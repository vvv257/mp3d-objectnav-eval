from __future__ import annotations

import csv
import fcntl
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from objectnav_eval.reporting.summary import summarize_results
from objectnav_eval.reporting.writer import _json_default


_COMPATIBILITY_KEYS = (
    "agent_entrypoint",
    "evaluator_version",
    "evaluator_git_commit",
    "evaluator_git_dirty",
    "evaluator_working_tree_sha256",
    "evaluator_config_sha256",
    "agent_code_sha256",
    "habitat_config",
    "habitat_config_sha256",
    "habitat_overrides",
    "habitat_lab_version",
    "habitat_sim_version",
)

_MANAGED_FILES = (
    "episodes.jsonl",
    "episodes.csv",
    "summary.json",
    "run_metadata.json",
    "evaluator_config.yaml",
    "errors.jsonl",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=_json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


class RunStore:
    """Durable run output with explicit, compatibility-checked resume."""

    def __init__(
        self,
        output_dir: str | Path,
        run_metadata: Mapping[str, Any],
        evaluator_config: str | Path,
        *,
        resume: bool = False,
    ) -> None:
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        _fsync_directory(self.output.parent)
        self._lock_stream = (self.output / ".run.lock").open("a+")
        try:
            fcntl.flock(
                self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            self._lock_stream.close()
            self._lock_stream = None
            error = RuntimeError if resume else FileExistsError
            message = "an evaluation is already active for this --output"
            if not resume:
                message += "; wait for it to stop instead of starting with --resume"
            raise error(message) from exc
        self.episodes_path = self.output / "episodes.jsonl"
        self.errors_path = self.output / "errors.jsonl"
        self.metadata_path = self.output / "run_metadata.json"

        try:
            if resume:
                self.resuming = True
                self._load_resume(run_metadata)
            else:
                self.resuming = False
                existing = [
                    name for name in _MANAGED_FILES if (self.output / name).exists()
                ]
                if existing:
                    raise FileExistsError(
                        "output already contains evaluation results "
                        f"({', '.join(existing)}); use --resume to continue the "
                        "same run or choose a new --output"
                    )
                self.results: list[dict[str, Any]] = []
                self.metadata = dict(run_metadata)
                started = _now()
                self.metadata.update(
                    {
                        "status": "running",
                        "completed_episodes": 0,
                        "total_episodes": None,
                        "failure_count": 0,
                        "started_at": started,
                        "updated_at": started,
                        "finished_at": None,
                    }
                )
                try:
                    with self.episodes_path.open("x", encoding="utf-8") as stream:
                        stream.flush()
                        os.fsync(stream.fileno())
                except FileExistsError as exc:
                    raise FileExistsError(
                        "another evaluation claimed this --output; use a different "
                        "directory or --resume after it stops"
                    ) from exc
                _fsync_directory(self.output)
                shutil.copyfile(
                    evaluator_config, self.output / "evaluator_config.yaml"
                )
                with (self.output / "evaluator_config.yaml").open("rb") as stream:
                    os.fsync(stream.fileno())
                _fsync_directory(self.output)
                _atomic_write_json(self.metadata_path, self.metadata)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._lock_stream is not None:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            self._lock_stream.close()
            self._lock_stream = None

    def _load_resume(self, current_metadata: Mapping[str, Any]) -> None:
        if not self.metadata_path.exists() or not self.episodes_path.exists():
            raise FileNotFoundError(
                "--resume requires run_metadata.json and episodes.jsonl in --output"
            )
        saved = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        mismatches = [
            key
            for key in _COMPATIBILITY_KEYS
            if saved.get(key) != current_metadata.get(key)
        ]
        if mismatches:
            raise ValueError(
                "cannot resume because run identity changed: " + ", ".join(mismatches)
            )
        self.metadata = saved
        self.results = self._load_episode_rows()

    def _load_episode_rows(self) -> list[dict[str, Any]]:
        data = self.episodes_path.read_bytes()
        rows: list[dict[str, Any]] = []
        offset = 0
        truncated_tail = False
        lines = data.splitlines(keepends=True)
        for position, raw_line in enumerate(lines):
            is_last = position == len(lines) - 1
            complete = raw_line.endswith((b"\n", b"\r"))
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if is_last and not complete:
                    with self.episodes_path.open("r+b") as stream:
                        stream.truncate(offset)
                    truncated_tail = True
                    break
                raise ValueError(
                    f"invalid episodes.jsonl line {position + 1}; refusing unsafe resume"
                ) from exc
            if row.get("episode_index") != position:
                raise ValueError(
                    "episodes.jsonl is not a consecutive episode_index prefix"
                )
            rows.append(row)
            offset += len(raw_line)

        if (
            rows
            and data
            and not truncated_tail
            and not data.endswith((b"\n", b"\r"))
        ):
            with self.episodes_path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        return rows

    def start(self, *, total_episodes: int) -> None:
        saved_total = self.metadata.get("total_episodes")
        if saved_total is not None and saved_total != total_episodes:
            raise ValueError(
                f"cannot resume: total_episodes changed from {saved_total} "
                f"to {total_episodes}"
            )
        if len(self.results) > total_episodes:
            raise ValueError("completed episode count exceeds total_episodes")
        self.metadata["total_episodes"] = total_episodes
        self.metadata["completed_episodes"] = len(self.results)
        if not self.resuming:
            self.metadata["status"] = "running"
            self.metadata["updated_at"] = _now()
            self.metadata["finished_at"] = None
        self._write_derived_outputs()
        _atomic_write_json(self.metadata_path, self.metadata)

    def append_episode(self, result: Mapping[str, Any]) -> None:
        expected_index = len(self.results)
        if result.get("episode_index") != expected_index:
            raise ValueError(
                f"expected episode_index {expected_index}, got "
                f"{result.get('episode_index')!r}"
            )
        row = dict(result)
        with self.episodes_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=_json_default) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.results.append(row)
        self.metadata["completed_episodes"] = len(self.results)
        self.metadata["status"] = "running"
        self.metadata["updated_at"] = _now()
        self.metadata["finished_at"] = None
        self._write_derived_outputs()
        _atomic_write_json(self.metadata_path, self.metadata)

    def record_failure(self, failure: Mapping[str, Any]) -> None:
        record = dict(failure)
        record.setdefault("recorded_at", _now())
        with self.errors_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=_json_default) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.metadata["status"] = "failed"
        self.metadata["completed_episodes"] = len(self.results)
        self.metadata["failure_count"] = int(
            self.metadata.get("failure_count", 0)
        ) + 1
        self.metadata["updated_at"] = _now()
        self.metadata["finished_at"] = self.metadata["updated_at"]
        _atomic_write_json(self.metadata_path, self.metadata)

    def complete(self) -> dict[str, Any]:
        total = self.metadata.get("total_episodes")
        if total is None or len(self.results) != total:
            raise ValueError(
                f"cannot complete run with {len(self.results)} of {total} episodes"
            )
        self.metadata["status"] = "completed"
        self.metadata["completed_episodes"] = len(self.results)
        self.metadata["updated_at"] = _now()
        self.metadata["finished_at"] = self.metadata["updated_at"]
        summary = self._write_derived_outputs()
        _atomic_write_json(self.metadata_path, self.metadata)
        return summary

    def _write_derived_outputs(self) -> dict[str, Any]:
        summary = summarize_results(self.results)
        _atomic_write_json(self.output / "summary.json", summary)

        path = self.output / "episodes.csv"
        temporary = path.with_name(f".{path.name}.tmp")
        flat_keys = [
            "episode_index",
            "episode_id",
            "episode_uid",
            "scene_id",
            "num_steps",
            "elapsed_seconds",
        ]
        metric_keys = sorted(
            {
                key
                for row in self.results
                for key in row.get("metrics", {}).keys()
            }
        )
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=flat_keys + metric_keys)
            writer.writeheader()
            for row in self.results:
                record = {key: row.get(key) for key in flat_keys}
                record.update(row.get("metrics", {}))
                writer.writerow(record)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return summary
