from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import platform
import subprocess
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from objectnav_eval import __version__
from objectnav_eval.agent import ObjectNavAgent
from objectnav_eval.backends.habitat import HabitatBackend
from objectnav_eval.config import habitat_overrides, load_yaml
from objectnav_eval.evaluator import (
    EpisodeEvaluationError,
    EvaluationConfig,
    Evaluator,
)
from objectnav_eval.loading import load_agent
from objectnav_eval.reporting import RunStore


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _git_state(
    repo_root: Path | None = None,
    excluded_paths: Sequence[str | Path] = (),
) -> dict[str, object]:
    root = repo_root or Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return {"commit": None, "dirty": None, "working_tree_sha256": None}

    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "working_tree_sha256": None}

    try:
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD", "--"],
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        excluded = [Path(path).resolve() for path in excluded_paths]

        def is_excluded(path: Path) -> bool:
            for excluded_path in excluded:
                try:
                    path.relative_to(excluded_path)
                    return True
                except ValueError:
                    pass
            return False

        filtered_untracked = [
            name
            for name in untracked
            if name
            and not is_excluded((root / name.decode()).resolve())
        ]
        if not diff and not filtered_untracked:
            return {
                "commit": commit,
                "dirty": False,
                "working_tree_sha256": None,
            }
        digest = hashlib.sha256(diff)
        for encoded_name in sorted(filtered_untracked):
            digest.update(b"\0" + encoded_name + b"\0")
            digest.update((root / encoded_name.decode()).read_bytes())
        working_tree_sha256: str | None = digest.hexdigest()
    except (OSError, UnicodeDecodeError, subprocess.CalledProcessError):
        working_tree_sha256 = None

    return {
        "commit": commit,
        "dirty": True,
        "working_tree_sha256": working_tree_sha256,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _agent_code_sha256(agent: ObjectNavAgent | None) -> str | None:
    if agent is None:
        return None
    try:
        module_name = type(agent).__module__
        top_level = module_name.split(".", 1)[0]
        spec = importlib.util.find_spec(top_level)
        if spec is not None and spec.submodule_search_locations:
            root = Path(next(iter(spec.submodule_search_locations)))
            files = sorted(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in {".py", ".json", ".toml", ".yaml", ".yml"}
            )
            digest = hashlib.sha256()
            for path in files:
                digest.update(str(path.relative_to(root)).encode() + b"\0")
                digest.update(path.read_bytes())
            return digest.hexdigest()
        source = inspect.getsourcefile(type(agent))
        return _file_sha256(Path(source)) if source else None
    except (ImportError, OSError, TypeError, ValueError):
        return None


def _habitat_config_sha256(config_path: str) -> str | None:
    path = Path(config_path)
    if path.is_file():
        return _file_sha256(path)
    spec = importlib.util.find_spec("habitat")
    if spec is None or not spec.submodule_search_locations:
        return None
    for root in spec.submodule_search_locations:
        candidate = Path(root) / "config" / config_path
        if candidate.is_file():
            return _file_sha256(candidate)
    return None


def build_run_metadata(
    *,
    agent_entrypoint: str,
    agent: ObjectNavAgent | None = None,
    evaluator_config: str | Path,
    habitat_config: str,
    overrides: Sequence[str],
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    config_path = Path(evaluator_config).resolve()
    excluded_paths = [Path(output_dir).resolve()] if output_dir is not None else []
    git_state = _git_state(excluded_paths=excluded_paths)
    return {
        "agent_entrypoint": agent_entrypoint,
        "evaluator_version": __version__,
        "evaluator_git_commit": git_state["commit"],
        "evaluator_git_dirty": git_state["dirty"],
        "evaluator_working_tree_sha256": git_state["working_tree_sha256"],
        "agent_code_sha256": _agent_code_sha256(agent),
        "evaluator_config": str(config_path),
        "evaluator_config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "habitat_config": habitat_config,
        "habitat_config_sha256": _habitat_config_sha256(habitat_config),
        "habitat_overrides": list(overrides),
        "habitat_lab_version": _distribution_version("habitat-lab"),
        "habitat_sim_version": _distribution_version("habitat-sim"),
        "python": sys.version,
        "platform": platform.platform(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an agent on MP3D ObjectNav")
    parser.add_argument("--agent", required=True, help="module:ClassOrFactory")
    parser.add_argument("--config", required=True, help="Evaluator YAML")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible interrupted run in --output",
    )
    return parser


def _print_progress(result: dict, completed: int, total: int, run_started: float) -> None:
    metrics = result.get("metrics", {})
    details = [
        f"[{completed}/{total}]",
        f"episode={result.get('episode_uid')}",
        f"steps={result.get('num_steps')}",
    ]
    for key in ("success", "spl"):
        if key in metrics:
            details.append(f"{key}={metrics[key]}")
    details.append(f"elapsed={time.perf_counter() - run_started:.1f}s")
    print(" ".join(details), file=sys.stderr, flush=True)


def _run_failure(stage: str, exc: BaseException) -> dict[str, object]:
    return {
        "episode_index": None,
        "episode_id": None,
        "episode_uid": None,
        "scene_id": None,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    benchmark = config["benchmark"]
    evaluation = config.get("evaluation", {})
    overrides = habitat_overrides(config)

    evaluation_config = EvaluationConfig(
        allowed_observations=tuple(
            evaluation.get(
                "allowed_observations",
                ["rgb", "depth", "gps", "compass", "objectgoal"],
            )
        ),
        max_episodes=args.num_episodes,
        strict_actions=bool(evaluation.get("strict_actions", True)),
    )
    agent = load_agent(args.agent)
    metadata = build_run_metadata(
        agent_entrypoint=args.agent,
        agent=agent,
        evaluator_config=args.config,
        habitat_config=benchmark["habitat_config"],
        overrides=overrides,
        output_dir=args.output,
    )
    store = RunStore(
        args.output,
        metadata,
        args.config,
        resume=args.resume,
    )
    try:
        backend = HabitatBackend(
            config_path=benchmark["habitat_config"], overrides=overrides
        )
    except BaseException as exc:
        store.record_failure(_run_failure("backend.init", exc))
        store.close()
        print(
            f"Evaluation stopped during backend.init: {exc}. Details were saved "
            f"in {args.output}/errors.jsonl.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc

    evaluator = Evaluator(backend, evaluation_config)
    try:
        total_episodes = evaluator.total_episodes
    except BaseException as exc:
        backend.close()
        store.record_failure(_run_failure("backend.episode_count", exc))
        store.close()
        print(
            f"Evaluation stopped during backend.episode_count: {exc}. Details "
            f"were saved in {args.output}/errors.jsonl.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc

    try:
        store.start(total_episodes=total_episodes)
    except BaseException:
        backend.close()
        store.close()
        raise

    if store.results:
        print(
            f"Resuming after {len(store.results)}/{total_episodes} "
            "completed episodes; validating saved prefix...",
            file=sys.stderr,
            flush=True,
        )

    run_started = time.perf_counter()
    iterator = evaluator.iter_evaluate(
        agent, store.results, total_episodes=total_episodes
    )
    try:
        for result in iterator:
            store.append_episode(result)
            _print_progress(
                result,
                len(store.results),
                total_episodes,
                run_started,
            )
    except EpisodeEvaluationError as exc:
        store.record_failure(exc.failure)
        store.close()
        print(
            f"Evaluation stopped: {exc}. Completed episodes were kept in "
            f"{args.output}. Fix the cause and rerun the same command with --resume.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc
    finally:
        iterator.close()

    summary = store.complete()
    store.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
