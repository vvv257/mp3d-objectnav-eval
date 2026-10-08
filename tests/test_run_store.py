import json
from pathlib import Path

import pytest

from objectnav_eval.reporting import RunStore


def metadata(agent: str = "agent.module:Agent") -> dict:
    return {
        "agent_entrypoint": agent,
        "evaluator_version": "0.1.0",
        "evaluator_git_commit": "abc123",
        "evaluator_git_dirty": False,
        "evaluator_working_tree_sha256": None,
        "evaluator_config_sha256": "config-hash",
        "agent_code_sha256": "agent-hash",
        "habitat_config": "objectnav.yaml",
        "habitat_config_sha256": "habitat-hash",
        "habitat_overrides": ["habitat.seed=100"],
        "habitat_lab_version": "0.2.4",
        "habitat_sim_version": "0.2.4",
    }


def episode(index: int) -> dict:
    return {
        "episode_index": index,
        "episode_id": str(index),
        "episode_uid": f"scene:{index}",
        "scene_id": "scene",
        "num_steps": index + 1,
        "elapsed_seconds": 0.25,
        "metrics": {"success": float(index == 0), "spl": 0.5},
    }


def test_run_store_persists_each_episode_and_marks_completion(tmp_path: Path):
    config = tmp_path / "eval.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    output = tmp_path / "run"
    store = RunStore(output, metadata(), config)
    store.start(total_episodes=2)

    store.append_episode(episode(0))

    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    current_metadata = json.loads((output / "run_metadata.json").read_text())
    current_summary = json.loads((output / "summary.json").read_text())
    assert rows == [episode(0)]
    assert current_metadata["status"] == "running"
    assert current_metadata["completed_episodes"] == 1
    assert current_summary["num_episodes"] == 1

    store.append_episode(episode(1))
    store.complete()

    final_metadata = json.loads((output / "run_metadata.json").read_text())
    assert final_metadata["status"] == "completed"
    assert final_metadata["completed_episodes"] == 2
    assert final_metadata["total_episodes"] == 2


def test_run_store_records_failure_without_losing_completed_rows(tmp_path: Path):
    config = tmp_path / "eval.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    output = tmp_path / "run"
    store = RunStore(output, metadata(), config)
    store.start(total_episodes=2)
    store.append_episode(episode(0))

    store.record_failure(
        {
            "episode_index": 1,
            "episode_uid": "scene:1",
            "stage": "agent.act",
            "exception_type": "RuntimeError",
            "message": "boom",
        }
    )

    rows = (output / "episodes.jsonl").read_text().splitlines()
    errors = [json.loads(line) for line in (output / "errors.jsonl").read_text().splitlines()]
    current_metadata = json.loads((output / "run_metadata.json").read_text())
    assert len(rows) == 1
    assert errors[0]["message"] == "boom"
    assert current_metadata["status"] == "failed"
    assert current_metadata["completed_episodes"] == 1
    assert current_metadata["failure_count"] == 1


def test_run_store_refuses_existing_results_without_resume(tmp_path: Path):
    config = tmp_path / "eval.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    output = tmp_path / "run"
    first = RunStore(output, metadata(), config)
    first.start(total_episodes=2)

    with pytest.raises(FileExistsError, match="--resume"):
        RunStore(output, metadata(), config)


def test_run_store_resumes_compatible_prefix_and_repairs_partial_tail(tmp_path: Path):
    config = tmp_path / "eval.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    output = tmp_path / "run"
    first = RunStore(output, metadata(), config)
    first.start(total_episodes=2)
    first.append_episode(episode(0))
    with (output / "episodes.jsonl").open("ab") as stream:
        stream.write(b'{"episode_index": 1')
    first.close()

    resumed = RunStore(output, metadata(), config, resume=True)
    resumed.start(total_episodes=2)

    assert resumed.results == [episode(0)]
    assert (output / "episodes.jsonl").read_bytes().endswith(b"\n")
    resumed.append_episode(episode(1))
    resumed.complete()
    assert len((output / "episodes.jsonl").read_text().splitlines()) == 2


def test_run_store_refuses_incompatible_resume(tmp_path: Path):
    config = tmp_path / "eval.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    output = tmp_path / "run"
    first = RunStore(output, metadata(), config)
    first.start(total_episodes=2)
    first.close()

    with pytest.raises(ValueError, match="agent_entrypoint"):
        RunStore(output, metadata("other.module:Agent"), config, resume=True)


def test_run_store_refuses_changed_episode_total_on_resume(tmp_path: Path):
    config = tmp_path / "eval.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    output = tmp_path / "run"
    first = RunStore(output, metadata(), config)
    first.start(total_episodes=2)
    first.record_failure({"stage": "agent.act", "message": "boom"})
    first.close()

    resumed = RunStore(output, metadata(), config, resume=True)
    with pytest.raises(ValueError, match="total_episodes"):
        resumed.start(total_episodes=3)

    saved_metadata = json.loads((output / "run_metadata.json").read_text())
    assert saved_metadata["status"] == "failed"


def test_run_store_keeps_failed_status_until_resume_writes_next_episode(
    tmp_path: Path,
):
    config = tmp_path / "eval.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    output = tmp_path / "run"
    first = RunStore(output, metadata(), config)
    first.start(total_episodes=1)
    first.record_failure({"stage": "agent.act", "message": "boom"})
    first.close()

    resumed = RunStore(output, metadata(), config, resume=True)
    resumed.start(total_episodes=1)

    before_episode = json.loads((output / "run_metadata.json").read_text())
    assert before_episode["status"] == "failed"
    resumed.append_episode(episode(0))
    after_episode = json.loads((output / "run_metadata.json").read_text())
    assert after_episode["status"] == "running"


def test_run_store_refuses_resume_when_agent_source_changed(tmp_path: Path):
    config = tmp_path / "eval.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    output = tmp_path / "run"
    first = RunStore(output, metadata(), config)
    first.start(total_episodes=1)
    changed = metadata()
    first.close()
    changed["agent_code_sha256"] = "different-agent-hash"

    with pytest.raises(ValueError, match="agent_code_sha256"):
        RunStore(output, changed, config, resume=True)


def test_run_store_refuses_concurrent_resume(tmp_path: Path):
    config = tmp_path / "eval.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    output = tmp_path / "run"
    active = RunStore(output, metadata(), config)
    active.start(total_episodes=1)

    with pytest.raises(RuntimeError, match="already active"):
        RunStore(output, metadata(), config, resume=True)

    active.close()
