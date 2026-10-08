import hashlib
import subprocess
from pathlib import Path

from objectnav_eval import cli


def test_git_state_records_commit_and_dirty_tree(tmp_path: Path):
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(
        ["git", "-C", tmp_path, "config", "user.name", "Test User"], check=True
    )
    subprocess.run(
        ["git", "-C", tmp_path, "config", "user.email", "test@example.com"],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", tmp_path, "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", tmp_path, "commit", "-qm", "initial"], check=True
    )

    clean = cli._git_state(tmp_path)
    tracked.write_text("dirty\n", encoding="utf-8")
    dirty = cli._git_state(tmp_path)

    assert len(clean["commit"]) == 40
    assert clean["dirty"] is False
    assert dirty["commit"] == clean["commit"]
    assert dirty["dirty"] is True


def test_run_metadata_records_config_hash_and_software_versions(tmp_path: Path):
    config_path = tmp_path / "eval.yaml"
    raw_config = b"benchmark:\n  split: val_mini\n"
    config_path.write_bytes(raw_config)

    metadata = cli.build_run_metadata(
        agent_entrypoint="agent.module:Agent",
        evaluator_config=config_path,
        habitat_config="benchmark/nav/objectnav/objectnav_mp3d.yaml",
        overrides=["habitat.seed=100"],
    )

    assert metadata["evaluator_config_sha256"] == hashlib.sha256(
        raw_config
    ).hexdigest()
    assert metadata["evaluator_version"]
    assert "evaluator_git_commit" in metadata
    assert "evaluator_git_dirty" in metadata
    assert metadata["habitat_lab_version"]
    assert metadata["habitat_sim_version"]
