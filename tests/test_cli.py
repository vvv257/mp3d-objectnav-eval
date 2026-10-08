import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from objectnav_eval import cli
from objectnav_eval.agent import ObjectNavAgent


class MetadataAgent(ObjectNavAgent):
    def act(self, observation):
        return "stop"


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
    assert clean["working_tree_sha256"] is None
    assert dirty["commit"] == clean["commit"]
    assert dirty["dirty"] is True
    assert len(dirty["working_tree_sha256"]) == 64


def test_git_state_excludes_the_current_output_directory(tmp_path: Path):
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
    output = tmp_path / "custom-run"
    output.mkdir()
    (output / "episodes.jsonl").write_text("{}\n", encoding="utf-8")

    state = cli._git_state(tmp_path, excluded_paths=[output])

    assert state["dirty"] is False
    assert state["working_tree_sha256"] is None


def test_run_metadata_records_config_hash_and_software_versions(tmp_path: Path):
    config_path = tmp_path / "eval.yaml"
    raw_config = b"benchmark:\n  split: val_mini\n"
    config_path.write_bytes(raw_config)
    habitat_config = tmp_path / "habitat.yaml"
    habitat_config.write_bytes(b"habitat:\n  seed: 100\n")

    metadata = cli.build_run_metadata(
        agent_entrypoint="agent.module:Agent",
        agent=MetadataAgent(),
        evaluator_config=config_path,
        habitat_config=str(habitat_config),
        overrides=["habitat.seed=100"],
    )

    assert metadata["evaluator_config_sha256"] == hashlib.sha256(
        raw_config
    ).hexdigest()
    assert metadata["evaluator_version"]
    assert "evaluator_git_commit" in metadata
    assert "evaluator_git_dirty" in metadata
    assert "evaluator_working_tree_sha256" in metadata
    assert metadata["agent_code_sha256"] == hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    assert metadata["habitat_config_sha256"] == hashlib.sha256(
        habitat_config.read_bytes()
    ).hexdigest()
    assert metadata["habitat_lab_version"]
    assert metadata["habitat_sim_version"]


def test_parser_accepts_explicit_resume_flag():
    args = cli.build_parser().parse_args(
        [
            "--agent",
            "agent.module:Agent",
            "--config",
            "eval.yaml",
            "--output",
            "results/run",
            "--resume",
        ]
    )

    assert args.resume is True


def test_cli_records_backend_initialization_failure(tmp_path: Path, monkeypatch):
    config = tmp_path / "eval.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    output = tmp_path / "results"
    fake_config = {
        "benchmark": {"habitat_config": "missing.yaml"},
        "evaluation": {},
    }
    monkeypatch.setattr(cli, "load_yaml", lambda path: fake_config)
    monkeypatch.setattr(cli, "habitat_overrides", lambda config: [])
    monkeypatch.setattr(cli, "load_agent", lambda entrypoint: MetadataAgent())

    def fail_backend(**kwargs):
        raise RuntimeError("simulator unavailable")

    monkeypatch.setattr(cli, "HabitatBackend", fail_backend)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "objectnav-eval",
            "--agent",
            "agent.module:Agent",
            "--config",
            str(config),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as caught:
        cli.main()

    error = json.loads((output / "errors.jsonl").read_text())
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert caught.value.code == 1
    assert error["stage"] == "backend.init"
    assert error["message"] == "simulator unavailable"
    assert metadata["status"] == "failed"
