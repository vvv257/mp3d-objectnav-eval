"""Opt-in real MP3D tests; requires OBJECTNAV_TEST_CONFIG pointing to local YAML."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from objectnav_eval.backends.habitat import HabitatBackend
from objectnav_eval.config import load_yaml, habitat_overrides
from objectnav_eval.tasks import load_protocol
from objectnav_eval.evaluator import EvaluationConfig, Evaluator

pytestmark = pytest.mark.skipif(not os.environ.get("OBJECTNAV_TEST_CONFIG"), reason="real MP3D integration is opt-in")


class RecordingBackend(HabitatBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []

    def step(self, action):
        obs = super().step(action)
        self.trace.append((action, {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in obs.items()}))
        return obs


class SequenceAgent:
    def __init__(self, commands):
        self.commands = commands

    def reset(self, episode):
        self.iterator = iter(self.commands)

    def act(self, observation):
        return next(self.iterator)


def test_real_primitive_and_motion_trajectories_match():
    config = load_yaml(os.environ["OBJECTNAV_TEST_CONFIG"])
    traces, rows = [], []
    for motion in (False, True):
        backend = RecordingBackend(config["benchmark"]["habitat_config"], habitat_overrides(config))
        try:
            runtime = backend.configure_protocol(load_protocol())
            params = runtime["action_parameters"]
            commands = ([{"type": "rotate", "degrees": 3 * params["turn_angle"]},
                         {"type": "move", "meters": 4 * params["forward_step_size"]}, "stop"]
                        if motion else ["turn_left"] * 3 + ["move_forward"] * 4 + ["stop"])
            result = Evaluator(backend, EvaluationConfig(max_episodes=1)).evaluate(SequenceAgent(commands))[0]
            rows.append(result)
            traces.append(backend.trace)
        finally:
            backend.close()
    assert traces[0] == traces[1]
    assert rows[0]["episode_uid"] == rows[1]["episode_uid"]
    assert rows[0]["metrics"] == rows[1]["metrics"]
    assert rows[0]["num_steps"] == rows[1]["num_steps"] == 8
    assert [r["agent_command_count"] for r in rows] == [8, 3]


def test_real_cli_adapter_failure_resume_and_changed_config(tmp_path):
    adapter = tmp_path / "adapter.yaml"
    adapter.write_text("type: discrete\nmapping:\n  0: stop\n")
    output = tmp_path / "run"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    env["OBJECTNAV_TEST_FAILURE_MARKER"] = str(tmp_path / "failure-marker")
    command = [sys.executable, "-m", "objectnav_eval.cli", "--agent", "action_fixture:FailOnceAgent",
               "--config", os.environ["OBJECTNAV_TEST_CONFIG"], "--adapter-config", str(adapter),
               "--output", str(output), "--num-episodes", "3"]
    failed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    assert failed.returncode == 1, failed.stderr
    before = (output / "episodes.jsonl").read_text()
    assert len(before.splitlines()) == 1
    assert json.loads((output / "errors.jsonl").read_text())["stage"] == "agent.act"
    adapter.write_text("type: discrete\nmapping:\n  0: turn_left\n")
    refused = subprocess.run(command + ["--resume"], env=env, capture_output=True, text=True, timeout=180)
    assert refused.returncode != 0 and "adapter_config_sha256" in refused.stderr
    assert (output / "episodes.jsonl").read_text() == before
    adapter.write_text("type: discrete\nmapping:\n  0: stop\n")
    resumed = subprocess.run(command + ["--resume"], env=env, capture_output=True, text=True, timeout=180)
    assert resumed.returncode == 0, resumed.stderr
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert [r["episode_index"] for r in rows] == [0, 1, 2]
    assert len({r["episode_uid"] for r in rows}) == 3
    assert all(r["primitive_action_count"] == r["num_steps"] == 1 for r in rows)
    meta = json.loads((output / "run_metadata.json").read_text())
    assert meta["status"] == "completed" and meta["failure_count"] == 1
