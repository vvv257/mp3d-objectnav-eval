"""Opt-in real VLN-CE task tests.

Requires:
  OBJECTNAV_TEST_VLN_SCENES_DIR      MatterPort3D root that contains `mp3d/`
  OBJECTNAV_TEST_VLN_EPISODE_SOURCE  an ObjectNav v1 `*.json.gz` whose poses the
                                     synthetic VLN dataset reuses

The episodes are synthetic (see `vln_ce_fixture`): these tests verify the VLN
task wiring on a real simulator, not VLN-CE benchmark accuracy.
"""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from objectnav_eval.backends.habitat import HabitatBackend
from objectnav_eval.config import habitat_overrides, load_yaml
from objectnav_eval.evaluator import EvaluationConfig, Evaluator
from objectnav_eval.tasks import load_protocol

from vln_ce_fixture import SequenceAgent, write_synthetic_vln_dataset

SCENES_DIR = os.environ.get("OBJECTNAV_TEST_VLN_SCENES_DIR")
EPISODE_SOURCE = os.environ.get("OBJECTNAV_TEST_VLN_EPISODE_SOURCE")

pytestmark = pytest.mark.skipif(
    not (SCENES_DIR and EPISODE_SOURCE and os.path.exists(EPISODE_SOURCE)),
    reason="real VLN-CE integration is opt-in",
)


def _vln_config(tmp_path, *, max_episode_steps=20):
    dataset = write_synthetic_vln_dataset(EPISODE_SOURCE, tmp_path / "episodes")
    config_path = tmp_path / "vln_ce_synthetic.yaml"
    config_path.write_text(
        "benchmark:\n"
        "  name: vln_ce_r2r_synthetic\n"
        "  habitat_config: benchmark/nav/vln_r2r.yaml\n"
        "  split: val_unseen\n"
        f"  data_path: {dataset.parent.parent}/{{split}}/{{split}}.json.gz\n"
        f"  scenes_dir: {SCENES_DIR}\n"
        f"  max_episode_steps: {max_episode_steps}\n"
        "  seed: 100\n"
        "  overrides:\n"
        "    - habitat.environment.iterator_options.shuffle=False\n"
        "task_protocol: vln_ce_r2r_v1\n"
    )
    return config_path


def _digest(value):
    payload = value.tobytes() if hasattr(value, "tobytes") else repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RecordingBackend(HabitatBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []

    def step(self, action):
        observation = super().step(action)
        self.trace.append((action, {key: _digest(value) for key, value in observation.items()}))
        return observation


def _trace_for(config_path, commands):
    config = load_yaml(config_path)
    backend = RecordingBackend(config["benchmark"]["habitat_config"], habitat_overrides(config))
    try:
        protocol = load_protocol("vln_ce_r2r_v1")
        backend.configure_protocol(protocol)
        config = EvaluationConfig(protocol=protocol, max_episodes=1)
        result = Evaluator(backend, config).evaluate(SequenceAgent(commands))[0]
        return backend.trace, result
    finally:
        backend.close()


def test_vln_ce_protocol_actions_and_runtime_parameters_match_the_real_task(tmp_path):
    config = load_yaml(_vln_config(tmp_path))
    backend = HabitatBackend(config["benchmark"]["habitat_config"], habitat_overrides(config))
    try:
        runtime = backend.configure_protocol(load_protocol("vln_ce_r2r_v1"))
        # Habitat's VLN-v0 task composes exactly these four actions.
        assert runtime["available_actions"] == ["move_forward", "stop", "turn_left", "turn_right"]
        # VLN-CE r2r resolves to 15 degrees, not the ObjectNav 30.
        assert runtime["action_parameters"]["turn_angle"] == 15
        assert runtime["action_parameters"]["forward_step_size"] == 0.25
        # Success criteria are read from the task's measurements, defaults included.
        assert runtime["success_criteria"]["distance_to"] == "POINT"
        assert isinstance(runtime["success_criteria"]["success_distance"], float)

        # An ObjectNav-only action must be refused before any episode runs.
        protocol = dataclasses.replace(
            load_protocol("vln_ce_r2r_v1"), allowed_actions=("stop", "look_up")
        )
        with pytest.raises(ValueError, match="does not support protocol actions"):
            backend.configure_protocol(protocol)

        # An evaluator left on the ObjectNav default must not run a VLN backend.
        backend.configure_protocol(load_protocol("vln_ce_r2r_v1"))
        with pytest.raises(ValueError, match="does not match the protocol the backend validated"):
            Evaluator(backend, EvaluationConfig())
    finally:
        backend.close()


def test_vln_ce_discrete_and_motion_commands_produce_the_same_trajectory(tmp_path):
    config_path = _vln_config(tmp_path)
    traces, rows = [], []
    for motion in (False, True):
        commands = (
            [{"type": "rotate", "degrees": 90}, {"type": "move", "meters": .5}, "stop"]
            if motion
            else ["turn_left"] * 6 + ["move_forward"] * 2 + ["stop"]
        )
        trace, row = _trace_for(config_path, commands)
        traces.append(trace)
        rows.append(row)

    assert traces[0] == traces[1]
    assert rows[0]["num_steps"] == rows[1]["num_steps"] == 9
    assert [r["agent_command_count"] for r in rows] == [9, 3]
    assert rows[0]["metrics"] == rows[1]["metrics"]
    assert set(rows[0]["metrics"]) >= {"success", "spl", "distance_to_goal"}
    assert rows[0]["task_id"] == "vln_ce_r2r_v1" and rows[0]["protocol_version"] == 1


def test_vln_ce_cli_run_delivers_language_observation_and_records_identity(tmp_path):
    config_path = _vln_config(tmp_path)
    output = tmp_path / "run"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "objectnav_eval.cli", "--agent", "action_fixture:VlnWalkAgent",
         "--config", str(config_path), "--output", str(output), "--num-episodes", "1"],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr

    row = json.loads((output / "episodes.jsonl").read_text().splitlines()[0])
    assert row["task_id"] == "vln_ce_r2r_v1" and row["protocol_version"] == 1
    assert row["primitive_action_count"] == row["num_steps"] == 3
    assert row["agent_command_count"] == 3
    assert set(row["metrics"]) >= {"success", "spl", "distance_to_goal"}
    # VLN-CE identity is scene + instruction + trajectory_id, not ObjectNav's.
    assert row["episode_uid"].startswith(Path(row["scene_id"]).stem + ":")

    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["task_protocol_sha256"] == load_protocol("vln_ce_r2r_v1").sha256
    assert metadata["task_protocol_file_sha256"] is None
    assert metadata["adapter_type"] == "identity"
    assert metadata["runtime"]["action_parameters"]["turn_angle"] == 15
