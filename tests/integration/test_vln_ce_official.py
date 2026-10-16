"""Opt-in real official R2R-VLNCE metric integration test."""

import hashlib
import os

import pytest

from objectnav_eval.backends.habitat import HabitatBackend
from objectnav_eval.evaluator import EvaluationConfig, Evaluator
from objectnav_eval.tasks import load_protocol

from vln_ce_fixture import SequenceAgent


DATA_PATH = os.environ.get("OBJECTNAV_TEST_R2R_DATA_PATH")
GT_PATH = os.environ.get("OBJECTNAV_TEST_R2R_GT_PATH")
SCENES_DIR = os.environ.get("OBJECTNAV_TEST_VLN_SCENES_DIR")

pytestmark = pytest.mark.skipif(
    not all(path and os.path.isfile(path) for path in (DATA_PATH, GT_PATH))
    or not SCENES_DIR,
    reason="official R2R-VLNCE integration is opt-in",
)


def test_real_official_r2r_episode_produces_full_metrics():
    protocol = load_protocol("vln_ce_r2r_official_v1")
    overrides = [
        "habitat.dataset.split=val_unseen",
        f"habitat.dataset.data_path={DATA_PATH}",
        f"habitat.dataset.scenes_dir={SCENES_DIR}",
        "habitat.environment.max_episode_steps=20",
        "habitat.environment.iterator_options.shuffle=False",
    ]
    backend = HabitatBackend(
        protocol.habitat_config,
        overrides,
        metric_options={"ground_truth_path": GT_PATH},
    )
    try:
        runtime = backend.configure_protocol(protocol)
        row = Evaluator(
            backend, EvaluationConfig(protocol=protocol, max_episodes=1)
        ).evaluate(SequenceAgent(["move_forward", "move_forward", "stop"]))[0]
    finally:
        backend.close()

    assert set(row["metrics"]) >= set(protocol.required_metrics)
    assert row["metrics"]["steps_taken"] == row["num_steps"] == 3
    assert row["metrics"]["navigation_error"] == row["metrics"]["distance_to_goal"]
    assert 0.0 <= row["metrics"]["ndtw"] <= 1.0
    expected_hash = hashlib.sha256(open(GT_PATH, "rb").read()).hexdigest()
    assert runtime["task_metrics"]["ground_truth_sha256"] == expected_hash
    assert runtime["task_metrics"]["success_distance"] == 3.0
    assert runtime["success_criteria"]["success_distance"] == 3.0
    assert runtime["success_criteria"]["source"].endswith(":VlnCeOfficialMetrics")
    assert "habitat_success_criteria" in runtime
