import gzip
import json
import math
from types import SimpleNamespace

import pytest

from objectnav_eval.metrics.vln_ce import VlnCeOfficialMetrics, fast_dtw_distance


class FakeSimulator:
    def __init__(self, position):
        self.position = list(position)

    def get_agent_state(self):
        return SimpleNamespace(position=self.position)


class FakeEnv:
    def __init__(self, position=(0.0, 0.0, 0.0), distance=4.0):
        self.sim = FakeSimulator(position)
        self.distance = distance

    def get_metrics(self):
        return {"distance_to_goal": self.distance}


def write_gt(path, payload):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_official_metrics_follow_vln_ce_definitions(tmp_path):
    gt_path = tmp_path / "val_unseen_gt.json.gz"
    write_gt(gt_path, {"7": {"locations": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]}})
    env = FakeEnv()
    metrics = VlnCeOfficialMetrics(
        env=env,
        options={"ground_truth_path": str(gt_path)},
        parameters={"success_distance": 3.0, "fast_dtw_radius": 1},
    )

    metrics.reset(SimpleNamespace(episode_id="7"))
    env.sim.position = [1.0, 0.0, 0.0]
    env.distance = 2.5
    metrics.update("move_forward")
    env.sim.position = [1.0, 0.0, 0.0]
    metrics.update("turn_left")
    metrics.update("stop")

    result = metrics.get_metrics()
    assert result["path_length"] == 1.0
    assert result["steps_taken"] == 3
    assert result["navigation_error"] == result["distance_to_goal"] == 2.5
    assert result["oracle_success"] == 1.0
    assert result["success"] == 1.0
    assert result["spl"] == 1.0
    expected_ndtw = math.exp(-1.0 / (3 * 3.0))
    assert result["ndtw"] == pytest.approx(expected_ndtw)
    assert result["sdtw"] == pytest.approx(expected_ndtw)


def test_official_success_requires_stop_and_oracle_success_does_not(tmp_path):
    gt_path = tmp_path / "gt.json.gz"
    write_gt(gt_path, {"1": {"locations": [[0, 0, 0]]}})
    env = FakeEnv(distance=2.0)
    metrics = VlnCeOfficialMetrics(
        env=env,
        options={"ground_truth_path": str(gt_path)},
        parameters={"success_distance": 3.0},
    )
    metrics.reset(SimpleNamespace(episode_id="1"))
    metrics.update("turn_left")

    result = metrics.get_metrics()
    assert result["oracle_success"] == 1.0
    assert result["success"] == result["spl"] == result["sdtw"] == 0.0
    assert result["steps_taken"] == 1


def test_official_spl_penalizes_actual_travel_distance(tmp_path):
    gt_path = tmp_path / "gt.json.gz"
    write_gt(gt_path, {"1": {"locations": [[0, 0, 0], [4, 0, 0]]}})
    env = FakeEnv(distance=4.0)
    metrics = VlnCeOfficialMetrics(
        env=env,
        options={"ground_truth_path": str(gt_path)},
        parameters={"success_distance": 3.0},
    )
    metrics.reset(SimpleNamespace(episode_id="1"))
    env.sim.position = [10.0, 0.0, 0.0]
    env.distance = 1.0
    metrics.update("move_forward")
    metrics.update("stop")

    assert metrics.get_metrics()["spl"] == pytest.approx(0.4)


def test_official_metrics_require_valid_matching_ground_truth(tmp_path):
    missing = tmp_path / "missing_gt.json.gz"
    with pytest.raises(FileNotFoundError, match="ground truth"):
        VlnCeOfficialMetrics(
            env=FakeEnv(), options={"ground_truth_path": str(missing)}, parameters={}
        )

    gt_path = tmp_path / "gt.json.gz"
    write_gt(gt_path, {"other": {"locations": [[0, 0, 0]]}})
    metrics = VlnCeOfficialMetrics(
        env=FakeEnv(), options={"ground_truth_path": str(gt_path)}, parameters={}
    )
    with pytest.raises(ValueError, match="episode_id.*not found"):
        metrics.reset(SimpleNamespace(episode_id="7"))


def test_fast_dtw_matches_known_path_cost():
    assert fast_dtw_distance([[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [2, 0, 0]]) == 1.0
