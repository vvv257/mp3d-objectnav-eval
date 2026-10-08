from typing import Any, Mapping

import pytest

from objectnav_eval.agent import EpisodeInfo, ObjectNavAgent
from objectnav_eval.backends.base import EvaluationBackend
from objectnav_eval.evaluator import EvaluationConfig, Evaluator
from objectnav_eval.reporting import summarize_results


class FakeBackend(EvaluationBackend):
    def __init__(self) -> None:
        self._episode = -1
        self._step = 0
        self.closed = False

    @property
    def episode_count(self) -> int:
        return 2

    @property
    def episode_over(self) -> bool:
        return self._step >= 2

    def reset(self):
        self._episode += 1
        self._step = 0
        return EpisodeInfo(str(self._episode), "fake_scene"), {
            "rgb": "rgb",
            "depth": "depth",
            "semantic": "MUST_NOT_LEAK",
        }

    def step(self, action: str):
        self._step += 1
        return {"rgb": "rgb", "depth": "depth", "semantic": "MUST_NOT_LEAK"}

    def get_metrics(self):
        return {"success": 1.0, "spl": 0.5}

    def close(self) -> None:
        self.closed = True


class FakeAgent(ObjectNavAgent):
    def reset(self, episode: EpisodeInfo) -> None:
        pass

    def act(self, observation: Mapping[str, Any]) -> str:
        assert "semantic" not in observation
        return "move_forward"


def test_evaluator_filters_and_aggregates():
    backend = FakeBackend()
    evaluator = Evaluator(
        backend,
        EvaluationConfig(allowed_observations=("rgb", "depth")),
    )
    results = evaluator.evaluate(FakeAgent())
    summary = summarize_results(results)

    assert len(results) == 2
    assert summary["success"] == 1.0
    assert summary["spl"] == 0.5
    assert summary["avg_num_steps"] == 2.0
    assert backend.closed


@pytest.mark.parametrize("max_episodes", [0, -1])
def test_evaluation_config_rejects_non_positive_episode_limit(max_episodes):
    with pytest.raises(ValueError, match="max_episodes must be positive"):
        EvaluationConfig(max_episodes=max_episodes)
