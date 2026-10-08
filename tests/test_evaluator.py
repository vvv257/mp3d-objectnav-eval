from typing import Any, Mapping

import pytest

from objectnav_eval.agent import EpisodeInfo, ObjectNavAgent
from objectnav_eval.backends.base import EvaluationBackend
from objectnav_eval.evaluator import (
    EpisodeEvaluationError,
    EvaluationConfig,
    Evaluator,
)
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
        return EpisodeInfo(
            str(self._episode),
            "fake_scene",
            f"fake_scene:{self._episode}",
        ), {
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


class FailingAgent(FakeAgent):
    def act(self, observation: Mapping[str, Any]) -> str:
        raise RuntimeError("agent crashed")


class EpisodeCountFailureBackend(FakeBackend):
    @property
    def episode_count(self) -> int:
        raise RuntimeError("cannot count episodes")


class EpisodeOverFailureBackend(FakeBackend):
    @property
    def episode_over(self) -> bool:
        raise RuntimeError("cannot read episode state")


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


def test_evaluator_resumes_after_validating_completed_episode_prefix():
    backend = FakeBackend()
    evaluator = Evaluator(backend, EvaluationConfig())
    completed = [{"episode_index": 0, "episode_uid": "fake_scene:0"}]

    results = list(evaluator.iter_evaluate(FakeAgent(), completed))

    assert [row["episode_index"] for row in results] == [1]
    assert backend.closed


def test_evaluator_rejects_resume_when_episode_identity_changed():
    backend = FakeBackend()
    evaluator = Evaluator(backend, EvaluationConfig())
    completed = [{"episode_index": 0, "episode_uid": "different:episode"}]

    with pytest.raises(ValueError, match="episode_uid"):
        list(evaluator.iter_evaluate(FakeAgent(), completed))

    assert backend.closed


def test_evaluator_reports_episode_context_when_agent_fails():
    backend = FakeBackend()
    evaluator = Evaluator(backend, EvaluationConfig())

    with pytest.raises(EpisodeEvaluationError) as caught:
        list(evaluator.iter_evaluate(FailingAgent()))

    assert caught.value.failure["episode_index"] == 0
    assert caught.value.failure["episode_uid"] == "fake_scene:0"
    assert caught.value.failure["stage"] == "agent.act"
    assert caught.value.failure["exception_type"] == "RuntimeError"
    assert caught.value.failure["message"] == "agent crashed"
    assert "RuntimeError: agent crashed" in caught.value.failure["traceback"]
    assert backend.closed


def test_evaluator_closes_backend_when_episode_count_fails():
    backend = EpisodeCountFailureBackend()
    evaluator = Evaluator(backend, EvaluationConfig())

    with pytest.raises(RuntimeError, match="cannot count episodes"):
        evaluator.evaluate(FakeAgent())

    assert backend.closed


def test_evaluator_labels_episode_over_failure():
    backend = EpisodeOverFailureBackend()
    evaluator = Evaluator(backend, EvaluationConfig())

    with pytest.raises(EpisodeEvaluationError) as caught:
        list(evaluator.iter_evaluate(FakeAgent()))

    assert caught.value.failure["stage"] == "backend.episode_over"
    assert backend.closed


@pytest.mark.parametrize("max_episodes", [0, -1])
def test_evaluation_config_rejects_non_positive_episode_limit(max_episodes):
    with pytest.raises(ValueError, match="max_episodes must be positive"):
        EvaluationConfig(max_episodes=max_episodes)
