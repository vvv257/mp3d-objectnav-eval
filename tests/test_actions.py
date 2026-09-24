import pytest

from objectnav_eval.agent import EpisodeInfo, ObjectNavAgent
from objectnav_eval.backends.base import EvaluationBackend
from objectnav_eval.evaluator import EvaluationConfig, Evaluator


class OneStepBackend(EvaluationBackend):
    @property
    def episode_count(self):
        return 1

    @property
    def episode_over(self):
        return False

    def reset(self):
        return EpisodeInfo("0", "scene"), {"rgb": 0}

    def step(self, action):
        raise AssertionError("invalid action should be rejected before env.step")

    def get_metrics(self):
        return {}

    def close(self):
        pass


class BadAgent(ObjectNavAgent):
    def act(self, observation):
        return "teleport"


def test_invalid_action_is_rejected():
    evaluator = Evaluator(OneStepBackend(), EvaluationConfig())
    with pytest.raises(ValueError, match="Invalid action"):
        evaluator.evaluate(BadAgent())
