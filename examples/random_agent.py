import random
from typing import Any, Mapping

from objectnav_eval.agent import EpisodeInfo, ObjectNavAgent


class RandomAgent(ObjectNavAgent):
    def __init__(self) -> None:
        self.rng = random.Random(0)

    def reset(self, episode: EpisodeInfo) -> None:
        pass

    def act(self, observation: Mapping[str, Any]) -> str:
        # Intentionally simple smoke-test baseline, not a meaningful ObjectNav method.
        if self.rng.random() < 0.02:
            return "stop"
        return self.rng.choice(("move_forward", "turn_left", "turn_right"))
