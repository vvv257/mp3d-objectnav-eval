from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from objectnav_eval.agent import ObjectNavAgent, VALID_ACTIONS
from objectnav_eval.backends.base import EvaluationBackend


@dataclass(frozen=True)
class EvaluationConfig:
    allowed_observations: tuple[str, ...] = (
        "rgb",
        "depth",
        "gps",
        "compass",
        "objectgoal",
    )
    max_episodes: int | None = None
    strict_actions: bool = True

    def __post_init__(self) -> None:
        if self.max_episodes is not None and self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive")


def filter_observation(
    observation: Mapping[str, Any], allowed: Iterable[str]
) -> dict[str, Any]:
    """Prevent accidental leakage of simulator-only/privileged observations."""
    allowed_set = set(allowed)
    return {key: value for key, value in observation.items() if key in allowed_set}


class Evaluator:
    def __init__(self, backend: EvaluationBackend, config: EvaluationConfig):
        self.backend = backend
        self.config = config

    def evaluate(self, agent: ObjectNavAgent) -> list[dict[str, Any]]:
        total = self.backend.episode_count
        if self.config.max_episodes is not None:
            total = min(total, self.config.max_episodes)

        results: list[dict[str, Any]] = []
        try:
            for index in range(total):
                episode, observation = self.backend.reset()
                agent.reset(episode)
                steps = 0
                started = time.perf_counter()

                while not self.backend.episode_over:
                    agent_obs = filter_observation(
                        observation, self.config.allowed_observations
                    )
                    action = agent.act(agent_obs)
                    if self.config.strict_actions and action not in VALID_ACTIONS:
                        raise ValueError(
                            f"Invalid action {action!r}; expected one of {VALID_ACTIONS}"
                        )
                    observation = self.backend.step(action)
                    steps += 1

                results.append(
                    {
                        "episode_index": index,
                        "episode_id": episode.episode_id,
                        "episode_uid": episode.episode_uid,
                        "scene_id": episode.scene_id,
                        "num_steps": steps,
                        "elapsed_seconds": time.perf_counter() - started,
                        "metrics": dict(self.backend.get_metrics()),
                    }
                )
        finally:
            self.backend.close()

        return results
