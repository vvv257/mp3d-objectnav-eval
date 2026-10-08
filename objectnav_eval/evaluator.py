from __future__ import annotations

import time
import traceback
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


class EpisodeEvaluationError(RuntimeError):
    """An episode failed, with enough context to persist and diagnose it."""

    def __init__(self, failure: Mapping[str, Any], original: BaseException):
        super().__init__(
            f"episode {failure.get('episode_index')} failed during "
            f"{failure.get('stage')}: {original}"
        )
        self.failure = dict(failure)
        self.original = original


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

    @property
    def total_episodes(self) -> int:
        total = self.backend.episode_count
        if self.config.max_episodes is not None:
            total = min(total, self.config.max_episodes)
        return total

    def iter_evaluate(
        self,
        agent: ObjectNavAgent,
        completed_results: Iterable[Mapping[str, Any]] = (),
        *,
        total_episodes: int | None = None,
    ) -> Iterable[dict[str, Any]]:
        completed = list(completed_results)
        try:
            total = self.total_episodes if total_episodes is None else total_episodes
            if len(completed) > total:
                raise ValueError(
                    f"resume has {len(completed)} completed episodes, "
                    f"but run total is {total}"
                )
            for index in range(total):
                stage = "backend.reset"
                episode = None
                try:
                    episode, observation = self.backend.reset()
                except BaseException as exc:
                    raise self._episode_error(index, episode, stage, exc) from exc

                if index < len(completed):
                    saved = completed[index]
                    if saved.get("episode_index") != index:
                        raise ValueError(
                            "resume results are not a consecutive episode_index prefix"
                        )
                    if saved.get("episode_uid") != episode.episode_uid:
                        raise ValueError(
                            "resume episode_uid mismatch at episode_index "
                            f"{index}: saved {saved.get('episode_uid')!r}, "
                            f"dataset {episode.episode_uid!r}"
                        )
                    continue

                steps = 0
                started = time.perf_counter()
                try:
                    stage = "agent.reset"
                    agent.reset(episode)

                    while True:
                        stage = "backend.episode_over"
                        if self.backend.episode_over:
                            break
                        stage = "observation.filter"
                        agent_obs = filter_observation(
                            observation, self.config.allowed_observations
                        )
                        stage = "agent.act"
                        action = agent.act(agent_obs)
                        stage = "action.validation"
                        if self.config.strict_actions and action not in VALID_ACTIONS:
                            raise ValueError(
                                f"Invalid action {action!r}; expected one of {VALID_ACTIONS}"
                            )
                        stage = "backend.step"
                        observation = self.backend.step(action)
                        steps += 1

                    stage = "backend.get_metrics"
                    metrics = dict(self.backend.get_metrics())
                except BaseException as exc:
                    raise self._episode_error(index, episode, stage, exc) from exc

                yield {
                    "episode_index": index,
                    "episode_id": episode.episode_id,
                    "episode_uid": episode.episode_uid,
                    "scene_id": episode.scene_id,
                    "num_steps": steps,
                    "elapsed_seconds": time.perf_counter() - started,
                    "metrics": metrics,
                }
        finally:
            self.backend.close()

    @staticmethod
    def _episode_error(index, episode, stage, exc) -> EpisodeEvaluationError:
        failure = {
            "episode_index": index,
            "episode_id": getattr(episode, "episode_id", None),
            "episode_uid": getattr(episode, "episode_uid", None),
            "scene_id": getattr(episode, "scene_id", None),
            "stage": stage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }
        return EpisodeEvaluationError(failure, exc)

    def evaluate(self, agent: ObjectNavAgent) -> list[dict[str, Any]]:
        try:
            return list(self.iter_evaluate(agent))
        except EpisodeEvaluationError as exc:
            # Preserve the original public behavior for callers using evaluate().
            raise exc.original.with_traceback(exc.original.__traceback__) from exc
