from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from objectnav_eval.agent import EpisodeInfo


class EvaluationBackend(ABC):
    """Simulator/backend boundary used by the evaluator."""

    @property
    @abstractmethod
    def episode_count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def episode_over(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> tuple[EpisodeInfo, Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
