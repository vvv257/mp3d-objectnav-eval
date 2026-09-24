from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


VALID_ACTIONS = (
    "stop",
    "move_forward",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
)


@dataclass(frozen=True)
class EpisodeInfo:
    """Non-privileged metadata supplied at episode reset."""

    # Habitat runtime ID. This may be renumbered by Habitat.
    episode_id: str

    # Scene path/identifier.
    scene_id: str

    # Stable evaluator-generated identifier.
    # It is designed for matching the same episode across methods/runs.
    episode_uid: str = ""


class ObjectNavAgent(ABC):
    """Minimal interface every evaluated method must implement.

    Mapping, recognition, exploration, planning, memory, learned policies, etc.
    all belong inside the agent implementation, not inside the evaluator.
    """

    def reset(self, episode: EpisodeInfo) -> None:
        """Reset method-specific state at the beginning of an episode."""

    @abstractmethod
    def act(self, observation: Mapping[str, Any]) -> str:
        """Return one Habitat ObjectNav action name."""
        raise NotImplementedError
