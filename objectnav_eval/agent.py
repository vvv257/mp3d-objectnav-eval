from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


# Compatibility alias. TaskProtocol is the authority for execution validation.
from objectnav_eval.tasks.objectnav import objectnav_protocol

VALID_ACTIONS = objectnav_protocol().allowed_actions


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
    def act(self, observation: Mapping[str, Any]) -> Any:
        """Return a primitive name or a typed primitive/move/rotate command."""
        raise NotImplementedError
