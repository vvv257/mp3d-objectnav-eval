from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskProtocol:
    task_id: str
    protocol_version: int
    allowed_actions: tuple[str, ...]
    allowed_observations: tuple[str, ...]
    required_metrics: tuple[str, ...]
    habitat_config: str
    episode_identity: str
    observation_specs: Mapping[str, Any] = field(default_factory=dict)
    action_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id or type(self.protocol_version) is not int or self.protocol_version < 1:
            raise ValueError("Invalid task_id or protocol_version")
        for name in ("allowed_actions", "allowed_observations", "required_metrics"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values or any(not isinstance(x, str) or not x for x in values):
                raise ValueError(f"Invalid {name}")
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate {name}")
            object.__setattr__(self, name, tuple(values))
        if not self.habitat_config or not self.episode_identity:
            raise ValueError("Task requires habitat_config and episode_identity")

    def validate_action(self, action):
        if not isinstance(action, str) or action not in self.allowed_actions:
            raise ValueError(f"Invalid action {action!r}; expected one of {self.allowed_actions}")

    def identity(self, episode):
        if self.episode_identity == "objectnav_v1":
            from objectnav_eval.backends.habitat import _stable_episode_uid
            return _stable_episode_uid(episode)
        module, name = self.episode_identity.split(":", 1)
        result = getattr(importlib.import_module(module), name)(episode)
        if not isinstance(result, str) or not result:
            raise ValueError("Episode identity strategy must return a non-empty string")
        return result

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
