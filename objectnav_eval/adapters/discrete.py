from enum import Enum
from numbers import Integral
from .base import BaseAdapter


class DiscreteActionAdapter(BaseAdapter):
    def __init__(self, agent, mapping, observation_adapter=None):
        super().__init__(agent, observation_adapter)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Discrete adapter requires an explicit nonempty mapping")
        if any(type(k) not in (int, str) or not isinstance(v, str) for k, v in mapping.items()):
            raise ValueError("Action mapping keys must be integer/string and values action names")
        self.mapping = mapping

    def convert_action(self, action):
        if isinstance(action, Enum):
            action = action.value
        if isinstance(action, bool) or not isinstance(action, (Integral, str)):
            raise ValueError("Action ID must be an integer or string")
        try:
            return self.mapping[action]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Unmapped action: {action!r}") from exc
