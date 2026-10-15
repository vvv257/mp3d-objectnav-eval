from dataclasses import dataclass
from typing import Any, Mapping
import math
import numbers


@dataclass(frozen=True)
class ActionCommand:
    type: str
    action: str = ""
    degrees: float = 0.0
    meters: float = 0.0


def normalize_command(value):
    if isinstance(value, ActionCommand):
        if value.type == "primitive" and value.degrees == value.meters == 0:
            value = {"type": value.type, "action": value.action}
        elif value.type == "rotate" and not value.action and value.meters == 0:
            value = {"type": value.type, "degrees": value.degrees}
        elif value.type == "move" and not value.action and value.degrees == 0:
            value = {"type": value.type, "meters": value.meters}
        else:
            raise ValueError("Conflicting command fields")
    if isinstance(value, str):
        return ActionCommand("primitive", action=value)
    if not isinstance(value, Mapping):
        raise ValueError("Expected action string or typed command mapping")
    kind = value.get("type")
    field = {"primitive": "action", "rotate": "degrees", "move": "meters"}.get(kind) if isinstance(kind, str) else None
    if field is None or set(value) != {"type", field}:
        raise ValueError(f"Unsupported or malformed action command: {value!r}")
    parameter = value[field]
    if kind == "primitive":
        if not isinstance(parameter, str):
            raise ValueError("Primitive action must be a string")
    elif isinstance(parameter, bool) or not isinstance(parameter, numbers.Real) or not math.isfinite(parameter):
        raise ValueError("Action parameter must be a finite number")
    elif parameter == 0 or (kind == "move" and parameter < 0):
        raise ValueError("move must be positive; rotate must be nonzero")
    return ActionCommand(kind, **{field: parameter})
