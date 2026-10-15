from __future__ import annotations

import math
from dataclasses import dataclass

from .schema import normalize_command

CONVERTER_VERSION = "1"


@dataclass(frozen=True)
class Conversion:
    actions: tuple[str, ...]
    requested: float | None = None
    quantized: float | None = None
    error: float = 0.0


def convert_command(command, protocol, parameters):
    command = normalize_command(command)
    if command.type == "primitive":
        protocol.validate_action(command.action)
        return Conversion((command.action,))
    rotate = command.type == "rotate"
    requested = command.degrees if rotate else command.meters
    parameter = "turn_angle" if rotate else "forward_step_size"
    step = parameters.get(parameter)
    if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0:
        raise ValueError(f"Missing or invalid effective Habitat {parameter}")
    ratio = abs(requested) / step
    if not math.isfinite(ratio) or ratio >= 64.5:
        raise ValueError("Command exceeds 64 primitive actions")
    count = math.floor(ratio + .5)
    if count == 0:
        raise ValueError("Nonzero command quantizes to zero primitive actions")
    action = ("turn_left" if requested > 0 else "turn_right") if rotate else "move_forward"
    protocol.validate_action(action)
    actual = count * step * (-1 if requested < 0 else 1)
    return Conversion((action,) * count, float(requested), actual, actual - requested)
