from .base import BaseAdapter
from .discrete import DiscreteActionAdapter
from .motion import MotionActionAdapter
from .observation import ObservationAdapter


def build_adapter(agent, config=None):
    config = config or {}
    kind = config.get("type", "identity")
    keys = {"type", "observations"}
    if kind == "discrete":
        keys.add("mapping")
    elif kind == "motion":
        keys.update(("rotate_field", "move_field"))
    elif kind != "identity":
        raise ValueError(f"Unknown adapter type: {kind!r}")
    if set(config) - keys:
        raise ValueError(f"Unsupported adapter settings: {set(config) - keys}")
    obs = ObservationAdapter(config["observations"]) if "observations" in config else None
    if kind == "discrete":
        return DiscreteActionAdapter(agent, config.get("mapping"), obs)
    if kind == "motion":
        return MotionActionAdapter(agent, config.get("rotate_field", "degrees"), config.get("move_field", "meters"), obs)
    return BaseAdapter(agent, obs)


__all__ = ["BaseAdapter", "DiscreteActionAdapter", "MotionActionAdapter", "ObservationAdapter", "build_adapter"]
