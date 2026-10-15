from __future__ import annotations

import importlib
import inspect
from typing import Any

from objectnav_eval.agent import ObjectNavAgent


def load_agent(entrypoint: str) -> ObjectNavAgent:
    """Load an explicit reset/act object; no third-party signature guessing."""
    if ":" not in entrypoint:
        raise ValueError("Agent entrypoint must look like 'package.module:AgentClass'")
    module_name, attr_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    target: Any = getattr(module, attr_name)

    if inspect.isclass(target):
        agent = target()
    elif callable(target):
        agent = target()
    else:
        agent = target

    if not all(callable(getattr(agent, name, None)) for name in ("reset", "act")):
        raise TypeError(
            f"Loaded object {entrypoint!r} must provide callable reset(episode) and act(observation); got {type(agent)}"
        )
    return agent
