from __future__ import annotations

import importlib
import inspect
from typing import Any

from objectnav_eval.agent import ObjectNavAgent


def load_agent(entrypoint: str) -> ObjectNavAgent:
    """Load `module:ClassOrFactory` and return an ObjectNavAgent."""
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

    if not isinstance(agent, ObjectNavAgent):
        raise TypeError(
            f"Loaded object {entrypoint!r} must implement ObjectNavAgent; got {type(agent)}"
        )
    return agent
