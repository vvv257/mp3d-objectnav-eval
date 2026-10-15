"""Agents used only by subprocess integration tests."""
import os
from pathlib import Path

from vln_ce_fixture import INSTRUCTION_TEXT


class VlnWalkAgent:
    """Consume the language observation, walk two steps, then stop."""

    def __init__(self):
        self.steps = 0

    def reset(self, episode):
        self.steps = 0

    def act(self, observation):
        assert observation["instruction"]["text"] == INSTRUCTION_TEXT, observation.keys()
        assert "semantic" not in observation
        self.steps += 1
        return "stop" if self.steps > 2 else "move_forward"


class FailOnceAgent:
    def __init__(self):
        self.index = -1

    def reset(self, episode):
        self.index += 1

    def act(self, observation):
        assert "semantic" not in observation
        marker = Path(os.environ["OBJECTNAV_TEST_FAILURE_MARKER"])
        # Resume skips the completed prefix without invoking agent.reset.
        if self.index == 1 and not marker.exists():
            marker.touch()
            raise RuntimeError("intentional integration failure")
        return 0
