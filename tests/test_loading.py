import subprocess
import sys
from types import ModuleType

from objectnav_eval.agent import ObjectNavAgent
from objectnav_eval.loading import load_agent


class ExternalAgent(ObjectNavAgent):
    def act(self, observation):
        return "stop"


def test_load_agent_accepts_external_agent_class(monkeypatch):
    module = ModuleType("external_agent_for_test")
    module.ExternalAgent = ExternalAgent
    monkeypatch.setitem(sys.modules, module.__name__, module)

    agent = load_agent("external_agent_for_test:ExternalAgent")

    assert isinstance(agent, ExternalAgent)


def test_installed_example_agent_is_importable_outside_repository(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-c", "import examples.random_agent"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
