import sys
from types import ModuleType, SimpleNamespace

from objectnav_eval.backends.habitat import HabitatBackend


class FakeEnv:
    def __init__(self, config):
        self.config = config
        self.episodes = [object()]
        self.episode_over = False
        self.current_episode = SimpleNamespace(
            episode_id="7",
            scene_id="/datasets/mp3d/scene/scene.glb",
            start_position=[1.0, 2.0, 3.0],
            start_rotation=[0.0, 0.0, 0.0, 1.0],
            object_category="chair",
        )
        self.last_action = None
        self.closed = False

    def reset(self):
        return {"rgb": "observation"}

    def step(self, action):
        self.last_action = action
        self.episode_over = True
        return {"rgb": "next observation"}

    def get_metrics(self):
        return {"success": 1.0, "spl": 0.5}

    def close(self):
        self.closed = True


def test_backend_uses_habitat_api_shared_by_024_and_033(monkeypatch):
    habitat = ModuleType("habitat")
    calls = {}

    def get_config(*, config_path, overrides):
        calls["get_config"] = (config_path, overrides)
        return {"loaded": True}

    habitat.get_config = get_config
    habitat.Env = FakeEnv
    monkeypatch.setitem(sys.modules, "habitat", habitat)

    backend = HabitatBackend("objectnav.yaml", ["habitat.seed=100"])
    episode, observation = backend.reset()
    next_observation = backend.step("turn_left")

    assert calls["get_config"] == (
        "objectnav.yaml",
        ["habitat.seed=100"],
    )
    assert backend.episode_count == 1
    assert episode.episode_id == "7"
    assert episode.episode_uid.startswith("scene:")
    assert observation == {"rgb": "observation"}
    assert next_observation == {"rgb": "next observation"}
    assert backend._env.last_action == {"action": "turn_left"}
    assert backend.episode_over is True
    assert backend.get_metrics() == {"success": 1.0, "spl": 0.5}

    backend.close()
    assert backend._env.closed is True
