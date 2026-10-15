import math
import sys
from types import ModuleType

import pytest

from objectnav_eval.actions.converters import convert_command
from objectnav_eval.adapters import build_adapter
from objectnav_eval.tasks import load_protocol
from objectnav_eval.loading import load_agent
from objectnav_eval.evaluator import EvaluationConfig, Evaluator
from objectnav_eval.agent import EpisodeInfo


def test_quantization_uses_environment_parameters_and_half_up():
    p = load_protocol()
    params = {"forward_step_size": 0.2, "turn_angle": 20}
    result = convert_command({"type": "rotate", "degrees": -50}, p, params)
    assert result.actions == ("turn_right",) * 3
    assert result.error == -10
    assert convert_command({"type": "move", "meters": 0.5}, p, params).actions == ("move_forward",) * 3


@pytest.mark.parametrize("command", [
    {"type": "move", "meters": 0}, {"type": "move", "meters": -1},
    {"type": "rotate", "degrees": 0}, {"type": "move", "meters": math.nan},
    {"type": "rotate", "degrees": math.inf}, {"type": "move", "meters": True},
    {"type": "move", "meters": 100}, {"type": "rotate", "degrees": 1},
    {"type": "waypoint", "position": [1, 2]}, "teleport", 1,
    {"type": "primitive", "action": "stop", "meters": 1},
])
def test_rejects_invalid_commands(command):
    with pytest.raises(ValueError):
        convert_command(command, load_protocol(), {"forward_step_size": .25, "turn_angle": 30})


class Native:
    def reset(self, episode):
        pass

    def act(self, obs):
        assert "semantic" not in obs
        return 3


def test_structural_loading_and_explicit_integer_adapter(monkeypatch):
    module = ModuleType("native_test")
    module.Agent = Native
    monkeypatch.setitem(sys.modules, "native_test", module)
    native = load_agent("native_test:Agent")
    adapted = build_adapter(native, {"type": "discrete", "mapping": {3: "turn_left"}})
    assert adapted.act({"rgb": 1}) == "turn_left"
    with pytest.raises(ValueError):
        build_adapter(native, {"type": "discrete", "mapping": {0: "stop"}}).act({})


class Backend:
    action_parameters = {"forward_step_size": .25, "turn_angle": 30}
    episode_count = 2

    def __init__(self):
        self.steps = 0
        self.index = -1
        self.trace = []

    @property
    def episode_over(self):
        return self.steps == 2

    def reset(self):
        self.steps = 0
        self.index += 1
        return EpisodeInfo(str(self.index), "s", str(self.index)), {"rgb": 1, "semantic": 2}

    def step(self, action):
        self.steps += 1
        self.trace.append(action)
        return {"rgb": 1, "semantic": 2}

    def get_metrics(self):
        return dict(success=0, spl=0, soft_spl=0, distance_to_goal=1)

    def close(self):
        pass


class Motion(Native):
    def act(self, obs):
        assert "semantic" not in obs
        return {"type": "move", "meters": 1}


def test_multi_step_counts_and_cancels_at_episode_end():
    backend = Backend()
    rows = Evaluator(backend, EvaluationConfig()).evaluate(Motion())
    assert backend.trace == ["move_forward"] * 4
    assert [r["num_steps"] for r in rows] == [2, 2]
    assert [r["agent_command_count"] for r in rows] == [1, 1]
    assert [r["primitive_action_count"] for r in rows] == [2, 2]
    assert rows[0]["action_conversions"][0]["executed_steps"] == 2


def test_task_specific_actions_and_permissions(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text('task_id: test\nprotocol_version: 1\nallowed_actions: [stop]\nallowed_observations: [rgb]\nrequired_metrics: [success]\nhabitat_config: test.yaml\nepisode_identity: objectnav_v1\n')
    protocol = load_protocol(path)
    assert convert_command("stop", protocol, {}).actions == ("stop",)
    with pytest.raises(ValueError):
        convert_command("turn_left", protocol, {})
    with pytest.raises(ValueError, match="observation"):
        EvaluationConfig(protocol=protocol, allowed_observations=("semantic",))


def test_adapter_cannot_override_task_physics():
    with pytest.raises(ValueError, match="Unsupported adapter"):
        build_adapter(Native(), {"type": "motion", "forward_step_size": 1.0})


def test_mapped_action_still_requires_protocol_permission():
    adapter = build_adapter(Native(), {"type": "discrete", "mapping": {3: "teleport"}})
    with pytest.raises(ValueError, match="Invalid action"):
        Evaluator(Backend(), EvaluationConfig(strict_actions=False)).evaluate(adapter)


def test_observation_transform_is_after_permission_filter():
    class Inspect(Native):
        def act(self, obs):
            assert set(obs) == {"image"}
            return "stop"
    adapter = build_adapter(Inspect(), {"observations": {"rgb": {"rename": "image"}, "semantic": {"rename": "hidden"}}})
    Evaluator(Backend(), EvaluationConfig()).evaluate(adapter)


def test_discrete_enum_and_string_mapping():
    from enum import Enum
    class Action(Enum):
        LEFT = 3
    adapter = build_adapter(Native(), {"type": "discrete", "mapping": {3: "turn_left", "halt": "stop"}})
    assert adapter.convert_action(Action.LEFT) == "turn_left"
    assert adapter.convert_action("halt") == "stop"
    with pytest.raises(ValueError):
        adapter.convert_action(True)
    for value in (3.0, 3 + 0j):
        with pytest.raises(ValueError):
            adapter.convert_action(value)


def test_motion_fields_and_64_step_boundary():
    adapter = build_adapter(Native(), {"type": "motion", "rotate_field": "angle", "move_field": "distance"})
    assert adapter.convert_action({"angle": 90}) == {"type": "rotate", "degrees": 90}
    assert adapter.convert_action({"distance": 1}) == {"type": "move", "meters": 1}
    command = {"type": "move", "meters": 16}
    assert len(convert_command(command, load_protocol(), {"forward_step_size": .25}).actions) == 64
    with pytest.raises(ValueError, match="effective Habitat"):
        convert_command(command, load_protocol(), {})


def test_missing_metrics_are_rejected():
    backend = Backend()
    backend.get_metrics = lambda: {"success": 0}
    with pytest.raises(ValueError, match="required task metrics"):
        Evaluator(backend, EvaluationConfig()).evaluate(Motion())


def test_quantization_log_and_failure_counts():
    from objectnav_eval.evaluator import EpisodeEvaluationError
    class Bad(Motion):
        def act(self, obs):
            return {"type": "move", "meters": float("nan")}
    with pytest.raises(EpisodeEvaluationError) as caught:
        list(Evaluator(Backend(), EvaluationConfig()).iter_evaluate(Bad()))
    assert caught.value.failure["primitive_action_count"] == 0
    assert caught.value.failure["agent_command_count"] == 1
    assert "finite" in caught.value.failure["action_conversion_error"]


def test_adapter_mapping_failure_has_conversion_error_record():
    from objectnav_eval.evaluator import EpisodeEvaluationError
    adapter = build_adapter(Native(), {"type": "discrete", "mapping": {0: "stop"}})
    with pytest.raises(EpisodeEvaluationError) as caught:
        list(Evaluator(Backend(), EvaluationConfig()).iter_evaluate(adapter))
    assert caught.value.failure["stage"] == "adapter.action"
    assert "Unmapped action" in caught.value.failure["action_conversion_error"]


def test_typed_command_objects():
    from objectnav_eval.actions import ActionCommand
    assert convert_command(ActionCommand("primitive", action="stop"), load_protocol(), {}).actions == ("stop",)
    assert convert_command(ActionCommand("rotate", degrees=90), load_protocol(), {"turn_angle": 30}).actions == ("turn_left",) * 3
    with pytest.raises(ValueError):
        convert_command(ActionCommand("rotate", degrees=90, meters=1), load_protocol(), {})


def test_default_protocol_file_matches_builtin():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "configs/tasks/mp3d_objectnav_v1.yaml"
    assert load_protocol(path).sha256 == load_protocol().sha256
