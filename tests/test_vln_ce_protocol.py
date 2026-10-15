from dataclasses import dataclass
from pathlib import Path

import pytest

from objectnav_eval.actions.converters import convert_command
from objectnav_eval.agent import EpisodeInfo
from objectnav_eval.evaluator import EvaluationConfig, Evaluator
from objectnav_eval.tasks import is_builtin_protocol, load_protocol, vln_ce_r2r_protocol
from objectnav_eval.tasks.vln_ce import vln_ce_episode_uid

SYNTHETIC_INSTRUCTION = "Leave the room and stop near the window."


@dataclass
class StubInstruction:
    instruction_text: str


class StubEpisode:
    def __init__(self, scene_id="/data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
                 instruction_text=SYNTHETIC_INSTRUCTION, trajectory_id=7):
        self.scene_id = scene_id
        self.instruction = StubInstruction(instruction_text)
        self.trajectory_id = trajectory_id


class VlnBackend:
    """FakeBackend with the LanguageNav task's observation and metrics."""

    action_parameters = {"forward_step_size": .25, "turn_angle": 15}
    protocol = vln_ce_r2r_protocol()
    episode_count = 1

    def __init__(self, instruction=SYNTHETIC_INSTRUCTION):
        self.instruction = instruction
        self.steps = 0
        self.seen = []

    @property
    def episode_over(self):
        return self.steps == 2

    def reset(self):
        self.steps = 0
        observation = {
            "rgb": 1,
            "depth": 2,
            "instruction": {"text": self.instruction, "tokens": ["go", "to", "the", "goal"]},
            "semantic": 9,
        }
        return EpisodeInfo("0", "mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb", "x8F5xyUWy9e:deadbeef"), observation

    def step(self, action):
        self.steps += 1
        self.seen.append(action)
        return {"rgb": 1, "instruction": {"text": self.instruction}}

    def get_metrics(self):
        return {"success": 1.0, "spl": .5, "distance_to_goal": .1}

    def close(self):
        pass


class InstructionAgent:
    def reset(self, episode):
        self.episodes = [episode]

    def act(self, observation):
        assert observation["instruction"]["text"] == SYNTHETIC_INSTRUCTION
        assert "semantic" not in observation
        return "move_forward"


def test_builtin_vln_protocol_matches_its_config_file():
    path = Path(__file__).resolve().parents[1] / "configs/tasks/vln_ce_r2r.yaml"
    assert is_builtin_protocol("vln_ce_r2r_v1") and not is_builtin_protocol("tasks/x.yaml")
    assert load_protocol("vln_ce_r2r_v1").sha256 == load_protocol(path).sha256


def test_vln_action_space_comes_from_the_task_config_not_objectnav():
    protocol = vln_ce_r2r_protocol()
    assert protocol.allowed_actions == ("stop", "move_forward", "turn_left", "turn_right")
    assert "look_up" not in protocol.allowed_actions
    assert protocol.allowed_observations == ("rgb", "depth", "instruction")
    # Habitat-Lab 0.2.4 `vln_r2r.yaml` provides these three and no ndtw.
    assert protocol.required_metrics == ("success", "spl", "distance_to_goal")
    with pytest.raises(ValueError):
        convert_command("look_up", protocol, {"turn_angle": 15})


def test_vln_motion_commands_use_the_effective_vln_turn_angle():
    protocol = vln_ce_r2r_protocol()
    result = convert_command({"type": "rotate", "degrees": 90}, protocol, {"turn_angle": 15})
    assert result.actions == ("turn_left",) * 6
    assert convert_command({"type": "move", "meters": .5}, protocol, {"forward_step_size": .25}).actions == ("move_forward",) * 2


def test_vln_episode_uid_uses_scene_instruction_and_trajectory():
    reference = vln_ce_episode_uid(StubEpisode())
    assert reference == vln_ce_episode_uid(StubEpisode(scene_id="/mnt/nas/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"))
    assert reference != vln_ce_episode_uid(StubEpisode(instruction_text="Turn left at the sofa."))
    assert reference != vln_ce_episode_uid(StubEpisode(trajectory_id=8))
    assert reference != vln_ce_episode_uid(StubEpisode(scene_id="/data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb"))
    assert reference.startswith("17DRP5sb8fy:")
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="instruction_text"):
            vln_ce_episode_uid(StubEpisode(instruction_text=blank))


def test_language_observation_reaches_agent_and_stays_permission_checked():
    protocol = vln_ce_r2r_protocol()
    backend = VlnBackend()
    rows = list(Evaluator(backend, EvaluationConfig(protocol=protocol)).iter_evaluate(InstructionAgent()))
    assert rows[0]["task_id"] == "vln_ce_r2r_v1"
    assert rows[0]["protocol_version"] == 1
    assert rows[0]["metrics"]["success"] == 1.0
    assert backend.seen == ["move_forward"] * 2

    # Withholding `instruction` must hide it from the agent, not summarize it.
    class Blind(InstructionAgent):
        def act(self, observation):
            assert "instruction" not in observation
            return "stop"

    backend = VlnBackend()
    list(Evaluator(backend, EvaluationConfig(protocol=protocol, allowed_observations=("rgb",))).iter_evaluate(Blind()))


def test_evaluator_refuses_a_protocol_the_backend_did_not_validate():
    with pytest.raises(ValueError, match="does not match the protocol the backend validated"):
        Evaluator(VlnBackend(), EvaluationConfig())


def test_vln_protocol_rejects_objectnav_only_observation_and_metrics():
    protocol = vln_ce_r2r_protocol()
    with pytest.raises(ValueError, match="observation"):
        EvaluationConfig(protocol=protocol, allowed_observations=("objectgoal",))
    assert EvaluationConfig(protocol=protocol).allowed_observations == ("rgb", "depth", "instruction")

    class WrongMetrics(VlnBackend):
        def get_metrics(self):
            return {"success": 1.0}

    with pytest.raises(ValueError, match="required task metrics"):
        Evaluator(WrongMetrics(), EvaluationConfig(protocol=protocol)).evaluate(InstructionAgent())
