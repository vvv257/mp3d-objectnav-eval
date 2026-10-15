from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .base import TaskProtocol


def vln_ce_episode_uid(episode) -> str:
    """Build a machine-independent ID for one VLN-CE instruction episode.

    Habitat renumbers `episode_id` while loading datasets, and one R2R
    trajectory can carry several instructions, so identity uses only
    task-defining fields: scene, language instruction and `trajectory_id`.
    This policy is separate from the ObjectNav one and is never shared with it.
    """

    scene_path = Path(str(episode.scene_id))

    # Keep only scene_id/scene_id.glb, dropping machine-specific prefixes
    # such as /home/user/datasets/.
    if len(scene_path.parts) >= 2:
        scene_key = "/".join(scene_path.parts[-2:])
    else:
        scene_key = scene_path.name

    instruction = getattr(episode, "instruction", None)
    text = getattr(instruction, "instruction_text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("VLN-CE episode requires a non-empty instruction_text")

    payload = {
        "scene": scene_key,
        "instruction": text,
        "trajectory_id": str(getattr(episode, "trajectory_id", "")),
    }

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]

    return f"{scene_path.stem}:{digest}"


def vln_ce_r2r_protocol() -> TaskProtocol:
    """VLN-CE R2R protocol mirroring Habitat-Lab 0.2.4 `benchmark/nav/vln_r2r.yaml`.

    The action space is the one that config composes (`VLN-v0`: stop,
    move_forward, turn_left, turn_right), not the ObjectNav six. `ndtw` is
    absent because that config provides only success, spl and distance_to_goal.
    Turn angle, forward step and the success distance are read from the resolved
    Habitat configuration at run time, never assumed here.
    """

    return TaskProtocol(
        task_id="vln_ce_r2r_v1",
        protocol_version=1,
        allowed_actions=("stop", "move_forward", "turn_left", "turn_right"),
        allowed_observations=("rgb", "depth", "instruction"),
        required_metrics=("success", "spl", "distance_to_goal"),
        habitat_config="benchmark/nav/vln_r2r.yaml",
        episode_identity="objectnav_eval.tasks.vln_ce:vln_ce_episode_uid",
        observation_specs={
            "rgb": {"layout": "HWC", "dtype": "uint8", "range": [0, 255]},
            "depth": {
                "layout": "HWC",
                "dtype": "float32",
                "units": "resolved Habitat depth configuration",
            },
            "instruction": {
                "dtype": "mapping",
                "fields": ["text", "tokens", "trajectory_id"],
                "source": "Habitat InstructionSensor",
            },
        },
        action_parameters={"source": "resolved Habitat configuration"},
    )
