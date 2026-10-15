"""Build a synthetic R2R-VLN dataset from ObjectNav episode poses.

The generated dataset reuses real MatterPort3D scenes, navmeshes and navigable
poses: each episode starts at an ObjectNav start pose and ends at a viewpoint of
that episode's goal object. Instructions are synthetic placeholders. The dataset
exists so the VLN-CE task wiring (language observation, episode identity,
metrics) can be exercised without the official R2R_VLNCE download. Results
collected on it are fixture smoke tests and must never be reported as VLN-CE
benchmark numbers.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

INSTRUCTION_TEXT = "Move forward into the room and stop at the target viewpoint."

# Habitat's SPL measure divides by the start-to-goal distance, so a goal at the
# start pose raises ZeroDivisionError while resetting the episode.
MIN_GOAL_DISTANCE = 1.0


def _goal_position(episode, goals_by_category):
    """Pick a navigable goal viewpoint of the episode's goal object."""

    scene_name = Path(str(episode["scene_id"])).name
    entries = goals_by_category.get(f"{scene_name}_{episode['object_category']}")
    if not entries:
        return None

    object_id = episode["info"].get("closest_goal_object_id")
    preferred = [entry for entry in entries if entry.get("object_id") == object_id]
    start = episode["start_position"]

    for entry in preferred or entries:
        for view_point in entry.get("view_points") or []:
            position = (view_point.get("agent_state") or {}).get("position")
            if position and math.dist(position, start) >= MIN_GOAL_DISTANCE:
                return position
    return None


class SequenceAgent:
    """Replay a fixed command list; used to compare execution paths."""

    def __init__(self, commands):
        self.commands = commands

    def reset(self, episode):
        self.iterator = iter(self.commands)

    def act(self, observation):
        assert "instruction" in observation
        return next(self.iterator)


def write_synthetic_vln_dataset(
    source_episodes,
    out_dir,
    *,
    split="val_unseen",
    count=2,
    instruction_text=INSTRUCTION_TEXT,
):
    """Write `<out_dir>/<split>/<split>.json.gz` in Habitat's R2RVLN-v1 format."""

    with gzip.open(source_episodes, "rt") as handle:
        source = json.load(handle)

    tokens = instruction_text.replace(".", "").lower().split()
    vocab = sorted(set(tokens))
    token_ids = [vocab.index(token) for token in tokens]

    goals_by_category = source.get("goals_by_category", {})

    episodes = []
    for episode in source["episodes"]:
        goal_position = _goal_position(episode, goals_by_category)
        if goal_position is None:
            continue

        episodes.append(
            {
                "episode_id": f"synthetic-{len(episodes)}",
                "scene_id": str(episode["scene_id"]),
                "start_position": episode["start_position"],
                "start_rotation": episode["start_rotation"],
                "goals": [{"position": goal_position, "radius": 3.0}],
                "reference_path": [episode["start_position"], goal_position],
                "instruction": {
                    "instruction_text": instruction_text,
                    "instruction_tokens": token_ids,
                },
                "trajectory_id": len(episodes),
            }
        )
        if len(episodes) == count:
            break

    if not episodes:
        raise ValueError(
            f"no reusable source episode in {source_episodes}: needs goals_by_category "
            "with object view points away from the start pose"
        )

    target_dir = Path(out_dir) / split
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{split}.json.gz"
    payload = {"instruction_vocab": {"word_list": vocab}, "episodes": episodes}
    with gzip.open(target, "wt") as handle:
        json.dump(payload, handle)

    return target
