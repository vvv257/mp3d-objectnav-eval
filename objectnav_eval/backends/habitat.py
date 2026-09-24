from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objectnav_eval.agent import EpisodeInfo
from objectnav_eval.backends.base import EvaluationBackend


def _stable_episode_uid(episode: Any) -> str:
    """Build a machine-independent ID for one ObjectNav episode.

    Habitat ObjectNav-v1 rewrites episode_id while loading datasets, so the
    runtime episode_id alone is not a stable identifier across dataset files.

    We instead identify an episode using task-defining information:
    scene + start pose + object category.
    """

    scene_path = Path(str(episode.scene_id))

    # Keep only:
    #   scene_id/scene_id.glb
    #
    # This removes machine-specific prefixes such as /home/user/datasets/.
    if len(scene_path.parts) >= 2:
        scene_key = "/".join(scene_path.parts[-2:])
    else:
        scene_key = scene_path.name

    def clean_floats(values: Any) -> list[float]:
        return [round(float(value), 8) for value in values]

    payload = {
        "scene": scene_key,
        "start_position": clean_floats(episode.start_position),
        "start_rotation": clean_floats(episode.start_rotation),
        "object_category": str(
            getattr(episode, "object_category", "")
        ),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256(encoded).hexdigest()[:16]

    scene_name = scene_path.stem

    return f"{scene_name}:{digest}"


class HabitatBackend(EvaluationBackend):
    """Thin adapter over Habitat-Lab's Env."""

    def __init__(
        self,
        config_path: str,
        overrides: Sequence[str] | None = None,
    ) -> None:
        try:
            import habitat
        except ImportError as exc:
            raise RuntimeError(
                "Habitat-Lab is not installed in this environment. "
                "Install the project inside a working Habitat environment."
            ) from exc

        self._habitat = habitat
        self._config_path = str(Path(config_path))
        self._overrides = list(overrides or [])

        self._config = habitat.get_config(
            config_path=self._config_path,
            overrides=self._overrides,
        )

        self._env = habitat.Env(config=self._config)

    @property
    def episode_count(self) -> int:
        return len(self._env.episodes)

    @property
    def episode_over(self) -> bool:
        return bool(self._env.episode_over)

    def reset(self) -> tuple[EpisodeInfo, Mapping[str, Any]]:
        observation = self._env.reset()
        episode = self._env.current_episode

        info = EpisodeInfo(
            episode_id=str(episode.episode_id),
            scene_id=str(episode.scene_id),
            episode_uid=_stable_episode_uid(episode),
        )

        return info, observation

    def step(self, action: str) -> Mapping[str, Any]:
        return self._env.step({"action": action})

    def get_metrics(self) -> Mapping[str, Any]:
        return self._env.get_metrics()

    def close(self) -> None:
        self._env.close()
