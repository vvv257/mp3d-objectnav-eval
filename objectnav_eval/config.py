from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of config: {path}")

    return data


def habitat_overrides(config: dict[str, Any]) -> list[str]:
    benchmark = config.get("benchmark", {})
    overrides = list(benchmark.get("overrides", []))

    split = benchmark.get("split")

    if split:
        overrides.append(
            f"habitat.dataset.split={split}"
        )

    data_path = benchmark.get("data_path")

    if data_path:
        # Allow reusable paths such as:
        # data/datasets/objectnav/mp3d/v1/{split}/{split}.json.gz
        if split:
            data_path = str(data_path).format(split=split)

        overrides.append(
            f"habitat.dataset.data_path={data_path}"
        )

    if benchmark.get("scenes_dir"):
        overrides.append(
            f"habitat.dataset.scenes_dir={benchmark['scenes_dir']}"
        )

    if benchmark.get("max_episode_steps") is not None:
        overrides.append(
            f"habitat.environment.max_episode_steps="
            f"{benchmark['max_episode_steps']}"
        )

    if benchmark.get("seed") is not None:
        overrides.append(
            f"habitat.seed={benchmark['seed']}"
        )

    return overrides
