from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# FastDTW helpers below are a dependency-free adaptation of slaypni/fastdtw
# (MIT, copyright 2015 Kazuaki Tanida); see THIRD_PARTY_NOTICES.md.


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("DTW locations must have equal dimensions")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _dtw(
    x: Sequence[Sequence[float]],
    y: Sequence[Sequence[float]],
    window: Iterable[tuple[int, int]] | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    if not x or not y:
        raise ValueError("DTW paths must not be empty")
    if window is None:
        window = ((i, j) for i in range(len(x)) for j in range(len(y)))
    costs: defaultdict[tuple[int, int], tuple[float, int, int]] = defaultdict(
        lambda: (float("inf"), 0, 0)
    )
    costs[0, 0] = (0.0, 0, 0)
    for raw_i, raw_j in window:
        i, j = raw_i + 1, raw_j + 1
        distance = _euclidean(x[i - 1], y[j - 1])
        previous = min(
            (costs[i - 1, j][0] + distance, i - 1, j),
            (costs[i, j - 1][0] + distance, i, j - 1),
            (costs[i - 1, j - 1][0] + distance, i - 1, j - 1),
            key=lambda item: item[0],
        )
        costs[i, j] = previous
    path: list[tuple[int, int]] = []
    i, j = len(x), len(y)
    while (i, j) != (0, 0):
        path.append((i - 1, j - 1))
        _, i, j = costs[i, j]
    path.reverse()
    return costs[len(x), len(y)][0], path


def _reduce_by_half(path: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [(float(a) + float(b)) / 2 for a, b in zip(path[i], path[i + 1])]
        for i in range(0, len(path) - len(path) % 2, 2)
    ]


def _expand_window(
    path: Sequence[tuple[int, int]], len_x: int, len_y: int, radius: int
) -> list[tuple[int, int]]:
    expanded = set(path)
    for i, j in path:
        for a in range(-radius, radius + 1):
            for b in range(-radius, radius + 1):
                expanded.add((i + a, j + b))
    doubled = set()
    for i, j in expanded:
        doubled.update(
            ((i * 2, j * 2), (i * 2, j * 2 + 1),
             (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1))
        )
    window: list[tuple[int, int]] = []
    start_j = 0
    for i in range(len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in doubled:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        if new_start_j is not None:
            start_j = new_start_j
    return window


def _fast_dtw(
    x: Sequence[Sequence[float]], y: Sequence[Sequence[float]], radius: int
) -> tuple[float, list[tuple[int, int]]]:
    if len(x) < radius + 2 or len(y) < radius + 2:
        return _dtw(x, y)
    _, path = _fast_dtw(_reduce_by_half(x), _reduce_by_half(y), radius)
    return _dtw(x, y, _expand_window(path, len(x), len(y), radius))


def fast_dtw_distance(
    x: Sequence[Sequence[float]], y: Sequence[Sequence[float]], radius: int = 1
) -> float:
    """Match VLN-CE's ``fastdtw(..., radius=1, dist=euclidean)`` call."""

    if type(radius) is not int or radius < 0:
        raise ValueError("fast_dtw_radius must be a non-negative integer")
    return _fast_dtw(x, y, radius)[0]


class VlnCeOfficialMetrics:
    """Accumulate the historical R2R-VLNCE metrics outside the Agent API.

    The formulas mirror ``jacobkrantz/VLN-CE``. Simulator state is consumed
    only inside the evaluator and is never added to Agent observations.
    """

    provided_metrics = frozenset(
        {
            "success", "spl", "distance_to_goal", "navigation_error",
            "oracle_success", "path_length", "steps_taken", "ndtw", "sdtw",
        }
    )

    def __init__(
        self, *, env: Any, options: Mapping[str, Any], parameters: Mapping[str, Any]
    ) -> None:
        self._env = env
        self._success_distance = float(parameters.get("success_distance", 3.0))
        self._radius = parameters.get("fast_dtw_radius", 1)
        if not math.isfinite(self._success_distance) or self._success_distance <= 0:
            raise ValueError("success_distance must be finite and positive")
        if type(self._radius) is not int or self._radius < 0:
            raise ValueError("fast_dtw_radius must be a non-negative integer")
        raw_path = options.get("ground_truth_path")
        self._gt_path = Path(str(raw_path)).expanduser() if raw_path else None
        if self._gt_path is None or not self._gt_path.is_file():
            raise FileNotFoundError(
                f"VLN-CE ground truth file not found: {raw_path!r}"
            )
        with gzip.open(self._gt_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("VLN-CE ground truth root must be a mapping")
        self._ground_truth = payload
        self._metric_version = str(
            parameters.get("metric_version", "vlnce-official-fastdtw-v1")
        )

    def _position(self) -> list[float]:
        values = [float(value) for value in self._env.sim.get_agent_state().position]
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("Simulator returned an invalid agent position")
        return values

    def _distance(self) -> float:
        value = float(self._env.get_metrics()["distance_to_goal"])
        if not math.isfinite(value):
            raise ValueError("Simulator returned an invalid distance_to_goal")
        return value

    def reset(self, episode: Any) -> None:
        episode_id = str(episode.episode_id)
        if episode_id not in self._ground_truth:
            raise ValueError(
                f"episode_id {episode_id!r} not found in VLN-CE ground truth"
            )
        locations = self._ground_truth[episode_id].get("locations")
        if not isinstance(locations, list) or not locations:
            raise ValueError(f"Invalid ground-truth locations for episode_id {episode_id!r}")
        self._gt_locations = [[float(item) for item in point] for point in locations]
        current = self._position()
        self._locations = [current]
        self._previous_position = current
        self._path_length = 0.0
        self._steps = 0
        self._stop_called = False
        self._start_distance = self._distance()
        self._minimum_distance = self._start_distance

    def update(self, action: str) -> None:
        self._steps += 1
        current = self._position()
        self._path_length += _euclidean(self._previous_position, current)
        self._previous_position = current
        if current != self._locations[-1]:
            self._locations.append(current)
        distance = self._distance()
        self._minimum_distance = min(self._minimum_distance, distance)
        self._stop_called = self._stop_called or action == "stop"

    def get_metrics(self) -> dict[str, float | int]:
        distance = self._distance()
        success = float(self._stop_called and distance < self._success_distance)
        oracle_success = float(self._minimum_distance < self._success_distance)
        ndtw = math.exp(
            -fast_dtw_distance(self._locations, self._gt_locations, self._radius)
            / (len(self._gt_locations) * self._success_distance)
        )
        spl = success * self._start_distance / max(
            self._start_distance, self._path_length
        )
        return {
            "success": success,
            "spl": spl,
            "distance_to_goal": distance,
            "navigation_error": distance,
            "oracle_success": oracle_success,
            "path_length": self._path_length,
            "steps_taken": self._steps,
            "ndtw": ndtw,
            "sdtw": success * ndtw,
        }

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "metric_provider": type(self).__module__ + ":" + type(self).__name__,
            "metric_version": self._metric_version,
            "success_distance": self._success_distance,
            "fast_dtw_radius": self._radius,
            "ground_truth_path": str(self._gt_path.resolve()),
            "ground_truth_sha256": hashlib.sha256(self._gt_path.read_bytes()).hexdigest(),
        }
