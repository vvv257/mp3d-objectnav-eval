from __future__ import annotations

from collections import defaultdict
from numbers import Number
from typing import Any, Iterable, Mapping


def summarize_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    metric_values: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        for key, value in row.get("metrics", {}).items():
            if isinstance(value, Number):
                metric_values[key].append(float(value))

    summary: dict[str, Any] = {"num_episodes": len(rows)}
    for key, values in sorted(metric_values.items()):
        if values:
            summary[key] = sum(values) / len(values)

    if rows:
        summary["avg_num_steps"] = sum(
            float(row.get("num_steps", 0)) for row in rows
        ) / len(rows)

    return summary
