from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objectnav_eval.reporting.summary import summarize_results


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_results(
    output_dir: str | Path,
    results: Sequence[Mapping[str, Any]],
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary = summarize_results(results)

    with (output / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, default=_json_default) + "\n")

    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    with (output / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(dict(run_metadata), f, indent=2, default=_json_default)

    flat_keys = ["episode_index", "episode_id", "scene_id", "num_steps"]
    metric_keys = sorted(
        {
            key
            for row in results
            for key in row.get("metrics", {}).keys()
        }
    )
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys + metric_keys)
        writer.writeheader()
        for row in results:
            record = {key: row.get(key) for key in flat_keys}
            record.update(row.get("metrics", {}))
            writer.writerow(record)

    return summary
