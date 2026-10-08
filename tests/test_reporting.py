import csv
from pathlib import Path

from objectnav_eval.reporting import write_results


def test_writer_preserves_episode_identity_timing_and_config(tmp_path: Path):
    config_path = tmp_path / "source.yaml"
    config_path.write_text("benchmark:\n  split: val_mini\n", encoding="utf-8")
    output = tmp_path / "results"
    results = [
        {
            "episode_index": 0,
            "episode_id": "7",
            "episode_uid": "scene:stable-id",
            "scene_id": "/datasets/mp3d/scene/scene.glb",
            "num_steps": 12,
            "elapsed_seconds": 1.25,
            "metrics": {"success": 1.0},
        }
    ]

    write_results(output, results, {"agent_entrypoint": "agent:Agent"}, config_path)

    with (output / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["episode_uid"] == "scene:stable-id"
    assert row["elapsed_seconds"] == "1.25"
    assert (output / "evaluator_config.yaml").read_text(encoding="utf-8") == (
        "benchmark:\n  split: val_mini\n"
    )
