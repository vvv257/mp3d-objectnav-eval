import hashlib
from pathlib import Path

from objectnav_eval import cli


def test_run_metadata_records_config_hash_and_software_versions(tmp_path: Path):
    config_path = tmp_path / "eval.yaml"
    raw_config = b"benchmark:\n  split: val_mini\n"
    config_path.write_bytes(raw_config)

    metadata = cli.build_run_metadata(
        agent_entrypoint="agent.module:Agent",
        evaluator_config=config_path,
        habitat_config="benchmark/nav/objectnav/objectnav_mp3d.yaml",
        overrides=["habitat.seed=100"],
    )

    assert metadata["evaluator_config_sha256"] == hashlib.sha256(
        raw_config
    ).hexdigest()
    assert metadata["evaluator_version"]
    assert metadata["habitat_lab_version"]
    assert metadata["habitat_sim_version"]
