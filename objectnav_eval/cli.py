from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from objectnav_eval import __version__
from objectnav_eval.backends.habitat import HabitatBackend
from objectnav_eval.config import habitat_overrides, load_yaml
from objectnav_eval.evaluator import EvaluationConfig, Evaluator
from objectnav_eval.loading import load_agent
from objectnav_eval.reporting import write_results


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _git_state(repo_root: Path | None = None) -> dict[str, object]:
    root = repo_root or Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return {"commit": None, "dirty": None}

    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}

    return {"commit": commit, "dirty": bool(status.strip())}


def build_run_metadata(
    *,
    agent_entrypoint: str,
    evaluator_config: str | Path,
    habitat_config: str,
    overrides: Sequence[str],
) -> dict[str, object]:
    config_path = Path(evaluator_config).resolve()
    git_state = _git_state()
    return {
        "agent_entrypoint": agent_entrypoint,
        "evaluator_version": __version__,
        "evaluator_git_commit": git_state["commit"],
        "evaluator_git_dirty": git_state["dirty"],
        "evaluator_config": str(config_path),
        "evaluator_config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "habitat_config": habitat_config,
        "habitat_overrides": list(overrides),
        "habitat_lab_version": _distribution_version("habitat-lab"),
        "habitat_sim_version": _distribution_version("habitat-sim"),
        "python": sys.version,
        "platform": platform.platform(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an agent on MP3D ObjectNav")
    parser.add_argument("--agent", required=True, help="module:ClassOrFactory")
    parser.add_argument("--config", required=True, help="Evaluator YAML")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--num-episodes", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    benchmark = config["benchmark"]
    evaluation = config.get("evaluation", {})
    overrides = habitat_overrides(config)

    evaluation_config = EvaluationConfig(
        allowed_observations=tuple(
            evaluation.get(
                "allowed_observations",
                ["rgb", "depth", "gps", "compass", "objectgoal"],
            )
        ),
        max_episodes=args.num_episodes,
        strict_actions=bool(evaluation.get("strict_actions", True)),
    )
    agent = load_agent(args.agent)
    metadata = build_run_metadata(
        agent_entrypoint=args.agent,
        evaluator_config=args.config,
        habitat_config=benchmark["habitat_config"],
        overrides=overrides,
    )
    backend = HabitatBackend(
        config_path=benchmark["habitat_config"],
        overrides=overrides,
    )
    evaluator = Evaluator(backend, evaluation_config)
    results = evaluator.evaluate(agent)

    summary = write_results(args.output, results, metadata, args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
