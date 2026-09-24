from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from objectnav_eval.backends.habitat import HabitatBackend
from objectnav_eval.config import habitat_overrides, load_yaml
from objectnav_eval.evaluator import EvaluationConfig, Evaluator
from objectnav_eval.loading import load_agent
from objectnav_eval.reporting import write_results


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

    backend = HabitatBackend(
        config_path=benchmark["habitat_config"],
        overrides=habitat_overrides(config),
    )
    evaluator = Evaluator(
        backend,
        EvaluationConfig(
            allowed_observations=tuple(
                evaluation.get(
                    "allowed_observations",
                    ["rgb", "depth", "gps", "compass", "objectgoal"],
                )
            ),
            max_episodes=args.num_episodes,
            strict_actions=bool(evaluation.get("strict_actions", True)),
        ),
    )
    agent = load_agent(args.agent)
    results = evaluator.evaluate(agent)

    metadata = {
        "agent_entrypoint": args.agent,
        "evaluator_config": str(Path(args.config).resolve()),
        "habitat_config": benchmark["habitat_config"],
        "habitat_overrides": habitat_overrides(config),
        "python": sys.version,
        "platform": platform.platform(),
    }
    summary = write_results(args.output, results, metadata)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
