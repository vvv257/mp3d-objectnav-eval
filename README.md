# mp3d-objectnav-eval

A small, method-agnostic evaluator for **MP3D ObjectNav** on Habitat-Lab.

The goal is to make different ObjectNav methods comparable without forcing
them to share mapping, recognition, exploration, planning, or policy code.
Each method implements the same tiny agent interface; the evaluator owns the
environment, episodes, allowed observations, action validation, official
Habitat metrics, and result serialization.

## Design

```text
MP3D ObjectNav episodes
        |
        v
  HabitatBackend
        |
   observation
        v
  ObjectNavAgent     <- method-specific recognition / mapping / planning
        |
      action
        v
  HabitatBackend
        |
        v
 Habitat metrics -> episodes.jsonl / episodes.csv / summary.json
```

The evaluator filters observations before they reach the agent, which helps
avoid accidental use of simulator-only ground truth.

## What this repository does not implement

It does not provide a detector, mapper, frontier explorer, planner, VLM, LLM,
or RL policy. Those are part of the method being evaluated.

## Status

**v0.1 scaffold.** The core evaluator, backend abstraction, result writer,
CLI, protocol document, fake-backend tests, and RandomAgent are implemented.
A real MP3D run additionally requires a working Habitat-Lab/Habitat-Sim setup
and licensed MP3D data.

## 1. Install the evaluator

```bash
pip install -e ".[test]"
pytest
```

The core package intentionally does not install Habitat for you, because
Habitat/Habitat-Sim versions are normally managed in their own conda
environment.

## 2. Prepare Habitat + MP3D

The reference evaluator config expects:

```text
data/
  datasets/objectnav/mp3d/v1/val/val.json.gz
  datasets/objectnav/mp3d/v1/val/content/...
  scene_datasets/mp3d/...
```

Habitat-Lab itself still exposes MP3D ObjectNav v1 as `ObjectNav-v1` with the
standard path `data/datasets/objectnav/mp3d/v1/{split}/{split}.json.gz`.

A practical initial compatibility pin is:

```text
habitat-lab==0.3.20231024
```

This is a reference engineering pin used by an existing public MP3D ObjectNav
evaluator; once your team confirms its exact environment, freeze the complete
Habitat-Lab + Habitat-Sim + Python/CUDA combination.

## 3. Implement an agent

```python
from objectnav_eval.agent import ObjectNavAgent

class MyAgent(ObjectNavAgent):
    def reset(self, episode):
        self.map = None

    def act(self, obs):
        # recognition / mapping / exploration / planning all live here
        return "move_forward"
```

Do not request the Habitat `env`, target coordinates, navmesh, goal viewpoints,
or GT semantics from the evaluator.

## 4. Smoke test with RandomAgent

Run from the repository root, inside the Habitat environment:

```bash
objectnav-eval \
  --agent examples.random_agent:RandomAgent \
  --config configs/mp3d_objectnav_v1.yaml \
  --output results/random \
  --num-episodes 5
```

The output directory contains:

```text
results/random/
  episodes.jsonl
  episodes.csv
  summary.json
  run_metadata.json
```

## 5. Evaluate another method

If a method repository exposes:

```text
my_method.agent:MyObjectNavAgent
```

install this evaluator into that method's Python environment, then run:

```bash
objectnav-eval \
  --agent my_method.agent:MyObjectNavAgent \
  --config /path/to/mp3d-objectnav-eval/configs/mp3d_objectnav_v1.yaml \
  --output results/my_method
```

This keeps the public agent contract stable while letting each method manage
its own model dependencies.

## Result schema

Per-episode records look like:

```json
{
  "episode_index": 0,
  "episode_id": "...",
  "scene_id": "...",
  "num_steps": 123,
  "elapsed_seconds": 4.12,
  "metrics": {
    "distance_to_goal": 0.0,
    "success": 1.0,
    "spl": 0.72,
    "soft_spl": 0.78
  }
}
```

`summary.json` averages every scalar metric reported by Habitat and also adds
`num_episodes` and `avg_num_steps`.

## Why not reimplement SPL/Success?

Habitat already owns ObjectNav task semantics and metrics. This project treats
`env.get_metrics()` as the source of truth and focuses on the missing glue:
standardizing the protocol, method interface, execution, and output format.

## Next milestones

1. Freeze the exact team environment (Habitat-Lab, Habitat-Sim, Python/CUDA).
2. Verify one real MP3D val episode end-to-end.
3. Save a copy/hash of the fully resolved Habitat config per run.
4. Add crash handling and resumable episode evaluation.
5. Add sharding + merge for parallel MP3D evaluation.
6. Add a process-isolated/ZMQ transport when method dependency conflicts make
   in-process execution inconvenient.

See [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) before comparing methods.
