# MP3D ObjectNav Evaluator

A small, method-agnostic evaluator for MP3D ObjectNav on Habitat-Lab. The
evaluator owns environment creation, episode iteration, observation filtering,
action validation, Habitat metrics, and result serialization. Recognition,
mapping, memory, exploration, planning, and policies remain inside each agent.

`val_mini` is a smoke-test split. Results from it are not formal benchmark
results.

## Reference environment

The verified reference environment is:

- Python 3.9
- Habitat-Lab 0.2.4
- Habitat-Sim 0.2.4
- MP3D ObjectNav v1 episodes and licensed Matterport3D scenes

Habitat and Habitat-Sim are intentionally not package dependencies: install the
evaluator into an environment where those matching packages already work.

## Installation

```bash
git clone git@github.com:vvv257/mp3d-objectnav-eval.git
cd mp3d-objectnav-eval
conda activate objectnav-eval
python -m pip install -e ".[test]"
pytest -q
```

Confirm the simulator versions before comparing runs:

```bash
python -c 'import habitat, habitat_sim; print(habitat.__version__, habitat_sim.__version__)'
```

## Data and configuration

The portable configs assume Habitat's conventional repository-relative layout:

```text
data/
  datasets/objectnav/mp3d/v1/val/val.json.gz
  datasets/objectnav/mp3d/v1/val/content/*.json.gz
  datasets/objectnav/mp3d/v1/val_mini/val_mini.json.gz
  datasets/objectnav/mp3d/v1/val_mini/content/*.json.gz
  scene_datasets/mp3d/<scene-id>/<scene-id>.glb
```

MP3D assets and episode files are not redistributed. Obtain them under their
original terms. If your files live elsewhere, make a private config:

```bash
cp configs/mp3d_objectnav_val_mini.yaml configs/mp3d_objectnav_val_mini.local.yaml
```

Edit `benchmark.data_path` and `benchmark.scenes_dir` in the `.local.yaml`
file. Local configs are ignored by Git. `data_path` may contain `{split}`;
the evaluator substitutes the configured split before giving the override to
Habitat.

## Agent interface

An agent is a no-argument class or factory loaded from `module:attribute`. It
must return one action per observation:

```python
from typing import Any, Mapping

from objectnav_eval import ObjectNavAgent
from objectnav_eval.agent import EpisodeInfo


class MyAgent(ObjectNavAgent):
    def reset(self, episode: EpisodeInfo) -> None:
        self.state = None

    def act(self, observation: Mapping[str, Any]) -> str:
        # All method-specific perception, mapping, and planning live here.
        return "move_forward"
```

The default allowed observation keys are `rgb`, `depth`, `gps`, `compass`, and
`objectgoal`. The evaluator never passes the Habitat environment, simulator
state, goal coordinates, goal viewpoints, navmesh, or semantic annotations to
the agent. `EpisodeInfo` contains only the runtime episode ID, scene ID, and a
stable evaluator-generated episode UID.

Allowed action strings are `stop`, `move_forward`, `turn_left`, `turn_right`,
`look_up`, and `look_down`. Invalid actions fail the run before they reach
Habitat. Agent dependencies and configuration belong to the method package;
the evaluator does not import or special-case RandomAgent.

## Usage

Smoke-test the installation from the repository root:

```bash
objectnav-eval \
  --agent examples.random_agent:RandomAgent \
  --config configs/mp3d_objectnav_val_mini.local.yaml \
  --output results/random_val_mini_smoke \
  --num-episodes 1
```

Evaluate another installed method in the same way:

```bash
objectnav-eval \
  --agent my_method.agent:MyObjectNavAgent \
  --config configs/mp3d_objectnav_v1.local.yaml \
  --output results/my_method_val
```

`--num-episodes` accepts a positive integer. Omitting it evaluates every
episode exposed by the configured split, in deterministic dataset order.

## Metrics

Metrics come directly from `Habitat.Env.get_metrics()`; this project does not
reimplement them. With the supplied Habitat 0.2.4 ObjectNav configuration:

- `distance_to_goal` is Habitat's geodesic distance to the nearest valid goal.
- `success` is 1 only when the agent calls `stop` within the configured success
  distance (0.2 m in the reference Habitat configuration).
- `spl` is success weighted by the ratio of shortest-path length to the longer
  of shortest-path and traveled-path lengths.
- `soft_spl` replaces binary success with normalized progress toward the goal,
  then applies the same path-efficiency weighting.

The result summary averages every scalar metric returned by Habitat. Compare
methods only when their evaluator config, Habitat versions, dataset split,
episode order, step limit, sensors, action space, and success distance match.

## Output format

Each output directory contains:

```text
episodes.jsonl         # one nested record per episode
episodes.csv           # flattened identity, timing, and scalar metrics
summary.json           # metric means, episode count, average steps
run_metadata.json      # entrypoint, versions, config hash, overrides, platform
evaluator_config.yaml  # exact input YAML used for the run
```

Example JSONL record:

```json
{
  "episode_index": 0,
  "episode_id": "0",
  "episode_uid": "x8F5xyUWy9e:0123456789abcdef",
  "scene_id": "data/scene_datasets/mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb",
  "num_steps": 123,
  "elapsed_seconds": 4.12,
  "metrics": {
    "distance_to_goal": 0.1,
    "success": 1.0,
    "spl": 0.72,
    "soft_spl": 0.78
  }
}
```

For a published experiment, also archive the evaluated method's commit,
dependency lockfile, model/config hashes, random seeds, and hardware details;
the evaluator cannot infer those from an entrypoint string.

See [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) for the comparison rules.
