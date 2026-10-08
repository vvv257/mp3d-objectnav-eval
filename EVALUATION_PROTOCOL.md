# MP3D ObjectNav Evaluation Protocol (v0.1)

This document freezes the public contract of the evaluator. Method-specific
recognition, mapping, memory, exploration, planning, and policy logic are part
of the **agent**, not the evaluator.

## Benchmark

- Task: Habitat ObjectNav (`ObjectNav-v1`)
- Scene dataset: Matterport3D (MP3D)
- Episode dataset: MP3D ObjectNav v1
- Default split: `val`
- Reference Habitat config: `benchmark/nav/objectnav/objectnav_mp3d.yaml`
- Default maximum episode length: 500 steps
- Episode order: deterministic (`shuffle=False`)
- Reference runtime: Python 3.9, Habitat-Lab 0.2.4, Habitat-Sim 0.2.4

The repository does not redistribute MP3D assets or ObjectNav episode files.
Users must obtain them under their original licenses/terms.

## Agent-visible information

The evaluator passes only these observation keys by default:

- `rgb`
- `depth`
- `gps`
- `compass`
- `objectgoal`

The evaluator deliberately does **not** pass the Habitat environment object to
the agent.

## Privileged information

An evaluated method must not use simulator-only ground truth to choose actions,
including target coordinates, goal viewpoints, semantic scene annotations,
navmesh queries, shortest-path queries, or simulator state not present in the
allowed observation set.

Privileged information may be used internally by Habitat to compute benchmark
metrics.

## Agent API

Every method must implement:

```python
class MyAgent(ObjectNavAgent):
    def reset(self, episode):
        ...

    def act(self, observation):
        return "move_forward"
```

Allowed action names in v0.1 are:

- `stop`
- `move_forward`
- `turn_left`
- `turn_right`
- `look_up`
- `look_down`

## Metrics

Metrics are read from `Habitat.Env.get_metrics()` rather than reimplemented in
this repository. The expected ObjectNav measurements are:

- `success`
- `spl`
- `soft_spl`
- `distance_to_goal`

The exact active measurements are determined by the frozen Habitat benchmark
configuration and are recorded in the output.

## Reproducibility

Every evaluation run must retain:

- the exact evaluator YAML and its SHA-256 hash
- Habitat config path
- all Habitat overrides
- agent entrypoint
- evaluator, Habitat-Lab, Habitat-Sim, Python, and platform versions
- per-episode metrics

Before publishing cross-method numbers, all compared methods must use the same
protocol file and evaluator configuration.

`val_mini` runs are installation smoke tests only and must not be reported as
formal MP3D ObjectNav benchmark results.
