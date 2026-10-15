# MP3D ObjectNav Evaluation Protocol (protocol version 1)

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

Every method must provide callable `reset` and `act` methods. Inheriting
`ObjectNavAgent` remains supported but is optional:

```python
class MyAgent(ObjectNavAgent):
    def reset(self, episode):
        ...

    def act(self, observation):
        return "move_forward"
```

The default ObjectNav task protocol permits these primitive actions:

- `stop`
- `move_forward`
- `turn_left`
- `turn_right`
- `look_up`
- `look_down`

Typed commands `primitive`, `move` (positive meters), and `rotate` (nonzero
degrees, positive left) are supported. Motion commands use the effective
Habitat forward step and turn angle, with half-up quantization
`floor(abs(request) / step_size + 0.5)`. Zero-step quantization, nonfinite
parameters, and expansions over 64 primitives are rejected. Each primitive
consumes one environment step. Environment termination cancels the rest of
the command; the executor does not add stop or compensate for collisions.

Explicit adapters translate action IDs and permitted observations. Observation
filtering occurs before adapter invocation. An adapter cannot change task
physics or success criteria. This API boundary is not a process sandbox:
submitted Agent and custom Adapter code must still be audited for privileged
access. Waypoints, velocity and semantic navigation skills are unsupported.

## Metrics

Metrics are read from `Habitat.Env.get_metrics()` rather than reimplemented in
this repository. The expected ObjectNav measurements are:

- `success`
- `spl`
- `soft_spl`
- `distance_to_goal`

The exact active measurements are determined by the frozen Habitat benchmark
configuration and are recorded in the output.

## Task protocols

A task protocol defines what an episode is allowed to ask for: legal actions,
permitted observations, required metrics, the Habitat config it belongs to, the
episode identity policy, and where motion parameters come from. Two protocols
ship with the evaluator:

- `mp3d_objectnav_v1` (default, protocol version 1): `configs/tasks/mp3d_objectnav_v1.yaml`
- `vln_ce_r2r_v1`: `configs/tasks/vln_ce_r2r.yaml`, mirroring Habitat-Lab 0.2.4
  `benchmark/nav/vln_r2r.yaml` — actions `stop`, `move_forward`, `turn_left`,
  `turn_right`; observations `rgb`, `depth`, `instruction`; metrics `success`,
  `spl`, `distance_to_goal` (that config provides no `ndtw`); episode identity
  from scene, instruction text and `trajectory_id`.

Each task reads its motion parameters and success criteria from the resolved
Habitat configuration; no protocol hard-codes step sizes or thresholds. The
backend refuses a protocol whose actions, observations or metrics the resolved
environment does not provide, and the evaluator refuses to run when its
protocol differs from the one the backend validated.

The VLN-CE protocol has only been exercised on a synthetic dataset built from
real scenes and navmeshes (see README). The official R2R_VLNCE episodes are not
integrated, so `vln_ce_r2r_v1` results are not VLN-CE benchmark results and must
not be compared with published VLN-CE numbers. Results from different task
protocols are not comparable with each other by construction.

## Reproducibility

Every evaluation run must retain:

- the exact evaluator YAML and its SHA-256 hash
- Habitat config path
- all Habitat overrides
- agent entrypoint
- evaluator, Habitat-Lab, Habitat-Sim, Python, and platform versions
- per-episode metrics
- task protocol contents/hash and protocol version
- adapter config/hash, source fingerprint and observation conversion rules
- effective Habitat configuration hash, motion parameters and sensor settings
- the resolved success criteria (`distance_to`, `success_distance`)
- primitive counts, command counts and motion quantization records

Changed task, adapter, converter, code or effective runtime configuration
prevents resume. Existing result files remain readable; runs started with an
older evaluator must be resumed using that version. Source fingerprints are
not complete dependency/model manifests; `reproducibility_complete=false`
records that limitation explicitly.

Before publishing cross-method numbers, all compared methods must use the same
protocol file and evaluator configuration.

`val_mini` runs are installation smoke tests only and must not be reported as
formal MP3D ObjectNav benchmark results.
