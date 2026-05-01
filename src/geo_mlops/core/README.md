# Core Layer

`geo_mlops.core` contains task-agnostic pipeline machinery.

Core is responsible for lifecycle mechanics:

- contracts and serialization,
- tiling engine and stage wrappers,
- splitting logic,
- generic training loop,
- generic full-scene evaluation loop,
- KPI gating,
- MLflow registry transitions,
- shared utilities.

Core should not know task semantics such as:

- what a building/water/cloud class means,
- which label is foreground,
- how a task computes macro metrics,
- how a task discovers golden evaluation scenes.

Those belong in `tasks/`.

## Subpackages

| Package | Responsibility |
|---|---|
| `contracts/` | Dataclasses defining stage outputs. |
| `io/` | Read/write helpers for contracts. |
| `tiling/` | Task-agnostic tiling engine and policy annotation. |
| `splitting/` | Group-aware deterministic splitting. |
| `training/` | Generic training loop and callbacks. |
| `evaluation/` | Generic full-scene sliding-window evaluation. |
| `gating/` | Threshold/KPI decision engine. |
| `registry/` | Task registry and MLflow model registry utilities. |
| `utils/` | Small shared helpers. |
