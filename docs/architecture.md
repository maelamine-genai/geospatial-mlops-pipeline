# Architecture

This repository is organized around a simple design principle: **core owns the lifecycle, tasks own domain semantics**.

![Core/plugin architecture](diagrams/core_plugin_architecture.png)

## Package boundaries

```text
src/geo_mlops/
  cli/
    Thin wrappers around each pipeline stage.

  core/
    Task-agnostic pipeline machinery: contracts, IO, tiling, splitting,
    training, full-scene evaluation, gating, MLflow registry, and utilities.

  models/
    Reusable architecture components such as SegFormer backbones and fusion heads.

  tasks/
    Concrete task plugins. A task plugin teaches the core pipeline how to tile,
    train, evaluate, and score a specific problem.
```

## Why the boundary matters

The core training engine does not know about building masks, context tensors, foreground labels, or segmentation thresholds. It only knows how to run a loop with injected callables:

```text
model, loss_fn, forward_fn, metrics_fn, train_ds, val_ds
```

The core evaluation engine does not know what macro metrics or Pareto analysis mean. It only knows how to:

1. iterate full scenes,
2. run sliding-window inference,
3. stitch scene predictions,
4. call task-provided metric and save hooks,
5. write a summary and manifest.

The task plugin supplies:

```text
tiling adapter
training dataset builder
model/loss/metrics/forward function
full-scene scene discovery and loading
prediction writer
formal evaluation accumulator
```

## CLI orchestration

The full pipeline is wired through `geo_mlops.cli.run_pipeline`:

![Pipeline lifecycle](diagrams/pipeline_lifecycle.png)

The orchestrator is intentionally thin. It calls the same stage CLIs that can be run individually during development:

```text
tile -> split -> train -> gate_a -> register candidate -> evaluate -> gate_b -> promote production
```

This keeps the local developer flow and the production orchestration flow aligned.

## Current reference implementation

The current reference plugin is `building_seg`:

```text
src/geo_mlops/tasks/segmentation/building/
  task.py
  config/
  data/
  evaluation/
  modeling/
  tiling/
```

The plugin proves that a realistic geospatial segmentation task can plug into the same lifecycle without embedding building-specific logic inside `core/`.

## Designed extension points

The most important extension points are:

| Extension | Where to add it | What remains unchanged |
|---|---|---|
| New segmentation task | `tasks/segmentation/<task_name>/` | CLI, contracts, gates, registry flow |
| New model architecture | `models/` or task-local `modeling/` | Training engine, registry, gates |
| New evaluation metrics | task-local `metrics.py` | Core evaluation loop |
| Distributed evaluation | `core/evaluation/ray_engine.py` | Task plugin hooks and eval contracts |
| Distributed training | `core/training/ray_engine.py` | Task plugin model/dataset/loss hooks |

