# Building Segmentation Task

`building_seg` is the reference task plugin for the MLOps pipeline.

It demonstrates how a concrete geospatial segmentation task plugs into the task-agnostic lifecycle.

## Layout

```text
building/
  task.py              # public plugin facade
  config/              # runnable task configs
  tiling/              # building tiling adapter and factory
  data/                # tile dataset and train/val dataset builders
  modeling/            # model, forward function, losses, metrics
  evaluation/          # golden full-scene eval hooks
```

## Key behavior

- Tiling emits all valid tiles and annotates train sampling decisions with columns like `sample__include`.
- Splitting is group-aware and deterministic.
- Training filters only the train partition by sampling policy; validation remains unfiltered.
- Golden evaluation is full-scene based, not tile/split based.
- Evaluation writes full-scene probability masks, binary masks, per-scene metrics, and Pareto/hardest-image tables.
