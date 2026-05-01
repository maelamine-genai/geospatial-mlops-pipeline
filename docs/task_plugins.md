# Task Plugins

Task plugins are the reason the repository can be task-agnostic without becoming generic to the point of being unusable.

The core pipeline owns the lifecycle. A task plugin owns task semantics.

![Task plugin architecture](diagrams/task_plugin_detail.png)

## Public plugin surface

A task plugin is loaded through `core/registry/task_registry.py`. The current reference task is:

```text
building_seg -> geo_mlops.tasks.segmentation.building.task:BuildingSegmentationTask
```

The plugin exposes methods such as:

```text
build_tiling_components(...)
build_training_cfg(...)
build_model(...)
build_loss(...)
build_metrics_fn(...)
build_forward_fn(...)
build_train_val_datasets(...)
build_evaluation_cfg(...)
iter_eval_scenes(...)
load_eval_scene(...)
build_eval_postprocessor(...)
save_eval_prediction(...)
build_eval_metric_accumulator(...)
load_checkpoint(...)
```

The core engines call those methods without importing task-specific classes directly.

## Building segmentation task layout

```text
tasks/segmentation/building/
  task.py
    Public facade loaded by the registry.

  config/
    default.yaml
    all-tiles.yaml
    hard-mining.yaml

  tiling/
    adapter.py
      Building-specific tiling adapter and CSV columns.
    factory.py
      Converts config into EngineConfig, adapter, and policy.

  data/
    dataset.py
      Tile-row dataset for training/validation.
    train_data.py
      Builds train/val datasets and applies train-only sampling.

  modeling/
    factory.py
      Builds the model.
    forward.py
      Adapts batch dictionaries to model calls.
    losses.py
      Building-specific loss function.
    metrics.py
      Training metrics and golden-evaluation accumulator.

  evaluation/
    eval.py
      Full-scene golden evaluation hooks: scene discovery, loading,
      postprocessing, saving rasters, and checkpoint loading.
```

## Why `task.py` is a facade

The task facade intentionally delegates to smaller task-local modules. This prevents `task.py` from becoming overloaded while preserving a clean registry-facing interface.

```text
task.py -> tiling/factory.py
        -> data/train_data.py
        -> modeling/factory.py
        -> modeling/losses.py
        -> modeling/metrics.py
        -> evaluation/eval.py
```

## Shared segmentation helper

The package includes:

```text
tasks/segmentation/segmentation_adapter.py
```

This is an intermediate helper between core's `BaseAdapter` and concrete segmentation tasks:

```text
BaseAdapter -> SegmentationAdapter -> BuildingSegmentationAdapter
```

`SegmentationAdapter` provides reusable segmentation behavior such as foreground-mask extraction, foreground presence, and generic prediction-vs-ground-truth difficulty. Concrete tasks override only what is task-specific.

## Adding a new task

A new segmentation task can follow the same shape:

```text
tasks/segmentation/water/
  task.py
  config/
  tiling/
  data/
  modeling/
  evaluation/
```

Then register it in `core/registry/task_registry.py`:

```python
_TASKS = {
    "building_seg": TaskSpec(...),
    "water_seg": TaskSpec(
        name="water_seg",
        plugin_path="geo_mlops.tasks.segmentation.water.task:WaterSegmentationTask",
    ),
}
```

The core pipeline should not need to change.

