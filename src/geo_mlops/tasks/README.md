# Tasks

`geo_mlops.tasks` contains task plugins.

A task plugin adapts the core MLOps lifecycle to a concrete ML problem. The current reference task is `building_seg` under `tasks/segmentation/building`.

A task is responsible for:

- task-specific tiling adapter logic,
- dataset construction,
- model/loss/metric construction,
- forward pass adaptation,
- full-scene evaluation scene discovery/loading,
- prediction artifact writing,
- formal task metrics and analytics.

The core pipeline calls a task through the task registry, not by importing task internals directly.
