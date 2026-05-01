# Roadmap

The pipeline is now a working end-to-end local MLOps system. The most valuable extensions are the ones that preserve the current contracts while scaling or enriching the workflow.

## 1. Distributed golden evaluation with Ray

Golden evaluation is the best first distributed target because each full scene can be processed independently:

```text
scene_001 -> sliding-window inference -> per-scene metrics
scene_002 -> sliding-window inference -> per-scene metrics
...
```

A Ray backend can distribute scene inference and merge per-scene metric outputs into the same `eval_summary.json` contract.

## 2. Ray/KubeRay training and sweeps

Training can later support:

- distributed data-parallel training,
- hyperparameter sweeps,
- multi-node execution through KubeRay,
- Argo steps that launch RayJobs.

The existing task plugin and contracts should remain the interface.

## 3. ROI analysis and data recommendation

The pipeline already produces enough evidence to drive data-centric analysis:

- training metrics,
- full-scene evaluation metrics,
- Pareto/hardest-image tables,
- prediction masks and probability maps,
- MLflow run history.

A future `analyze_run` stage can use DINO/SatMAE/Prithvi-style embeddings to cluster ROIs, identify failure modes, and recommend where to collect or label more data.

## 4. Add a second task plugin

A second task proves the plugin interface. Good candidates:

- water segmentation,
- cloud/snow segmentation,
- change detection,
- image-level classification.

The goal is to add a task without modifying core pipeline logic.

## 5. Self-supervised and generative task families

The same lifecycle can extend beyond segmentation:

```text
self-supervised:
  tile/pair data -> train encoder -> evaluate embeddings/linear probe -> gate -> register encoder

generative:
  pair/condition data -> train generator -> evaluate visual/geospatial metrics -> gate -> register generator
```

