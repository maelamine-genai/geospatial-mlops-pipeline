# Pipeline Walkthrough

This walkthrough describes the full local workflow. The same stage contracts are intended to be usable by larger orchestrators such as Argo Workflows.

## 0. Start MLflow locally

```bash
mkdir -p /tmp/geo_mlops_mlruns

mlflow server \
  --backend-store-uri sqlite:////tmp/geo_mlops_mlflow.db \
  --default-artifact-root /tmp/geo_mlops_mlruns \
  --host 127.0.0.1 \
  --port 5000
```

## 1. One-command pipeline

```bash
python -m geo_mlops.cli.run_pipeline \
  --task building_seg \
  --task-cfg src/geo_mlops/tasks/segmentation/building/config/default.yaml \
  --dataset-root /path/to/train_val_dataset_root \
  --golden-root /path/to/golden_full_scene_dataset_root \
  --run-dir /path/to/output/run \
  --csv-name bldg_tiles_regular.csv \
  --mlflow \
  --mlflow-tracking-uri http://127.0.0.1:5000 \
  --mlflow-experiment building_seg_debug \
  --force-tiling
```

## 2. Run stages manually

The pipeline can also be debugged stage by stage.

### Tile

```bash
python -m geo_mlops.cli.tile \
  --task building_seg \
  --task-cfg src/geo_mlops/tasks/segmentation/building/config/default.yaml \
  --dataset-root /path/to/train_val_dataset_root \
  --csv-name bldg_tiles_regular.csv \
  --out-dir /path/to/run/tiles \
  --force
```

If `--dataset-buckets` is omitted, all immediate subdirectories under `--dataset-root` are processed.

### Split

```bash
python -m geo_mlops.cli.split \
  --task building_seg \
  --task-cfg src/geo_mlops/tasks/segmentation/building/config/default.yaml \
  --tiles-dir /path/to/run/tiles \
  --out-dir /path/to/run/split
```

### Train

```bash
python -m geo_mlops.cli.train \
  --task building_seg \
  --task-cfg src/geo_mlops/tasks/segmentation/building/config/default.yaml \
  --tiles-dir /path/to/run/tiles \
  --split-dir /path/to/run/split \
  --out-dir /path/to/run/train \
  --mlflow \
  --mlflow-tracking-uri http://127.0.0.1:5000 \
  --mlflow-experiment building_seg_debug
```

### Gate A

```bash
python -m geo_mlops.cli.gate \
  --task building_seg \
  --task-cfg src/geo_mlops/tasks/segmentation/building/config/default.yaml \
  --gate-name gate_a \
  --metrics-file /path/to/run/train/metrics.json \
  --out-dir /path/to/run/gate_a \
  --train-manifest /path/to/run/train/train_manifest.json \
  --split-json /path/to/run/split/split.json \
  --tiles-manifest /path/to/run/tiles/tiles_manifest.json
```

### Register candidate

```bash
python -m geo_mlops.cli.register \
  --task building_seg \
  --task-cfg src/geo_mlops/tasks/segmentation/building/config/default.yaml \
  --action register-candidate \
  --gate-contract /path/to/run/gate_a/gate_decision.json \
  --out-dir /path/to/run/registry_candidate
```

`register.py` can infer the MLflow run ID from `train_manifest.json` if the Gate A contract includes the upstream training manifest path.

### Golden full-scene evaluation

```bash
python -m geo_mlops.cli.evaluate \
  --task building_seg \
  --task-cfg src/geo_mlops/tasks/segmentation/building/config/default.yaml \
  --dataset-root /path/to/golden_full_scene_dataset_root \
  --train-manifest /path/to/run/train/train_manifest.json \
  --out-dir /path/to/run/golden_eval
```

This stage does not use tile splits. It loads full scenes, runs sliding-window inference, saves full-scene outputs, and computes golden-set metrics.

### Gate B

```bash
python -m geo_mlops.cli.gate \
  --task building_seg \
  --task-cfg src/geo_mlops/tasks/segmentation/building/config/default.yaml \
  --gate-name gate_b \
  --metrics-file /path/to/run/golden_eval/eval_summary.json \
  --out-dir /path/to/run/gate_b \
  --eval-manifest /path/to/run/golden_eval/eval_manifest.json \
  --train-manifest /path/to/run/train/train_manifest.json
```

### Promote production

```bash
python -m geo_mlops.cli.register \
  --task building_seg \
  --task-cfg src/geo_mlops/tasks/segmentation/building/config/default.yaml \
  --action promote-production \
  --gate-contract /path/to/run/gate_b/gate_decision.json \
  --model-version <candidate-version> \
  --out-dir /path/to/run/registry_production
```

`run_pipeline.py` automatically reads the candidate version from `registry_candidate/registry_result.json`.

