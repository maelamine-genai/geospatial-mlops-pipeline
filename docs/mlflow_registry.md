# MLflow Tracking and Registry

MLflow is used for two separate responsibilities:

1. **Tracking** training metrics, parameters, system metrics, checkpoints, and model artifacts.
2. **Registry** candidate/production model version transitions after gates pass.

![MLflow promotion lifecycle](diagrams/mlflow_promotion.png)

## Start a local MLflow server

```bash
mkdir -p /tmp/geo_mlops_mlruns

mlflow server \
  --backend-store-uri sqlite:////tmp/geo_mlops_mlflow.db \
  --default-artifact-root /tmp/geo_mlops_mlruns \
  --host 127.0.0.1 \
  --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

## Training-time tracking

When `--mlflow` is passed to `cli/train.py` or `cli/run_pipeline.py`, the training callback logs:

- training and validation metrics per epoch,
- engine/model/dataset/loss/metric/sampler parameters,
- raw checkpoint artifacts,
- a logged MLflow PyTorch model artifact suitable for registry,
- system metrics when enabled.

The training manifest records MLflow metadata:

```json
{
  "tracking": {
    "mlflow_run_id": "...",
    "mlflow_run_name": "building_seg/train/default/20260501-144956",
    "mlflow_tracking_uri": "http://127.0.0.1:5000",
    "mlflow_model_artifact_path": "model",
    "mlflow_model_uri": "runs:/.../model"
  }
}
```

## Candidate registration after Gate A

After Gate A passes:

```bash
python -m geo_mlops.cli.register \
  --task building_seg \
  --task-cfg src/geo_mlops/tasks/segmentation/building/config/default.yaml \
  --action register-candidate \
  --gate-contract /path/to/run/gate_a/gate_decision.json \
  --out-dir /path/to/run/registry_candidate
```

The CLI resolves the MLflow run ID from `train_manifest.json` through the gate contract's upstream lineage. It then registers the MLflow model artifact and writes `registry_result.json`.

## Production promotion after Gate B

After full-scene golden evaluation and Gate B pass:

```bash
python -m geo_mlops.cli.register \
  --task building_seg \
  --task-cfg src/geo_mlops/tasks/segmentation/building/config/default.yaml \
  --action promote-production \
  --gate-contract /path/to/run/gate_b/gate_decision.json \
  --model-version <candidate-version> \
  --out-dir /path/to/run/registry_production
```

The full `run_pipeline.py` command performs this automatically by reading the candidate model version from:

```text
registry_candidate/registry_result.json
```

## Registry config

The task config contains the registry policy:

```yaml
registry:
  model_name: building_seg
  model_artifact_path: model
  candidate_alias: candidate
  production_alias: production
  archive_candidate_alias: false
  tracking_uri: http://127.0.0.1:5000
  registry_uri: null
```

