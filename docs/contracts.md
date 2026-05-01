# Contracts

The pipeline is contract-driven. Every major stage writes a JSON artifact that records its outputs, upstream inputs, and enough metadata for the next stage to run without guessing.

![Contracts flow](diagrams/contracts_flow.png)

## Contract inventory

| File | Producer | Main consumer | Description |
|---|---|---|---|
| `tiles_manifest.json` | `cli/tile.py` | split, train | Points to the master tile CSV and records tiling configuration, task, adapter, policy, and row count. |
| `split.json` | `cli/split.py` | train | Records deterministic group-aware train/validation partitions. |
| `train_manifest.json` | `cli/train.py` | gate, registry, evaluation | Records checkpoint path, metrics path, upstream data artifacts, selection metric, best epoch, and MLflow run metadata. |
| `metrics.json` | `cli/train.py` | Gate A | Stores training/validation metric history and best metric. |
| `gate_decision.json` | `cli/gate.py` | registry, run_pipeline | Records KPI checks, actual values, pass/fail decision, and warnings. |
| `eval_summary.json` | `cli/evaluate.py` | Gate B | Stores full-scene golden evaluation metrics and artifact paths. |
| `eval_manifest.json` | `cli/evaluate.py` | audit/debug | Records evaluation inputs, scene IDs, config, checkpoint/model URI, and artifact paths. |
| `registry_result.json` | `cli/register.py` | run_pipeline | Records registered model name, version, action, and MLflow URI. |

## Why contracts matter

Contracts make the workflow robust in several ways:

- **No hidden state:** downstream stages read explicit artifact paths.
- **Reproducibility:** config and upstream lineage are recorded.
- **Auditing:** gate decisions include both thresholds and actual metric values.
- **Restartability:** stages can be skipped or rerun without recomputing the full pipeline.
- **Future orchestration:** Argo/Ray jobs can exchange the same JSON contracts used locally.

## Example: training manifest

A typical `train_manifest.json` includes:

```json
{
  "task": "building_seg",
  "tiles_manifest": ".../tiles_manifest.json",
  "split_json": ".../split.json",
  "train_cfg": ".../default.yaml",
  "model_path": ".../train/model.pt",
  "metrics_path": ".../train/metrics.json",
  "selection_metric": "val/micro_f1",
  "selection_mode": "max",
  "best_metric_value": 0.32,
  "best_epoch": 3,
  "tracking": {
    "mlflow_run_id": "...",
    "mlflow_model_artifact_path": "model",
    "mlflow_model_uri": "runs:/.../model"
  }
}
```

## Example: gate decision

A gate records exactly why a model passed or failed:

```json
{
  "gate_name": "gate_b",
  "decision": "pass",
  "passed": true,
  "checks": [
    {
      "scope": "micro",
      "metric": "f1",
      "comparator": ">=",
      "threshold": 0.30,
      "actual": 0.306,
      "passed": true
    }
  ]
}
```

## Gate-compatible metric shapes

The gate engine supports both flat and nested metric layouts:

```json
{"val/micro_f1": 0.31}
```

or:

```json
{
  "micro": {"f1": 0.31},
  "macro": {"f1": 0.24}
}
```

That lets Gate A consume training validation metrics and Gate B consume golden evaluation metrics using the same engine.

