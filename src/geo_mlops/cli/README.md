# CLI Layer

`geo_mlops.cli` contains thin command-line wrappers around the core pipeline stages.

The CLIs should stay intentionally small:

```text
parse args -> load task plugin/config -> call core engine/stage -> print artifact paths
```

## Commands

| Command | Purpose |
|---|---|
| `tile.py` | Generate tile records and `tiles_manifest.json`. |
| `split.py` | Create deterministic group-aware train/validation splits. |
| `train.py` | Train a task model and write training artifacts. |
| `gate.py` | Evaluate KPI thresholds and write `gate_decision.json`. |
| `register.py` | Register candidate models and promote production versions in MLflow. |
| `evaluate.py` | Run full-scene golden evaluation with sliding-window inference. |
| `run_pipeline.py` | Orchestrate the full A-to-Z workflow. |

## Design rule

Do not put heavy logic here. Heavy logic belongs in `core/` or in task plugins under `tasks/`.
