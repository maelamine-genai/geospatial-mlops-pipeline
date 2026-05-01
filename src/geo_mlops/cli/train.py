from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence
from datetime import datetime

import torch

from geo_mlops.core.io.split_io import load_split_contract
from geo_mlops.core.io.tile_io import load_tiles_contract
from geo_mlops.core.io.train_io import resolve_training_inputs
from geo_mlops.core.registry.task_registry import get_task
from geo_mlops.core.training.engine import TrainConfig, train_one_run
from geo_mlops.core.training.mlflow_callbacks import MLflowTrainingCallback


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train a task model from tiling + split contracts."
    )

    ap.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task key registered in task_registry, e.g. building_seg.",
    )
    ap.add_argument(
        "--task-cfg",
        "--task_cfg",
        dest="task_cfg",
        type=Path,
        required=True,
        help="Unified task config YAML/JSON containing a `training:` section.",
    )
    ap.add_argument(
        "--tiles-dir",
        "--tiles_dir",
        dest="tiles_dir",
        type=Path,
        required=True,
        help="Tiling output directory containing tiles_manifest.json.",
    )
    ap.add_argument(
        "--split-dir",
        "--split_dir",
        dest="split_dir",
        type=Path,
        required=True,
        help="Split output directory containing split.json.",
    )
    ap.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        required=True,
        help="Output directory for training artifacts.",
    )

    # Optional runtime overrides. If omitted, values come from training.engine.
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--selection-metric", type=str, default=None)
    ap.add_argument("--selection-mode", type=str, choices=("min", "max"), default=None)

    ap.add_argument("--mlflow", action="store_true", help="Enable MLflow logging.")
    ap.add_argument("--mlflow-tracking-uri", type=str, default=None)
    ap.add_argument("--mlflow-experiment", type=str, default=None)
    ap.add_argument("--mlflow-run-name", type=str, default=None)

    return ap


def _build_train_config(train_cfg: dict) -> TrainConfig:
    engine_cfg = train_cfg.get("engine", {}) or {}

    return TrainConfig(
        batch_size=int(engine_cfg.get("batch_size", 8)),
        num_workers=int(engine_cfg.get("num_workers", 4)),
        epochs=int(engine_cfg.get("epochs", 5)),
        lr=float(engine_cfg.get("lr", 3e-4)),
        seed=int(engine_cfg.get("seed", 1337)),
        selection_metric=str(engine_cfg.get("selection_metric", "val/loss")),
        selection_mode=str(engine_cfg.get("selection_mode", "min")),
    )


def _apply_cli_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    updates = {}

    if args.batch_size is not None:
        updates["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        updates["num_workers"] = int(args.num_workers)
    if args.epochs is not None:
        updates["epochs"] = int(args.epochs)
    if args.lr is not None:
        updates["lr"] = float(args.lr)
    if args.seed is not None:
        updates["seed"] = int(args.seed)
    if args.selection_metric is not None:
        updates["selection_metric"] = str(args.selection_metric)
    if args.selection_mode is not None:
        updates["selection_mode"] = str(args.selection_mode)

    return replace(cfg, **updates) if updates else cfg


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def default_run_name(
    task: str,
    stage: str = "train",
    cfg_name: str | None = None,
) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [task, stage]
    if cfg_name:
        parts.append(cfg_name)
    parts.append(ts)
    return "/".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    task_plugin = get_task(args.task)

    # -------------------------------------------------------------------------
    # Load stage contracts
    # -------------------------------------------------------------------------
    tiles = load_tiles_contract(args.tiles_dir)
    split = load_split_contract(args.split_dir)

    if tiles.task != args.task:
        raise ValueError(
            f"Task mismatch: --task={args.task!r}, tiles.task={tiles.task!r}"
        )

    # -------------------------------------------------------------------------
    # Load task training config
    # -------------------------------------------------------------------------
    train_cfg = task_plugin.build_training_cfg(args.task_cfg)
    engine_cfg = _apply_cli_overrides(
        _build_train_config(train_cfg),
        args,
    )

    # -------------------------------------------------------------------------
    # Resolve canonical training inputs
    # -------------------------------------------------------------------------
    train_inputs = resolve_training_inputs(
        tiles_manifest_path=tiles.tiles_dir / "tiles_manifest.json",
        split_json_path=split.split_dir / "split.json",
        train_cfg_path=args.task_cfg,
        out_dir=args.out_dir,
    )

    # -------------------------------------------------------------------------
    # Build task-specific components through plugin
    # -------------------------------------------------------------------------
    train_ds, val_ds = task_plugin.build_train_val_datasets(
        tiles=tiles,
        split=split,
        train_cfg=train_cfg,
    )

    model = task_plugin.build_model(train_cfg)
    loss_fn = task_plugin.build_loss(train_cfg)
    metrics_fn = task_plugin.build_metrics_fn(train_cfg)
    forward_fn = task_plugin.get_forward_fn()

    device = _resolve_device(args.device)

    callbacks = []

    if args.mlflow:
        run_name = args.mlflow_run_name or default_run_name(
            task=args.task,
            stage="train",
            cfg_name=args.task_cfg.stem,
        )
        callbacks.append(
            MLflowTrainingCallback(
                tracking_uri=args.mlflow_tracking_uri,
                experiment_name=args.mlflow_experiment or args.task,
                run_name=run_name,
                tags={
                    "task": args.task,
                    "stage": "train",
                    "task_cfg": str(args.task_cfg),
                    "tiles_dir": str(args.tiles_dir),
                    "split_dir": str(args.split_dir),
                }
            )
        )
    # -------------------------------------------------------------------------
    # Run generic core trainer
    # -------------------------------------------------------------------------
    run_outputs = train_one_run(
        model=model,
        loss_fn=loss_fn,
        train_ds=train_ds,
        val_ds=val_ds,
        out_dir=args.out_dir,
        device=device,
        cfg=engine_cfg,
        train_inputs=train_inputs,
        forward_fn=forward_fn,
        metrics_fn=metrics_fn,
        callbacks=callbacks,
        callback_context={
            "train_cfg": train_cfg,
            "task_cfg_path": str(args.task_cfg),
            "tiles_dir": str(args.tiles_dir),
            "split_dir": str(args.split_dir),
        },
    )

    print(f"[train] done")
    print(f"[train] model={run_outputs.model_path}")
    print(f"[train] metrics={run_outputs.metrics_path}")
    print(f"[train] manifest={run_outputs.train_manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
