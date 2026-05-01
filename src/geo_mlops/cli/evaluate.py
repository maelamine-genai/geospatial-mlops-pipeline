from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from geo_mlops.core.config.loader import load_cfg
from geo_mlops.core.evaluation.engine import run_full_scene_evaluation
from geo_mlops.core.registry.task_registry import get_task


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run full-scene sliding-window evaluation on a golden dataset using "
            "a trained task model."
        )
    )

    p.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task key registered in task_registry, e.g. building_seg.",
    )
    p.add_argument(
        "--task-cfg",
        "--task_cfg",
        dest="task_cfg",
        type=Path,
        required=True,
        help="Unified task config YAML/JSON containing training/evaluation sections.",
    )
    p.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        type=Path,
        required=True,
        help="Golden evaluation dataset root. This is full-scene based, not tile/split based.",
    )
    p.add_argument(
        "--train-manifest",
        "--train_manifest",
        dest="train_manifest",
        type=Path,
        default=None,
        help=(
            "Optional train_manifest.json. Used to infer checkpoint path when "
            "--checkpoint is not provided."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to local model checkpoint/state_dict. Overrides train_manifest model_path.",
    )
    p.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        required=True,
        help="Output directory for eval_summary.json, eval_manifest.json, masks, probabilities, and tables.",
    )

    p.add_argument("--device", type=str, default="cuda")

    # Optional eval engine overrides.
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)

    return p


def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")

    obj = json.loads(p.read_text())

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at root of {p}")

    return obj


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[evaluate] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def _resolve_checkpoint(
    *,
    checkpoint: Optional[Path],
    train_manifest: Optional[Path],
) -> Path:
    if checkpoint is not None:
        return checkpoint

    if train_manifest is None:
        raise ValueError("Provide --checkpoint or --train-manifest.")

    manifest = _load_json(train_manifest)

    model_path = manifest.get("model_path")
    if not model_path:
        raise ValueError(
            f"train_manifest does not contain model_path: {train_manifest}"
        )

    return Path(str(model_path))


def _load_training_cfg_from_task_cfg(task_cfg_path: Path) -> Dict[str, Any]:
    cfg = load_cfg(task_cfg_path)

    if not isinstance(cfg, dict):
        raise ValueError(f"Task config root must be a mapping: {task_cfg_path}")

    training = cfg.get("training")
    if not isinstance(training, dict):
        raise ValueError("Task config must include a 'training' mapping.")

    return training


def _apply_eval_overrides(eval_engine_cfg, args: argparse.Namespace):
    """
    Avoid importing dataclasses.replace unless needed.
    EvalConfig is frozen, so construct a new one if overrides are present.
    """
    updates = {
        "tile_size": args.tile_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "seed": args.seed,
    }

    updates = {k: v for k, v in updates.items() if v is not None}

    if not updates:
        return eval_engine_cfg

    from dataclasses import replace

    return replace(eval_engine_cfg, **updates)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)

    task_plugin = get_task(args.task)

    eval_cfg = task_plugin.build_evaluation_cfg(args.task_cfg)
    eval_engine_cfg = task_plugin.build_eval_engine_cfg(eval_cfg)
    eval_engine_cfg = _apply_eval_overrides(eval_engine_cfg, args)

    train_cfg = _load_training_cfg_from_task_cfg(args.task_cfg)

    checkpoint_path = _resolve_checkpoint(
        checkpoint=args.checkpoint,
        train_manifest=args.train_manifest,
    )

    model = task_plugin.build_model(train_cfg)
    model = task_plugin.load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    scenes = task_plugin.iter_eval_scenes(
        dataset_root=args.dataset_root,
        eval_cfg=eval_cfg,
    )

    outputs = run_full_scene_evaluation(
        task=args.task,
        model=model,
        scenes=scenes,
        out_dir=args.out_dir,
        device=device,
        cfg=eval_engine_cfg,
        load_scene_fn=lambda scene: task_plugin.load_eval_scene(scene, eval_cfg),
        forward_fn=task_plugin.get_forward_fn(),
        postprocess_fn=task_plugin.build_eval_postprocessor(eval_cfg),
        save_prediction_fn=task_plugin.save_eval_prediction,
        metric_accumulator=task_plugin.build_eval_metric_accumulator(eval_cfg),
        eval_cfg_raw=eval_cfg,
        checkpoint_path=checkpoint_path,
        model_uri=None,
    )

    print("[evaluate] done")
    print(f"[evaluate] scenes={outputs.summary.get('num_scenes')}")
    print(f"[evaluate] summary={outputs.summary_path}")
    print(f"[evaluate] manifest={outputs.manifest_path}")
    print(f"[evaluate] per_scene_table={outputs.per_scene_table_path}")
    print(f"[evaluate] probabilities={outputs.probability_dir}")
    print(f"[evaluate] masks={outputs.mask_dir}")

    metrics = outputs.summary.get("metrics", {})
    if metrics:
        print(f"[evaluate] metrics={json.dumps(metrics, indent=2)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())