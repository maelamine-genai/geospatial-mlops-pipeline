from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from geo_mlops.core.io.train_io import resolve_training_inputs
from geo_mlops.models.engine import train_one_run
from geo_mlops.tasks.segmentation.building.dataset import BuildingSegWithContextDataset, BuildingSegConfig
from geo_mlops.tasks.segmentation.building.model import build_building_model  # you create this tiny builder
from geo_mlops.tasks.segmentation.building.loss import build_building_loss    # tiny builder


def main() -> None:
    ap = argparse.ArgumentParser("Train a model from tiles + split contracts")

    ap.add_argument("--tiles-manifest", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--train-cfg", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Resolve contracts → inputs
    # -----------------------------
    train_inputs = resolve_training_inputs(
        tiles_manifest_path=args.tiles_manifest,
        split_json_path=args.split_json,
        train_cfg_path=args.train_cfg,
        out_dir=args.out_dir,
    )

    # Load master CSV once, use indices subsets
    tiles_df = pd.read_csv(train_inputs.tiles_master_csv)

    # -----------------------------
    # Build datasets
    # -----------------------------
    ds_cfg = BuildingSegConfig(
        reflectance_max=float(train_inputs.train_cfg.get("data", {}).get("reflectance_max", 10_000)),
        use_context=bool(train_inputs.train_cfg.get("data", {}).get("use_context", True)),
        do_aug=bool(train_inputs.train_cfg.get("augment", {}).get("enabled", False)),
        aug_flip=bool(train_inputs.train_cfg.get("augment", {}).get("flip", True)),
        aug_rot90=bool(train_inputs.train_cfg.get("augment", {}).get("rot90", True)),
        aug_noise_std=float(train_inputs.train_cfg.get("augment", {}).get("noise_std", 0.0)),
        tile_out_channels=int(train_inputs.train_cfg.get("data", {}).get("tile_out_channels", 1)),
        context_out_channels=int(train_inputs.train_cfg.get("data", {}).get("context_out_channels", 1)),
    )

    train_ds = BuildingSegWithContextDataset(
        tiles_df=tiles_df,
        indices=train_inputs.train_row_indices,
        cfg=ds_cfg,
        cache_context=True,
        context_cache_max_items=int(train_inputs.train_cfg.get("data", {}).get("context_cache_max_items", 256)),
    )
    val_ds = BuildingSegWithContextDataset(
        tiles_df=tiles_df,
        indices=train_inputs.val_row_indices,
        cfg=BuildingSegConfig(**{**ds_cfg.__dict__, "do_aug": False}),  # no aug on val
        cache_context=True,
        context_cache_max_items=int(train_inputs.train_cfg.get("data", {}).get("context_cache_max_items", 256)),
    )

    # -----------------------------
    # Build model + loss
    # -----------------------------
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    model = build_building_model(train_inputs.train_cfg).to(device)
    loss_fn = build_building_loss(train_inputs.train_cfg)

    # -----------------------------
    # Run training (engine owns loop, metrics, saving)
    # -----------------------------
    run_outputs = train_one_run(
        model=model,
        loss_fn=loss_fn,
        train_ds=train_ds,
        val_ds=val_ds,
        out_dir=args.out_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        train_inputs=train_inputs,
    )

    print(f"[train] done. model={run_outputs.model_path}")
    print(f"[train] metrics={run_outputs.metrics_path}")
    print(f"[train] manifest={run_outputs.train_manifest_path}")


if __name__ == "__main__":
    main()