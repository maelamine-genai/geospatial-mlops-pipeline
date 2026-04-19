from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


import torch
from torch.utils.data import DataLoader

from geo_mlops.core.contracts.train_contract import TrainInputs, TrainOutputs
from geo_mlops.models.evals import _iou_binary
from geo_mlops.models.utils import _seed_everything


def train_one_run(
    *,
    model: torch.nn.Module,
    loss_fn,
    train_ds,
    val_ds,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    epochs: int,
    lr: float,
    seed: int,
    train_inputs: TrainInputs,
) -> TrainOutputs:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_iou = -1.0
    metrics: Dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for batch in train_loader:
            tile = batch["tile_tensor"].to(device)                # [B,C,H,W]
            mask = batch["mask"].to(device)                       # [B,H,W]
            ctx = batch.get("context_tensor", None)
            if ctx is not None:
                ctx = ctx.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(tile, ctx) if ctx is not None else model(tile)  # [B,1,H,W] ideally

            loss = loss_fn(logits, mask)
            loss.backward()
            opt.step()

            train_loss_sum += float(loss.item()) * int(tile.shape[0])
            n_train += int(tile.shape[0])

        train_loss = train_loss_sum / max(1, n_train)

        # ---- val
        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                tile = batch["tile_tensor"].to(device)
                mask = batch["mask"].to(device)
                ctx = batch.get("context_tensor", None)
                if ctx is not None:
                    ctx = ctx.to(device)

                logits = model(tile, ctx) if ctx is not None else model(tile)
                loss = loss_fn(logits, mask)

                val_loss_sum += float(loss.item()) * int(tile.shape[0])
                val_iou_sum += _iou_binary(logits, mask) * int(tile.shape[0])
                n_val += int(tile.shape[0])

        val_loss = val_loss_sum / max(1, n_val)
        val_iou = val_iou_sum / max(1, n_val)

        metrics[f"epoch_{epoch}"] = {"train_loss": train_loss, "val_loss": val_loss, "val_iou": val_iou}
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

        # checkpoint best
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), out_dir / "model.pt")

    # write metrics + manifest
    metrics_path = out_dir / "metrics.json"
    metrics["best_val_iou"] = {best_val_iou}
    metrics_path.write_text(json.dumps(metrics, indent=2))

    manifest = {
        "task": train_inputs.task,
        "tiles_manifest": str(train_inputs.tiles_manifest_path),
        "split_json": str(train_inputs.split_json_path),
        "train_cfg": str(train_inputs.train_cfg_path),
        "tiles_master_csv": str(train_inputs.tiles_master_csv),
        "num_train_tiles": int(len(train_inputs.train_row_indices)),
        "num_val_tiles": int(len(train_inputs.val_row_indices)),
        "model_path": str(out_dir / "model.pt"),
        "metrics_path": str(metrics_path),
        "best_val_iou": float(best_val_iou),
    }
    train_manifest_path = out_dir / "train_manifest.json"
    train_manifest_path.write_text(json.dumps(manifest, indent=2))

    return TrainOutputs(
        run_dir=out_dir,
        model_path=out_dir / "model.pt",
        metrics_path=metrics_path,
        train_manifest_path=train_manifest_path,
    )