from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from geo_mlops.core.contracts.train_contract import TrainInputs, TrainOutputs
from geo_mlops.core.utils.random import seed_everything


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    seed: int
    selection_metric: str = "val/loss"
    selection_mode: str = "min"  # "min" or "max"


def _is_better(value: float, best: Optional[float], mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    raise ValueError(f"selection_mode must be 'min' or 'max', got {mode!r}")


def _prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}/{k}": float(v) for k, v in metrics.items()}


def train_one_run(
    *,
    model: torch.nn.Module,
    loss_fn: Callable[[Any, Dict[str, Any]], torch.Tensor],
    train_ds,
    val_ds,
    out_dir: Path,
    device: torch.device,
    cfg: TrainConfig,
    train_inputs: TrainInputs,
    forward_fn: Callable[[torch.nn.Module, Dict[str, Any], torch.device], Any],
    metrics_fn: Optional[Callable[[Any, Dict[str, Any]], Dict[str, float]]] = None,
    optimizer_factory: Optional[Callable[[torch.nn.Module, TrainConfig], torch.optim.Optimizer]] = None,
) -> TrainOutputs:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    opt = (
        optimizer_factory(model, cfg)
        if optimizer_factory is not None
        else torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    )

    best_metric_value: Optional[float] = None
    best_epoch: Optional[int] = None
    history: Dict[str, Any] = {}

    model_path = out_dir / "model.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        n_train = 0

        for batch in train_loader:
            opt.zero_grad(set_to_none=True)

            outputs = forward_fn(model, batch, device)
            loss = loss_fn(outputs, batch)

            loss.backward()
            opt.step()

            batch_size = _infer_batch_size(batch)
            train_loss_sum += float(loss.item()) * batch_size
            n_train += batch_size

        train_loss = train_loss_sum / max(1, n_train)

        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        val_metric_sums: Dict[str, float] = {}

        with torch.no_grad():
            for batch in val_loader:
                outputs = forward_fn(model, batch, device)
                loss = loss_fn(outputs, batch)

                batch_size = _infer_batch_size(batch)
                val_loss_sum += float(loss.item()) * batch_size
                n_val += batch_size

                if metrics_fn is not None:
                    batch_metrics = metrics_fn(outputs, batch)
                    for name, value in batch_metrics.items():
                        val_metric_sums[name] = val_metric_sums.get(name, 0.0) + float(value) * batch_size

        val_loss = val_loss_sum / max(1, n_val)
        val_metrics = {
            name: total / max(1, n_val)
            for name, total in val_metric_sums.items()
        }

        epoch_metrics = {
            "train/loss": float(train_loss),
            "val/loss": float(val_loss),
            **_prefix_metrics("val", val_metrics),
        }

        history[f"epoch_{epoch}"] = epoch_metrics

        if cfg.selection_metric not in epoch_metrics:
            raise KeyError(
                f"selection_metric={cfg.selection_metric!r} not found. "
                f"Available metrics: {sorted(epoch_metrics.keys())}"
            )

        current_value = float(epoch_metrics[cfg.selection_metric])

        print(
            f"[epoch {epoch}] "
            + " ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items())
        )

        if _is_better(current_value, best_metric_value, cfg.selection_mode):
            best_metric_value = current_value
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

    metrics_path = out_dir / "metrics.json"

    metrics_payload = {
        "selection_metric": cfg.selection_metric,
        "selection_mode": cfg.selection_mode,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "history": history,
    }

    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    train_manifest_path = out_dir / "train_manifest.json"

    manifest = {
        "task": train_inputs.task,
        "tiles_manifest": str(train_inputs.tiles_manifest_path),
        "split_json": str(train_inputs.split_json_path),
        "train_cfg": str(train_inputs.train_cfg_path),
        "tiles_master_csv": str(train_inputs.tiles_master_csv),
        "num_train_tiles": int(len(train_inputs.train_row_indices)),
        "num_val_tiles": int(len(train_inputs.val_row_indices)),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "selection_metric": cfg.selection_metric,
        "selection_mode": cfg.selection_mode,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
    }

    train_manifest_path.write_text(json.dumps(manifest, indent=2))

    return TrainOutputs(
        run_dir=out_dir,
        model_path=model_path,
        metrics_path=metrics_path,
        train_manifest_path=train_manifest_path,
    )


def _infer_batch_size(batch: Dict[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
    raise ValueError("Could not infer batch size from batch; no tensor values found.")