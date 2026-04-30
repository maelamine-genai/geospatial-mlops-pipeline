from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader

from geo_mlops.models.evals import _iou_binary
from geo_mlops.core.utils.utils import _seed_everything
from geo_mlops.tasks.segmentation.building.dataset import (
    BuildingSegConfig,
    BuildingSegWithContextDataset,
)
from geo_mlops.tasks.segmentation.building.model_factory import build_model


@dataclass(frozen=True)
class EvalOutputs:
    metrics: Dict[str, Any]
    num_eval_tiles: int
    eval_indices: List[int]


def _build_dataset_cfg(train_cfg: Mapping[str, Any]) -> BuildingSegConfig:
    data_cfg = train_cfg.get("data", {})
    return BuildingSegConfig(
        reflectance_max=float(data_cfg.get("reflectance_max", 10_000)),
        use_context=bool(data_cfg.get("use_context", True)),
        do_aug=False,
        aug_flip=False,
        aug_rot90=False,
        aug_noise_std=0.0,
        tile_out_channels=int(data_cfg.get("tile_out_channels", 1)),
        context_out_channels=int(data_cfg.get("context_out_channels", 1)),
    )


def select_eval_indices(
    *,
    tiles_df: pd.DataFrame,
    group_col: str,
    groups: Sequence[str],
) -> List[int]:
    """
    Select evaluation rows from the master CSV by group membership.

    Example:
      group_col="region"
      groups=["AOI_3_Paris", "AOI_5_Khartoum"]
    """
    if group_col not in tiles_df.columns:
        raise KeyError(f"group_col '{group_col}' not found in tiles DataFrame")

    groups_set = {str(g) for g in groups}
    mask = tiles_df[group_col].astype(str).isin(groups_set)
    indices = tiles_df.index[mask].tolist()

    if not indices:
        raise ValueError(
            f"No rows matched groups under column '{group_col}'. "
            f"Requested groups={list(groups)[:5]}{'...' if len(groups) > 5 else ''}"
        )

    return indices


def load_groups_file(path: str | Path) -> List[str]:
    """
    Read a newline-delimited text file of evaluation groups.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Groups file not found: {p}")

    groups = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not groups:
        raise ValueError(f"No non-empty groups found in file: {p}")
    return groups


def build_eval_dataset(
    *,
    tiles_df: pd.DataFrame,
    eval_indices: Sequence[int],
    train_cfg: Mapping[str, Any],
) -> BuildingSegWithContextDataset:
    """
    Build the evaluation dataset using the same task dataset class as training,
    but with augmentation disabled.
    """
    ds_cfg = _build_dataset_cfg(train_cfg)
    data_cfg = train_cfg.get("data", {})

    return BuildingSegWithContextDataset(
        tiles_df=tiles_df,
        indices=list(eval_indices),
        cfg=ds_cfg,
        cache_context=True,
        context_cache_max_items=int(data_cfg.get("context_cache_max_items", 256)),
    )


def load_trained_model(
    *,
    train_cfg: Mapping[str, Any],
    model_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Rebuild the task model and load the saved state_dict from training.
    """
    model = build_model(train_cfg).to(device)

    state = torch.load(Path(model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def compute_eval_metrics(
    *,
    model: torch.nn.Module,
    eval_ds,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    split_name: str,
) -> Dict[str, Any]:
    """
    Run batched evaluation and return nested metrics keyed by split_name.

    Output shape is intentionally compatible with the gate engine's
    nested metric format:
      {
        "golden_test": {
          "iou": 0.731,
          "num_samples": 1280
        }
      }
    """
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    iou_sum = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in eval_loader:
            tile = batch["tile_tensor"].to(device)
            mask = batch["mask"].to(device)

            ctx = batch.get("context_tensor", None)
            if ctx is not None:
                ctx = ctx.to(device)

            logits = model(tile, ctx) if ctx is not None else model(tile)

            bs = int(tile.shape[0])
            iou_sum += _iou_binary(logits, mask) * bs
            n_samples += bs

    mean_iou = iou_sum / max(1, n_samples)

    return {
        str(split_name): {
            "iou": float(mean_iou),
            "num_samples": int(n_samples),
        }
    }


def run_evaluation(
    *,
    tiles_df: pd.DataFrame,
    train_cfg: Mapping[str, Any],
    model_path: str | Path,
    group_col: str,
    groups: Sequence[str],
    split_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int = 1337,
) -> EvalOutputs:
    """
    Full evaluation entrypoint used by the CLI.

    Responsibilities:
      - select eval subset from tiles DataFrame
      - build eval dataset
      - rebuild/load trained model
      - run batched evaluation
      - return structured metrics and selected indices
    """
    _seed_everything(seed)

    eval_indices = select_eval_indices(
        tiles_df=tiles_df,
        group_col=group_col,
        groups=groups,
    )

    eval_ds = build_eval_dataset(
        tiles_df=tiles_df,
        eval_indices=eval_indices,
        train_cfg=train_cfg,
    )

    model = load_trained_model(
        train_cfg=train_cfg,
        model_path=model_path,
        device=device,
    )

    metrics = compute_eval_metrics(
        model=model,
        eval_ds=eval_ds,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        split_name=split_name,
    )

    return EvalOutputs(
        metrics=metrics,
        num_eval_tiles=len(eval_indices),
        eval_indices=eval_indices,
    )