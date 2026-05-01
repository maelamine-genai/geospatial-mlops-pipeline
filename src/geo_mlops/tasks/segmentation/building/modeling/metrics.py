from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch


# -----------------------------------------------------------------------------
# Training-time validation metrics
# -----------------------------------------------------------------------------
def build_metrics_fn(train_cfg: Dict[str, Any]):
    metrics_cfg = train_cfg.get("metrics", {}) or {}

    threshold = float(metrics_cfg.get("threshold", 0.5))
    foreground_label = int(metrics_cfg.get("foreground_label", 1))
    eps = float(metrics_cfg.get("eps", 1e-7))

    def metrics_fn(outputs: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Lightweight binary segmentation metrics for training-time validation.

        This is intentionally batch-level and cheap. Formal golden evaluation
        uses BuildingSegmentationEvalAccumulator below.
        """
        if "mask" not in batch:
            raise KeyError("Building metrics expects batch['mask'].")

        if outputs.ndim != 4 or outputs.shape[1] != 1:
            raise ValueError(
                f"Building metrics expect outputs shaped [B,1,H,W], got {tuple(outputs.shape)}."
            )

        mask = batch["mask"].to(outputs.device)
        target = mask == foreground_label

        probs = torch.sigmoid(outputs[:, 0])
        pred = probs >= threshold

        counts = _torch_binary_counts(pred=pred, target=target)
        metrics = _metrics_from_counts(counts, eps=eps)

        return {
            "iou": metrics["iou"],
            "micro_precision": metrics["precision"],
            "micro_recall": metrics["recall"],
            "micro_f1": metrics["f1"],
            "pixel_accuracy": metrics["pixel_accuracy"],
        }

    return metrics_fn


# -----------------------------------------------------------------------------
# Formal full-scene golden evaluation metrics
# -----------------------------------------------------------------------------
class BuildingSegmentationEvalAccumulator:
    """
    Formal evaluation accumulator for full-scene building segmentation.

    Responsibilities:
      - compute per-scene binary segmentation metrics
      - aggregate global pixel-count micro metrics
      - aggregate per-scene macro metrics
      - write task-specific analytics tables, including Pareto/hardest images

    Core evaluation engine calls:
      update(scene=..., arrays=..., prediction=..., artifacts=...)
      finalize(out_dir=...)
    """

    def __init__(self, metrics_cfg: Optional[Dict[str, Any]] = None) -> None:
        metrics_cfg = metrics_cfg or {}

        self.eps = float(metrics_cfg.get("eps", 1e-7))
        self.pareto_top_k = int(metrics_cfg.get("pareto_top_k", 50))

        # Optional: ignore images without GT instead of failing.
        self.require_target = bool(metrics_cfg.get("require_target", True))

        self.global_counts: Dict[str, int] = _empty_counts()
        self.rows: List[Dict[str, Any]] = []
        self.warnings: List[str] = []

    def update(
        self,
        *,
        scene: Any,
        arrays: Any,
        prediction: Any,
        artifacts: Any,
    ) -> Dict[str, Any]:
        """
        Update metrics from one full-scene prediction.

        Args are intentionally typed as Any to avoid importing core evaluation
        dataclasses into this task metrics file. Expected fields:
          - scene.scene_id / region / subregion
          - arrays.target
          - prediction.mask
          - prediction.probability
          - artifacts.probability_path / mask_path
        """
        if getattr(arrays, "target", None) is None:
            msg = f"Scene {scene.scene_id!r} has no target/GT mask."
            if self.require_target:
                raise ValueError(msg)

            self.warnings.append(msg)
            row = {
                "scene_id": scene.scene_id,
                "has_target": False,
                "warning": msg,
            }
            self.rows.append(row)
            return row

        target = _to_numpy_bool(arrays.target)
        pred = _to_numpy_bool(prediction.mask)

        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch for scene={scene.scene_id!r}: "
                f"pred={pred.shape}, target={target.shape}"
            )

        counts = _numpy_binary_counts(pred=pred, target=target)
        self.global_counts = _add_counts(self.global_counts, counts)

        metrics = _metrics_from_counts(counts, eps=self.eps)

        prob = getattr(prediction, "probability", None)
        prob_stats = _probability_stats(prob)

        row: Dict[str, Any] = {
            "scene_id": scene.scene_id,
            "region": getattr(scene, "region", "") or "",
            "subregion": getattr(scene, "subregion", "") or "",
            "has_target": True,
            **counts,
            **metrics,
            "gt_foreground_pixels": int(target.sum()),
            "pred_foreground_pixels": int(pred.sum()),
            "gt_foreground_frac": float(target.mean()),
            "pred_foreground_frac": float(pred.mean()),
            "false_positive_pixels": int(counts["fp"]),
            "false_negative_pixels": int(counts["fn"]),
            "probability_path": str(getattr(artifacts, "probability_path", "") or ""),
            "mask_path": str(getattr(artifacts, "mask_path", "") or ""),
            **prob_stats,
        }

        self.rows.append(row)
        return row

    def finalize(self, *, out_dir: Path) -> Dict[str, Any]:
        out_dir = Path(out_dir)
        tables_dir = out_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.rows)

        per_image_csv = tables_dir / "building_per_image_metrics.csv"
        df.to_csv(per_image_csv, index=False)

        pareto_df = self._build_pareto_table(df)
        pareto_csv = tables_dir / "building_pareto_images.csv"
        pareto_df.to_csv(pareto_csv, index=False)

        micro = _metrics_from_counts(self.global_counts, eps=self.eps)
        macro = self._macro_metrics(df)

        analytics = {
            "num_images_with_target": int(df["has_target"].sum()) if "has_target" in df.columns else 0,
            "num_images_total": int(len(df)),
            "global_counts": dict(self.global_counts),
            "warnings": list(self.warnings),
        }

        return {
            "metrics": {
                "micro": micro,
                "macro": macro,
            },
            "artifacts": {
                "building_per_image_metrics_csv": str(per_image_csv),
                "building_pareto_images_csv": str(pareto_csv),
            },
            "analytics": analytics,
        }

    def _macro_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {}

        if "has_target" in df.columns:
            df = df[df["has_target"] == True].copy()  # noqa: E712

        metric_cols = [
            "precision",
            "recall",
            "f1",
            "iou",
            "pixel_accuracy",
        ]

        out: Dict[str, float] = {}

        for col in metric_cols:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce")
                out[col] = float(values.mean())

        return out

    def _build_pareto_table(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        df = df.copy()

        if "has_target" in df.columns:
            df = df[df["has_target"] == True].copy()  # noqa: E712

        if df.empty:
            return df

        # Lower f1/iou are worse. Higher fp/fn are worse.
        if "f1" in df.columns:
            df["rank_low_f1"] = pd.to_numeric(
                df["f1"], errors="coerce"
            ).rank(method="min", ascending=True)

        if "iou" in df.columns:
            df["rank_low_iou"] = pd.to_numeric(
                df["iou"], errors="coerce"
            ).rank(method="min", ascending=True)

        if "false_positive_pixels" in df.columns:
            df["rank_high_fp"] = pd.to_numeric(
                df["false_positive_pixels"], errors="coerce"
            ).rank(method="min", ascending=False)

        if "false_negative_pixels" in df.columns:
            df["rank_high_fn"] = pd.to_numeric(
                df["false_negative_pixels"], errors="coerce"
            ).rank(method="min", ascending=False)

        rank_cols = [c for c in df.columns if c.startswith("rank_")]

        if rank_cols:
            df["pareto_score"] = df[rank_cols].mean(axis=1)
            df.sort_values("pareto_score", inplace=True, ignore_index=True)

        return df.head(self.pareto_top_k)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def _empty_counts() -> Dict[str, int]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }


def _add_counts(a: Mapping[str, int], b: Mapping[str, int]) -> Dict[str, int]:
    return {
        "tp": int(a.get("tp", 0)) + int(b.get("tp", 0)),
        "fp": int(a.get("fp", 0)) + int(b.get("fp", 0)),
        "fn": int(a.get("fn", 0)) + int(b.get("fn", 0)),
        "tn": int(a.get("tn", 0)) + int(b.get("tn", 0)),
    }


def _torch_binary_counts(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, int]:
    pred = pred.bool()
    target = target.bool()

    tp = torch.logical_and(pred, target).sum()
    fp = torch.logical_and(pred, ~target).sum()
    fn = torch.logical_and(~pred, target).sum()
    tn = torch.logical_and(~pred, ~target).sum()

    return {
        "tp": int(tp.detach().cpu()),
        "fp": int(fp.detach().cpu()),
        "fn": int(fn.detach().cpu()),
        "tn": int(tn.detach().cpu()),
    }


def _numpy_binary_counts(
    *,
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, int]:
    pred = pred.astype(bool)
    target = target.astype(bool)

    return {
        "tp": int(np.logical_and(pred, target).sum()),
        "fp": int(np.logical_and(pred, ~target).sum()),
        "fn": int(np.logical_and(~pred, target).sum()),
        "tn": int(np.logical_and(~pred, ~target).sum()),
    }


def _metrics_from_counts(
    counts: Mapping[str, int],
    *,
    eps: float = 1e-7,
) -> Dict[str, float]:
    tp = float(counts.get("tp", 0))
    fp = float(counts.get("fp", 0))
    fn = float(counts.get("fn", 0))
    tn = float(counts.get("tn", 0))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "pixel_accuracy": float(pixel_accuracy),
    }


def _to_numpy_bool(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    arr = np.asarray(x)

    # If [1,H,W], squeeze to [H,W].
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    return arr.astype(bool)


def _probability_stats(probability: Any) -> Dict[str, float]:
    if probability is None:
        return {}

    if torch.is_tensor(probability):
        probability = probability.detach().cpu().numpy()

    arr = np.asarray(probability)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    arr = arr.astype(np.float32, copy=False)

    return {
        "prob_mean": float(np.nanmean(arr)),
        "prob_std": float(np.nanstd(arr)),
        "prob_min": float(np.nanmin(arr)),
        "prob_max": float(np.nanmax(arr)),
    }