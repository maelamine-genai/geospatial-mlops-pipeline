from __future__ import annotations

from typing import Any, Dict

import torch


def build_metrics_fn(train_cfg: Dict[str, Any]):
    metrics_cfg = train_cfg.get("metrics", {}) or {}

    threshold = float(metrics_cfg.get("threshold", 0.5))
    foreground_label = int(metrics_cfg.get("foreground_label", 1))
    eps = float(metrics_cfg.get("eps", 1e-7))

    def metrics_fn(outputs: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Lightweight binary segmentation metrics for validation monitoring.

        Args:
            outputs: logits shaped [B, 1, H, W]
            batch: dataset batch containing "mask" shaped [B, H, W]

        Returns:
            dictionary of scalar metrics.
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

        tp = torch.logical_and(pred, target).sum().float()
        fp = torch.logical_and(pred, ~target).sum().float()
        fn = torch.logical_and(~pred, target).sum().float()
        tn = torch.logical_and(~pred, ~target).sum().float()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        return {
            "iou": float(iou.detach().cpu()),
            "micro_precision": float(precision.detach().cpu()),
            "micro_recall": float(recall.detach().cpu()),
            "micro_f1": float(f1.detach().cpu()),
            "pixel_accuracy": float(accuracy.detach().cpu()),
        }

    return metrics_fn