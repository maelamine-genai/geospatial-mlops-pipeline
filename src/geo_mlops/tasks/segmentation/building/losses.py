from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def build_loss(train_cfg: Dict[str, Any]):
    loss_cfg = train_cfg.get("loss", {}) or {}

    kind = str(loss_cfg.get("kind", "bce")).lower().strip()
    foreground_label = int(loss_cfg.get("foreground_label", 1))
    pos_weight = loss_cfg.get("pos_weight", None)

    if kind != "bce":
        raise ValueError(f"Unsupported building loss kind={kind!r}. Expected: 'bce'.")

    def loss_fn(outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Building binary segmentation loss.

        Args:
            outputs: logits shaped [B, 1, H, W]
            batch: dataset batch containing "mask" shaped [B, H, W]

        Returns:
            scalar loss tensor
        """
        if "mask" not in batch:
            raise KeyError("Building loss expects batch['mask'].")

        if outputs.ndim != 4 or outputs.shape[1] != 1:
            raise ValueError(
                f"Building BCE loss expects outputs shaped [B,1,H,W], got {tuple(outputs.shape)}."
            )

        mask = batch["mask"].to(outputs.device)
        target = (mask == foreground_label).float().unsqueeze(1)

        weight = None
        if pos_weight is not None:
            weight = torch.tensor(
                float(pos_weight),
                dtype=outputs.dtype,
                device=outputs.device,
            )

        return F.binary_cross_entropy_with_logits(
            outputs,
            target,
            pos_weight=weight,
        )

    return loss_fn