from __future__ import annotations
from typing import Any, Dict
import torch
import torch.nn.functional as F


def build_building_loss(train_cfg: Dict[str, Any]):
    # simplest: BCEWithLogits for binary building mask
    def loss_fn(logits: torch.Tensor, target_hw: torch.Tensor) -> torch.Tensor:
        # logits [B,1,H,W], target [B,H,W] {0,1}
        target = (target_hw > 0).float().unsqueeze(1)
        return F.binary_cross_entropy_with_logits(logits, target)
    return loss_fn