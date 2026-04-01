from __future__ import annotations

import torch

@torch.no_grad()
def _iou_binary(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.0) -> float:
    """
    logits: [B,1,H,W]
    targets: [B,H,W] int64 {0,1}
    thresh=0.0 means sigmoid(logits) > 0.5 is equivalent to logits > 0
    """
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).to(torch.bool)
    gt = (targets > 0).to(torch.bool)

    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return float(inter / max(1, union))