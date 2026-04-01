from __future__ import annotations

import torch

def _binary_dice_from_logits(logits, target, mask=None, eps=1e-6):
    """
    Binary Dice loss for the WATER class only.
    logits: [B, 3, H, W]
    target: [B, 1, H, W] (binary water mask)
    mask:   [B, 1, H, W] or None
    """
    # Take water channel only
    probs = torch.sigmoid(logits[:, 1:2, :, :])  # [B,1,H,W]

    # Flatten
    probs = probs.contiguous().view(probs.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    if mask is not None:
        mask = mask.contiguous().view(mask.size(0), -1).float()
        probs = probs * mask
        target = target * mask

    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1 - dice.mean()
