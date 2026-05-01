from __future__ import annotations

from typing import Any, Dict

import torch


def building_forward_fn(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    if "tile_tensor" not in batch:
        raise KeyError("Building forward_fn expects batch['tile_tensor'].")

    tile = batch["tile_tensor"].to(device)
    ctx = batch.get("context_tensor", None)

    if ctx is not None:
        ctx = ctx.to(device)
        return model(tile, ctx)

    return model(tile)