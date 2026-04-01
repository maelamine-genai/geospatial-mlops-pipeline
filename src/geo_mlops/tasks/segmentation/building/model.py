from __future__ import annotations
from typing import Any, Dict
import torch

from geo_mlops.models.backbones.segformer import SegFormerBackbone
from geo_mlops.models.fusion.concat_fusion import ConcatFusionHead


def build_building_model(train_cfg: Dict[str, Any]) -> torch.nn.Module:
    mcfg = train_cfg.get("model", {})
    bcfg = mcfg.get("backbone", {})

    # Decide fusion layout (must match what your dataset provides)
    add_coords = bool(mcfg.get("add_coords", True))
    expect_ctx = bool(mcfg.get("expect_ctx", True))

    tile_ch = int(mcfg.get("tile_channels", 1))
    ctx_ch  = int(mcfg.get("ctx_channels", 1))

    fused_in_ch = tile_ch + (ctx_ch if expect_ctx else 0) + (2 if add_coords else 0)

    # Single-input SegFormer expecting fused tensor
    inner = SegFormerBackbone(
        base_name=bcfg["base_name"],
        num_classes=int(bcfg["num_classes"]),
        image_size=int(bcfg["image_size"]),
        in_channels=fused_in_ch,
        proj_to_rgb=True,
    )

    # Fusion head concatenates and calls inner(x)
    model = ConcatFusionHead(
        inner_model=inner,
        tile_channels=tile_ch,
        ctx_channels=ctx_ch,
        add_coords=add_coords,
        expect_ctx=expect_ctx,
        inner_in_channels=fused_in_ch,
        use_1x1_proj=False,  # not needed since inner expects fused channels
    )
    return model