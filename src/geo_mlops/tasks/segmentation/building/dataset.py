from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from geo_mlops.core.data.base_dataset import BaseRasterTileDataset, TileRecord


@dataclass(frozen=True)
class BuildingSegConfig:
    # Normalization for PAN reflectance -> [0,1]
    reflectance_max: float = 10_000.0

    # Whether context is required/used
    use_context: bool = True

    # Output channels (mostly for convenience if you later want to expand channels)
    tile_out_channels: int = 1
    context_out_channels: int = 1

    # Augmentation knobs
    do_aug: bool = False
    aug_flip: bool = True
    aug_rot90: bool = True
    aug_noise_std: float = 0.0  # applied to tile only


class BuildingSegWithContextDataset(BaseRasterTileDataset):
    """
    Building semantic segmentation dataset built on the new BaseRasterTileDataset.

    Expected tiles_master.csv columns (contract-driven):
      - scene_id, image_src, gt_src, context_src
      - x0, y0, x1, y1
      - plus optional meta columns (region/subregion/stem/gsd/stride/etc.)

    Returns:
      {
        "tile_tensor":    torch.float32 [C,H,W] in [0,1]
        "context_tensor": torch.float32 [C,h,w] in [0,1]   (if use_context)
        "mask":           torch.int64   [H,W]   (binary or multi-class depending on GT)
        "meta":           dict
      }
    """

    def __init__(
        self,
        *,
        tiles_df: pd.DataFrame,
        indices: Optional[Sequence[int]] = None,
        cfg: BuildingSegConfig = BuildingSegConfig(),
        # caching (maps to BaseRasterTileDataset context cache)
        cache_context: bool = True,
        context_cache_max_items: int = 256,
        # optional transforms
        tile_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        context_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.cfg = cfg
        self.tile_transform = tile_transform
        self.context_transform = context_transform

        super().__init__(
            tiles_df=tiles_df,
            indices=indices,
            cache_context=cache_context,
            context_cache_max_items=context_cache_max_items,
        )

    # -------------------------
    # Schema
    # -------------------------
    @classmethod
    def required_columns(cls) -> Tuple[str, ...]:
        # base requires scene_id, image_src, x0,y0,x1,y1
        # for building seg we require gt_src; context_src only if enabled
        return ("scene_id", "image_src", "gt_src", "x0", "y0", "x1", "y1")

    # -------------------------
    # IO helpers
    # -------------------------
    def _pan_window_float01(self, rec: TileRecord) -> np.ndarray:
        """
        Read single-band PAN window and normalize to [0,1] using reflectance_max.
        Returns (H, W) float32.
        """
        win = self._window_from_record(rec)
        pan_hw = self.read_window(rec.image_src, win, band=1, out_dtype=np.float32)  # (H,W)
        pan_hw = np.clip(pan_hw, 0.0, float(self.cfg.reflectance_max)) / max(1e-6, float(self.cfg.reflectance_max))
        return pan_hw.astype(np.float32, copy=False)

    def _gt_window_int(self, rec: TileRecord) -> np.ndarray:
        """
        Read single-band GT window. Returns (H,W) int64.
        """
        if rec.gt_src is None:
            raise ValueError(f"gt_src missing for scene_id={rec.scene_id}")
        win = self._window_from_record(rec)
        gt_hw = self.read_window(rec.gt_src, win, band=1)  # dtype as stored
        return gt_hw.astype(np.int64, copy=False)

    def _context_float01(self, rec: TileRecord) -> torch.Tensor:
        """
        Read full context raster and normalize to [0,1].
        Context is expected to be uint8 (from your make_context stage).
        Returns torch tensor [C,h,w] float32.
        """
        ctx = self.read_context(rec)  # (C,h,w) numpy
        ctx = np.asarray(ctx, dtype=np.float32) / 255.0
        ctx_t = torch.from_numpy(ctx)
        # enforce channel count if desired
        ctx_t = self._to_channels(ctx_t, int(self.cfg.context_out_channels))
        return ctx_t
    
    def _resize_context_to_tile(self, ctx_t: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # ctx_t: [C,h,w] -> [C,H,W]
        ctx = ctx_t.unsqueeze(0)  # [1,C,h,w]
        ctx = F.interpolate(ctx, size=(H, W), mode="bilinear", align_corners=False)
        return ctx.squeeze(0)

    @staticmethod
    def _to_channels(x: torch.Tensor, out_channels: int) -> torch.Tensor:
        """
        Ensure tensor is [C,H,W] with out_channels.
        If C==1 and out_channels==3, repeat. If C==3 and out_channels==1, take first band.
        Otherwise, no-op.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected [C,H,W], got shape {tuple(x.shape)}")
        c, h, w = x.shape
        if out_channels == c:
            return x
        if c == 1 and out_channels > 1:
            return x.repeat(out_channels, 1, 1)
        if c > 1 and out_channels == 1:
            return x[:1, :, :]
        # fallback: pad/truncate
        if c > out_channels:
            return x[:out_channels, :, :]
        pad = out_channels - c
        return torch.cat([x, x[-1:, :, :].repeat(pad, 1, 1)], dim=0)

    # -------------------------
    # Sample building
    # -------------------------
    def build_sample(self, rec: TileRecord) -> Dict[str, Any]:
        # Enforce context requirement at dataset level if enabled
        if self.cfg.use_context and rec.context_src is None:
            raise ValueError(f"context_src missing but use_context=True for scene_id={rec.scene_id}")

        # Read
        pan_hw = self._pan_window_float01(rec)            # (H,W) float32
        gt_hw = self._gt_window_int(rec)                  # (H,W) int64

        # To tensors
        tile_t = torch.from_numpy(pan_hw).unsqueeze(0)    # [1,H,W]
        tile_t = self._to_channels(tile_t, int(self.cfg.tile_out_channels))

        mask_t = torch.from_numpy(gt_hw)                  # [H,W] int64

        # Optional transforms
        if self.tile_transform is not None:
            tile_t = self.tile_transform(tile_t)

        ctx_t: Optional[torch.Tensor] = None
        if self.cfg.use_context:
            ctx_t = self._context_float01(rec)
            _, H, W = tile_t.shape
            ctx_t = self._resize_context_to_tile(ctx_t, H, W)  # [C,H,W]
            if self.context_transform is not None:
                ctx_t = self.context_transform(ctx_t)

        # Augmentations (geom sync between tile/mask/context)
        if self.cfg.do_aug:
            tile_t, mask_t, ctx_t = self._geom_augment_sync(tile_t, mask_t, ctx_t)

            if self.cfg.aug_noise_std > 0.0:
                noise = torch.randn_like(tile_t) * float(self.cfg.aug_noise_std)
                tile_t = torch.clamp(tile_t + noise, 0.0, 1.0)

        meta = self._build_meta(rec)

        out: Dict[str, Any] = {
            "tile_tensor": tile_t,
            "mask": mask_t,
            "meta": meta,
        }
        if ctx_t is not None:
            out["context_tensor"] = ctx_t
        return out

    def _build_meta(self, rec: TileRecord) -> Dict[str, Any]:
        # Keep meta lightweight; training can log more if needed
        return {
            "scene_id": rec.scene_id,
            "image_src": str(rec.image_src),
            "gt_src": str(rec.gt_src) if rec.gt_src else "",
            "context_src": str(rec.context_src) if rec.context_src else "",
            "window": (rec.x0, rec.y0, rec.x1, rec.y1),
            "region": rec.region or "",
            "subregion": rec.subregion or "",
            "stem": rec.stem or "",
        }

    @staticmethod
    def _geom_augment_sync(
        tile_t: torch.Tensor,              # [C,H,W]
        mask_t: torch.Tensor,              # [H,W]
        ctx_t: Optional[torch.Tensor],     # [C,h,w] or None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Horizontal flip
        if np.random.rand() < 0.5:
            tile_t = torch.flip(tile_t, dims=[2])
            mask_t = torch.flip(mask_t, dims=[1])
            if ctx_t is not None:
                ctx_t = torch.flip(ctx_t, dims=[2])

        # Vertical flip
        if np.random.rand() < 0.5:
            tile_t = torch.flip(tile_t, dims=[1])
            mask_t = torch.flip(mask_t, dims=[0])
            if ctx_t is not None:
                ctx_t = torch.flip(ctx_t, dims=[1])

        # Rot90
        if np.random.rand() < 0.5:
            k = int(np.random.randint(0, 4))
            if k:
                tile_t = torch.rot90(tile_t, k, dims=[1, 2])
                mask_t = torch.rot90(mask_t, k, dims=[0, 1])
                if ctx_t is not None:
                    ctx_t = torch.rot90(ctx_t, k, dims=[1, 2])

        return tile_t, mask_t, ctx_t