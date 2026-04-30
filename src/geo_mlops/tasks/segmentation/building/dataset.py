from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from geo_mlops.core.data.base import BaseRasterTileDataset, TileRecord


@dataclass(frozen=True)
class BuildingDatasetConfig:
    # PAN reflectance normalization
    reflectance_max: float = 10_000.0

    # Context usage
    use_context: bool = True

    # Output channels
    tile_out_channels: int = 1
    context_out_channels: int = 1

    # Augmentation
    do_aug: bool = False
    aug_flip: bool = True
    aug_rot90: bool = True
    aug_noise_std: float = 0.0

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any] | None) -> "BuildingDatasetConfig":
        cfg = cfg or {}
        return cls(
            reflectance_max=float(cfg.get("reflectance_max", cls.reflectance_max)),
            use_context=bool(cfg.get("use_context", cls.use_context)),
            tile_out_channels=int(cfg.get("tile_out_channels", cls.tile_out_channels)),
            context_out_channels=int(cfg.get("context_out_channels", cls.context_out_channels)),
            do_aug=bool(cfg.get("do_aug", cls.do_aug)),
            aug_flip=bool(cfg.get("aug_flip", cls.aug_flip)),
            aug_rot90=bool(cfg.get("aug_rot90", cls.aug_rot90)),
            aug_noise_std=float(cfg.get("aug_noise_std", cls.aug_noise_std)),
        )


class BuildingDataset(BaseRasterTileDataset):
    """
    Building segmentation dataset.

    Expected tiles_master.csv columns:
      - scene_id
      - image_src
      - gt_src
      - x0, y0, x1, y1
      - context_src if use_context=True

    Returns:
      {
        "tile_tensor": torch.float32 [C,H,W],
        "context_tensor": torch.float32 [C,H,W], optional,
        "mask": torch.int64 [H,W],
        "meta": dict
      }
    """

    def __init__(
        self,
        *,
        tiles_df: pd.DataFrame,
        indices: Optional[Sequence[int]] = None,
        cfg: BuildingDatasetConfig = BuildingDatasetConfig(),
        cache_context: bool = True,
        context_cache_max_items: int = 256,
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

    @classmethod
    def required_columns(cls) -> Tuple[str, ...]:
        return ("scene_id", "image_src", "gt_src", "x0", "y0", "x1", "y1")

    def _pan_window_float01(self, rec: TileRecord) -> np.ndarray:
        win = self._window_from_record(rec)

        pan_hw = self.read_window(
            rec.image_src,
            win,
            band=1,
            out_dtype=np.float32,
        )

        ref_max = max(1e-6, float(self.cfg.reflectance_max))
        pan_hw = np.clip(pan_hw, 0.0, ref_max) / ref_max
        return pan_hw.astype(np.float32, copy=False)

    def _gt_window_int(self, rec: TileRecord) -> np.ndarray:
        if rec.gt_src is None:
            raise ValueError(f"gt_src missing for scene_id={rec.scene_id}")

        win = self._window_from_record(rec)
        gt_hw = self.read_window(rec.gt_src, win, band=1)
        return gt_hw.astype(np.int64, copy=False)

    def _context_float01(self, rec: TileRecord) -> torch.Tensor:
        if rec.context_src is None:
            raise ValueError(f"context_src missing for scene_id={rec.scene_id}")

        ctx = self.read_context(rec)
        ctx = np.asarray(ctx, dtype=np.float32) / 255.0

        ctx_t = torch.from_numpy(ctx)
        ctx_t = self._to_channels(ctx_t, int(self.cfg.context_out_channels))
        return ctx_t

    @staticmethod
    def _resize_context_to_tile(ctx_t: torch.Tensor, height: int, width: int) -> torch.Tensor:
        ctx = ctx_t.unsqueeze(0)
        ctx = F.interpolate(
            ctx,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return ctx.squeeze(0)

    @staticmethod
    def _to_channels(x: torch.Tensor, out_channels: int) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [C,H,W], got shape {tuple(x.shape)}")

        channels = int(x.shape[0])

        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        if channels == out_channels:
            return x

        if channels == 1 and out_channels > 1:
            return x.repeat(out_channels, 1, 1)

        if channels > out_channels:
            return x[:out_channels, :, :]

        pad = out_channels - channels
        return torch.cat([x, x[-1:, :, :].repeat(pad, 1, 1)], dim=0)

    def build_sample(self, rec: TileRecord) -> Dict[str, Any]:
        if self.cfg.use_context and rec.context_src is None:
            raise ValueError(
                f"context_src missing but use_context=True for scene_id={rec.scene_id}"
            )

        pan_hw = self._pan_window_float01(rec)
        gt_hw = self._gt_window_int(rec)

        tile_t = torch.from_numpy(pan_hw).unsqueeze(0)
        tile_t = self._to_channels(tile_t, int(self.cfg.tile_out_channels))

        mask_t = torch.from_numpy(gt_hw)

        if self.tile_transform is not None:
            tile_t = self.tile_transform(tile_t)

        ctx_t: Optional[torch.Tensor] = None

        if self.cfg.use_context:
            ctx_t = self._context_float01(rec)
            _, height, width = tile_t.shape
            ctx_t = self._resize_context_to_tile(ctx_t, height, width)

            if self.context_transform is not None:
                ctx_t = self.context_transform(ctx_t)

        if self.cfg.do_aug:
            tile_t, mask_t, ctx_t = self._geom_augment_sync(
                tile_t=tile_t,
                mask_t=mask_t,
                ctx_t=ctx_t,
                aug_flip=self.cfg.aug_flip,
                aug_rot90=self.cfg.aug_rot90,
            )

            if self.cfg.aug_noise_std > 0.0:
                noise = torch.randn_like(tile_t) * float(self.cfg.aug_noise_std)
                tile_t = torch.clamp(tile_t + noise, 0.0, 1.0)

        out: Dict[str, Any] = {
            "tile_tensor": tile_t.contiguous(),
            "mask": mask_t.long().contiguous(),
            "meta": self._build_meta(rec),
        }

        if ctx_t is not None:
            out["context_tensor"] = ctx_t.contiguous()

        return out

    @staticmethod
    def _build_meta(rec: TileRecord) -> Dict[str, Any]:
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
        *,
        tile_t: torch.Tensor,
        mask_t: torch.Tensor,
        ctx_t: Optional[torch.Tensor],
        aug_flip: bool,
        aug_rot90: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if aug_flip:
            if np.random.rand() < 0.5:
                tile_t = torch.flip(tile_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[1])
                if ctx_t is not None:
                    ctx_t = torch.flip(ctx_t, dims=[2])

            if np.random.rand() < 0.5:
                tile_t = torch.flip(tile_t, dims=[1])
                mask_t = torch.flip(mask_t, dims=[0])
                if ctx_t is not None:
                    ctx_t = torch.flip(ctx_t, dims=[1])

        if aug_rot90 and np.random.rand() < 0.5:
            k = int(np.random.randint(0, 4))
            if k:
                tile_t = torch.rot90(tile_t, k, dims=[1, 2])
                mask_t = torch.rot90(mask_t, k, dims=[0, 1])
                if ctx_t is not None:
                    ctx_t = torch.rot90(ctx_t, k, dims=[1, 2])

        return tile_t, mask_t, ctx_t
    