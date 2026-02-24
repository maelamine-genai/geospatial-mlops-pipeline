from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from geo_mlops.core.data.base_dataset import BaseRasterTileDataset


class BuildingShadowWithContextDataset(BaseRasterTileDataset):
    """
    Building+Shadow semantic segmentation dataset (SegFormer-ready).

    Reads chips from:
      - image_src (PAN) as rasterio window
      - mask_src (GT) as rasterio window3
      - context_path (downsampled full-scene context) with small LRU cache

    Returns dict:
      {
        "tile_tensor":    torch.float32 [C,H,W] in [0,1]
        "context_tensor": torch.float32 [C,h,w] in [0,1]
        "mask":           torch.int64   [H,W]   in {0,1,2}
        "meta":           dict (paths, window, scene/geo fields if present)
        "Building_ratio":    float (nan if missing)
        "shadow_ratio":   float (nan if missing)
      }
    """

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        csv_name: str = "Building_tiles_hard_fullval.csv",
        *,
        # discovery (when not using csv_paths/df directly)
        dataset_buckets: Optional[Sequence[str]] = None,  # e.g. ["Golden-Train-Regions"]
        min_any_ratio: Optional[float] = None,
        # caching
        cache_images: bool = True,  # mapped to BaseRasterTileDataset.cache_context
        max_cached_images: int = 64,  # mapped to BaseRasterTileDataset.max_cached_context
        # augmentation knobs
        do_aug: bool = False,
        aug_flip: bool = True,
        aug_rot90: bool = True,
        aug_noise_std: float = 0.0,
        # normalization / channels
        normalize: bool = True,
        reflectance_max: int = 10_000,
        tile_out_channels: int = 1,
        context_out_channels: int = 1,
        # optional extra transforms
        tile_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        context_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        # preload options
        preload: bool = False,
        preload_max_gb: float = 16.0,
        verbose: bool = True,
        # advanced: allow passing df/csv_paths directly (trainer convenience)
        csv_paths: Optional[Union[Path, Sequence[Path]]] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.csv_name = str(csv_name)

        self.min_any_ratio = float(min_any_ratio) if min_any_ratio is not None else None

        self.do_aug = bool(do_aug)
        self.aug_flip = bool(aug_flip)
        self.aug_rot90 = bool(aug_rot90)
        self.aug_noise_std = float(aug_noise_std)

        self.tile_transform = tile_transform
        self.context_transform = context_transform

        self.dataset_buckets = list(dataset_buckets) if dataset_buckets is not None else None

        # Resolve dataframe source:
        #  1) explicit df=...
        #  2) explicit csv_paths=...
        #  3) scan root_dir[/bucket]/**/csv_name like your old dataset
        if df is None and csv_paths is None:
            if self.root_dir is None:
                raise ValueError("BuildingShadowWithContextDataset: provide root_dir=... or df=... or csv_paths=...")
            csv_paths = self._discover_csvs(self.root_dir, self.csv_name, self.dataset_buckets)
            if not csv_paths:
                roots = (
                    [self.root_dir]
                    if self.dataset_buckets is None
                    else [self.root_dir / b for b in self.dataset_buckets]
                )
                raise FileNotFoundError(f"No '{self.csv_name}' found under {', '.join(map(str, roots))}")

        # Build df if csv_paths provided (so we can filter/sort/ensure optional columns)
        if df is None:
            assert csv_paths is not None
            dfs: List[pd.DataFrame] = []
            needed = {"image_src", "mask_src", "x0", "y0", "x1", "y1", "context_path"}
            for p in [Path(x) for x in (csv_paths if isinstance(csv_paths, (list, tuple)) else [csv_paths])]:
                dfi = pd.read_csv(p)
                if not needed.issubset(set(dfi.columns)):
                    continue

                for c in ("x0", "y0", "x1", "y1"):
                    dfi[c] = dfi[c].astype(int)

                # Ensure optional columns exist (so meta building is safe)
                for opt in (
                    "Building_ratio",
                    "shadow_ratio",
                    "scene_id",
                    "tile_cx_norm",
                    "tile_cy_norm",
                    "gsd_mpp",
                    "tile_size_px",
                    "stride_px",
                    "overlap",
                    "region",
                    "subregion",
                ):
                    if opt not in dfi.columns:
                        dfi[opt] = np.nan

                if self.min_any_ratio is not None:
                    wr = pd.to_numeric(dfi["Building_ratio"], errors="coerce").fillna(0.0)
                    sr = pd.to_numeric(dfi["shadow_ratio"], errors="coerce").fillna(0.0)
                    dfi = dfi[(wr + sr) >= float(self.min_any_ratio)]

                if len(dfi):
                    dfs.append(dfi)

            if not dfs:
                raise RuntimeError("No valid rows found after filtering CSVs.")
            df = pd.concat(dfs, ignore_index=True)

            # Stable sort (nice for debugging/repro)
            sort_cols = [c for c in ["region", "subregion", "image_src", "y0", "x0"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols, ignore_index=True)

        # Now initialize the base with df
        super().__init__(
            df=df,
            cache_context=bool(cache_images),
            max_cached_context=int(max_cached_images),
            normalize=bool(normalize),
            reflectance_max=float(reflectance_max),
            use_context=True,
            tile_out_channels=int(tile_out_channels),
            context_out_channels=int(context_out_channels),
            preload=bool(preload),
            preload_max_gb=float(preload_max_gb),
            verbose=bool(verbose),
        )

    # ---------------------------------------------------------------------
    # Discovery
    # ---------------------------------------------------------------------
    @staticmethod
    def _discover_csvs(root_dir: Path, csv_name: str, buckets: Optional[Sequence[str]]) -> List[Path]:
        if buckets is None:
            search_roots = [root_dir]
        else:
            search_roots = [root_dir / b for b in buckets]
        csvs: List[Path] = []
        for sr in search_roots:
            if not sr.exists():
                raise FileNotFoundError(f"Dataset bucket root not found: {sr}")
            csvs.extend(sorted(sr.rglob(csv_name)))
        return csvs

    # ---------------------------------------------------------------------
    # Schema
    # ---------------------------------------------------------------------
    @classmethod
    def required_columns(cls) -> Sequence[str]:
        # segmentation requires mask_src + context_path
        return ("image_src", "mask_src", "context_path", "x0", "y0", "x1", "y1")

    # ---------------------------------------------------------------------
    # Core sample building
    # ---------------------------------------------------------------------
    def build_preload_sample(self, row_idx: int, row: pd.Series) -> Dict[str, Any]:
        """
        When preloading, we usually want *no random augmentation* so that:
          - RAM sample == deterministic
          - augmentations happen in training collate/transform stage if desired
        This matches your previous behavior (you didn’t augment during preload).
        """
        return self._build_sample_impl(row_idx, row, allow_aug=False)

    def build_sample(self, row_idx: int, row: pd.Series) -> Dict[str, Any]:
        return self._build_sample_impl(row_idx, row, allow_aug=True)

    def _build_sample_impl(self, row_idx: int, row: pd.Series, *, allow_aug: bool) -> Dict[str, Any]:
        img_p = Path(row["image_src"])
        msk_p = Path(row["mask_src"])
        ctx_p = Path(row["context_path"])
        x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])

        # window reads
        pan_hw = self.read_single_band_window01(img_p, x0, y0, x1, y1)  # (H,W) float32 [0,1]
        mask_hw = self.read_mask_window_int(msk_p, img_p, x0, y0, x1, y1, invalid_to=0, mask_dtype=np.int64)

        # context
        context_hw = self.load_context_thumb01(ctx_p)  # (h,w) or (h,w,3) float32 [0,1]

        # tensors
        tile_t = self.to_chw_tensor(pan_hw, self.tile_out_channels)  # float32
        ctx_t = self.to_chw_tensor(context_hw, self.context_out_channels)
        mask_t = torch.from_numpy(mask_hw.astype(np.int64, copy=False))

        # optional transforms
        if self.tile_transform is not None:
            tile_t = self.tile_transform(tile_t)
        if self.context_transform is not None:
            ctx_t = self.context_transform(ctx_t)

        # augmentation (geom sync)
        if allow_aug and self.do_aug:
            tile_t, ctx_t, mask_t = self._geom_augment_sync(tile_t, ctx_t, mask_t)

        # noise aug (tile only) - keep identical to your knobs
        if allow_aug and self.do_aug and self.aug_noise_std > 0.0:
            noise = torch.randn_like(tile_t) * float(self.aug_noise_std)
            tile_t = torch.clamp(tile_t + noise, 0.0, 1.0)

        meta = self._build_meta(row, img_p, msk_p, ctx_p, x0, y0, x1, y1)

        return {
            "tile_tensor": tile_t,
            "context_tensor": ctx_t,
            "mask": mask_t,
            "meta": meta,
            "Building_ratio": float(row.get("Building_ratio", np.nan)),
            "shadow_ratio": float(row.get("shadow_ratio", np.nan)),
        }

    def _build_meta(
        self,
        row: pd.Series,
        img_p: Path,
        msk_p: Path,
        ctx_p: Path,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> Dict[str, Any]:
        # Keep meta structure identical to your previous dataset
        def _to_int_or0(v: Any) -> int:
            if pd.isna(v):
                return 0
            return int(v)

        def _to_float_or_nan(v: Any) -> float:
            if pd.isna(v):
                return float("nan")
            return float(v)

        return {
            "image_src": str(img_p),
            "mask_src": str(msk_p),
            "context_path": str(ctx_p),
            "window": (int(x0), int(y0), int(x1), int(y1)),
            "scene_id": row.get("scene_id", ""),
            "tile_cx_norm": _to_float_or_nan(row.get("tile_cx_norm", np.nan)),
            "tile_cy_norm": _to_float_or_nan(row.get("tile_cy_norm", np.nan)),
            "gsd_mpp": _to_float_or_nan(row.get("gsd_mpp", np.nan)),
            "tile_size_px": _to_int_or0(row.get("tile_size_px", (x1 - x0))),
            "stride_px": _to_int_or0(row.get("stride_px", 0)),
            "overlap": _to_float_or_nan(row.get("overlap", np.nan)),
            "region": row.get("region", ""),
            "subregion": row.get("subregion", ""),
        }

    # ---------------------------------------------------------------------
    # Augmentations
    # ---------------------------------------------------------------------
    def _geom_augment_sync(
        self,
        tile_t: torch.Tensor,  # (C,H,W), float in [0,1]
        ctx_t: torch.Tensor,  # (C,h,w), float in [0,1]
        mask_t: torch.Tensor,  # (H,W),   int64
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random H/V flips and 90° rotations applied to all three in lockstep."""
        # random horizontal flip
        if self.aug_flip and (np.random.rand() < 0.5):
            tile_t = torch.flip(tile_t, dims=[2])  # W
            ctx_t = torch.flip(ctx_t, dims=[2])
            mask_t = torch.flip(mask_t, dims=[1])

        # random vertical flip
        if self.aug_flip and (np.random.rand() < 0.5):
            tile_t = torch.flip(tile_t, dims=[1])  # H
            ctx_t = torch.flip(ctx_t, dims=[1])
            mask_t = torch.flip(mask_t, dims=[0])

        # random 0/90/180/270 rotation
        if self.aug_rot90:
            k = int(np.random.randint(0, 4))
            if k:
                tile_t = torch.rot90(tile_t, k, dims=[1, 2])
                ctx_t = torch.rot90(ctx_t, k, dims=[1, 2])
                mask_t = torch.rot90(mask_t, k, dims=[0, 1])

        return tile_t, ctx_t, mask_t

    # --------------------------
    # Convenience constructor
    # --------------------------
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        cache_images: bool = True,
        max_cached_images: int = 64,
        do_aug: bool = False,
        aug_flip: bool = True,
        aug_rot90: bool = True,
        aug_noise_std: float = 0.0,
        normalize: bool = True,
        reflectance_max: int = 10_000,
        tile_out_channels: int = 1,
        context_out_channels: int = 1,
        min_any_ratio: Optional[float] = None,
        tile_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        context_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        preload: bool = False,
        preload_max_gb: float = 16.0,
        verbose: bool = True,
    ) -> "BuildingShadowWithContextDataset":
        if df is None or len(df) == 0:
            raise RuntimeError("from_dataframe: df is empty.")

        needed = {"image_src", "mask_src", "context_path", "x0", "y0", "x1", "y1"}
        missing = needed - set(df.columns)
        if missing:
            raise RuntimeError(f"from_dataframe: missing required columns: {sorted(missing)}")

        # normalize types + ensure optional cols exist
        df = df.copy()
        for c in ("x0", "y0", "x1", "y1"):
            df[c] = df[c].astype(int)

        for opt in (
            "Building_ratio",
            "shadow_ratio",
            "scene_id",
            "tile_cx_norm",
            "tile_cy_norm",
            "gsd_mpp",
            "tile_size_px",
            "stride_px",
            "overlap",
            "region",
            "subregion",
        ):
            if opt not in df.columns:
                df[opt] = np.nan

        if min_any_ratio is not None:
            wr = pd.to_numeric(df["Building_ratio"], errors="coerce").fillna(0.0)
            sr = pd.to_numeric(df["shadow_ratio"], errors="coerce").fillna(0.0)
            df = df[(wr + sr) >= float(min_any_ratio)].copy()
            if len(df) == 0:
                raise RuntimeError("from_dataframe: df empty after min_any_ratio filtering.")

        # Create directly via init(df=...)
        return cls(
            root_dir=".",
            csv_name="",
            df=df.reset_index(drop=True),
            cache_images=cache_images,
            max_cached_images=max_cached_images,
            do_aug=do_aug,
            aug_flip=aug_flip,
            aug_rot90=aug_rot90,
            aug_noise_std=aug_noise_std,
            normalize=normalize,
            reflectance_max=reflectance_max,
            tile_out_channels=tile_out_channels,
            context_out_channels=context_out_channels,
            min_any_ratio=None,
            tile_transform=tile_transform,
            context_transform=context_transform,
            preload=preload,
            preload_max_gb=preload_max_gb,
            verbose=verbose,
        )
