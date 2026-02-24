from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset


class BaseRasterTileDataset(Dataset):
    """
    Shared dataset base for "tile window from raster" tasks (seg + cls),

    What this base provides:
      - CSV loading & concatenation
      - required column checks + int casting for x0/y0/x1/y1
      - fast window reads for PAN/imagery via rasterio.windows
      - LRU cache for context thumbnails (small images)
      - robust NumPy->torch conversion that fixes negative strides after aug flips/rots
      - optional preload pattern (subclass-controlled sample construction)

    What subclasses implement:
      - build_sample(row_idx, row) -> Dict[str, Any]
        (and optionally build_preload_sample(...) if you want different behavior)
      - required_columns() override if needed
    """

    # ---- required columns for "windowed tile" datasets
    _REQ_WINDOW_COLS = ("image_src", "x0", "y0", "x1", "y1")

    def __init__(
        self,
        csv_paths: Optional[Union[Path, Sequence[Path]]] = None,
        *,
        df: Optional[pd.DataFrame] = None,
        # file/column sanitation hook
        sanitize_df: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        # caching
        cache_context: bool = True,
        max_cached_context: int = 64,
        # normalize/scaling
        normalize: bool = True,
        reflectance_max: float = 10_000.0,
        # context
        use_context: bool = True,
        # output channels
        tile_out_channels: int = 1,
        context_out_channels: int = 1,
        # preload (generic; subclasses decide what goes into sample)
        preload: bool = False,
        preload_max_gb: float = 16.0,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self.normalize = bool(normalize)
        self.reflectance_max = float(reflectance_max)

        self.use_context = bool(use_context)
        self.tile_out_channels = int(tile_out_channels)
        self.context_out_channels = int(context_out_channels)

        self.cache_context = bool(cache_context)
        self.max_cached_context = int(max_cached_context)
        self.verbose = bool(verbose)

        self.preload = bool(preload)
        self.preload_max_gb = float(preload_max_gb)

        # LRU cache for context thumbnails (key: str(path) -> np.ndarray)
        self._ctx_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # storage
        self.df = self._load_df(csv_paths=csv_paths, df=df, sanitize_df=sanitize_df)

        # optional preload
        self._preloaded: Optional[List[Dict[str, Any]]] = None
        if self.preload:
            self._maybe_preload_all()

    # -------------------------------------------------------------------------
    # Loading / validation
    # -------------------------------------------------------------------------
    @classmethod
    def required_columns(cls) -> Sequence[str]:
        """Subclasses may override (e.g., segmentation requires mask_src)."""
        return cls._REQ_WINDOW_COLS

    def _load_df(
        self,
        *,
        csv_paths: Optional[Union[Path, Sequence[Path]]],
        df: Optional[pd.DataFrame],
        sanitize_df: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
    ) -> pd.DataFrame:
        if df is None:
            if csv_paths is None:
                raise ValueError("BaseRasterTileDataset: provide either df=... or csv_paths=...")
            if isinstance(csv_paths, (str, Path)):
                paths = [Path(csv_paths)]
            else:
                paths = [Path(p) for p in csv_paths]
            if not paths:
                raise ValueError("BaseRasterTileDataset: csv_paths resolved to an empty list.")
            dfs = [pd.read_csv(p) for p in paths]
            if not dfs:
                raise RuntimeError(f"BaseRasterTileDataset: no CSVs found for {csv_paths}")
            df0 = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        else:
            df0 = df.copy()

        if sanitize_df is not None:
            df0 = sanitize_df(df0)

        # required columns
        req = set(self.required_columns())
        missing = req - set(df0.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        # cast window coords
        for c in ("x0", "y0", "x1", "y1"):
            df0[c] = df0[c].astype(int)

        return df0.reset_index(drop=True)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> "BaseRasterTileDataset":
        """
        Generic from_dataframe for subclasses that use the same init signature.
        Subclasses with extra init args can override or just rely on **kwargs.
        """
        return cls(df=df, **kwargs)

    # --------------------------------------------
    # Dataset API
    # --------------------------------------------
    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._preloaded is not None:
            return self._preloaded[int(idx)]
        row = self.df.iloc[int(idx)]
        return self.build_sample(int(idx), row)

    # --------------------------------------------
    # Abstract hooks (implement in subclasses)
    # --------------------------------------------
    def build_sample(self, row_idx: int, row: pd.Series) -> Dict[str, Any]:
        """
        Subclasses implement task-specific output formatting here.
        Typical patterns:
          - classification: return tile_tensor/context_tensor/class_id/row_idx/aug stuff
          - segmentation: return tile_tensor/context_tensor/mask/meta/ratios
        """
        raise NotImplementedError

    def build_preload_sample(self, row_idx: int, row: pd.Series) -> Dict[str, Any]:
        """
        By default, preload uses build_sample. Subclasses can override if:
          - we want to skip augmentation during preload
          - we want to store lighter-weight data and post-process later
        """
        return self.build_sample(row_idx, row)

    # -------------------
    # Raster window I/O
    # -------------------
    @staticmethod
    def _window(x0: int, y0: int, x1: int, y1: int) -> Window:
        return Window.from_slices((int(y0), int(y1)), (int(x0), int(x1)))

    def read_single_band_window01(
        self,
        img_path: Path,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> np.ndarray:
        """
        Shared logic from datasets:
          - read band 1 window
          - set nodata to 0
          - optionally scale/clip to [0,1] using reflectance_max
        """
        with rasterio.Env():
            with rasterio.open(img_path) as ds:
                w = self._window(x0, y0, x1, y1)
                arr = ds.read(1, window=w, boundless=False)
                arr = np.asarray(arr, dtype=np.float32, order="C")
                nodata = ds.nodata
                if nodata is not None:
                    arr[arr == nodata] = 0.0

        if self.normalize:
            np.clip(arr, 0.0, self.reflectance_max, out=arr)
            arr /= float(self.reflectance_max)

        return arr

    def read_mask_window_int(
        self,
        mask_path: Path,
        pan_path: Path,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        invalid_to: int = 0,
        mask_dtype: np.dtype = np.int64,
    ) -> np.ndarray:
        """
        Water dataset behavior:
          - read mask window
          - read PAN window to identify invalid pixels (nodata/non-finite)
          - set those mask pixels to `invalid_to`
        """
        w = self._window(x0, y0, x1, y1)
        with rasterio.Env():
            with rasterio.open(mask_path) as mds:
                m = mds.read(1, window=w, boundless=False)
                m = np.asarray(m, dtype=mask_dtype, order="C")

            with rasterio.open(pan_path) as pds:
                p = pds.read(1, window=w, boundless=False)
                p = np.asarray(p, dtype=np.float32, order="C")
                inv = ~np.isfinite(p)
                if pds.nodata is not None:
                    inv |= p == pds.nodata

        if np.any(inv):
            m = m.copy()
            m[inv] = int(invalid_to)

        return m

    # -------------------------------------------------------------------------
    # Context caching / path inference
    # -------------------------------------------------------------------------
    def _get_from_ctx_cache(self, key: str) -> Optional[np.ndarray]:
        if not self.cache_context:
            return None
        arr = self._ctx_cache.get(key)
        if arr is not None:
            self._ctx_cache.move_to_end(key, last=True)
        return arr

    def _put_in_ctx_cache(self, key: str, arr: np.ndarray) -> None:
        if not self.cache_context:
            return
        self._ctx_cache[key] = arr
        self._ctx_cache.move_to_end(key, last=True)
        while len(self._ctx_cache) > self.max_cached_context:
            self._ctx_cache.popitem(last=False)

    def load_context_thumb01(self, ctx_path: Path) -> np.ndarray:
        """
        Load context image and scale to [0,1] assuming uint8 0..255 input
        """
        key = str(ctx_path)
        cached = self._get_from_ctx_cache(key)
        if cached is not None:
            return cached

        if not ctx_path.exists():
            raise FileNotFoundError(f"Context image missing: {ctx_path}")

        with rasterio.open(ctx_path) as src:
            # Read all bands
            arr = src.read()  # shape: (bands, height, width)
        arr = np.asarray(arr, dtype=np.float32)
        arr /= 255.0

        self._put_in_ctx_cache(key, arr)
        return arr

    # ---------------------------------
    # NumPy -> Torch
    # ---------------------------------
    def to_chw_tensor(self, hw: np.ndarray, out_channels: int) -> torch.Tensor:
        """
        Robust conversion:
          - supports (H,W) or (H,W,3)
          - fixes negative strides via np.ascontiguousarray
          - returns float32 torch tensor (C,H,W)
        """
        hw = np.asarray(hw, dtype=np.float32)

        if hw.ndim == 2:
            if out_channels == 1:
                base = np.ascontiguousarray(hw)
                t = torch.from_numpy(base).unsqueeze(0)
            elif out_channels == 3:
                base = np.ascontiguousarray(hw)
                hw3 = np.repeat(base[..., None], 3, axis=2)
                hw3 = np.ascontiguousarray(hw3)
                t = torch.from_numpy(hw3).permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported out_channels={out_channels}")

        else:
            # (H,W,3)
            if out_channels == 1:
                base = np.ascontiguousarray(hw)
                m = base.mean(axis=2, dtype=np.float32)
                m = np.ascontiguousarray(m)
                t = torch.from_numpy(m).unsqueeze(0)
            elif out_channels == 3:
                base = np.ascontiguousarray(hw)
                t = torch.from_numpy(base).permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported out_channels={out_channels}")

        return t.contiguous()

    # ---------------------------
    # Preload (generic)
    # ---------------------------
    def estimate_preload_gb(self) -> float:
        """
        Conservative default estimate:
          - assumes tile is float32 [tile_out_channels, H, W]
          - context assumed roughly 1/16 the pixel count of tile
          - does NOT include masks/labels (subclasses can override)
        """
        if len(self.df) == 0:
            return 0.0
        r0 = self.df.iloc[0]
        H = int(r0["y1"]) - int(r0["y0"])
        W = int(r0["x1"]) - int(r0["x0"])

        tile_bytes = H * W * self.tile_out_channels * 4  # float32
        ctx_bytes = max(1, H // 4) * max(1, W // 4) * self.context_out_channels * 4
        total = len(self.df) * (tile_bytes + ctx_bytes)
        return float(total) / float(1024**3)

    def _maybe_preload_all(self) -> None:
        est = self.estimate_preload_gb()
        if est > self.preload_max_gb:
            if self.verbose:
                print(f"[preload] estimated {est:.2f} GB > cap {self.preload_max_gb:.2f} GB; skipping preload.")
            self.preload = False
            self._preloaded = None
            return

        if self.verbose:
            print(f"[preload] estimated {est:.2f} GB, loading all samples into RAM...")

        preloaded: List[Dict[str, Any]] = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            sample = self.build_preload_sample(int(i), row)
            preloaded.append(sample)

        self._preloaded = preloaded
        if self.verbose:
            print(f"[preload] loaded {len(preloaded)} samples to RAM.")
