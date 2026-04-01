from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


@dataclass(frozen=True)
class TileRecord:
    """
    Minimal, task-agnostic tile record parsed from a row in tiles_master.csv.

    NOTE: Task-specific datasets should override `row_to_record()` if they need more fields.
    """
    # tile_id: str
    scene_id: str
    image_src: Path
    x0: int
    y0: int
    x1: int
    y1: int

    # Optional common fields (may be empty / missing in some tasks)
    gt_src: Optional[Path] = None
    pred_src: Optional[Path] = None
    context_src: Optional[Path] = None

    # Optional metadata (not required for loading)
    region: Optional[str] = None
    subregion: Optional[str] = None
    stem: Optional[str] = None


class BaseRasterTileDataset:
    """
    Base dataset for raster tile training/inference.

    Responsibilities (and ONLY these):
      - Hold tiles DataFrame (tiles_master.csv) and optional subset indices.
      - Parse rows into a task-agnostic TileRecord.
      - Read raster windows from image_src (and optionally gt/pred/context if subclasses request).
      - Return a dict payload; subclasses decide exact keys and tensor conversion.

    Explicitly NOT handled here (because tiling stage already handled/decided it):
      - Any nodata/validity checks, "skip invalid tiles", presence thresholds, mining logic.
      - Any task semantics (classes, label meanings, loss targets).
      - Any geospatial correctness beyond reading the requested window.
    """

    def __init__(
        self,
        *,
        tiles_df: pd.DataFrame,
        indices: Optional[Sequence[int]] = None,
        cache_context: bool = True,
        context_cache_max_items: int = 256,
    ):
        self.df = tiles_df.reset_index(drop=True)

        self._indices: Optional[np.ndarray]
        if indices is None:
            self._indices = None
        else:
            self._indices = np.asarray(indices, dtype=np.int64)

        self.cache_context = bool(cache_context)
        self.context_cache_max_items = int(max(0, context_cache_max_items))
        self._context_cache: Dict[str, np.ndarray] = {}

        self._validate_required_columns()

    # -------------------------
    # Required columns contract
    # -------------------------
    @classmethod
    def required_columns(cls) -> Tuple[str, ...]:
        """
        Minimum columns needed to read image windows. Task datasets typically add:
          - gt_src (for supervised training)
          - context_src (if using context)
          - any task-specific fields (class_id, etc.)
        """
        return ("scene_id", "image_src", "x0", "y0", "x1", "y1")

    def _validate_required_columns(self) -> None:
        missing = [c for c in self.required_columns() if c not in self.df.columns]
        if missing:
            raise ValueError(f"Tiles dataframe missing required columns: {missing}")

    # -------------------------
    # Indexing / length
    # -------------------------
    def __len__(self) -> int:
        return int(len(self._indices)) if self._indices is not None else int(len(self.df))

    def _resolve_df_index(self, idx: int) -> int:
        if self._indices is None:
            return int(idx)
        return int(self._indices[int(idx)])

    # -------------------------
    # Row parsing
    # -------------------------
    def row_to_record(self, row: Mapping[str, Any]) -> TileRecord:
        """
        Convert a df row to a TileRecord. Subclasses may override to add fields.
        """
        def _p(x: Any) -> Optional[Path]:
            if x is None:
                return None
            s = str(x)
            if not s or s.lower() == "nan":
                return None
            return Path(s)

        return TileRecord(
            # tile_id=str(row["tile_id"]),
            scene_id=str(row["scene_id"]),
            image_src=Path(str(row["image_src"])),
            x0=int(row["x0"]),
            y0=int(row["y0"]),
            x1=int(row["x1"]),
            y1=int(row["y1"]),
            gt_src=_p(row.get("gt_src", None)),
            pred_src=_p(row.get("pred_src", None)),
            context_src=_p(row.get("context_src", None)),
            region=str(row.get("region")) if "region" in row else None,
            subregion=str(row.get("subregion")) if "subregion" in row else None,
            stem=str(row.get("stem")) if "stem" in row else None,
        )

    # -------------------------
    # Raster IO helpers
    # -------------------------
    @staticmethod
    def _window_from_record(rec: TileRecord) -> Window:
        w = int(rec.x1 - rec.x0)
        h = int(rec.y1 - rec.y0)
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid window for tile_id={rec.tile_id}: "
                             f"(x0,y0,x1,y1)=({rec.x0},{rec.y0},{rec.x1},{rec.y1})")
        return Window(col_off=int(rec.x0), row_off=int(rec.y0), width=w, height=h)

    @staticmethod
    def read_window(
        raster_path: Path,
        window: Window,
        *,
        band: Optional[int] = None,
        out_dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """
        Read a raster window. Returns:
          - (H, W) if band is provided
          - (C, H, W) if band is None (all bands)
        """
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster not found: {raster_path}")

        with rasterio.open(raster_path) as src:
            if band is None:
                arr = src.read(window=window)  # (C,H,W)
            else:
                arr = src.read(band, window=window)  # (H,W)

        if out_dtype is not None:
            arr = np.asarray(arr, dtype=out_dtype)
        return arr

    def read_image_window(self, rec: TileRecord) -> np.ndarray:
        """
        Read the input image chip. Default behavior reads all bands.
        Subclasses can override if they want band=1 for PAN, etc.
        """
        win = self._window_from_record(rec)
        return self.read_window(rec.image_src, win, band=None)

    def read_mask_window(self, rec: TileRecord, *, src_path: Path) -> np.ndarray:
        """
        Generic mask window reader (single-band). Subclasses may cast / remap labels.
        """
        win = self._window_from_record(rec)
        return self.read_window(src_path, win, band=1)

    # -------------------------
    # Context caching (optional)
    # -------------------------
    def _ctx_cache_get(self, key: str) -> Optional[np.ndarray]:
        if not self.cache_context:
            return None
        return self._context_cache.get(key)

    def _ctx_cache_put(self, key: str, value: np.ndarray) -> None:
        if not self.cache_context:
            return
        if self.context_cache_max_items <= 0:
            return
        if key in self._context_cache:
            self._context_cache[key] = value
            return
        # naive cap: pop an arbitrary item when full (good enough for v0)
        if len(self._context_cache) >= self.context_cache_max_items:
            self._context_cache.pop(next(iter(self._context_cache)))
        self._context_cache[key] = value

    def read_context(self, rec: TileRecord) -> np.ndarray:
        """
        Read full context raster (not windowed). Context is expected to be "small"
        (e.g., downsampled overview created in a DataOps stage).

        Returns (C,H,W). No normalization is applied here; task datasets can normalize.
        """
        if rec.context_src is None:
            raise ValueError(f"context_src missing for tile_id={rec.tile_id}")

        key = str(rec.context_src)
        cached = self._ctx_cache_get(key)
        if cached is not None:
            return cached

        if not rec.context_src.exists():
            raise FileNotFoundError(f"Context raster not found: {rec.context_src}")

        with rasterio.open(rec.context_src) as src:
            arr = src.read()  # (C,H,W)

        self._ctx_cache_put(key, arr)
        return arr

    # -------------------------
    # Output contract
    # -------------------------
    def build_sample(self, rec: TileRecord) -> Dict[str, Any]:
        """
        Build a task-agnostic sample dict.

        Task-specific datasets should override this to:
          - add gt/pred/context reads
          - convert to torch tensors
          - apply normalization/augmentations
        """
        img = self.read_image_window(rec)
        return {
            "tile_id": rec.tile_id,
            "scene_id": rec.scene_id,
            "image": img,
            "x0": rec.x0,
            "y0": rec.y0,
            "x1": rec.x1,
            "y1": rec.y1,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        df_idx = self._resolve_df_index(idx)
        row = self.df.iloc[df_idx].to_dict()
        rec = self.row_to_record(row)
        return self.build_sample(rec)