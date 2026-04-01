from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rasterio

import numpy as np
from geo_mlops.core.tiling.adapters.base import (
    SceneArrays,
    SceneInputs,
    TaskAdapter,
    TileWindow,
    TilingPolicy,
)
from geo_mlops.core.tiling.utils import (
    _relaxed_lookup,
    compute_gsd_from_gcps,
    gsd_from_epsg4326,
    gen_tiles_cover,
)


# -----------------------------------------
# Engine config (task-agnostic)
# -----------------------------------------
@dataclass(frozen=True)
class EngineConfig:
    # optional
    preds_dirname: Optional[str]
    context_dirname: Optional[str]
    context_max_side_cap: Optional[int]

    # discovery
    pan_dirname: str = "PAN"
    gt_dirname: str = "GT"

    # tiling geometry
    target_size_m: float = 250.0
    overlap: float = 0.5
    reflectance_max: int = 10_000

    # shared behavior toggles
    verbose: bool = False
    skip_tiles_with_nodata: bool = True


# -----------------------------------------
# Unified ROI tiling engine (task-agnostic)
# -----------------------------------------
class RoiTilingEngine:
    """
    Engine responsibilities ONLY:
      - discover scene files (PAN/gt/pred/context)
      - load arrays (PAN + optional gt/pred)
      - compute scene metadata (H/W/GSD) and nodata mask
      - generate tile windows and apply nodata skip
      - emit core CSV row fields
      - delegate inclusion to policy, and task columns to adapter

    All task semantics live in:
      - adapters/ (presence/difficulty/task columns, etc.)
      - policies.py (All/Regular/HardMining)
    """

    def __init__(self, *, cfg: EngineConfig, adapter: TaskAdapter, policy: TilingPolicy):
        self.cfg = cfg
        self.adapter = adapter
        self.policy = policy
        self._overlap = float(np.clip(cfg.overlap, 0.0, 0.95))

    # -----------------------------
    # Public scanning API
    # -----------------------------
    def scan_subdir(self, subdir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        stats: Dict[str, Any] = dict(
            region=subdir.parent.name,
            subregion=subdir.name,
            roi_pred_missing=False,
            roi_context_missing=False,
            scenes_processed=0,
            scenes_skipped_no_pan=0,
            scenes_skipped_no_context=0,
            scenes_read_error=0,
            tiles_considered=0,
            tiles_included=0,
            tiles_skipped=0,
            tiles_skipped_nodata=0,
        )

        # -----------------------------
        # Discover directories
        # -----------------------------
        pan_dir = subdir / self.cfg.pan_dirname
        if not pan_dir.is_dir():
            print(f"Scene {subdir} skipped: pan_dir not found at {pan_dir}")
            return [], stats

        gt_dir = subdir / self.cfg.gt_dirname
        if self.adapter.require_gt_dir() and not gt_dir.is_dir():
            print(f"Scene {subdir} skipped: gt_dir required but not found at {gt_dir}")
            return [], stats

        ctx_dir = subdir / self.cfg.context_dirname
        if self.adapter.require_context_dir() and not ctx_dir.is_dir():
            print(f"Scene {subdir} skipped: context_dir required but not found at {ctx_dir}")
            stats["roi_context_missing"] = True
            return [], stats

        pred_dir = subdir / self.cfg.preds_dirname if self.cfg.preds_dirname else None

        # -----------------------------
        # Build stem->path maps
        # -----------------------------
        pan_map: Dict[str, Path] = {p.stem: p for p in pan_dir.glob("*.tif")}
        gt_map: Dict[str, Path] = {p.stem: p for p in gt_dir.glob("*.tif")} if gt_dir.is_dir() else {}
        ctx_map: Dict[str, Path] = {p.stem: p for p in ctx_dir.glob("*.tif")} if ctx_dir.is_dir() else {}
        pred_map: Dict[str, Path] = (
            {p.stem: p for p in pred_dir.glob("*.tif")} if pred_dir and pred_dir.is_dir() else {}
        )

        if self.adapter.require_nonempty_gt_map() and not gt_map:
            return [], stats

        if self.adapter.require_nonempty_context_map() and not ctx_map:
            stats["roi_context_missing"] = True
            return [], stats

        roi_pred_missing = (self.cfg.preds_dirname is not None) and (not (pred_dir and pred_dir.is_dir()))
        stats["roi_pred_missing"] = bool(roi_pred_missing)

        # -----------------------------
        # stems processing
        # -----------------------------
        stems = self.adapter.stems_to_process(pan_map=pan_map, gt_map=gt_map)

        rows: List[Dict[str, Any]] = []

        for stem in stems:
            pan_path = _relaxed_lookup(stem, pan_map)
            if pan_path is None:
                stats["scenes_skipped_no_pan"] += 1
                continue

            gt_path = _relaxed_lookup(stem, gt_map) if gt_map else None
            pred_path = _relaxed_lookup(stem, pred_map) if pred_map else None
            ctx_path = _relaxed_lookup(stem, ctx_map) if ctx_map else None

            if ctx_path is None and not self.adapter.allow_missing_context_per_scene():
                stats["scenes_skipped_no_context"] += 1
                continue

            # -----------------------------
            # Read PAN + compute GSD
            # -----------------------------
            try:
                with rasterio.open(pan_path) as src:
                    pan_2d = src.read(1)
                    # print("crs:", src.crs)
                    # print("transform:", src.transform)
                    # print("bounds:", src.bounds)
                    # meta = src.meta.copy()
                    # print(meta)
                H, W = pan_2d.shape
                # gsd_mpp = float(compute_gsd_from_gcps(meta.gcps[0]))
                gsd_mpp = gsd_from_epsg4326(pan_path)
            except (FileNotFoundError, OSError):
                stats["scenes_read_error"] += 1
                continue

            # -----------------------------
            # Read gt (or fabricate zeros)
            # -----------------------------
            if gt_path is not None:
                with rasterio.open(gt_path) as src:
                    gt2d = src.read()
                    gt2d = gt2d.astype(np.int64)
            else:
                if self.adapter.allow_fabricated_zero_gt():
                    gt2d = np.zeros((H, W), dtype=np.int64)
                else:
                    gt2d = None

            # -----------------------------
            # Read pred (optional)
            # -----------------------------
            if pred_path is not None:
                with rasterio.open(pred_path) as src:
                    pred2d = src.read()
                    pred2d = pred2d.astype(np.int64)
            else:
                pred2d = None

            scene_id = f"{subdir.parent.name}/{subdir.name}/{pan_path.stem}"
            scene = SceneInputs(
                region=subdir.parent.name,
                subregion=subdir.name,
                stem=stem,
                pan_path=pan_path,
                gt_path=gt_path,
                pred_path=pred_path,
                context_path=ctx_path,
                scene_id=scene_id,
            )

            arr = SceneArrays(
                H=int(H),
                W=int(W),
                gsd_mpp=float(gsd_mpp),
                gt2d=gt2d,
                pred2d=pred2d,
            )

            # Adapter may populate derived layers (optional)
            self.adapter.build_convenience_layers(arr)

            # -----------------------------
            # Tiling params
            # -----------------------------
            tile_px = max(8, int(round(self.cfg.target_size_m / max(1e-6, arr.gsd_mpp))))
            stride = max(1, int(round(tile_px * (1.0 - self._overlap))))
            th = tw = tile_px
            sh = sw = stride

            stats["scenes_processed"] += 1
            tile_idx = 0

            # -----------------------------
            # Tile loop
            # -----------------------------
            for x0, y0, x1, y1, r, c in gen_tiles_cover(int(H), int(W), th, tw, sh, sw):
                tot = int((y1 - y0) * (x1 - x0))
                stats["tiles_considered"] += 1

                win = TileWindow(
                    x0=int(x0),
                    y0=int(y0),
                    x1=int(x1),
                    y1=int(y1),
                    r=int(r),
                    c=int(c),
                    tile_idx=int(tile_idx),
                    tot=int(tot),
                )

                # shared nodata skip
                if self.cfg.skip_tiles_with_nodata and arr.pan_mask is not None:
                    if arr.pan_mask[win.y0 : win.y1, win.x0 : win.x1].any():
                        stats["tiles_skipped_nodata"] += 1
                        stats["tiles_skipped"] += 1
                        tile_idx += 1
                        continue

                include, extra = self.policy.decide_include(
                    adapter=self.adapter,
                    scene=scene,
                    arr=arr,
                    tw=win,
                    roi_pred_missing=roi_pred_missing,
                )
                if not include:
                    stats["tiles_skipped"] += 1
                    tile_idx += 1
                    continue

                # -----------------------------
                # Core row fields (engine-owned)
                # -----------------------------
                cx = 0.5 * (win.x0 + win.x1)
                cy = 0.5 * (win.y0 + win.y1)

                row: Dict[str, Any] = dict(
                    region=scene.region,
                    subregion=scene.subregion,
                    scene_id=scene.scene_id,
                    stem=scene.stem,
                    image_src=str(scene.pan_path),
                    gt_src=str(scene.gt_path) if scene.gt_path else "",
                    pred_src=str(scene.pred_path) if scene.pred_path else "",
                    context_src=str(scene.context_path) if scene.context_path else "",
                    x0=win.x0,
                    y0=win.y0,
                    x1=win.x1,
                    y1=win.y1,
                    tile_w_px=int(win.x1 - win.x0),
                    tile_h_px=int(win.y1 - win.y0),
                    gsd_mpp=float(arr.gsd_mpp),
                    scene_h=int(arr.H),
                    scene_w=int(arr.W),
                    tile_row=int(win.r),
                    tile_col=int(win.c),
                    tile_idx=int(win.tile_idx),
                    tile_cx_norm=float(cx / max(1, arr.W)),
                    tile_cy_norm=float(cy / max(1, arr.H)),
                    tile_size_px=int(tile_px),
                    stride_px=int(stride),
                    overlap=float(self._overlap),
                )

                # Policy schema defaults + policy extra + task columns
                row.update(self.policy.extra_row_fields())
                row.update(extra)
                row.update(self.adapter.build_task_row(scene=scene, arr=arr, tw=win))

                rows.append(row)
                stats["tiles_included"] += 1
                tile_idx += 1

        return rows, stats
