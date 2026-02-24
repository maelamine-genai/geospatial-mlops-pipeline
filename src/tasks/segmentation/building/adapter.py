from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from geo_mlops.core.tiling.adapters.base import (
    DifficultyResult,
    SceneArrays,
    SceneInputs,
    TileWindow,
)
from geo_mlops.tasks.segmentation.tiling_adapter import SegmentationAdapter


@dataclass
class BuildingSegmentationAdapter(SegmentationAdapter):
    """
    Building segmentation adapter.

    Uses SegmentationAdapter's generic:
      - build_convenience_layers(): target_gt/target_pred from fg_from_mask()
      - gt_presence(): foreground ratio in GT

    Adds:
      - building-specific task columns (building_ratio, building_pixels, total_pixels, optional shadow_*)
      - building-specific difficulty details (change_pixels/ratio, add/rem breakdown)
      - optional min_change_pixels gate baked into DifficultyResult.value
    """

    class_of_interest_id: int = 1
    shadow_id: Optional[int] = 2
    emit_shadow: bool = True

    # Gate for "hardness" to mimic old (min_change_pixels AND min_change_ratio) logic
    # Used inside difficulty(): if change_pixels < min_change_pixels => value=0.0
    min_change_pixels: int = 1

    def fg_from_mask(self, mask2d: np.ndarray) -> np.ndarray:
        return mask2d == int(self.class_of_interest_id)

    def difficulty(self, *, scene: SceneInputs, arr: SceneArrays, tw: TileWindow) -> DifficultyResult:
        _ = scene

        # Keep adapter resilient; strict hard-mining policy will raise before this if preds are required.
        if arr.pred2d is None and arr.target_pred is None:
            return DifficultyResult(value=0.0, details={"pred_missing": True, "pred_available": 0})

        if arr.gt2d is None and arr.target_gt is None:
            return DifficultyResult(value=0.0, details={"gt_missing": True})

        # Prefer precomputed boolean layers
        if arr.target_pred is not None:
            pred_fg = arr.target_pred[tw.y0 : tw.y1, tw.x0 : tw.x1]
        else:
            pred_chip = arr.pred2d[tw.y0 : tw.y1, tw.x0 : tw.x1]
            pred_fg = self.fg_from_mask(pred_chip)

        if arr.target_gt is not None:
            gt_fg = arr.target_gt[tw.y0 : tw.y1, tw.x0 : tw.x1]
        else:
            gt_chip = arr.gt2d[tw.y0 : tw.y1, tw.x0 : tw.x1]
            gt_fg = self.fg_from_mask(gt_chip)

        tot = int(pred_fg.size) if pred_fg.size > 0 else int(tw.tot)

        # Disagreement
        diff = np.logical_xor(pred_fg, gt_fg)
        change_pixels = int(diff.sum())
        change_ratio = float(change_pixels / max(1, tot))

        # Directional diagnostics (naming kept from previous building policy)
        building_add_pixels = int((~pred_fg & gt_fg).sum())  # FN-ish (missed GT building)
        building_rem_pixels = int((pred_fg & ~gt_fg).sum())  # FP-ish (extra pred building)

        details = dict(
            pred_available=1,
            change_pixels=change_pixels,
            change_ratio=change_ratio,
            building_add_pixels=building_add_pixels,
            building_add_ratio=float(building_add_pixels / max(1, tot)),
            building_rem_pixels=building_rem_pixels,
            building_rem_ratio=float(building_rem_pixels / max(1, tot)),
        )

        # Gate: enforce min_change_pixels without a building-specific policy
        value = change_ratio if change_pixels >= int(self.min_change_pixels) else 0.0

        return DifficultyResult(value=value, details=details)

    def build_task_row(self, *, scene: SceneInputs, arr: SceneArrays, tw: TileWindow) -> Dict[str, Any]:
        _ = scene

        # Use base presence (foreground ratio); we only rename columns for building
        pres = super().gt_presence(scene=scene, arr=arr, tw=tw)
        # base keys: gt_fg_px, gt_tot_px, gt_fg_ratio
        d = pres.details or {}

        row: Dict[str, Any] = {
            "building_ratio": float(pres.value),
            "building_pixels": int(d.get("gt_fg_px", 0)),
            "total_pixels": int(d.get("gt_tot_px", tw.tot)),
        }

        if self.emit_shadow and self.shadow_id is not None and arr.gt2d is not None:
            chip = arr.gt2d[tw.y0 : tw.y1, tw.x0 : tw.x1]
            tot = int(chip.size) if chip.size > 0 else int(tw.tot)
            shadow_px = int((chip == int(self.shadow_id)).sum())
            row["shadow_pixels"] = shadow_px
            row["shadow_ratio"] = float(shadow_px / max(1, tot))

        return row
