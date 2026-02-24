from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from cvrs.machine_learning.core.tiling.adapters.base import (
    BaseAdapter,
    DifficultyResult,
    PresenceResult,
    SceneArrays,
    SceneInputs,
    TileWindow,
)


@dataclass
class SegmentationAdapter(BaseAdapter):
    """
    Helper base for segmentation-style tasks where:
      - arr.gt2d and arr.pred2d are integer masks aligned to PAN
      - "presence" is measured as foreground pixel ratio
      - "difficulty" is measured from gt vs pred disagreement

    This class is intentionally generic and reusable across segmentation tasks.

    Typical usage:
      class WaterAdapter(SegmentationAdapter):
          def build_task_row(...): ...
          # optionally override fg_from_mask() for multi-class / remaps
          # optionally override difficulty() if you want a different metric

    Key extension hooks:
      - fg_from_mask(mask2d) -> bool mask of class(es) of interest
      - build_task_row(...)  -> task-specific CSV columns
    """

    # -----------------------------
    # Foreground definition
    # -----------------------------
    class_of_interest_id: int = 1  # default matches current conventions

    def fg_from_mask(self, mask2d: np.ndarray) -> np.ndarray:
        """
        Return a boolean foreground mask for the class(es) of interest.

        Override this for:
          - multi-class interest: return np.isin(mask2d, ids)
          - remaps: handle ignore labels, etc.
        """
        return mask2d == int(self.class_of_interest_id)

    # -----------------------------
    # Optional derived layers
    # -----------------------------
    def build_convenience_layers(self, arr: SceneArrays) -> None:
        # Populate target_gt / target_pred using fg_from_mask
        if arr.gt2d is not None:
            arr.target_gt = self.fg_from_mask(arr.gt2d)
        if arr.pred2d is not None:
            arr.target_pred = self.fg_from_mask(arr.pred2d)

    # -----------------------------
    # Tile-level metrics for universal policies
    # -----------------------------
    def gt_presence(self, *, scene: SceneInputs, arr: SceneArrays, tw: TileWindow) -> PresenceResult:
        """
        Presence = foreground ratio in gt (0..1).
        """
        _ = scene  # not used by default; kept for overrides

        if arr.gt2d is None and arr.target_gt is None:
            return PresenceResult(value=0.0, details={"gt_missing": True})

        if arr.target_gt is None:
            # compute on the fly if convenience layers weren't built
            chip = arr.gt2d[tw.y0 : tw.y1, tw.x0 : tw.x1]
            w = self.fg_from_mask(chip)
        else:
            w = arr.target_gt[tw.y0 : tw.y1, tw.x0 : tw.x1]

        fg = int(w.sum())
        tot = int(w.size) if w.size > 0 else int(tw.tot)
        ratio = float(fg / max(1, tot))

        return PresenceResult(
            value=ratio,
            details={
                "gt_fg_px": fg,
                "gt_tot_px": tot,
                "gt_fg_ratio": ratio,
            },
        )

    def difficulty(self, *, scene: SceneInputs, arr: SceneArrays, tw: TileWindow) -> DifficultyResult:
        """
        Default disagreement metric:
          - change_ratio = mean(pred_fg != gt_fg) over the tile

        This is intentionally simple + generic.
        Override for:
          - IoU-based hardness
          - FN-heavy mining
          - multi-class confusion breakdowns
        """
        _ = scene  # not used by default

        if arr.pred2d is None and arr.target_pred is None:
            return DifficultyResult(value=0.0, details={"pred_missing": True})

        if arr.gt2d is None and arr.target_gt is None:
            return DifficultyResult(value=0.0, details={"gt_missing": True})

        # Get boolean fg chips
        if arr.target_gt is None:
            gt_chip = arr.gt2d[tw.y0 : tw.y1, tw.x0 : tw.x1]
            gt_fg = self.fg_from_mask(gt_chip)
        else:
            gt_fg = arr.target_gt[tw.y0 : tw.y1, tw.x0 : tw.x1]

        if arr.target_pred is None:
            pred_chip = arr.pred2d[tw.y0 : tw.y1, tw.x0 : tw.x1]
            pred_fg = self.fg_from_mask(pred_chip)
        else:
            pred_fg = arr.target_pred[tw.y0 : tw.y1, tw.x0 : tw.x1]

        # Disagreement
        diff = pred_fg ^ gt_fg
        change_pixels = int(diff.sum())
        tot = int(diff.size) if diff.size > 0 else int(tw.tot)
        change_ratio = float(change_pixels / max(1, tot))

        # Optional extra confusion-like counts (binary)
        tp = int((pred_fg & gt_fg).sum())
        fp = int((pred_fg & ~gt_fg).sum())
        fn = int((~pred_fg & gt_fg).sum())
        tn = int((~pred_fg & ~gt_fg).sum())

        return DifficultyResult(
            value=change_ratio,
            details={
                "change_pixels": change_pixels,
                "change_ratio": change_ratio,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            },
        )

    # -----------------------------
    # Task-specific CSV columns
    # -----------------------------
    def build_task_row(self, *, scene: SceneInputs, arr: SceneArrays, tw: TileWindow) -> Dict[str, Any]:
        """
        Base class does not impose any task columns.
        Concrete task adapters should override this.

        Example for water:
          pres = self.gt_presence(...)
          return {"water_ratio": pres.value}
        """
        raise NotImplementedError(
            "SegmentationAdapter is a helper base. Create a concrete task adapter and implement build_task_row()."
        )
