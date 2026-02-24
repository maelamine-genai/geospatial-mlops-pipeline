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
class ClassificationAdapter(BaseAdapter):
    """
    Helper base for classification-style tasks.
    Intended patterns:
      - We may still have arr.gt2d as a raster mask (labels per pixel)
        and reduce a tile window to a single label.

    How this differs from SegmentationAdapter:
      - It reduces a chip to a *single label* (class_id)
      - Presence is usually boolean-as-float or a ratio of "non-clear" pixels,
        depending on our use case.
      - Difficulty is typically 0/1 mismatch (or could be entropy/margin if
        we later feed logits/probs to the adapter).

    Concrete tasks should override:
      - reduce_gt_to_label(...)
      - build_task_row(...)
      - optionally difficulty(...) for fancier mining

    If the task needs gt always, enforce it at the adapter layer
    This keeps the engine generic
    """

    def require_gt_dir(self) -> bool:
        return True

    def require_nonempty_gt_map(self) -> bool:
        return True

    def allow_fabricated_zero_gt(self) -> bool:
        return False

    def stems_to_process(self, *, pan_map: Dict[str, Any], gt_map: Dict[str, Any]) -> list[str]:
        # classification usually uses only labeled scenes
        return sorted(set(gt_map.keys()))

    # -----------------------------
    # Label reduction
    # -----------------------------
    def reduce_gt_to_label(self, gt_chip: np.ndarray) -> int:
        """
        Reduce an HxW GT raster chip to a single integer class label.
        Override this with your project’s rule, e.g.:
          - majority vote
          - "any non-clear => non-clear"
          - custom chip_label_from_mask(mask_chip)

        Must return an int class_id.
        """
        raise NotImplementedError

    def label_from_pred(self, pred_chip: np.ndarray) -> int:
        """
        Reduce an HxW prediction raster chip to a single integer label.
        Default mirrors gt reduction.
        Override if prediction representation differs.
        """
        return self.reduce_gt_to_label(pred_chip)

    # -----------------------------
    # Optional derived layers
    # -----------------------------
    def build_convenience_layers(self, arr: SceneArrays) -> None:
        """
        For classification, we usually don't need w_gt/w_pred.
        Keep empty by default to avoid implying "class-of-interest == 1".
        """
        return

    # -----------------------------
    # Tile-level metrics for universal policies
    # -----------------------------
    def gt_presence(self, *, scene: SceneInputs, arr: SceneArrays, tw: TileWindow) -> PresenceResult:
        """
        Default presence signal for classification: 1.0 if label != 0 else 0.0.

        This is a useful universal default for things like:
          - "include non-clear tiles"
          - "include positive class tiles"
        where 0 is treated as background/clear.

        Override if you want:
          - non-clear pixel ratio
          - presence based on a specific class_id set
        """
        _ = scene

        if arr.gt2d is None:
            return PresenceResult(value=0.0, details={"gt_missing": True})

        gt_chip = arr.gt2d[tw.y0 : tw.y1, tw.x0 : tw.x1]
        label = int(self.reduce_gt_to_label(gt_chip))
        present = 0.0 if label == 0 else 1.0

        return PresenceResult(
            value=float(present),
            details={
                "gt_label": label,
                "gt_present": int(present),
            },
        )

    def difficulty(self, *, scene: SceneInputs, arr: SceneArrays, tw: TileWindow) -> DifficultyResult:
        """
        Default difficulty for classification: mismatch (0/1) between reduced labels.
        This works well for hard-example mining where we have per-pixel predicted labels
        (or predicted masks), but we can override to use probs/logits later.
        """
        _ = scene

        if arr.gt2d is None:
            return DifficultyResult(value=0.0, details={"gt_missing": True})

        if arr.pred2d is None:
            return DifficultyResult(value=0.0, details={"pred_missing": True})

        gt_chip = arr.gt2d[tw.y0 : tw.y1, tw.x0 : tw.x1]
        pred_chip = arr.pred2d[tw.y0 : tw.y1, tw.x0 : tw.x1]

        gt_label = int(self.reduce_gt_to_label(gt_chip))
        pred_label = int(self.label_from_pred(pred_chip))

        mismatch = 1.0 if pred_label != gt_label else 0.0

        return DifficultyResult(
            value=float(mismatch),
            details={
                "gt_label": gt_label,
                "pred_label": pred_label,
                "mismatch": int(mismatch),
            },
        )

    # -----------------------------
    # Task-specific CSV columns
    # -----------------------------
    def build_task_row(self, *, scene: SceneInputs, arr: SceneArrays, tw: TileWindow) -> Dict[str, Any]:
        """
        Base class does not impose any task columns.
        Concrete task adapters should override this.

        Common output for classification tiling CSV:
          - gt_label
          - maybe pred_label (if present)
        """
        raise NotImplementedError(
            "ClassificationAdapter is a helper base. Create a concrete task adapter and implement build_task_row()."
        )
