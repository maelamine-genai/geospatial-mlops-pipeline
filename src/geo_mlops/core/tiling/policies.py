from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from geo_mlops.core.tiling.adapters.base import (
    SceneArrays,
    SceneInputs,
    TaskAdapter,
    TileWindow,
)


# -------------------------
# AllPolicy
# -------------------------
@dataclass
class AllPolicy:
    """
    Annotate every valid tile as selectable.

    Important:
      - The tiling engine may still skip invalid/nodata tiles.
      - This policy never filters valid tiles out of the master CSV.
    """

    sample_prefix: str = "sample__"

    def extra_row_fields(self) -> Dict[str, Any]:
        return {
            f"{self.sample_prefix}include": True,
            f"{self.sample_prefix}policy": "all",
            f"{self.sample_prefix}reason": "all",
        }

    def decide_include(
        self,
        *,
        adapter: TaskAdapter,
        scene: SceneInputs,
        arr: SceneArrays,
        tw: TileWindow,
        roi_pred_missing: bool,
    ) -> Tuple[bool, Dict[str, Any]]:
        _ = (adapter, scene, arr, tw, roi_pred_missing)

        extra = {
            f"{self.sample_prefix}include": True,
            f"{self.sample_prefix}policy": "all",
            f"{self.sample_prefix}reason": "all",
        }

        # Always emit valid tiles.
        return True, extra


# -------------------------
# RegularPolicy
# -------------------------
@dataclass
class RegularPolicy:
    """
    GT-presence scoring policy.

    This no longer filters tiles during tiling.

    It annotates each tile with:
      - presence__* fields
      - sample__include: whether this tile should be used by train-time sampling
      - sample__policy: regular
      - sample__reason

    The validation/test partitions should ignore sample__include and use all valid rows.
    """

    gt_presence_threshold: float = 1e-6
    require_presence: bool = True

    details_prefix: str = "presence__"
    sample_prefix: str = "sample__"

    def extra_row_fields(self) -> Dict[str, Any]:
        return {
            f"{self.details_prefix}value": 0.0,
            f"{self.sample_prefix}include": False,
            f"{self.sample_prefix}policy": "regular",
            f"{self.sample_prefix}reason": "",
        }

    def decide_include(
        self,
        *,
        adapter: TaskAdapter,
        scene: SceneInputs,
        arr: SceneArrays,
        tw: TileWindow,
        roi_pred_missing: bool,
    ) -> Tuple[bool, Dict[str, Any]]:
        _ = roi_pred_missing

        pres = adapter.gt_presence(scene=scene, arr=arr, tw=tw)

        presence_value = float(pres.value)
        sample_include = (
            True
            if not self.require_presence
            else presence_value >= float(self.gt_presence_threshold)
        )

        extra: Dict[str, Any] = {}

        if pres.details:
            if self.details_prefix:
                extra.update(
                    {f"{self.details_prefix}{k}": v for k, v in pres.details.items()}
                )
            else:
                extra.update(pres.details)

        extra.setdefault(f"{self.details_prefix}value", presence_value)

        extra[f"{self.sample_prefix}include"] = bool(sample_include)
        extra[f"{self.sample_prefix}policy"] = "regular"
        extra[f"{self.sample_prefix}reason"] = (
            "presence_pass"
            if sample_include
            else "presence_below_threshold"
        )

        # Always emit valid tiles into master CSV.
        return True, extra


# -------------------------
# HardMiningPolicy
# -------------------------
@dataclass
class HardMiningPolicy:
    """
    Hard-mining scoring policy.

    This no longer filters tiles during tiling.

    It annotates each tile with:
      - presence__* fields
      - difficulty__* fields
      - sample__include: whether this tile should be used by train-time sampling
      - sample__policy: hard_mining
      - sample__reason

    The validation/test partitions should ignore sample__include and use all valid rows.
    """

    min_difficulty: float = 1e-6

    include_if_gt_present: bool = True
    gt_presence_threshold: float = 1e-6

    presence_prefix: str = "presence__"
    difficulty_prefix: str = "difficulty__"
    sample_prefix: str = "sample__"

    def extra_row_fields(self) -> Dict[str, Any]:
        return {
            f"{self.presence_prefix}value": 0.0,
            f"{self.difficulty_prefix}value": 0.0,
            f"{self.sample_prefix}include": False,
            f"{self.sample_prefix}policy": "hard_mining",
            f"{self.sample_prefix}reason": "",
        }

    def decide_include(
        self,
        *,
        adapter: TaskAdapter,
        scene: SceneInputs,
        arr: SceneArrays,
        tw: TileWindow,
        roi_pred_missing: bool,
    ) -> Tuple[bool, Dict[str, Any]]:
        if roi_pred_missing:
            raise FileNotFoundError(
                "HardMiningPolicy requires predictions, but preds directory is missing for ROI: "
                f"{scene.region}/{scene.subregion}"
            )

        if arr.pred2d is None:
            raise ValueError(
                "HardMiningPolicy requires predictions, but pred2d is None for scene/tile: "
                f"{scene.region}/{scene.subregion}"
            )

        extra: Dict[str, Any] = {}

        pres = adapter.gt_presence(scene=scene, arr=arr, tw=tw)
        presence_value = float(pres.value)
        extra.update(
            self._pack_details(
                presence_value,
                pres.details,
                prefix=self.presence_prefix,
            )
        )

        diff = adapter.difficulty(scene=scene, arr=arr, tw=tw)
        difficulty_value = float(diff.value)
        extra.update(
            self._pack_details(
                difficulty_value,
                diff.details,
                prefix=self.difficulty_prefix,
            )
        )

        presence_pass = presence_value >= float(self.gt_presence_threshold)
        difficulty_pass = difficulty_value >= float(self.min_difficulty)

        sample_include = difficulty_pass or (
            bool(self.include_if_gt_present) and presence_pass
        )

        if difficulty_pass:
            reason = "difficulty_pass"
        elif bool(self.include_if_gt_present) and presence_pass:
            reason = "presence_pass"
        else:
            reason = "not_hard_and_no_presence"

        extra[f"{self.sample_prefix}include"] = bool(sample_include)
        extra[f"{self.sample_prefix}policy"] = "hard_mining"
        extra[f"{self.sample_prefix}reason"] = reason

        # Always emit valid tiles into master CSV.
        return True, extra

    @staticmethod
    def _pack_details(
        value: float,
        details: Dict[str, Any] | None,
        *,
        prefix: str,
    ) -> Dict[str, Any]:
        key = f"{prefix}value" if prefix else "value"
        out: Dict[str, Any] = {key: float(value)}

        if details:
            if prefix:
                out.update({f"{prefix}{k}": v for k, v in details.items()})
            else:
                out.update(details)

        return out
    