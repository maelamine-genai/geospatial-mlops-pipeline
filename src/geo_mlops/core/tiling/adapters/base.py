from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


# -----------------------------
# Shared data bundles (engine + adapters + policies)
# -----------------------------
@dataclass(frozen=True)
class SceneInputs:
    region: str
    subregion: str
    stem: str

    pan_path: Path
    gt_path: Optional[Path]
    pred_path: Optional[Path]
    context_path: Optional[Path]

    scene_id: str


@dataclass
class SceneArrays:
    H: int
    W: int
    gsd_mpp: float

    gt2d: Optional[np.ndarray]
    pred2d: Optional[np.ndarray]

    # optional nodata mask derived from PAN masked array
    pan_mask: Optional[np.ndarray] = None

    # optional derived layers (adapter may populate)
    target_gt: Optional[np.ndarray] = None
    target_pred: Optional[np.ndarray] = None


@dataclass(frozen=True)
class TileWindow:
    x0: int
    y0: int
    x1: int
    y1: int
    r: int
    c: int
    tile_idx: int
    tot: int


# -----------------------------
# Adapter-facing generic results
# -----------------------------
@dataclass(frozen=True)
class PresenceResult:
    """
    Normalized measure of GT presence for a tile.
    Examples:
      - segmentation: water_ratio (0..1)
      - classification: non_clear_ratio (0..1) or boolean-as-float (0/1)
    """

    value: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DifficultyResult:
    """
    Normalized measure of model difficulty / disagreement for a tile.
    Examples:
      - segmentation: change_ratio, fp_ratio, fn_ratio, etc.
      - classification: mismatch (0/1), margin, entropy, etc.
    """

    value: float
    details: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Core interfaces (policies depend on these)
# -----------------------------
class TaskAdapter(Protocol):
    """
    Task-specific adapter used by universal tiling policies.
    Engine owns: traversal + IO + tiling + core row schema.
    Adapter owns:
      - directory requirements & stems selection
      - optional derived layers (target_gt/target_pred or others)
      - presence + difficulty metrics consumed by universal policies
      - task-specific CSV columns
    """

    # -----------------------
    # Scene-level behavior
    # -----------------------
    def require_gt_dir(self) -> bool: ...
    def require_nonempty_gt_map(self) -> bool: ...
    def allow_fabricated_zero_gt(self) -> bool: ...

    def require_context_dir(self) -> bool: ...
    def require_nonempty_context_map(self) -> bool: ...
    def allow_missing_context_per_scene(self) -> bool: ...

    def stems_to_process(self, *, pan_map: Dict[str, Path], gt_map: Dict[str, Path]) -> List[str]: ...

    # -----------------------
    # Mask interpretation (optional)
    # -----------------------
    def build_convenience_layers(self, arr: SceneArrays) -> None: ...

    # -----------------------
    # Tile-level metrics for universal policies
    # -----------------------
    def gt_presence(self, *, scene: SceneInputs, arr: SceneArrays, tw: TileWindow) -> PresenceResult: ...
    def difficulty(self, *, scene: SceneInputs, arr: SceneArrays, tw: TileWindow) -> DifficultyResult: ...

    # -----------------------
    # CSV task columns
    # -----------------------
    def build_task_row(self, *, scene: SceneInputs, arr: SceneArrays, tw: TileWindow) -> Dict[str, Any]: ...


class TilingPolicy(Protocol):
    """
    Universal include policy interface.
    Policies should be task-agnostic and rely on adapter.gt_presence / adapter.difficulty.
    """

    def extra_row_fields(self) -> Dict[str, Any]: ...

    def decide_include(
        self,
        *,
        adapter: TaskAdapter,
        scene: SceneInputs,
        arr: SceneArrays,
        tw: TileWindow,
        roi_pred_missing: bool,
    ) -> tuple[bool, Dict[str, Any]]: ...


# --------------------------------
# Small optional convenience base
# --------------------------------
class BaseAdapter:
    """
    Minimal convenience base with safe defaults.

    IMPORTANT: This is intentionally NOT "SegmentationAdapter" or "ClassificationAdapter".
    Those belong in adapters/segmentation.py and adapters/classification.py.

    Defaults:
      - gt dir is optional
      - stems come from gt_map if present, else pan_map
      - fabricate zeros when gt missing
      - context required by default (matches your current engine behavior),
        but tasks can override.
    """

    # ---- scene-level defaults ----
    def require_gt_dir(self) -> bool:
        return False

    def require_nonempty_gt_map(self) -> bool:
        return False

    def allow_fabricated_zero_gt(self) -> bool:
        return True

    def require_context_dir(self) -> bool:
        return True

    def require_nonempty_context_map(self) -> bool:
        return True

    def allow_missing_context_per_scene(self) -> bool:
        return False

    def stems_to_process(self, *, pan_map: Dict[str, Path], gt_map: Dict[str, Path]) -> List[str]:
        return sorted(set(gt_map.keys()) or set(pan_map.keys()))

    # ---- derived layers (default assumes class-of-interest == 1) ----
    def build_convenience_layers(self, arr: SceneArrays) -> None:
        if arr.gt2d is not None:
            arr.target_gt = arr.gt2d == 1
        if arr.pred2d is not None:
            arr.target_pred = arr.pred2d == 1
