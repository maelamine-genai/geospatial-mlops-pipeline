from __future__ import annotations

from typing import Any, Dict

from geo_mlops.core.tiling.engine import EngineConfig
from geo_mlops.core.tiling.policies import AllPolicy, HardMiningPolicy, RegularPolicy
from cli.cfg_loader import load_cfg
from geo_mlops.tasks.segmentation.building.adapter import BuildingSegmentationAdapter


def _build_engine_cfg(engine_cfg: Dict[str, Any]) -> EngineConfig:
    return EngineConfig(
        pan_dirname=engine_cfg.get("pan_dirname"),
        gt_dirname=engine_cfg.get("gt_dirname"),
        preds_dirname=engine_cfg.get("preds_dirname"),
        context_dirname=engine_cfg.get("context_dirname"),
        target_size_m=float(engine_cfg.get("target_size_m", 250.0)),
        overlap=float(engine_cfg.get("overlap", 0.5)),
        reflectance_max=int(engine_cfg.get("reflectance_max", 10_000)),
        context_max_side_cap=engine_cfg.get("context_max_side_cap", None),
        verbose=bool(engine_cfg.get("verbose", False)),
        skip_tiles_with_nodata=bool(engine_cfg.get("skip_tiles_with_nodata", True)),
    )


def _build_adapter(adapter_cfg: Dict[str, Any]) -> BuildingSegmentationAdapter:
    return BuildingSegmentationAdapter(
        class_of_interest_id=int(adapter_cfg.get("class_of_interest_id", 1)),
        shadow_id=adapter_cfg.get("shadow_id", 2),
        emit_shadow=bool(adapter_cfg.get("emit_shadow", True)),
        min_change_pixels=int(adapter_cfg.get("min_change_pixels", 1)),
    )


def _build_policy(policy_cfg: Dict[str, Any]) -> AllPolicy | RegularPolicy | HardMiningPolicy:
    kind = str(policy_cfg.get("kind", "all")).lower().strip()

    if kind == "all":
        return AllPolicy()

    if kind == "regular":
        return RegularPolicy(
            gt_presence_threshold=float(policy_cfg.get("gt_presence_threshold", 1e-6)),
            require_presence=bool(policy_cfg.get("require_presence", True)),
            details_prefix=str(policy_cfg.get("details_prefix", "presence__")),
        )

    if kind == "hard_mining":
        return HardMiningPolicy(
            min_difficulty=float(policy_cfg["min_difficulty"]),  # required
            include_if_gt_present=bool(policy_cfg.get("include_if_gt_present", True)),
            gt_presence_threshold=float(policy_cfg.get("gt_presence_threshold", 1e-6)),
            presence_prefix=str(policy_cfg.get("presence_prefix", "presence__")),
            difficulty_prefix=str(policy_cfg.get("difficulty_prefix", "difficulty__")),
        )

    raise ValueError(f"Unknown policy kind={kind!r}. Expected one of: all, regular, hard_mining")


def build_from_cfg(
    task_cfg_path: str,
) -> tuple[
    EngineConfig,
    BuildingSegmentationAdapter,
    AllPolicy | RegularPolicy | HardMiningPolicy,
    dict[Any, Any] | dict[Any, Any],
]:
    """
    Entry point referenced by task_registry.

    Reads YAML/JSON config and returns:
      - EngineConfig
      - BuildingSegmentationAdapter (TaskAdapter)
      - Policy (All/Regular/HardMiningPolicy)
      - meta dict
    """
    cfg = load_cfg(task_cfg_path)

    engine_cfg = cfg.get("engine", {}) or {}
    adapter_cfg = cfg.get("adapter", {}) or {}
    policy_cfg = cfg.get("policy", {}) or {}
    meta = cfg.get("meta", {}) or {}

    engine = _build_engine_cfg(engine_cfg)
    adapter = _build_adapter(adapter_cfg)
    policy = _build_policy(policy_cfg)

    return engine, adapter, policy, meta
