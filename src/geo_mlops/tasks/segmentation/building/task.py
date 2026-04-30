from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import torch

from geo_mlops.core.config.loader import load_cfg
from geo_mlops.core.contracts.split_contract import SplitContract
from geo_mlops.core.contracts.tile_contract import TilesContract
from geo_mlops.core.tiling.engine import EngineConfig
from geo_mlops.core.tiling.policies import AllPolicy, HardMiningPolicy, RegularPolicy
from geo_mlops.tasks.segmentation.building.adapter import BuildingSegmentationAdapter
from geo_mlops.tasks.segmentation.building.dataset import BuildingDataset, BuildingDatasetConfig
from geo_mlops.tasks.segmentation.building.losses import build_loss
from geo_mlops.tasks.segmentation.building.metrics import build_metrics_fn
from geo_mlops.tasks.segmentation.building.model_factory import build_model


TilingPolicyT = AllPolicy | RegularPolicy | HardMiningPolicy


@dataclass(frozen=True)
class BuildingSegmentationTask:
    """
    Public task plugin for building segmentation.

    Core code should interact with this object instead of importing
    building-specific datasets, losses, metrics, adapters, or model factories directly.
    """

    name: str = "building_seg"

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    def load_task_cfg(self, task_cfg_path: str | Path) -> Dict[str, Any]:
        cfg = load_cfg(task_cfg_path)
        if not isinstance(cfg, dict):
            raise ValueError(
                f"Task config root must be a mapping. Got {type(cfg).__name__}."
            )
        return cfg

    def require_section(self, cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
        value = cfg.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"Task config must include a '{section}' mapping.")
        return value

    # -------------------------------------------------------------------------
    # Tiling
    # -------------------------------------------------------------------------
    def build_tiling_components(
        self,
        task_cfg_path: str | Path,
    ) -> tuple[EngineConfig, BuildingSegmentationAdapter, TilingPolicyT, Dict[str, Any]]:
        cfg = self.load_task_cfg(task_cfg_path)
        tiling_cfg = self.require_section(cfg, "tiling")

        engine_cfg = self.require_section(tiling_cfg, "engine")
        adapter_cfg = self.require_section(tiling_cfg, "adapter")
        policy_cfg = self.require_section(tiling_cfg, "policy")
        meta = tiling_cfg.get("meta", {}) or {}

        if not isinstance(meta, dict):
            raise ValueError("tiling.meta must be a mapping if provided.")

        engine = self._build_engine_cfg(engine_cfg)
        adapter = self._build_adapter(adapter_cfg)
        policy = self._build_policy(policy_cfg)

        return engine, adapter, policy, meta

    def _build_engine_cfg(self, engine_cfg: Dict[str, Any]) -> EngineConfig:
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

    def _build_adapter(self, adapter_cfg: Dict[str, Any]) -> BuildingSegmentationAdapter:
        return BuildingSegmentationAdapter(
            class_of_interest_id=int(adapter_cfg.get("class_of_interest_id", 1)),
            shadow_id=adapter_cfg.get("shadow_id", 2),
            emit_shadow=bool(adapter_cfg.get("emit_shadow", True)),
            min_change_pixels=int(adapter_cfg.get("min_change_pixels", 1)),
        )

    def _build_policy(self, policy_cfg: Dict[str, Any]) -> TilingPolicyT:
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
            if "min_difficulty" not in policy_cfg:
                raise ValueError("tiling.policy.min_difficulty is required for hard_mining policy.")

            return HardMiningPolicy(
                min_difficulty=float(policy_cfg["min_difficulty"]),
                include_if_gt_present=bool(policy_cfg.get("include_if_gt_present", True)),
                gt_presence_threshold=float(policy_cfg.get("gt_presence_threshold", 1e-6)),
                presence_prefix=str(policy_cfg.get("presence_prefix", "presence__")),
                difficulty_prefix=str(policy_cfg.get("difficulty_prefix", "difficulty__")),
            )

        raise ValueError(
            f"Unknown tiling policy kind={kind!r}. "
            "Expected one of: all, regular, hard_mining."
        )

    # -------------------------------------------------------------------------
    # Training / evaluation components
    # -------------------------------------------------------------------------
    def build_training_cfg(self, task_cfg_path: str | Path) -> Dict[str, Any]:
        cfg = self.load_task_cfg(task_cfg_path)
        return self.require_section(cfg, "training")

    def build_model(self, train_cfg: Dict[str, Any]) -> torch.nn.Module:
        return build_model(train_cfg)

    def build_loss(self, train_cfg: Dict[str, Any]):
        return build_loss(train_cfg)

    def build_metrics_fn(self, train_cfg: Dict[str, Any]):
        return build_metrics_fn(train_cfg)

    def build_forward_fn(self, train_cfg: Dict[str, Any]):
        """
        Task-specific forward function used by generic training/evaluation engines.

        Core calls:
            outputs = forward_fn(model, batch, device)

        Core does not know about tile_tensor/context_tensor.
        """

        def forward_fn(
            model: torch.nn.Module,
            batch: Dict[str, Any],
            device: torch.device,
        ) -> torch.Tensor:
            if "tile_tensor" not in batch:
                raise KeyError("Building forward_fn expects batch['tile_tensor'].")

            tile = batch["tile_tensor"].to(device)
            ctx = batch.get("context_tensor", None)

            if ctx is not None:
                ctx = ctx.to(device)
                return model(tile, ctx)

            return model(tile)

        return forward_fn

    # -------------------------------------------------------------------------
    # Dataset builders
    # -------------------------------------------------------------------------
    def build_dataset(
        self,
        *,
        tiles_df: pd.DataFrame,
        indices: Optional[Sequence[int]],
        cfg: Dict[str, Any],
    ) -> BuildingDataset:
        dataset_cfg = cfg.get("dataset", {}) or {}

        return BuildingDataset(
            tiles_df=tiles_df,
            indices=indices,
            cfg=BuildingDatasetConfig.from_dict(dataset_cfg),
            cache_context=bool(dataset_cfg.get("cache_context", True)),
            context_cache_max_items=int(dataset_cfg.get("context_cache_max_items", 256)),
        )

    def build_train_val_datasets(
        self,
        *,
        tiles: TilesContract,
        split: SplitContract,
        train_cfg: Dict[str, Any],
    ) -> tuple[BuildingDataset, BuildingDataset]:
        df = pd.read_csv(tiles.master_csv)

        split_cfg = split.meta.get("cfg", {}) if split.meta else {}
        group_col = str(split_cfg.get("group_col", "scene_id"))

        if group_col not in df.columns:
            raise ValueError(
                f"Split group_col={group_col!r} not found in tiles CSV. "
                f"Available columns: {list(df.columns)}"
            )

        train_groups = set(map(str, split.train_regions))
        val_groups = set(map(str, split.val_regions))

        group_values = df[group_col].astype(str)

        train_indices = df.index[group_values.isin(train_groups)].tolist()
        val_indices = df.index[group_values.isin(val_groups)].tolist()

        if not train_indices:
            raise ValueError("No training rows matched split.train_regions.")
        if not val_indices:
            raise ValueError("No validation rows matched split.val_regions.")

        train_ds = self.build_dataset(
            tiles_df=df,
            indices=train_indices,
            cfg=train_cfg,
            split_name="train",
        )

        val_ds = self.build_dataset(
            tiles_df=df,
            indices=val_indices,
            cfg=train_cfg,
            split_name="val",
        )

        return train_ds, val_ds

    def build_eval_dataset(
        self,
        *,
        tiles: TilesContract,
        split: SplitContract,
        eval_cfg: Dict[str, Any],
        partition: str,
    ) -> BuildingDataset:
        df = pd.read_csv(tiles.master_csv)

        split_cfg = split.meta.get("cfg", {}) if split.meta else {}
        group_col = str(split_cfg.get("group_col", "scene_id"))

        if group_col not in df.columns:
            raise ValueError(
                f"Split group_col={group_col!r} not found in tiles CSV. "
                f"Available columns: {list(df.columns)}"
            )

        if partition == "train":
            groups = split.train_regions
        elif partition == "val":
            groups = split.val_regions
        else:
            groups = split.extra_partitions.get(partition)
            if groups is None:
                raise ValueError(
                    f"Unknown split partition={partition!r}. "
                    f"Available: train, val, {sorted(split.extra_partitions.keys())}"
                )

        group_set = set(map(str, groups))
        group_values = df[group_col].astype(str)
        indices = df.index[group_values.isin(group_set)].tolist()

        if not indices:
            raise ValueError(f"No rows matched split partition={partition!r}.")

        return self.build_dataset(
            tiles_df=df,
            indices=indices,
            cfg=eval_cfg,
            split_name=partition,
        )
    