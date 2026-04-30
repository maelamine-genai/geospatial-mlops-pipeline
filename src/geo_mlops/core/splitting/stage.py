from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from geo_mlops.core.config.loader import load_cfg
from geo_mlops.core.contracts.split_contract import SplitContract
from geo_mlops.core.io.split_io import write_split_contract
from geo_mlops.core.io.tile_io import load_tiles_contract
from geo_mlops.core.splitting.split import (
    SplitConfig,
    make_splits_from_csvs,
    parse_ratios,
)


def _require_mapping(spec: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = spec.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"split config field '{key}' must be a mapping.")
    return value


def _build_split_config(spec: Dict[str, Any]) -> tuple[SplitConfig, str]:
    stage = _require_mapping(spec, "stage")
    grouping = _require_mapping(spec, "grouping")
    strat = _require_mapping(spec, "stratification")
    outputs = _require_mapping(spec, "outputs")

    splits = stage.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("stage.splits must be a mapping, e.g. {train: 0.8, val: 0.2}")

    train = float(splits.get("train", 0.0))
    val = float(splits.get("val", 0.0))
    test = float(splits.get("test", 0.0))

    if train <= 0 or val <= 0:
        raise ValueError("stage.splits must include positive 'train' and 'val' values.")

    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"stage.splits must sum to 1.0. Got {total:.6f}.")

    ratios = parse_ratios([train, val, test] if test > 0 else [train, val])

    seed = int(stage.get("seed", 42))
    group_col = str(grouping.get("group_col", "scene_id"))

    kind = str(strat.get("kind", "grouped")).strip().lower()

    group_metric_mode: Optional[str] = None
    group_metric_col: Optional[str] = None
    presence_eps: float = 0.001
    bins: Optional[List[float]] = None

    if kind in ("none", "grouped"):
        policy = "grouped"

    elif kind == "predefined":
        policy = "predefined"

    elif kind in ("binned_group_metric", "categorical_group_metric"):
        policy = "stratified"

        group_metric = strat.get("group_metric")
        if not isinstance(group_metric, dict):
            raise ValueError(
                "stratification.group_metric must be a mapping for stratified splitting."
            )

        group_metric_mode = str(group_metric.get("mode", "")).strip()
        group_metric_col = str(group_metric.get("source_col", "")).strip()

        if not group_metric_mode:
            raise ValueError("stratification.group_metric.mode is required.")
        if not group_metric_col:
            raise ValueError("stratification.group_metric.source_col is required.")

        if group_metric_mode == "presence_frac":
            presence_eps = float(group_metric.get("presence_eps", presence_eps))

        if kind == "binned_group_metric":
            bins_raw = strat.get("bins")
            if not isinstance(bins_raw, list) or len(bins_raw) < 2:
                raise ValueError("stratification.bins must be a list of at least 2 numeric edges.")
            bins = [float(x) for x in bins_raw]

    else:
        raise ValueError(
            f"Unknown stratification.kind={kind!r}. Expected one of: "
            "none, grouped, predefined, binned_group_metric, categorical_group_metric."
        )

    group_list_prefix = str(outputs.get("group_list_prefix", outputs.get("prefix", "tiles")))

    cfg = SplitConfig(
        policy=policy,
        seed=seed,
        ratios=ratios,
        group_col=group_col,
        predefined_col=str(grouping.get("predefined_col", "split")),
        min_any_ratio=stage.get("min_any_ratio"),
        ratio_cols=stage.get("ratio_cols"),
        dedupe_key=stage.get("dedupe_key"),
        group_metric_mode=group_metric_mode,
        group_metric_col=group_metric_col,
        presence_eps=presence_eps,
        bins=bins,
        prefix=str(outputs.get("prefix", "tiles")),
    )

    return cfg, group_list_prefix


def _unique_group_values(df, group_col: str) -> list[str]:
    if group_col not in df.columns:
        raise ValueError(f"Expected group column {group_col!r} in split dataframe.")
    return sorted(map(str, df[group_col].dropna().unique().tolist()))


def run_split_stage(
    *,
    task: str,
    task_cfg_path: Path,
    tiles_dir: Path,
    out_dir: Path,
    write_group_lists: bool = True,
    write_group_stats: bool = True,
) -> SplitContract:
    task_cfg_path = Path(task_cfg_path)
    tiles_dir = Path(tiles_dir)
    out_dir = Path(out_dir)

    task_cfg = load_cfg(task_cfg_path)

    if not isinstance(task_cfg, dict):
        raise ValueError(f"Task config root must be a mapping. Got {type(task_cfg).__name__}.")

    spec = task_cfg.get("splitting")
    if not isinstance(spec, dict):
        raise ValueError("Task config must include a `splitting:` mapping.")

    cfg, group_list_prefix = _build_split_config(spec)

    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = load_tiles_contract(tiles_dir)
    res = make_splits_from_csvs([tiles.master_csv], config=cfg)

    train_groups = _unique_group_values(res.train, cfg.group_col)
    val_groups = _unique_group_values(res.val, cfg.group_col)

    if write_group_lists:
        (out_dir / f"train_{group_list_prefix}.txt").write_text(
            "\n".join(train_groups) + "\n"
        )
        (out_dir / f"val_{group_list_prefix}.txt").write_text(
            "\n".join(val_groups) + "\n"
        )

    if write_group_stats and getattr(res, "group_stats", None) is not None:
        res.group_stats.to_csv(out_dir / "group_stats.csv", index=False)

    extra_partitions = getattr(res, "extra_partitions", None) or {}

    meta: Dict[str, Any] = {
        "task": task,
        "task_cfg_path": str(task_cfg_path),
        "split_cfg_section": "splitting",
        "cfg": {
            "policy": cfg.policy,
            "group_col": cfg.group_col,
            "seed": cfg.seed,
            "ratios": {
                "train": cfg.ratios.train,
                "val": cfg.ratios.val,
                "test": cfg.ratios.test,
            },
            "group_metric_mode": cfg.group_metric_mode,
            "group_metric_col": cfg.group_metric_col,
            "presence_eps": cfg.presence_eps,
            "bins": cfg.bins,
        },
        "upstream": {
            "tiles_manifest": str(tiles.tiles_dir / "tiles_manifest.json"),
            "tiles_master_csv": str(tiles.master_csv),
            "tiles_schema_version": tiles.schema_version,
            "tiles_task": tiles.task,
            "tiles_adapter": tiles.adapter,
            "tiles_policy": tiles.policy,
        },
        "warnings": list(getattr(res, "warnings", [])),
        "counts": {
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            **{
                f"{name}_groups": len(groups)
                for name, groups in extra_partitions.items()
            },
        },
    }

    contract = SplitContract(
        split_dir=out_dir,
        train_regions=train_groups,
        val_regions=val_groups,
        extra_partitions={
            name: sorted(map(str, values))
            for name, values in extra_partitions.items()
        },
        meta=meta,
    )

    write_split_contract(contract)
    return contract