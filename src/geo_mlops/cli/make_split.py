"""
Task-agnostic MLOps CLI: create train/val(/test) splits from a tiling stage output directory.

Input:
- A tiling output directory containing:
    - tiles_manifest.json
    - master tiles CSV (path referenced by manifest)

Outputs (in --out-dir):
- split.json       (canonical SplitContract via core/contracts/splits_io.py; includes rich meta)
- group_stats.csv  (if available from splitter)
- train_<prefix>.txt / val_<prefix>.txt (optional convenience outputs)

Notes:
- `split.json` is the canonical stage artifact. Downstream stages should load it via splits_io.py.
- Convenience artifacts (txt/csv) are for humans and quick debugging; they are not required for downstream code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from geo_mlops.core.contracts.split_contract import SplitContract
from geo_mlops.core.io.split_io import write_split_contract
from geo_mlops.core.io.tile_io import load_tiles_contract
from geo_mlops.core.splitting.split import (
    SplitConfig,
    make_splits_from_csvs,
    parse_ratios,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create deterministic group-aware train/val(/test) splits from a tiling output directory."
    )

    p.add_argument("--task", required=True, help="Task name, e.g. water_segmentation")
    p.add_argument(
        "--split-config",
        required=True,
        help="Path to split config (YAML).",
    )
    p.add_argument("--tiles-dir", required=True, help="Tiling output directory containing tiles_manifest.json")
    p.add_argument("--out-dir", required=True, help="Output directory for split artifacts")

    # convenience outputs
    p.add_argument("--no-group-lists", action="store_true", help="Do not write train_*.txt / val_*.txt")
    p.add_argument("--no-group-stats", action="store_true", help="Do not write group_stats.csv")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg_path = Path(args.split_config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Split config not found: {cfg_path}")

    spec = yaml.safe_load(cfg_path.read_text())
    if not isinstance(spec, dict):
        raise ValueError(f"Split config root must be a mapping. Got: {type(spec)}")

    stage = spec.get("stage")
    grouping = spec.get("grouping")
    strat = spec.get("stratification")
    outputs = spec.get("outputs")

    splits = stage.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("stage.splits must be a mapping, e.g. {train: 0.8, val: 0.2}")

    train = float(splits.get("train"))
    val = float(splits.get("val"))
    test = float(splits.get("test", 0.0))
    if train <= 0 or val <= 0:
        raise ValueError("stage.splits must include positive 'train' and 'val'")
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"stage.splits must sum to 1.0 (got {total:.6f}). Fix the config.")
    ratios = parse_ratios([train, val, test] if test > 0 else [train, val])

    seed = int(stage.get("seed"))
    group_col = str(grouping.get("group_col", "scene_id"))

    group_list_prefix = str(outputs.get("group_list_prefix"))

    # ----- stratification mapping -----
    kind = str(strat.get("kind")).strip().lower()

    group_metric_mode: Optional[str] = None
    group_metric_col: Optional[str] = None
    presence_eps: float = 0.001
    bins: Optional[List[float]] = None

    if kind in "grouped":
        policy = "grouped"
    elif kind == "predefined":
        policy = "predefined"
    elif kind in ("binned_group_metric", "categorical_group_metric"):
        policy = "stratified"
        gm = strat.get("group_metric")
        if not isinstance(gm, dict):
            raise ValueError("stratification.group_metric must be a mapping for stratified splitting")
        group_metric_mode = str(gm.get("mode")).strip()
        group_metric_col = str(gm.get("source_col")).strip()
        if group_metric_mode == "presence_frac":
            presence_eps = float(gm.get("presence_eps"))

        if kind == "binned_group_metric":
            bins_raw = strat.get("bins")
            if not isinstance(bins_raw, list) or len(bins_raw) < 2:
                raise ValueError("stratification.bins must be a list of >=2 numeric edges")
            bins = [float(x) for x in bins_raw]
    else:
        raise ValueError(
            f"Unknown stratification.kind='{kind}'. Expected one of: "
            "none, grouped, predefined, binned_group_metric, categorical_group_metric"
        )

    cfg = SplitConfig(
        policy=policy,
        seed=seed,
        ratios=ratios,
        group_col=group_col,
        group_metric_mode=group_metric_mode,
        group_metric_col=group_metric_col,
        presence_eps=presence_eps,
        bins=bins,
        prefix=str(outputs.get("prefix", "tiles")),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = load_tiles_contract(Path(args.tiles_dir))
    tiles_csv = tiles.master_csv

    res = make_splits_from_csvs([tiles_csv], config=cfg)

    train_groups = sorted(map(str, res.train[group_col].dropna().unique().tolist()))
    val_groups = sorted(map(str, res.val[group_col].dropna().unique().tolist()))

    if not args.no_group_lists:
        (out_dir / f"train_{group_list_prefix}.txt").write_text("\n".join(train_groups) + "\n")
        (out_dir / f"val_{group_list_prefix}.txt").write_text("\n".join(val_groups) + "\n")

    if (not args.no_group_stats) and getattr(res, "group_stats", None) is not None:
        res.group_stats.to_csv(out_dir / "group_stats.csv", index=False)

    meta: Dict[str, Any] = {
        "task": args.task,
        "split_config_path": str(cfg_path),
        "split_config_name": str(args.split_config),
        "cfg": {
            "policy": cfg.policy,
            "group_col": cfg.group_col,
            "seed": cfg.seed,
            "ratios": {"train": cfg.ratios.train, "val": cfg.ratios.val, "test": cfg.ratios.test},
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
        "counts": {"train_groups": len(train_groups), "val_groups": len(val_groups)},
    }

    contract = SplitContract(
        split_dir=out_dir,
        train_regions=train_groups,
        val_regions=val_groups,
        extra_partitions={k: list(v) for k, v in (getattr(res, "extra_partitions", None) or {}).items()},
        meta=meta,
    )
    write_split_contract(contract)

    print(f"[make_splits] wrote split artifacts to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
