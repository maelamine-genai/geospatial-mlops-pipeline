from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from geo_mlops.core.contracts.tile_contract import (
    TILES_SCHEMA_VERSION_V1,
    TilesContract,
)
from geo_mlops.core.io.tile_io import write_tiles_contract
from geo_mlops.core.tiling.adapters.base import TaskAdapter, TilingPolicy
from geo_mlops.core.tiling.engine import EngineConfig, RoiTilingEngine


def write_subdir_csv(
    subdir: Path,
    rows: List[Dict[str, Any]],
    csv_name: str,
) -> Optional[Path]:
    if not rows:
        return None

    df = pd.DataFrame(rows)

    sort_keys = [k for k in ("image_src", "y0", "x0") if k in df.columns]
    if sort_keys:
        df.sort_values(sort_keys, inplace=True, ignore_index=True)

    out_path = subdir / csv_name
    df.to_csv(out_path, index=False)
    return out_path


def _ensure_bucket_column(df: pd.DataFrame, bucket_name: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "dataset_bucket" not in df.columns:
        df["dataset_bucket"] = bucket_name
    else:
        df["dataset_bucket"] = df["dataset_bucket"].fillna(bucket_name)

    return df


def _apply_schema_defaults(
    df: pd.DataFrame,
    schema_defaults: Dict[str, Any],
) -> pd.DataFrame:
    if df.empty or not schema_defaults:
        return df

    df = df.copy()

    for col, default in schema_defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def _sort_master(
    df: pd.DataFrame,
    sort_keys: Sequence[str] = (
        "dataset_bucket",
        "region",
        "subregion",
        "image_src",
        "y0",
        "x0",
    ),
) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    keys = [k for k in sort_keys if k in df.columns]

    if keys:
        df.sort_values(keys, inplace=True, ignore_index=True)

    return df


def _resolve_dataset_roots(
    dataset_root: Path,
    dataset_buckets: Optional[Sequence[str]],
) -> List[Path]:
    root = Path(dataset_root)

    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"dataset_root is not a directory: {root}")

    if dataset_buckets is None:
        roots = [p for p in sorted(root.iterdir()) if p.is_dir()]
    else:
        roots = [root / str(bucket) for bucket in dataset_buckets]

    if not roots:
        raise ValueError(f"No dataset bucket directories found under: {root}")

    missing = [p for p in roots if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset bucket directory/directories: "
            + ", ".join(str(p) for p in missing)
        )

    not_dirs = [p for p in roots if not p.is_dir()]
    if not_dirs:
        raise NotADirectoryError(
            "Dataset bucket path(s) are not directories: "
            + ", ".join(str(p) for p in not_dirs)
        )

    return roots


def scan_root(
    root_dir: Path,
    regions: Optional[Sequence[str]],
    engine: RoiTilingEngine,
    csv_name: str,
    *,
    force: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root}")

    region_dirs = (
        [root / r for r in regions]
        if regions
        else [d for d in sorted(root.iterdir()) if d.is_dir()]
    )

    all_rows: List[Dict[str, Any]] = []
    bucket_name = root.name

    for region_dir in tqdm(region_dirs, desc=f"Regions ({bucket_name})"):
        if not region_dir.exists():
            tqdm.write(f"[WARN] missing region dir: {region_dir}")
            continue

        region_stats = defaultdict(int)
        subdirs = [d for d in sorted(region_dir.iterdir()) if d.is_dir()]

        for subdir in tqdm(subdirs, leave=False, desc=region_dir.name):
            existing_csv = subdir / csv_name

            if existing_csv.exists() and not force:
                df_existing = pd.read_csv(existing_csv)
                df_existing = _ensure_bucket_column(df_existing, bucket_name)

                if not df_existing.empty:
                    all_rows.extend(df_existing.to_dict(orient="records"))

                region_stats["roi_total"] += 1
                region_stats["skipped_existing_csv"] += 1

                if verbose:
                    tqdm.write(
                        f"[SKIP] {bucket_name}/{region_dir.name}/{subdir.name}: "
                        f"found existing {csv_name} ({len(df_existing)} rows)"
                    )

                continue

            rows, stats = engine.scan_subdir(subdir)

            for row in rows:
                row.setdefault("dataset_bucket", bucket_name)

            csv_path = write_subdir_csv(subdir, rows, csv_name)
            if csv_path is not None:
                all_rows.extend(rows)

            for k, v in stats.items():
                if isinstance(v, int) and k not in ("region", "subregion"):
                    region_stats[k] += v

            if stats.get("roi_pred_missing"):
                region_stats["roi_pred_missing_count"] += 1
            if stats.get("roi_context_missing"):
                region_stats["roi_context_missing_count"] += 1

            region_stats["roi_total"] += 1

            if verbose:
                tqdm.write(
                    f"[ROI] {bucket_name}/{region_dir.name}/{subdir.name}: "
                    f"rows={len(rows)} "
                    f"considered={stats.get('tiles_considered')} "
                    f"included={stats.get('tiles_included')}"
                )

        if verbose:
            tqdm.write(
                f"[REGION] {bucket_name}/{region_dir.name} | "
                f"ROIs={region_stats['roi_total']} "
                f"skipped_existing_csv={region_stats.get('skipped_existing_csv', 0)} | "
                f"scenes_ok={region_stats.get('scenes_processed', 0)} "
                f"no_pan={region_stats.get('scenes_skipped_no_pan', 0)} "
                f"io_err={region_stats.get('scenes_read_error', 0)} | "
                f"tiles_considered={region_stats.get('tiles_considered', 0)} "
                f"tiles_included={region_stats.get('tiles_included', 0)} "
                f"tiles_skipped={region_stats.get('tiles_skipped', 0)} "
                f"tiles_skipped_nodata={region_stats.get('tiles_skipped_nodata', 0)}"
            )

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    df = _ensure_bucket_column(df, bucket_name)
    return _sort_master(df)


def scan_datasets(
    dataset_root: Path,
    dataset_buckets: Optional[Sequence[str]],
    regions: Optional[Sequence[str]],
    engine: RoiTilingEngine,
    csv_name: str,
    *,
    force: bool,
    verbose: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    roots = _resolve_dataset_roots(dataset_root, dataset_buckets)
    resolved_bucket_names = [root.name for root in roots]

    dfs: List[pd.DataFrame] = []

    for root in roots:
        df_bucket = scan_root(
            root,
            regions=regions,
            engine=engine,
            csv_name=csv_name,
            force=force,
            verbose=verbose,
        )

        if not df_bucket.empty:
            df_bucket = df_bucket.copy()
            df_bucket["dataset_bucket"] = root.name

        dfs.append(df_bucket)

    non_empty = [df for df in dfs if not df.empty]

    if not non_empty:
        return pd.DataFrame(), resolved_bucket_names

    return _sort_master(pd.concat(non_empty, ignore_index=True)), resolved_bucket_names


def _as_plain_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}

    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, dict):
        return dict(obj)

    raise TypeError(f"Expected dataclass or dict, got {type(obj).__name__}")


def run_tiling_stage(
    *,
    task: str,
    task_cfg_path: Path,
    dataset_root: Path,
    dataset_buckets: Optional[Sequence[str]],
    regions: Optional[Sequence[str]],
    csv_name: str,
    out_dir: Path,
    engine_cfg: EngineConfig,
    adapter: TaskAdapter,
    policy: TilingPolicy,
    meta: Optional[Dict[str, Any]] = None,
    force: bool = False,
    verbose: bool = False,
) -> TilesContract:
    meta = meta or {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = RoiTilingEngine(
        cfg=engine_cfg,
        adapter=adapter,
        policy=policy,
    )

    df, resolved_dataset_buckets = scan_datasets(
        dataset_root=Path(dataset_root),
        dataset_buckets=dataset_buckets,
        regions=regions,
        engine=engine,
        csv_name=csv_name,
        force=force,
        verbose=verbose,
    )

    df = _apply_schema_defaults(
        df,
        schema_defaults=meta.get("schema_defaults", {}),
    )

    master_csv = out_dir / f"{task}_tiles_master.csv"
    df.to_csv(master_csv, index=False)

    contract = TilesContract(
        tiles_dir=out_dir,
        master_csv=master_csv,
        schema_version=TILES_SCHEMA_VERSION_V1,
        task=task,
        datasets_root=Path(dataset_root),
        dataset_buckets=resolved_dataset_buckets,
        regions=list(regions) if regions is not None else None,
        engine_cfg=_as_plain_dict(engine_cfg),
        adapter={
            "module": type(adapter).__module__,
            "name": type(adapter).__name__,
        },
        policy={
            "module": type(policy).__module__,
            "name": type(policy).__name__,
        },
        csv_name=csv_name,
        row_count=int(len(df)),
        meta={
            **meta,
            "task_cfg_path": str(task_cfg_path),
            "dataset_buckets_source": "cli" if dataset_buckets is not None else "discovered_from_dataset_root",
        },
    )

    write_tiles_contract(contract)

    return contract