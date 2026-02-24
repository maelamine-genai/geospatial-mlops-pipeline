from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from geo_mlops.core.contracts.tile_contract import(
    TILES_SCHEMA_VERSION_V1,
    TilesContract,
)
from geo_mlops.core.io.tile_io import write_tiles_contract
from geo_mlops.core.tiling.adapters.base import TaskAdapter, TilingPolicy
from geo_mlops.core.tiling.engine import EngineConfig, RoiTilingEngine

# Task registry: maps --task to a factory that builds (cfg, adapter, policy, meta)
from cli.task_registry import build_tiling_components
from tqdm import tqdm

ALLOWED_BUCKETS = {"SN2_buildings_train_AOI_3_Paris", "SN2_buildings_train_AOI_5_Khartoum"}


# -----------------------------
# Task factory contract
# -----------------------------
@dataclass(frozen=True)
class TilingBundle:
    cfg: EngineConfig
    adapter: TaskAdapter
    policy: TilingPolicy
    meta: Dict[str, Any]


# -----------------------------
# CSV helpers
# -----------------------------
def write_subdir_csv(subdir: Path, rows: List[Dict[str, Any]], csv_name: str) -> Optional[Path]:
    if not rows:
        print("No rows")
        return None
    df = pd.DataFrame(rows)
    # generic stable sort if present
    sort_keys = [k for k in ("image_src", "y0", "x0") if k in df.columns]
    if sort_keys:
        df.sort_values(sort_keys, inplace=True, ignore_index=True)
    out_path = subdir / csv_name
    df.to_csv(out_path, index=False)
    return out_path


def _ensure_bucket_column(df: pd.DataFrame, bucket_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    if "dataset_bucket" not in df.columns:
        df["dataset_bucket"] = bucket_name
    else:
        df["dataset_bucket"] = df["dataset_bucket"].fillna(bucket_name)
    return df


def _apply_schema_defaults(df: pd.DataFrame, schema_defaults: Dict[str, Any]) -> pd.DataFrame:
    """
    Optional: tasks can provide schema_defaults to keep downstream schema stable.
    Only fills missing columns; never overwrites existing.
    """
    if df.empty or not schema_defaults:
        return df
    for col, default in schema_defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def _sort_master(df: pd.DataFrame, sort_keys: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    keys = [k for k in sort_keys if k in df.columns]
    if keys:
        df.sort_values(keys, inplace=True, ignore_index=True)
    return df


# -----------------------------
# Scanners
# -----------------------------
def scan_root(
    root_dir: Path,
    regions: Optional[List[str]],
    engine: RoiTilingEngine,
    csv_name: str,
    *,
    force: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root}")

    region_dirs = [root / r for r in regions] if regions else [d for d in root.iterdir() if d.is_dir()]
    all_rows: List[Dict[str, Any]] = []
    bucket_name = root.name

    for region_dir in tqdm(region_dirs, desc=f"Regions ({root.name})"):
        if not region_dir.exists():
            tqdm.write(f"[WARN] missing region dir: {region_dir}")
            continue

        R = defaultdict(int)
        subdirs = [d for d in sorted(region_dir.iterdir()) if d.is_dir()]
        for subdir in tqdm(subdirs, leave=False, desc=region_dir.name):
            existing_csv = subdir / csv_name

            # --- IDEMPOTENCE ---
            if existing_csv.exists() and not force:
                df_existing = pd.read_csv(existing_csv)
                if not df_existing.empty:
                    df_existing = _ensure_bucket_column(df_existing, bucket_name)
                    all_rows.extend(df_existing.to_dict(orient="records"))
                if verbose:
                    tqdm.write(
                        f"[SKIP] {root.name}/{region_dir.name}/{subdir.name}: "
                        f"found existing {csv_name} ({len(df_existing)} rows)"
                    )
                R["roi_total"] += 1
                R["skipped_existing_csv"] += 1
                continue

            # --- compute fresh ---
            rows, stats = engine.scan_subdir(subdir)
            print(
                f"[DEBUG] {region_dir.name}/{subdir.name}: rows={len(rows)} stats_tiles_considered={stats.get('tiles_considered')} included={stats.get('tiles_included')}"
            )

            # annotate dataset bucket (universal, not task-specific)
            for r in rows:
                r.setdefault("dataset_bucket", bucket_name)

            csv_path = write_subdir_csv(subdir, rows, csv_name)
            if csv_path is not None:
                all_rows.extend(rows)

            # accumulate numeric stats
            for k, v in stats.items():
                if isinstance(v, int) and k not in ("region", "subregion"):
                    R[k] += v
            if stats.get("roi_pred_missing"):
                R["roi_pred_missing_count"] += 1
            if stats.get("roi_context_missing"):
                R["roi_context_missing_count"] += 1
            R["roi_total"] += 1

        if verbose:
            tqdm.write(
                f"[REGION] {root.name}/{region_dir.name} | ROIs={R['roi_total']} "
                f"(preds_missing={R.get('roi_pred_missing_count', 0)}, "
                f"context_missing={R.get('roi_context_missing_count', 0)}, "
                f"skipped_existing_csv={R.get('skipped_existing_csv', 0)}) | "
                f"scenes: ok={R.get('scenes_processed', 0)}, "
                f"no_pan={R.get('scenes_skipped_no_pan', 0)}, "
                f"io_err={R.get('scenes_read_error', 0)} | "
                f"tiles: considered={R.get('tiles_considered', 0)}, "
                f"included={R.get('tiles_included', 0)}, "
                f"skipped={R.get('tiles_skipped', 0)}, "
                f"skipped_nodata={R.get('tiles_skipped_nodata', 0)}"
            )

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    df = _ensure_bucket_column(df, bucket_name)

    # Generic master sort (works for all tasks as long as engine provides these core cols)
    df = _sort_master(
        df,
        sort_keys=("dataset_bucket", "region", "subregion", "image_src", "y0", "x0"),
    )
    return df


def scan_datasets(
    dataset_root: Path,
    dataset_buckets: Sequence[str],
    regions: Optional[List[str]],
    engine: RoiTilingEngine,
    csv_name: str,
    *,
    force: bool,
    verbose: bool,
) -> pd.DataFrame:
    roots: List[Path] = []
    for b in dataset_buckets:
        if b not in ALLOWED_BUCKETS:
            raise ValueError(f"Unknown dataset bucket '{b}'. Allowed: {sorted(ALLOWED_BUCKETS)}")
        roots.append(Path(dataset_root) / b)

    dfs: List[pd.DataFrame] = []
    for root in roots:
        df_b = scan_root(root, regions=regions, engine=engine, csv_name=csv_name, force=force, verbose=verbose)
        if not df_b.empty:
            df_b["dataset_bucket"] = root.name  # enforce consistency
        dfs.append(df_b)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df = _sort_master(df, sort_keys=("dataset_bucket", "region", "subregion", "image_src", "y0", "x0"))
    return df


# -----------------------------
# Build tiling components from --task
# -----------------------------
def build_bundle_from_args(args) -> TilingBundle:
    """
    Delegates task-specific construction to the registry.
    The task factory should return:
      cfg: EngineConfig
      adapter: TaskAdapter
      policy: TilingPolicy
      meta: dict (optional)
    """
    cfg, adapter, policy, meta = build_tiling_components(
        task=args.task,
        task_cfg_path=str(args.task_cfg),
    )

    return TilingBundle(cfg=cfg, adapter=adapter, policy=policy, meta=(meta or {}))


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        "Unified tiler CSV generator over Datasets/<bucket>/<region>/<subdir>/... "
        "(task-agnostic engine + task adapters + universal policies)"
    )

    # required
    ap.add_argument("--task", type=str, required=True, help="Task key registered in task_registry (e.g., water_seg).")
    ap.add_argument("--task_cfg", type=Path, required=True)
    # dataset traversal
    ap.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Root directory containing dataset buckets (Golden-Train-Regions, Golden-Test-Regions, Dataops-Regions).",
    )
    ap.add_argument(
        "--dataset-buckets",
        nargs="+",
        default=["Golden-Train-Regions", "Golden-Test-Regions"],
        help="One or more dataset buckets under --dataset_root.",
    )
    ap.add_argument(
        "--regions",
        type=str,
        nargs="*",
        default=None,
        help="Optional: only process these region directory names (under each bucket).",
    )

    # outputs
    ap.add_argument("--csv_name", type=str, required=True, help="Per-subdir CSV filename (defaults to task meta).")
    ap.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for the tiling stage (will contain master CSV + tiles_manifest.json).",
    )

    # idempotence / logging
    ap.add_argument("--force", action="store_true", help="Recompute even if per-subdir CSV already exists")
    ap.add_argument("--verbose", action="store_true", help="Verbose progress logging")

    # NOTE: task-specific flags are intentionally NOT defined here.
    # We parse known args and pass the rest to the task factory.
    args, unknown = ap.parse_known_args()

    # validate buckets early
    unknown_buckets = set(args.dataset_buckets) - ALLOWED_BUCKETS
    if unknown_buckets:
        ap.error(f"Unknown dataset bucket(s): {sorted(unknown_buckets)}. Allowed: {sorted(ALLOWED_BUCKETS)}")

    bundle = build_bundle_from_args(args)

    engine = RoiTilingEngine(cfg=bundle.cfg, adapter=bundle.adapter, policy=bundle.policy)

    df = scan_datasets(
        dataset_root=args.dataset_root,
        dataset_buckets=args.dataset_buckets,
        regions=args.regions,
        engine=engine,
        csv_name=args.csv_name,
        force=args.force,
        verbose=args.verbose,
    )

    # Optional: enforce task schema defaults (columns) if provided
    df = _apply_schema_defaults(df, schema_defaults=bundle.meta.get("schema_defaults", {}))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    master_csv = out_dir / f"{args.task}_tiles_master.csv"

    df.to_csv(master_csv, index=False)
    print(f"[master] wrote {len(df)} rows -> {master_csv}")

    # -----------------------------
    # Write tiling stage contract (canonical)
    # -----------------------------
    tiles_dir = out_dir

    contract = TilesContract(
        tiles_dir=tiles_dir,
        master_csv=master_csv,
        schema_version=TILES_SCHEMA_VERSION_V1,
        task=args.task,
        datasets_root=args.dataset_root,
        dataset_buckets=list(args.dataset_buckets),
        regions=list(args.regions) if args.regions is not None else None,
        engine_cfg=asdict(bundle.cfg) if is_dataclass(bundle.cfg) else dict(bundle.cfg),
        adapter={
            "module": type(bundle.adapter).__module__,
            "name": type(bundle.adapter).__name__,
        },
        policy={
            "module": type(bundle.policy).__module__,
            "name": type(bundle.policy).__name__,
        },
        csv_name=args.csv_name,
        row_count=int(len(df)),
        meta=bundle.meta,
    )

    manifest_path = write_tiles_contract(contract)
    print(f"[manifest] wrote -> {manifest_path}")

    print("Done.")


if __name__ == "__main__":
    main()
