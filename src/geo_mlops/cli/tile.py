from __future__ import annotations

import argparse
from pathlib import Path

from geo_mlops.core.contracts.tile_contract import TILES_MANIFEST_NAME
from geo_mlops.core.registry.task_registry import get_task
from geo_mlops.core.tiling.stage import run_tiling_stage


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate tile CSVs using a task-agnostic tiling engine "
            "and task-specific adapter/policy components."
        )
    )

    ap.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task key registered in task_registry, e.g. building_seg.",
    )
    ap.add_argument(
        "--task-cfg",
        "--task_cfg",
        dest="task_cfg",
        type=Path,
        required=True,
        help="Unified task config YAML/JSON.",
    )
    ap.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        type=Path,
        required=True,
        help="Directory containing dataset bucket subdirectories.",
    )
    ap.add_argument(
        "--dataset-buckets",
        nargs="+",
        default=None,
        help="One or more dataset bucket names under --dataset-root.",
    )
    ap.add_argument(
        "--regions",
        type=str,
        nargs="*",
        default=None,
        help="Optional region directory names to process under each bucket.",
    )
    ap.add_argument(
        "--csv-name",
        "--csv_name",
        dest="csv_name",
        type=str,
        required=True,
        help="Per-subdir CSV filename to write or reuse.",
    )
    ap.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=Path,
        required=True,
        help="Output directory for master CSV and tiles manifest.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute per-subdir CSVs even if they already exist.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress logs.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    task_plugin = get_task(args.task)

    engine_cfg, adapter, policy, meta = task_plugin.build_tiling_components(
        task_cfg_path=args.task_cfg,
    )

    contract = run_tiling_stage(
        task=args.task,
        task_cfg_path=args.task_cfg,
        dataset_root=args.dataset_root,
        dataset_buckets=args.dataset_buckets,
        regions=args.regions,
        csv_name=args.csv_name,
        out_dir=args.out_dir,
        engine_cfg=engine_cfg,
        adapter=adapter,
        policy=policy,
        meta=meta or {},
        force=args.force,
        verbose=args.verbose,
    )

    manifest_path = contract.tiles_dir / TILES_MANIFEST_NAME

    print(f"[tiling] wrote {contract.row_count} rows")
    print(f"[master] {contract.master_csv}")
    print(f"[manifest] {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()