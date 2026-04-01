from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from geo_mlops.core.contracts.train_contract import TrainInputs


def resolve_training_inputs(
    *,
    tiles_manifest_path: Path,
    split_json_path: Path,
    train_cfg_path: Path,
    out_dir: Path,
) -> TrainInputs:
    tiles_manifest_path = Path(tiles_manifest_path)
    split_json_path = Path(split_json_path)
    train_cfg_path = Path(train_cfg_path)
    out_dir = Path(out_dir)

    tiles = json.loads(tiles_manifest_path.read_text())
    split = json.loads(split_json_path.read_text())
    train_cfg = yaml.safe_load(train_cfg_path.read_text()) or {}

    master_csv = Path(tiles["master_csv"])
    task = str(tiles["task"])

    df = pd.read_csv(master_csv)

    # Your split.json is subregion-based (train_regions/val_regions)
    train_regions: List[str] = list(split.get("train_regions", []))
    val_regions: List[str] = list(split.get("val_regions", []))

    if not train_regions or not val_regions:
        raise ValueError("split.json must include non-empty train_regions and val_regions")

    if "subregion" not in df.columns:
        raise ValueError("tiles_master.csv must contain 'subregion' column for region-based split")

    train_idx = df.index[df["region"].isin(train_regions)].to_numpy(dtype=np.int64)
    val_idx = df.index[df["region"].isin(val_regions)].to_numpy(dtype=np.int64)

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"Empty split after filtering: train={len(train_idx)} val={len(val_idx)}")

    return TrainInputs(
        task=task,
        tiles_manifest_path=tiles_manifest_path,
        split_json_path=split_json_path,
        train_cfg_path=train_cfg_path,
        out_dir=out_dir,
        tiles_master_csv=master_csv,
        train_row_indices=train_idx,
        val_row_indices=val_idx,
        train_cfg=train_cfg,
    )