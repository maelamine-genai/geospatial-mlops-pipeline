from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from datetime import datetime


@dataclass(frozen=True)
class TrainInputs:
    task: str
    tiles_manifest_path: Path
    split_json_path: Path
    train_cfg_path: Path
    out_dir: Path

    tiles_master_csv: Path
    train_row_indices: np.ndarray
    val_row_indices: np.ndarray

    # parsed config dict (resolved once)
    train_cfg: Dict[str, Any]

@dataclass(frozen=True)
class TrainOutputs:
    run_dir: Path
    model_path: Path
    metrics_path: Path
    train_manifest_path: Path