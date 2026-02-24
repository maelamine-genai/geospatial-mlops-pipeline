from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

TILES_MANIFEST_NAME = "tiles_manifest.json"
TILES_SCHEMA_VERSION_V1 = "tiles_v1"


@dataclass(frozen=True)
class TilesContract:
    """
    Output of the tiling stage (generate_tiles_csv) as consumed by downstream stages.
    Canonical artifact is the master tile CSV.
    Per-subdir CSVs are cache artifacts and not required for the contract.
    """

    tiles_dir: Path  # directory that contains master CSV + manifest
    master_csv: Path  # canonical CSV path

    schema_version: str  # e.g. "tiles_v1"
    task: str

    datasets_root: Path
    dataset_buckets: List[str]
    regions: Optional[List[str]]

    engine_cfg: Dict[str, Any]  # EngineConfig serialized to dict
    adapter: Dict[str, str]  # {"module": "...", "name": "..."}
    policy: Dict[str, str]  # {"module": "...", "name": "..."}
    csv_name: str  # per-subdir CSV filename used/expected
    row_count: int

    meta: Dict[str, Any]  # free-form task/stage metadata
