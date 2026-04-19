from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

GATE_SCHEMA_VERSION_V1 = "gate.v1"

@dataclass(frozen=True)
class GateCheckResult:
    metric: str
    scope: str                 # e.g. "micro", "macro", "class:building"
    comparator: str            # ">=", "<=", etc.
    threshold: float
    actual: float
    passed: bool

@dataclass(frozen=True)
class GateContract:
    gate_dir: Path
    schema_version: str

    gate_name: str             # "gate_a" or "gate_b"
    task: str

    decision: str              # "pass" | "fail"
    passed: bool

    checks: List[Dict[str, Any]]
    summary: Dict[str, Any]    # counts, key metrics, etc.

    upstream: Dict[str, Any]   # train_manifest, eval_manifest, split_json, tiles_manifest
    threshold_spec: Dict[str, Any]
    meta: Dict[str, Any]