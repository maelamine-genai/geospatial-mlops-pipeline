from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_cfg(path: str | Path) -> Dict[str, Any]:
    """
    Load a task configuration from YAML or JSON.

    Returns a plain dict with keys typically like:
      - engine: {...}
      - adapter: {...}
      - policy: {...}
      - meta: {...}

    Notes:
      - This loader does not enforce a schema; task factories should validate.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"task_cfg not found: {p}")

    suffix = p.suffix.lower()
    if suffix in (".json",):
        return json.loads(p.read_text())

    if suffix in (".yml", ".yaml"):
        return yaml.safe_load(p.read_text())

    raise ValueError(f"Unsupported config extension '{suffix}'. Use .yaml/.yml or .json.")


def require_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    value = cfg.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"Task config must include a '{section}' mapping.")
    return value