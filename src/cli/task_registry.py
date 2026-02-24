from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from geo_mlops.core.tiling.adapters.base import TaskAdapter, TilingPolicy
from geo_mlops.core.tiling.engine import EngineConfig


@dataclass(frozen=True)
class TaskSpec:
    name: str
    build_path: str  # "pkg.module:function"


_TASKS: Dict[str, TaskSpec] = {
    "building_seg": TaskSpec("building_seg", "src.tasks.segmentation.building.tiling_factory:build_from_cfg"),
    "noise_cls": TaskSpec(
        "noise_cls", "src.tasks.classification.noise.tiling_factory:build_from_cfg"
    ),
}


def list_tasks() -> list[str]:
    return sorted(_TASKS.keys())


def get_task_spec(task_name: str) -> TaskSpec:
    if task_name not in _TASKS:
        raise KeyError(f"Unknown task '{task_name}'. Available: {list_tasks()}")
    return _TASKS[task_name]


def _load_build_fn(path: str):
    mod_name, fn_name = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def build_tiling_components(
    *,
    task: str,
    task_cfg_path: str,
) -> Tuple[EngineConfig, TaskAdapter, TilingPolicy, Dict[str, Any]]:
    spec = get_task_spec(task)
    build_fn = _load_build_fn(spec.build_path)
    return build_fn(task_cfg_path)
