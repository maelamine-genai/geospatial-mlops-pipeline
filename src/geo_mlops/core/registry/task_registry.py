from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple, runtime_checkable

from geo_mlops.core.tiling.adapters.base import TaskAdapter, TilingPolicy
from geo_mlops.core.tiling.engine import EngineConfig


@runtime_checkable
class TaskPlugin(Protocol):
    name: str

    def build_tiling_components(
        self,
        task_cfg_path: str,
    ) -> Tuple[EngineConfig, TaskAdapter, TilingPolicy, Dict[str, Any]]:
        ...


@dataclass(frozen=True)
class TaskSpec:
    name: str
    plugin_path: str  # "pkg.module:object_or_factory"


_TASKS: Dict[str, TaskSpec] = {
    "building_seg": TaskSpec(
        name="building_seg",
        plugin_path="geo_mlops.tasks.segmentation.building.task:BuildingSegmentationTask",
    ),
    # Keep commented until implemented with the same plugin interface:
    # "noise_cls": TaskSpec(
    #     name="noise_cls",
    #     plugin_path="geo_mlops.tasks.classification.noise.task:NoiseClassificationTask",
    # ),
}


_PLUGIN_CACHE: Dict[str, TaskPlugin] = {}


def list_tasks() -> list[str]:
    return sorted(_TASKS.keys())


def get_task_spec(task_name: str) -> TaskSpec:
    if task_name not in _TASKS:
        raise KeyError(f"Unknown task {task_name!r}. Available: {list_tasks()}")
    return _TASKS[task_name]


def _load_symbol(path: str):
    mod_name, symbol_name = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, symbol_name)


def get_task(task_name: str) -> TaskPlugin:
    if task_name in _PLUGIN_CACHE:
        return _PLUGIN_CACHE[task_name]

    spec = get_task_spec(task_name)
    symbol = _load_symbol(spec.plugin_path)

    # Supports either:
    #   plugin_path="...:BuildingSegmentationTask"  # class
    # or:
    #   plugin_path="...:TASK"                      # instance
    if isinstance(symbol, type):
        plugin = symbol()
    else:
        plugin = symbol

    if not isinstance(plugin, TaskPlugin):
        raise TypeError(
            f"Task plugin for {task_name!r} does not implement TaskPlugin protocol. "
            f"Loaded object: {plugin!r}"
        )

    if plugin.name != task_name:
        raise ValueError(
            f"Task plugin name mismatch: registry key={task_name!r}, "
            f"plugin.name={plugin.name!r}"
        )

    _PLUGIN_CACHE[task_name] = plugin
    return plugin


def build_tiling_components(
    *,
    task: str,
    task_cfg_path: str,
) -> Tuple[EngineConfig, TaskAdapter, TilingPolicy, Dict[str, Any]]:
    return get_task(task).build_tiling_components(task_cfg_path)