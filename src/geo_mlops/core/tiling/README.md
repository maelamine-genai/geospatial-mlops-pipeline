# Tiling System — Developer Guide

This document explains **how tiling works end‑to‑end** in the CVRS ML platform, and how ML developers should **extend or reuse it for new tasks**.

The goal of tiling is to convert large geospatial scenes into **task‑ready tile rows** (CSV) in a **task‑agnostic, reproducible, and extensible** way.

---

## 1. Mental Model (Read This First)

Tiling is a **pipeline stage** with three layers:

1. **Core tiling engine** – owns the scanning loop and geometry
2. **Task adapter** – explains task semantics (GT, preds, context)
3. **Policy** – decides *which* tiles to include and *why*

> The engine never knows what “water”, “cloud”, or “rooftop” means.
> Tasks never re‑implement scanning logic.

The output of the stage is:

* a **master tiles CSV** (canonical artifact)
* a **tiles manifest** describing how it was produced

---

## 2. High‑Level Data Flow

```
Datasets root
   ↓
[ core/tiling/engine.py ]
   ↓  (TileWindow + SceneInputs)
[ TaskAdapter ]
   ↓  (task semantics)
[ TilingPolicy ]
   ↓  (include / exclude + extra fields)
rows → DataFrame → master CSV
                    ↓
              tiles_manifest.json
```

---

## 3. Core Tiling Components

### 3.1 `EngineConfig`

**Location:** `core/tiling/engine.py`

Defines *how* scenes are scanned:

* input glob patterns (PAN, GT, preds, context)
* target tile size (meters)
* overlap
* nodata behavior

This config is:

* task‑agnostic
* serializable
* embedded into the tiling manifest for provenance

---

### 3.2 `RoiTilingEngine`

**Location:** `core/tiling/engine.py`

Responsibilities:

* iterate over regions / subregions
* load rasters
* generate `TileWindow`'s
* compute geometric metadata
* assemble **core row fields**

Guarantees **stable core columns**, such as:

* region / subregion
* scene_id
* image_src / gt_src / pred_src / context_src
* x0, y0, x1, y1
* tile index, row, col
* gsd, tile size, stride, overlap

The engine **never**:

* filters tiles
* interprets labels
* computes task metrics

---

## 4. Task Adapters

### 4.1 What an Adapter Is

**Location:** `core/tiling/adapters/base.py`

A `TaskAdapter` explains:

* what inputs the task requires (GT, preds, context)
* how to compute task‑specific signals (presence, difficulty)
* how to add task‑specific columns to a row

The adapter is the **only place** where task semantics enter tiling.

---

### 4.2 Base Adapters

Provided by core:

* `BaseAdapter`
* `SegmentationAdapter`
* `ClassificationAdapter`

They implement common patterns and defaults.

---

### 4.3 Task‑Specific Adapters

**Location:** `tasks/<task>/adapter.py`

Example:

```python
class WaterSegmentationAdapter(SegmentationAdapter):
    ...
```

Task adapters may:

* compute GT presence
* compute prediction difficulty
* add task‑specific row fields

They should **not**:

* filter tiles
* decide inclusion

---

## 5. Tiling Policies

### 5.1 What a Policy Is

**Location:** `core/tiling/policies.py`

A policy answers two questions:

1. Should this tile be included?
2. What extra metadata should be attached?

Examples:

* `AllPolicy` – include everything
* `RegularPolicy` – include tiles with GT presence
* `HardMiningPolicy` – include difficult tiles

---

### 5.2 Policy Interface

Policies implement:

* `extra_row_fields()` → column schema additions
* `decide_include(...)` → `(bool, extra_fields)`

Policies **do not** know about training or splits.

---

## 6. MLOps Packaging Layer

### 6.1 `generate_tiles_csv.py`

**Location:** `mlops_tools/generate_tiles_csv.py`

Responsibilities:

* parse CLI args
* load task tiling factory via `task_registry`
* construct `EngineConfig`, `TaskAdapter`, `TilingPolicy`
* scan datasets using `RoiTilingEngine`
* aggregate rows into DataFrames
* write master CSV
* write tiling manifest

This script is the **stage runner** for tiling.

---

### 6.2 Task Registry

**Location:** `mlops_tools/task_registry.py`

Maps a task name to a tiling factory:

```python
water_seg → tasks.water_segmentation.tiling_factory:build_from_cfg
```

This keeps core free of task imports.

---

## 7. Tiling Stage Contract

### 7.1 Canonical Artifact

The canonical output of tiling is:

* **Master tiles CSV**

Per‑subdir CSVs are cache artifacts.

---

### 7.2 Tiles Manifest

A `tiles_manifest.json` is written next to the master CSV.

It records:

* task name
* engine config
* adapter + policy identity
* dataset buckets / regions
* row count
* schema version
* path to master CSV

This makes tiling a **first‑class pipeline stage**, like splits and training.

---

## 8. How Downstream Stages Use Tiling

* **Splitting** consumes the master CSV via the tiles manifest
* **Training** usually consumes splits, not tiles directly
* Tile provenance is preserved via manifests

---

## 9. How to Add a New Task

1. Create `tasks/<task>/adapter.py`
2. Create tiling config YAMLs under `tasks/<task>/config/tiling/`
3. Add task entry to `task_registry`
4. Run `generate_tiles_csv.py --task <task>`

No core code changes required.

---

## 10. Design Principles (Do Not Break These)

* Engine owns geometry and iteration
* Adapter owns task semantics
* Policy owns inclusion logic
* CSV schema is additive and extensible
* Manifests define stage boundaries

If you follow these rules, tiling stays reusable and stable.

---

## 11. Common Anti‑Patterns

* Putting task logic in the engine
* Hardcoding label IDs in core
* Filtering tiles inside adapters
* Bypassing the tiling manifest

---

## 12. Summary

Tiling is already modular and scalable.

The only recent addition is making its **implicit contract explicit** via a manifest.

Once that exists, tiling, splitting, and training follow the **same pipeline pattern**.
