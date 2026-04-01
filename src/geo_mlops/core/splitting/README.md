# Core Splitting Layer (`core/splitting`) — Splitting Guide

This document explains **what lives in `core/splitting/splits.py`**, what does **not**, and how it interacts with:

- `mlops_tools/make_splits.py`
- `core/contracts/splits_contract.py`
- `core/contracts/splits_io.py`

If you are confused about *where splitting logic belongs*, this file is the source of truth.

---

## 1. Mental Model (TL;DR)

> **`core/splitting/splits.py` is a library.**  
> **`mlops_tools/make_splits.py` is a pipeline stage runner.**

- Core computes *how* to split.
- MLOps tools decide *when, from what inputs, and where outputs go*.
- Contracts define *what a split artifact looks like*.

If you remember only one thing, remember this:

> **Core never knows about pipeline stages. Pipelines never re-implement core logic.**

---

## 2. Purpose of `core/splitting/splits.py`

`core/splitting/splits.py` implements the **group-aware splitting algorithms** used across tasks.

It answers questions like:

- How do we split without leakage?
- How do we stratify by a group-level metric?
- How do we enforce ratios deterministically?

Splits are **deterministic** given the same inputs and `SplitConfig`.

It does **not** answer questions like:

- Where does the input CSV come from?
- What task is running?
- What directory structure should outputs follow?

---

## 3. What Belongs in `core/splitting/splits.py`

### 3.1 Algorithmic Data Structures

These are *algorithm-level concepts*, not pipeline artifacts:

- `SplitConfig`
- `SplitRatios`
- `SplitResult`
- `LeakageCheckResult`

They describe *how splitting works*, not *how splits are stored on disk*.

---

### 3.2 Pure Helper Functions

Functions that:

- operate on DataFrames
- operate on in-memory structures
- are deterministic and testable

Examples include:

- deduplication helpers
- ratio normalization
- group metric computation
- binning and stratification logic

These must stay in core so they can be reused, tested, and reasoned about independently.

---

### 3.3 Split Policies

Policy implementations such as:

- grouped splits
- predefined splits
- stratified splits

These are the **heart of the splitting algorithm**.

They are task-agnostic and must never import task code.

---

### 3.4 Library Entry Points

These functions define the *public API* of the splitting library:

- `make_splits_from_csvs(csv_paths, config)`

They:

- take data + config
- return a `SplitResult`
- do **not** write pipeline contracts or artifacts

---

## 4. What Does NOT Belong in `core/splitting/splits.py`

The following must **never** live in core:

- CLI argument parsing
- YAML config discovery under `tasks/`
- knowledge of tiling manifests or training stages
- dynamic imports / task registries
- filesystem layout assumptions
- canonical pipeline artifact writing (e.g. `split.json`)

If core starts caring about these, it stops being reusable.

---

## 5. Why `mlops_tools/make_splits.py` Exists (and Why It’s Separate)

`mlops_tools/make_splits.py` exists because **splitting is a pipeline stage**, not just a function call.

Its responsibilities are:

- Discovering and loading split specs (YAML)
- Resolving **input artifacts** (via `tiles_manifest.json`)
- Applying CLI overrides
- Choosing output directories
- Calling `core/splitting/splits.py`
- Writing pipeline artifacts

It is allowed to be opinionated and pipeline-aware.

---

## 6. The Role of Contracts in Splitting

### 6.1 `SplitContract`

Defined in:

- `core/contracts/splits_contract.py`

This dataclass defines the **stage boundary**:

- what downstream stages can rely on
- what fields exist in a split artifact

It is intentionally minimal and task-agnostic.

---

### 6.2 `splits_io.py`

Defined in:

- `core/contracts/splits_io.py`

This module:

- reads and writes `split.json`
- maps JSON ↔ `SplitContract`
- centralizes schema evolution

Downstream stages must **never** parse `split.json` directly.

---

## 7. How the Pieces Fit Together

```text
Tiling stage
  ↓
TilesContract (tiles_manifest.json)
  ↓
mlops_tools/make_splits.py
  ↓
core/splitting/splits.py   ← pure algorithm
  ↓
SplitResult (in memory)
  ↓
SplitContract (split.json)
