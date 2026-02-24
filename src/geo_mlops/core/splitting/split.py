from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# ==============
# Dataclasses
# ==============


@dataclass(frozen=True)
class SplitRatios:
    """
    Split ratios container.
    Can be used for train/val-only by leaving test=0.
    """

    train: float
    val: float
    test: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"train": float(self.train), "val": float(self.val), "test": float(self.test)}


@dataclass(frozen=True)
class SplitConfig:
    """
    Core splitter config.
    policy:
      - "grouped": split groups randomly (no stratification)
      - "stratified": split groups within strata (binned metric or categorical majority label)
      - "predefined": respect an existing column defining split membership

    Notes:
      - For task-agnostic stratification, use:
          group_metric_mode:
            - "mean"           : group mean of group_metric_col
            - "presence_frac"  : frac(group_metric_col > presence_eps)
            - "majority_label" : group majority label of group_metric_col (categorical)
          bins:
            - required for numeric stratification modes ("mean", "presence_frac") unless you do categorical
    """

    policy: str  # "grouped" | "stratified" | "predefined"
    seed: int = 1337

    ratios: SplitRatios = SplitRatios(train=0.8, val=0.2, test=0.0)

    # leakage / grouping
    group_col: str = "scene_id"

    # predefined
    predefined_col: str = "split"

    # filters / hygiene
    min_any_ratio: Optional[float] = None
    ratio_cols: Optional[List[str]] = None
    dedupe_key: Optional[List[str]] = None  # e.g. ["image_src","x0","y0","x1","y1"]

    # stratification (group-level)
    group_metric_mode: Optional[str] = None  # "mean" | "presence_frac" | "majority_label"
    group_metric_col: Optional[str] = None  # row-level source col to aggregate per group
    presence_eps: float = 0.001  # used only for presence_frac
    bins: Optional[List[float]] = None  # used for mean/presence_frac

    # output naming helper
    prefix: str = "tiles"


@dataclass
class LeakageCheckResult:
    name: str
    ok: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: Optional[pd.DataFrame]
    checks: List[LeakageCheckResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Added for MLOps artifacts / debugging
    resolved_group_col: str = ""
    group_assignment: Dict[str, str] = field(default_factory=dict)  # group_id -> split
    group_stats: Optional[pd.DataFrame] = None


# ======================================================================================
# Helpers
# ======================================================================================


def parse_ratios(vals: Sequence[float]) -> SplitRatios:
    """
    Accepts [train, val] or [train, val, test]. Normalizes if they don't sum to 1.
    """
    if len(vals) not in (2, 3):
        raise ValueError("--ratios must have 2 values (train val) or 3 values (train val test)")

    t = float(vals[0])
    v = float(vals[1])
    s = float(vals[2]) if len(vals) == 3 else 0.0

    if any(x < 0 for x in (t, v, s)):
        raise ValueError("ratios must be non-negative")
    if (t + v + s) <= 0:
        raise ValueError("ratios must sum to > 0")

    total = t + v + s
    t, v, s = t / total, v / total, s / total

    # allow test=0 for train/val-only stages
    if t <= 0 or v <= 0:
        raise ValueError("train and val ratios must be > 0")
    return SplitRatios(train=t, val=v, test=s)


def resolve_group_col(df: pd.DataFrame, preferred: str = "scene_id") -> Tuple[str, List[str]]:
    """
    Chooses a grouping column for no-leakage splits.
    If preferred missing, fall back to image_src. Else create a surrogate.
    """
    warnings: List[str] = []
    if preferred in df.columns:
        return preferred, warnings
    if "image_src" in df.columns:
        warnings.append(f"[split] group_col='{preferred}' missing; falling back to group_col='image_src'")
        return "image_src", warnings

    warnings.append("[split] group_col and image_src missing; using row_index surrogate grouping (weak)")
    return "__row_group__", warnings


def _apply_min_any_ratio_filter(
    df: pd.DataFrame,
    *,
    min_any_ratio: Optional[float],
    ratio_cols: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, Optional[str]]:
    if min_any_ratio is None:
        return df, None

    thr = float(min_any_ratio)
    cols = list(ratio_cols) if ratio_cols else []
    if not cols:
        return df, f"[split] min_any_ratio={thr} requested but ratio_cols not provided; skipping filter"

    missing = [c for c in cols if c not in df.columns]
    present = [c for c in cols if c in df.columns]
    if not present:
        return df, f"[split] min_any_ratio={thr} requested but none of ratio_cols exist: {missing}; skipping filter"

    out = df.copy()
    # keep rows where ANY ratio col >= thr
    vals = out[present].apply(pd.to_numeric, errors="coerce")
    keep = (vals >= thr).any(axis=1).fillna(False)

    before = len(out)
    out = out.loc[keep].copy()
    dropped = before - len(out)

    msg = f"[split] min_any_ratio={thr} dropped {dropped} rows using cols={present}"
    if missing:
        msg += f" (missing cols ignored: {missing})"
    return out, msg


def _apply_dedupe(df: pd.DataFrame, dedupe_key: Optional[Sequence[str]]) -> Tuple[pd.DataFrame, Optional[str]]:
    if not dedupe_key:
        return df, None

    key = list(dedupe_key)
    missing = [c for c in key if c not in df.columns]
    if missing:
        return df, f"[split] dedupe_key missing columns {missing}; skipping dedupe"

    before = len(df)
    out = df.drop_duplicates(subset=key, keep="first").copy()
    after = len(out)
    if after < before:
        return out, f"[split] dedupe dropped {before - after} duplicate rows using key={key}"
    return out, None


def read_tiles_csvs(
    inputs: Sequence[Union[str, Path]],
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """
    Reads one or more tile CSVs and concatenates them. Adds '_csv_path' provenance.
    CSV-only (no raster reads).
    """
    paths: List[Path] = []
    for x in inputs:
        p = Path(x)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.csv")))
        else:
            paths.append(p)

    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError("No CSV inputs found (paths do not exist or no *.csv under provided directories)")

    dfs: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            raise RuntimeError(f"Failed reading CSV: {p} ({e})") from e

        if df.empty:
            if allow_empty:
                continue
            continue

        df = df.copy()
        df["_csv_path"] = str(p)
        dfs.append(df)

    if not dfs:
        if allow_empty:
            return pd.DataFrame()
        raise RuntimeError("All provided CSVs were empty or unreadable")

    out = pd.concat(dfs, ignore_index=True)
    return out


def _bin_series(x: pd.Series, edges: Sequence[float]) -> pd.Series:
    e = list(map(float, edges))
    if len(e) < 2:
        raise ValueError("bins must include at least two edges (e.g., 0 0.2 0.6 1.0)")
    if any(e[i] > e[i + 1] for i in range(len(e) - 1)):
        raise ValueError("bins edges must be increasing")
    b = pd.cut(pd.to_numeric(x, errors="coerce"), bins=e, include_lowest=True, right=True)
    return b.astype(str)


def _group_majority_label(df: pd.DataFrame, group_col: str, label_col: str) -> pd.Series:
    def mode_or_nan(s: pd.Series) -> Any:
        s2 = s.dropna()
        if s2.empty:
            return np.nan
        vc = s2.value_counts()
        m = vc.max()
        winners = sorted(vc[vc == m].index.tolist())
        return winners[0]

    return df.groupby(group_col, dropna=False)[label_col].apply(mode_or_nan)


def _group_mean_metric(df: pd.DataFrame, group_col: str, metric_col: str) -> pd.Series:
    x = pd.to_numeric(df[metric_col], errors="coerce")
    tmp = df.assign(__metric=x)
    return tmp.groupby(group_col, dropna=False)["__metric"].mean()


def _group_presence_frac(df: pd.DataFrame, group_col: str, metric_col: str, eps: float) -> pd.Series:
    x = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0)
    present = (x.to_numpy(dtype=float) > float(eps)).astype(np.float32)
    tmp = df.assign(__present=present)
    return tmp.groupby(group_col, dropna=False)["__present"].mean()


def _deterministic_shuffle(items: List[Any], rng: np.random.Generator) -> List[Any]:
    idx = np.arange(len(items))
    rng.shuffle(idx)
    return [items[i] for i in idx.tolist()]


def _assign_groups_random(
    groups: List[str],
    *,
    ratios: SplitRatios,
    seed: int,
) -> Dict[str, str]:
    """
    Deterministic random assignment of group ids to splits according to ratios.
    """
    rng = np.random.default_rng(int(seed))
    groups_shuf = _deterministic_shuffle(list(groups), rng)
    n = len(groups_shuf)

    n_train = int(round(n * ratios.train))
    n_val = int(round(n * ratios.val))

    # ensure train/val non-empty when feasible
    if n >= 2 and n_train == 0:
        n_train = 1
    if n >= 2 and n_val == 0:
        n_val = 1

    assignment: Dict[str, str] = {}
    for g in groups_shuf[:n_train]:
        assignment[g] = "train"
    for g in groups_shuf[n_train : n_train + n_val]:
        assignment[g] = "val"
    for g in groups_shuf[n_train + n_val :]:
        assignment[g] = "test" if ratios.test > 0 else "train"

    return assignment


def _assign_groups_stratified(
    groups_by_stratum: Mapping[str, List[str]],
    *,
    ratios: SplitRatios,
    seed: int,
) -> Dict[str, str]:
    assignment: Dict[str, str] = {}
    for stratum in sorted(groups_by_stratum.keys()):
        groups = list(groups_by_stratum[stratum])
        if not groups:
            continue
        sub_seed = int(seed) ^ (hash(stratum) & 0xFFFFFFFF)
        sub_assign = _assign_groups_random(groups, ratios=ratios, seed=sub_seed)
        assignment.update(sub_assign)
    return assignment


def materialize_splits(
    df: pd.DataFrame,
    group_col: str,
    group_assignment: Mapping[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    tmp = df.copy()
    if group_col not in tmp.columns:
        raise ValueError(f"group_col '{group_col}' not in dataframe")

    split = tmp[group_col].astype(str).map(group_assignment)
    tmp["__split"] = split

    train = tmp[tmp["__split"] == "train"].drop(columns=["__split"]).copy()
    val = tmp[tmp["__split"] == "val"].drop(columns=["__split"]).copy()
    test = tmp[tmp["__split"] == "test"].drop(columns=["__split"]).copy()

    return train, val, (None if test.empty else test)


# ======================================================================================
# Leakage checks
# ======================================================================================


def check_group_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: Optional[pd.DataFrame],
    *,
    group_col: str,
) -> LeakageCheckResult:
    if (
        group_col not in train.columns
        or group_col not in val.columns
        or (test is not None and group_col not in test.columns)
    ):
        return LeakageCheckResult(
            name="group_overlap",
            ok=False,
            details={"error": f"group_col '{group_col}' missing in one or more splits"},
        )

    def uniq(df: pd.DataFrame) -> set:
        return set(df[group_col].dropna().astype(str).tolist())

    a = uniq(train)
    b = uniq(val)
    c = uniq(test) if test is not None else set()

    ab = sorted(a.intersection(b))
    ac = sorted(a.intersection(c))
    bc = sorted(b.intersection(c))

    ok = (len(ab) == 0) and (len(ac) == 0) and (len(bc) == 0)
    return LeakageCheckResult(
        name="group_overlap",
        ok=ok,
        details={
            "group_col": group_col,
            "overlap_counts": {"train_val": len(ab), "train_test": len(ac), "val_test": len(bc)},
            "examples": {"train_val": ab[:20], "train_test": ac[:20], "val_test": bc[:20]},
        },
    )


def check_key_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: Optional[pd.DataFrame],
    *,
    key_cols: Sequence[str],
) -> LeakageCheckResult:
    key = list(key_cols)
    missing: List[str] = []
    for c in key:
        if c not in train.columns or c not in val.columns or (test is not None and c not in test.columns):
            missing.append(c)

    if missing:
        return LeakageCheckResult(
            name="dedupe_key_overlap",
            ok=True,
            details={"skipped": True, "reason": f"missing columns: {missing}", "key_cols": key},
        )

    def keys(df: pd.DataFrame) -> set:
        return set(df[key].astype(str).agg("|".join, axis=1).tolist())

    a = keys(train)
    b = keys(val)
    c = keys(test) if test is not None else set()

    ab = len(a.intersection(b))
    ac = len(a.intersection(c))
    bc = len(b.intersection(c))

    ok = (ab == 0) and (ac == 0) and (bc == 0)
    return LeakageCheckResult(
        name="dedupe_key_overlap",
        ok=ok,
        details={"key_cols": key, "overlap_counts": {"train_val": ab, "train_test": ac, "val_test": bc}},
    )


def check_image_src_overlap(train: pd.DataFrame, val: pd.DataFrame, test: Optional[pd.DataFrame]) -> LeakageCheckResult:
    if (
        "image_src" not in train.columns
        or "image_src" not in val.columns
        or (test is not None and "image_src" not in test.columns)
    ):
        return LeakageCheckResult(
            name="image_src_overlap", ok=True, details={"skipped": True, "reason": "image_src missing"}
        )

    def uniq(df: pd.DataFrame) -> set:
        return set(df["image_src"].dropna().astype(str).tolist())

    a = uniq(train)
    b = uniq(val)
    c = uniq(test) if test is not None else set()

    ab = len(a.intersection(b))
    ac = len(a.intersection(c))
    bc = len(b.intersection(c))

    ok = (ab == 0) and (ac == 0) and (bc == 0)
    return LeakageCheckResult(
        name="image_src_overlap",
        ok=ok,
        details={"overlap_counts": {"train_val": ab, "train_test": ac, "val_test": bc}},
    )


# ======================
# Split policies
# ======================


def _compute_group_stats_for_stratification(
    df: pd.DataFrame,
    *,
    group_col: str,
    mode: str,
    source_col: str,
    presence_eps: float,
    bins: Optional[Sequence[float]],
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Returns:
      - group_stats df (indexed by group id col)
      - strata label series indexed by group id (string)
      - strat_info dict for manifest
    """
    if source_col not in df.columns:
        raise ValueError(f"Requested group_metric_col '{source_col}' missing from dataframe")

    group_ids = df[group_col].astype(str)
    tmp = df.copy()
    tmp[group_col] = group_ids

    grp_size = tmp.groupby(group_col, dropna=False).size().rename("n_rows")

    strat_info: Dict[str, Any] = {
        "mode": mode,
        "group_col": group_col,
        "group_metric_col": source_col,
        "presence_eps": float(presence_eps) if mode == "presence_frac" else None,
        "bins": list(map(float, bins)) if bins else None,
    }

    if mode == "majority_label":
        metric = _group_majority_label(tmp, group_col, source_col).astype(str)
        strata = metric.fillna("NA").astype(str)
        group_stats = pd.DataFrame(
            {
                "group_id": metric.index.astype(str),
                "n_rows": grp_size.reindex(metric.index).values,
                "metric": metric.values,
            }
        )
        group_stats["stratum"] = strata.values
        return group_stats, strata, strat_info

    if mode == "mean":
        if not bins:
            raise ValueError("group_metric_mode='mean' requires bins for stratification")
        metric = _group_mean_metric(tmp, group_col, source_col)
        strata = _bin_series(metric, bins).fillna("NA").astype(str)
        group_stats = pd.DataFrame(
            {
                group_col: metric.index.astype(str),
                "n_rows": grp_size.reindex(metric.index).values,
                "metric": metric.values,
            }
        )
        group_stats["stratum"] = strata.values
        return group_stats, strata, strat_info

    if mode == "presence_frac":
        if not bins:
            raise ValueError("group_metric_mode='presence_frac' requires bins for stratification")
        metric = _group_presence_frac(tmp, group_col, source_col, presence_eps)
        strata = _bin_series(metric, bins).fillna("NA").astype(str)
        group_stats = pd.DataFrame(
            {
                group_col: metric.index.astype(str),
                "n_rows": grp_size.reindex(metric.index).values,
                "metric": metric.values,
            }
        )
        group_stats["stratum"] = strata.values
        return group_stats, strata, strat_info

    raise ValueError(f"Unknown group_metric_mode='{mode}' (expected mean|presence_frac|majority_label)")


def split_grouped(
    df: pd.DataFrame, *, config: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], List[str], str, Dict[str, str], pd.DataFrame]:
    warnings: List[str] = []
    group_col, w = resolve_group_col(df, preferred=config.group_col)
    warnings.extend(w)

    tmp = df.copy()
    if group_col == "__row_group__":
        tmp[group_col] = np.arange(len(tmp), dtype=np.int64)

    tmp[group_col] = tmp[group_col].astype(str)
    uniq_groups = sorted(tmp[group_col].unique().tolist())

    assignment = _assign_groups_random(uniq_groups, ratios=config.ratios, seed=config.seed)
    train, val, test = materialize_splits(tmp, group_col, assignment)

    # minimal group_stats for grouped split
    grp_size = tmp.groupby(group_col, dropna=False).size().rename("n_rows")
    group_stats = grp_size.reset_index().rename(columns={group_col: "group_id"})
    group_stats["split"] = group_stats["group_id"].astype(str).map(assignment)

    return train, val, test, warnings, group_col, assignment, group_stats


def split_predefined(
    df: pd.DataFrame, *, config: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], List[str], str, Dict[str, str], pd.DataFrame]:
    warnings: List[str] = []
    group_col, w = resolve_group_col(df, preferred=config.group_col)
    warnings.extend(w)

    tmp = df.copy()
    if group_col == "__row_group__":
        tmp[group_col] = np.arange(len(tmp), dtype=np.int64)
    tmp[group_col] = tmp[group_col].astype(str)

    col = config.predefined_col
    if col not in tmp.columns:
        warnings.append(
            f"[split] predefined policy requested but predefined_col='{col}' missing; falling back to grouped"
        )
        return split_grouped(tmp, config=config)

    labels = tmp[col]
    tmp["__predef"] = labels

    fixed = tmp[tmp["__predef"].isin(["train", "val", "test"])].copy()
    free = tmp[~tmp.index.isin(fixed.index)].copy()

    # assign unlabeled groups deterministically
    assignment: Dict[str, str] = {}
    if not free.empty:
        uniq_groups = sorted(free[group_col].unique().tolist())
        free_assign = _assign_groups_random(uniq_groups, ratios=config.ratios, seed=config.seed)
        assignment.update(free_assign)

    # fixed assignment from labels (group-consistency not guaranteed by user input; we enforce by majority)
    if not fixed.empty:
        fixed_group_label = fixed.groupby(group_col)["__predef"].apply(lambda s: s.value_counts().index[0])
        assignment.update({gid: lab for gid, lab in fixed_group_label.items()})

    # materialize with assignment; any groups still missing -> train
    tmp["__split"] = tmp[group_col].map(assignment).fillna("train")
    train = tmp[tmp["__split"] == "train"].drop(columns=["__split", "__predef"]).copy()
    val = tmp[tmp["__split"] == "val"].drop(columns=["__split", "__predef"]).copy()
    test_df = tmp[tmp["__split"] == "test"].drop(columns=["__split", "__predef"]).copy()
    test = None if (test_df.empty or config.ratios.test <= 0) else test_df

    if config.ratios.test <= 0 and not test_df.empty:
        warnings.append("[split] ratios.test=0 but predefined produced test rows; folding them into train")
        train = pd.concat([train, test_df], ignore_index=True)
        test = None

    grp_size = tmp.groupby(group_col, dropna=False).size().rename("n_rows")
    group_stats = grp_size.reset_index().rename(columns={group_col: "group_id"})
    group_stats["split"] = group_stats["group_id"].map(assignment).fillna("train")

    return train, val, test, warnings, group_col, assignment, group_stats


def split_stratified(
    df: pd.DataFrame, *, config: SplitConfig
) -> Tuple[
    pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], List[str], str, Dict[str, str], Dict[str, Any], pd.DataFrame
]:
    warnings: List[str] = []
    group_col, w = resolve_group_col(df, preferred=config.group_col)
    warnings.extend(w)

    tmp = df.copy()
    if group_col == "__row_group__":
        tmp[group_col] = np.arange(len(tmp), dtype=np.int64)
    tmp[group_col] = tmp[group_col].astype(str)

    mode = (config.group_metric_mode or "").strip()
    source_col = (config.group_metric_col or "").strip()

    if not mode or not source_col:
        warnings.append(
            "[split] stratified policy requested but group_metric_mode/col not set; falling back to grouped"
        )
        train, val, test, w2, gcol, assignment, group_stats = split_grouped(tmp, config=config)
        warnings.extend(w2)
        return (
            train,
            val,
            test,
            warnings,
            gcol,
            assignment,
            {"stratified": False, "reason": "missing_group_metric"},
            group_stats,
        )

    group_stats, strata_series, strat_info = _compute_group_stats_for_stratification(
        tmp,
        group_col=group_col,
        mode=mode,
        source_col=source_col,
        presence_eps=config.presence_eps,
        bins=config.bins,
    )

    # build mapping stratum -> groups
    groups_by_stratum: Dict[str, List[str]] = {}
    for gid, lab in strata_series.items():
        groups_by_stratum.setdefault(str(lab), []).append(str(gid))

    assignment = _assign_groups_stratified(groups_by_stratum, ratios=config.ratios, seed=config.seed)

    train, val, test = materialize_splits(tmp, group_col, assignment)

    # enrich group_stats with split assignment
    group_stats = group_stats.copy()
    # group_stats has a group_col column with group ids
    if group_col in group_stats.columns:
        group_stats["split"] = group_stats[group_col].astype(str).map(assignment)
    elif "group_id" in group_stats.columns:
        group_stats["split"] = group_stats["group_id"].astype(str).map(assignment)

    strat_out = {
        "stratified": True,
        **strat_info,
        "n_strata": int(len(groups_by_stratum)),
        "strata_counts": {k: int(len(v)) for k, v in sorted(groups_by_stratum.items())},
    }
    return train, val, test, warnings, group_col, assignment, strat_out, group_stats


# ======================================================================================
# Main orchestration API
# ======================================================================================


def make_splits(df: pd.DataFrame, *, config: SplitConfig) -> SplitResult:
    warnings: List[str] = []
    tmp = df.copy()

    tmp, w = _apply_min_any_ratio_filter(tmp, min_any_ratio=config.min_any_ratio, ratio_cols=config.ratio_cols)
    if w:
        warnings.append(w)

    tmp, w = _apply_dedupe(tmp, config.dedupe_key)
    if w:
        warnings.append(w)

    if config.policy == "grouped":
        train, val, test, w2, resolved_group_col, assignment, group_stats = split_grouped(tmp, config=config)
        warnings.extend(w2)

    elif config.policy == "predefined":
        train, val, test, w2, resolved_group_col, assignment, group_stats = split_predefined(tmp, config=config)
        warnings.extend(w2)

    elif config.policy == "stratified":
        train, val, test, w2, resolved_group_col, assignment, strat_info, group_stats = split_stratified(
            tmp, config=config
        )
        warnings.extend(w2)

    else:
        raise ValueError(f"Unknown policy '{config.policy}'. Expected: grouped|stratified|predefined")

    checks: List[LeakageCheckResult] = [
        check_group_leakage(train, val, test, group_col=resolved_group_col),
        check_image_src_overlap(train, val, test),
    ]
    if config.dedupe_key:
        checks.append(check_key_leakage(train, val, test, key_cols=config.dedupe_key))

    return SplitResult(
        train=train,
        val=val,
        test=test,
        checks=checks,
        warnings=warnings,
        resolved_group_col=resolved_group_col,
        group_assignment=assignment,
        group_stats=group_stats,
    )


def make_splits_from_csvs(inputs: Sequence[Union[str, Path]], *, config: SplitConfig) -> SplitResult:
    df = read_tiles_csvs(inputs)
    return make_splits(df, config=config)
