from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from geo_mlops.core.contracts.gate_contract import (
    GATE_SCHEMA_VERSION_V1,
    GateCheckResult,
    GateContract,
)


# -----------------------------
# Engine config / helpers
# -----------------------------
@dataclass(frozen=True)
class GateEngineConfig:
    """
    Minimal runtime behavior controls for gate evaluation.

    fail_on_missing:
        If True, missing metrics cause the corresponding check to fail.
        If False, missing metrics are skipped and recorded in warnings.

    allow_nan:
        If False, NaN actual values are treated as missing/invalid and fail.
    """
    fail_on_missing: bool = True
    allow_nan: bool = False


def _is_nan(value: Any) -> bool:
    try:
        return value != value
    except Exception:
        return False


def _normalize_op(op: str) -> str:
    op = str(op).strip()
    aliases = {
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
        "==": "==",
        "=": "==",
        "eq": "==",
        "ne": "!=",
        "!=": "!=",
        "ge": ">=",
        "gt": ">",
        "le": "<=",
        "lt": "<",
    }
    if op not in aliases:
        raise ValueError(
            f"Unsupported comparator '{op}'. "
            f"Allowed: {sorted(set(aliases.keys()))}"
        )
    return aliases[op]


def _compare(actual: float, comparator: str, threshold: float) -> bool:
    comparator = _normalize_op(comparator)

    if comparator == ">=":
        return actual >= threshold
    if comparator == ">":
        return actual > threshold
    if comparator == "<=":
        return actual <= threshold
    if comparator == "<":
        return actual < threshold
    if comparator == "==":
        return actual == threshold
    if comparator == "!=":
        return actual != threshold

    raise ValueError(f"Unhandled comparator: {comparator}")


def _metric_key(metric: str, scope: str) -> str:
    """
    Canonical in-memory metric lookup key.

    Examples:
      metric='iou', scope='micro'           -> 'micro/iou'
      metric='f1', scope='macro'            -> 'macro/f1'
      metric='iou', scope='class:building'  -> 'class:building/iou'
    """
    return f"{scope}/{metric}"


def flatten_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """
    Normalize common metric layouts into a flat dict:
        { "<scope>/<metric>": float }

    Supported layouts:

    1) Already flat:
       {
         "micro/iou": 0.74,
         "macro/f1": 0.71,
         "class:building/iou": 0.68,
       }

    2) Nested by scope:
       {
         "micro": {"iou": 0.74, "f1": 0.80},
         "macro": {"iou": 0.70},
         "class:building": {"iou": 0.68}
       }

    Non-numeric leaves are ignored.
    """
    flat: Dict[str, float] = {}

    for k, v in metrics.items():
        print(k, v)
        # already flat: "micro/iou": 0.74
        if isinstance(k, str) and "/" in k and isinstance(v, (int, float)):
            flat[k] = float(v)
            continue

        # nested scope dict: "micro": {"iou": 0.74}
        if isinstance(v, Mapping):
            scope = str(k)
            for metric_name, metric_val in v.items():
                if isinstance(metric_val, (int, float)):
                    flat[_metric_key(str(metric_name), scope)] = float(metric_val)

    return flat


def resolve_metric(
    metrics: Mapping[str, float],
    *,
    metric: str,
    scope: str,
) -> Optional[float]:
    """
    Resolve a metric from the flattened metric dict.
    """
    return metrics.get(_metric_key(metric=metric, scope=scope))


# -----------------------------
# Check evaluation
# -----------------------------
def evaluate_check(
    *,
    metrics: Mapping[str, float],
    metric: str,
    scope: str,
    comparator: str,
    threshold: float,
    config: GateEngineConfig,
) -> tuple[GateCheckResult, Optional[str]]:
    """
    Evaluate a single gate check.

    Returns:
      (GateCheckResult, warning_or_none)
    """
    actual = resolve_metric(metrics, metric=metric, scope=scope)

    if actual is None:
        warning = f"Missing metric for check: scope='{scope}', metric='{metric}'"
        return (
            GateCheckResult(
                metric=metric,
                scope=scope,
                comparator=_normalize_op(comparator),
                threshold=float(threshold),
                actual=float("nan"),
                passed=(False if config.fail_on_missing else True),
            ),
            warning,
        )

    if _is_nan(actual) and not config.allow_nan:
        warning = f"NaN metric for check: scope='{scope}', metric='{metric}'"
        return (
            GateCheckResult(
                metric=metric,
                scope=scope,
                comparator=_normalize_op(comparator),
                threshold=float(threshold),
                actual=float(actual),
                passed=False,
            ),
            warning,
        )

    passed = _compare(float(actual), comparator, float(threshold))
    return (
        GateCheckResult(
            metric=metric,
            scope=scope,
            comparator=_normalize_op(comparator),
            threshold=float(threshold),
            actual=float(actual),
            passed=bool(passed),
        ),
        None,
    )


def evaluate_gate_checks(
    *,
    metrics: Mapping[str, Any],
    checks_spec: Iterable[Mapping[str, Any]],
    config: GateEngineConfig,
) -> tuple[List[GateCheckResult], List[str]]:
    """
    Evaluate all checks against a metrics mapping.

    checks_spec entries are expected to have:
      - metric
      - scope
      - comparator (or 'op')
      - threshold
    """
    flat_metrics = flatten_metrics(metrics)
    # print(flat_metrics)
    results: List[GateCheckResult] = []
    warnings: List[str] = []

    for idx, spec in enumerate(checks_spec):
        if not isinstance(spec, Mapping):
            raise TypeError(f"Check spec at index {idx} must be a mapping, got {type(spec)}")

        metric = str(spec["metric"])
        scope = str(spec["scope"])
        comparator = str(spec.get("comparator", spec.get("op", ">=")))
        threshold = float(spec["threshold"])

        result, warning = evaluate_check(
            metrics=flat_metrics,
            metric=metric,
            scope=scope,
            comparator=comparator,
            threshold=threshold,
            config=config,
        )
        results.append(result)

        if warning is not None:
            warnings.append(warning)

    return results, warnings


def summarize_results(results: Iterable[GateCheckResult]) -> Dict[str, Any]:
    results = list(results)
    total_checks = len(results)
    passed_checks = sum(1 for r in results if r.passed)
    failed_checks = total_checks - passed_checks

    failed_metrics = [
        {
            "metric": r.metric,
            "scope": r.scope,
            "actual": r.actual,
            "threshold": r.threshold,
            "comparator": r.comparator,
        }
        for r in results
        if not r.passed
    ]

    return {
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "failed_metrics": failed_metrics,
    }


# -----------------------------
# Main engine entrypoint
# -----------------------------
def run_gate(
    *,
    gate_dir: str | Path,
    gate_name: str,
    task: str,
    metrics: Mapping[str, Any],
    threshold_spec: Mapping[str, Any],
    upstream: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    schema_version: str = GATE_SCHEMA_VERSION_V1,
) -> GateContract:
    """
    Build a GateContract from metrics + threshold spec.

    Expected threshold_spec shape:
    {
      "fail_on_missing": true,
      "allow_nan": false,
      "checks": [
        {"metric": "iou", "scope": "micro", "comparator": ">=", "threshold": 0.72},
        {"metric": "iou", "scope": "macro", "comparator": ">=", "threshold": 0.70},
      ]
    }
    """
    if not isinstance(threshold_spec, Mapping):
        raise TypeError(f"threshold_spec must be a mapping, got {type(threshold_spec)}")

    checks_spec = threshold_spec.get("checks")
    if not isinstance(checks_spec, list) or not checks_spec:
        raise ValueError("threshold_spec['checks'] must be a non-empty list")

    config = GateEngineConfig(
        fail_on_missing=bool(threshold_spec.get("fail_on_missing", True)),
        allow_nan=bool(threshold_spec.get("allow_nan", False)),
    )

    check_results, warnings = evaluate_gate_checks(
        metrics=metrics,
        checks_spec=checks_spec,
        config=config,
    )

    summary = summarize_results(check_results)
    if warnings:
        summary["warnings"] = warnings

    passed = all(r.passed for r in check_results)
    decision = "pass" if passed else "fail"

    meta_out: Dict[str, Any] = dict(meta or {})
    if warnings:
        meta_out.setdefault("warnings", warnings)

    contract = GateContract(
        gate_dir=Path(gate_dir),
        schema_version=schema_version,
        gate_name=str(gate_name),
        task=str(task),
        decision=decision,
        passed=passed,
        checks=[
            {
                "metric": r.metric,
                "scope": r.scope,
                "comparator": r.comparator,
                "threshold": r.threshold,
                "actual": r.actual,
                "passed": r.passed,
            }
            for r in check_results
        ],
        summary=summary,
        upstream=dict(upstream or {}),
        threshold_spec=dict(threshold_spec),
        meta=meta_out,
    )
    return contract