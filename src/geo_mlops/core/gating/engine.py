from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from geo_mlops.core.contracts.gate_contract import (
    GATE_SCHEMA_VERSION_V1,
    GateCheckResult,
    GateContract,
)


@dataclass(frozen=True)
class GateEngineConfig:
    """
    Runtime behavior controls for gate evaluation.

    fail_on_missing:
        If True, missing metrics cause the corresponding check to fail.
        If False, missing metrics are skipped as pass=True but still warned.

    allow_nan:
        If False, NaN actual values are treated as invalid and fail.
    """

    fail_on_missing: bool = True
    allow_nan: bool = False


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_nan(value: Any) -> bool:
    try:
        return value != value
    except Exception:
        return False


def _normalize_op(op: str) -> str:
    op = str(op).strip().lower()
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
            f"Unsupported comparator {op!r}. Allowed: {sorted(set(aliases.keys()))}"
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


def _metric_key(metric: str, scope: Optional[str]) -> str:
    metric = str(metric)
    if scope is None or str(scope).strip() == "":
        return metric
    return f"{scope}/{metric}"


def _add_metric(
    flat: Dict[str, float],
    *,
    key: str,
    value: Any,
) -> None:
    if _is_number(value):
        flat[str(key)] = float(value)


def _flatten_plain_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """
    Normalize common metric layouts into a flat dict.

    Supported:
      1. {"val/micro_f1": 0.5}
      2. {"val": {"micro_f1": 0.5}}
      3. {"micro_f1": 0.5}
    """
    flat: Dict[str, float] = {}

    for key, value in metrics.items():
        key = str(key)

        if _is_number(value):
            _add_metric(flat, key=key, value=value)
            continue

        if isinstance(value, Mapping):
            scope = key
            for metric_name, metric_value in value.items():
                if _is_number(metric_value):
                    _add_metric(
                        flat,
                        key=_metric_key(metric=str(metric_name), scope=scope),
                        value=metric_value,
                    )

    return flat


def _extract_best_epoch_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """
    Extract best-epoch metrics from a training metrics.json payload.

    Expected shape:
      {
        "best_epoch": 3,
        "history": {
          "epoch_3": {
            "train/loss": ...,
            "val/micro_f1": ...
          }
        }
      }
    """
    history = metrics.get("history")
    if not isinstance(history, Mapping):
        return {}

    best_epoch = metrics.get("best_epoch")
    if best_epoch is None:
        return {}

    epoch_key = f"epoch_{int(best_epoch)}"
    epoch_metrics = history.get(epoch_key)

    if not isinstance(epoch_metrics, Mapping):
        return {}

    flat = _flatten_plain_metrics(epoch_metrics)

    # Preserve useful top-level selectors too.
    if _is_number(metrics.get("best_metric_value")):
        flat["best_metric_value"] = float(metrics["best_metric_value"])

    selection_metric = metrics.get("selection_metric")
    if isinstance(selection_metric, str) and _is_number(metrics.get("best_metric_value")):
        # This lets users gate directly on metric: best_metric_value
        # or on the actual selected metric name, e.g. val/micro_f1.
        flat.setdefault(selection_metric, float(metrics["best_metric_value"]))

    return flat


def flatten_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """
    Normalize metrics into flat lookup keys.

    Supports:
      - Direct flat metrics:
          {"val/micro_f1": 0.42}

      - Nested metrics:
          {"val": {"micro_f1": 0.42}}

      - Training metrics payload:
          {
            "best_epoch": 3,
            "history": {
              "epoch_3": {"val/micro_f1": 0.42}
            }
          }

    Output:
      {"val/micro_f1": 0.42, ...}
    """
    flat: Dict[str, float] = {}

    # First include any direct/nested numeric metrics present at root.
    flat.update(_flatten_plain_metrics(metrics))

    # Then, if this is a training metrics payload, expose best epoch metrics.
    # Best-epoch values should win over top-level duplicate keys.
    flat.update(_extract_best_epoch_metrics(metrics))

    return flat


def resolve_metric(
    metrics: Mapping[str, float],
    *,
    metric: str,
    scope: Optional[str] = None,
) -> Optional[float]:
    """
    Resolve a metric from flattened metrics.

    Supported check styles:
      - scope: val, metric: micro_f1       -> val/micro_f1
      - metric: val/micro_f1, no scope     -> val/micro_f1
      - metric: best_metric_value          -> best_metric_value
    """
    metric = str(metric)

    candidate_keys: List[str] = []

    if scope is not None and str(scope).strip():
        candidate_keys.append(_metric_key(metric=metric, scope=str(scope)))

    candidate_keys.append(metric)

    for key in candidate_keys:
        if key in metrics:
            return metrics[key]

    return None


def evaluate_check(
    *,
    metrics: Mapping[str, float],
    metric: str,
    scope: Optional[str],
    comparator: str,
    threshold: float,
    config: GateEngineConfig,
) -> tuple[GateCheckResult, Optional[str]]:
    actual = resolve_metric(metrics, metric=metric, scope=scope)

    scope_for_contract = str(scope) if scope is not None else ""

    if actual is None:
        warning = f"Missing metric for check: scope={scope_for_contract!r}, metric={metric!r}"
        return (
            GateCheckResult(
                metric=metric,
                scope=scope_for_contract,
                comparator=_normalize_op(comparator),
                threshold=float(threshold),
                actual=float("nan"),
                passed=(False if config.fail_on_missing else True),
            ),
            warning,
        )

    if _is_nan(actual) and not config.allow_nan:
        warning = f"NaN metric for check: scope={scope_for_contract!r}, metric={metric!r}"
        return (
            GateCheckResult(
                metric=metric,
                scope=scope_for_contract,
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
            scope=scope_for_contract,
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
    flat_metrics = flatten_metrics(metrics)

    results: List[GateCheckResult] = []
    warnings: List[str] = []

    for idx, spec in enumerate(checks_spec):
        if not isinstance(spec, Mapping):
            raise TypeError(f"Check spec at index {idx} must be a mapping, got {type(spec)}")

        if "metric" not in spec:
            raise ValueError(f"Check spec at index {idx} is missing required field 'metric'")
        if "threshold" not in spec:
            raise ValueError(f"Check spec at index {idx} is missing required field 'threshold'")

        metric = str(spec["metric"])
        scope = spec.get("scope", None)
        scope = None if scope is None else str(scope)

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

    Recommended threshold_spec shape:
      {
        "fail_on_missing": true,
        "allow_nan": false,
        "checks": [
          {"scope": "val", "metric": "micro_f1", "comparator": ">=", "threshold": 0.40}
        ]
      }

    Also supported:
      {"metric": "val/micro_f1", "comparator": ">=", "threshold": 0.40}
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

    flat_metrics = flatten_metrics(metrics)
    summary["resolved_metrics"] = {
        "available_keys": sorted(flat_metrics.keys()),
    }

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