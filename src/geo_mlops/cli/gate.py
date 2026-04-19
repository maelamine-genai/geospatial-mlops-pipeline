from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from geo_mlops.core.gating.engine import run_gate
from geo_mlops.core.io.gate_io import summarize_gate_contract, write_gate_contract


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a gating stage against a metrics file and write a canonical gate contract."
    )

    p.add_argument("--task", required=True, help="Task name, e.g. building_seg")
    p.add_argument("--gate-name", required=True, help="Gate name, e.g. gate_a or gate_b")

    p.add_argument(
        "--gate-config",
        required=True,
        help="Path to gate config YAML/JSON containing threshold spec.",
    )
    p.add_argument(
        "--metrics-file",
        required=True,
        help="Path to metrics JSON/YAML file to evaluate.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for gate artifacts.",
    )

    # optional lineage inputs
    p.add_argument("--train-manifest", default=None, help="Optional upstream train_manifest.json")
    p.add_argument("--eval-manifest", default=None, help="Optional upstream eval_manifest.json")
    p.add_argument("--split-json", default=None, help="Optional upstream split.json")
    p.add_argument("--tiles-manifest", default=None, help="Optional upstream tiles_manifest.json")

    # optional extra metadata
    p.add_argument(
        "--meta-json",
        default=None,
        help="Optional path to extra metadata JSON to embed in gate contract.",
    )

    return p


def _load_structured_file(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix == ".json":
        obj = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        obj = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported file extension '{suffix}' for {p}. Use .json/.yaml/.yml")

    if not isinstance(obj, dict):
        raise ValueError(f"Expected mapping/object at root of {p}, got {type(obj)}")
    return obj


def _extract_threshold_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either:
      1) raw threshold spec at root:
         {fail_on_missing: true, checks: [...]}

      2) nested under 'gate':
         {gate: {fail_on_missing: true, checks: [...]}}

    This keeps the CLI forgiving while the config format stabilizes.
    """
    if "gate" in cfg:
        gate_spec = cfg["gate"]
        if not isinstance(gate_spec, dict):
            raise ValueError("Expected 'gate' section in config to be a mapping")
        return gate_spec
    return cfg


def _build_upstream_dict(args: argparse.Namespace) -> Dict[str, Any]:
    upstream: Dict[str, Any] = {}

    if args.train_manifest:
        upstream["train_manifest"] = str(Path(args.train_manifest))
    if args.eval_manifest:
        upstream["eval_manifest"] = str(Path(args.eval_manifest))
    if args.split_json:
        upstream["split_json"] = str(Path(args.split_json))
    if args.tiles_manifest:
        upstream["tiles_manifest"] = str(Path(args.tiles_manifest))

    return upstream


def _load_optional_meta(meta_json_path: Optional[str]) -> Dict[str, Any]:
    if not meta_json_path:
        return {}
    return _load_structured_file(meta_json_path)


def _print_summary(summary: Dict[str, Any]) -> None:
    print(
        f"[gate] {summary['gate_name']} | task={summary['task']} | "
        f"decision={summary['decision']} | passed={summary['passed']}"
    )
    print(
        f"[gate] checks: total={summary['total_checks']} "
        f"passed={summary['passed_checks']} failed={summary['failed_checks']}"
    )
    print(f"[gate] out_dir: {summary['gate_dir']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_cfg = _load_structured_file(args.gate_config)
    threshold_spec = _extract_threshold_spec(gate_cfg)

    metrics = _load_structured_file(args.metrics_file)
    upstream = _build_upstream_dict(args)
    meta = _load_optional_meta(args.meta_json)

    print("XXX")

    contract = run_gate(
        gate_dir=out_dir,
        gate_name=args.gate_name,
        task=args.task,
        metrics=metrics,
        threshold_spec=threshold_spec,
        upstream=upstream,
        meta=meta,
    )

    contract_path = write_gate_contract(contract)
    summary = summarize_gate_contract(contract)

    _print_summary(summary)
    print(f"[gate] contract: {contract_path}")

    if not contract.passed:
        print("[gate] result: FAILED")
        return 1

    print("[gate] result: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())