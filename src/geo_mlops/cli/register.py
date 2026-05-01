from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from geo_mlops.core.config.loader import load_cfg
from geo_mlops.core.registry.model_registry import (
    promote_model_to_production,
    register_candidate_model,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Register or promote MLflow model versions based on gate contracts."
    )

    p.add_argument("--task", type=str, required=True, help="Task name, e.g. building_seg.")
    p.add_argument(
        "--task-cfg",
        "--task_cfg",
        dest="task_cfg",
        type=Path,
        required=True,
        help="Unified task config containing registry settings.",
    )
    p.add_argument(
        "--action",
        required=True,
        choices=("register-candidate", "promote-production"),
        help="Registry transition to perform.",
    )
    p.add_argument(
        "--gate-contract",
        type=Path,
        required=True,
        help="Path to gate contract JSON. Must have passed=true.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory for registry_result.json.",
    )

    p.add_argument("--mlflow-tracking-uri", type=str, default=None)
    p.add_argument("--mlflow-registry-uri", type=str, default=None)

    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Registered model name. Overrides registry.model_name in task config.",
    )
    p.add_argument(
        "--model-artifact-path",
        type=str,
        default=None,
        help="MLflow model artifact path within the run.",
    )

    # Optional for register-candidate; inferred from train manifest when omitted.
    p.add_argument(
        "--mlflow-run-id",
        type=str,
        default=None,
        help="Training MLflow run ID containing the logged model artifact.",
    )
    p.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help=(
            "Optional train_manifest.json. If omitted, register.py attempts to read "
            "gate_contract.upstream.train_manifest."
        ),
    )

    # Required for promote-production.
    p.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Registered model version to promote.",
    )

    return p


def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")

    obj = json.loads(p.read_text())

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at root of {p}")

    return obj


def _registry_cfg(task_cfg_path: Path) -> Dict[str, Any]:
    cfg = load_cfg(task_cfg_path)

    if not isinstance(cfg, dict):
        raise ValueError(f"Task config root must be a mapping: {task_cfg_path}")

    reg = cfg.get("registry", {}) or {}

    if not isinstance(reg, dict):
        raise ValueError("registry section must be a mapping if provided.")

    return reg


def _resolve_model_name(
    *,
    task: str,
    cli_model_name: Optional[str],
    reg_cfg: Dict[str, Any],
) -> str:
    if cli_model_name:
        return cli_model_name

    if reg_cfg.get("model_name"):
        return str(reg_cfg["model_name"])

    return task


def _resolve_train_manifest_path(
    *,
    cli_train_manifest: Optional[Path],
    gate_contract: Dict[str, Any],
) -> Optional[Path]:
    if cli_train_manifest is not None:
        return cli_train_manifest

    upstream = gate_contract.get("upstream", {}) or {}
    if not isinstance(upstream, dict):
        return None

    train_manifest = upstream.get("train_manifest")
    if train_manifest:
        return Path(str(train_manifest))

    return None


def _resolve_tracking_from_manifest(
    train_manifest_path: Optional[Path],
) -> Dict[str, Any]:
    if train_manifest_path is None:
        return {}

    manifest = _load_json(train_manifest_path)

    tracking = manifest.get("tracking", {}) or {}
    if not isinstance(tracking, dict):
        raise ValueError(f"train_manifest tracking field must be a mapping: {train_manifest_path}")

    return tracking


def _resolve_required_run_id(
    *,
    cli_run_id: Optional[str],
    tracking: Dict[str, Any],
) -> str:
    if cli_run_id:
        return cli_run_id

    run_id = tracking.get("mlflow_run_id")
    if run_id:
        return str(run_id)

    raise ValueError(
        "Could not resolve MLflow run ID. Provide --mlflow-run-id, or ensure "
        "train_manifest.json contains tracking.mlflow_run_id and the gate contract "
        "contains upstream.train_manifest."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    reg_cfg = _registry_cfg(args.task_cfg)
    gate_contract = _load_json(args.gate_contract)

    model_name = _resolve_model_name(
        task=args.task,
        cli_model_name=args.model_name,
        reg_cfg=reg_cfg,
    )

    train_manifest_path = _resolve_train_manifest_path(
        cli_train_manifest=args.train_manifest,
        gate_contract=gate_contract,
    )
    tracking = _resolve_tracking_from_manifest(train_manifest_path)

    model_artifact_path = (
        args.model_artifact_path
        or tracking.get("mlflow_model_artifact_path")
        or reg_cfg.get("model_artifact_path")
        or "model"
    )
    model_artifact_path = str(model_artifact_path)

    tracking_uri = (
        args.mlflow_tracking_uri
        or tracking.get("mlflow_tracking_uri")
        or reg_cfg.get("tracking_uri")
    )
    registry_uri = args.mlflow_registry_uri or reg_cfg.get("registry_uri")

    if args.action == "register-candidate":
        mlflow_run_id = _resolve_required_run_id(
            cli_run_id=args.mlflow_run_id,
            tracking=tracking,
        )

        result = register_candidate_model(
            model_name=model_name,
            mlflow_run_id=mlflow_run_id,
            gate_contract_path=args.gate_contract,
            model_artifact_path=model_artifact_path,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            candidate_alias=str(reg_cfg.get("candidate_alias", "candidate")),
            out_dir=args.out_dir,
            extra_tags={
                "task": args.task,
                "task_cfg": str(args.task_cfg),
                "train_manifest": str(train_manifest_path) if train_manifest_path else "",
            },
        )

        print("[registry] registered candidate")
        print(f"[registry] model={result.model_name}")
        print(f"[registry] version={result.model_version}")
        print(f"[registry] uri={result.model_uri}")
        return 0

    if args.action == "promote-production":
        if not args.model_version:
            raise ValueError("--model-version is required for --action promote-production")

        result = promote_model_to_production(
            model_name=model_name,
            model_version=args.model_version,
            gate_contract_path=args.gate_contract,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            production_alias=str(reg_cfg.get("production_alias", "production")),
            archive_candidate_alias=bool(reg_cfg.get("archive_candidate_alias", False)),
            out_dir=args.out_dir,
            extra_tags={
                "task": args.task,
                "task_cfg": str(args.task_cfg),
            },
        )

        print("[registry] promoted model to production")
        print(f"[registry] model={result.model_name}")
        print(f"[registry] version={result.model_version}")
        return 0

    raise ValueError(f"Unhandled action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())