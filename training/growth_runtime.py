"""Fail-closed runtime loading for a parity-approved V4 growth child."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from training.checkpoint_durability import FULL_RESUME
from training.csii import (
    CrossScaleIdentityInheritance,
    GrowthAlignmentController,
    model_architecture_sha256,
)
from training.growth_contract import build_growth_parent_lineage
from training.v2_config import (
    ANRA_V4_GROWTH_MODEL_PROFILE,
    CANONICAL_MODEL_PROFILE,
    CHECKPOINT_SCHEMA_VERSION,
)
from training.v2_runtime import build_model_for_profile


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weights_only_load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Growth artifact must contain a dictionary: {path}")
    return payload


def _model_state(payload: Mapping[str, object], *, label: str) -> dict[str, torch.Tensor]:
    raw = payload.get("model_state_dict", payload.get("model"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{label} has no model state")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in raw.items()
    ):
        raise TypeError(f"{label} model state contains a non-tensor entry")
    return raw


def _load_exact(model: object, state: dict[str, torch.Tensor], *, label: str) -> None:
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"{label} did not load exactly: {incompatible}")


def _parent_progress(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "continuation_token_counts": dict(payload.get("continuation_token_counts", {})),
        "raw_window_consumption": dict(payload.get("raw_window_consumption", {})),
        "data_sampler_state": dict(payload.get("data_sampler_state", {})),
        "data_profile": str(payload.get("data_profile", "unknown")),
        "training_data_layout": str(payload.get("training_data_layout", "unknown")),
        "seed_contract": dict(payload.get("seed_contract", {})),
        "data_manifests": dict(
            payload.get("data_manifests", payload.get("dataset_manifest_hashes", {}))
        ),
        "best_validation_loss": float(payload.get("best_validation_loss", float("inf"))),
        "best_answer_validation_loss": float(
            payload.get("best_answer_validation_loss", float("inf"))
        ),
        "validation_history": list(payload.get("validation_history", [])),
    }


def load_growth_training_pair(
    target: object,
    *,
    initialization_path: str | Path,
    growth_manifest_path: str | Path,
    parent_checkpoint_path: str | Path,
) -> tuple[object, GrowthAlignmentController, dict[str, object]]:
    """Load child+teacher while proving hashes, architecture and fresh optimizer semantics."""

    initialization = Path(initialization_path).resolve()
    report_path = Path(growth_manifest_path).resolve()
    parent_path = Path(parent_checkpoint_path).resolve()
    for label, path in (
        ("growth initialization", initialization),
        ("growth manifest", report_path),
        ("growth parent", parent_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    report_raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report_raw, dict):
        raise ValueError("Growth manifest must contain a JSON object")
    report = CrossScaleIdentityInheritance.validate_growth_report(report_raw)
    if report["source_profile"] != CANONICAL_MODEL_PROFILE:
        raise ValueError("Growth runtime requires the canonical 181M parent")
    if report["target_profile"] != ANRA_V4_GROWTH_MODEL_PROFILE:
        raise ValueError("Growth runtime requires the registered 500M child")
    parent_sha256 = sha256_file(parent_path)
    if parent_sha256 != report["source_checkpoint_sha256"]:
        raise ValueError("Growth teacher checkpoint hash does not match the manifest")

    metadata_path = initialization.with_suffix(initialization.suffix + ".meta.json")
    if not metadata_path.is_file():
        raise FileNotFoundError("Growth initialization metadata sidecar is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    initialization_sha256 = sha256_file(initialization)
    expected_metadata = {
        "schema": "anra-growth-initialization/v1",
        "artifact_class": "growth_initialization",
        "artifact_sha256": initialization_sha256,
        "growth_manifest_sha256": sha256_file(report_path),
        "source_checkpoint_sha256": parent_sha256,
        "source_profile": CANONICAL_MODEL_PROFILE,
        "target_profile": ANRA_V4_GROWTH_MODEL_PROFILE,
        "optimizer_restart_required": True,
        "optimizer_state_inherited": False,
        "training_resume_allowed": False,
    }
    if metadata != expected_metadata:
        raise ValueError("Growth initialization metadata does not match its artifacts")

    CrossScaleIdentityInheritance.apply_attention_mode_mapping(target, report)
    if model_architecture_sha256(target) != report["target_architecture_sha256"]:
        raise ValueError("Active child architecture does not match the growth manifest")
    child_payload = _weights_only_load(initialization)
    if (
        child_payload.get("artifact_class") != "growth_initialization"
        or child_payload.get("training_resume_allowed") is not False
        or child_payload.get("optimizer_restart_required") is not True
        or child_payload.get("optimizer_state_inherited") is not False
        or child_payload.get("model_profile") != ANRA_V4_GROWTH_MODEL_PROFILE
    ):
        raise ValueError("Growth initialization artifact has invalid optimizer semantics")
    embedded_report = child_payload.get("growth_manifest")
    if not isinstance(embedded_report, dict):
        raise ValueError("Growth initialization does not embed its growth manifest")
    embedded_validated = CrossScaleIdentityInheritance.validate_growth_report(embedded_report)
    if json.dumps(embedded_validated, sort_keys=True) != json.dumps(report, sort_keys=True):
        raise ValueError("Growth initialization embeds a different growth manifest")
    _load_exact(target, _model_state(child_payload, label="growth initialization"), label="child")

    embedded_parent_lineage = child_payload.get("parent_lineage")
    if not isinstance(embedded_parent_lineage, dict):
        raise ValueError("Growth initialization has no immutable parent_lineage")
    parent_payload = _weights_only_load(parent_path)
    expected_parent_lineage = build_growth_parent_lineage(
        parent_payload,
        checkpoint_sha256=parent_sha256,
        parent_stage_policy=str(embedded_parent_lineage.get("parent_stage_policy", "")),
    )
    if json.dumps(embedded_parent_lineage, sort_keys=True) != json.dumps(
        expected_parent_lineage, sort_keys=True
    ):
        raise ValueError("Growth initialization parent lineage differs from its parent checkpoint")

    source, controller, provenance = load_growth_teacher(
        target,
        growth_manifest_path=report_path,
        parent_checkpoint_path=parent_path,
    )
    if child_payload.get("parent_progress") != provenance["parent_progress"]:
        raise ValueError("Growth initialization parent cursor differs from its parent checkpoint")
    return source, controller, {
        **provenance,
        "initialization_sha256": initialization_sha256,
        "parent_lineage": expected_parent_lineage,
    }


def load_growth_teacher(
    target: object,
    *,
    growth_manifest_path: str | Path,
    parent_checkpoint_path: str | Path,
) -> tuple[object, GrowthAlignmentController, dict[str, object]]:
    """Validate the growth architecture and load its immutable 181M teacher."""

    report_path = Path(growth_manifest_path).resolve()
    parent_path = Path(parent_checkpoint_path).resolve()
    if not report_path.is_file() or not parent_path.is_file():
        raise FileNotFoundError("Growth manifest and parent checkpoint are required")
    report_raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report_raw, dict):
        raise ValueError("Growth manifest must contain a JSON object")
    report = CrossScaleIdentityInheritance.validate_growth_report(report_raw)
    if (
        report["source_profile"] != CANONICAL_MODEL_PROFILE
        or report["target_profile"] != ANRA_V4_GROWTH_MODEL_PROFILE
    ):
        raise ValueError("Growth manifest does not describe the registered V4 lineage")
    parent_sha256 = sha256_file(parent_path)
    if parent_sha256 != report["source_checkpoint_sha256"]:
        raise ValueError("Growth teacher checkpoint hash does not match the manifest")
    CrossScaleIdentityInheritance.apply_attention_mode_mapping(target, report)
    if model_architecture_sha256(target) != report["target_architecture_sha256"]:
        raise ValueError("Active child architecture does not match the growth manifest")

    parent_payload = _weights_only_load(parent_path)
    if (
        parent_payload.get("checkpoint_artifact_class") != FULL_RESUME
        or int(parent_payload.get("checkpoint_schema_version", 0)) != CHECKPOINT_SCHEMA_VERSION
        or parent_payload.get("completed_optimizer_boundary") is not True
    ):
        raise ValueError("Growth teacher must be a complete schema-9 full-resume checkpoint")
    source = build_model_for_profile(CANONICAL_MODEL_PROFILE)
    _load_exact(source, _model_state(parent_payload, label="growth parent"), label="parent")
    if model_architecture_sha256(source) != report["source_architecture_sha256"]:
        raise ValueError("Active parent architecture does not match the growth manifest")

    identity_layers = tuple(int(value) for value in report["identity_layers"])
    controller = GrowthAlignmentController(
        source,
        target,
        identity_layers=identity_layers,
    )
    return source, controller, {
        "schema": "anra-growth-runtime/v1",
        "growth_manifest_sha256": sha256_file(report_path),
        "parent_checkpoint_sha256": parent_sha256,
        "identity_layers": list(identity_layers),
        "optimizer_restart_required": True,
        "alignment_steps": controller.alignment_steps,
        "new_only_steps": controller.new_only_steps,
        "parent_progress": _parent_progress(parent_payload),
    }
