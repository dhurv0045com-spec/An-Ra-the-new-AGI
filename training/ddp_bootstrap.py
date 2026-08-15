"""Signed, model-only migration from a single-GPU V4 lineage into canonical DDP.

This is deliberately not an exact resume.  The learned tensors and auditable
progress are inherited, while optimizer, scheduler, scaler, and RNG topology
are restarted under a new child lineage.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from training.checkpoint_durability import FULL_RESUME, build_checkpoint_lineage, sha256_file

BOOTSTRAP_SCHEMA = "anra-single-gpu-to-ddp-bootstrap/v1"
SAMPLER_POLICIES = frozenset(
    {
        "preserve_global_cursor_repartition_by_rank_v1",
        "reset_for_new_signed_data_window_v1",
    }
)
RESTART_CONTRACT = {
    "optimizer": "restart_adamw_from_empty_state",
    "scheduler": "restart_canonical_cosine_from_step_zero",
    "scaler": "restart_for_active_precision_backend",
    "rng": "fresh_rank_seeded_streams",
    "forbidden_parent_state": ["optimizer", "scheduler", "scaler", "rng_states"],
}
PRESERVED_PROGRESS_FIELDS = (
    "global_step",
    "tokens_seen",
    "sessions_completed",
    "continuation_token_counts",
    "raw_window_consumption",
    "data_sampler_state",
    "unique_token_ids_seen",
    "best_loss",
    "best_validation_loss",
    "best_answer_validation_loss",
    "validation_history",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def current_source_commit(root: Path) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("DDP bootstrap requires a resolvable source commit") from exc
    if len(commit) != 40:
        raise RuntimeError("DDP bootstrap source commit is not a full Git SHA")
    return commit


def file_bindings(paths: Mapping[str, str | Path]) -> dict[str, dict[str, object]]:
    bindings: dict[str, dict[str, object]] = {}
    for role, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"DDP bootstrap data manifest is missing: {path}")
        bindings[str(role)] = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not bindings:
        raise ValueError("DDP bootstrap requires at least one immutable data manifest")
    return bindings


def _load_weights_only_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - old torch is unsupported operationally
        raise RuntimeError("DDP bootstrap requires torch.load(weights_only=True) support") from exc
    if not isinstance(payload, dict):
        raise TypeError("DDP bootstrap parent must be a structured checkpoint")
    return payload


def _validated_parent(path: Path) -> tuple[dict[str, Any], dict[str, object]]:
    payload = _load_weights_only_checkpoint(path)
    if int(payload.get("checkpoint_schema_version", 0) or 0) != 9:
        raise RuntimeError("DDP bootstrap parent must use checkpoint schema 9")
    if payload.get("checkpoint_artifact_class") != FULL_RESUME:
        raise RuntimeError("DDP bootstrap parent must be a full_resume artifact")
    if payload.get("completed_optimizer_boundary") is not True:
        raise RuntimeError("DDP bootstrap parent is not on a completed optimizer boundary")
    lineage = payload.get("checkpoint_lineage")
    if not isinstance(lineage, Mapping):
        lineage = build_checkpoint_lineage(payload)
    lineage = dict(lineage)
    parent_lineage_id = str(lineage.get("lineage_id", "")).strip()
    if not parent_lineage_id:
        raise RuntimeError("DDP bootstrap parent has no lineage identity")
    source_commit = str(payload.get("source_commit", ""))
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise RuntimeError("DDP bootstrap parent has no verifiable source commit")
    if str(lineage.get("source_commit", "")) != source_commit:
        raise RuntimeError("DDP bootstrap parent source commit disagrees with its lineage")
    for component in ("optimizer", "scheduler", "scaler", "rng_states"):
        if component not in payload:
            raise RuntimeError(f"DDP bootstrap parent full_resume lacks {component}")
    if dict(lineage.get("architecture", {})).get("config") != payload.get("model_config"):
        raise RuntimeError("DDP bootstrap parent architecture disagrees with its lineage")
    if lineage.get("tokenizer") != payload.get("tokenizer_contract"):
        raise RuntimeError("DDP bootstrap parent tokenizer disagrees with its lineage")
    model = payload.get("model")
    if not isinstance(model, Mapping) or not model:
        raise RuntimeError("DDP bootstrap parent has no model state")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in model.items()
    ):
        raise RuntimeError("DDP bootstrap parent model state contains unsafe non-tensor values")
    progress = {name: payload.get(name) for name in PRESERVED_PROGRESS_FIELDS}
    progress["global_step"] = int(payload.get("global_step", payload.get("step", 0)) or 0)
    progress["tokens_seen"] = int(payload.get("tokens_seen", 0) or 0)
    progress["sessions_completed"] = int(payload.get("sessions_completed", 0) or 0)
    progress["continuation_token_counts"] = dict(payload.get("continuation_token_counts", {}))
    progress["raw_window_consumption"] = dict(payload.get("raw_window_consumption", {}))
    progress["data_sampler_state"] = dict(payload.get("data_sampler_state", {}))
    progress["unique_token_ids_seen"] = list(payload.get("unique_token_ids_seen", []))
    progress["validation_history"] = list(payload.get("validation_history", []))
    return payload, {"lineage": lineage, "progress": progress}


def create_bootstrap_manifest(
    *,
    parent_checkpoint: str | Path,
    child_checkpoint: str | Path,
    output_manifest: str | Path,
    child_lineage_id: str,
    target_source_commit: str,
    target_ddp_contract: Mapping[str, object],
    target_data_bindings: Mapping[str, object],
    seed: int,
    sampler_policy: str = "preserve_global_cursor_repartition_by_rank_v1",
    signing_key: str | None = None,
) -> dict[str, object]:
    parent = Path(parent_checkpoint)
    child = Path(child_checkpoint)
    target = Path(output_manifest)
    key = signing_key or os.environ.get("ANRA_MANIFEST_SIGNING_KEY", "")
    if not key:
        raise PermissionError("ANRA_MANIFEST_SIGNING_KEY is required")
    if not parent.is_file():
        raise FileNotFoundError(f"DDP bootstrap parent is missing: {parent}")
    if child.exists():
        raise FileExistsError(f"DDP bootstrap child destination already exists: {child}")
    if parent.resolve() == child.resolve():
        raise ValueError("DDP bootstrap child must not overwrite its parent")
    if target.exists():
        raise FileExistsError(f"DDP bootstrap manifest already exists: {target}")
    if sampler_policy not in SAMPLER_POLICIES:
        raise ValueError(f"unsupported DDP bootstrap sampler policy: {sampler_policy}")
    parent_payload, inspected = _validated_parent(parent)
    lineage = inspected["lineage"]
    assert isinstance(lineage, dict)
    parent_lineage_id = str(lineage["lineage_id"])
    child_id = child_lineage_id.strip()
    if not child_id or child_id == parent_lineage_id:
        raise ValueError("DDP bootstrap requires a distinct non-empty child lineage id")
    if len(target_source_commit) != 40 or any(
        char not in "0123456789abcdef" for char in target_source_commit
    ):
        raise ValueError("DDP bootstrap target source commit must be a full Git SHA")
    if not str(child_checkpoint).strip():
        raise ValueError("DDP bootstrap child checkpoint path cannot be empty")
    manifest: dict[str, object] = {
        "schema": BOOTSTRAP_SCHEMA,
        "parent": {
            "path": str(parent_checkpoint),
            "sha256": sha256_file(parent),
            "size_bytes": parent.stat().st_size,
            "artifact_class": FULL_RESUME,
            "source_commit": str(parent_payload.get("source_commit", "unknown")),
            "lineage_id": parent_lineage_id,
            "lineage_sha256": json_sha256(lineage),
            "architecture": lineage.get("architecture", {}),
            "tokenizer": lineage.get("tokenizer", {}),
            "data": lineage.get("data", {}),
            "progress": inspected["progress"],
        },
        "child": {
            "lineage_id": child_id,
            "checkpoint_path": str(child_checkpoint),
            "source_commit": str(target_source_commit),
            "ddp_contract": dict(target_ddp_contract),
            "data_bindings": dict(target_data_bindings),
            "sampler_policy": sampler_policy,
            "continuation_counts_policy": "preserve_cumulative_counts_v1",
            "seed": int(seed),
            "rng_policy": "fresh_base_seed_plus_rank_times_1000003_v1",
            "restart": dict(RESTART_CONTRACT),
        },
    }
    body_sha = json_sha256(manifest)
    manifest["body_sha256"] = body_sha
    manifest["signature"] = hmac.new(key.encode(), body_sha.encode(), hashlib.sha256).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)
    return manifest


def load_and_verify_bootstrap_manifest(
    path: str | Path, *, signing_key: str | None = None
) -> dict[str, object]:
    key = signing_key or os.environ.get("ANRA_MANIFEST_SIGNING_KEY", "")
    if not key:
        raise PermissionError("ANRA_MANIFEST_SIGNING_KEY is required")
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema") != BOOTSTRAP_SCHEMA:
        raise ValueError("unsupported DDP bootstrap manifest")
    unsigned = {
        key_: value
        for key_, value in payload.items()
        if key_ not in {"body_sha256", "signature"}
    }
    body_sha = json_sha256(unsigned)
    if not hmac.compare_digest(str(payload.get("body_sha256", "")), body_sha):
        raise PermissionError("DDP bootstrap manifest body hash is invalid")
    expected = hmac.new(key.encode(), body_sha.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(payload.get("signature", "")), expected):
        raise PermissionError("DDP bootstrap manifest signature is invalid")
    return payload


def validate_runtime_contract(
    manifest: Mapping[str, object],
    *,
    parent_checkpoint: str | Path,
    child_checkpoint: str | Path,
    source_commit: str,
    ddp_contract: Mapping[str, object],
    model_config: Mapping[str, object],
    tokenizer_contract: Mapping[str, object],
    data_bindings: Mapping[str, object],
    seed: int,
) -> None:
    parent = Path(parent_checkpoint)
    child = Path(child_checkpoint)
    parent_contract = dict(manifest.get("parent", {}))
    child_contract = dict(manifest.get("child", {}))
    if not parent.is_file() or sha256_file(parent) != parent_contract.get("sha256"):
        raise RuntimeError("DDP bootstrap parent checkpoint hash mismatch")
    if child.exists() or child.resolve() == parent.resolve():
        raise RuntimeError("DDP bootstrap child destination must be new and non-overwriting")
    expected_child = Path(str(child_contract.get("checkpoint_path", "")))
    if str(child) != str(expected_child):
        raise RuntimeError("DDP bootstrap child path differs from its signed destination")
    comparisons = {
        "source commit": (source_commit, child_contract.get("source_commit")),
        "DDP contract": (dict(ddp_contract), child_contract.get("ddp_contract")),
        "architecture": (
            dict(model_config),
            dict(parent_contract.get("architecture", {})).get("config"),
        ),
        "tokenizer": (dict(tokenizer_contract), parent_contract.get("tokenizer")),
        "data bindings": (dict(data_bindings), child_contract.get("data_bindings")),
        "seed": (int(seed), child_contract.get("seed")),
    }
    mismatched = [name for name, (active, signed) in comparisons.items() if active != signed]
    if mismatched:
        raise RuntimeError("DDP bootstrap runtime contract mismatch: " + ", ".join(mismatched))
    if child_contract.get("restart") != RESTART_CONTRACT:
        raise RuntimeError("DDP bootstrap restart declaration is incomplete")
    if child_contract.get("sampler_policy") not in SAMPLER_POLICIES:
        raise RuntimeError("DDP bootstrap sampler policy is unsupported")


def load_parent_model_and_progress(
    manifest: Mapping[str, object], parent_checkpoint: str | Path, model: torch.nn.Module
) -> dict[str, object]:
    parent = Path(parent_checkpoint)
    payload, inspected = _validated_parent(parent)
    parent_contract = dict(manifest["parent"])
    lineage = inspected["lineage"]
    if sha256_file(parent) != parent_contract.get("sha256"):
        raise RuntimeError("DDP bootstrap parent changed after manifest validation")
    if json_sha256(lineage) != parent_contract.get("lineage_sha256"):
        raise RuntimeError("DDP bootstrap parent lineage differs from the signed lineage")
    state = payload["model"]
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"DDP bootstrap model accounting failed: {result}")
    progress = dict(inspected["progress"])
    return {
        "progress": progress,
        "provenance": {
            "schema": BOOTSTRAP_SCHEMA,
            "manifest_body_sha256": manifest["body_sha256"],
            "parent_checkpoint_sha256": parent_contract["sha256"],
            "parent_lineage_id": parent_contract["lineage_id"],
            "child_lineage_id": dict(manifest["child"])["lineage_id"],
            "restart": dict(RESTART_CONTRACT),
            "sampler_policy": dict(manifest["child"])["sampler_policy"],
        },
    }
