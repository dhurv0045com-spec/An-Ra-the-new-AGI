"""Signed, reproducible training launch manifests."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import torch
from anra.anra_paths import ROOT

from training.v2_runtime import active_tokenizer_path

REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "git_commit",
    "dirty_state_hash",
    "model_profile",
    "extension_profile",
    "tokenizer_hash",
    "data_manifests",
    "stage",
    "optimizer",
    "batch_size",
    "accumulation",
    "learning_rate_schedule",
    "seeds",
    "checkpoint_source",
    "expected_tokens",
    "owner_authorized",
    "worker_id",
    "worker_role",
    "artifact_path",
    "shard_assignment",
    "checkpoint_read_only",
    "signature",
}


def _git(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *command], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_launch_manifest(
    *,
    model_profile: str,
    extension_profile: str,
    tokenizer_hash: str,
    data_manifests: list[str],
    stage: str,
    optimizer: str,
    batch_size: int,
    accumulation: int,
    schedule: dict[str, object],
    seeds: list[int],
    checkpoint_source: str,
    expected_tokens: int,
    runtime_estimate_hours: float | None,
    owner_authorized: bool,
    worker_id: str = "coordinator",
    worker_role: str = "coordinator",
    artifact_path: str = "",
    shard_assignment: list[int] | None = None,
    checkpoint_read_only: bool = True,
) -> dict[str, object]:
    dirty = _git(["status", "--porcelain"])
    return {
        "schema_version": 2,
        "run_id": str(uuid.uuid4()),
        "created_at": time.time(),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "cuda": torch.version.cuda,
        },
        "runtime": {"python": os.sys.version, "torch": torch.__version__},
        "git_commit": _git(["rev-parse", "HEAD"]),
        "dirty_state_hash": hashlib.sha256(dirty.encode("utf-8")).hexdigest(),
        "model_profile": model_profile,
        "extension_profile": extension_profile,
        "tokenizer_hash": tokenizer_hash,
        "data_manifests": data_manifests,
        "stage": stage,
        "optimizer": optimizer,
        "batch_size": int(batch_size),
        "accumulation": int(accumulation),
        "learning_rate_schedule": schedule,
        "seeds": seeds,
        "checkpoint_source": checkpoint_source,
        "expected_tokens": int(expected_tokens),
        "runtime_estimate_hours": runtime_estimate_hours,
        "owner_authorized": bool(owner_authorized),
        "worker_id": worker_id,
        "worker_role": worker_role,
        "artifact_path": artifact_path,
        "shard_assignment": list(shard_assignment or []),
        "checkpoint_read_only": bool(checkpoint_read_only),
    }


def sign_manifest(
    manifest: dict[str, object], path: str | Path, *, key: str | None = None
) -> dict[str, object]:
    signing_key = key or os.environ.get("ANRA_MANIFEST_SIGNING_KEY", "")
    if not signing_key:
        raise PermissionError("ANRA_MANIFEST_SIGNING_KEY is required to sign a launch manifest.")
    unsigned = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signed = {
        **manifest,
        "signature": hmac.new(signing_key.encode("utf-8"), unsigned, hashlib.sha256).hexdigest(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(signed, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return signed


def verify_manifest(manifest: dict[str, object], *, key: str | None = None) -> bool:
    signing_key = key or os.environ.get("ANRA_MANIFEST_SIGNING_KEY", "")
    signature = str(manifest.get("signature", ""))
    unsigned = {k: v for k, v in manifest.items() if k != "signature"}
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = hmac.new(signing_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return bool(signing_key and hmac.compare_digest(signature, expected))


def load_and_validate_manifest(
    path: str | Path,
    *,
    key: str | None = None,
) -> dict[str, object]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Launch manifest must contain a JSON object.")
    missing = sorted(REQUIRED_FIELDS - payload.keys())
    if missing:
        raise ValueError(f"Launch manifest missing fields: {missing}")
    if int(payload["schema_version"]) != 2:
        raise ValueError("Unsupported launch-manifest schema version.")
    if not verify_manifest(payload, key=key):
        raise PermissionError("Launch manifest signature verification failed.")
    if not bool(payload["owner_authorized"]):
        raise PermissionError("Launch manifest lacks explicit owner authorization.")
    if str(payload["extension_profile"]) not in {"none", "cognition-v1"}:
        raise ValueError("Unsupported cognitive extension profile.")
    schedule = payload["learning_rate_schedule"]
    if not isinstance(schedule, dict) or str(schedule.get("kind", "")).lower() not in {
        "cosine",
        "cosine_with_warmup",
    }:
        raise ValueError("Canonical launches require a cosine learning-rate schedule.")
    if abs(float(schedule.get("warmup_fraction", 0.0)) - 0.02) > 1e-9:
        raise ValueError("Canonical continuation launches require exactly 2% warmup.")
    if abs(float(schedule.get("min_lr", 0.0)) - 1e-5) > 1e-12:
        raise ValueError("Canonical continuation launches must decay to min_lr=1e-5.")
    tokenizer_path = active_tokenizer_path()
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if not hmac.compare_digest(str(payload["tokenizer_hash"]), tokenizer_hash):
        raise ValueError("Launch manifest tokenizer hash does not match the canonical tokenizer.")
    for entry in payload["data_manifests"]:
        manifest_path = Path(str(entry))
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Launch data manifest is missing: {manifest_path}")
    artifact_raw = str(payload["artifact_path"]).strip()
    checkpoint_raw = str(payload["checkpoint_source"]).strip()
    if artifact_raw and checkpoint_raw and Path(artifact_raw) == Path(checkpoint_raw):
        raise ValueError("A worker artifact path must not overwrite its source checkpoint")
    if not bool(payload["checkpoint_read_only"]):
        raise ValueError("Experiment-farm workers must treat source checkpoints as read-only")
    return payload


def build_experiment_farm_manifests(
    *,
    output_dir: str | Path,
    base: dict[str, object],
    key: str | None = None,
) -> list[dict[str, object]]:
    """Create seven signed, non-overlapping An-Ra experiment jobs."""
    roles = (
        "shard_validation",
        "tokenizer_fertility",
        "mod_ablation",
        "rim_esv_ablation",
        "dstp_hal_ablation",
        "continuation_candidate",
        "evaluation_reproducibility",
    )
    root = Path(output_dir)
    manifests: list[dict[str, object]] = []
    for index, role in enumerate(roles):
        worker_id = f"colab-{index + 1:02d}"
        artifact = root / "artifacts" / worker_id / "candidate.pt"
        manifest = build_launch_manifest(
            **base,
            worker_id=worker_id,
            worker_role=role,
            artifact_path=str(artifact),
            shard_assignment=[index],
            checkpoint_read_only=True,
        )
        manifests.append(
            sign_manifest(
                manifest,
                root / "jobs" / f"{worker_id}.json",
                key=key,
            )
        )
    return manifests


def select_experiment_candidate(report_paths: list[str | Path]) -> dict[str, object]:
    """Select one proven worker artifact; never average unrelated optimizer states."""
    candidates: list[dict[str, object]] = []
    for report_path in report_paths:
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if not payload.get("completed") or not payload.get("reproducible"):
            continue
        if not payload.get("checkpoint_tensor_accounting"):
            continue
        candidates.append(payload)
    if not candidates:
        raise RuntimeError("No experiment-farm candidate passed reproducibility and tensor gates")
    selected = max(
        candidates,
        key=lambda item: (
            float(item.get("capability_score", 0.0)),
            -float(item.get("validation_loss", float("inf"))),
        ),
    )
    return {
        "selected_worker": selected.get("worker_id"),
        "checkpoint": selected.get("artifact_path"),
        "capability_score": selected.get("capability_score"),
        "validation_loss": selected.get("validation_loss"),
        "selection_policy": "capability_then_validation_loss_no_weight_averaging",
    }
