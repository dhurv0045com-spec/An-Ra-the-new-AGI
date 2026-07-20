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
    "tokenizer_path",
    "tokenizer_metadata_hash",
    "tokenizer_metadata_path",
    "data_manifests",
    "data_manifest_hashes",
    "data_manifest_roles",
    "stage",
    "optimizer",
    "batch_size",
    "accumulation",
    "learning_rate_schedule",
    "seed",
    "seeds",
    "checkpoint_source",
    "checkpoint_source_hash",
    "expected_tokens",
    "owner_authorized",
    "worker_id",
    "worker_role",
    "artifact_path",
    "shard_assignment",
    "checkpoint_read_only",
    "allow_data_profile_change",
    "reset_data_sampler",
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
    tokenizer_path: str | None = None,
    data_manifests: list[str],
    data_manifest_roles: dict[str, str] | None = None,
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
    allow_data_profile_change: bool = False,
    reset_data_sampler: bool = False,
) -> dict[str, object]:
    if len(seeds) != 1:
        raise ValueError("One launch manifest must represent exactly one training seed.")
    seed = int(seeds[0])
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("Launch seed must be in [0, 2**32-1].")
    if reset_data_sampler and not allow_data_profile_change:
        raise ValueError("Sampler reset requires an explicit data-profile change")
    dirty = _git(["status", "--porcelain"])
    bound_tokenizer = Path(tokenizer_path) if tokenizer_path else active_tokenizer_path()
    if not bound_tokenizer.is_absolute():
        bound_tokenizer = (ROOT / bound_tokenizer).resolve()
    tokenizer_metadata = bound_tokenizer.with_suffix(bound_tokenizer.suffix + ".meta.json")
    if not tokenizer_metadata.is_file():
        raise FileNotFoundError(
            f"Launch tokenizer metadata sidecar is missing: {tokenizer_metadata}"
        )
    data_manifest_hashes: dict[str, str] = {}
    for entry in data_manifests:
        manifest_path = Path(str(entry))
        if not manifest_path.is_absolute():
            manifest_path = (ROOT / manifest_path).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Launch data manifest is missing: {manifest_path}")
        data_manifest_hashes[str(entry)] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
    roles = {str(key): str(value) for key, value in (data_manifest_roles or {}).items()}
    if set(roles) != set(data_manifest_hashes):
        if data_manifests:
            raise ValueError("Every launch data manifest requires an explicit role.")
        roles = {}
    if any(role not in {"train", "validation", "test"} for role in roles.values()):
        raise ValueError("Launch data manifest roles must be train, validation, or test.")
    checkpoint_source_value = str(checkpoint_source).strip() or "scratch"
    checkpoint_source_hash = ""
    if checkpoint_source_value.lower() != "scratch":
        checkpoint_path = Path(checkpoint_source_value)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (ROOT / checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Launch source checkpoint is missing: {checkpoint_path}")
        checkpoint_source_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return {
        "schema_version": 3,
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
        "tokenizer_path": str(bound_tokenizer),
        "tokenizer_metadata_hash": hashlib.sha256(
            tokenizer_metadata.read_bytes()
        ).hexdigest(),
        "tokenizer_metadata_path": str(tokenizer_metadata),
        "data_manifests": data_manifests,
        "data_manifest_hashes": data_manifest_hashes,
        "data_manifest_roles": roles,
        "stage": stage,
        "optimizer": optimizer,
        "batch_size": int(batch_size),
        "accumulation": int(accumulation),
        "learning_rate_schedule": schedule,
        "seed": seed,
        "seeds": [seed],
        "checkpoint_source": checkpoint_source_value,
        "checkpoint_source_hash": checkpoint_source_hash,
        "expected_tokens": int(expected_tokens),
        "runtime_estimate_hours": runtime_estimate_hours,
        "owner_authorized": bool(owner_authorized),
        "worker_id": worker_id,
        "worker_role": worker_role,
        "artifact_path": artifact_path,
        "shard_assignment": list(shard_assignment or []),
        "checkpoint_read_only": bool(checkpoint_read_only),
        "allow_data_profile_change": bool(allow_data_profile_change),
        "reset_data_sampler": bool(reset_data_sampler),
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
    allow_blocked: bool = False,
) -> dict[str, object]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Launch manifest must contain a JSON object.")
    missing = sorted(REQUIRED_FIELDS - payload.keys())
    if missing:
        raise ValueError(f"Launch manifest missing fields: {missing}")
    if int(payload["schema_version"]) != 3:
        raise ValueError("Unsupported launch-manifest schema version.")
    seeds = payload["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 1
        or int(seeds[0]) != int(payload["seed"])
    ):
        raise ValueError("Launch manifest must bind exactly one matching training seed.")
    seed = int(payload["seed"])
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("Launch seed must be in [0, 2**32-1].")
    if not verify_manifest(payload, key=key):
        raise PermissionError("Launch manifest signature verification failed.")
    if not bool(payload["owner_authorized"]):
        raise PermissionError("Launch manifest lacks explicit owner authorization.")
    if str(payload.get("blocked_on", "")).strip() and not allow_blocked:
        raise PermissionError(
            f"Launch manifest is blocked on: {payload['blocked_on']}"
        )
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
    tokenizer_path = Path(str(payload["tokenizer_path"]))
    if not tokenizer_path.is_absolute():
        tokenizer_path = (ROOT / tokenizer_path).resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Launch tokenizer artifact is missing: {tokenizer_path}")
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if not hmac.compare_digest(str(payload["tokenizer_hash"]), tokenizer_hash):
        raise ValueError("Launch manifest tokenizer hash does not match its bound artifact.")
    # Downstream runtime code must consume the exact artifact validated above,
    # independent of the process working directory or pack installation path.
    payload["tokenizer_path"] = str(tokenizer_path)
    tokenizer_metadata_path = Path(str(payload["tokenizer_metadata_path"]))
    if not tokenizer_metadata_path.is_absolute():
        tokenizer_metadata_path = (ROOT / tokenizer_metadata_path).resolve()
    expected_metadata_path = tokenizer_path.with_suffix(tokenizer_path.suffix + ".meta.json")
    if tokenizer_metadata_path != expected_metadata_path:
        raise ValueError("Launch tokenizer metadata is not the bound tokenizer sidecar.")
    if not tokenizer_metadata_path.is_file():
        raise FileNotFoundError(
            f"Launch tokenizer metadata artifact is missing: {tokenizer_metadata_path}"
        )
    tokenizer_metadata_hash = hashlib.sha256(
        tokenizer_metadata_path.read_bytes()
    ).hexdigest()
    if not hmac.compare_digest(
        str(payload["tokenizer_metadata_hash"]), tokenizer_metadata_hash
    ):
        raise ValueError(
            "Launch manifest tokenizer metadata hash does not match its bound artifact."
        )
    payload["tokenizer_metadata_path"] = str(tokenizer_metadata_path)
    data_manifests = payload["data_manifests"]
    data_manifest_hashes = payload["data_manifest_hashes"]
    data_manifest_roles = payload["data_manifest_roles"]
    if (
        not isinstance(data_manifests, list)
        or not isinstance(data_manifest_hashes, dict)
        or not isinstance(data_manifest_roles, dict)
    ):
        raise ValueError(
            "Launch data manifest bindings must include a list, hash object, and role object."
        )
    if len(data_manifests) != len({str(entry) for entry in data_manifests}):
        raise ValueError("Launch data manifests must be unique.")
    if set(data_manifest_hashes) != {str(entry) for entry in data_manifests}:
        raise ValueError("Launch data manifest hash keys do not match declared paths.")
    if set(data_manifest_roles) != set(data_manifest_hashes):
        raise ValueError("Launch data manifest role keys do not match declared paths.")
    if any(
        str(role) not in {"train", "validation", "test"}
        for role in data_manifest_roles.values()
    ):
        raise ValueError("Launch data manifest contains an unsupported role.")
    for entry in data_manifests:
        manifest_path = Path(str(entry))
        if not manifest_path.is_absolute():
            manifest_path = (ROOT / manifest_path).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Launch data manifest is missing: {manifest_path}")
        actual_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if not hmac.compare_digest(str(data_manifest_hashes[str(entry)]), actual_hash):
            raise ValueError(
                f"Launch data manifest hash does not match its bound artifact: {manifest_path}"
            )
    artifact_raw = str(payload["artifact_path"]).strip()
    checkpoint_raw = str(payload["checkpoint_source"]).strip()
    checkpoint_hash = str(payload["checkpoint_source_hash"]).strip()
    if checkpoint_raw.lower() == "scratch":
        if checkpoint_hash:
            raise ValueError("Scratch launches must not declare a checkpoint-source hash")
    else:
        checkpoint_path = Path(checkpoint_raw)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (ROOT / checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Launch source checkpoint is missing: {checkpoint_path}"
            )
        actual_checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if not hmac.compare_digest(checkpoint_hash, actual_checkpoint_hash):
            raise ValueError("Launch checkpoint hash does not match its bound artifact")
    if artifact_raw and checkpoint_raw and Path(artifact_raw) == Path(checkpoint_raw):
        raise ValueError("A worker artifact path must not overwrite its source checkpoint")
    if not bool(payload["checkpoint_read_only"]):
        raise ValueError("Experiment-farm workers must treat source checkpoints as read-only")
    if bool(payload["reset_data_sampler"]) and not bool(
        payload["allow_data_profile_change"]
    ):
        raise ValueError("Launch sampler reset requires a signed data-profile change")
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
