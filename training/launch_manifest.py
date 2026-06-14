"""Signed, reproducible training launch manifests."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import torch

from anra.anra_paths import ROOT, V3_TOKENIZER_FILE


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
    "signature",
}


def _git(command: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *command], text=True, stderr=subprocess.DEVNULL).strip()
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
) -> dict[str, object]:
    dirty = _git(["status", "--porcelain"])
    return {
        "schema_version": 1,
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
    }


def sign_manifest(manifest: dict[str, object], path: str | Path, *, key: str | None = None) -> dict[str, object]:
    signing_key = key or os.environ.get("ANRA_MANIFEST_SIGNING_KEY", "")
    if not signing_key:
        raise PermissionError("ANRA_MANIFEST_SIGNING_KEY is required to sign a launch manifest.")
    unsigned = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signed = {**manifest, "signature": hmac.new(signing_key.encode("utf-8"), unsigned, hashlib.sha256).hexdigest()}
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
    if int(payload["schema_version"]) != 1:
        raise ValueError("Unsupported launch-manifest schema version.")
    if not verify_manifest(payload, key=key):
        raise PermissionError("Launch manifest signature verification failed.")
    if not bool(payload["owner_authorized"]):
        raise PermissionError("Launch manifest lacks explicit owner authorization.")
    if str(payload["extension_profile"]) not in {"none", "cognition-v1"}:
        raise ValueError("Unsupported cognitive extension profile.")
    schedule = payload["learning_rate_schedule"]
    if not isinstance(schedule, dict) or str(schedule.get("kind", "")).lower() != "wsd":
        raise ValueError("Canonical launches require a WSD learning-rate schedule.")
    tokenizer_hash = hashlib.sha256(V3_TOKENIZER_FILE.read_bytes()).hexdigest()
    if not hmac.compare_digest(str(payload["tokenizer_hash"]), tokenizer_hash):
        raise ValueError("Launch manifest tokenizer hash does not match the canonical tokenizer.")
    for entry in payload["data_manifests"]:
        manifest_path = Path(str(entry))
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Launch data manifest is missing: {manifest_path}")
    return payload
