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

from training.v2_config import (
    ANRA_V4_GROWTH_MODEL_PROFILE,
    CANONICAL_FOUNDATION_OPTIMIZER,
    CANONICAL_MODEL_PROFILE,
    CANONICAL_TRAINING_SEED,
    CANONICAL_V4_VOCAB_SIZE,
    model_parameter_count,
    model_profile_registration,
    resolve_model_profile,
)
from training.v2_runtime import active_tokenizer_path

TRAINING_CONTRACT_ID = "anra-training-contract/v4"
LAUNCH_MANIFEST_SCHEMA_VERSION = 4
DEFAULT_HOT_STORAGE_LIMIT_BYTES = 12 * 1024**3
GROWTH_HOT_STORAGE_LIMIT_BYTES = 32 * 1024**3
DEFAULT_CHECKPOINT_STEPS = 100
DEFAULT_CHECKPOINT_MINUTES = 15
OPERATIONAL_STAGE_LABELS = frozenset(
    {
        "canary",
        "foundation",
        "foundation_200m",
        "foundation_500m",
        "foundation_1b",
        "foundation_3_6b",
        "architecture_pilot",
        "growth_alignment",
        "growth_continuation",
    }
)

REQUIRED_FIELDS = {
    "contract_id",
    "schema_version",
    "run_id",
    "git_commit",
    "dirty_state_hash",
    "model_profile",
    "model_contract",
    "model_contract_hash",
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
    "token_window",
    "artifact_destinations",
    "resource_limits",
    "checkpoint_parent",
    "growth_manifest",
    "checkpoint_source_kind",
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


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _model_contract(profile: str) -> tuple[dict[str, object], str]:
    experimental = str(profile).strip().lower() != CANONICAL_MODEL_PROFILE
    config, _training = resolve_model_profile(
        profile,
        allow_experimental=experimental,
    )
    registration = model_profile_registration(profile)
    contract = {
        "profile": registration.name,
        "status": registration.status,
        "parameter_count": model_parameter_count(config, config.vocab_size),
        "vocab_size": int(config.vocab_size),
        "context_length": int(config.target_seq_len),
        "d_model": int(config.n_embd),
        "n_layers": int(config.n_layer),
        "n_query_heads": int(config.n_head),
        "n_kv_heads": int(config.n_kv_head),
        "d_ff": int(config.d_ff or 0),
        "head_dim": int(config.n_embd // config.n_head),
        "optimizer": CANONICAL_FOUNDATION_OPTIMIZER,
        "tokenizer_line": "v4",
    }
    if int(contract["parameter_count"]) != int(registration.expected_parameters):
        raise AssertionError(
            f"Model profile {profile!r} parameter contract drifted: "
            f"{contract['parameter_count']:,} != {registration.expected_parameters:,}"
        )
    return contract, _sha256_json(contract)


def _hot_storage_ceiling(model_profile: str) -> int:
    return (
        GROWTH_HOT_STORAGE_LIMIT_BYTES
        if str(model_profile).strip().lower() == ANRA_V4_GROWTH_MODEL_PROFILE
        else DEFAULT_HOT_STORAGE_LIMIT_BYTES
    )


def _default_resource_limits(model_profile: str) -> dict[str, object]:
    return {
        "gpu_required": True,
        "session_budget_minutes": 600,
        "drain_reserve_minutes": 30,
        "checkpoint_steps": DEFAULT_CHECKPOINT_STEPS,
        "checkpoint_minutes": DEFAULT_CHECKPOINT_MINUTES,
        "hot_storage_limit_bytes": _hot_storage_ceiling(model_profile),
        "max_attempts": 5,
    }


def _normalise_token_window(
    token_window: dict[str, object] | None,
    *,
    data_manifest_hashes: dict[str, str],
    shard_assignment: list[int],
    expected_tokens: int,
) -> dict[str, object]:
    if int(expected_tokens) <= 0:
        raise ValueError("Expected token count must be positive")
    if token_window is None:
        start = 0
        end = int(expected_tokens)
        source = {
            "data_manifest_hashes": data_manifest_hashes,
            "shards": shard_assignment,
            "start_token": start,
            "end_token": end,
        }
        return {"window_id": _sha256_json(source), **source}
    result = dict(token_window)
    start = int(result.get("start_token", -1))
    end = int(result.get("end_token", -1))
    if start < 0 or end <= start:
        raise ValueError("Token window requires 0 <= start_token < end_token")
    if end > int(expected_tokens):
        raise ValueError("Token window cannot extend beyond expected_tokens")
    declared_hashes = result.setdefault("data_manifest_hashes", data_manifest_hashes)
    declared_shards = result.setdefault("shards", shard_assignment)
    if declared_hashes != data_manifest_hashes:
        raise ValueError("Token window data hashes do not match the launch manifests")
    if declared_shards != shard_assignment:
        raise ValueError("Token window shards do not match the launch assignment")
    identity_source = {key: value for key, value in result.items() if key != "window_id"}
    expected_window_id = _sha256_json(identity_source)
    declared = str(result.get("window_id", expected_window_id))
    if not hmac.compare_digest(declared, expected_window_id):
        raise ValueError("Token-window id does not match its signed contents")
    result["window_id"] = expected_window_id
    return result


def _validate_artifact_destinations(destinations: object) -> list[dict[str, object]]:
    if not isinstance(destinations, list) or not destinations:
        raise ValueError("Launch requires at least one artifact destination")
    normalised: list[dict[str, object]] = []
    identities: set[tuple[str, str]] = set()
    for destination in destinations:
        if not isinstance(destination, dict):
            raise ValueError("Artifact destinations must be objects")
        kind = str(destination.get("kind", "")).strip()
        uri = str(destination.get("uri", "")).strip()
        if kind not in {"full_resume", "fp16_inference"} or not uri:
            raise ValueError("Artifact destination requires a supported kind and non-empty URI")
        identity = (kind, uri)
        if identity in identities:
            raise ValueError("Artifact destinations must be unique")
        identities.add(identity)
        normalised.append(dict(destination))
    if not any(
        destination["kind"] == "full_resume"
        and bool(destination.get("required"))
        for destination in normalised
    ):
        raise ValueError("Launch requires a mandatory full_resume artifact destination")
    return normalised


def _validate_resource_limits(
    limits: object,
    *,
    model_profile: str,
) -> dict[str, object]:
    if not isinstance(limits, dict):
        raise ValueError("Launch resource limits must be an object")
    checkpoint_steps = int(limits.get("checkpoint_steps", 0))
    checkpoint_minutes = int(limits.get("checkpoint_minutes", 0))
    if checkpoint_steps <= 0 or checkpoint_steps > DEFAULT_CHECKPOINT_STEPS:
        raise ValueError("Checkpoint cadence must be at most 100 optimizer steps")
    if checkpoint_minutes <= 0 or checkpoint_minutes > DEFAULT_CHECKPOINT_MINUTES:
        raise ValueError("Checkpoint cadence must be at most 15 minutes")
    session_budget = int(limits.get("session_budget_minutes", 0))
    drain_reserve = int(limits.get("drain_reserve_minutes", 0))
    if session_budget <= drain_reserve or drain_reserve < 15:
        raise ValueError("Session budget must leave at least 15 minutes for draining")
    hot_storage = int(limits.get("hot_storage_limit_bytes", 0))
    ceiling = _hot_storage_ceiling(model_profile)
    if hot_storage <= 0 or hot_storage > ceiling:
        raise ValueError(
            "Hot storage limit must be positive and cannot exceed "
            f"{ceiling // 1024**3} GiB for {model_profile}"
        )
    if int(limits.get("max_attempts", 0)) <= 0:
        raise ValueError("Launch resource limits require at least one bounded attempt")
    return dict(limits)


def _growth_binding(
    profile: str,
    growth_manifest: str | Path | None,
    growth_parent_checkpoint: str | Path | None,
) -> dict[str, object]:
    registration = model_profile_registration(profile)
    if not registration.requires_growth_manifest:
        if growth_manifest or growth_parent_checkpoint:
            raise ValueError(
                "Canonical scratch/continuation profiles cannot bind a growth manifest"
            )
        return {}
    if not growth_manifest:
        raise ValueError(f"Model profile {profile!r} requires a validated growth manifest")
    if not growth_parent_checkpoint:
        raise ValueError(f"Model profile {profile!r} requires its frozen parent checkpoint")
    path = Path(growth_manifest)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Growth manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Growth manifest must contain a JSON object")
    # Import locally so ordinary 181M launch-contract validation does not pay
    # for the model-growth implementation unless a growth child is requested.
    from training.csii import CrossScaleIdentityInheritance

    validated = CrossScaleIdentityInheritance.validate_growth_report(payload)
    if str(validated["target_profile"]).strip().lower() != registration.name:
        raise ValueError("Growth manifest target does not match the launch profile")
    if (
        str(validated["source_profile"]).strip().lower()
        != str(registration.parent_profile).strip().lower()
    ):
        raise ValueError("Growth manifest source does not match the registered parent profile")
    parent_path = Path(growth_parent_checkpoint)
    if not parent_path.is_absolute():
        parent_path = (ROOT / parent_path).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"Growth parent checkpoint is missing: {parent_path}")
    parent_sha256 = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    if not hmac.compare_digest(
        parent_sha256, str(validated["source_checkpoint_sha256"])
    ):
        raise ValueError("Growth parent checkpoint does not match the parity-gated manifest")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parent_profile": registration.parent_profile,
        "parent_checkpoint_path": str(parent_path),
        "parent_checkpoint_sha256": parent_sha256,
        "source_checkpoint_sha256": str(validated["source_checkpoint_sha256"]),
        "source_architecture_sha256": str(validated["source_architecture_sha256"]),
        "target_architecture_sha256": str(validated["target_architecture_sha256"]),
        "parity_cosine": float(validated["parity_cosine"]),
        "optimizer_restart_required": True,
    }


def _growth_initialization_metadata(
    checkpoint_path: Path,
    *,
    checkpoint_sha256: str,
    growth_binding: dict[str, object],
    target_profile: str,
) -> dict[str, object] | None:
    metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".meta.json")
    if not metadata_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "anra-growth-initialization/v1":
        return None
    checks = {
        "artifact_sha256": checkpoint_sha256,
        "growth_manifest_sha256": str(growth_binding["sha256"]),
        "source_checkpoint_sha256": str(growth_binding["source_checkpoint_sha256"]),
        "target_profile": str(target_profile),
        "source_profile": str(growth_binding["parent_profile"]),
    }
    mismatches = {
        key: {"declared": payload.get(key), "expected": expected}
        for key, expected in checks.items()
        if str(payload.get(key, "")) != expected
    }
    if mismatches:
        raise ValueError(f"Growth initialization metadata mismatch: {mismatches}")
    if (
        payload.get("artifact_class") != "growth_initialization"
        or payload.get("optimizer_restart_required") is not True
        or payload.get("optimizer_state_inherited") is not False
        or payload.get("training_resume_allowed") is not False
    ):
        raise ValueError("Growth initialization metadata violates fresh-optimizer semantics")
    return {
        "path": str(metadata_path),
        "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
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
    token_window: dict[str, object] | None = None,
    artifact_destinations: list[dict[str, object]] | None = None,
    resource_limits: dict[str, object] | None = None,
    growth_manifest: str | Path | None = None,
    growth_parent_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    if int(batch_size) <= 0 or int(accumulation) <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if not owner_authorized:
        raise PermissionError("A launch contract requires explicit owner authorization")
    if not checkpoint_read_only:
        raise ValueError("A launch must treat its source checkpoint as immutable")
    if len(seeds) != 1:
        raise ValueError("One launch manifest must represent exactly one training seed.")
    seed = int(seeds[0])
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("Launch seed must be in [0, 2**32-1].")
    if reset_data_sampler and not allow_data_profile_change:
        raise ValueError("Sampler reset requires an explicit data-profile change")
    stage_key = str(stage).strip().lower().replace("-", "_")
    if stage_key not in OPERATIONAL_STAGE_LABELS:
        raise ValueError(
            f"Unsupported V4 training stage {stage!r}; legacy Stage A-E labels are retired"
        )
    if str(optimizer).strip().lower() != CANONICAL_FOUNDATION_OPTIMIZER:
        raise ValueError(
            f"V4 foundation launches require {CANONICAL_FOUNDATION_OPTIMIZER}, got {optimizer!r}"
        )
    profile_key = str(model_profile).strip().lower()
    if (
        profile_key in {CANONICAL_MODEL_PROFILE, ANRA_V4_GROWTH_MODEL_PROFILE}
        and seed != CANONICAL_TRAINING_SEED
    ):
        raise ValueError(
            f"Canonical V4 lineages use seed {CANONICAL_TRAINING_SEED}; "
            "replicated pilot launches must be declared separately"
        )
    if str(schedule.get("kind", "")).lower() not in {"cosine", "cosine_with_warmup"}:
        raise ValueError("Canonical launches require a cosine learning-rate schedule")
    if abs(float(schedule.get("warmup_fraction", 0.0)) - 0.02) > 1e-9:
        raise ValueError("Canonical launches require exactly 2% warmup")
    expected_min_lr = 5e-6 if profile_key == ANRA_V4_GROWTH_MODEL_PROFILE else 1e-5
    if abs(float(schedule.get("min_lr", 0.0)) - expected_min_lr) > 1e-12:
        raise ValueError(
            f"Model profile {profile_key} requires min_lr={expected_min_lr:g}"
        )
    dirty = _git(["status", "--porcelain"])
    bound_tokenizer = Path(tokenizer_path) if tokenizer_path else active_tokenizer_path()
    if not bound_tokenizer.is_absolute():
        bound_tokenizer = (ROOT / bound_tokenizer).resolve()
    tokenizer_metadata = bound_tokenizer.with_suffix(bound_tokenizer.suffix + ".meta.json")
    if not bound_tokenizer.is_file():
        raise FileNotFoundError(f"Launch tokenizer artifact is missing: {bound_tokenizer}")
    actual_tokenizer_hash = hashlib.sha256(bound_tokenizer.read_bytes()).hexdigest()
    if not hmac.compare_digest(str(tokenizer_hash), actual_tokenizer_hash):
        raise ValueError("Launch tokenizer hash does not match its bound artifact")
    if not tokenizer_metadata.is_file():
        raise FileNotFoundError(
            f"Launch tokenizer metadata sidecar is missing: {tokenizer_metadata}"
        )
    try:
        tokenizer_metadata_payload = json.loads(tokenizer_metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Launch tokenizer metadata is not valid JSON") from exc
    if not isinstance(tokenizer_metadata_payload, dict) or int(
        tokenizer_metadata_payload.get("schema_version", 0)
    ) != 4:
        raise ValueError("Operational launches require tokenizer schema V4")
    if int(tokenizer_metadata_payload.get("vocab_size", 0)) != CANONICAL_V4_VOCAB_SIZE:
        raise ValueError("Operational launches require the 32,768-token V4 vocabulary")
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
    if str(artifact_path).strip() and checkpoint_source_value.lower() != "scratch":
        output_path = Path(str(artifact_path))
        if not output_path.is_absolute():
            output_path = (ROOT / output_path).resolve()
        source_path = Path(checkpoint_source_value)
        if not source_path.is_absolute():
            source_path = (ROOT / source_path).resolve()
        if output_path == source_path:
            raise ValueError("A worker artifact path must not overwrite its source checkpoint")
    run_id = str(uuid.uuid4())
    model_contract, model_contract_hash = _model_contract(model_profile)
    assignment = list(shard_assignment or [])
    signed_window = _normalise_token_window(
        token_window,
        data_manifest_hashes=data_manifest_hashes,
        shard_assignment=assignment,
        expected_tokens=int(expected_tokens),
    )
    limits = _validate_resource_limits(
        {**_default_resource_limits(profile_key), **dict(resource_limits or {})},
        model_profile=profile_key,
    )
    destinations = list(artifact_destinations or [])
    if not destinations and str(artifact_path).strip():
        destinations = [
            {
                "kind": "full_resume",
                "uri": str(artifact_path),
                "required": True,
            }
        ]
    destinations = _validate_artifact_destinations(destinations)
    growth_binding = _growth_binding(
        model_profile,
        growth_manifest,
        growth_parent_checkpoint,
    )
    if growth_binding and checkpoint_source_value.lower() == "scratch":
        raise ValueError("A growth child cannot be launched from scratch")
    checkpoint_source_kind = (
        "scratch" if checkpoint_source_value.lower() == "scratch" else "full_resume"
    )
    if growth_binding:
        checkpoint_path = Path(checkpoint_source_value)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (ROOT / checkpoint_path).resolve()
        initialization_metadata = _growth_initialization_metadata(
            checkpoint_path,
            checkpoint_sha256=checkpoint_source_hash,
            growth_binding=growth_binding,
            target_profile=profile_key,
        )
        if initialization_metadata is not None:
            checkpoint_source_kind = "growth_initialization"
            growth_binding["initialization_metadata"] = initialization_metadata
    checkpoint_parent = {
        "kind": checkpoint_source_kind,
        "path": checkpoint_source_value,
        "sha256": checkpoint_source_hash,
    }
    return {
        "contract_id": TRAINING_CONTRACT_ID,
        "schema_version": LAUNCH_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": time.time(),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "cuda": torch.version.cuda,
        },
        "runtime": {"python": os.sys.version, "torch": torch.__version__},
        "git_commit": _git(["rev-parse", "HEAD"]),
        "dirty_state_hash": hashlib.sha256(dirty.encode("utf-8")).hexdigest(),
        "model_profile": model_profile,
        "model_contract": model_contract,
        "model_contract_hash": model_contract_hash,
        "extension_profile": extension_profile,
        "tokenizer_hash": actual_tokenizer_hash,
        "tokenizer_path": str(bound_tokenizer),
        "tokenizer_metadata_hash": hashlib.sha256(
            tokenizer_metadata.read_bytes()
        ).hexdigest(),
        "tokenizer_metadata_path": str(tokenizer_metadata),
        "data_manifests": data_manifests,
        "data_manifest_hashes": data_manifest_hashes,
        "data_manifest_roles": roles,
        "stage": stage_key,
        "optimizer": optimizer,
        "batch_size": int(batch_size),
        "accumulation": int(accumulation),
        "learning_rate_schedule": schedule,
        "seed": seed,
        "seeds": [seed],
        "checkpoint_source": checkpoint_source_value,
        "checkpoint_source_hash": checkpoint_source_hash,
        "checkpoint_source_kind": checkpoint_source_kind,
        "expected_tokens": int(expected_tokens),
        "token_window": signed_window,
        "artifact_destinations": destinations,
        "resource_limits": limits,
        "checkpoint_parent": checkpoint_parent,
        "growth_manifest": growth_binding,
        "runtime_estimate_hours": runtime_estimate_hours,
        "owner_authorized": bool(owner_authorized),
        "worker_id": worker_id,
        "worker_role": worker_role,
        "artifact_path": artifact_path,
        "shard_assignment": assignment,
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
    if str(payload["contract_id"]) != TRAINING_CONTRACT_ID:
        raise ValueError("Unsupported training contract id.")
    if int(payload["schema_version"]) != LAUNCH_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported launch-manifest schema version.")
    if int(payload["batch_size"]) <= 0 or int(payload["accumulation"]) <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if int(payload["expected_tokens"]) <= 0:
        raise ValueError("Expected token count must be positive")
    if str(payload["stage"]) not in OPERATIONAL_STAGE_LABELS:
        raise ValueError("Launch manifest contains a retired or unsupported V4 stage")
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
    profile = str(payload["model_profile"]).strip().lower()
    model_contract, model_contract_hash = _model_contract(profile)
    declared_model_contract = payload["model_contract"]
    if not isinstance(declared_model_contract, dict):
        raise ValueError("Launch model contract must be an object")
    if declared_model_contract != model_contract or not hmac.compare_digest(
        str(payload["model_contract_hash"]), model_contract_hash
    ):
        raise ValueError("Launch model contract does not match the registered profile")
    if str(payload["optimizer"]).strip().lower() != CANONICAL_FOUNDATION_OPTIMIZER:
        raise ValueError("V4 training contracts require AdamW")
    if profile in {CANONICAL_MODEL_PROFILE, ANRA_V4_GROWTH_MODEL_PROFILE} and seed != int(
        CANONICAL_TRAINING_SEED
    ):
        raise ValueError("Canonical V4 training contracts require seed 1301")
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
    expected_min_lr = 5e-6 if profile == ANRA_V4_GROWTH_MODEL_PROFILE else 1e-5
    if abs(float(schedule.get("min_lr", 0.0)) - expected_min_lr) > 1e-12:
        raise ValueError(
            f"Model profile {profile} requires min_lr={expected_min_lr:g}"
        )
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
    try:
        tokenizer_metadata = json.loads(tokenizer_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Launch tokenizer metadata is not valid JSON") from exc
    if not isinstance(tokenizer_metadata, dict) or int(
        tokenizer_metadata.get("schema_version", 0)
    ) != 4:
        raise ValueError("Operational launches require tokenizer schema V4")
    if int(tokenizer_metadata.get("vocab_size", 0)) != CANONICAL_V4_VOCAB_SIZE:
        raise ValueError("Operational launches require the 32,768-token V4 vocabulary")
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
    checkpoint_source_kind = str(payload["checkpoint_source_kind"]).strip()
    if checkpoint_source_kind not in {"scratch", "full_resume", "growth_initialization"}:
        raise ValueError("Launch checkpoint source has an unsupported artifact kind")
    if checkpoint_raw.lower() == "scratch":
        if checkpoint_hash:
            raise ValueError("Scratch launches must not declare a checkpoint-source hash")
        if checkpoint_source_kind != "scratch":
            raise ValueError("Scratch checkpoint source must declare kind=scratch")
    else:
        if checkpoint_source_kind == "scratch":
            raise ValueError("A checkpoint file cannot declare kind=scratch")
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
    if artifact_raw and checkpoint_raw.lower() != "scratch":
        artifact_path = Path(artifact_raw)
        if not artifact_path.is_absolute():
            artifact_path = (ROOT / artifact_path).resolve()
        source_path = Path(checkpoint_raw)
        if not source_path.is_absolute():
            source_path = (ROOT / source_path).resolve()
        if artifact_path == source_path:
            raise ValueError("A worker artifact path must not overwrite its source checkpoint")
    if not bool(payload["checkpoint_read_only"]):
        raise ValueError("Experiment-farm workers must treat source checkpoints as read-only")
    if bool(payload["reset_data_sampler"]) and not bool(
        payload["allow_data_profile_change"]
    ):
        raise ValueError("Launch sampler reset requires a signed data-profile change")
    token_window = payload["token_window"]
    if not isinstance(token_window, dict):
        raise ValueError("Launch token window must be an object")
    expected_window = _normalise_token_window(
        token_window,
        data_manifest_hashes={str(k): str(v) for k, v in data_manifest_hashes.items()},
        shard_assignment=[int(value) for value in payload["shard_assignment"]],
        expected_tokens=int(payload["expected_tokens"]),
    )
    if expected_window != token_window:
        raise ValueError("Launch token window is not canonical")
    _validate_artifact_destinations(payload["artifact_destinations"])
    _validate_resource_limits(
        payload["resource_limits"],
        model_profile=profile,
    )
    parent = payload["checkpoint_parent"]
    if not isinstance(parent, dict):
        raise ValueError("Launch checkpoint parent must be an object")
    expected_parent = {
        "kind": checkpoint_source_kind,
        "path": checkpoint_raw,
        "sha256": checkpoint_hash,
    }
    if parent != expected_parent:
        raise ValueError("Launch checkpoint parent does not match checkpoint_source")
    growth_binding = payload["growth_manifest"]
    registration = model_profile_registration(profile)
    if registration.requires_growth_manifest:
        if checkpoint_raw.lower() == "scratch" or not isinstance(growth_binding, dict):
            raise ValueError("Growth launches require a parent full-resume checkpoint")
        growth_path = Path(str(growth_binding.get("path", "")))
        if not growth_path.is_file():
            raise FileNotFoundError(f"Growth manifest is missing: {growth_path}")
        actual_growth_hash = hashlib.sha256(growth_path.read_bytes()).hexdigest()
        if not hmac.compare_digest(str(growth_binding.get("sha256", "")), actual_growth_hash):
            raise ValueError("Growth manifest hash does not match its artifact")
        growth_payload = json.loads(growth_path.read_text(encoding="utf-8"))
        if not isinstance(growth_payload, dict):
            raise ValueError("Growth manifest must contain a JSON object")
        from training.csii import CrossScaleIdentityInheritance

        validated_growth = CrossScaleIdentityInheritance.validate_growth_report(growth_payload)
        if str(validated_growth["target_profile"]).strip().lower() != registration.name:
            raise ValueError("Growth manifest target does not match the launch profile")
        if (
            str(validated_growth["source_profile"]).strip().lower()
            != str(registration.parent_profile).strip().lower()
        ):
            raise ValueError("Growth manifest source does not match the registered parent profile")
        parent_path = Path(str(growth_binding.get("parent_checkpoint_path", "")))
        if not parent_path.is_file():
            raise FileNotFoundError(f"Growth parent checkpoint is missing: {parent_path}")
        parent_sha256 = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        if not hmac.compare_digest(
            parent_sha256, str(validated_growth["source_checkpoint_sha256"])
        ):
            raise ValueError("Growth parent checkpoint does not match the growth manifest")
        expected_binding: dict[str, object] = {
            "path": str(growth_path),
            "sha256": actual_growth_hash,
            "parent_profile": registration.parent_profile,
            "parent_checkpoint_path": str(parent_path),
            "parent_checkpoint_sha256": parent_sha256,
            "source_checkpoint_sha256": str(validated_growth["source_checkpoint_sha256"]),
            "source_architecture_sha256": str(
                validated_growth["source_architecture_sha256"]
            ),
            "target_architecture_sha256": str(
                validated_growth["target_architecture_sha256"]
            ),
            "parity_cosine": float(validated_growth["parity_cosine"]),
            "optimizer_restart_required": True,
        }
        if checkpoint_source_kind == "growth_initialization":
            initialization_metadata = _growth_initialization_metadata(
                checkpoint_path,
                checkpoint_sha256=checkpoint_hash,
                growth_binding=expected_binding,
                target_profile=profile,
            )
            if initialization_metadata is None:
                raise ValueError(
                    "Growth initialization launches require a verified metadata sidecar"
                )
            expected_binding["initialization_metadata"] = initialization_metadata
        elif checkpoint_source_kind != "full_resume":
            raise ValueError("A growth child must start from growth initialization or full resume")
        if growth_binding != expected_binding:
            raise ValueError("Growth launch binding does not match its validated manifest")
    elif growth_binding or checkpoint_source_kind == "growth_initialization":
        raise ValueError("Canonical 181M launches cannot bind a growth artifact")
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
