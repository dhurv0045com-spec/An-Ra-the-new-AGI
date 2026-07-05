from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import TypedDict

import torch
from anra.anra_paths import (
    DRIVE_DIR,
    DRIVE_V2_CHECKPOINTS,
    DRIVE_V2_DIR,
    OUTPUT_V2_DIR,
    ROOT,
    TOKENIZER_MANIFEST,
    V2_BRAIN_CHECKPOINT,
    V2_IDENTITY_CHECKPOINT,
    V2_OUROBOROS_CHECKPOINT,
    V3_TOKENIZER_FILE,
    ensure_dirs,
    get_dataset_file,
    get_identity_file,
    get_teacher_files,
    get_v2_checkpoint,
)
from anra_brain import CausalTransformerV2
from runtime.drive_session_manager import DriveSessionManager
from runtime.safe_load import safe_torch_load
from tokenizer.subword_tokenizer import SubwordTokenizer
from tokenizer.tokenizer_adapter import TokenizerAdapter

from training.anra_optimizer import restore_optimizer_state_for_resume
from training.v2_config import (
    CANONICAL_VOCAB_SIZE,
    CHECKPOINT_SCHEMA_VERSION,
    EXPECTED_PAD_TOKEN_ID,
    EXPECTED_SPECIAL_TOKEN_IDS,
    EXPECTED_SPECIAL_TOKENS,
    EXPECTED_TOKENIZER_VOCAB_SIZE,
    TOKENIZER_SCHEMA_VERSION,
    TOKENIZER_V4_VOCAB_SIZE,
    V2_FRONTIER,
    V2_MODEL,
    V2_REPORT_FILES,
    frontier_parameter_count,
    resolve_model_profile,
)

ensure_dirs()
logger = logging.getLogger(__name__)

DRIVE_SESSION_MANAGER = DriveSessionManager(DRIVE_DIR)


def active_tokenizer_path() -> Path:
    configured = os.environ.get("ANRA_TOKENIZER_PATH", "").strip()
    if not configured:
        return V3_TOKENIZER_FILE
    path = Path(configured).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot satisfy the requested model contract."""


class CheckpointLoadReport(TypedDict, total=False):
    loaded_keys: list[str]
    loaded_key_count: int
    target_key_count: int
    missing_keys: list[str]
    unexpected_keys: list[str]
    mismatched_shapes: dict[str, object]
    source_shape_changes: dict[str, object]
    newly_initialized_keys: list[str]
    core_missing_keys: list[str]
    core_mismatched_keys: list[str]
    subsystem_missing_keys: list[str]
    subsystem_mismatched_keys: list[str]
    migration: dict[str, object]
    exact_core_load: bool
    exact_native_load: bool
    all_target_tensors_accounted: bool


def canonical_v2_checkpoint(kind: str = "brain") -> Path:
    mapping = {
        "brain": V2_BRAIN_CHECKPOINT,
        "identity": V2_IDENTITY_CHECKPOINT,
        "ouroboros": V2_OUROBOROS_CHECKPOINT,
    }
    return mapping.get(kind, V2_BRAIN_CHECKPOINT)


def v2_output_file(name: str) -> Path:
    OUTPUT_V2_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_V2_DIR / name


def v2_report_path(key: str) -> Path:
    filename = V2_REPORT_FILES.get(key, key)
    return v2_output_file(filename)


def atomic_save(
    payload: dict, output_path: Path, *, drive_dir: Path | None = DRIVE_V2_CHECKPOINTS
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(output_path)
    if drive_dir is not None:
        try:
            drive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, drive_dir / output_path.name)
        except Exception as exc:
            logger.warning("Drive checkpoint mirror failed for %s: %s", output_path, exc)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_step(path: Path) -> int:
    """Safely read step number from checkpoint without loading full model."""
    try:
        ckpt = safe_torch_load(path, map_location="cpu")
        return int(ckpt.get("step", ckpt.get("global_step", 0)))
    except Exception as exc:
        logger.warning("Could not read checkpoint step from %s: %s", path, exc)
        return 0


def _drive_artifact_path(name: str) -> Path:
    drive_filenames = {
        "brain": "anra_v2_brain.pt",
        "identity": "anra_v2_identity.pt",
        "ouroboros": "anra_v2_ouroboros.pt",
        "tokenizer": "tokenizer_v3.json",
        "eval_summary": "anra_v2_eval_summary.json",
    }
    filename = drive_filenames.get(name, f"anra_v2_{name}.pt")
    # Tokenizer and reports are immutable/small V2 artifacts. Checkpoints use
    # the dedicated checkpoint directory. Each artifact has one canonical
    # Drive path; legacy restore candidates remain supported below.
    if name in {"brain", "identity", "ouroboros"}:
        return DRIVE_V2_CHECKPOINTS / filename
    return DRIVE_V2_DIR / filename


def _local_artifact_path(name: str) -> Path:
    local_map = {
        "brain": V2_BRAIN_CHECKPOINT,
        "identity": V2_IDENTITY_CHECKPOINT,
        "ouroboros": V2_OUROBOROS_CHECKPOINT,
        "tokenizer": V3_TOKENIZER_FILE,
        "eval_summary": v2_report_path("eval_summary"),
    }
    return local_map.get(name, ROOT / f"anra_v2_{name}.pt")


def _drive_restore_candidates(name: str) -> list[Path]:
    canonical = _drive_artifact_path(name)
    candidates = [
        DRIVE_V2_CHECKPOINTS / canonical.name,
        canonical,
    ]
    if name == "tokenizer":
        # Older sessions may keep the canonical tokenizer in legacy checkpoint folders.
        candidates.extend(
            [
                DRIVE_V2_CHECKPOINTS / "tokenizer_v3.json",
                DRIVE_V2_CHECKPOINTS / "tokenizer_v2.json",
                DRIVE_DIR / "tokenizer_v3.json",
                DRIVE_DIR / "tokenizer_v2.json",
            ]
        )
    return candidates


def restore_v2_artifact(name: str = "brain") -> bool:
    """
    Check Drive for checkpoint. If found, copy to local output dir.
    Returns True if restored, False if starting fresh.
    """
    local_path = _local_artifact_path(name)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    source = None
    for candidate in _drive_restore_candidates(name):
        if candidate.exists():
            source = candidate
            break

    if source is None:
        print(f"[Restore] {name}: not on Drive — will start fresh")
        return False

    if local_path.exists() and local_path.stat().st_mtime >= source.stat().st_mtime:
        step = _read_step(local_path) if local_path.suffix == ".pt" else 0
        print(f"[Restore] {name}: already current (step={step})")
        return True

    try:
        shutil.copy2(source, local_path)
    except Exception as exc:
        logger.warning("Drive restore failed for %s from %s: %s", local_path, source, exc)
        return False

    step = _read_step(local_path) if local_path.suffix == ".pt" else 0
    print(f"[Restore] {name}: restored from Drive (step={step})")
    return True


def sync_to_drive(name: str = "brain") -> bool:
    """
    Sync one legacy artifact to its canonical Drive path.

    This compatibility helper intentionally writes one file, not the old pair
    of root and checkpoint-directory copies. Frontier training checkpoints use
    the shared-master publisher instead.
    """
    local_path = _local_artifact_path(name)
    if not local_path.exists():
        print(f"[Drive] {name}: local file not found, skipping")
        return False

    drive_target = _drive_artifact_path(name)

    try:
        drive_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, drive_target)
        step = _read_step(local_path) if local_path.suffix == ".pt" else 0
        size_kb = local_path.stat().st_size // 1024
        print(f"[Drive] {name}: saved (step={step}, {size_kb}KB)")
        return True
    except Exception as e:
        print(f"[Drive] {name}: save failed ({e})")
        return False


def sync_v2_artifacts(
    checkpoint_path: Path,
    *,
    tokenizer_path: Path | None = None,
    extra_paths: list[Path] | None = None,
) -> None:
    del checkpoint_path, tokenizer_path
    sync_to_drive("brain")
    sync_to_drive("tokenizer")
    for extra in extra_paths or []:
        if extra.name == "v2_eval_summary.json":
            target = get_v2_checkpoint("brain").parent / "anra_v2_eval_summary.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extra, target)
            sync_to_drive("eval_summary")


def _collect_tokenizer_texts(dataset_path: Path) -> list[str]:
    texts = [dataset_path.read_text(encoding="utf-8", errors="replace")]
    identity_path = get_identity_file()
    if identity_path is not None and identity_path.exists():
        texts.append(identity_path.read_text(encoding="utf-8", errors="replace"))
    for teacher_path in get_teacher_files():
        lines = teacher_path.read_text(encoding="utf-8", errors="replace").splitlines()
        texts.extend(line for line in lines if line.strip())
    return texts


def load_or_build_v2_tokenizer(
    *,
    dataset_path: Path | None = None,
    vocab_size: int = EXPECTED_TOKENIZER_VOCAB_SIZE,
) -> SubwordTokenizer:
    dataset_path = dataset_path or get_dataset_file()
    local = active_tokenizer_path()
    if local.exists():
        if local == V3_TOKENIZER_FILE:
            _migrate_tokenizer_surface(local)
        tokenizer = SubwordTokenizer.load(local)
        assert_tokenizer_contract(local, tokenizer)
        return tokenizer
    if local != V3_TOKENIZER_FILE:
        raise FileNotFoundError(f"Configured append-only tokenizer does not exist: {local}")
    restored = restore_v2_artifact("tokenizer")
    if restored and local.exists():
        _migrate_tokenizer_surface(local)
        tokenizer = SubwordTokenizer.load(local)
        assert_tokenizer_contract(local, tokenizer)
        return tokenizer

    texts = _collect_tokenizer_texts(dataset_path)
    print(f"[build_brain] Building tokenizer_v3 from {dataset_path} ...", flush=True)
    tokenizer = SubwordTokenizer.train_from_texts(texts, vocab_size=vocab_size)
    tokenizer.save(local)
    assert_tokenizer_contract(local, tokenizer)
    try:
        drive_tok = DRIVE_V2_DIR / local.name
        drive_tok.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, drive_tok)
    except Exception as exc:
        logger.warning("Drive tokenizer mirror failed for %s: %s", local, exc)
    print(
        f"[build_brain] tokenizer_v3 built + mirrored to Drive. vocab_size={tokenizer.vocab_size}",
        flush=True,
    )
    return tokenizer


def _migrate_tokenizer_surface(path: Path) -> bool:
    """Append V3 controls at IDs 8192..8208 while preserving all legacy IDs."""
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("backend") != "fallback":
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        id_to_token = list(payload["id_to_token"])
        token_to_id = dict(payload["token_to_id"])
    except Exception:
        return False
    current = {token: token_to_id.get(token) for token in EXPECTED_SPECIAL_TOKENS}
    if current == EXPECTED_SPECIAL_TOKEN_IDS:
        return False
    if len(id_to_token) < EXPECTED_TOKENIZER_VOCAB_SIZE:
        return False
    for token, expected_id in EXPECTED_SPECIAL_TOKEN_IDS.items():
        if expected_id < 13:
            if id_to_token[expected_id] != token:
                raise AssertionError(
                    f"Legacy tokenizer row {expected_id} is {id_to_token[expected_id]!r}, "
                    f"expected {token!r}"
                )
            continue
        previous = id_to_token[expected_id]
        token_to_id.pop(previous, None)
        id_to_token[expected_id] = token
        token_to_id[token] = expected_id
    migrated = {"token_to_id": token_to_id, "id_to_token": id_to_token}
    temporary = path.with_suffix(path.suffix + ".migrating")
    temporary.write_text(json.dumps(migrated, indent=2), encoding="utf-8")
    temporary.replace(path)
    meta.update(
        {
            "schema_version": TOKENIZER_SCHEMA_VERSION,
            "vocab_size": EXPECTED_TOKENIZER_VOCAB_SIZE,
            "special_tokens": EXPECTED_SPECIAL_TOKENS,
            "special_token_ids": EXPECTED_SPECIAL_TOKEN_IDS,
            "migration": {
                "source_vocab_size": 8192,
                "appended_rows": [8192, 8208],
                "legacy_rows_preserved": True,
            },
        }
    )
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta_tmp.replace(meta_path)
    return True


def assert_tokenizer_contract(path: Path, tokenizer: SubwordTokenizer) -> None:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    vocab_size = int(meta.get("vocab_size", tokenizer.vocab_size))
    special_tokens = list(meta.get("special_tokens", tokenizer.special_tokens))
    allowed_vocab_sizes = {EXPECTED_TOKENIZER_VOCAB_SIZE, TOKENIZER_V4_VOCAB_SIZE}
    if vocab_size not in allowed_vocab_sizes:
        raise AssertionError(
            f"Tokenizer contract mismatch: vocab_size={vocab_size}, "
            f"expected one of {sorted(allowed_vocab_sizes)} "
            f"(meta={meta_path})"
        )
    schema_version = int(meta.get("schema_version", TOKENIZER_SCHEMA_VERSION))
    if vocab_size == TOKENIZER_V4_VOCAB_SIZE and schema_version != 4:
        raise AssertionError("A 16,384-token tokenizer must declare schema_version=4")
    if special_tokens != EXPECTED_SPECIAL_TOKENS:
        raise AssertionError(
            f"Tokenizer contract mismatch: special_tokens={special_tokens}, "
            f"expected={EXPECTED_SPECIAL_TOKENS} "
            f"(meta={meta_path})"
        )
    if tokenizer.vocab_size != vocab_size:
        raise AssertionError(
            f"Tokenizer contract mismatch: tokenizer.vocab_size={tokenizer.vocab_size}, "
            f"metadata={vocab_size} ({path})"
        )
    if tokenizer.pad_token_id != EXPECTED_PAD_TOKEN_ID:
        raise AssertionError(
            f"Tokenizer contract mismatch: pad_token_id={tokenizer.pad_token_id}, "
            f"expected={EXPECTED_PAD_TOKEN_ID} ({path})"
        )
    token_to_id = getattr(tokenizer, "token_to_id", {})
    missing = [
        (expected_id, token)
        for token, expected_id in EXPECTED_SPECIAL_TOKEN_IDS.items()
        if token_to_id.get(token) != expected_id
    ]
    if missing:
        raise AssertionError(f"Tokenizer special-token ID mismatch: {missing[:5]}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    probe_sha256 = _tokenizer_probe_fingerprint(tokenizer)
    payload = json.loads(path.read_text(encoding="utf-8"))
    vocabulary = payload.get("token_to_id", {}) if isinstance(payload, dict) else {}
    vocabulary_sha256 = hashlib.sha256(
        json.dumps(vocabulary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": schema_version,
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "special_tokens": list(EXPECTED_SPECIAL_TOKENS),
        "special_token_ids": {token: int(token_to_id[token]) for token in EXPECTED_SPECIAL_TOKENS},
        "tokenizer_path": str(path),
        "tokenizer_sha256": digest,
        "vocabulary_sha256": vocabulary_sha256,
        "probe_count": 500,
        "probe_sha256": probe_sha256,
    }
    TOKENIZER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = TOKENIZER_MANIFEST.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(TOKENIZER_MANIFEST)


def _tokenizer_probe_fingerprint(
    tokenizer: SubwordTokenizer,
    count: int = 500,
) -> str:
    """Fingerprint fixed encode/decode behavior, not JSON formatting."""
    encoded_probes: list[list[int]] = []
    for index in range(count):
        text = (
            f"H: An-Ra tokenizer probe {index:03d}: "
            f"code_{index % 17} = ({index % 97} + {index % 31}); "
            f"logic, math, science, memory.\nANRA: verified {index % 11}."
        )
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        if tokenizer.encode(decoded, add_special_tokens=False) != encoded:
            raise AssertionError(f"Tokenizer encode/decode probe {index} is not ID-stable")
        encoded_probes.append([int(token_id) for token_id in encoded])
    material = json.dumps(encoded_probes, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _active_tokenizer_identity() -> dict[str, object]:
    path = active_tokenizer_path()
    if not path.exists():
        return {"available": False}
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    vocabulary = payload.get("token_to_id", {}) if isinstance(payload, dict) else {}
    tokenizer = SubwordTokenizer.load(path)
    return {
        "available": True,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "vocabulary_sha256": hashlib.sha256(
            json.dumps(vocabulary, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "probe_count": 500,
        "probe_sha256": _tokenizer_probe_fingerprint(tokenizer),
    }


def assert_model_tokenizer_contract(
    model: CausalTransformerV2, tokenizer: SubwordTokenizer
) -> None:
    """Ensure model embeddings and tokenizer IDs share the same training contract."""
    if tokenizer.vocab_size != model.vocab_size:
        raise AssertionError(
            f"Tokenizer/model vocab mismatch: tokenizer.vocab_size={tokenizer.vocab_size}, "
            f"model.vocab_size={model.vocab_size}"
        )
    if tokenizer.pad_token_id != model.pad_token_id:
        raise AssertionError(
            f"Tokenizer/model pad mismatch: tokenizer.pad_token_id={tokenizer.pad_token_id}, "
            f"model.pad_token_id={model.pad_token_id}"
        )


def _checkpoint_vocab_size(model_state: dict[str, torch.Tensor]) -> int | None:
    for key, weight in model_state.items():
        if (
            (key.endswith("token_embedding_table.weight") or key.endswith("lm_head.weight"))
            and isinstance(weight, torch.Tensor)
            and weight.ndim >= 2
        ):
            return int(weight.shape[0])
    return None


def _adapt_state_vocab_rows(
    model_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Expand tokenizer tensors while preserving legacy rows bit-for-bit."""
    adapted = dict(model_state)
    for key, old_weight in model_state.items():
        target_weight = target_state.get(key)
        if (
            isinstance(old_weight, torch.Tensor)
            and isinstance(target_weight, torch.Tensor)
            and old_weight.ndim >= 2
            and target_weight.ndim == old_weight.ndim
            and old_weight.shape[1:] == target_weight.shape[1:]
            and old_weight.shape[0] < target_weight.shape[0]
            and (key.endswith("token_embedding_table.weight") or key.endswith("lm_head.weight"))
        ):
            new_weight = torch.empty_like(target_weight)
            legacy = old_weight.to(device=new_weight.device, dtype=new_weight.dtype)
            new_weight[: old_weight.shape[0]] = legacy
            appended_count = target_weight.shape[0] - old_weight.shape[0]
            scale = legacy.float().std().clamp_min(1e-8) * 0.01
            rows = torch.arange(
                1,
                appended_count + 1,
                device=new_weight.device,
                dtype=torch.float32,
            ).unsqueeze(1)
            columns = torch.arange(
                1,
                legacy.shape[1] + 1,
                device=new_weight.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            offsets = torch.sin(rows * columns * 0.017453292519943295)
            offsets = offsets - offsets.mean(dim=1, keepdim=True)
            offsets = offsets / offsets.norm(dim=1, keepdim=True).clamp_min(1e-8)
            tokenizer_payload = json.loads(active_tokenizer_path().read_text(encoding="utf-8"))
            tokenizer_meta_path = active_tokenizer_path().with_suffix(
                active_tokenizer_path().suffix + ".meta.json"
            )
            tokenizer_meta = (
                json.loads(tokenizer_meta_path.read_text(encoding="utf-8"))
                if tokenizer_meta_path.is_file()
                else {}
            )
            decompositions = tokenizer_meta.get("token_decompositions", {})
            id_to_token = list(tokenizer_payload.get("id_to_token", []))
            token_to_id = dict(tokenizer_payload.get("token_to_id", {}))
            bases: list[torch.Tensor] = []
            for row_index in range(old_weight.shape[0], target_weight.shape[0]):
                token = id_to_token[row_index] if row_index < len(id_to_token) else ""
                declared = decompositions.get(token, []) if isinstance(decompositions, dict) else []
                constituent_ids = [
                    int(value)
                    for value in declared
                    if isinstance(value, int) and 0 <= int(value) < old_weight.shape[0]
                ]
                position = 0
                while not constituent_ids and position < len(token):
                    matched_id = None
                    matched_end = position
                    for end in range(len(token), position, -1):
                        candidate_id = token_to_id.get(token[position:end])
                        if candidate_id is not None and int(candidate_id) < old_weight.shape[0]:
                            matched_id = int(candidate_id)
                            matched_end = end
                            break
                    if matched_id is None:
                        position += 1
                    else:
                        constituent_ids.append(matched_id)
                        position = matched_end
                if not constituent_ids:
                    constituent_ids = [min(1, legacy.shape[0] - 1)]
                bases.append(legacy[constituent_ids].float().mean(dim=0))
            base_rows = torch.stack(bases, dim=0)
            appended = base_rows + offsets * scale
            new_weight[old_weight.shape[0] :] = appended.to(dtype=new_weight.dtype)
            adapted[key] = new_weight
    return adapted


def migrate_checkpoint_state(
    model_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Apply versioned, deterministic state migrations before loading."""
    migrated = _adapt_state_vocab_rows(model_state, target_state)
    changes: list[str] = []
    if "dstp_logits" in migrated and "residual_depth_logits" in target_state:
        migrated["residual_depth_logits"] = migrated.pop("dstp_logits")
        changes.append("dstp_logits->residual_depth_logits")
    if "dstp_temperature_log" not in migrated and "dstp_temperature_log" in target_state:
        migrated["dstp_temperature_log"] = target_state["dstp_temperature_log"].detach().clone()
        changes.append("initialize_dstp_temperature_log")
    neutral_native_prefixes = (
        "esv_module.",
        "rim_modules.",
        "mod_routers.",
    )
    neutral_native_exact = {
        "embedding_input_scale",
        "residual_depth_logits",
        "layer_temperature_bias",
    }
    for key in sorted(target_state):
        if key in migrated:
            continue
        if key not in neutral_native_exact and not key.startswith(neutral_native_prefixes):
            continue
        value = target_state[key].detach().clone()
        if key.endswith("mod_routers") or ".gate.weight" in key:
            value.zero_()
        migrated[key] = value
        changes.append(f"initialize_native:{key}")
    report = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tokenizer_schema_version": (
            4
            if _checkpoint_vocab_size(target_state) == TOKENIZER_V4_VOCAB_SIZE
            else TOKENIZER_SCHEMA_VERSION
        ),
        "changes": changes,
        "source_vocab_size": _checkpoint_vocab_size(model_state),
        "target_vocab_size": _checkpoint_vocab_size(target_state),
        "vocabulary_initialization": "declared-decomposition-mean-plus-deterministic-sinusoid-v2",
        "legacy_rows_preserved": True,
    }
    source_vocab = report["source_vocab_size"]
    target_vocab = report["target_vocab_size"]
    if isinstance(source_vocab, int) and isinstance(target_vocab, int):
        report["appended_token_rows"] = max(0, target_vocab - source_vocab)
    return migrated, report


def _load_state_with_base_fallback(
    model: CausalTransformerV2,
    model_state: dict[str, torch.Tensor],
    *,
    strict: bool,
) -> object:
    try:
        return model.load_state_dict(model_state, strict=strict)
    except RuntimeError as exc:
        base = getattr(model, "model", None)
        if base is None:
            raise exc
        base_migrated, _ = migrate_checkpoint_state(model_state, base.state_dict())
        return base.load_state_dict(base_migrated, strict=strict)


def checkpoint_load_report(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    migrated_state: dict[str, torch.Tensor],
    incompatible: object,
    migration: dict[str, object],
) -> CheckpointLoadReport:
    """Describe every tensor disposition instead of treating strict=False as proof."""
    source_keys = set(source_state)
    target_keys = set(target_state)
    migrated_keys = set(migrated_state)
    source_shape_changes = {
        key: {
            "checkpoint": list(source_state[key].shape),
            "model": list(target_state[key].shape),
        }
        for key in sorted(source_keys & target_keys)
        if isinstance(source_state[key], torch.Tensor)
        and isinstance(target_state[key], torch.Tensor)
        and source_state[key].shape != target_state[key].shape
    }
    mismatched_shapes = {
        key: {
            "migrated": list(migrated_state[key].shape),
            "model": list(target_state[key].shape),
        }
        for key in sorted(migrated_keys & target_keys)
        if isinstance(migrated_state[key], torch.Tensor)
        and isinstance(target_state[key], torch.Tensor)
        and migrated_state[key].shape != target_state[key].shape
    }
    runtime_missing = list(getattr(incompatible, "missing_keys", []))
    runtime_unexpected = list(getattr(incompatible, "unexpected_keys", []))
    missing = sorted((target_keys - migrated_keys) | set(runtime_missing))
    unexpected = sorted((migrated_keys - target_keys) | set(runtime_unexpected))
    initialized = sorted(target_keys - source_keys - set(missing))
    loaded = sorted(
        key
        for key in target_keys & migrated_keys
        if key not in mismatched_shapes and key not in missing
    )
    core_prefixes = (
        "token_embedding_table",
        "token_embedding",
        "embedding_input_scale",
        "lm_head",
        "blocks.",
        "norm_f",
    )
    subsystem_prefixes = (
        "esv_module.",
        "rim_modules.",
        "mod_routers.",
        "residual_depth_logits",
        "dstp_temperature_log",
        "layer_temperature_bias",
    )
    core_missing = [key for key in missing if key.startswith(core_prefixes)]
    core_mismatched = [key for key in mismatched_shapes if key.startswith(core_prefixes)]
    subsystem_missing = [key for key in missing if key.startswith(subsystem_prefixes)]
    subsystem_mismatched = [key for key in mismatched_shapes if key.startswith(subsystem_prefixes)]
    return {
        "loaded_keys": loaded,
        "loaded_key_count": len(loaded),
        "target_key_count": len(target_keys),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_shapes": mismatched_shapes,
        "source_shape_changes": source_shape_changes,
        "newly_initialized_keys": initialized,
        "core_missing_keys": core_missing,
        "core_mismatched_keys": core_mismatched,
        "subsystem_missing_keys": subsystem_missing,
        "subsystem_mismatched_keys": subsystem_mismatched,
        "migration": migration,
        "exact_core_load": not core_missing and not core_mismatched,
        "exact_native_load": not subsystem_missing and not subsystem_mismatched,
        "all_target_tensors_accounted": (not missing and not mismatched_shapes and not unexpected),
    }


def _load_hal(config: object) -> object | None:
    if not getattr(config, "use_hal", False):
        return None
    try:
        from anra.anra_paths import DRIVE_LOGS, HAL_STATE_FILE

        from identity.hal import HALModule

        drive_path = DRIVE_LOGS / "hal_state.json"
        path = drive_path if drive_path.exists() else HAL_STATE_FILE
        return HALModule.load(str(path)) if path.exists() else HALModule()
    except Exception as exc:
        logger.warning("HAL initialization failed: %s", exc)
        return None


def build_model_from_config(
    config: object,
    *,
    hal_module: object | None = None,
    block_size: int | None = None,
    vocab_size: int | None = None,
) -> CausalTransformerV2:
    effective_vocab_size = int(vocab_size or config.vocab_size)
    if effective_vocab_size not in {
        EXPECTED_TOKENIZER_VOCAB_SIZE,
        TOKENIZER_V4_VOCAB_SIZE,
    }:
        raise AssertionError(
            "Canonical model vocab must preserve IDs 0-8208 and use either "
            f"{EXPECTED_TOKENIZER_VOCAB_SIZE} or {TOKENIZER_V4_VOCAB_SIZE} rows; "
            f"got {effective_vocab_size}"
        )
    hal_module = _load_hal(config) if hal_module is None else hal_module
    effective_block_size = int(block_size or config.block_size)
    model = CausalTransformerV2(
        vocab_size=effective_vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        n_layer=config.n_layer,
        block_size=effective_block_size,
        rms_norm_eps=config.rms_norm_eps,
        dropout=config.dropout,
        mod_layers=set(config.mod_layers),
        base_seq_len=config.base_seq_len,
        target_seq_len=max(int(config.target_seq_len), effective_block_size),
        pad_token_id=config.pad_token_id,
        use_layer_temperature_bias=True,
        use_hal=config.use_hal,
        hal_module=hal_module,
        use_rim=True,
        use_dstp=True,
    )
    if getattr(config, "gradient_checkpointing", config.n_layer >= 36):
        model.gradient_checkpointing_enable()
    model.disable_kv_cache()
    return model


def build_v2_model(
    *, vocab_size: int, block_size: int = V2_MODEL.block_size
) -> CausalTransformerV2:
    if vocab_size not in {EXPECTED_TOKENIZER_VOCAB_SIZE, TOKENIZER_V4_VOCAB_SIZE}:
        raise AssertionError(
            f"Model/tokenizer vocab mismatch at construction: vocab_size={vocab_size}, "
            f"expected one of {{{V2_MODEL.vocab_size}, {EXPECTED_TOKENIZER_VOCAB_SIZE}}}"
        )
    return build_model_from_config(V2_MODEL, block_size=block_size, vocab_size=vocab_size)


def build_frontier_model(
    *,
    hal_module: object | None = None,
    block_size: int | None = None,
    vocab_size: int | None = None,
) -> CausalTransformerV2:
    """
    Build the branch frontier model from V2_FRONTIER config.
    KV cache is disabled for training. HAL may be None.
    """
    cfg = V2_FRONTIER

    if cfg.vocab_size not in {EXPECTED_TOKENIZER_VOCAB_SIZE, CANONICAL_VOCAB_SIZE}:
        raise AssertionError(
            f"frontier vocab mismatch: config={cfg.vocab_size} "
            f"tokenizer={EXPECTED_TOKENIZER_VOCAB_SIZE}"
        )

    if block_size is not None and int(block_size) not in {1024, 1536, 2048}:
        raise ValueError("Frontier context length must be 1024, 1536, or 2048")
    model = build_model_from_config(
        cfg,
        hal_module=hal_module,
        block_size=block_size,
        vocab_size=vocab_size,
    )

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("  [build_frontier_model] Gradient checkpointing: ENABLED")
    else:
        model.gradient_checkpointing = True
        print("  [build_frontier_model] Gradient checkpointing flag: ENABLED")

    if hasattr(model, "disable_kv_cache"):
        model.disable_kv_cache()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[build_frontier_model] Built {total_params / 1e6:.0f}M param model")
    expected_params = frontier_parameter_count(model.vocab_size)
    if total_params != expected_params:
        print(
            f"[build_frontier_model] WARNING: expected {expected_params:,} "
            f"params for this branch, got {total_params:,}"
        )
    print(
        f"  n_embd={cfg.n_embd}  n_layer={cfg.n_layer}  "
        f"n_head={cfg.n_head}  n_kv_head={cfg.n_kv_head}"
    )
    print(f"  block_size={model.block_size}  vocab={cfg.vocab_size}")
    print(f"  HAL: {'enabled' if cfg.use_hal else 'disabled'}")
    return model


def build_model_for_profile(
    profile: str,
    *,
    hal_module: object | None = None,
    block_size: int | None = None,
    vocab_size: int | None = None,
) -> CausalTransformerV2:
    config, _ = resolve_model_profile(profile)
    return build_model_from_config(
        config,
        hal_module=hal_module,
        block_size=block_size,
        vocab_size=vocab_size,
    )


def load_checkpoint(
    model: CausalTransformerV2,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    mp_trainer: object | None,
    checkpoint_path: Path,
    *,
    device: torch.device,
    strict: bool = False,
) -> dict[str, object]:
    state = {
        "loaded": False,
        "global_step": 0,
        "epoch": 0,
        "best_loss": float("inf"),
        "sessions_completed": 0,
        "data_profile": "unknown",
        "training_data_layout": "unknown",
        "tokens_seen": 0,
        "unique_token_ids_seen": [],
        "continuation_token_counts": {},
        "best_validation_loss": float("inf"),
        "validation_history": [],
        "appended_row_optimizer_steps": 0,
        "raw_window_consumption": {},
        "tokenizer_identity": {"verified": False, "reason": "checkpoint_not_loaded"},
    }
    ckpt = checkpoint_path
    if not ckpt.exists():
        kind = "brain"
        if "identity" in ckpt.name:
            kind = "identity"
        elif "ouroboros" in ckpt.name:
            kind = "ouroboros"
        restored = restore_v2_artifact(kind)
        if restored:
            ckpt = get_v2_checkpoint(kind)
    if not ckpt.exists():
        return state

    blob = safe_torch_load(ckpt, map_location=device)
    model_state = (
        blob.get("model_state_dict", blob.get("model", blob)) if isinstance(blob, dict) else blob
    )
    if isinstance(model_state, dict):
        ckpt_vocab_size = _checkpoint_vocab_size(model_state)
        if ckpt_vocab_size is not None and ckpt_vocab_size > model.vocab_size:
            raise AssertionError(
                f"Checkpoint/model vocab mismatch: checkpoint vocab_size={ckpt_vocab_size}, "
                f"model.vocab_size={model.vocab_size} ({ckpt})"
            )
    if isinstance(blob, dict):
        ckpt_config = blob.get("model_config", {})
        if isinstance(ckpt_config, dict):
            ckpt_pad = ckpt_config.get("pad_token_id")
            if ckpt_pad is not None and int(ckpt_pad) != model.pad_token_id:
                raise AssertionError(
                    f"Checkpoint/model pad mismatch: checkpoint pad_token_id={ckpt_pad}, "
                    f"model.pad_token_id={model.pad_token_id} ({ckpt})"
                )
        saved_tokenizer = blob.get("tokenizer_contract", {})
        active_tokenizer = _active_tokenizer_identity()
        tokenizer_identity = {
            "verified": False,
            "saved": saved_tokenizer,
            "active": active_tokenizer,
            "reason": "legacy_checkpoint_missing_tokenizer_fingerprint",
        }
        if isinstance(saved_tokenizer, dict):
            comparisons = {
                key: (saved_tokenizer.get(key), active_tokenizer.get(key))
                for key in ("vocabulary_sha256", "probe_sha256")
                if saved_tokenizer.get(key)
            }
            mismatches = {
                key: {"checkpoint": values[0], "active": values[1]}
                for key, values in comparisons.items()
                if values[0] != values[1]
            }
            if mismatches:
                raise CheckpointCompatibilityError(
                    f"Checkpoint tokenizer IDs differ from the active tokenizer: {mismatches}"
                )
            if len(comparisons) == 2:
                tokenizer_identity["verified"] = True
                tokenizer_identity["reason"] = "vocabulary_and_500_probes_match"
            tokenizer_identity["file_sha256_matches"] = (
                saved_tokenizer.get("sha256") == active_tokenizer.get("sha256")
                if saved_tokenizer.get("sha256")
                else None
            )
        state["tokenizer_identity"] = tokenizer_identity
    try:
        target_state = model.state_dict()
        migrated_state, migration = migrate_checkpoint_state(model_state, target_state)
        incompatible = _load_state_with_base_fallback(model, migrated_state, strict=strict)
        load_report = checkpoint_load_report(
            model_state,
            target_state,
            migrated_state,
            incompatible,
            migration,
        )
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"Checkpoint {ckpt} is incompatible with the requested model architecture."
        ) from exc
    if isinstance(blob, dict):
        if optimizer is not None:
            try:
                repaired = restore_optimizer_state_for_resume(
                    optimizer,
                    blob.get("optimizer_state_dict", blob.get("optimizer", {})),
                )
                if repaired:
                    logger.warning(
                        "Using safe optimizer resume policy for %s: %s",
                        ckpt,
                        ", ".join(repaired),
                    )
            except Exception as exc:
                logger.warning("Optimizer state restore failed from %s: %s", ckpt, exc)
        if scheduler is not None:
            try:
                scheduler.load_state_dict(
                    blob.get("scheduler_state_dict", blob.get("scheduler", {}))
                )
            except Exception as exc:
                logger.warning("Scheduler state restore failed from %s: %s", ckpt, exc)
        if mp_trainer is not None:
            try:
                scaler_state = blob.get("scaler_state_dict", blob.get("scaler"))
                if scaler_state:
                    mp_trainer.load_state_dict(scaler_state)
            except Exception as exc:
                logger.warning("Mixed-precision scaler restore failed from %s: %s", ckpt, exc)
        state["global_step"] = int(blob.get("global_step", blob.get("step", 0)))
        state["epoch"] = int(blob.get("epoch", 0))
        state["best_loss"] = float(blob.get("best_loss", float("inf")))
        state["sessions_completed"] = int(blob.get("sessions_completed", 0))
        state["data_profile"] = str(blob.get("data_profile", "unknown"))
        state["training_data_layout"] = str(blob.get("training_data_layout", "unknown"))
        state["tokens_seen"] = int(blob.get("tokens_seen", 0))
        state["unique_token_ids_seen"] = list(blob.get("unique_token_ids_seen", []))
        state["continuation_token_counts"] = dict(blob.get("continuation_token_counts", {}))
        state["best_validation_loss"] = float(blob.get("best_validation_loss", float("inf")))
        state["validation_history"] = list(blob.get("validation_history", []))
        state["appended_row_optimizer_steps"] = int(blob.get("appended_row_optimizer_steps", 0))
        state["raw_window_consumption"] = dict(blob.get("raw_window_consumption", {}))
        state["data_manifests"] = dict(
            blob.get("data_manifests", blob.get("dataset_manifest_hashes", {}))
        )
        state["data_manifest_payloads"] = dict(blob.get("data_manifest_payloads", {}))
        state["model_config"] = dict(blob.get("model_config", {}))
        state["source_commit"] = str(blob.get("source_commit", "unknown"))
        restore_hal_state(model, blob.get("hal_state", {}))
    state["loaded"] = True
    state["migration"] = migration
    state["load_report"] = load_report
    return state


@torch.no_grad()
def tokenizer_special_ids(tokenizer: object) -> dict[str, int]:
    """Return special token IDs for both tokenizer surfaces used by AN-RA."""
    special_attr = getattr(tokenizer, "special_ids", None)
    if callable(special_attr):
        special = dict(special_attr())
    elif isinstance(special_attr, dict):
        special = dict(special_attr)
    else:
        special = {}

    if "<bos>" not in special and hasattr(tokenizer, "bos_token_id"):
        special["<bos>"] = int(tokenizer.bos_token_id)
    if "<eos>" not in special and hasattr(tokenizer, "eos_token_id"):
        special["<eos>"] = int(tokenizer.eos_token_id)
    if "<pad>" not in special and hasattr(tokenizer, "pad_token_id"):
        special["<pad>"] = int(tokenizer.pad_token_id)
    if "<unk>" not in special and hasattr(tokenizer, "unk_token_id"):
        special["<unk>"] = int(tokenizer.unk_token_id)

    missing = [token for token in ("<bos>", "<eos>") if token not in special]
    if missing:
        raise TypeError(f"Tokenizer is missing required special token IDs: {missing}")
    return special


@torch.no_grad()
def generate_text(
    model: CausalTransformerV2,
    tokenizer: TokenizerAdapter,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 96,
    temperature: float = 0.9,
    top_k: int = 40,
    greedy: bool = False,
    seed: int | None = None,
) -> str:
    """Generate a continuation.

    ``greedy=True`` selects argmax decoding, the deterministic recovery
    baseline. Otherwise sampling uses a local generator seeded by ``seed``
    when provided, so evaluation evidence replays exactly without mutating
    the global RNG state.
    """
    model.eval()
    special = tokenizer_special_ids(tokenizer)
    ids = [special["<bos>"]] + tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generator: torch.Generator | None = None
    if seed is not None and not greedy:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.block_size :]
        logits, _ = model(x_cond)
        next_logits = logits[:, -1, :] / max(temperature, 1e-4)
        if top_k > 0:
            values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits = next_logits.masked_fill(next_logits < values[:, [-1]], float("-inf"))
        if greedy:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        x = torch.cat([x, next_token], dim=1)
        if int(next_token.item()) == special["<eos>"]:
            break
    prompt_token_count = 1 + len(tokenizer.encode(prompt))
    answer_ids = x[0].tolist()[prompt_token_count:]
    return tokenizer.decode(answer_ids).strip()


def model_summary(model: torch.nn.Module) -> dict[str, int]:
    return {
        "parameters": sum(param.numel() for param in model.parameters()),
        "trainable_parameters": sum(
            param.numel() for param in model.parameters() if param.requires_grad
        ),
    }


def ensure_tied_lm_head(model: torch.nn.Module) -> bool:
    """
    Re-assert AN-RA's tied embedding/LM-head contract after device moves.

    Some accelerator backends can materialize the tied Parameter as a separate
    object during ``model.to(device)``. The architecture still expects tied
    embeddings, so training entrypoints call this after moving the model.
    """
    target = getattr(model, "model", model)
    embedding = getattr(target, "token_embedding_table", None)
    lm_head = getattr(target, "lm_head", None)
    if embedding is None or lm_head is None:
        return False
    if getattr(lm_head, "weight", None) is not getattr(embedding, "weight", None):
        lm_head.weight = embedding.weight
    return getattr(lm_head, "weight", None) is getattr(embedding, "weight", None)


def hal_state_dict(model: torch.nn.Module) -> dict[str, object]:
    target = getattr(model, "model", model)
    hal = getattr(target, "hal_module", None)
    state = getattr(hal, "state", None)
    if state is None:
        return {}
    if hasattr(state, "__dataclass_fields__"):
        return {name: getattr(state, name) for name in state.__dataclass_fields__}
    return dict(getattr(state, "__dict__", {}))


def restore_hal_state(model: torch.nn.Module, payload: object) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    target = getattr(model, "model", model)
    hal = getattr(target, "hal_module", None)
    state = getattr(hal, "state", None)
    if state is None:
        return False
    for key, value in payload.items():
        if not hasattr(state, key):
            continue
        current = getattr(state, key)
        try:
            setattr(state, key, type(current)(value))
        except Exception:
            setattr(state, key, value)
    return True


def get_hal_module(model: torch.nn.Module) -> object | None:
    target = getattr(model, "model", model)
    return getattr(target, "hal_module", None)


def update_hal_from_training(
    model: torch.nn.Module,
    *,
    loss: float,
    best_loss: float,
    gradient_norm: float,
    step: int,
) -> dict[str, object]:
    hal = get_hal_module(model)
    if hal is None:
        return {}
    improved = float(loss) <= float(best_loss) if math.isfinite(float(best_loss)) else True
    verifier_score = max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, float(loss)))))
    context = {
        "training_step": int(step),
        "loss_improved": bool(improved),
        "near_capability_boundary": bool(float(loss) < 1.5),
        "model_incoherence_self_detected": bool(not math.isfinite(float(loss))),
        "high_gradient_norm": bool(
            math.isfinite(float(gradient_norm)) and float(gradient_norm) > 1.0
        ),
    }
    if hasattr(hal, "update"):
        try:
            hal.update(
                verifier_result=verifier_score,
                civ_score=None,
                session_context=context,
                decay_turns=1 if step % 10 == 0 else 0,
            )
        except TypeError:
            hal.update(verifier_score, None, context)
    elif hasattr(hal, "decay"):
        hal.decay(1 if step % 10 == 0 else 0)
    return hal_state_dict(model)


def load_session_state() -> dict[str, object]:
    path = v2_report_path("session_state")
    if not path.exists():
        return {"successful_sessions": 0, "eval_scores": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Session state read failed from %s: %s", path, exc)
        return {"successful_sessions": 0, "eval_scores": []}


def update_session_state(*, eval_score: float | None = None) -> dict[str, object]:
    state = load_session_state()
    state["successful_sessions"] = int(state.get("successful_sessions", 0)) + 1
    scores = list(state.get("eval_scores", []))
    if eval_score is not None and not math.isnan(eval_score):
        scores.append({"score": float(eval_score), "ts": time.time()})
    state["eval_scores"] = scores[-12:]
    write_json(v2_report_path("session_state"), state)
    return state
