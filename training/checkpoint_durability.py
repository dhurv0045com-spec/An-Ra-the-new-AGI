"""Immutable, resumable publication for An-Ra training checkpoints.

The trainer owns the checkpoint payload.  This module owns its durable life
after the local atomic save: content-addressed chunks, an immutable manifest,
verified replica receipts, and a canonical pointer that is never published
before every referenced byte has been verified.

Mounted Google Drive and ordinary filesystem replicas deliberately share the
same backend.  The byte-level contract does not depend on a cloud SDK, which
makes it usable from Colab, a laptop sync folder, or the cluster publisher.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import queue
import shutil
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import torch

DURABILITY_SCHEMA_VERSION = 1
DEFAULT_CHUNK_SIZE_BYTES = 128 * 1024 * 1024
DEFAULT_COPY_BLOCK_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_COPY_STREAMS = 2
HOT_STORAGE_LIMIT_BYTES = 12 * 1024**3
DEFAULT_KEEP_FULL = 2
DEFAULT_KEEP_COMPACT = 2

FULL_RESUME = "full_resume"
FP16_INFERENCE = "fp16_inference"


class ArtifactClass(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    FULL_RESUME = FULL_RESUME
    FP16_INFERENCE = FP16_INFERENCE


class DurabilityState(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    LOCAL_SAVED = "local_saved"
    STAGED = "staged"
    CANONICAL_VERIFIED = "canonical_verified"
    PROTECTED = "protected"


STATE_ORDER = {
    DurabilityState.LOCAL_SAVED: 0,
    DurabilityState.STAGED: 1,
    DurabilityState.CANONICAL_VERIFIED: 2,
    DurabilityState.PROTECTED: 3,
}


class DurabilityError(RuntimeError):
    """Base class for durability contract failures."""


class DurabilityCorruptionError(DurabilityError):
    """A content-addressed object does not match its declared digest."""


class PublicationError(DurabilityError):
    """A snapshot could not reach the requested replica state."""


class ResumeArtifactError(DurabilityError):
    """An artifact is not eligible for exact training resume."""


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, block_size: int = DEFAULT_COPY_BLOCK_BYTES) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"sha256": _sha256_bytes(value), "size_bytes": len(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_bytes(path, _canonical_json(payload) + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DurabilityError(f"Expected a JSON object: {path}")
    return payload


def _truthy_environment(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _emit_evidence(
    kind: str,
    payload: Mapping[str, object],
    *,
    artifact_refs: list[dict[str, str]] | None = None,
) -> None:
    """Publish durability truth to the stream shared by Matrix and ThirdEye."""

    from runtime.evidence_stream import append_evidence

    append_evidence(
        source="training.checkpoint_durability",
        kind=kind,
        payload=dict(_json_safe(dict(payload))),
        run_id=os.environ.get("ANRA_CLUSTER_JOB_ID", ""),
        artifact_refs=artifact_refs,
        require_signature=_truthy_environment("ANRA_REQUIRE_SIGNED_EVIDENCE"),
    )


def _validate_identifier(value: str, *, label: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if not value or len(value) > 200 or any(character not in allowed for character in value):
        raise DurabilityError(f"Unsafe {label}: {value!r}")
    return value


def _validate_sha256(value: str, *, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise DurabilityCorruptionError(f"Invalid {label} SHA-256: {value!r}")
    return normalized


def durability_required_from_environment() -> bool:
    """Return whether this process must receive a remote durability ACK."""
    return any(
        _truthy_environment(name)
        for name in (
            "ANRA_REQUIRE_DURABLE_ACK",
            "ANRA_REQUIRE_SHARED_MASTER",
            "ANRA_CLUSTER_MODE",
        )
    ) or bool(os.environ.get("ANRA_CLUSTER_JOB_ID", "").strip())


def build_checkpoint_lineage(payload: Mapping[str, object]) -> dict[str, object]:
    """Expose the complete schema-9 continuity contract without tensor data."""
    model_config = _json_safe(payload.get("model_config", {}))
    tokenizer = _json_safe(payload.get("tokenizer_contract", {}))
    dataset_hashes = _json_safe(
        payload.get("dataset_manifest_hashes", payload.get("data_manifests", {}))
    )
    training_recipe = _json_safe(payload.get("training_recipe", {}))
    architecture_sha256 = _sha256_bytes(_canonical_json(model_config))
    dataset_contract_sha256 = _sha256_bytes(_canonical_json(dataset_hashes))
    recipe_sha256 = _sha256_bytes(_canonical_json(training_recipe))
    recipe = dict(training_recipe) if isinstance(training_recipe, Mapping) else {}
    seed_contract = payload.get("seed_contract", {})
    seed_payload = dict(seed_contract) if isinstance(seed_contract, Mapping) else {}
    explicit_lineage = str(
        payload.get("lineage_id")
        or os.environ.get("ANRA_CHECKPOINT_LINEAGE_ID", "")
        or os.environ.get("ANRA_CLUSTER_CAMPAIGN_ID", "")
    ).strip()
    model_profile = str(recipe.get("model_profile", "unknown"))
    lineage_id = (
        f"{explicit_lineage}/{model_profile}"
        if explicit_lineage
        else f"local/{architecture_sha256[:16]}/seed-{seed_payload.get('seed', 'unknown')}"
    )
    return {
        "lineage_id": lineage_id,
        "checkpoint_schema_version": int(
            payload.get("checkpoint_schema_version", 0) or 0
        ),
        "source_commit": str(payload.get("source_commit", "unknown")),
        "architecture": {
            "sha256": architecture_sha256,
            "config": model_config,
        },
        "tokenizer": tokenizer,
        "data": {
            "profile": str(payload.get("data_profile", "unknown")),
            "layout": str(payload.get("training_data_layout", "unknown")),
            "manifest_hashes": dataset_hashes,
            "contract_sha256": dataset_contract_sha256,
        },
        "training": {
            "recipe": training_recipe,
            "recipe_sha256": recipe_sha256,
            "seed_contract": _json_safe(seed_contract),
            "migration_provenance": _json_safe(
                payload.get("migration_provenance")
            ),
        },
        "progress": {
            "global_step": int(
                payload.get("global_step", payload.get("step", 0)) or 0
            ),
            "tokens_seen": int(payload.get("tokens_seen", 0) or 0),
            "sessions_completed": int(payload.get("sessions_completed", 0) or 0),
            "continuation_token_counts": _json_safe(
                payload.get("continuation_token_counts", {})
            ),
            "raw_window_consumption": _json_safe(
                payload.get("raw_window_consumption", {})
            ),
            "token_window": _json_safe(payload.get("token_window", {})),
        },
        "continuity": {
            "completed_optimizer_boundary": (
                payload.get("completed_optimizer_boundary") is True
            ),
            "accum_micro_steps": int(payload.get("accum_micro_steps", 0) or 0),
            "data_sampler_state": _json_safe(payload.get("data_sampler_state", {})),
            "components": {
                name: name in payload
                for name in ("model", "optimizer", "scheduler", "scaler", "rng_states")
            },
        },
    }


def assert_resume_artifact_class(blob: object, checkpoint: Path) -> None:
    """Reject inference-only artifacts before any model tensors are applied."""
    if not isinstance(blob, Mapping):
        return
    artifact_class = str(blob.get("checkpoint_artifact_class", FULL_RESUME))
    if artifact_class != FULL_RESUME:
        raise ResumeArtifactError(
            f"Checkpoint {checkpoint} is {artifact_class!r}; exact training resume "
            f"requires {FULL_RESUME!r}."
        )


def create_fp16_inference_artifact(
    checkpoint: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create a hash-bound model-only artifact that can never resume training."""

    source = Path(checkpoint)
    output = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Full-resume checkpoint does not exist: {source}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Compact artifact already exists: {output}")
    blob = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(blob, Mapping):
        raise ResumeArtifactError("Compact export requires a structured full-resume checkpoint")
    assert_resume_artifact_class(blob, source)
    lineage = blob.get("checkpoint_lineage")
    if not isinstance(lineage, Mapping):
        lineage = build_checkpoint_lineage(blob)
    _validate_resume_lineage(lineage)
    raw_model = blob.get("model_state_dict", blob.get("model"))
    if not isinstance(raw_model, Mapping) or not raw_model:
        raise ResumeArtifactError("Full-resume checkpoint has no model tensors")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in raw_model.items()
    ):
        raise ResumeArtifactError("Full-resume checkpoint model state contains non-tensors")
    compact_model = {
        name: (
            value.detach().to(device="cpu", dtype=torch.float16)
            if value.is_floating_point()
            else value.detach().to(device="cpu")
        )
        for name, value in raw_model.items()
    }
    payload: dict[str, object] = {
        "checkpoint_schema_version": int(blob.get("checkpoint_schema_version", 0)),
        "checkpoint_artifact_class": FP16_INFERENCE,
        "training_resume_allowed": False,
        "model": compact_model,
        "model_config": _json_safe(blob.get("model_config", {})),
        "tokenizer_contract": _json_safe(blob.get("tokenizer_contract", {})),
        "checkpoint_lineage": dict(_json_safe(lineage)),
        "source_full_resume_sha256": sha256_file(source),
        "source_commit": str(blob.get("source_commit", "unknown")),
        "global_step": int(blob.get("global_step", blob.get("step", 0)) or 0),
        "tokens_seen": int(blob.get("tokens_seen", 0) or 0),
        "growth_provenance": _json_safe(blob.get("growth_provenance", {})),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
        "source_full_resume_sha256": payload["source_full_resume_sha256"],
        "global_step": payload["global_step"],
        "lineage": payload["checkpoint_lineage"],
    }


def _validate_resume_lineage(lineage: Mapping[str, object]) -> None:
    if int(lineage.get("checkpoint_schema_version", 0) or 0) != 9:
        raise DurabilityError("A full_resume snapshot requires checkpoint schema 9 lineage")
    architecture = dict(lineage.get("architecture", {}))
    tokenizer = dict(lineage.get("tokenizer", {}))
    data = dict(lineage.get("data", {}))
    training = dict(lineage.get("training", {}))
    continuity = dict(lineage.get("continuity", {}))
    components = dict(continuity.get("components", {}))
    missing: list[str] = []
    if not str(lineage.get("lineage_id", "")).strip():
        missing.append("lineage_id")
    if not architecture.get("sha256"):
        missing.append("architecture.sha256")
    if not (tokenizer.get("sha256") or tokenizer.get("vocabulary_sha256")):
        missing.append("tokenizer.sha256")
    if not data.get("contract_sha256"):
        missing.append("data.contract_sha256")
    if not training.get("recipe_sha256"):
        missing.append("training.recipe_sha256")
    if continuity.get("completed_optimizer_boundary") is not True:
        missing.append("continuity.completed_optimizer_boundary")
    for component in ("model", "optimizer", "scheduler", "scaler", "rng_states"):
        if components.get(component) is not True:
            missing.append(f"continuity.components.{component}")
    if missing:
        raise DurabilityError(
            "A full_resume snapshot has incomplete schema-9 lineage: " + ", ".join(missing)
        )


@dataclass(frozen=True)
class ChunkRecord:
    index: int
    offset: int
    size_bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ChunkRecord:
        record = cls(
            index=int(payload["index"]),
            offset=int(payload["offset"]),
            size_bytes=int(payload["size_bytes"]),
            sha256=_validate_sha256(str(payloa×Ž¶ÖÚ$z{-®éÜj×6µ÷F–ÖV÷WE÷6V6öæG0¢6VÆbæ†÷E÷7F÷&vUöÆ–Ö—Eö'—FW2Ò–çB††÷E÷7F÷&vUöÆ–Ö—Eö'—FW2¢–b6VÆbæ†÷E÷7F÷&vUöÆ–Ö—Eö'—FW2ÃÒ ¢&—6RfÇVTW'&÷"‚&†÷E÷7F÷&vUöÆ–Ö—Eö'—FW2×W7B&R÷6—F—fR"¢6VÆbæ–æ—F–Åö6¶æ÷vÆVFvVBÒfÇ6P¢6VÆbæÆFW7C¢6æ6†÷E&VbÂæöæRÒæöæP ¢6Æ76ÖWF†ö@¢FVbg&öÕöVçf—&öæÖVçB€¢6Ç2À¢FVfVÇEö÷WF&÷ƒ¢F‚À¢¢À¢67&F6…÷'Vã¢&ööÂÀ¢’Óâ6†V6·ö–çDGW&&–Æ—G•6W76–öã ¢&WV—&VBÒGW&&–Æ—G•÷&WV—&VEög&öÕöVçf—&öæÖVçB‚¢Væ&ÆVBÒ&WV—&VB÷"÷G'WF‡•öVçf—&öæÖVçB‚$å$ôTä$ÄUôEU$$”Ä•E•ôõUD$õ‚"¢–b÷G'WF‡•öVçf—&öæÖVçB‚$å$õ$UT•$Uõ4”täTEôUd”DTä4R"’æBæ÷B÷2æVçf—&öâævWB€¢$å$ôUd”DTä4Uõ4”tä”äuô´U’"Â" ¢“ ¢&—6RV&Æ–6F–öäW'&÷"€¢$å$õ$UT•$Uõ4”täTEôUd”DTä4SÓ&WV—&W2å$ôUd”DTä4Uõ4”tä”äuô´U’ ¢¢&ö÷BÒF‚†÷2æVçf—&öâævWB‚$å$ôEU$$”Ä•E•ôõUD$õ‚"Â7G"†FVfVÇEö÷WF&÷‚’’¢÷WF&÷‚Ò6†V6·ö–çD÷WF&÷‚‡&ö÷B¢&WÆ–62Ò÷'6U÷&WÆ–6öVçf—&öæÖVçB‚’–bVæ&ÆVBVÇ6RµÐ¢–b&WV—&VBæBæ÷B&WÆ–63 ¢&—6RV&Æ–6F–öäW'&÷"€¢$GW&&ÆR&VÖ÷FR4²—2&WV—&VBÂ'WBå$ôEU$$”Ä•E•õ$UÄ”42†÷" ¢$å$õ4„$TEô4„T4µô”åEôD•"’FöW2æ÷BFVf–æR&WÆ–6â ¢¢Ö–å÷&÷FV7FVBÒ–çB€¢÷2æVçf—&öâævWB€¢$å$ôEU$$”Ä•E•ôÔ”åõ$õDT5DTEõ$UÄ”42"À¢7G"†Ö–âƒ"ÂÆVâ‡&WÆ–62’’’–b&WÆ–62VÇ6R#"À¢¢¢V&Æ—6†W"Ò€¢6æ6†÷EV&Æ—6†W"€¢÷WF&÷‚À¢&WÆ–62À¢Ö–å÷&÷FV7FVE÷&WÆ–63ÖÖ–å÷&÷FV7FVBÀ¢Ö…ö6÷•÷7G&V×3Ö–çB€¢÷2æVçf—&öâævWB€¢$å$ôEU$$”Ä•E•ô4õ•õ5E$TÕ2"À¢7G"„DTdTÅEôÔ…ô4õ•õ5E$TÕ2’À¢¢’À¢†÷E÷7F÷&vUöÆ–Ö—Eö'—FW3Ö–çB€¢÷2æVçf—&öâævWB€¢$å$ôEU$$”Ä•E•ô„õEôÄ”Ô•Eô%•DU2"À¢7G"„„õEõ5Dõ$tUôÄ”Ô•Eô%•DU2’À¢¢’À¢¢–b&WÆ–60¢VÇ6RæöæP¢¢&WGW&â6Ç2€¢÷WF&÷‚À¢V&Æ—6†W"À¢Væ&ÆVCÖVæ&ÆVBÀ¢&WV—&VC×&WV—&VBÀ¢67&F6…÷'Vã×67&F6…÷'VâÀ¢6µ÷F–ÖV÷WE÷6V6öæG3ÖfÆöB€¢÷2æVçf—&öâævWB‚$å$ôEU$$”Ä•E•ô4µõD”ÔTõUEõ4T4ôäE2"Â#ƒ"¢’À¢†÷E÷7F÷&vUöÆ–Ö—Eö'—FW3Ö–çB€¢÷2æVçf—&öâævWB€¢$å$ôEU$$”Ä•E•ô„õEôÄ”Ô•Eô%•DU2"À¢7G"„„õEõ5Dõ$tUôÄ”Ô•Eô%•DU2’À¢¢’À¢ ¢&÷W'G¢FVb&WV—&W5ö–æ—F–Åö&÷VæF'’‡6VÆb’Óâ&ööÃ ¢&WGW&â6VÆbç&WV—&VBæB6VÆbç67&F6…÷'VâæBæ÷B6VÆbæ–æ—F–Åö6¶æ÷vÆVFvV@ ¢FVbV&Æ—6…ö6†V6·ö–çB€¢6VÆbÀ¢6†V6·ö–çC¢F‚À¢–ÆöC¢Ö–æu·7G"Âö&¦V7EÒÀ¢¢À¢f–æÃ¢&ööÂÒfÇ6RÀ¢’Óâ6æ6†÷E&VbÂæöæS ¢–bæ÷B6VÆbæVæ&ÆVC ¢&WGW&âæöæP¢Æ–æVvRÒ–ÆöBævWB‚&6†V6·ö–çEöÆ–æVvR"¢–bæ÷B—6–ç7Fæ6R†Æ–æVvRÂÖ–ær“ ¢Æ–æVvRÒ'V–ÆEö6†V6·ö–çEöÆ–æVvR‡–ÆöB¢Æ–æVvUö–BÒ7G"†Æ–æVvRævWB‚&Æ–æVvUö–B"Â""’’ç7G&—‚¢–bæ÷BÆ–æVvUö–C ¢&—6RGW&&–Æ—G”W'&÷"‚$GW&&ÆR6†V6·ö–çB&WV—&W27F&ÆRÆ–æVvUö–B"¢2W†7FÇ’öæRWÆöBÖ’&R–âfÆ–v‡Bâ–bG&—fR—26Æ÷vW"F†âF†P¢2G&–æW"ÂF†RæW‡B6†V6·ö–çB&÷VæF'’Æ–W2&6·&W77W&R&F†W ¢2F†âf–ÆÆ–ærW†VÖW&ÂF—6²v—F‚âVæ&÷VæFVBVWVRà¢–b6VÆbçV&Æ—6†W"—2æ÷BæöæRæB6VÆbæÆFW7B—2æ÷BæöæS ¢6VÆbçV&Æ—6†W"çv—Eöf÷"€¢6VÆbæÆFW7BÀ¢GW&&–Æ—G•7FFRå$õDT5DTBÀ¢F–ÖV÷WE÷6V6öæG3×6VÆbæ6µ÷F–ÖV÷WE÷6V6öæG2À¢¢ÆâÒÆåö†÷E÷&WFVçF–öâ€¢6VÆbæ÷WF&÷‚À¢–åöfÆ–v‡Eö'—FW3ÕF‚†6†V6·ö–çB’ç7FB‚’ç7E÷6—¦RÀ¢†÷EöÆ–Ö—Eö'—FW3×6VÆbæ†÷E÷7F÷&vUöÆ–Ö—Eö'—FW2À¢Æ–æVvUö–CÖÆ–æVvUö–BÀ¢¢–bæ÷BÆâæf—G3 ¢&—6RV&Æ–6F–öäW'&÷"€¢$6†V6·ö–çB†÷B×7F÷&vR6öçG&7B6ææ÷Bf—BGvò&W7VÖR7FFW2Â ¢'Gvò6ö×7B7FFW2æBöæR–âÖfÆ–v‡B'F–f7C¢ ¢b&FVf–6—C×·ÆâæFVf–6—Eö'—FW7Ò'—FW2 ¢¢–bÆâæFVÆWFU÷6æ6†÷Eö–G3 ¢–b6VÆbçV&Æ—6†W"—2æ÷BæöæS ¢6VÆbçV&Æ—6†W"ç'VæU÷6æ6†÷G2‡ÆâæFVÆWFU÷6æ6†÷Eö–G2¢6VÆbæ÷WF&÷‚ç'VæR‡ÆâæFVÆWFU÷6æ6†÷Eö–G2¢&VbÒ6VÆbæ÷WF&÷‚ç&Vv—7FW%ö6†V6·ö–çB€¢6†V6·ö–çBÀ¢'F–f7Eö6Æ73Ô'F–f7D6Æ72äeTÄÅõ$U5TÔRÀ¢Æ–æVvSÖÆ–æVvRÀ¢¢6VÆbæÆFW7BÒ&V`¢–b6VÆbçV&Æ—6†W"—2æ÷BæöæS ¢6VÆbçV&Æ—6†W"ç7V&Ö—B‡&Vb¢v—Eöf÷%ö6²Ò6VÆbç&WV—&VBæB‡6VÆbç&WV—&W5ö–æ—F–Åö&÷VæF'’÷"f–æÂ¢–bv—Eöf÷%ö6³ ¢–b6VÆbçV&Æ—6†W"—2æöæS ¢&—6RV&Æ–6F–öäW'&÷"‚$&WV—&VBGW&&–Æ—G’4²†2æòV&Æ—6†W""¢F&vWBÒ€¢GW&&–Æ—G•7FFRå$õDT5DT@¢–bf–æÀ¢VÇ6RGW&&–Æ—G•7FFRä4äôä”4ÅõdU$”d”T@¢¢6VÆbçV&Æ—6†W"çv—Eöf÷"€¢&VbÀ¢F&vWBÀ¢F–ÖV÷WE÷6V6öæG3×6VÆbæ6µ÷F–ÖV÷WE÷6V6öæG2À¢¢6VÆbæ–æ—F–Åö6¶æ÷vÆVFvVBÒG'VP¢–bf–æÂæB6VÆbçV&Æ—6†W"—2æ÷BæöæS ¢6VÆbçV&Æ—6†W"çv—Eöf÷"€¢&VbÀ¢GW&&–Æ—G•7FFRå$õDT5DTBÀ¢F–ÖV÷WE÷6V6öæG3×6VÆbæ6µ÷F–ÖV÷WE÷6V6öæG2À¢¢f–æÅ÷ÆâÒÆåö†÷E÷&WFVçF–öâ€¢6VÆbæ÷WF&÷‚À¢†÷EöÆ–Ö—Eö'—FW3×6VÆbæ†÷E÷7F÷&vUöÆ–Ö—Eö'—FW2À¢Æ–æVvUö–CÖÆ–æVvUö–BÀ¢¢–bf–æÅ÷ÆâæFVÆWFU÷6æ6†÷Eö–G3 ¢6VÆbçV&Æ—6†W"ç'VæU÷6æ6†÷G2†f–æÅ÷ÆâæFVÆWFU÷6æ6†÷Eö–G2¢6VÆbæ÷WF&÷‚ç'VæR†f–æÅ÷ÆâæFVÆWFU÷6æ6†÷Eö–G2¢&WGW&â&V` ¢FVb6Æ÷6R‡6VÆb’ÓâæöæS ¢–b6VÆbçV&Æ—6†W"—2æöæS ¢&WGW&à¢–b6VÆbç&WV—&VBæB6VÆbæÆFW7B—2æ÷BæöæS ¢6VÆbçV&Æ—6†W"çv—Eöf÷"€¢6VÆbæÆFW7BÀ¢GW&&–Æ—G•7FFRå$õDT5DTBÀ¢F–ÖV÷WE÷6V6öæG3×6VÆbæ6µ÷F–ÖV÷WE÷6V6öæG2À¢¢6VÆbçV&Æ—6†W"æ6Æ÷6R‡v—CÕG'VRÂF–ÖV÷WE÷6V6öæG3×6VÆbæ6µ÷F–ÖV÷WE÷6V6öæG2  ¤FF6Æ72†g&÷¦VãÕG'VR¦6Æ72&WFVçF–öåÆã ¢¶VW÷6æ6†÷Eö–G3¢GWÆU·7G"ÂââåÐ¢FVÆWFU÷6æ6†÷Eö–G3¢GWÆU·7G"ÂââåÐ¢&WF–æVEöÆöv–6Åö'—FW3¢–ç@¢–åöfÆ–v‡Eö'—FW3¢–ç@¢†÷EöÆ–Ö—Eö'—FW3¢–ç@¢f—G3¢&ööÀ¢FVf–6—Eö'—FW3¢–ç@ ¢FVbFõöF–7B‡6VÆb’ÓâF–7E·7G"Âö&¦V7EÓ ¢&WGW&â°¢&¶VW÷6æ6†÷Eö–G2#¢Æ—7B‡6VÆbæ¶VW÷6æ6†÷Eö–G2’À¢&FVÆWFU÷6æ6†÷Eö–G2#¢Æ—7B‡6VÆbæFVÆWFU÷6æ6†÷Eö–G2’À¢'&WF–æVEöÆöv–6Åö'—FW2#¢6VÆbç&WF–æVEöÆöv–6Åö'—FW2À¢&–åöfÆ–v‡Eö'—FW2#¢6VÆbæ–åöfÆ–v‡Eö'—FW2À¢&†÷EöÆ–Ö—Eö'—FW2#¢6VÆbæ†÷EöÆ–Ö—Eö'—FW2À¢&f—G2#¢6VÆbæf—G2À¢&FVf–6—Eö'—FW2#¢6VÆbæFVf–6—Eö'—FW2À¢Ð  ¦FVb6æ6†÷EöÆ–æVvUö–B†÷WF&÷ƒ¢6†V6·ö–çD÷WF&÷‚Â&Vc¢6æ6†÷E&Vb’Óâ7G# ¢Æ–æVvRÒF–7B†÷WF&÷‚æÆöEöÖæ–fW7B‡&Vbç6æ6†÷Eö–B’ævWB‚&Æ–æVvR"Â·Ò’¢W‡Æ–6—BÒ7G"†Æ–æVvRævWB‚&Æ–æVvUö–B"Â""’’ç7G&—‚¢–bW‡Æ–6—C ¢&WGW&âW‡Æ–6—@¢266†VÖ×c÷WF&÷†W27&VFVB&Vf÷&RÆ–æVvUö–Bv2ÖæFF÷'’&VÖ–à¢2w&÷W&ÆRf÷"âW‡Æ–6—BÖ–w&F–öâ÷&WFVçF–öâ72à¢&6†—FV7GW&RÒF–7B†Æ–æVvRævWB‚&&6†—FV7GW&R"Â·Ò’¢G&–æ–ærÒF–7B†Æ–æVvRævWB‚'G&–æ–ær"Â·Ò’¢6VVBÒF–7B‡G&–æ–ærævWB‚'6VVEö6öçG&7B"Â·Ò’’ævWB‚'6VVB"Â'Væ¶æ÷vâ"¢&WGW&âb&ÆVv7’÷¶&6†—FV7GW&RævWB‚w6†#SbrÂwVæ¶æ÷vâr—Ò÷6VVB×·6VVGÒ   ¦FVbÆåö†÷E÷&WFVçF–öâ€¢÷WF&÷ƒ¢6†V6·ö–çD÷WF&÷‚À¢¢À¢¶VWögVÆÃ¢–çBÒDTdTÅEô´TUôeTÄÂÀ¢¶VWö6ö×7C¢–çBÒDTdTÅEô´TUô4ôÕ5BÀ¢–åöfÆ–v‡Eö'—FW3¢–çBÒÀ¢†÷EöÆ–Ö—Eö'—FW3¢–çBÒ„õEõ5Dõ$tUôÄ”Ô•Eô%•DU2À¢Æ–æVvUö–C¢7G"ÂæöæRÒæöæRÀ¢’Óâ&WFVçF–öåÆã ¢ÆÅ÷&Vg2Ò÷WF&÷‚ç6æ6†÷G2‚¢&Vg2Ò€¢·&Vbf÷"&Vb–âÆÅ÷&Vg2–b6æ6†÷EöÆ–æVvUö–B†÷WF&÷‚Â&Vb’ÓÒÆ–æVvUö–EÐ¢–bÆ–æVvUö–B—2æ÷BæöæP¢VÇ6RÆÅ÷&Vg0¢¢'•ö6Æ73¢F–7E´'F–f7D6Æ72ÂÆ—7Eµ6æ6†÷E&VeÕÒÒ°¢'F–f7D6Æ72äeTÄÅõ$U5TÔS¢µÒÀ¢'F–f7D6Æ72äeeô”ädU$Tä4S¢µÒÀ¢Ð¢f÷"&Vb–â&Vg3 ¢'•ö6Æ75·&Vbæ'F–f7Eö6Æ75ÒæVæB‡&Vb¢f÷"w&÷W–â'•ö6Æ72çfÇVW2‚“ ¢w&÷Wç6÷'B†¶W“ÖÆÖ&F&Vc¢‡&VbævÆö&Å÷7FWÂ&Vbç6æ6†÷Eö–B’Â&WfW'6SÕG'VR¢¶WBÒ€¢'•ö6Æ75´'F–f7D6Æ72äeTÄÅõ$U5TÔUÕ³¢Ö‚ƒÂ¶VWögVÆÂ•Ð¢²'•ö6Æ75´'F–f7D6Æ72äeeô”ädU$Tä4UÕ³¢Ö‚ƒÂ¶VWö6ö×7B•Ð¢¢¶VWö–G2Ò·&Vbç6æ6†÷Eö–Bf÷"&Vb–â¶WGÐ¢&WF–æVBÒ ¢f÷"&Vb–â¶WC ¢Öæ–fW7BÒ÷WF&÷‚æÆöEöÖæ–fW7B‡&Vbç6æ6†÷Eö–B¢&WF–æVB³Ò–çB†F–7B†Öæ–fW7E²'6÷W&6R%Ò•²'6—¦Uö'—FW2%Ò¢&WV—&VBÒ&WF–æVB²Ö‚ƒÂ–çB†–åöfÆ–v‡Eö'—FW2’¢&WGW&â&WFVçF–öåÆâ€¢¶VW÷6æ6†÷Eö–G3×GWÆR‡6÷'FVB†¶VWö–G2’’À¢FVÆWFU÷6æ6†÷Eö–G3×GWÆR€¢6÷'FVB‡&Vbç6æ6†÷Eö–Bf÷"&Vb–â&Vg2–b&Vbç6æ6†÷Eö–Bæ÷B–â¶VWö–G2¢’À¢&WF–æVEöÆöv–6Åö'—FW3×&WF–æVBÀ¢–åöfÆ–v‡Eö'—FW3ÖÖ‚ƒÂ–çB†–åöfÆ–v‡Eö'—FW2’’À¢†÷EöÆ–Ö—Eö'—FW3Ö–çB††÷EöÆ–Ö—Eö'—FW2’À¢f—G3×&WV—&VBÃÒ†÷EöÆ–Ö—Eö'—FW2À¢FVf–6—Eö'—FW3ÖÖ‚ƒÂ&WV—&VBÒ†÷EöÆ–Ö—Eö'—FW2’À¢  ¦FVb÷&WÆ–65ög&öÕö6Æ’€¢fÇVW3¢—FW&&ÆU·7G%ÒÀ¢G&—fU÷fÇVW3¢—FW&&ÆU·7G%ÒÀ¢’ÓâÆ—7E´f–ÆW7—7FVÕ&WÆ–6Ó ¢&WÆ–63¢Æ—7E´f–ÆW7—7FVÕ&WÆ–6ÒÒµÐ¢VçG&–W2Ò²‡fÇVRÂ&f–ÆW7—7FVÒ"’f÷"fÇVR–âfÇVW5Ð¢VçG&–W2³Ò²‡fÇVRÂ&Ö÷VçFVEöG&—fR"’f÷"fÇVR–âG&—fU÷fÇVW5Ð¢f÷"–æFW‚Â†VçG'’Â¶–æB’–âVçVÖW&FR†VçG&–W2“ ¢–b#Ò"æ÷B–âVçG'“ ¢&—6RfÇVTW'&÷"‚%&WÆ–6×W7B&RäÔSÕD‚"¢æÖRÂF‚ÒVçG'’ç7Æ—B‚#Ò"Â¢&WÆ–62æVæB€¢f–ÆW7—7FVÕ&WÆ–6€¢æÖRÀ¢F‚‡F‚’À¢¶–æCÖ¶–æBÀ¢6æöæ–6ÃÖ–æFW‚ÓÒÀ¢¢¢&WGW&â&WÆ–60  ¦FVbÖ–â†&wc¢6WVVæ6U·7G%ÒÂæöæRÒæöæR’Óâ–çC ¢'6W"Ò&w'6Rä&wVÖVçE'6W"†FW67&—F–öãÕõöFö5õò¢7V''6W'2Ò'6W"æFE÷7V''6W'2†FW7CÒ&6öÖÖæB"Â&WV—&VCÕG'VR ¢&Vv—7FW"Ò7V''6W'2æFE÷'6W"‚'&Vv—7FW""Â†VÇÒ'&Vv—7FW"â–Ö×WF&ÆR6†V6·ö–çB"¢&Vv—7FW"æFEö&wVÖVçB‚"ÒÖ÷WF&÷‚"Â&WV—&VCÕG'VRÂG—SÕF‚¢&Vv—7FW"æFEö&wVÖVçB‚"ÒÖ6†V6·ö–çB"Â&WV—&VCÕG'VRÂG—SÕF‚¢&Vv—7FW"æFEö&wVÖVçB€¢"ÒÖ'F–f7BÖ6Æ72"À¢6†ö–6W3Õ¶—FVÒçfÇVRf÷"—FVÒ–â'F–f7D6Æ75ÒÀ¢FVfVÇCÔ'F–f7D6Æ72äeTÄÅõ$U5TÔRçfÇVRÀ¢¢&Vv—7FW"æFEö&wVÖVçB‚"ÒÖÆ–æVvRÖ§6öâ"ÂG—SÕF‚ ¢V&Æ—6‚Ò7V''6W'2æFE÷'6W"‚'V&Æ—6‚"Â†VÇÒ'V&Æ—6‚öæR÷"ÆÂ6æ6†÷G2"¢V&Æ—6‚æFEö&wVÖVçB‚"ÒÖ÷WF&÷‚"Â&WV—&VCÕG'VRÂG—SÕF‚¢V&Æ—6‚æFEö&wVÖVçB‚"Ò×6æ6†÷BÖ–B"¢V&Æ—6‚æFEö&wVÖVçB‚"Ò×&WÆ–6"Â7F–öãÒ&VæB"ÂFVfVÇCÕµÒÂÖWFf#Ò$äÔSÕD‚"¢V&Æ—6‚æFEö&wVÖVçB€¢"ÒÖG&—fR×&WÆ–6"À¢7F–öãÒ&VæB"À¢FVfVÇCÕµÒÀ¢ÖWFf#Ò$äÔSÔÔõTåDTEõD‚"À¢¢V&Æ—6‚æFEö&wVÖVçB‚"ÒÖÖ–â×&÷FV7FVB×&WÆ–62"ÂG—SÖ–çBÂFVfVÇCÓ ¢ÖFW&–Æ—¦RÒ7V''6W'2æFE÷'6W"‚&ÖFW&–Æ—¦R"Â†VÇÒ'&V76VÖ&ÆR6æ6†÷B"¢ÖFW&–Æ—¦RæFEö&wVÖVçB‚"ÒÖ÷WF&÷‚"Â&WV—&VCÕG'VRÂG—SÕF‚¢ÖFW&–Æ—¦RæFEö&wVÖVçB‚"Ò×6æ6†÷BÖ–B"Â&WV—&VCÕG'VR¢ÖFW&–Æ—¦RæFEö&wVÖVçB‚"ÒÖ÷WGWB"Â&WV—&VCÕG'VRÂG—SÕF‚¢ÖFW&–Æ—¦RæFEö&wVÖVçB‚"ÒÖf÷"×&W7VÖR"Â7F–öãÒ'7F÷&U÷G'VR" ¢6ö×7BÒ7V''6W'2æFE÷'6W"€¢&6ö×7B"À¢†VÇÒ&7&VFRæB&Vv—7FW"âgbÖöFVÂÖöæÇ’'F–f7Bg&öÒgVÆÂ&W7VÖR"À¢¢6ö×7BæFEö&wVÖVçB‚"ÒÖ÷WF&÷‚"Â&WV—&VCÕG'VRÂG—SÕF‚¢6ö×7BæFEö&wVÖVçB‚"ÒÖ6†V6·ö–çB"Â&WV—&VCÕG'VRÂG—SÕF‚¢6ö×7BæFEö&wVÖVçB‚"ÒÖ÷WGWB"Â&WV—&VCÕG'VRÂG—SÕF‚¢6ö×7BæFEö&wVÖVçB‚"ÒÖ÷fW'w&—FR"Â7F–öãÒ'7F÷&U÷G'VR" ¢&WFVçF–öâÒ7V''6W'2æFE÷'6W"‚'&WFVçF–öâ"Â†VÇÒ'&–çBF†RæöâÖFW7G'V7F—fR†÷BÆâ"¢&WFVçF–öâæFEö&wVÖVçB‚"ÒÖ÷WF&÷‚"Â&WV—&VCÕG'VRÂG—SÕF‚¢&WFVçF–öâæFEö&wVÖVçB‚"ÒÖ–âÖfÆ–v‡BÖ'—FW2"ÂG—SÖ–çBÂFVfVÇCÓ ¢&w2Ò'6W"ç'6Uö&w2†&wb¢÷WF&÷‚Ò6†V6·ö–çD÷WF&÷‚†&w2æ÷WF&÷‚¢–b&w2æ6öÖÖæBÓÒ&6ö×7B# ¢&W÷'BÒ7&VFUögeö–æfW&Væ6Uö'F–f7B€¢&w2æ6†V6·ö–çBÀ¢&w2æ÷WGWBÀ¢÷fW'w&—FSÖ&w2æ÷fW'w&—FRÀ¢¢&VbÒ÷WF&÷‚ç&Vv—7FW%ö6†V6·ö–çB€¢&w2æ÷WGWBÀ¢'F–f7Eö6Æ73Ô'F–f7D6Æ72äeeô”ädU$Tä4RÀ¢Æ–æVvSÖF–7B‡&W÷'E²&Æ–æVvR%Ò’À¢¢&–çB€¢§6öâæGV×2€¢°¢¢§¶¶W“¢fÇVRf÷"¶W’ÂfÇVR–â&W÷'Bæ—FV×2‚’–b¶W’Ò&Æ–æVvR'ÒÀ¢'6æ6†÷Eö–B#¢&Vbç6æ6†÷Eö–BÀ¢ÒÀ¢–æFVçCÓ"À¢¢¢&WGW&â ¢–b&w2æ6öÖÖæBÓÒ'&Vv—7FW"# ¢–b&w2æÆ–æVvUö§6öã ¢Æ–æVvRÒ÷&VEö§6öâ†&w2æÆ–æVvUö§6öâ¢VÆ–b&w2æ'F–f7Eö6Æ72ÓÒ'F–f7D6Æ72äeTÄÅõ$U5TÔRçfÇVS ¢g&öÒ'VçF–ÖRç6fUöÆöB–×÷'B6fU÷F÷&6…öÆö@ ¢&Æö"Ò6fU÷F÷&6…öÆöB†&w2æ6†V6·ö–çBÂÖöÆö6F–öãÒ&7R"¢–bæ÷B—6–ç7Fæ6R†&Æö"ÂÖ–ær“ ¢&—6RGW&&–Æ—G”W'&÷"‚$gVÆÂ×&W7VÖR6†V6·ö–çB×W7B6öçF–âÖ–ær–ÆöB"¢Æ–æVvRÒ'V–ÆEö6†V6·ö–çEöÆ–æVvR†&Æö"¢VÇ6S ¢Æ–æVvRÒ·Ð¢&VbÒ÷WF&÷‚ç&Vv—7FW%ö6†V6·ö–çB€¢&w2æ6†V6·ö–çBÀ¢'F–f7Eö6Æ73Ö&w2æ'F–f7Eö6Æ72À¢Æ–æVvSÖÆ–æVvRÀ¢¢&–çB†§6öâæGV×2‡²'6æ6†÷Eö–B#¢&Vbç6æ6†÷Eö–GÒÂ–æFVçCÓ"’¢&WGW&â ¢–b&w2æ6öÖÖæBÓÒ&ÖFW&–Æ—¦R# ¢&W7VÇBÒ÷WF&÷‚æÖFW&–Æ—¦R€¢&w2ç6æ6†÷Eö–BÀ¢&w2æ÷WGWBÀ¢f÷%÷&W7VÖSÖ&w2æf÷%÷&W7VÖRÀ¢¢&–çB‡&W7VÇB¢&WGW&â ¢–b&w2æ6öÖÖæBÓÒ'&WFVçF–öâ# ¢&–çB€¢§6öâæGV×2€¢Æåö†÷E÷&WFVçF–öâ€¢÷WF&÷‚À¢–åöfÆ–v‡Eö'—FW3Ö&w2æ–åöfÆ–v‡Eö'—FW2À¢’çFõöF–7B‚’À¢–æFVçCÓ"À¢¢¢&WGW&â ¢&WÆ–62Ò÷&WÆ–65ög&öÕö6Æ’†&w2ç&WÆ–6Â&w2æG&—fU÷&WÆ–6¢V&Æ—6†W"Ò6æ6†÷EV&Æ—6†W"€¢÷WF&÷‚À¢&WÆ–62À¢Ö–å÷&÷FV7FVE÷&WÆ–63Ö&w2æÖ–å÷&÷FV7FVE÷&WÆ–62À¢¢&Vg2Ò¶÷WF&÷‚æÆöE÷&Vb†&w2ç6æ6†÷Eö–B•Ò–b&w2ç6æ6†÷Eö–BVÇ6R÷WF&÷‚ç6æ6†÷G2‚¢G'“ ¢f÷"&Vb–â&Vg3 ¢V&Æ—6†W"ç7V&Ö—B‡&Vb¢f÷"&Vb–â&Vg3 ¢&W7VÇBÒV&Æ—6†W"çv—Eöf÷"‡&VbÂGW&&–Æ—G•7FFRä4äôä”4ÅõdU$”d”TB¢&–çB†§6öâæGV×2‡²'6æ6†÷Eö–B#¢&Vbç6æ6†÷Eö–BÂ'7FFR#¢&W7VÇBç7FFRçfÇVWÒ’¢f–æÆÇ“ ¢V&Æ—6†W"æ6Æ÷6R‡v—CÕG'VR¢&WGW&â   ¦–bõöæÖUõòÓÒ%õöÖ–åõò# ¢&—6R7—7FVÔW†—B†Ö–â‚’ 