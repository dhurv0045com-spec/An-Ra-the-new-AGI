"""Canonical GPU SFT child-lineage trainer for the V4 181M model.

This is intentionally separate from foundation pretraining and the retired
identity fine-tuner.  It trains only assistant-answer tokens, starts a fresh
optimizer from a verified foundation parent, and publishes its checkpoint into
``<shared training home>/sft-v4`` so it cannot replace foundation weights.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import pickle
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 - canonical PyTorch alias
from anra.sft_conversation import (
    render_chat_prompt,
    render_prompt_from_context,
    split_training_conversation,
)
from evaluation.sft_behavior_gate import check_smoke_response
from torch.utils.data import Dataset

from training.anra_optimizer import build_optimizer
from training.checkpoint_durability import (
    CheckpointDurabilitySession,
    DurabilityState,
    build_checkpoint_lineage,
    sha256_file,
)
from training.mixed_precision import MixedPrecisionTrainer
from training.posttraining_contract import (
    REQUIRED_SFT_CATEGORIES,
    verify_sft_lineage_manifest,
    write_sft_lineage_manifest,
)
from training.reproducibility import capture_rng_states, make_data_generator, seed_everything
from training.scheduler import get_cosine_schedule_with_warmup
from training.sft_dataset_v4 import SFT_DATASET_SCHEMA, SFT_SOURCE_RECEIPTS_SCHEMA
from training.v2_config import (
    ANRA_V4_TRAINING,
    CANONICAL_FOUNDATION_OPTIMIZER,
    CANONICAL_MODEL_PROFILE,
    CANONICAL_TRAINING_SEED,
    CHECKPOINT_SCHEMA_VERSION,
)
from training.v2_runtime import (
    active_tokenizer_identity,
    active_tokenizer_path,
    atomic_save,
    build_model_for_profile,
    ensure_tied_lm_head,
    load_checkpoint,
    load_or_build_v2_tokenizer,
    write_json,
)

SFT_STAGE = "sft"
SFT_CHECKPOINT_SCHEMA = "anra-v4-sft-checkpoint/v1"
SFT_DEFAULT_LEARNING_RATE = 5e-5
SFT_DEFAULT_TOTAL_STEPS = 5_000
SFT_DEFAULT_CHECKPOINT_STEPS = 200
SFT_DEFAULT_CHECKPOINT_MINUTES = 15
SFT_FULL_APPROVAL_SCHEMA = "anra-v4-sft-full-approval/v1"


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _seal(payload: Mapping[str, object], signing_key: str) -> str:
    return hmac.new(
        signing_key.encode("utf-8"), _canonical_json(payload), hashlib.sha256
    ).hexdigest()


def _write_full_sft_approval(
    *,
    vault_root: Path,
    lineage: Mapping[str, object],
    signing_key: str,
    owner_approval: str,
) -> Path:
    """Record explicit owner approval after a real protected pilot."""

    statement = owner_approval.strip()
    if len(statement) < 12:
        raise ValueError("full SFT approval must contain an explicit owner statement")
    sft_root = vault_root / "sft-v4"
    report_path = sft_root / "latest_sft_report.json"
    checkpoint = sft_root / "anra-v4-current-full-resume.pt"
    if not report_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("a protected SFT pilot report and checkpoint are required")
    report = _read_json(report_path)
    expected_report = {
        "lineage_id": str(lineage["lineage_id"]),
        "base_checkpoint_sha256": str(dict(lineage["parent"])["base_checkpoint_sha256"]),
        "train_manifest_sha256": str(dict(lineage["dataset"])["manifest_sha256"]),
        "validation_manifest_sha256": str(dict(lineage["evaluation"])["manifest_sha256"]),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    for field, expected in expected_report.items():
        if report.get(field) != expected:
            raise RuntimeError(
                f"SFT pilot report {field} is not bound to current checkpoint lineage"
            )
    measured_losses = (
        float(report.get("parent_validation_loss", math.inf)),
        float(report.get("best_validation_loss", math.inf)),
    )
    if int(report.get("global_step", 0)) <= 0 or not all(
        math.isfinite(value) for value in measured_losses
    ):
        raise RuntimeError("SFT pilot has no valid parent-versus-child validation evidence")
    behavior = report.get("behavior_smoke")
    if not isinstance(behavior, Mapping) or behavior.get("passed") is not True:
        raise RuntimeError(
            "SFT pilot lacks a passing behavior smoke report; review generated outputs "
            "before approving full SFT"
        )
    readiness_path = sft_root / "ready_to_sft.json"
    if not readiness_path.is_file():
        raise FileNotFoundError(
            "The final pilot has not produced ready_to_sft.json; rerun the pilot first"
        )
    readiness = _read_json(readiness_path)
    expected_readiness = {
        "lineage_id": str(lineage["lineage_id"]),
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_manifest_sha256": str(dict(lineage["dataset"])["manifest_sha256"]),
        "validation_manifest_sha256": str(dict(lineage["evaluation"])["manifest_sha256"]),
    }
    for field, expected in expected_readiness.items():
        if readiness.get(field) != expected:
            raise RuntimeError(f"SFT readiness {field} is not bound to the current pilot")
    if readiness.get("full_sft_ready") is not True:
        raise RuntimeError("The final SFT readiness gate did not pass")
    body: dict[str, object] = {
        "schema": SFT_FULL_APPROVAL_SCHEMA,
        "lineage_manifest_sha256": str(lineage["manifest_sha256"]),
        "base_checkpoint_sha256": str(dict(lineage["parent"])["base_checkpoint_sha256"]),
        "pilot_checkpoint_sha256": sha256_file(checkpoint),
        "pilot_global_step": int(report["global_step"]),
        "pilot_validation_loss": float(report["best_validation_loss"]),
        "parent_validation_loss": float(report["parent_validation_loss"]),
        "validation_delta_from_parent": float(report["best_validation_loss"])
        - float(report["parent_validation_loss"]),
        "owner_approval": statement,
    }
    sealed = {**body, "signature": _seal(body, signing_key)}
    target = sft_root / "full_sft_approval.json"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _verify_full_sft_approval(
    *, vault_root: Path, lineage: Mapping[str, object], signing_key: str
) -> dict[str, object]:
    path = vault_root / "sft-v4" / "full_sft_approval.json"
    if not path.is_file():
        raise PermissionError(
            "Full SFT requires a signed full_sft_approval.json created after the pilot review"
        )
    payload = _read_json(path)
    signature = str(payload.pop("signature", ""))
    if not hmac.compare_digest(signature, _seal(payload, signing_key)):
        raise PermissionError("full SFT approval signature is invalid")
    if payload.get("schema") != SFT_FULL_APPROVAL_SCHEMA:
        raise PermissionError("full SFT approval schema is invalid")
    expected = {
        "lineage_manifest_sha256": str(lineage["manifest_sha256"]),
        "base_checkpoint_sha256": str(dict(lineage["parent"])["base_checkpoint_sha256"]),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise PermissionError(f"full SFT approval does not match active {name}")
    checkpoint = vault_root / "sft-v4" / "anra-v4-current-full-resume.pt"
    if not checkpoint.is_file():
        raise PermissionError(
            "full SFT approval is not bound to a protected pilot checkpoint"
        )

    current_checkpoint_sha256 = sha256_file(checkpoint)
    approved_checkpoint_sha256 = str(payload.get("pilot_checkpoint_sha256", ""))
    if current_checkpoint_sha256 != approved_checkpoint_sha256:
        # The first full run replaces the pilot with a newer child checkpoint.
        # Approval is intentionally bound to the immutable pilot hash, so a
        # later resume must prove that the current file is a descendant of the
        # same signed SFT lineage instead of requiring the hash to remain equal
        # to the pilot forever.
        try:
            _verify_resume_checkpoint_binding(checkpoint, lineage)
            current_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            pickle.UnpicklingError,
        ) as error:
            raise PermissionError(
                "full SFT approval is not bound to the current protected pilot "
                "checkpoint or a valid descendant checkpoint"
            ) from error
        if not isinstance(current_payload, Mapping):
            raise PermissionError(
                "current full SFT checkpoint is not a valid lineage-bound mapping"
            )
        try:
            current_step = int(
                current_payload.get(
                    "global_step", current_payload.get("step", -1)
                )
            )
            approved_step = int(payload.get("pilot_global_step", -1))
        except (TypeError, ValueError) as error:
            raise PermissionError(
                "full SFT approval has invalid pilot/checkpoint step metadata"
            ) from error
        if current_step < approved_step:
            raise PermissionError(
                "current SFT checkpoint predates the approved pilot checkpoint"
            )
    return payload


def _require_cuda(*, allow_cpu_pilot: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu_pilot:
        return torch.device("cpu")
    raise RuntimeError("V4 SFT requires CUDA. Select a T4 GPU before starting the notebook.")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_source_receipt_hash(path: Path, expected: str) -> dict[str, Any]:
    receipt = _read_json(path)
    if receipt.get("schema") != SFT_SOURCE_RECEIPTS_SCHEMA:
        raise ValueError("SFT source receipt schema is invalid")
    if len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError("SFT source receipt does not match the signed dataset lineage")
    return receipt


def _verify_source_receipt(path: Path, lineage: Mapping[str, object]) -> dict[str, Any]:
    """Require the source receipt whose digest the signed lineage recorded."""

    return _verify_source_receipt_hash(
        path,
        str(dict(lineage["dataset"]).get("source_receipt_sha256", "")),
    )


def _render_prompt(messages: Sequence[Mapping[str, object]]) -> tuple[str, str]:
    return split_training_conversation(messages)


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    answer: str
    category: str
    conversation_sha256: str
    split_group: str


def load_sft_split_examples(
    dataset_manifest_path: str | Path,
    *,
    expected_split: str,
    require_all_categories: bool,
) -> tuple[list[SFTExample], dict[str, Any]]:
    """Load one immutable SFT split and validate its own manifest and artifact."""

    manifest_path = Path(dataset_manifest_path).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != SFT_DATASET_SCHEMA or manifest.get("split") != expected_split:
        raise ValueError(
            f"SFT runner expected an anra-sft-dataset/v1 {expected_split} manifest"
        )
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != 1
        or not isinstance(artifacts[0], Mapping)
    ):
        raise ValueError(f"SFT {expected_split} manifest must declare exactly one JSONL artifact")
    artifact_info = artifacts[0]
    artifact = manifest_path.parent / str(artifact_info.get("path", ""))
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if int(artifact_info.get("size_bytes", -1)) != artifact.stat().st_size:
        raise ValueError(f"SFT {expected_split} artifact size mismatch")
    if str(artifact_info.get("sha256", "")) != sha256_file(artifact):
        raise ValueError(f"SFT {expected_split} artifact hash mismatch")
    examples: list[SFTExample] = []
    seen: set[str] = set()
    with artifact.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"SFT record {line_number} is not an object")
            category = str(raw.get("category", ""))
            if category not in REQUIRED_SFT_CATEGORIES:
                raise ValueError(f"SFT record {line_number} has invalid category {category!r}")
            messages = raw.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"SFT record {line_number} has no messages")
            prompt, answer = _render_prompt(
                [dict(item) for item in messages if isinstance(item, Mapping)]
            )
            identity = str(raw.get("conversation_sha256", ""))
            if len(identity) != 64 or identity in seen:
                raise ValueError(f"SFT record {line_number} has invalid or duplicate identity")
            split_group = str(raw.get("split_group", "")).strip()
            if not split_group:
                raise ValueError(f"SFT record {line_number} has no split_group")
            seen.add(identity)
            examples.append(SFTExample(prompt, answer, category, identity, split_group))
    declared_counts = manifest.get("category_counts", {})
    actual_counts = {
        category: sum(item.category == category for item in examples)
        for category in REQUIRED_SFT_CATEGORIES
    }
    if require_all_categories and any(
        actual_counts[category] <= 0 for category in REQUIRED_SFT_CATEGORIES
    ):
        raise ValueError("SFT train artifact is missing one or more required categories")
    observed_counts = {name: count for name, count in actual_counts.items() if count > 0}
    if dict(sorted(observed_counts.items())) != dict(sorted(dict(declared_counts).items())):
        raise ValueError("SFT category counts disagree with the signed data manifest")
    if int(manifest.get("accepted_examples", -1)) != len(examples):
        raise ValueError("SFT accepted example count disagrees with the train artifact")
    return examples, manifest


def load_sft_examples(dataset_manifest_path: str | Path) -> tuple[list[SFTExample], dict[str, Any]]:
    """Load the training split, which must cover every canonical SFT category."""

    return load_sft_split_examples(
        dataset_manifest_path,
        expected_split="train",
        require_all_categories=True,
    )


def load_sft_validation_examples(
    dataset_manifest_path: str | Path,
) -> tuple[list[SFTExample], dict[str, Any]]:
    """Load held-out SFT data; validation need not contain every category."""

    return load_sft_split_examples(
        dataset_manifest_path,
        expected_split="validation",
        require_all_categories=False,
    )


class SFTConversationDataset(Dataset[dict[str, torch.Tensor]]):
    """Causal SFT samples with loss exactly on assistant tokens and EOS."""

    def __init__(self, examples: Sequence[SFTExample], tokenizer: object, block_size: int) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.pad_id = int(tokenizer.pad_token_id)
        self.bos_id = int(tokenizer.bos_token_id)
        self.eos_id = int(tokenizer.eos_token_id)
        if not self.examples:
            raise ValueError("SFT dataset is empty")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        prefix = self.tokenizer.encode(
            render_prompt_from_context(example.prompt), add_special_tokens=False
        )
        answer = self.tokenizer.encode(f" {example.answer}", add_special_tokens=False)
        # Preserve the full answer/EOS whenever possible; trim only older prompt context.
        answer = answer[: max(1, self.block_size - 2)]
        prompt_budget = max(0, self.block_size - 1 - len(answer))
        prefix = prefix[-prompt_budget:] if prompt_budget else []
        full = [self.bos_id, *prefix, *answer, self.eos_id]
        inputs = full[:-1]
        targets = full[1:]
        answer_start = len(prefix)
        weights = [0.0] * len(targets)
        # targets[answer_start:] corresponds to answer tokens plus EOS.
        for offset in range(answer_start, len(targets)):
            weights[offset] = 1.0
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
        }

    def collate(self, rows: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        width = max(int(row["input_ids"].numel()) for row in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        targets = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        weights = torch.zeros((len(rows), width), dtype=torch.float32)
        for index, row in enumerate(rows):
            length = int(row["input_ids"].numel())
            ids[index, :length] = row["input_ids"]
            targets[index, :length] = row["targets"]
            weights[index, :length] = row["weights"]
        return {"input_ids": ids, "targets": targets, "weights": weights}


def assistant_only_loss(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Return the mean cross-entropy over assistant/EOS targets only."""

    per_token = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    denominator = weights.sum().clamp_min(1.0)
    return (per_token * weights).sum() / denominator


def _sft_recipe(
    *, seed: int, batch_size: int, accumulation: int, total_steps: int
) -> dict[str, object]:
    return {
        "stage": SFT_STAGE,
        "model_profile": CANONICAL_MODEL_PROFILE,
        "training_layout": "assistant_only_sft_v1",
        "optimizer": CANONICAL_FOUNDATION_OPTIMIZER,
        "seed": int(seed),
        "micro_batch_size": int(batch_size),
        "gradient_accumulation": int(accumulation),
        "total_steps": int(total_steps),
        "schedule": "cosine_with_warmup",
    }


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    mp: MixedPrecisionTrainer,
    global_step: int,
    epoch: int,
    microbatch_cursor: int,
    skipped_updates: int,
    input_tokens_processed: int,
    supervised_tokens_processed: int,
    best_validation_loss: float,
    parent_validation_loss: float,
    train_loss: float,
    tokenizer_contract: Mapping[str, object],
    recipe: Mapping[str, object],
    lineage: Mapping[str, object],
    dataset_manifest: Mapping[str, object],
    data_generator: torch.Generator,
    behavior_history: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_artifact_class": "full_resume",
        "sft_checkpoint_schema": SFT_CHECKPOINT_SCHEMA,
        "sft": {
            "stage": SFT_STAGE,
            "lineage_manifest_sha256": lineage["manifest_sha256"],
            "base_checkpoint_sha256": dict(lineage["parent"])["base_checkpoint_sha256"],
            "dataset_manifest_sha256": dict(lineage["dataset"])["manifest_sha256"],
            "validation_manifest_sha256": dict(lineage["evaluation"])["manifest_sha256"],
            "assistant_only_loss": True,
            "category_counts": dict(dataset_manifest["category_counts"]),
        },
        "lineage_id": str(lineage["lineage_id"]),
        "tokenizer_schema_version": int(tokenizer_contract.get("schema_version", 4)),
        "tokenizer_contract": dict(tokenizer_contract),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": mp.state_dict(),
        "global_step": int(global_step),
        "step": int(global_step),
        "epoch": int(epoch),
        "sft_microbatch_cursor": int(microbatch_cursor),
        "sft_skipped_updates": int(skipped_updates),
        "sft_input_tokens_processed": int(input_tokens_processed),
        "sft_supervised_tokens_processed": int(supervised_tokens_processed),
        # Loss is necessary but cannot establish useful language behavior. Keep
        # the bounded fixed-prompt history on every resumable child so an
        # apparent validation improvement cannot erase a collapse signal from
        # an earlier worker session.
        "sft_behavior_history": [dict(row) for row in behavior_history],
        "tokens_seen": int(input_tokens_processed),
        "best_loss": float(best_validation_loss),
        "best_training_loss": float(train_loss),
        "best_validation_loss": float(best_validation_loss),
        "parent_validation_loss": float(parent_validation_loss),
        "completed_optimizer_boundary": True,
        "accum_micro_steps": 0,
        "model_config": model.model_config(),
        "training_recipe": dict(recipe),
        "data_profile": "sft-v4",
        "training_data_layout": "assistant_only_sft_v1",
        "dataset_manifest_hashes": {
            "sft_train": str(dict(lineage["dataset"])["manifest_sha256"]),
            "sft_validation": str(dict(lineage["evaluation"])["manifest_sha256"]),
        },
        "data_manifests": {
            "sft_train": str(dict(lineage["dataset"])["manifest_sha256"]),
            "sft_validation": str(dict(lineage["evaluation"])["manifest_sha256"]),
        },
        "mix_report": {
            "total_examples": int(dataset_manifest["accepted_examples"]),
            "realized_counts": dict(dataset_manifest["category_counts"]),
            "active_weights": {"assistant_only": 1.0},
        },
        "rng_states": capture_rng_states(data_generator=data_generator),
        "source_commit": _source_commit(),
    }
    payload["checkpoint_lineage"] = build_checkpoint_lineage(payload)
    return payload


def _copy_verified(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".resume.tmp")
    try:
        shutil.copy2(source, temporary)
        if source.stat().st_size != temporary.stat().st_size or sha256_file(source) != sha256_file(
            temporary
        ):
            raise ValueError("SFT checkpoint copy failed hash verification")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_resume_checkpoint_binding(path: Path, lineage: Mapping[str, object]) -> None:
    """Reject a valid-looking SFT child that belongs to another signed lineage."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"cannot inspect SFT resume checkpoint: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("SFT resume checkpoint is not a mapping")
    sft = payload.get("sft")
    if not isinstance(sft, Mapping):
        raise ValueError("resume checkpoint has no SFT lineage metadata")
    expected = {
        "stage": SFT_STAGE,
        "lineage_manifest_sha256": str(lineage["manifest_sha256"]),
        "base_checkpoint_sha256": str(dict(lineage["parent"])["base_checkpoint_sha256"]),
        "dataset_manifest_sha256": str(dict(lineage["dataset"])["manifest_sha256"]),
        "validation_manifest_sha256": str(dict(lineage["evaluation"])["manifest_sha256"]),
        "assistant_only_loss": True,
    }
    if payload.get("sft_checkpoint_schema") != SFT_CHECKPOINT_SCHEMA:
        raise ValueError("resume checkpoint has an unsupported SFT checkpoint schema")
    for field, expected_value in expected.items():
        if sft.get(field) != expected_value:
            raise ValueError(f"resume checkpoint {field} does not match signed SFT lineage")


def _resume_parent_validation_loss(
    loaded: Mapping[str, object],
    *,
    resuming: bool,
    mode: str,
    signed_approval: object,
) -> tuple[float, bool]:
    """Recover a legacy child baseline only from the verified full approval.

    Early pilot workers wrote a valid lineage-bound child checkpoint without
    persisting ``parent_validation_loss`` in the payload.  Full-mode preflight
    already verifies that the signed approval is bound to the current pilot
    checkpoint and lineage, so its signed baseline is the only safe migration
    source.  Pilot mode still rejects the incomplete checkpoint.
    """

    if not resuming:
        return math.inf, False
    try:
        stored = float(loaded.get("parent_validation_loss", math.inf))
    except (TypeError, ValueError):
        stored = math.inf
    if math.isfinite(stored):
        return stored, False
    if mode != "full" or not isinstance(signed_approval, Mapping):
        return stored, False
    try:
        approved = float(signed_approval.get("parent_validation_loss", math.inf))
    except (TypeError, ValueError):
        approved = math.inf
    if math.isfinite(approved):
        return approved, True
    return stored, False


def _compatibility_commit_authorized(lineage_source: object, current_source: str) -> bool:
    """Allow only an explicitly declared, owner-approved runtime patch."""

    return (
        bool(current_source)
        and str(lineage_source) == os.environ.get("ANRA_SFT_COMPATIBILITY_BASE_COMMIT", "")
        and current_source == os.environ.get("ANRA_SFT_COMPATIBILITY_TARGET_COMMIT", "")
    )


def _configure_durability(vault_root: Path, outbox: Path, *, lineage_id: str) -> None:
    destination = vault_root / "sft-v4"
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["ANRA_DURABILITY_OUTBOX"] = str(outbox)
    os.environ["ANRA_DURABILITY_REPLICAS"] = json.dumps(
        [
            {
                "name": "drive-sft-vault",
                "path": str(destination),
                "kind": "mounted_drive_single_file",
                "canonical": True,
            }
        ]
    )
    os.environ["ANRA_REQUIRE_DURABLE_ACK"] = "1"
    os.environ["ANRA_DURABILITY_MIN_PROTECTED_REPLICAS"] = "1"
    os.environ["ANRA_DURABILITY_COPY_STREAMS"] = "2"
    os.environ["ANRA_CHECKPOINT_LINEAGE_ID"] = lineage_id
    os.environ["ANRA_DATA_PROFILE"] = "sft-v4"
    os.environ["ANRA_TRAINING_DATA_LAYOUT"] = "assistant_only_sft_v1"


def _prune_sft_checkpoint_copies(vault_root: Path) -> tuple[str, ...]:
    """Keep the SFT Drive vault to one portable full-resume checkpoint.

    Older notebook revisions moved the 2+ GiB checkpoint into an ``archive``
    directory whenever a lineage changed.  That preserved audit metadata but
    silently multiplied Drive usage on every restart.  Lineage JSON/reports
    remain useful evidence; old full-resume payloads do not belong in the hot
    training vault because they can never be resumed by the active lineage.

    The current root checkpoint is never touched.  Legacy step-named files in
    the root are retained until a current checkpoint exists so a first-run
    migration cannot destroy the only recoverable state.
    """

    sft_root = (vault_root / "sft-v4").resolve()
    if not sft_root.is_dir():
        return ()
    current = (sft_root / "anra-v4-current-full-resume.pt").resolve()
    removed: list[str] = []

    def is_checkpoint_copy(path: Path) -> bool:
        name = path.name
        return (
            name == "anra-v4-current-full-resume.pt"
            or name.startswith("anra-v4-current-full-resume")
            or (name.startswith("anra-v4-step-") and name.endswith("-full-resume.pt"))
        )

    for path in sft_root.rglob("*"):
        if not path.is_file() or not is_checkpoint_copy(path):
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved == current:
            continue
        # Never remove a root legacy checkpoint before a canonical replacement
        # exists; the notebook can still migrate it on a first run.
        if resolved.parent == sft_root and not current.is_file():
            continue
        try:
            resolved.unlink()
        except OSError:
            continue
        removed.append(str(resolved))
    return tuple(sorted(removed))


@dataclass(frozen=True)
class SFTRunConfig:
    dataset_manifest: Path
    validation_manifest: Path
    source_receipt: Path
    lineage_manifest: Path
    base_checkpoint: Path
    vault_root: Path
    local_checkpoint: Path
    signing_key: str
    mode: str = "pilot"
    max_minutes: int = 15
    max_examples: int | None = None
    batch_size: int = 1
    accumulation: int = 8
    seed: int = CANONICAL_TRAINING_SEED
    total_steps: int = SFT_DEFAULT_TOTAL_STEPS
    checkpoint_steps: int = SFT_DEFAULT_CHECKPOINT_STEPS
    checkpoint_minutes: int = SFT_DEFAULT_CHECKPOINT_MINUTES
    behavior_probe_steps: int = 500
    allow_cpu_pilot: bool = False


def preflight_sft_v4(config: SFTRunConfig) -> dict[str, object]:
    device = _require_cuda(allow_cpu_pilot=config.allow_cpu_pilot)
    tokenizer_path = active_tokenizer_path()
    lineage = verify_sft_lineage_manifest(
        config.lineage_manifest,
        signing_key=config.signing_key,
        artifact_paths={
            "base_checkpoint_path": config.base_checkpoint,
            "tokenizer_path": tokenizer_path,
            "dataset_manifest_path": config.dataset_manifest,
            "validation_manifest_path": config.validation_manifest,
        },
    )
    current_source = _source_commit()
    lineage_source = str(lineage.get("source_commit", ""))
    approval = None
    if config.mode == "full":
        approval = _verify_full_sft_approval(
            vault_root=config.vault_root,
            lineage=lineage,
            signing_key=config.signing_key,
        )
    if lineage_source != current_source:
        if not (
            approval is not None
            and _compatibility_commit_authorized(lineage_source, current_source)
        ):
            raise RuntimeError(
                "SFT lineage source commit differs from this checkout. Clone the exact "
                "commit that created the lineage instead of changing training code mid-run."
            )
        print(
            "[SFT] Using the explicitly approved compatibility runtime patch "
            f"{lineage_source[:8]} -> {current_source[:8]}; data and checkpoint "
            "lineage remain unchanged."
        )
    if not isinstance(lineage.get("evaluation"), Mapping):
        raise ValueError(
            "canonical V4 SFT requires a signed, hash-bound validation split in its lineage"
        )
    _verify_source_receipt(config.source_receipt, lineage)
    examples, dataset_manifest = load_sft_examples(config.dataset_manifest)
    validation_examples, validation_manifest = load_sft_validation_examples(
        config.validation_manifest
    )
    if not validation_examples:
        raise ValueError("SFT validation split is empty")
    validation_categories = {row.category for row in validation_examples}
    missing_validation = sorted(
        set(REQUIRED_SFT_CATEGORIES) - validation_categories
    )
    if missing_validation:
        raise ValueError(
            "SFT validation split is missing required categories: "
            + ", ".join(missing_validation)
        )
    if {row.split_group for row in examples} & {
        row.split_group for row in validation_examples
    }:
        raise ValueError("SFT training and validation split groups overlap")
    if {row.conversation_sha256 for row in examples} & {
        row.conversation_sha256 for row in validation_examples
    }:
        raise ValueError("SFT training and validation conversation identities overlap")
    if config.mode not in {"pilot", "full"}:
        raise ValueError("SFT mode must be 'pilot' or 'full'")
    if config.mode == "pilot" and len(examples) < 8:
        raise ValueError("SFT pilot needs at least one example in each required category")
    if config.mode == "full" and len(examples) < 1_000:
        raise ValueError("Full SFT requires at least 1,000 audited examples")
    if (
        config.batch_size < 1
        or config.accumulation < 1
        or config.max_minutes < 1
        or config.behavior_probe_steps < 1
    ):
        raise ValueError("SFT batch size, accumulation, and session minutes must be positive")
    if config.max_examples is not None:
        raise ValueError(
            "--max-examples is not allowed for canonical SFT: use an audited, "
            "category-complete pilot corpus instead"
        )
    destination = config.vault_root / "sft-v4"
    destination.mkdir(parents=True, exist_ok=True)
    pruned_checkpoint_copies = _prune_sft_checkpoint_copies(config.vault_root)
    if pruned_checkpoint_copies:
        print(
            "[SFT] Removed obsolete archived checkpoint copies; canonical file kept: "
            f"{destination / 'anra-v4-current-full-resume.pt'}"
        )
    probe = destination / ".sft-v4-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
    finally:
        probe.unlink(missing_ok=True)
    report = {
        "passed": True,
        "mode": config.mode,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu-pilot",
        "lineage_id": lineage["lineage_id"],
        "base_checkpoint_sha256": dict(lineage["parent"])["base_checkpoint_sha256"],
        "tokenizer_sha256": dict(lineage["tokenizer"])["sha256"],
        "dataset_examples": len(examples),
        "validation_examples": len(validation_examples),
        "category_counts": dict(dataset_manifest["category_counts"]),
        "validation_category_counts": dict(validation_manifest["category_counts"]),
        "sft_vault": str(destination),
        "pruned_checkpoint_copies": list(pruned_checkpoint_copies),
        "full_approval": approval,
        "next_action": "run SFT pilot" if config.mode == "pilot" else "run approved full SFT",
    }
    write_json(destination / "ready_to_sft.json", report)
    return report


def _next_batch_indices(
    length: int, batch_size: int, seed: int, epoch: int, cursor: int
) -> tuple[list[int], int, int]:
    generator = torch.Generator().manual_seed(int(seed) + int(epoch))
    ordering = torch.randperm(length, generator=generator).tolist()
    if cursor >= len(ordering):
        epoch += 1
        cursor = 0
        generator = torch.Generator().manual_seed(int(seed) + int(epoch))
        ordering = torch.randperm(length, generator=generator).tolist()
    result = ordering[cursor : cursor + batch_size]
    return result, epoch, cursor + len(result)


def _validate_sft_sampler_position(
    *,
    dataset_size: int,
    global_step: int,
    batch_size: int,
    accumulation: int,
    epoch: int,
    cursor: int,
) -> int:
    """Prove that optimizer progress and the deterministic data cursor agree."""

    if dataset_size <= 0 or not 0 <= cursor <= dataset_size or epoch < 0:
        raise ValueError("invalid SFT sampler position")
    observed_examples = epoch * dataset_size + cursor
    expected_examples = global_step * batch_size * accumulation
    if observed_examples != expected_examples:
        raise ValueError(
            "SFT checkpoint progress is inconsistent: "
            f"step/recipe imply {expected_examples:,} examples but epoch/cursor "
            f"record {observed_examples:,}"
        )
    return observed_examples


def _recover_sft_token_counters(
    dataset: SFTConversationDataset,
    *,
    seed: int,
    epoch: int,
    cursor: int,
) -> tuple[int, int]:
    """Recover exact legacy counters from a verified deterministic sampler position."""

    lengths: list[tuple[int, int]] = []
    for index in range(len(dataset)):
        row = dataset[index]
        lengths.append(
            (int(row["input_ids"].numel()), int(row["weights"].sum().item()))
        )
    all_input = sum(item[0] for item in lengths)
    all_supervised = sum(item[1] for item in lengths)
    generator = torch.Generator().manual_seed(int(seed) + int(epoch))
    ordering = torch.randperm(len(dataset), generator=generator).tolist()[:cursor]
    return (
        epoch * all_input + sum(lengths[index][0] for index in ordering),
        epoch * all_supervised + sum(lengths[index][1] for index in ordering),
    )


_BEHAVIOR_SMOKE_PROMPTS: tuple[tuple[str, str], ...] = (
    ("instruction_following", "Give two concise steps for organizing a small project."),
    ("dialogue", "Respond warmly to a person who says they had a difficult day."),
    ("code", "Write a Python function that returns the larger of two numbers."),
    ("mathematics", "What is 17 plus 28? Show the arithmetic briefly."),
    ("decomposition", "Break preparing a healthy breakfast into three steps."),
    ("tool_contracts", "Show a minimal JSON object describing a successful tool result."),
    ("uncertainty", "How should you answer when you do not have enough evidence?"),
    ("correction", "Rewrite this sentence clearly: The results was not consistent."),
)


def _behavior_smoke_report(
    model: torch.nn.Module,
    tokenizer: object,
    *,
    device: torch.device,
    max_new_tokens: int = 24,
) -> dict[str, object]:
    """Run a tiny deterministic generation gate before full-SFT approval."""

    was_training = model.training
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for category, prompt in _BEHAVIOR_SMOKE_PROMPTS:
            prefix = tokenizer.encode(render_chat_prompt([], prompt), add_special_tokens=False)
            ids = [int(tokenizer.bos_token_id), *prefix]
            ids = ids[-2047:]
            generated: list[int] = []
            for _ in range(max_new_tokens):
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                output = model(input_ids)
                logits = output[0] if isinstance(output, tuple) else output
                next_id = int(torch.argmax(logits[0, -1, :].float()).item())
                if next_id == int(tokenizer.eos_token_id):
                    break
                generated.append(next_id)
                ids.append(next_id)
            text = str(tokenizer.decode(generated)).strip()
            tokens = len(generated)
            unique_tokens = len(set(generated))
            rows.append(
                {
                    "category": category,
                    "prompt": prompt,
                    "output": text,
                    "generated_tokens": tokens,
                    "unique_tokens": unique_tokens,
                    "nonempty": bool(text),
                }
            )
            behavior_pass, requirement = check_smoke_response(category, text)
            rows[-1]["behavior_pass"] = behavior_pass
            rows[-1]["requirement"] = requirement
    unique_outputs = len({" ".join(str(row["output"]).split()) for row in rows})
    passed = bool(rows) and all(bool(row["behavior_pass"]) for row in rows)
    report = {
        "schema": "anra-sft-behavior-smoke/v1",
        "passed": passed,
        "prompt_count": len(rows),
        "unique_output_count": unique_outputs,
        "max_new_tokens": max_new_tokens,
        "rows": rows,
    }
    model.train(was_training)
    return report


def run_sft_v4(config: SFTRunConfig) -> dict[str, object]:
    """Run or resume an auditable SFT session, publishing only the SFT child."""

    report = preflight_sft_v4(config)
    signed_approval = report.get("full_approval")
    device = torch.device(str(report["device"]))
    lineage = verify_sft_lineage_manifest(
        config.lineage_manifest,
        signing_key=config.signing_key,
        artifact_paths={
            "base_checkpoint_path": config.base_checkpoint,
            "tokenizer_path": active_tokenizer_path(),
            "dataset_manifest_path": config.dataset_manifest,
            "validation_manifest_path": config.validation_manifest,
        },
    )
    if not isinstance(lineage.get("evaluation"), Mapping):
        raise ValueError("SFT lineage has no signed validation split")
    examples, dataset_manifest = load_sft_examples(config.dataset_manifest)
    validation_examples, validation_manifest = load_sft_validation_examples(
        config.validation_manifest
    )
    seed_everything(config.seed)
    tokenizer = load_or_build_v2_tokenizer(dataset_path=config.dataset_manifest)
    tokenizer_contract = active_tokenizer_identity()
    if tokenizer_contract.get("available") is not True:
        raise RuntimeError("active V4 tokenizer identity is unavailable")
    dataset = SFTConversationDataset(examples, tokenizer, block_size=2048)
    validation_dataset = SFTConversationDataset(validation_examples, tokenizer, block_size=2048)
    model = build_model_for_profile(
        CANONICAL_MODEL_PROFILE,
        block_size=2048,
        vocab_size=tokenizer.vocab_size,
        use_mtp=False,
        use_moe=False,
    ).to(device)
    ensure_tied_lm_head(model)
    recipe = _sft_recipe(
        seed=config.seed,
        batch_size=config.batch_size,
        accumulation=config.accumulation,
        total_steps=config.total_steps,
    )
    native_model = getattr(model, "model", model)
    sft_remote_checkpoint = config.vault_root / "sft-v4" / "anra-v4-current-full-resume.pt"
    resuming = sft_remote_checkpoint.is_file()
    if resuming:
        _verify_resume_checkpoint_binding(sft_remote_checkpoint, lineage)
        _copy_verified(sft_remote_checkpoint, config.local_checkpoint)
        native_model.training_recipe = dict(recipe)
    else:
        # Empty recipe intentionally loads the V4 parent as weights only. The
        # parent optimizer, scheduler, RNG and causal-data cursor never cross
        # into the separate SFT lineage.
        native_model.training_recipe = {}
    optimizer = build_optimizer(
        model,
        lr=SFT_DEFAULT_LEARNING_RATE,
        weight_decay=float(getattr(ANRA_V4_TRAINING, "weight_decay", 0.1)),
        optimizer_name=CANONICAL_FOUNDATION_OPTIMIZER,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=max(25, int(config.total_steps * 0.02)),
        total_steps=config.total_steps,
        min_lr_ratio=0.1,
    )
    mp = MixedPrecisionTrainer(device=device)
    data_generator = make_data_generator(config.seed)
    load_path = config.local_checkpoint if resuming else config.base_checkpoint
    loaded = load_checkpoint(
        model,
        optimizer if resuming else None,
        scheduler if resuming else None,
        mp if resuming else None,
        load_path,
        device=device,
        strict=True,
        resume_training=resuming,
        data_generator=data_generator if resuming else None,
    )
    load_report = loaded.get("load_report", {})
    if (
        not loaded.get("loaded")
        or not isinstance(load_report, Mapping)
        or not load_report.get("exact_core_load")
    ):
        raise RuntimeError("V4 SFT parent/resume checkpoint failed exact core loading")
    native_model.training_recipe = dict(recipe)
    global_step = int(loaded.get("global_step", 0)) if resuming else 0
    epoch = int(loaded.get("epoch", 0)) if resuming else 0
    cursor = int(loaded.get("sft_microbatch_cursor", 0)) if resuming else 0
    skipped_updates = int(loaded.get("sft_skipped_updates", 0)) if resuming else 0
    input_tokens_processed = int(loaded.get("sft_input_tokens_processed", 0)) if resuming else 0
    supervised_tokens_processed = (
        int(loaded.get("sft_supervised_tokens_processed", 0)) if resuming else 0
    )
    if resuming:
        _validate_sft_sampler_position(
            dataset_size=len(dataset),
            global_step=global_step,
            batch_size=config.batch_size,
            accumulation=config.accumulation,
            epoch=epoch,
            cursor=cursor,
        )
        has_token_counters = {
            "sft_input_tokens_processed",
            "sft_supervised_tokens_processed",
        }.issubset(loaded)
        if not has_token_counters:
            input_tokens_processed, supervised_tokens_processed = _recover_sft_token_counters(
                dataset,
                seed=config.seed,
                epoch=epoch,
                cursor=cursor,
            )
            print(
                "[SFT] Recovered exact token counters from the verified legacy sampler cursor",
                flush=True,
            )
    best_validation_loss = (
        float(loaded.get("best_validation_loss", math.inf)) if resuming else math.inf
    )
    parent_validation_loss, migrated_parent_baseline = _resume_parent_validation_loss(
        loaded,
        resuming=resuming,
        mode=config.mode,
        signed_approval=signed_approval,
    )
    if migrated_parent_baseline:
        print(
            "[SFT] Migrated the legacy resume checkpoint's parent validation "
            "baseline from the verified full-SFT approval. It will be persisted "
            "in the next checkpoint."
        )
    if resuming and not math.isfinite(parent_validation_loss):
        raise RuntimeError(
            "SFT resume checkpoint lacks the immutable parent validation baseline; "
            "start a fresh signed SFT lineage instead of relabeling child metrics"
        )
    _configure_durability(
        config.vault_root,
        config.local_checkpoint.parent / "sft-durability-outbox",
        lineage_id=str(lineage["lineage_id"]),
    )
    durability = CheckpointDurabilitySession.from_environment(
        config.local_checkpoint.parent / "sft-durability-outbox",
        scratch_run=not resuming,
    )
    started = time.monotonic()
    deadline = started + 60 * int(config.max_minutes)
    next_checkpoint = time.monotonic() + 60 * int(config.checkpoint_minutes)
    behavior_history = (
        [dict(row) for row in loaded.get("sft_behavior_history", [])]
        if resuming
        else []
    )
    # Bound checkpoint growth while retaining a complete enough trajectory to
    # diagnose a late behavioral collapse across notebook handoffs.
    behavior_history = behavior_history[-24:]

    def save(*, final: bool, train_loss: float) -> None:
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            mp=mp,
            global_step=global_step,
            epoch=epoch,
            microbatch_cursor=cursor,
            skipped_updates=skipped_updates,
            input_tokens_processed=input_tokens_processed,
            supervised_tokens_processed=supervised_tokens_processed,
            best_validation_loss=best_validation_loss,
            parent_validation_loss=parent_validation_loss,
            train_loss=train_loss,
            tokenizer_contract=tokenizer_contract,
            recipe=recipe,
            lineage=lineage,
            dataset_manifest=dataset_manifest,
            data_generator=data_generator,
            behavior_history=behavior_history,
        )
        atomic_save(payload, config.local_checkpoint, drive_dir=None)
        ref = durability.publish_checkpoint(config.local_checkpoint, payload, final=final)
        if ref is not None and (final or durability.requires_initial_boundary):
            assert durability.publisher is not None
            durability.publisher.wait_for(
                ref,
                DurabilityState.PROTECTED,
                timeout_seconds=durability.ack_timeout_seconds,
            )

    def validation_loss(limit: int = 64) -> float:
        model.eval()
        total_nll = 0.0
        total_supervised_tokens = 0.0
        with torch.no_grad():
            for start in range(0, min(len(validation_dataset), limit), config.batch_size):
                rows = [
                    validation_dataset[index]
                    for index in range(
                        start, min(len(validation_dataset), start + config.batch_size)
                    )
                ]
                batch = validation_dataset.collate(rows)
                ids = batch["input_ids"].to(device)
                targets = batch["targets"].to(device)
                weights = batch["weights"].to(device)
                output = model(ids)
                logits = output[0] if isinstance(output, tuple) else output
                per_token = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
                total_nll += float((per_token * weights).sum().item())
                total_supervised_tokens += float(weights.sum().item())
        model.train()
        return total_nll / max(1.0, total_supervised_tokens)

    model.train()
    if not resuming:
        # On a fresh SFT lineage the currently loaded weights are exactly the
        # frozen foundation parent. Persist this baseline so every later
        # handoff evaluates progress against the same held-out measurement.
        parent_validation_loss = validation_loss()
    last_loss = math.inf
    while time.monotonic() < deadline and global_step < config.total_steps:
        optimizer.zero_grad(set_to_none=True)
        pending_epoch, pending_cursor = epoch, cursor
        pending_input_tokens = 0
        pending_supervised_tokens = 0
        for _ in range(config.accumulation):
            indices, pending_epoch, pending_cursor = _next_batch_indices(
                len(dataset), config.batch_size, config.seed, pending_epoch, pending_cursor
            )
            batch = dataset.collate([dataset[index] for index in indices])
            ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            weights = batch["weights"].to(device)
            pending_input_tokens += int((batch["input_ids"] != dataset.pad_id).sum().item())
            pending_supervised_tokens += int(batch["weights"].sum().item())
            with mp.autocast():
                output = model(ids)
                logits = output[0] if isinstance(output, tuple) else output
                loss = assistant_only_loss(logits, targets, weights) / config.accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "SFT loss became non-finite; checkpoint remains at the last safe boundary"
                )
            mp.backward(loss)
            last_loss = float(loss.item() * config.accumulation)
        mp.clip_gradients(
            model,
            optimizer,
            max_norm=float(getattr(ANRA_V4_TRAINING, "max_grad_norm", 1.0)),
        )
        mp.step(optimizer)
        optimizer_advanced = mp.update()
        if optimizer_advanced:
            epoch, cursor = pending_epoch, pending_cursor
            input_tokens_processed += pending_input_tokens
            supervised_tokens_processed += pending_supervised_tokens
            scheduler.step()
            global_step += 1
        else:
            skipped_updates += 1
            print(
                "[SFT] AMP skipped a non-finite optimizer update; retrying same data boundary",
                flush=True,
            )
        optimizer.zero_grad(set_to_none=True)
        if optimizer_advanced and global_step % min(config.checkpoint_steps, 50) == 0:
            validation = validation_loss()
            best_validation_loss = min(best_validation_loss, validation)
            print(
                f"[SFT] step={global_step} train_loss={last_loss:.4f} "
                f"validation_loss={validation:.4f} "
                f"input_tokens={input_tokens_processed:,} "
                f"supervised_tokens={supervised_tokens_processed:,}",
                flush=True,
            )
        # A falling assistant-token NLL can coexist with one generic response
        # being emitted for unrelated prompts. Probe the same fixed, strict
        # behavior suite during a run so that artifact selection has evidence
        # beyond loss. This records evidence; it does not silently terminate a
        # legitimate early-learning phase.
        if optimizer_advanced and global_step % config.behavior_probe_steps == 0:
            behavior_probe = _behavior_smoke_report(model, tokenizer, device=device)
            behavior_history.append(
                {
                    "step": global_step,
                    "input_tokens_processed": input_tokens_processed,
                    "supervised_tokens_processed": supervised_tokens_processed,
                    "validation_loss": best_validation_loss,
                    "behavior": behavior_probe,
                }
            )
            behavior_history = behavior_history[-24:]
            print(
                "[SFT] behavior_probe "
                f"step={global_step} passed={behavior_probe['passed']} "
                f"unique_outputs={behavior_probe['unique_output_count']}/"
                f"{behavior_probe['prompt_count']}",
                flush=True,
            )
        if (
            durability.requires_initial_boundary
            or global_step % config.checkpoint_steps == 0
            or time.monotonic() >= next_checkpoint
        ):
            save(final=False, train_loss=last_loss)
            next_checkpoint = time.monotonic() + 60 * int(config.checkpoint_minutes)
    if math.isinf(best_validation_loss):
        best_validation_loss = validation_loss()
    try:
        behavior_smoke = _behavior_smoke_report(model, tokenizer, device=device)
    except Exception as exc:  # pragma: no cover - device-specific diagnostics
        behavior_smoke = {
            "schema": "anra-sft-behavior-smoke/v1",
            "passed": False,
            "prompt_count": 0,
            "unique_output_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not behavior_history or int(behavior_history[-1].get("step", -1)) != global_step:
        behavior_history.append(
            {
                "step": global_step,
                "input_tokens_processed": input_tokens_processed,
                "supervised_tokens_processed": supervised_tokens_processed,
                "validation_loss": best_validation_loss,
                "behavior": behavior_smoke,
            }
        )
        behavior_history = behavior_history[-24:]
    # The final durable state must include the final behavioral evidence, not
    # only the pre-final loss. Otherwise a crash after the report write would
    # let a later worker resume without the decisive acceptance signal.
    save(final=True, train_loss=last_loss)
    durability.close()
    result = {
        **report,
        "resumed": resuming,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "parent_validation_loss": parent_validation_loss,
        "validation_delta_from_parent": best_validation_loss - parent_validation_loss,
        "last_train_loss": last_loss,
        "skipped_optimizer_updates": skipped_updates,
        "validation_manifest_sha256": sha256_file(config.validation_manifest),
        "train_manifest_sha256": sha256_file(config.dataset_manifest),
        "base_checkpoint_sha256": str(dict(lineage["parent"])["base_checkpoint_sha256"]),
        "lineage_id": str(lineage["lineage_id"]),
        "validation_examples": len(validation_examples),
        "input_tokens_processed": input_tokens_processed,
        "supervised_tokens_processed": supervised_tokens_processed,
        "checkpoint": str(sft_remote_checkpoint),
        "checkpoint_sha256": sha256_file(sft_remote_checkpoint),
        "elapsed_minutes": round((time.monotonic() - started) / 60, 2),
        "behavior_smoke": behavior_smoke,
        "behavior_history": behavior_history,
    }
    report_path = config.vault_root / "sft-v4" / "latest_sft_report.json"
    _atomic_write_json(report_path, result)
    readiness = {
        "schema": "anra-sft-readiness/v1",
        "lineage_id": str(lineage["lineage_id"]),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "train_manifest_sha256": result["train_manifest_sha256"],
        "validation_manifest_sha256": result["validation_manifest_sha256"],
        "global_step": result["global_step"],
        "validation_improved": (
            math.isfinite(float(result["parent_validation_loss"]))
            and math.isfinite(float(result["best_validation_loss"]))
            and float(result["best_validation_loss"])
            < float(result["parent_validation_loss"])
        ),
        "behavior_smoke_passed": bool(
            isinstance(behavior_smoke, Mapping) and behavior_smoke.get("passed") is True
        ),
        "full_sft_ready": bool(
            isinstance(behavior_smoke, Mapping)
            and behavior_smoke.get("passed") is True
            and float(result["best_validation_loss"])
            < float(result["parent_validation_loss"])
        ),
        "next_action": "approve_full" if (
            isinstance(behavior_smoke, Mapping)
            and behavior_smoke.get("passed") is True
            and float(result["best_validation_loss"])
            < float(result["parent_validation_loss"])
        ) else "review_pilot_and_fix_before_full",
    }
    _atomic_write_json(config.vault_root / "sft-v4" / "ready_to_sft.json", readiness)
    return result


def _config_from_args(args: argparse.Namespace) -> SFTRunConfig:
    key = os.environ.get(args.signing_key_env, "")
    if not key:
        raise PermissionError(
            f"missing SFT signing key in environment variable {args.signing_key_env}"
        )
    return SFTRunConfig(
        dataset_manifest=Path(args.dataset_manifest).resolve(),
        validation_manifest=Path(args.validation_manifest).resolve(),
        source_receipt=Path(args.source_receipt).resolve(),
        lineage_manifest=Path(args.lineage_manifest).resolve(),
        base_checkpoint=Path(args.base_checkpoint).resolve(),
        vault_root=Path(args.vault_root).resolve(),
        local_checkpoint=Path(args.local_checkpoint).resolve(),
        signing_key=key,
        mode=args.mode,
        max_minutes=args.max_minutes,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
        seed=args.seed,
        total_steps=args.total_steps,
        checkpoint_steps=args.checkpoint_steps,
        checkpoint_minutes=args.checkpoint_minutes,
        behavior_probe_steps=args.behavior_probe_steps,
        allow_cpu_pilot=args.allow_cpu_pilot,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical An-Ra V4 SFT trainer")
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare-lineage")
    prepare.add_argument("--lineage-id", required=True)
    prepare.add_argument("--dataset-manifest", required=True)
    prepare.add_argument("--validation-manifest", required=True)
    prepare.add_argument("--source-receipt", required=True)
    prepare.add_argument("--base-checkpoint", required=True)
    prepare.add_argument("--tokenizer", default=str(active_tokenizer_path()))
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--signing-key-env", default="ANRA_MANIFEST_SIGNING_KEY")
    prepare.add_argument("--key-id", default="owner")
    approve = subcommands.add_parser("approve-full")
    approve.add_argument("--dataset-manifest", required=True)
    approve.add_argument("--validation-manifest", required=True)
    approve.add_argument("--lineage-manifest", required=True)
    approve.add_argument("--base-checkpoint", required=True)
    approve.add_argument("--vault-root", required=True)
    approve.add_argument("--signing-key-env", default="ANRA_MANIFEST_SIGNING_KEY")
    approve.add_argument("--owner-approval", required=True)
    for name in ("preflight", "run"):
        command = subcommands.add_parser(name)
        command.add_argument("--dataset-manifest", required=True)
        command.add_argument("--validation-manifest", required=True)
        command.add_argument("--source-receipt", required=True)
        command.add_argument("--lineage-manifest", required=True)
        command.add_argument("--base-checkpoint", required=True)
        command.add_argument("--vault-root", required=True)
        command.add_argument("--local-checkpoint", required=True)
        command.add_argument("--signing-key-env", default="ANRA_MANIFEST_SIGNING_KEY")
        command.add_argument("--mode", choices=["pilot", "full"], default="pilot")
        command.add_argument("--max-minutes", type=int, default=15)
        command.add_argument("--max-examples", type=int, default=None)
        command.add_argument("--batch-size", type=int, default=1)
        command.add_argument("--accumulation", type=int, default=8)
        command.add_argument("--seed", type=int, default=CANONICAL_TRAINING_SEED)
        command.add_argument("--total-steps", type=int, default=SFT_DEFAULT_TOTAL_STEPS)
        command.add_argument("--checkpoint-steps", type=int, default=SFT_DEFAULT_CHECKPOINT_STEPS)
        command.add_argument(
            "--checkpoint-minutes", type=int, default=SFT_DEFAULT_CHECKPOINT_MINUTES
        )
        command.add_argument(
            "--behavior-probe-steps",
            type=int,
            default=500,
            help="Run and persist the fixed strict SFT behavior probe at this optimizer interval.",
        )
        command.add_argument("--allow-cpu-pilot", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-lineage":
        key = os.environ.get(args.signing_key_env, "")
        if not key:
            raise PermissionError(f"missing signing key in {args.signing_key_env}")
        dataset = _read_json(Path(args.dataset_manifest))
        _verify_source_receipt_hash(
            Path(args.source_receipt), str(dataset.get("source_receipt_sha256", ""))
        )
        payload = write_sft_lineage_manifest(
            args.output,
            lineage_id=args.lineage_id,
            dataset_manifest_path=args.dataset_manifest,
            validation_manifest_path=args.validation_manifest,
            base_checkpoint_path=args.base_checkpoint,
            tokenizer_path=args.tokenizer,
            source_commit=_source_commit(),
            signing_key=key,
            key_id=args.key_id,
        )
        print(json.dumps(payload, indent=2))
        return
    if args.command == "approve-full":
        key = os.environ.get(args.signing_key_env, "")
        if not key:
            raise PermissionError(f"missing signing key in {args.signing_key_env}")
        lineage = verify_sft_lineage_manifest(
            args.lineage_manifest,
            signing_key=key,
            artifact_paths={
                "base_checkpoint_path": args.base_checkpoint,
                "tokenizer_path": active_tokenizer_path(),
                "dataset_manifest_path": args.dataset_manifest,
                "validation_manifest_path": args.validation_manifest,
            },
        )
        approval = _write_full_sft_approval(
            vault_root=Path(args.vault_root).resolve(),
            lineage=lineage,
            signing_key=key,
            owner_approval=args.owner_approval,
        )
        print(json.dumps({"approval": str(approval), "lineage_id": lineage["lineage_id"]}))
        return
    config = _config_from_args(args)
    result = preflight_sft_v4(config) if args.command == "preflight" else run_sft_v4(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
