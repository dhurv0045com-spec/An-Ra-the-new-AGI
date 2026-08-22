"""Pure, CPU-testable training-state contracts used by the XLA trainer."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data.distributed import DistributedSampler


def tokens_per_optimizer_step(
    *, batch_size: int, grad_accum_steps: int, world_size: int, sequence_length: int
) -> int:
    values = (batch_size, grad_accum_steps, world_size, sequence_length)
    if any(int(value) <= 0 for value in values):
        raise ValueError("batch, accumulation, world size, and sequence length must be positive")
    return math.prod(int(value) for value in values)


def dataset_fingerprint(path: Path) -> str:
    """Return a path-independent identity for a text file or token-shard pack.

    Shard content hashes are intentionally included.  This preflight is slower
    than hashing names and sizes, but it prevents a resumed run from silently
    consuming different tokens under the same filenames.
    """
    path = path.expanduser().resolve()
    files = sorted(path.rglob("*.npy")) if path.is_dir() else [path]
    if not files or any(not item.is_file() for item in files):
        raise FileNotFoundError(path)
    root = path if path.is_dir() else path.parent
    digest = hashlib.sha256()
    digest.update(b"anra-training-dataset/v1\0")
    for item in files:
        relative = item.relative_to(root).as_posix() if path.is_dir() else item.name
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(item.stat().st_size).encode("ascii") + b"\0")
        with item.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DataPosition:
    epoch: int
    batch_in_epoch: int
    microbatches_consumed: int

    @classmethod
    def from_microbatches(cls, consumed: int, batches_per_epoch: int) -> "DataPosition":
        if consumed < 0 or batches_per_epoch <= 0:
            raise ValueError("data cursor and batches_per_epoch must be valid")
        epoch, batch = divmod(int(consumed), int(batches_per_epoch))
        return cls(epoch=epoch, batch_in_epoch=batch, microbatches_consumed=int(consumed))


class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler that can begin at a saved sample offset in an epoch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def set_start_index(self, start_index: int) -> None:
        if start_index < 0 or start_index > self.num_samples:
            raise ValueError("sampler start index is outside this rank's epoch")
        self.start_index = int(start_index)

    def __iter__(self) -> Iterator[int]:
        indices = list(super().__iter__())
        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return self.num_samples - self.start_index


@dataclass(frozen=True, slots=True)
class CosineSchedule:
    """Checkpoint-stable cosine decay anchored to a continuation boundary."""

    base_lr: float
    min_lr: float
    origin_step: int
    decay_steps: int

    def __post_init__(self) -> None:
        if self.base_lr <= 0 or self.min_lr < 0 or self.min_lr > self.base_lr:
            raise ValueError("invalid learning-rate bounds")
        if self.origin_step < 0 or self.decay_steps <= 0:
            raise ValueError("invalid schedule origin or duration")

    def lr_at(self, update_step: int) -> float:
        progress = min(1.0, max(0.0, (update_step - self.origin_step) / self.decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def to_dict(self) -> dict[str, object]:
        return {"name": "cosine_continuation_v1", **asdict(self)}

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict[str, Any],
        *,
        start_step: int,
        checkpoint_lr: float,
        decay_steps: int,
        min_lr_ratio: float,
    ) -> "CosineSchedule":
        saved = payload.get("lr_schedule")
        if isinstance(saved, dict) and saved.get("name") == "cosine_continuation_v1":
            return cls(
                base_lr=float(saved["base_lr"]),
                min_lr=float(saved["min_lr"]),
                origin_step=int(saved["origin_step"]),
                decay_steps=int(saved["decay_steps"]),
            )
        return cls(
            base_lr=float(checkpoint_lr),
            min_lr=float(checkpoint_lr) * float(min_lr_ratio),
            origin_step=int(start_step),
            decay_steps=int(decay_steps),
        )


def validate_full_resume(payload: dict[str, Any], *, minimum_step: int) -> int:
    if payload.get("checkpoint_artifact_class") != "full_resume":
        raise ValueError("--resume-from must be a full_resume checkpoint")
    if not isinstance(payload.get("optimizer_state_dict"), dict):
        raise ValueError("resume checkpoint is missing optimizer_state_dict")
    schema_version = payload.get("checkpoint_schema_version")
    if schema_version not in {1, 2}:
        raise ValueError(f"unsupported full-resume schema version: {schema_version!r}")
    if schema_version == 2:
        trainer_state = payload.get("trainer_state")
        schedule = payload.get("lr_schedule")
        if not isinstance(trainer_state, dict) or trainer_state.get("schema") != "anra-training-state/v2":
            raise ValueError("schema-v2 checkpoint is missing valid trainer_state")
        if not isinstance(schedule, dict) or schedule.get("name") != "cosine_continuation_v1":
            raise ValueError("schema-v2 checkpoint is missing its cosine LR schedule")
    step = payload.get("global_step")
    if not isinstance(step, int) or step < minimum_step:
        raise ValueError(
            f"resume checkpoint step must be at least {minimum_step:,}; got {step!r}"
        )
    trainer_step = payload.get("trainer_state", {}).get("global_step")
    if trainer_step is not None and int(trainer_step) != step:
        raise ValueError(
            f"checkpoint global_step={step} disagrees with trainer_state={trainer_step}"
        )
    return step


def build_training_state(
    *,
    step: int,
    optimizer_updates: int,
    position: DataPosition,
    dataset_sha256: str,
    dataset_windows: int,
    batch_size: int,
    grad_accum_steps: int,
    world_size: int,
    sequence_length: int,
    seed: int,
    attention_chunk_size: int,
    gradient_checkpointing: bool,
    gradient_clip_norm: float,
    precision: str = "bf16",
) -> dict[str, object]:
    return {
        "schema": "anra-training-state/v2",
        "global_step": int(step),
        "optimizer_updates": int(optimizer_updates),
        "data": asdict(position),
        "dataset_sha256": dataset_sha256,
        "dataset_windows": int(dataset_windows),
        "batch_size_per_core": int(batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "world_size": int(world_size),
        "sequence_length": int(sequence_length),
        "tokens_per_optimizer_step": tokens_per_optimizer_step(
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            world_size=world_size,
            sequence_length=sequence_length,
        ),
        "sampler_seed": int(seed),
        "rng_scheme": "rank_seed_plus_global_step_v1",
        "attention_chunk_size": int(attention_chunk_size),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "gradient_clip_norm": float(gradient_clip_norm),
        "precision": str(precision),
    }


def validate_training_state(
    saved: dict[str, Any], current: dict[str, object], *, allow_legacy: bool
) -> None:
    if not saved:
        if allow_legacy:
            return
        raise ValueError(
            "checkpoint predates exact data-resume metadata; pass --allow-legacy-resume once "
            "to establish a v2 continuation boundary"
        )
    if saved.get("schema") != "anra-training-state/v2":
        raise ValueError(f"unsupported trainer_state schema: {saved.get('schema')!r}")
    fields = (
        "dataset_sha256",
        "dataset_windows",
        "batch_size_per_core",
        "grad_accum_steps",
        "world_size",
        "sequence_length",
        "tokens_per_optimizer_step",
        "sampler_seed",
        "rng_scheme",
        "attention_chunk_size",
        "gradient_checkpointing",
        "gradient_clip_norm",
        "precision",
    )
    drift = {
        name: {"checkpoint": saved.get(name), "current": current.get(name)}
        for name in fields
        if saved.get(name) != current.get(name)
    }
    if drift:
        raise ValueError(f"resume training recipe drift: {json.dumps(drift, sort_keys=True)}")
