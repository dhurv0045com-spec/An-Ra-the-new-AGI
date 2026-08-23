"""Canonical TPU/XLA trainer for An-Ra V4.

This is intentionally separate from the CUDA trainer.  It uses one XLA process
per TPU core, a distributed sampler, BF16 autocast, synchronized optimizer
steps, and one atomically replaced full-resume checkpoint.  The module imports
torch_xla lazily so local CPU tooling and tests do not require TPU packages.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

from anra_core.config import CANONICAL_CONFIG
from anra_core.model import AnRaCore
from anra_core.tokenizer import V4Tokenizer
from training.state import (
    CosineSchedule,
    DataPosition,
    ResumableDistributedSampler,
    build_training_state,
    dataset_fingerprint,
    tokens_per_optimizer_step,
    validate_training_state,
)
from training.wsd_scheduler import PackWsdSchedule


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Deterministic fixed-length next-token windows."""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if len(token_ids) <= block_size:
            raise ValueError("dataset must contain more tokens than block_size")
        self.block_size = block_size
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.count = (len(token_ids) - 1) // block_size

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.block_size
        chunk = self.tokens[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class TokenShardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read the immutable V4 ``train/*.npy`` token pack without concatenating it.

    Each pack shard contains independent ``block_size`` windows plus one target
    token.  Memory-mapped shards keep the TPU host from making eight full copies
    of a 330M-token continuation pack when ``xmp.spawn`` starts workers.
    """

    def __init__(self, root: Path, block_size: int) -> None:
        files = sorted(root.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"no token shards (*.npy) found in {root}")
        self.block_size = int(block_size)
        self._arrays: list[np.ndarray] = []
        self._ends: list[int] = []
        total = 0
        for path in files:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"token shard must be a 1-D integer array: {path}")
            windows = (int(array.shape[0]) - 1) // self.block_size
            if windows <= 0:
                continue
            self._arrays.append(array)
            total += windows
            self._ends.append(total)
        if not self._arrays or total <= 0:
            raise ValueError(f"token pack contains no complete training windows: {root}")
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        shard = bisect_right(self._ends, index)
        previous_end = 0 if shard == 0 else self._ends[shard - 1]
        local_index = index - previous_end
        start = local_index * self.block_size
        # Copy the tiny window before converting to int64; the source is a
        # read-only mmap and torch must never retain a writable alias to it.
        values = np.asarray(
            self._arrays[shard][start : start + self.block_size + 1], dtype=np.int64
        ).copy()
        tokens = torch.from_numpy(values)
        return tokens[:-1], tokens[1:]


def _require_xla() -> tuple[Any, Any, Any]:
    """Load XLA only inside a TPU worker and provide an actionable error."""
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ImportError as exc:  # pragma: no cover - exercised on non-TPU hosts
        raise RuntimeError(
            "TPU training requires the Kaggle TPU runtime with torch_xla installed. "
            "Select Accelerator = TPU v5e-8 and restart the session."
        ) from exc
    return xm, pl, xmp


def _read_text(path: Path) -> str:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        pieces: list[str] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                value = record.get("text", record.get("content", record.get("prompt", "")))
                pieces.append(str(value))
            else:
                pieces.append(str(record))
        return "\n".join(pieces)
    return path.read_text(encoding="utf-8", errors="replace")


def _load_dataset(path: Path, tokenizer: V4Tokenizer, block_size: int) -> TokenBlockDataset:
    if path.is_dir():
        train_root = path / "train" if (path / "train").is_dir() else path
        dataset = TokenShardDataset(train_root, block_size)
        print(
            f"[TPU data] token pack {train_root} -> {len(dataset):,} windows",
            flush=True,
        )
        return dataset
    if not path.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {path}")
    text = _read_text(path)
    token_ids = tokenizer.encode(text)
    print(f"[TPU data] {path} -> {len(token_ids):,} tokens", flush=True)
    return TokenBlockDataset(token_ids, block_size)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


# --------------------------------------------------------------------------
# CPU preflight: verify pack semantics + explicit checkpoint BEFORE workers.
# --------------------------------------------------------------------------


def preflight(
    *,
    dataset_path: Path,
    checkpoint_path: Path | None,
    block_size: int,
    vocab_size: int,
    expected_resume_step: int = 0,
    start_new_pack: bool = False,
    allow_legacy_resume: bool = False,
) -> dict[str, Any]:
    """Fail-closed validation on CPU before any TPU worker spawns.

    Returns the run identity block used in the receipt. Raises on:
    - pack that is missing/malformed/semantically invalid
    - resume requested without an exact checkpoint path (never guess)
    - checkpoint that cannot be verified
    """
    from training.pack_verify import PackVerificationError, verify_pack

    train_root = dataset_path / "train" if (dataset_path / "train").is_dir() else dataset_path
    try:
        pack = verify_pack(
            train_root.parent if (dataset_path / "train").is_dir() else dataset_path,
            vocab_size=vocab_size,
            expected_block_size=block_size,
        )
    except PackVerificationError as exc:
        raise RuntimeError(f"REFUSING TO TRAIN - pack verification failed: {exc}") from exc

    identity_block: dict[str, Any] = {
        "pack_manifest_sha256": pack.manifest_sha256,
        "pack_total_tokens": pack.total_tokens,
        "pack_windows": pack.total_windows,
        "pack_shards": len(pack.shard_paths),
        "block_size": block_size,
    }

    if checkpoint_path is not None:
        from training.resume import restore_training_state

        # Verify the checkpoint loads through the CANONICAL restore path on a
        # throwaway model - the exact path the workers will use.
        probe_model = AnRaCore(CANONICAL_CONFIG)
        probe_optimizer = torch.optim.AdamW(probe_model.parameters(), lr=1e-4)
        restored = restore_training_state(
            str(checkpoint_path), probe_model, probe_optimizer,
            mode="new_pack_parent" if start_new_pack else "same_pack",
            current_pack_manifest_sha256=pack.manifest_sha256,
            allow_legacy_checkpoint=allow_legacy_resume,
        )
        if restored.global_step < expected_resume_step:
            raise RuntimeError(
                f"parent global_step {restored.global_step:,} is below expected "
                f"{expected_resume_step:,}"
            )
        identity_block.update(
            {
                "parent_checkpoint": str(checkpoint_path),
                "parent_global_step": restored.global_step,
                "parent_parameter_sha256": restored.checkpoint_parameter_sha256,
                "parent_optimizer_restored": restored.optimizer_restored,
                "parent_mode": restored.mode,
                "parent_checkpoint_schema": restored.checkpoint_schema_version,
            }
        )
        del probe_model, probe_optimizer
    else:
        raise RuntimeError(
            "REFUSING TO TRAIN: no checkpoint selected. Set ANRA_TPU_CHECKPOINT "
            "or pass --resume-from with the EXACT parent path (never auto-pick "
            "by highest step: step-30.4k measured worse than step-20k)."
        )
    return identity_block


def write_run_receipt(
    run_dir: Path,
    *,
    identity_block: dict[str, Any],
    config: dict[str, Any],
    world_size: int,
) -> Path:
    """One receipt that makes the run reconstructable. Written pre-training;
    post-training fields are appended by the trainer on completion."""
    import subprocess

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return ""

    receipt = {
        "run_id": time.strftime("%Y%m%d-%H%M%S"),
        "schema": "anra-run-receipt/v1",
        "source_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        **identity_block,
        "world_size": world_size,
        "precision": "bf16",
        **{k: config.get(k) for k in (
            "batch_size", "grad_accum_steps", "learning_rate",
            "weight_decay", "seed", "max_steps", "max_minutes",
            "save_interval", "candidate_interval",
        )},
        "sequence_length": CANONICAL_CONFIG.block_size,
        "tokens_per_optimizer_step": (
            config.get("batch_size", 1) * config.get("grad_accum_steps", 1)
            * world_size * CANONICAL_CONFIG.block_size
        ),
        "schedule": {
            "name": "wsd_pack_v1",
            "parameters": {
                "base_lr": config.get("learning_rate"),
                "min_lr_ratio": config.get("min_lr_ratio"),
                "warmup_fraction": config.get("warmup_fraction"),
                "decay_fraction": config.get("decay_fraction"),
                "total_steps": config.get("_pack_total_steps"),
            },
        },
    }
    path = run_dir / "receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return path
def _optimizer_update_count(state_dict: dict[str, Any]) -> int:
    counts: list[int] = []
    for state in state_dict.get("state", {}).values():
        value = state.get("step")
        if isinstance(value, torch.Tensor):
            counts.append(int(value.detach().cpu().item()))
        elif isinstance(value, (int, float)):
            counts.append(int(value))
    return max(counts, default=0)


def _clip_global_grad_norm(parameters: Iterator[torch.nn.Parameter], max_norm: float) -> torch.Tensor:
    """Clip already-reduced gradients without XLA-host scalar control flow."""
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("optimizer step has no gradients")
    total_squared = torch.zeros((), device=gradients[0].device, dtype=torch.float32)
    for gradient in gradients:
        total_squared.add_(gradient.detach().float().pow(2).sum())
    total_norm = total_squared.sqrt()
    coefficient = torch.clamp(
        torch.tensor(float(max_norm), device=total_norm.device) / (total_norm + 1e-6),
        max=1.0,
    )
    for gradient in gradients:
        gradient.mul_(coefficient.to(gradient.dtype))
    return total_norm


def _bucketed_traceable_all_reduce(
    parameters: Iterator[torch.nn.Parameter],
    *,
    dist: Any,
    xm: Any,
    world_size: int,
    bucket_cap_mb: float = 16.0,
) -> None:
    """Average accumulated gradients with bounded XLA temporary memory.

    PyTorch/XLA DDP can flatten this model's complete gradient set into one
    roughly 799 MiB buffer, which does not fit in the remaining v5e HBM.  This
    uses the supported traceable ``torch.distributed`` collective, but executes
    deterministic small buckets one at a time.  Oversized individual tensors
    are reduced in place and never copied into another full-size buffer.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    bucket_cap_bytes = max(1, int(bucket_cap_mb * 1024 * 1024))
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("optimizer step has no gradients")

    bucket: list[torch.Tensor] = []
    bucket_bytes = 0

    def flush() -> None:
        nonlocal bucket, bucket_bytes
        if not bucket:
            return
        if len(bucket) == 1:
            reduced = bucket[0]
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced.mul_(1.0 / world_size)
        else:
            reduced = torch.cat([gradient.reshape(-1) for gradient in bucket])
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced.mul_(1.0 / world_size)
            offset = 0
            for gradient in bucket:
                count = gradient.numel()
                gradient.copy_(reduced.narrow(0, offset, count).view_as(gradient))
                offset += count
        # Materialize and release this bucket before constructing the next one.
        xm.mark_step()
        bucket = []
        bucket_bytes = 0

    for gradient in gradients:
        gradient_bytes = gradient.numel() * gradient.element_size()
        if bucket and bucket_bytes + gradient_bytes > bucket_cap_bytes:
            flush()
        bucket.append(gradient)
        bucket_bytes += gradient_bytes
        if gradient_bytes >= bucket_cap_bytes:
            flush()
    flush()


def payload_for_schedule(prepared) -> dict[str, Any]:
    """Expose only validated schedule state carried by the prepared resume."""
    return {
        "lr_schedule": prepared.lr_schedule,
        "trainer_state": prepared.trainer_state,
    }


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    model_config,  # CoreConfig dataclass
    training_config: dict[str, Any],  # runtime dict
    tokenizer: V4Tokenizer,
    step: int,
    metrics: dict[str, float],
    source_checkpoint: str | None,
    world_size: int,
    training_state: dict[str, object] | None = None,
    schedule=None,
    artifact_class: str = "full_resume",
) -> dict[str, object]:
    """Checkpoint creation with UNAMBIGUOUS config types (P0-5):
    model_config is the CoreConfig dataclass; training_config is the runtime
    dict. They are never the same variable."""
    raw_model = model.module if hasattr(model, "module") else model
    state = {name: value.detach().cpu() for name, value in raw_model.state_dict().items()}
    state["lm_head.weight"] = state["token_embedding_table.weight"]
    from training.resume import canonical_parameter_sha256

    return {
        "checkpoint_artifact_class": artifact_class,
        "checkpoint_schema_version": (
            3
            if schedule is not None
            and schedule.to_dict().get("name") == "wsd_pack_v1"
            else 2
        ),
        "global_step": int(step),
        "pack_manifest_sha256": training_config.get("pack_manifest_sha256"),
        "training_stage": "pretraining_tpu_xla",
        "source_commit": os.environ.get("ANRA_SOURCE_COMMIT", "unknown"),
        "source_checkpoint": source_checkpoint,
        "model_config": asdict(model_config),
        "parameter_sha256": canonical_parameter_sha256(state),
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "trainer_state": training_state,
        "lr_schedule": schedule.to_dict() if schedule is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "tokenizer_contract": {"available": True, **tokenizer.identity(probe_count=500)},
        "metrics": metrics,
        "execution": {
            "backend": "torch_xla",
            "precision": "bf16",
            "world_size": int(world_size),
        },
    }


def _atomic_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.uploading")
    torch.save(payload, str(temporary))
    os.replace(temporary, path)


def _save_latest(
    xm: Any,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    tokenizer: V4Tokenizer,
    step: int,
    metrics: dict[str, float],
    source_checkpoint: str | None,
    world_size: int,
    training_state: dict[str, object],
    schedule: CosineSchedule | PackWsdSchedule,
) -> None:
    """Write exactly one checkpoint object; never create numbered copies."""
    if not xm.is_master_ordinal():
        xm.rendezvous("checkpoint-written")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.uploading")
    payload = _checkpoint_payload(
        model, optimizer,
        model_config=CANONICAL_CONFIG,
        training_config=config,
        tokenizer=tokenizer, step=step, metrics=metrics,
        source_checkpoint=source_checkpoint, world_size=world_size,
        training_state=training_state, schedule=schedule,
    )
    xm.save(payload, str(temporary), master_only=True)
    os.replace(temporary, path)
    print(f"[TPU checkpoint] step={step:,} path={path}", flush=True)
    xm.rendezvous("checkpoint-written")


def save_candidate(
    output: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_config: dict[str, Any],
    tokenizer: V4Tokenizer,
    step: int,
    metrics: dict[str, float],
    source_checkpoint: str | None,
    world_size: int,
    *,
    training_state: dict[str, object],
    schedule: CosineSchedule | PackWsdSchedule,
) -> Path:
    """Write an immutable, fully resumable sparse recovery candidate.

    Returns the written path. Refuses to overwrite an existing candidate -
    intermediate useful states must never disappear or mutate.
    """
    candidate_path = output.parent / "candidates" / f"anra-v4-step-{step:05d}.pt"
    if candidate_path.exists():
        print(f"[candidate] step={step:,} already exists - preserved", flush=True)
        return candidate_path
    raw_model = model.module if hasattr(model, "module") else model
    payload = _checkpoint_payload(
        raw_model, optimizer,
        model_config=CANONICAL_CONFIG,
        training_config=training_config,
        tokenizer=tokenizer, step=step, metrics=metrics,
        source_checkpoint=source_checkpoint, world_size=world_size,
        training_state=training_state,
        schedule=schedule,
        artifact_class="full_resume",
    )
    _atomic_save(payload, candidate_path)
    return candidate_path


@contextlib.contextmanager
def _bf16_autocast(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    try:
        with torch.autocast(device_type="xla", dtype=torch.bfloat16):
            yield
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(
            "This TPU runtime does not support the required XLA BF16 autocast path; "
            "refusing a silent FP32 fallback."
        ) from exc


def _worker(index: int, config: dict[str, object]) -> None:
    xm, pl, _ = _require_xla()
    import torch_xla
    import torch.distributed as dist
    import torch_xla.distributed.xla_backend  # noqa: F401 - registers xla://
    import torch_xla.runtime as xr

    if not hasattr(torch, "xla") and hasattr(torch, "_register_device_module"):
        torch._register_device_module("xla", torch_xla)

    device = xm.xla_device()
    ordinal = getattr(xr, "global_ordinal", None)
    rank = int(ordinal() if callable(ordinal) else xm.get_ordinal())
    world_size = int(xr.world_size())
    required_world_size = int(config["require_world_size"])
    if world_size != required_world_size:
        raise RuntimeError(
            f"Expected exactly {required_world_size} TPU workers, got {world_size}; "
            "refusing to change the global batch or checkpoint topology"
        )
    # PyTorch/XLA 2.8 supports the upstream distributed API and uses its
    # traceable collectives.  Do not use the legacy xm.all_reduce /
    # xm.reduce_gradients path here: on Kaggle v5e-8 it can segfault inside
    # torch_xla::tensor_methods::all_reduce before the first logged update.
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")

    seed = int(config["seed"])
    # All replicas must construct identical fresh weights. Rank-specific RNG is
    # established only after model construction/loading.
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = V4Tokenizer.load_canonical()
    resume_path = Path(str(config["resume_from"])).expanduser() if config.get("resume_from") else None
    source_checkpoint: str | None = str(resume_path) if resume_path else None

    # CANONICAL PREPARATION PATH (P0-7): one model, one optimizer, one restore.
    # CPU-testable; _worker consumes the returned struct and never re-restores
    # or re-reads raw payloads.
    from training.resume import prepare_training_state

    prepared = prepare_training_state(
        parent_checkpoint=str(resume_path) if resume_path else None,
        model_config=CANONICAL_CONFIG,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        expected_resume_step=int(config.get("expected_resume_step", 0)),
        resume_mode=(
            "new_pack_parent" if bool(config.get("start_new_pack")) else "same_pack"
        ),
        current_pack_manifest_sha256=str(config["pack_manifest_sha256"]),
        allow_legacy_checkpoint=bool(config.get("allow_legacy_resume")),
    )
    model = prepared.model
    optimizer = prepared.optimizer
    start_step = prepared.global_step
    optimizer_updates = prepared.optimizer_updates
    if rank == 0:
        print(
            f"[TPU prepare] mode={prepared.resume_mode} step={start_step:,} "
            f"optimizer_updates={optimizer_updates:,} "
            f"parameter_sha256={prepared.checkpoint_parameter_sha256[:16]}",
            flush=True,
        )

    random.seed(seed + rank + start_step)
    torch.manual_seed(seed + rank + start_step)

    model = model.to(device)
    model.train()
    model.enable_gradient_checkpointing(bool(config["gradient_checkpointing"]))
    model.enable_memory_efficient_attention(int(config["attention_chunk_size"]))
    # Optimizer state moves to XLA exactly once (restore already populated it).
    _move_optimizer_state(optimizer, device)

    checkpoint_lr = float(optimizer.param_groups[0]["lr"])
    pack_total_steps = int(config["_pack_total_steps"])
    saved_schedule = payload_for_schedule(prepared).get("lr_schedule")
    if prepared.checkpoint_schema_version == 3 and not config.get("start_new_pack"):
        schedule = PackWsdSchedule.from_dict(saved_schedule or {})
        if schedule.total_steps != pack_total_steps:
            raise ValueError("saved WSD horizon disagrees with the verified pack horizon")
    elif prepared.checkpoint_schema_version == 2 and not config.get("start_new_pack"):
        schedule = CosineSchedule.from_checkpoint(
            payload_for_schedule(prepared), start_step=start_step,
            checkpoint_lr=checkpoint_lr, decay_steps=int(config["lr_decay_steps"]),
            min_lr_ratio=float(config["min_lr_ratio"]),
        )
    else:
        schedule = PackWsdSchedule(
            base_lr=float(config["learning_rate"]),
            total_steps=pack_total_steps,
            warmup_steps=int(pack_total_steps * float(config["warmup_fraction"])),
            min_lr_ratio=float(config["min_lr_ratio"]),
            decay_fraction=float(config["decay_fraction"]),
        )

    sequence_length = int(config["sequence_length"])
    dataset = _load_dataset(Path(str(config["dataset_path"])), tokenizer, sequence_length)
    sampler = ResumableDistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True
    )
    batch_size = int(config["batch_size"])
    batches_per_epoch = sampler.num_samples // batch_size
    if batches_per_epoch <= 0:
        raise RuntimeError("dataset is too small for one distributed training batch")
    # A new-pack continuation from a legacy parent has NO historical cursor;
    # that boundary is established honestly here and persisted exactly onward.
    saved_training_state = prepared.trainer_state or {}
    saved_data = saved_training_state.get("data", {})
    if saved_data:
        position = DataPosition(
            epoch=int(saved_data["epoch"]),
            batch_in_epoch=int(saved_data["batch_in_epoch"]),
            microbatches_consumed=int(saved_data["microbatches_consumed"]),
        )
        canonical_position = DataPosition.from_microbatches(
            position.microbatches_consumed, batches_per_epoch
        )
        if position != canonical_position:
            raise ValueError(
                f"checkpoint data cursor is inconsistent: {position} != {canonical_position}"
            )
    else:
        position = DataPosition(epoch=0, batch_in_epoch=0, microbatches_consumed=0)
    pack_step = prepared.pack_step
    expected_microbatches = pack_step * int(config["grad_accum_steps"])
    if position.microbatches_consumed != expected_microbatches:
        raise ValueError(
            "checkpoint pack step disagrees with its sampler cursor: "
            f"{pack_step} updates vs {position.microbatches_consumed} microbatches"
        )
    current_training_state = build_training_state(
        step=start_step,
        pack_step=pack_step,
        optimizer_updates=optimizer_updates,
        position=position,
        dataset_sha256=str(config["_dataset_sha256"]),
        dataset_windows=len(dataset),
        batch_size=batch_size,
        grad_accum_steps=int(config["grad_accum_steps"]),
        world_size=world_size,
        sequence_length=sequence_length,
        seed=seed,
        attention_chunk_size=int(config["attention_chunk_size"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        gradient_clip_norm=float(config["gradient_clip_norm"]),
        pack_manifest_sha256=str(config["pack_manifest_sha256"]),
        pack_total_steps=pack_total_steps,
    )
    if resume_path and not config.get("start_new_pack"):
        validate_training_state(
            saved_training_state,
            current_training_state,
            allow_legacy=(
                bool(config["allow_legacy_resume"])
                and prepared.checkpoint_schema_version == 1
            ),
        )
    sampler.set_epoch(position.epoch)
    sampler.set_start_index(position.batch_in_epoch * batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    # One execution per microbatch bounds HBM for 2,048-token activations.
    device_loader = pl.MpDeviceLoader(loader, device, batches_per_execution=1)
    iterator = iter(device_loader)
    max_steps = pack_total_steps
    if pack_step >= max_steps:
        raise ValueError("verified pack is already complete; attach a new pack explicitly")
    max_minutes = float(config["max_minutes"])
    deadline = time.monotonic() + max_minutes * 60.0 if max_minutes > 0 else None
    grad_accum = int(config["grad_accum_steps"])
    save_interval = int(config["save_interval"])
    log_interval = int(config["log_interval"])
    output = Path(str(config["output_checkpoint"])).expanduser()
    loss_window = torch.zeros((), device=device)
    loss_window_steps = 0
    latest_loss: torch.Tensor | None = None
    session_started = time.monotonic()
    window_started = session_started
    last_report_step = start_step
    tokens_per_step = tokens_per_optimizer_step(
        batch_size=batch_size,
        grad_accum_steps=grad_accum,
        world_size=world_size,
        sequence_length=sequence_length,
    )
    latest_grad_norm: torch.Tensor | None = None

    if rank == 0:
        print(
            f"[TPU ready] cores={world_size} device={device} precision=bf16 "
            f"resume_step={start_step:,} pack={pack_step:,}/{max_steps:,} "
            f"optimizer_updates={optimizer_updates:,} tokens/step={tokens_per_step:,} "
            f"lr={checkpoint_lr:.3e}",
            flush=True,
        )

    initial_pack_step = pack_step
    for pack_update in range(initial_pack_step, max_steps):
        step = start_step + (pack_update - initial_pack_step)
        control_boundary = pack_update == initial_pack_step or pack_update % log_interval == 0
        if deadline is not None and control_boundary and step > start_step:
            local_stop = int(step > start_step and time.monotonic() >= deadline)
            # Keep control-plane synchronization off the XLA tensor graph.
            # A tensor all_reduce here can crash PJRT before the first model
            # graph is compiled (observed as a null-pointer SIGSEGV on v5e).
            stop = xm.mesh_reduce(
                f"anra_deadline_{pack_update}", local_stop, sum
            )
            if int(stop) > 0:
                break

        effective_lr = schedule.lr_at(
            pack_update if isinstance(schedule, PackWsdSchedule) else step
        )
        for group in optimizer.param_groups:
            group["lr"] = effective_lr
        optimizer.zero_grad(set_to_none=True)
        # Keep loss accumulation on TPU.  Calling ``.cpu()`` for every
        # microbatch synchronizes host and device and substantially reduces
        # throughput on a v5e-8.
        loss_sum = torch.zeros((), device=device)
        for _ in range(grad_accum):
            try:
                x, y = next(iterator)
            except StopIteration:
                # The cursor moves to the next epoch as soon as the last batch
                # is consumed; do not increment it a second time here.
                sampler.set_epoch(position.epoch)
                sampler.set_start_index(0)
                iterator = iter(device_loader)
                x, y = next(iterator)
            with _bf16_autocast(True):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, CANONICAL_CONFIG.vocab_size), y.reshape(-1)
                )
                scaled = loss / grad_accum
            scaled.backward()
            # Execute one reusable forward/backward graph per microbatch.
            # Without this boundary XLA captures all grad-accum microbatches
            # into one enormous first-step graph, making v5e compilation look
            # hung before the first optimizer update. Gradients remain live on
            # device and continue accumulating across mark_step boundaries.
            xm.mark_step()
            loss_sum.add_(loss.detach())
            position = DataPosition.from_microbatches(
                position.microbatches_consumed + 1, batches_per_epoch
            )
        # Average the complete accumulated gradient across all ranks in bounded
        # traceable buckets, then clip that shared gradient and step once.
        _bucketed_traceable_all_reduce(
            model.parameters(), dist=dist, xm=xm, world_size=world_size
        )
        latest_grad_norm = _clip_global_grad_norm(
            model.parameters(), float(config["gradient_clip_norm"])
        )
        optimizer.step()
        xm.mark_step()
        optimizer_updates += 1
        latest_loss = loss_sum / grad_accum
        if not math.isfinite(effective_lr) or effective_lr <= 0:
            raise RuntimeError(
                f"INVALID LEARNING RATE at step {step + 1}: {effective_lr!r}. "
                "Refusing to save; last healthy recovery checkpoint is preserved."
            )
        loss_window.add_(latest_loss)
        loss_window_steps += 1
        completed = step + 1
        pack_completed = pack_update + 1

        report = pack_completed % log_interval == 0 or pack_completed == initial_pack_step + 1
        if report:
            # Logging does not need another device collective. Rank 0's shard
            # loss is an unbiased sample metric; avoiding a redundant legacy
            # all-reduce also keeps metrics outside the crash-prone path.
            mean_loss_tensor = loss_window / max(1, loss_window_steps)
            finite_loss = bool(torch.isfinite(mean_loss_tensor).all().cpu().item())
            finite_grad = bool(torch.isfinite(latest_grad_norm).all().cpu().item())
            if not finite_loss:
                raise RuntimeError(
                    f"NON-FINITE LOSS at step {completed}. Refusing to save; "
                    "last healthy recovery checkpoint is preserved."
                )
            if not finite_grad:
                raise RuntimeError(
                    f"NON-FINITE GRADIENT NORM at step {completed}. Refusing to "
                    "save; last healthy recovery checkpoint is preserved."
                )
        if rank == 0 and report:
            mean_loss = float(mean_loss_tensor.cpu())
            elapsed = max(1e-6, time.monotonic() - window_started)
            report_steps = completed - last_report_step
            tok_per_sec = report_steps * tokens_per_step / elapsed
            print(
                f"step={completed} pack={pack_completed}/{max_steps} loss={mean_loss:.4f} "
                f"lr={effective_lr:.3e} grad_norm={float(latest_grad_norm.cpu()):.3f} "
                f"global_tok/s={tok_per_sec:.1f} elapsed_total="
                f"{(time.monotonic() - session_started) / 60:.1f}m",
                flush=True,
            )
        if report:
            loss_window = torch.zeros((), device=device)
            loss_window_steps = 0
            last_report_step = completed
            window_started = time.monotonic()

        if pack_completed % save_interval == 0:
            training_state = build_training_state(
                step=completed, pack_step=pack_completed,
                optimizer_updates=optimizer_updates, position=position,
                dataset_sha256=str(config["_dataset_sha256"]), dataset_windows=len(dataset),
                batch_size=batch_size, grad_accum_steps=grad_accum, world_size=world_size,
                sequence_length=sequence_length, seed=seed,
                attention_chunk_size=int(config["attention_chunk_size"]),
                gradient_checkpointing=bool(config["gradient_checkpointing"]),
                gradient_clip_norm=float(config["gradient_clip_norm"]),
                pack_manifest_sha256=str(config["pack_manifest_sha256"]),
                pack_total_steps=max_steps,
            )
            _save_latest(
                xm, output, model, optimizer, config, tokenizer, completed,
                {"loss": float(latest_loss.cpu()), "learning_rate": effective_lr,
                 "gradient_norm": float(latest_grad_norm.cpu())},
                source_checkpoint, world_size, training_state, schedule,
            )

        # Sparse immutable candidates (never overwritten): research lineage.
        # All ranks reach this boundary together; only rank 0 writes.
        candidate_interval = int(config.get("candidate_interval") or 0)
        if candidate_interval > 0 and pack_completed % candidate_interval == 0:
            candidate_training_state = build_training_state(
                step=completed, pack_step=pack_completed,
                optimizer_updates=optimizer_updates, position=position,
                dataset_sha256=str(config["_dataset_sha256"]),
                dataset_windows=len(dataset), batch_size=batch_size,
                grad_accum_steps=grad_accum, world_size=world_size,
                sequence_length=sequence_length, seed=seed,
                attention_chunk_size=int(config["attention_chunk_size"]),
                gradient_checkpointing=bool(config["gradient_checkpointing"]),
                gradient_clip_norm=float(config["gradient_clip_norm"]),
                pack_manifest_sha256=str(config["pack_manifest_sha256"]),
                pack_total_steps=max_steps,
            )
            if rank == 0:
                save_candidate(
                    output, model, optimizer, config, tokenizer, completed,
                    {"loss": float(latest_loss.cpu())}, source_checkpoint, world_size,
                    training_state=candidate_training_state,
                    schedule=schedule,
                )
            # Rendezvous so no rank enters the next collective while rank 0
            # is still serializing (PJRT graph-sync safety).
            xm.rendezvous(f"candidate-{completed}")

    final_step = completed if "completed" in locals() else start_step
    final_pack_step = pack_completed if "pack_completed" in locals() else initial_pack_step
    if final_pack_step % save_interval != 0 or not output.is_file():
        training_state = build_training_state(
            step=final_step, pack_step=final_pack_step,
            optimizer_updates=optimizer_updates, position=position,
            dataset_sha256=str(config["_dataset_sha256"]), dataset_windows=len(dataset),
            batch_size=batch_size, grad_accum_steps=grad_accum, world_size=world_size,
            sequence_length=sequence_length, seed=seed,
            attention_chunk_size=int(config["attention_chunk_size"]),
            gradient_checkpointing=bool(config["gradient_checkpointing"]),
            gradient_clip_norm=float(config["gradient_clip_norm"]),
            pack_manifest_sha256=str(config["pack_manifest_sha256"]),
            pack_total_steps=max_steps,
        )
        _save_latest(
            xm, output, model, optimizer, config, tokenizer, final_step,
            {"loss": float(latest_loss.cpu()) if latest_loss is not None else 0.0,
             "learning_rate": schedule.lr_at(
                 min(final_pack_step, max_steps - 1)
                 if isinstance(schedule, PackWsdSchedule) else final_step
             )},
            source_checkpoint, world_size, training_state, schedule,
        )
    if rank == 0:
        print("[TPU complete] one protected checkpoint is ready", flush=True)


def run(args: argparse.Namespace) -> None:
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be zero (full pack) or positive")
    if args.max_minutes < 0:
        raise ValueError("--max-minutes must be zero (step-only) or positive")
    if args.batch_size <= 0 or args.grad_accum_steps <= 0:
        raise ValueError("--batch-size and --grad-accum-steps must be positive")
    if args.sequence_length <= 0 or args.sequence_length > CANONICAL_CONFIG.block_size:
        raise ValueError(
            f"--sequence-length must be in [1, {CANONICAL_CONFIG.block_size}]"
        )
    if CANONICAL_CONFIG.block_size % args.sequence_length:
        raise ValueError("--sequence-length must divide the canonical context length")
    if args.attention_chunk_size <= 0 or args.require_world_size <= 0:
        raise ValueError("attention chunk size and required world size must be positive")
    if args.gradient_clip_norm <= 0:
        raise ValueError("--gradient-clip-norm must be positive")
    if args.lr_decay_steps <= 0 or not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("invalid cosine learning-rate schedule")
    if not 0 <= args.warmup_fraction < 1 or not 0 < args.decay_fraction <= 1:
        raise ValueError("invalid WSD warmup/decay fractions")
    if args.save_interval <= 0 or args.log_interval <= 0:
        raise ValueError("--save-interval and --log-interval must be positive")

    dataset_path = Path(args.dataset_path).expanduser()
    resume_from = (
        os.environ.get("ANRA_TPU_CHECKPOINT") or args.resume_from
        if (os.environ.get("ANRA_TPU_CHECKPOINT") or args.resume_from)
        else None
    )
    checkpoint_path = Path(resume_from).expanduser() if resume_from else None

    # CPU preflight BEFORE spawning workers: pack semantics + exact parent.
    identity_block = preflight(
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        block_size=CANONICAL_CONFIG.block_size,
        vocab_size=CANONICAL_CONFIG.vocab_size,
        expected_resume_step=args.expected_resume_step,
        start_new_pack=args.start_new_pack,
        allow_legacy_resume=args.allow_legacy_resume,
    )
    if identity_block["parent_checkpoint_schema"] in {1, 9} and not args.allow_legacy_resume:
        raise ValueError(
            "legacy step-20k migration requires --allow-legacy-resume once"
        )

    config = vars(args).copy()
    config["pack_manifest_sha256"] = identity_block["pack_manifest_sha256"]
    config["_dataset_sha256"] = dataset_fingerprint(Path(args.dataset_path))
    available_pack_steps = identity_block["pack_windows"] // (
        args.batch_size * args.grad_accum_steps * args.require_world_size
    )
    if available_pack_steps <= 0:
        raise ValueError("verified pack cannot form one complete distributed optimizer step")
    if args.max_steps > available_pack_steps:
        raise ValueError(
            f"--max-steps={args.max_steps:,} exceeds the pack's "
            f"{available_pack_steps:,} unique-data updates"
        )
    config["_pack_total_steps"] = args.max_steps or available_pack_steps
    receipt = write_run_receipt(
        Path(args.output_checkpoint).expanduser().parent,
        identity_block=identity_block, config=config,
        world_size=args.require_world_size,
    )
    print(f"[preflight OK] receipt: {receipt}", flush=True)

    # ``torch_xla.launch`` is the current PJRT entrypoint: it launches exactly
    # the TPU workers granted by Kaggle instead of assuming a fixed process
    # topology.  Keep the legacy launcher only for older Kaggle XLA images.
    try:
        import torch_xla

        launch = getattr(torch_xla, "launch", None)
    except ImportError:  # pragma: no cover - _require_xla already explains this
        launch = None
    if callable(launch):
        launch(_worker, args=(config,), start_method="spawn")
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp

        xmp.spawn(_worker, args=(config,), start_method="spawn")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="An-Ra V4 TPU v5e-8 trainer")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--resume-from",
                        help="EXACT parent checkpoint path (never auto-selected)")
    parser.add_argument("--expected-resume-step", type=int, default=20_000)
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help="establish a schema-v3 pack/WSD boundary from a legacy full-resume checkpoint",
    )
    parser.add_argument("--candidate-interval", type=int, default=1000,
                        help="sparse immutable candidate checkpoints (0 disables)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="pack-local update cap; 0 consumes every complete unique window once")
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=CANONICAL_CONFIG.block_size)
    parser.add_argument("--attention-chunk-size", type=int, default=128)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--require-world-size", type=int, default=8)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--lr-decay-steps", type=int, default=1_000_000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-fraction", type=float, default=0.0)
    parser.add_argument("--decay-fraction", type=float, default=0.1)
    parser.add_argument("--start-new-pack", action="store_true",
                        help="reset pack-local state for an explicitly different pack")
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1301)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
