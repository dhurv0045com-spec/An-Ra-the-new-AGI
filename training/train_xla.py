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

from anra_core.checkpoint import load_core_checkpoint
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
    validate_full_resume,
    validate_training_state,
)


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
            mode="new_pack_parent",
        )
        identity_block.update(
            {
                "parent_checkpoint": str(checkpoint_path),
                "parent_global_step": restored.global_step,
                "parent_parameter_sha256": restored.checkpoint_parameter_sha256,
                "parent_optimizer_restored": restored.optimizer_restored,
                "parent_mode": restored.mode,
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
    import hashlib
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
        "schedule": "wsd_2pct_warmup_stable_linear_decay_to_10pct",
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


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    tokenizer: V4Tokenizer,
    step: int,
    metrics: dict[str, float],
    source_checkpoint: str | None,
    world_size: int,
    training_state: dict[str, object] | None = None,
    schedule=None,
    *,
    artifact_class: str = "full_resume",
) -> dict[str, object]:
    raw_model = model.module if hasattr(model, "module") else model
    state = {name: value.detach().cpu() for name, value in raw_model.state_dict().items()}
    state["lm_head.weight"] = state["token_embedding_table.weight"]
    return {
        "checkpoint_artifact_class": artifact_class,
        "checkpoint_schema_version": 2,
        "global_step": int(step),
        "pack_manifest_sha256": config.get("pack_manifest_sha256"),
        "training_stage": "pretraining_tpu_xla",
        "source_commit": os.environ.get("ANRA_SOURCE_COMMIT", "unknown"),
        "source_checkpoint": source_checkpoint,
        "model_config": asdict(config),
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "trainer_state": training_state,
        "lr_schedule": schedule.to_dict(),
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
    schedule: CosineSchedule,
) -> None:
    """Write exactly one checkpoint object; never create numbered copies."""
    if not xm.is_master_ordinal():
        xm.rendezvous("checkpoint-written")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.uploading")
    payload = _checkpoint_payload(
        model, optimizer, config, tokenizer, step, metrics, source_checkpoint, world_size,
        training_state, schedule,
    )
    xm.save(payload, str(temporary), master_only=True)
    os.replace(temporary, path)
    print(f"[TPU checkpoint] step={step:,} path={path}", flush=True)
    xm.rendezvous("checkpoint-written")


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

    seed = int(config["seed"])
    # All replicas must construct identical fresh weights. Rank-specific RNG is
    # established only after model construction/loading.
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = V4Tokenizer.load_canonical()
    resume_path = Path(str(config["resume_from"])).expanduser() if config.get("resume_from") else None
    source_checkpoint: str | None = str(resume_path) if resume_path else None

    if resume_path:
        # Canonical restore: install the checkpoint INTO this worker's model
        # and verify per tensor (P0 fix). The parent becomes a schema-v2
        # continuation boundary - honestly, without pretending a historical
        # data cursor exists. Their guards are kept: full-resume validation,
        # minimum step, and world-size drift refusal.
        from training.resume import restore_training_state

        restored = restore_training_state(
            str(resume_path), model, optimizer, mode="new_pack_parent",
        )
        start_step = validate_full_resume(
            {"global_step": restored.global_step},
            minimum_step=int(config["expected_resume_step"]),
        )
        payload = {}
        if rank == 0:
            print(
                f"[TPU resume] mode={restored.mode} step={start_step:,} "
                f"parameter_sha256={restored.checkpoint_parameter_sha256[:16]} "
                f"optimizer={'restored' if restored.optimizer_restored else 'FRESH'} "
                f"-> schema-v2 continuation boundary established",
                flush=True,
            )
    else:
        model, payload = AnRaCore(CANONICAL_CONFIG), {}
        start_step = 0

    random.seed(seed + rank + start_step)
    torch.manual_seed(seed + rank + start_step)

    model = model.to(device)
    model.train()
    model.enable_gradient_checkpointing(bool(config["gradient_checkpointing"]))
    model.enable_memory_efficient_attention(int(config["attention_chunk_size"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(config["weight_decay"]),
    )
    if resume_path:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
    saved_optimizer_updates = payload.get("trainer_state", {}).get("optimizer_updates")
    optimizer_updates = (
        int(saved_optimizer_updates)
        if saved_optimizer_updates is not None
        else _optimizer_update_count(payload.get("optimizer_state_dict", {}))
    )
    checkpoint_lr = float(optimizer.param_groups[0]["lr"])
    schedule = CosineSchedule.from_checkpoint(
        payload,
        start_step=start_step,
        checkpoint_lr=checkpoint_lr,
        decay_steps=int(config["lr_decay_steps"]),
        min_lr_ratio=float(config["min_lr_ratio"]),
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
    saved_training_state = payload.get("trainer_state", {}) if resume_path else {}
    saved_data = saved_training_state.get("data", {}) if saved_training_state else {}
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
    elif resume_path:
        # V1 checkpoints used epoch=global_step and batch zero on every resume.
        # Reproduce that boundary once, then persist an exact v2 cursor.
        position = DataPosition(
            epoch=start_step,
            batch_in_epoch=0,
            microbatches_consumed=start_step * batches_per_epoch,
        )
    else:
        position = DataPosition(epoch=0, batch_in_epoch=0, microbatches_consumed=0)
    current_training_state = build_training_state(
        step=start_step,
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
    )
    if resume_path:
        validate_training_state(
            saved_training_state,
            current_training_state,
            allow_legacy=(
                bool(config["allow_legacy_resume"])
                and payload.get("checkpoint_schema_version") == 1
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
    max_steps = int(config["max_steps"])
    if max_steps <= start_step:
        raise ValueError(
            f"--max-steps is an absolute target and must exceed resume step {start_step:,}"
        )
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
            f"resume_step={start_step:,} target_step={max_steps:,} "
            f"optimizer_updates={optimizer_updates:,} tokens/step={tokens_per_step:,} "
            f"lr={checkpoint_lr:.3e}",
            flush=True,
        )

    for step in range(start_step, max_steps):
        control_boundary = step == start_step or step % log_interval == 0
        if deadline is not None and control_boundary:
            local_stop = int(step > start_step and time.monotonic() >= deadline)
            stop = xm.all_reduce(xm.REDUCE_SUM, torch.tensor([local_stop], device=device))
            if int(stop.cpu().item()) > 0:
                break

        effective_lr = schedule.lr_at(step)
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
                loss = F.cross_entropy(logits.reshape(-1, CANONICAL_CONFIG.vocab_size), y.reshape(-1))
                scaled = loss / grad_accum
            scaled.backward()
            loss_sum.add_(loss.detach())
            position = DataPosition.from_microbatches(
                position.microbatches_consumed + 1, batches_per_epoch
            )
        # Clip the averaged global gradient, not a different local gradient on
        # every rank. Do not call xm.optimizer_step afterwards (it would reduce twice).
        xm.reduce_gradients(optimizer)
        latest_grad_norm = _clip_global_grad_norm(
            model.parameters(), float(config["gradient_clip_norm"])
        )
        optimizer.step()
        xm.mark_step()
        optimizer_updates += 1
        latest_loss = loss_sum / grad_accum
        # NaN/Inf guard (fail closed): never checkpoint a corrupted run.
        if not bool(torch.isfinite(latest_loss).all()):
            raise RuntimeError(
                f"NON-FINITE LOSS at step {step + 1}: {float(latest_loss.cpu())}. "
                "Refusing to save; last healthy recovery checkpoint is preserved."
            )
        loss_window.add_(latest_loss)
        loss_window_steps += 1
        completed = step + 1

        report = completed % log_interval == 0 or completed == start_step + 1
        if report:
            global_loss = xm.all_reduce(xm.REDUCE_SUM, loss_window) / world_size
        if rank == 0 and report:
            mean_loss = float(global_loss.cpu()) / max(1, loss_window_steps)
            elapsed = max(1e-6, time.monotonic() - window_started)
            report_steps = completed - last_report_step
            tok_per_sec = report_steps * tokens_per_step / elapsed
            print(
                f"step={completed} loss={mean_loss:.4f} "
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

        if completed % save_interval == 0:
            training_state = build_training_state(
                step=completed, optimizer_updates=optimizer_updates, position=position,
                dataset_sha256=str(config["_dataset_sha256"]), dataset_windows=len(dataset),
                batch_size=batch_size, grad_accum_steps=grad_accum, world_size=world_size,
                sequence_length=sequence_length, seed=seed,
                attention_chunk_size=int(config["attention_chunk_size"]),
                gradient_checkpointing=bool(config["gradient_checkpointing"]),
                gradient_clip_norm=float(config["gradient_clip_norm"]),
            )
            _save_latest(
                xm, output, model, optimizer, CANONICAL_CONFIG, tokenizer, completed,
                {"loss": float(latest_loss.cpu()), "learning_rate": effective_lr,
                 "gradient_norm": float(latest_grad_norm.cpu())},
                source_checkpoint, world_size, training_state, schedule,
            )

        # Sparse immutable candidates (never overwritten): research lineage.
        candidate_interval = int(config.get("candidate_interval") or 0)
        if rank == 0 and candidate_interval > 0 and completed % candidate_interval == 0:
            raw_model = model.module if hasattr(model, "module") else model
            candidate_payload = _checkpoint_payload(
                raw_model, optimizer, config, tokenizer, completed,
                {"loss": float(latest_loss.cpu())}, source_checkpoint, world_size,
                artifact_class="candidate_model_only",
            )
            candidate_payload.pop("optimizer_state_dict", None)  # small lineage files
            candidate_path = (
                output.parent / "candidates" / f"anra-v4-step-{completed:05d}.pt"
            )
            if not candidate_path.exists():
                _atomic_save(candidate_payload, candidate_path)
                print(f"[candidate] step={completed:,} -> {candidate_path}", flush=True)

    final_step = completed if "completed" in locals() else start_step
    if final_step % save_interval != 0 or not output.is_file():
        training_state = build_training_state(
            step=final_step, optimizer_updates=optimizer_updates, position=position,
            dataset_sha256=str(config["_dataset_sha256"]), dataset_windows=len(dataset),
            batch_size=batch_size, grad_accum_steps=grad_accum, world_size=world_size,
            sequence_length=sequence_length, seed=seed,
            attention_chunk_size=int(config["attention_chunk_size"]),
            gradient_checkpointing=bool(config["gradient_checkpointing"]),
            gradient_clip_norm=float(config["gradient_clip_norm"]),
        )
        _save_latest(
            xm, output, model, optimizer, CANONICAL_CONFIG, tokenizer, final_step,
            {"loss": float(latest_loss.cpu()) if latest_loss is not None else 0.0,
             "learning_rate": schedule.lr_at(final_step)},
            source_checkpoint, world_size, training_state, schedule,
        )
    if rank == 0:
        print("[TPU complete] one protected checkpoint is ready", flush=True)


def run(args: argparse.Namespace) -> None:
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
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
    )

    config = vars(args).copy()
    config["pack_manifest_sha256"] = identity_block["pack_manifest_sha256"]
    config["_dataset_sha256"] = dataset_fingerprint(Path(args.dataset_path))
    receipt = write_run_receipt(
        output_dir := Path(args.output_checkpoint).expanduser().parent,
        identity_block=identity_block, config=config,
        world_size=int(os.environ.get("EXPECTED_TPU_WORKERS", "8")),
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
        help="establish a v2 data/scheduler boundary from a schema-v1 full-resume checkpoint",
    )
    parser.add_argument("--candidate-interval", type=int, default=1000,
                        help="sparse immutable candidate checkpoints (0 disables)")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
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
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1301)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
