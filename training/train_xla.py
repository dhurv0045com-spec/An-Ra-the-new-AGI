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

from anra_core.checkpoint import load_core_checkpoint
from anra_core.config import CANONICAL_CONFIG
from anra_core.model import AnRaCore
from anra_core.tokenizer import V4Tokenizer


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


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    tokenizer: V4Tokenizer,
    step: int,
    metrics: dict[str, float],
    source_checkpoint: str | None,
    world_size: int,
) -> dict[str, object]:
    raw_model = model.module if hasattr(model, "module") else model
    state = {name: value.detach().cpu() for name, value in raw_model.state_dict().items()}
    state["lm_head.weight"] = state["token_embedding_table.weight"]
    return {
        "checkpoint_artifact_class": "full_resume",
        "checkpoint_schema_version": 1,
        "global_step": int(step),
        "training_stage": "pretraining_tpu_xla",
        "source_commit": os.environ.get("ANRA_SOURCE_COMMIT", "unknown"),
        "source_checkpoint": source_checkpoint,
        "model_config": asdict(config),
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "tokenizer_contract": {"available": True, **tokenizer.identity(probe_count=500)},
        "metrics": metrics,
        "execution": {
            "backend": "torch_xla",
            "precision": "bf16",
            "world_size": int(world_size),
        },
    }


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
) -> None:
    """Write exactly one checkpoint object; never create numbered copies."""
    if not xm.is_master_ordinal():
        if world_size > 1:
            xm.rendezvous("checkpoint-written")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.uploading")
    payload = _checkpoint_payload(
        model, optimizer, config, tokenizer, step, metrics, source_checkpoint, world_size
    )
    xm.save(payload, str(temporary), master_only=True)
    os.replace(temporary, path)
    print(f"[TPU checkpoint] step={step:,} path={path}", flush=True)
    if world_size > 1:
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

    # Some Kaggle torch_xla wheels expose the XLA package but do not register
    # it as ``torch.xla``.  PyTorch's non-reentrant checkpoint helper needs
    # that registration to identify the execution device; register the
    # already-loaded runtime rather than falling back to CPU or disabling
    # rematerialization.
    if not hasattr(torch, "xla") and hasattr(torch, "_register_device_module"):
        torch._register_device_module("xla", torch_xla)

    device = xm.xla_device()
    # ``xm.get_ordinal`` was removed from current Kaggle torch_xla; runtime
    # owns the PJRT ordinal now.  Keep a zero-rank fallback for the supported
    # single-worker topology.
    rank = int(getattr(xr, "global_ordinal", lambda: 0)())
    # Modern PJRT/XLA exposes the worker topology through runtime.world_size;
    # xrt_world_size was removed from current Kaggle torch_xla builds.
    world_size_hint = config.get("_world_size_hint")
    world_size = int(world_size_hint) if world_size_hint is not None else int(xr.world_size())
    # Kaggle can expose a TPU device while the PJRT slice is provisioned as a
    # single worker (for example while a v5e-8 session is still attaching).
    # In that topology, spawning eight processes is invalid: PJRT reports one
    # worker address and aborts every child.  Run one honest worker instead;
    # the checkpoint records world_size=1 so this cannot be mistaken for an
    # eight-core run.  Once Kaggle grants the full slice, world_size is 8 and
    # the same code uses all workers.
    if world_size < 1:
        raise RuntimeError(f"Invalid TPU runtime world_size={world_size}")

    seed = int(config["seed"])
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    tokenizer = V4Tokenizer.load_canonical()
    resume_path = Path(str(config["resume_from"])).expanduser() if config.get("resume_from") else None
    source_checkpoint: str | None = str(resume_path) if resume_path else None

    if resume_path:
        model, payload, identity = load_core_checkpoint(resume_path)
        if rank == 0:
            print(
                f"[TPU resume] step={identity.global_step or 0:,} "
                f"parameter_sha256={identity.parameter_sha256}",
                flush=True,
            )
    else:
        model, payload = AnRaCore(CANONICAL_CONFIG), {}

    model = model.to(device)
    model.train()
    model.enable_gradient_checkpointing(True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(config["weight_decay"]),
    )
    if isinstance(payload, dict) and isinstance(payload.get("optimizer_state_dict"), dict):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
    start_step = int(payload.get("global_step", 0)) if isinstance(payload, dict) else 0

    dataset = _load_dataset(Path(str(config["dataset_path"])), tokenizer, CANONICAL_CONFIG.block_size)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    # One XLA execution contains a complete accumulated update.  The previous
    # default of one batch per execution materialized every microbatch and
    # left most of gradient accumulation's TPU fusion benefit on the table.
    device_loader = pl.MpDeviceLoader(
        loader, device, batches_per_execution=int(config["grad_accum_steps"])
    )
    sampler_epoch = start_step
    sampler.set_epoch(sampler_epoch)
    iterator = iter(device_loader)
    max_steps = int(config["max_steps"])
    max_minutes = float(config["max_minutes"])
    deadline = time.monotonic() + max_minutes * 60.0 if max_minutes > 0 else None
    grad_accum = int(config["grad_accum_steps"])
    save_interval = int(config["save_interval"])
    log_interval = int(config["log_interval"])
    output = Path(str(config["output_checkpoint"])).expanduser()
    loss_window = torch.zeros((), device=device)
    loss_window_steps = 0
    latest_loss: torch.Tensor | None = None
    window_started = time.monotonic()

    if rank == 0:
        print(
            f"[TPU ready] cores={world_size} device={device} precision=bf16 "
            f"resume_step={start_step:,} target_step={max_steps:,}",
            flush=True,
        )

    for step in range(start_step, max_steps):
        if deadline is not None and step > start_step and time.monotonic() >= deadline:
            stop = torch.tensor([1], device=device)
        else:
            stop = torch.tensor([0], device=device)
        stop = xm.all_reduce(xm.REDUCE_MAX, stop)
        if int(stop.item()):
            break

        optimizer.zero_grad(set_to_none=True)
        # Keep loss accumulation on TPU.  Calling ``.cpu()`` for every
        # microbatch synchronizes host and device and substantially reduces
        # throughput on a v5e-8.
        loss_sum = torch.zeros((), device=device)
        for _ in range(grad_accum):
            try:
                x, y = next(iterator)
            except StopIteration:
                sampler_epoch += 1
                sampler.set_epoch(sampler_epoch)
                iterator = iter(device_loader)
                x, y = next(iterator)
            with _bf16_autocast(True):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, CANONICAL_CONFIG.vocab_size), y.reshape(-1))
                scaled = loss / grad_accum
            scaled.backward()
            loss_sum.add_(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        xm.optimizer_step(optimizer, barrier=world_size > 1)
        latest_loss = loss_sum / grad_accum
        loss_window.add_(latest_loss)
        loss_window_steps += 1
        completed = step + 1

        if rank == 0 and (completed % log_interval == 0 or completed == start_step + 1):
            # This is the intentionally infrequent host sync for reporting.
            mean_loss = float(loss_window.cpu()) / max(1, loss_window_steps)
            elapsed = max(1e-6, time.monotonic() - window_started)
            tokens = completed - start_step
            tok_per_sec = tokens * grad_accum * int(config["batch_size"]) * world_size * CANONICAL_CONFIG.block_size / elapsed
            print(
                f"step={completed} loss={mean_loss:.4f} "
                f"tok/s={tok_per_sec:.1f} elapsed={elapsed / 60:.1f}m",
                flush=True,
            )
            loss_window = torch.zeros((), device=device)
            loss_window_steps = 0

        if completed % save_interval == 0:
            _save_latest(
                xm, output, model, optimizer, CANONICAL_CONFIG, tokenizer, completed,
                {"loss": float(latest_loss.cpu())}, source_checkpoint, world_size,
            )

    _save_latest(
        xm, output, model, optimizer, CANONICAL_CONFIG, tokenizer, completed if "completed" in locals() else start_step,
        {"loss": float(latest_loss.cpu()) if latest_loss is not None else 0.0},
        source_checkpoint, world_size,
    )
    if rank == 0:
        print("[TPU complete] one protected checkpoint is ready", flush=True)


def run(args: argparse.Namespace) -> None:
    _, _, xmp = _require_xla()
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.max_minutes < 0:
        raise ValueError("--max-minutes must be zero (step-only) or positive")
    if args.batch_size <= 0 or args.grad_accum_steps <= 0:
        raise ValueError("--batch-size and --grad-accum-steps must be positive")
    if args.save_interval <= 0 or args.log_interval <= 0:
        raise ValueError("--save-interval and --log-interval must be positive")
    config = vars(args).copy()
    # Inspect the already-provisioned PJRT topology before choosing a launch
    # mode.  A one-worker Kaggle TPU must not be handed to the eight-process
    # launcher; doing so produces the misleading "expected 8 worker
    # addresses, got 1" crash.  Direct execution reuses the existing client
    # safely and remains resumable, while a real v5e-8 slice uses spawn below.
    try:
        import torch_xla.runtime as xr

        parent_world_size = int(xr.world_size())
    except Exception as exc:
        # The notebook preflight may already have initialized PJRT.  Some
        # Kaggle images then refuse a second topology query with a transient
        # device-busy error even though the existing client is usable.  The
        # preflight's world-size=1 topology is the only safe fallback; record
        # it explicitly and keep the run single-process rather than spawning
        # a mismatched eight-worker slice.
        if os.environ.get("ANRA_TPU_PREFLIGHT_WORLD_SIZE"):
            parent_world_size = int(os.environ["ANRA_TPU_PREFLIGHT_WORLD_SIZE"])
        else:
            parent_world_size = 1
            print(
                f"[TPU topology] PJRT query unavailable after preflight ({exc}); "
                "using one existing worker",
                flush=True,
            )
    config["_world_size_hint"] = parent_world_size
    if parent_world_size == 1:
        _worker(0, config)
        return
    # ``torch_xla.launch`` is the current PJRT entrypoint: it launches exactly
    # the TPU workers granted by Kaggle instead of assuming a fixed process
    # topology.  Keep the legacy launcher only for older Kaggle XLA images.
    try:
        import torch_xla

        launch = getattr(torch_xla, "launch", None)
    except ImportError:  # pragma: no cover - _require_xla already explains this
        launch = None
    if callable(launch):
        # A Kaggle notebook may have imported torch_xla during preflight.
        # Spawn fresh workers so the parent PJRT client is never inherited;
        # fork would make each child attempt a second client initialization.
        launch(_worker, args=(config,), start_method="spawn")
    else:
        xmp.spawn(_worker, args=(config,), start_method="spawn")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="An-Ra V4 TPU v5e-8 trainer")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1301)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
