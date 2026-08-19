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

    def __init__(self, root: Path, block_size: int, *, source_block_size: int = 2_048) -> None:
        files = sorted(root.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"no token shards (*.npy) found in {root}")
        self.block_size = int(block_size)
        self.source_block_size = int(source_block_size)
        if self.block_size <= 0 or self.source_block_size <= 0:
            raise ValueError("training and source block sizes must be positive")
        if self.block_size > self.source_block_size or self.source_block_size % self.block_size:
            raise ValueError(
                "training sequence length must divide the canonical source window "
                f"({self.source_block_size}); got {self.block_size}"
            )
        self._segments_per_source_window = self.source_block_size // self.block_size
        self._arrays: list[np.ndarray] = []
        self._ends: list[int] = []
        total = 0
        for path in files:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"token shard must be a 1-D integer array: {path}")
            windows = (int(array.shape[0]) - 1) // self.source_block_size
            if windows <= 0:
                continue
            self._arrays.append(array)
            total += windows * self._segments_per_source_window
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
        source_window, segment = divmod(local_index, self._segments_per_source_window)
        start = (
            source_window * self.source_block_size
            + segment * self.block_size
        )
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


def _runtime_topology(runtime: Any) -> tuple[int, int]:
    """Return ``(process_world_size, device_count)`` for the PJRT runtime.

    On Kaggle TPU VMs the notebook process commonly reports
    ``world_size()==1`` before the multiprocessing launcher starts, while
    ``global_device_count()==8`` already proves that the complete v5e-8 slice
    was granted.  Treating the pre-launch process count as the slice size
    incorrectly rejects (or serializes) a healthy eight-core allocation.
    """
    try:
        process_world_size = int(runtime.world_size())
    except Exception:
        process_world_size = 1
    device_count = process_world_size
    for name in ("global_device_count", "global_runtime_device_count"):
        getter = getattr(runtime, name, None)
        if getter is None:
            continue
        try:
            device_count = max(device_count, int(getter()))
        except Exception:
            continue
    if process_world_size < 1 or device_count < 1:
        raise RuntimeError(
            f"Invalid PJRT topology: process_world_size={process_world_size}, "
            f"device_count={device_count}"
        )
    return process_world_size, device_count


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
        dataset = TokenShardDataset(
            train_root,
            block_size,
            source_block_size=CANONICAL_CONFIG.block_size,
        )
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
    # Modern PJRT/XLA exposes the child worker topology through
    # runtime.world_size. The parent hint is only an expected value; never
    # use it to mask a partial or misconfigured child launch.
    world_size_hint = config.get("_world_size_hint")
    runtime_world_size = int(xr.world_size())
    if world_size_hint is not None and runtime_world_size != int(world_size_hint):
        raise RuntimeError(
            f"TPU worker topology mismatch: expected {int(world_size_hint)} workers, "
            f"runtime reported {runtime_world_size}; refusing partial launch"
        )
    world_size = runtime_world_size
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
    model.enable_memory_efficient_attention(int(config["attention_chunk_size"]))
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

    sequence_length = int(config["sequence_length"])
    dataset = _load_dataset(Path(str(config["dataset_path"])), tokenizer, sequence_length)
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
            tok_per_sec = tokens * grad_accum * int(config["batch_size"]) * world_size * sequence_length / elapsed
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
    # Kaggle may leave a legacy one-address override in the environment.
    # PJRT interprets it as a one-worker topology and aborts when the full
    # v5e-8 launch creates eight local workers. The TPU runtime supplies the
    # correct local topology without this override.
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    # Keep the parent a pure launcher.  In particular, do not import
    # ``torch_xla.core.xla_model`` or the parallel loader here: Kaggle's PJRT
    # runtime can initialize its metrics client as a side effect of that
    # import.  Importing the full worker stack in the parent and then calling
    # ``torch_xla.launch`` creates a second PJRT client in the child process
    # and aborts with ``runtime_metric_aggregator``/``reporting_closure``.
    # Workers load the XLA worker stack after PJRT assigns their topology.
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
        raise ValueError(
            "--sequence-length must evenly divide the canonical 2048-token source window"
        )
    if args.attention_chunk_size <= 0:
        raise ValueError("--attention-chunk-size must be positive")
    if args.save_interval <= 0 or args.log_interval <= 0:
        raise ValueError("--save-interval and --log-interval must be positive")
    config = vars(args).copy()
    # The worker is the only authoritative topology observer.  The parent
    # records the operator's requested slice (when provided); each worker
    # compares it with xr.world_size() after launch.  This avoids both stale
    # preflight values and a second PJRT initialization in the parent.
    config["_world_size_hint"] = (
        int(args.require_world_size) if args.require_world_size is not None else None
    )
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
        # Older XLA images expose only xmp.spawn.  Import it lazily here so
        # the parent still never imports xla_model/parallel_loader before the
        # launcher has established the worker topology.
        import torch_xla.distributed.xla_multiprocessing as xmp

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
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=CANONICAL_CONFIG.block_size,
        help="training window length; model context remains the canonical 2048 tokens",
    )
    parser.add_argument(
        "--attention-chunk-size",
        type=int,
        default=128,
        help="query tile for exact bounded-workspace attention on TPU/XLA",
    )
    parser.add_argument(
        "--require-world-size",
        type=int,
        help="fail closed unless Kaggle grants exactly this TPU worker count",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1301)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
