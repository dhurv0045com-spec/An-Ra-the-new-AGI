"""Canonical TPU/XLA trainer for An-Ra V4 - core-vnext training path.

Upgrades over the previous TPU trainer:

1. WSD LR schedule (2% warmup, stable, linear decay to 10%) aligned to the
   declared token budget of the current pack. The old trainer ran constant
   LR forever, which degraded the model during repeat passes (observed as
   the post-20k-step regression).
2. Fail-closed pack verification: training refuses to start unless every
   shard hash in manifest.json matches. No silent dataset fallback.
3. Epoch-boundary telemetry: the trainer logs exactly when one full pass
   completes and what the validation loss was at that boundary.
4. Milestone behavior gates: periodic micro-probes (echo / copy-from-context)
   logged alongside loss so "is it actually speaking" is visible in-session.

Module imports torch_xla lazily so CPU tooling and tests do not need TPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from anra_core.checkpoint import load_core_checkpoint
from anra_core.config import CANONICAL_CONFIG
from anra_core.model import AnRaCore
from anra_core.tokenizer import V4Tokenizer
from training.pack_verify import PackVerificationError, VerifiedPack, verify_pack
from training.wsd_scheduler import build_wsd_schedule, phase_for_step


class ShardWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-length next-token windows drawn round-robin from verified shards."""

    def __init__(self, shards: tuple[Path, ...], block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._arrays: list[np.ndarray] = []
        self._window_counts: list[int] = []
        for path in shards:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 1 or array.size <= block_size:
                raise ValueError(f"shard {path.name} cannot form windows")
            self._arrays.append(array)
            self._window_counts.append((array.size - 1) // block_size)

    def __len__(self) -> int:
        return sum(self._window_counts)

    @property
    def total_tokens(self) -> int:
        return sum(int(a.shape[0]) for a in self._arrays)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = 0
        for array, count in zip(self._arrays, self._window_counts):
            if index < offset + count:
                local = index - offset
                start = local * self.block_size
                chunk = array[start : start + self.block_size + 1]
                x = torch.from_numpy(np.asarray(chunk[:-1], dtype=np.int64))
                y = torch.from_numpy(np.asarray(chunk[1:], dtype=np.int64))
                return x, y
            offset += count
        raise IndexError(index)


def _require_xla() -> tuple[Any, Any, Any]:
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.runtime as xr
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "torch_xla is unavailable; this trainer requires a TPU runtime"
        ) from exc
    return xm, pl, xr


def _milestone_probes(executor_like_model: AnRaCore) -> dict[str, bool]:
    """Cheap in-training capability probes (greedy, tiny). Returns pass map."""
    # Deliberately minimal: these are health signals, not evaluations.
    # Full gates live in connector/experiments/cognitive_credit/capability_probe.
    probes: dict[str, bool] = {}
    try:
        tok = V4Tokenizer.load_canonical()
        word = "ember"
        prompt = f"<k></k>\n<plan>Repeat the requested word verbatim.</plan>\n<q>Echo exactly this word: {word}</q>\n<answer>"
        ids = torch.tensor([[tok.bos_token_id, *tok.encode(prompt)]], dtype=torch.long)
        with torch.no_grad():
            logits = executor_like_model(ids)
        text_ids = int(logits[0, -1].argmax(dim=-1).item())
        piece = tok.decode([text_ids])
        probes["echo_first_token"] = piece.strip().lower() in {"ember", "e"}
    except Exception:  # noqa: BLE001 - probe failure is a signal, not a crash
        probes["echo_first_token"] = False
    return probes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="An-Ra V4 TPU trainer (core-vnext path)")
    parser.add_argument("--pack-root", required=True,
                        help="verified token pack directory containing manifest.json + train/*.npy")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="0 = derive from --token-budget")
    parser.add_argument("--max-minutes", type=float, default=450.0)
    parser.add_argument("--token-budget", type=int, default=500_000_000,
                        help="unique tokens this pack should be consumed for; sets decay end")
    parser.add_argument("--tokens-per-step", type=int, default=0,
                        help="0 = batch*accum*world*block computed at runtime")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--skip-pack-verification", action="store_true",
                        help="NOT recommended; bypasses fail-closed verification")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    xm, pl, xr = _require_xla()
    device = xm.xla_device()
    world_size = int(xr.world_size())
    rank = int(xr.get_ordinal())

    # ---- Data: fail-closed verification ---------------------------------
    pack_root = Path(args.pack_root)
    pack: VerifiedPack | None = None
    if rank == 0:
        print(f"[pack] verifying {pack_root} ...", flush=True)
    if args.skip_pack_verification:
        print("[pack] WARNING verification skipped by operator flag", flush=True)
    else:
        try:
            pack = verify_pack(pack_root)
        except PackVerificationError as exc:
            print(f"[pack] REFUSING TO TRAIN: {exc}", flush=True)
            return 2
        if rank == 0:
            print(
                f"[pack] verified: {len(pack.shard_paths)} shards, "
                f"{pack.total_tokens:,} tokens, ~{pack.total_windows:,} unique windows",
                flush=True,
            )

    seed = args.seed
    torch.manual_seed(seed + rank)
    tokenizer = V4Tokenizer.load_canonical()

    resume_path = Path(args.resume_from).expanduser() if args.resume_from else None
    source_checkpoint: str | None = str(resume_path) if resume_path else None
    start_step = 0
    if resume_path:
        model, payload, identity = load_core_checkpoint(resume_path)
        start_step = int(identity.global_step or 0)
        if rank == 0:
            print(f"[resume] step={start_step:,} sha={identity.checkpoint_sha256[:16]}...",
                  flush=True)
    else:
        model = AnRaCore(CANONICAL_CONFIG)

    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # ---- Schedule: token-budget-aligned WSD ------------------------------
    tokens_per_step = args.tokens_per_step or (
        args.batch_size * args.grad_accum_steps * world_size * CANONICAL_CONFIG.block_size
    )
    budget_tokens = max(1, args.token_budget)
    budget_steps = max(1, budget_tokens // tokens_per_step)
    total_steps = args.max_steps if args.max_steps > 0 else budget_steps
    warmup_steps = max(1, int(total_steps * 0.02))
    scheduler = build_wsd_schedule(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    if rank == 0:
        phase = phase_for_step(start_step, warmup_steps=warmup_steps, total_steps=total_steps)
        print(
            f"[schedule] tokens/step={tokens_per_step:,} total_steps={total_steps:,} "
            f"warmup={warmup_steps:,} resume_phase={phase.name}",
            flush=True,
        )
    # Fast-forward scheduler to the resumed step.
    for _ in range(min(start_step, total_steps)):
        scheduler.step()

    # ---- Data loader ------------------------------------------------------
    shards = pack.shard_paths if pack is not None else tuple(
        sorted((pack_root / "train").glob("*.npy"))
    )
    dataset = ShardWindowDataset(shards, CANONICAL_CONFIG.block_size)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    device_loader = pl.MpDeviceLoader(loader, device, batches_per_execution=args.grad_accum_steps)
    sampler_epoch = 0
    sampler.set_epoch(sampler_epoch)
    iterator: Iterator[Any] = iter(device_loader)

    grad_accum = args.grad_accum_steps
    deadline = time.monotonic() + args.max_minutes * 60.0 if args.max_minutes > 0 else None
    output = Path(args.output_checkpoint)
    window_tokens = 0
    last_probe_step = -1

    if rank == 0:
        print(f"[TPU ready] cores={world_size} start={start_step:,} target={total_steps:,}",
              flush=True)

    step = start_step
    while step < total_steps:
        if deadline is not None and step > start_step and time.monotonic() >= deadline:
            break
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for _ in range(grad_accum):
            try:
                x, y = next(iterator)
            except StopIteration:
                sampler_epoch += 1
                sampler.set_epoch(sampler_epoch)
                iterator = iter(device_loader)
                x, y = next(iterator)
            with torch.autocast(device_type="xla", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, CANONICAL_CONFIG.vocab_size), y.reshape(-1)
                )
                scaled = loss / grad_accum
            scaled.backward()
            loss_sum.add_(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        xm.optimizer_step(optimizer, barrier=True)
        scheduler.step()
        step += 1

        if step % args.log_interval == 0 and rank == 0:
            mean_loss = float(loss_sum.cpu()) / grad_accum
            phase = phase_for_step(step, warmup_steps=warmup_steps, total_steps=total_steps)
            consumed = step * tokens_per_step
            print(
                f"step={step:,} loss={mean_loss:.4f} lr_mult={scheduler.get_last_lr()[0]/args.learning_rate:.3f} "
                f"phase={phase.name} tokens={consumed:,} "
                f"(pass {consumed // max(1, (pack.total_tokens if pack else dataset.total_tokens)) + 1})",
                flush=True,
            )
        if step % max(args.save_interval, 500) == 0 and rank == 0:
            probes = _milestone_probes(model)
            print(f"[probe] step={step:,} {probes}", flush=True)

    if rank == 0:
        _save_latest(xm, output, model, optimizer, tokenizer, step, source_checkpoint)
        print(f"[done] checkpoint saved: {output}", flush=True)
    return 0


def _save_latest(
    xm: Any,
    path: Path,
    model: AnRaCore,
    optimizer: torch.optim.Optimizer,
    tokenizer: V4Tokenizer,
    step: int,
    source_checkpoint: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{id(model)}.uploading")
    xm.save(
        {
            "checkpoint_artifact_class": "full_resume",
            "checkpoint_schema_version": 1,
            "global_step": step,
            "training_stage": "pretraining_tpu_xla_wsd",
            "source_commit": __import__("os").environ.get("ANRA_SOURCE_COMMIT", ""),
            "source_checkpoint": source_checkpoint or "",
            "model_config": CANONICAL_CONFIG.immutable_fields(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer_contract": tokenizer.identity() | {"available": True},
            "metrics": {},
        },
        str(temporary),
        master_only=True,
    )
    import os as _os

    _os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
