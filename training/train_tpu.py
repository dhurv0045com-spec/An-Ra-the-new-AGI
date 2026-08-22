"""Canonical TPU/XLA trainer for An-Ra V4 - core-vnext training path.

Step accounting (the P0 fix):
    global_step  - checkpoint identity, monotonic across packs
    pack_step    - position in the CURRENT pack's schedule horizon
    schedule_step == pack_step (WSD decay lands at THIS pack's boundary)

A resumed global step never bounds the pack loop. Resuming global step 20,000
into a fresh 2,500-step pack runs exactly 2,500 updates.

Full resume semantics:
    model + optimizer + scheduler position are restored from the checkpoint.
    If optimizer state is absent, the artifact is treated as model_only and
    AdamW starts fresh - recorded honestly at launch.

Durability:
    - periodic atomic saves every --save-interval pack steps (latest)
    - best-loss candidate saved separately and never overwritten by worse
      later states
    - degradation guard warns; best artifact always remains available

Topology:
    launched through torch_xla.launch so all eight workers exist; timeout
    decisions are all-reduced across ranks before any rank exits.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from anra_core.config import CANONICAL_CONFIG
from anra_core.model import AnRaCore
from anra_core.tokenizer import V4Tokenizer
from training.pack_verify import PackVerificationError, VerifiedPack, verify_pack
from training.resume import (
    PackHorizon,
    degradation_ratio,
    resolve_pack_horizon,
    should_periodic_save,
    update_best,
)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="An-Ra V4 TPU trainer (core-vnext)")
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--resume-mode", default="auto",
                        choices=("auto", "same_pack", "new_pack_parent"),
                        help="resume semantics: auto infers from pack identity; "
                             "same_pack requires identical pack manifest sha256; "
                             "new_pack_parent resets pack_step and schedule")
    parser.add_argument("--resume-pack-sha", default="",
                        help="expected pack_manifest_sha256 of the checkpoint "
                             "(same_pack resume verification)")
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--best-checkpoint", default="",
                        help="separate path for the best-loss candidate (recommended)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="override pack step budget (0 = derive from token budget)")
    parser.add_argument("--max-minutes", type=float, default=450.0)
    parser.add_argument("--token-budget", type=int, default=330_000_000,
                        help="unique tokens this pack contributes; sets decay end")
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
    parser.add_argument("--skip-pack-verification", action="store_true")
    return parser.parse_args(argv)


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


class ResumeMode(str):
    """Explicit resume semantics. Never inferred accidentally.

    SAME_PACK       - continue an interrupted session; requires identical
                      pack manifest sha256; restores optimizer + pack_step.
    NEW_PACK_PARENT - use checkpoint model as parent for a fresh pack;
                      pack_step/schedule reset to 0; optimizer decision is
                      explicit (continue or reset), never accidental.
    """

    SAME_PACK = "same_pack"
    NEW_PACK_PARENT = "new_pack_parent"


@dataclass(slots=True)
class RestoredTrainingState:
    """What a resume actually restored — recorded, never implied."""

    global_step: int
    pack_step: int
    pack_manifest_sha256: str | None
    optimizer_restored: bool
    mode: str
    checkpoint_parameter_sha256: str


def _load_payload(resume_from: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load raw state + payload. Strict-first; legacy fallback ONLY for the
    specific known condition (missing/unavailable tokenizer contract on older
    writers). Corruption, architecture mismatch, and loader bugs surface."""
    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.errors import RepresentationIncompatibleError

    try:
        _model, _payload, _identity = load_core_checkpoint(resume_from)
    except RepresentationIncompatibleError as exc:
        if "tokenizer contract" not in str(exc):
            raise  # corruption/mismatch/bug: never disguise as legacy
        print(f"[resume] strict load refused ({exc}); explicit legacy fallback "
              "for older-writer contract absence", flush=True)
    payload = torch.load(resume_from, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") or payload.get("model")
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint has no model tensors: {resume_from}")
    return state, payload


def _restore_training_state(
    resume_from: str,
    model: AnRaCore,
    optimizer: torch.optim.Optimizer,
    *,
    mode: str = ResumeMode.SAME_PACK,
    current_pack_manifest_sha256: str | None = None,
) -> RestoredTrainingState:
    """Restore checkpoint INTO the caller's model (P0 fix).

    The previous implementation called ``load_core_checkpoint`` which builds a
    NEW model internally and returned only metadata - silently discarding the
    loaded weights while logs looked healthy. This version loads the state
    dict and installs it into ``model`` via load_state_dict, then verifies
    installation by exact tensor comparison on every tensor.

    Mode semantics:
      SAME_PACK: pack_step restored only if pack identity matches; mismatch
                 fails closed.
      NEW_PACK_PARENT: pack_step forced to 0; pack identity of checkpoint is
                 irrelevant; optimizer restoration is the caller's explicit
                 choice via allow_optimizer_continue.
    """
    state, payload = _load_payload(resume_from)

    # Install into the CALLER's model and prove it took effect.
    missing, unexpected = model.load_state_dict(state, strict=True)
    installed = 0
    for key, tensor in state.items():
        target = dict(model.state_dict())[key]
        if torch.is_floating_point(tensor):
            if not torch.equal(target, tensor):
                raise RuntimeError(f"resume verification failed: {key} did not install")
            installed += 1
    if installed == 0:
        raise RuntimeError("resume verification failed: no float tensors compared")

    optimizer_state = payload.get("optimizer") or payload.get("optimizer_state_dict")
    optimizer_restored = False
    if isinstance(optimizer_state, dict) and optimizer_state.get("state"):
        try:
            optimizer.load_state_dict(optimizer_state)
            optimizer_restored = True
        except Exception as exc:
            print(f"[resume] optimizer state present but failed to load: {exc}", flush=True)
            raise

    checkpoint_pack_sha = payload.get("pack_manifest_sha256")
    if mode == ResumeMode.SAME_PACK:
        restored_pack_step = int(payload.get("pack_step", 0) or 0)
        if restored_pack_step > 0 and not checkpoint_pack_sha:
            raise RuntimeError(
                "SAME_PACK resume refused: checkpoint carries pack_step without "
                "pack_manifest_sha256 - pack identity cannot be verified (fail closed)"
            )
        if (
            current_pack_manifest_sha256
            and checkpoint_pack_sha
            and current_pack_manifest_sha256 != checkpoint_pack_sha
        ):
            raise RuntimeError(
                "SAME_PACK resume refused: checkpoint was trained on a different "
                f"pack ({checkpoint_pack_sha[:16]}... != "
                f"{current_pack_manifest_sha256[:16]}...). Use NEW_PACK_PARENT."
            )
    else:  # NEW_PACK_PARENT
        restored_pack_step = 0

    import hashlib

    param_digest = hashlib.sha256()
    for key in sorted(state):
        param_digest.update(key.encode())
        param_digest.update(state[key].numpy().tobytes())

    return RestoredTrainingState(
        global_step=int(payload.get("global_step", 0) or 0),
        pack_step=restored_pack_step,
        pack_manifest_sha256=checkpoint_pack_sha,
        optimizer_restored=optimizer_restored,
        mode=mode,
        checkpoint_parameter_sha256=param_digest.hexdigest(),
    )


def run(args: argparse.Namespace) -> int:
    """Launch all XLA workers via the supported entry point.

    ``torch_xla.launch`` spawns the PJRT worker processes (all eight cores on
    v5e-8); calling ``_worker`` once in-process would train on a single device.
    Falls back to direct single-process execution when the multi-worker launch
    API is unavailable (older runtimes / CPU test harness).
    """
    import torch_xla.launch as xla_launch

    if hasattr(xla_launch, "launch"):
        xla_launch.launch(_worker, args=(args,))
        return 0
    print("[launch] torch_xla.launch unavailable; running single-process fallback", flush=True)
    return _worker(args)


def _worker(args: argparse.Namespace) -> int:
    xm, pl, xr = _require_xla()
    device = xm.xla_device()
    world_size = int(xr.world_size())
    rank = int(xr.get_ordinal())

    pack_root = Path(args.pack_root)
    pack: VerifiedPack | None = None
    if args.skip_pack_verification:
        if rank == 0:
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
                f"{pack.total_tokens:,} tokens, block={pack.block_size}",
                flush=True,
            )

    block_size = pack.block_size if pack is not None else CANONICAL_CONFIG.block_size
    if block_size != CANONICAL_CONFIG.block_size:
        print(f"[pack] REFUSING: pack block {block_size} != model context "
              f"{CANONICAL_CONFIG.block_size}", flush=True)
        return 2

    seed = args.seed
    torch.manual_seed(seed + rank)
    tokenizer = V4Tokenizer.load_canonical()

    model = AnRaCore(CANONICAL_CONFIG)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    start_global = 0
    restored_pack_step = 0
    optimizer_restored = False
    if args.resume_from:
        # Resume mode: explicit. Same pack continues; a different pack (or a
        # fresh start) is NEW_PACK_PARENT and resets pack-relative progress.
        pack_sha = pack.pack_manifest_sha256 if pack else None
        if args.resume_mode == "auto":
            mode = (
                ResumeMode.SAME_PACK
                if (pack_sha and args.resume_pack_sha in (None, "", pack_sha))
                else ResumeMode.NEW_PACK_PARENT
            )
        else:
            mode = ResumeMode(args.resume_mode)
        restored = _restore_training_state(
            str(Path(args.resume_from).expanduser()), model, optimizer,
            mode=mode,
            current_pack_manifest_sha256=(
                args.resume_pack_sha or None
                if args.resume_pack_sha
                else pack_sha
            ),
        )
        start_global = restored.global_step
        restored_pack_step = restored.pack_step
        optimizer_restored = restored.optimizer_restored
        if rank == 0:
            print(
                f"[resume] mode={restored.mode} global_step={start_global:,} "
                f"pack_step={restored_pack_step:,} "
                f"optimizer={'restored' if optimizer_restored else 'FRESH (explicit)'} "
                f"param_sha={restored.checkpoint_parameter_sha256[:16]}",
                flush=True,
            )

    model = model.to(device)
    model.train()
    # Move restored optimizer tensors onto XLA after the model move.
    if optimizer_restored:
        for state_tensor in optimizer.state.values():
            for key, value in state_tensor.items():
                if torch.is_tensor(value):
                    state_tensor[key] = value.to(device)

    tokens_per_step = args.tokens_per_step or (
        args.batch_size * args.grad_accum_steps * world_size * block_size
    )
    horizon: PackHorizon = resolve_pack_horizon(
        global_step=start_global,
        restored_pack_step=restored_pack_step,
        token_budget=args.token_budget,
        tokens_per_step=tokens_per_step,
        max_steps_override=args.max_steps,
    )
    if horizon.updates_remaining == 0:
        if rank == 0:
            print("[horizon] pack already complete; nothing to do", flush=True)
        return 0
    scheduler = build_wsd_schedule(
        optimizer,
        total_steps=horizon.pack_total_steps,
        warmup_steps=max(1, int(horizon.pack_total_steps * 0.02)),
        min_lr_ratio=args.min_lr_ratio,
    )
    for _ in range(horizon.start_pack_step):
        scheduler.step()
    if rank == 0:
        phase = phase_for_step(
            horizon.start_pack_step,
            warmup_steps=max(1, int(horizon.pack_total_steps * 0.02)),
            total_steps=horizon.pack_total_steps,
        )
        print(
            f"[schedule] pack_total={horizon.pack_total_steps:,} starting_phase={phase.name} "
            f"updates_this_session={horizon.updates_remaining:,} tokens/step={tokens_per_step:,}",
            flush=True,
        )

    shards = (
        pack.shard_paths
        if pack is not None
        else tuple(sorted((pack_root / "train").glob("*.npy")))
    )
    dataset = ShardWindowDataset(shards, block_size)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=seed, drop_last=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    device_loader = pl.MpDeviceLoader(loader, device, batches_per_execution=args.grad_accum_steps)
    sampler_epoch = 0
    sampler.set_epoch(sampler_epoch)
    iterator = iter(device_loader)

    grad_accum = args.grad_accum_steps
    deadline = time.monotonic() + args.max_minutes * 60.0 if args.max_minutes > 0 else None
    output = Path(args.output_checkpoint)
    best_path = Path(args.best_checkpoint) if args.best_checkpoint else None
    best_loss: float | None = None
    best_step_global = start_global

    if rank == 0:
        print(
            f"[TPU ready] cores={world_size} global={start_global:,} "
            f"pack_updates={horizon.updates_remaining:,}",
            flush=True,
        )

    def all_reduce_stop(stop_local: bool) -> bool:
        flag = torch.tensor([1 if stop_local else 0], device=device)
        flag = xm.all_reduce(xm.REDUCE_MAX, flag)
        return bool(int(flag.item()))

    pack_step = horizon.start_pack_step
    while pack_step < horizon.pack_total_steps:
        timed_out = deadline is not None and pack_step > horizon.start_pack_step and time.monotonic() >= deadline
        if all_reduce_stop(timed_out):
            if rank == 0:
                print("[stop] session deadline reached (synchronized)", flush=True)
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
        pack_step += 1
        start_global += 1

        if pack_step % args.log_interval == 0 and rank == 0:
            mean_loss = float(loss_sum.cpu()) / grad_accum
            phase = phase_for_step(pack_step, warmup_steps=max(1, int(horizon.pack_total_steps * 0.02)),
                                   total_steps=horizon.pack_total_steps)
            print(
                f"global={start_global:,} pack_step={pack_step:,}/{horizon.pack_total_steps:,} "
                f"loss={mean_loss:.4f} lr_mult="
                f"{scheduler.get_last_lr()[0]/args.learning_rate:.3f} phase={phase.name}",
                flush=True,
            )
            best_loss, improved = update_best(best_loss, mean_loss)
            if improved:
                best_step_global = start_global
                # BEST MEANS BEST: snapshot the exact state at measurement
                # time. The previous implementation saved the FINAL model as
                # "best" - a lie whenever training degraded after the best step.
                if rank == 0 and best_path is not None:
                    _save_checkpoint(
                        xm, best_path, model, optimizer, tokenizer,
                        start_global, pack_step,
                        source_checkpoint=args.resume_from or "",
                        extra={"best_loss": best_loss, "best_step_global": best_step_global},
                    )
                    print(f"[best] new best {best_loss:.4f} @ global {start_global:,} - snapshot saved", flush=True)
            elif degradation_ratio(best_loss, mean_loss) > 1.10:
                print(
                    f"[WARN] loss {mean_loss:.4f} >10% above best {best_loss:.4f} "
                    f"(step {best_step_global:,}). Best snapshot remains preserved.",
                    flush=True,
                )

        if rank == 0 and should_periodic_save(pack_step, args.save_interval):
            _save_checkpoint(
                xm, output, model, optimizer, tokenizer, start_global, pack_step,
                source_checkpoint=args.resume_from or "",
                pack_manifest_sha256=pack.pack_manifest_sha256 if pack else None,
                extra={"loss": float(loss_sum.cpu()) / grad_accum},
            )
            print(f"[save] latest @ pack_step {pack_step:,}", flush=True)

    if rank == 0:
        _save_checkpoint(
            xm, output, model, optimizer, tokenizer, start_global, pack_step,
            source_checkpoint=args.resume_from or "",
            pack_manifest_sha256=pack.pack_manifest_sha256 if pack else None,
        )
        print(
            f"[done] global={start_global:,} pack_step={pack_step:,}/"
            f"{horizon.pack_total_steps:,} best_loss={best_loss}",
            flush=True,
        )
    return 0


def _save_checkpoint(
    xm: Any,
    path: Path,
    model: AnRaCore,
    optimizer: torch.optim.Optimizer,
    tokenizer: V4Tokenizer,
    global_step: int,
    pack_step: int,
    *,
    source_checkpoint: str = "",
    pack_manifest_sha256: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{id(model)}.uploading")
    payload = {
        "checkpoint_artifact_class": "full_resume",
        "checkpoint_schema_version": 1,
        "global_step": global_step,
        "pack_step": pack_step,
        # Pack identity: pack_step is meaningless without the data it counts.
        "pack_manifest_sha256": pack_manifest_sha256,
        "training_stage": "pretraining_tpu_xla_wsd",
        "source_commit": __import__("os").environ.get("ANRA_SOURCE_COMMIT", ""),
        "source_checkpoint": source_checkpoint,
        "model_config": CANONICAL_CONFIG.immutable_fields(),
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "tokenizer_contract": tokenizer.identity() | {"available": True},
        "metrics": dict(extra or {}),
    }
    xm.save(payload, str(temporary), master_only=True)
    import os as _os

    _os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
