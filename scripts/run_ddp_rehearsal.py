"""Two-GPU exact-resume rehearsal for the An-Ra DDP contract.

The rehearsal intentionally stays separate from the canonical trainer.  It
uses deterministic synthetic windows and a tiny dropout model to prove:

* one rank-strided global data sequence with no replay;
* real DDP gradient accumulation using ``no_sync``;
* a rank-zero checkpoint only at completed optimizer boundaries;
* rank-local RNG restoration after DataLoader construction; and
* identical uninterrupted and interrupted/resumed training-state fingerprints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from training.curriculum_sampler import (
    DeterministicPermutationSampler,
    RankStridedSampler,
)
from training.distributed import (
    all_gather_objects,
    all_reduce_bool_or,
    all_reduce_mean,
    barrier_or_raise,
    destroy_distributed,
    initialize_distributed,
)

CHECKPOINT_SCHEMA = "anra-ddp-rehearsal/v2"
REPORT_SCHEMA = "anra-ddp-rehearsal-report/v2"


class InjectedRehearsalInterruptionError(RuntimeError):
    """Expected failure used to prove that partial accumulation is discarded."""


class _FixedWindows(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, count: int, sequence_length: int, vocab_size: int) -> None:
        self.count = int(count)
        self.sequence_length = int(sequence_length)
        self.vocab_size = int(vocab_size)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        base = torch.arange(self.sequence_length + 1, dtype=torch.long)
        tokens = (base * 17 + int(index) * 31 + 7) % self.vocab_size
        return tokens[:-1], tokens[1:], int(index)


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.norm(self.embedding(tokens))))


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def _seed_rank_training_rng(seed: int, rank: int) -> None:
    rank_seed = int(seed) + (int(rank) + 1) * 1_000_003
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)


def _atomic_torch_save(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_value(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        for item in value:
            _hash_value(digest, item)
    else:
        digest.update(type(value).__name__.encode() + b"\0")
        digest.update(repr(value).encode())


def _training_state_fingerprint(
    model_state: dict[str, object],
    optimizer_state: dict[str, object],
    *,
    global_step: int,
    global_cursor: int,
    consumed_indices: list[int],
    distributed_rng_states: dict[str, object],
) -> str:
    digest = hashlib.sha256()
    _hash_value(
        digest,
        {
            "model": model_state,
            "optimizer": optimizer_state,
            "global_step": global_step,
            "global_cursor": global_cursor,
            "consumed_indices": consumed_indices,
            "distributed_rng_states": distributed_rng_states,
        },
    )
    return digest.hexdigest()


def _rehearsal_contract(args: argparse.Namespace, context: Any) -> dict[str, object]:
    return {
        "distributed": context.contract(
            micro_batch_size_per_rank=1,
            gradient_accumulation=int(args.accumulation),
        ),
        "seed": int(args.seed),
        "windows": int(args.windows),
        "sequence_length": int(args.sequence_length),
        "vocab_size": int(args.vocab_size),
        "width": int(args.width),
        "learning_rate": float(args.learning_rate),
        "weight_decay": 0.1,
        "model": "tiny_tied_embedding_dropout_v1",
        "sampler": "counter_based_sha256_v1",
    }


def _validate_resume_payload(
    payload: dict[str, object],
    *,
    expected_contract: dict[str, object],
    base_sampler: DeterministicPermutationSampler,
) -> None:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("DDP rehearsal checkpoint schema is not exact-resume compatible")
    if payload.get("rehearsal_contract") != expected_contract:
        raise RuntimeError("DDP rehearsal topology or lineage changed across exact resume")
    distributed = expected_contract["distributed"]
    assert isinstance(distributed, dict)
    sequences_per_step = int(distributed["global_sequences_per_step"])
    global_step = int(payload["global_step"])
    global_cursor = int(payload["global_cursor"])
    consumed = [int(value) for value in payload["consumed_indices"]]
    if global_cursor != global_step * sequences_per_step:
        raise RuntimeError("DDP rehearsal cursor is not an optimizer-step boundary")
    if len(consumed) != global_cursor:
        raise RuntimeError("DDP rehearsal consumed history does not match its cursor")
    expected_prefix = [base_sampler.index_at(position) for position in range(global_cursor)]
    if consumed != expected_prefix:
        raise RuntimeError("DDP rehearsal consumed history is not the canonical sampler prefix")
    expected_ranks = {
        str(rank) for rank in range(int(distributed["world_size"]))
    }
    rank_states = payload.get("distributed_rng_states")
    if not isinstance(rank_states, dict) or set(rank_states) != expected_ranks:
        raise RuntimeError("DDP rehearsal checkpoint lacks an exact per-rank RNG set")


def _collective_topology_check(context: Any) -> None:
    rows = all_gather_objects(
        {
            "rank": context.rank,
            "local_rank": context.local_rank,
            "hostname": socket.gethostname(),
            "visible_cuda_devices": torch.cuda.device_count(),
        },
        context,
    )
    if {str(row["hostname"]) for row in rows} != {socket.gethostname()}:
        raise RuntimeError("initial An-Ra DDP rehearsal is restricted to one physical host")
    if {int(row["rank"]) for row in rows} != set(range(context.world_size)):
        raise RuntimeError("DDP rehearsal did not gather one record from every global rank")
    if len({int(row["local_rank"]) for row in rows}) != context.world_size:
        raise RuntimeError("same-host DDP ranks do not own unique CUDA devices")


def _collective_resume_load(
    checkpoint: Path,
    *,
    context: Any,
    expected_contract: dict[str, object],
    base_sampler: DeterministicPermutationSampler,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    payload: dict[str, object] | None = None
    local_error: str | None = None
    checkpoint_hash: str | None = None
    try:
        checkpoint_hash = _file_sha256(checkpoint)
        loaded = torch.load(checkpoint, map_location=context.device, weights_only=False)
        if not isinstance(loaded, dict):
            raise RuntimeError("checkpoint root is not a mapping")
        _validate_resume_payload(
            loaded,
            expected_contract=expected_contract,
            base_sampler=base_sampler,
        )
        model.load_state_dict(loaded["model"], strict=True)
        optimizer.load_state_dict(loaded["optimizer"])
        payload = loaded
    except Exception as exc:
        local_error = f"rank {context.rank}: {type(exc).__name__}: {exc}"
    outcomes = all_gather_objects(
        {"error": local_error, "checkpoint_sha256": checkpoint_hash}, context
    )
    errors = [str(row["error"]) for row in outcomes if row["error"]]
    if errors:
        raise RuntimeError("collective resume validation failed: " + " | ".join(errors))
    hashes = {str(row["checkpoint_sha256"]) for row in outcomes}
    if len(hashes) != 1 or None in {row["checkpoint_sha256"] for row in outcomes}:
        raise RuntimeError("DDP ranks did not load identical checkpoint bytes")
    assert payload is not None
    return payload


def run_rehearsal(args: argparse.Namespace) -> dict[str, object]:
    context = None
    try:
        context = initialize_distributed("ddp")
        if context.world_size != int(args.world_size):
            raise RuntimeError(
                f"rehearsal expected world_size={args.world_size}, received {context.world_size}"
            )
        if int(args.accumulation) < 1:
            raise ValueError("accumulation must be positive")
        if int(args.fault_after_microsteps) < 0:
            raise ValueError("fault-after-microsteps cannot be negative")
        _collective_topology_check(context)

        seed = int(args.seed)
        # Model initialization is deliberately identical before rank-local RNG begins.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        dataset = _FixedWindows(args.windows, args.sequence_length, args.vocab_size)
        base_sampler = DeterministicPermutationSampler(
            len(dataset), num_samples=len(dataset), seed=seed
        )
        model = _TinyLM(args.vocab_size, args.width).to(context.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=0.1
        )
        contract = _rehearsal_contract(args, context)
        global_step = 0
        global_cursor = 0
        consumed: list[int] = []
        resume_rng: dict[str, object] | None = None
        checkpoint = Path(args.checkpoint).resolve()
        if args.resume:
            payload = _collective_resume_load(
                checkpoint,
                context=context,
                expected_contract=contract,
                base_sampler=base_sampler,
                model=model,
                optimizer=optimizer,
            )
            global_step = int(payload["global_step"])
            global_cursor = int(payload["global_cursor"])
            consumed = [int(value) for value in payload["consumed_indices"]]
            rank_states = payload["distributed_rng_states"]
            assert isinstance(rank_states, dict)
            resume_rng = rank_states[str(context.rank)]

        accumulation = int(args.accumulation)
        sequences_per_step = context.world_size * accumulation
        if global_cursor % sequences_per_step:
            raise RuntimeError("resume cursor is inside an accumulation boundary")
        remaining = len(dataset) - global_cursor
        if remaining % sequences_per_step:
            raise RuntimeError(
                "remaining rehearsal windows must divide into complete optimizer steps"
            )

        train_model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        sampler = RankStridedSampler(
            base_sampler,
            rank=context.rank,
            world_size=context.world_size,
            global_cursor=global_cursor,
        )
        # Iterator creation consumes only this private generator, never training RNG.
        loader_generator = torch.Generator().manual_seed(seed + 97_409)
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            generator=loader_generator,
        )
        loader_iterator = iter(loader)
        if resume_rng is None:
            _seed_rank_training_rng(seed, context.rank)
        else:
            _restore_rng_state(resume_rng)

        target_step = min(int(args.steps), global_step + len(loader) // accumulation)
        optimizer.zero_grad(set_to_none=True)
        pending_indices: list[int] = []
        pending_loss = torch.zeros((), dtype=torch.float32, device=context.device)
        session_microsteps = 0

        while global_step < target_step:
            for micro_step in range(accumulation):
                inputs, targets, indices = next(loader_iterator)
                inputs = inputs.to(context.device)
                targets = targets.to(context.device)
                synchronize = micro_step + 1 == accumulation
                sync_context = nullcontext() if synchronize else train_model.no_sync()
                with sync_context:
                    logits = train_model(inputs)
                    raw_loss = nn.functional.cross_entropy(
                        logits.flatten(0, 1), targets.flatten()
                    )
                    (raw_loss / accumulation).backward()
                pending_loss.add_(raw_loss.detach())

                rank_indices = [int(value) for value in indices.tolist()]
                gathered = all_gather_objects(rank_indices, context)
                accepted = [value for rank_values in gathered for value in rank_values]
                if len(accepted) != len(set(accepted)):
                    raise RuntimeError("DDP rehearsal consumed a window on multiple ranks")
                if set(accepted).intersection(consumed) or set(accepted).intersection(
                    pending_indices
                ):
                    raise RuntimeError("DDP rehearsal replayed an already accepted window")
                pending_indices.extend(accepted)
                session_microsteps += 1

                requested_fault = (
                    int(args.fault_after_microsteps) > 0
                    and session_microsteps == int(args.fault_after_microsteps)
                    and (
                        int(args.fault_rank) < 0
                        or context.rank == int(args.fault_rank)
                    )
                )
                if all_reduce_bool_or(requested_fault, context):
                    optimizer.zero_grad(set_to_none=True)
                    raise InjectedRehearsalInterruptionError(
                        "injected after backward and before the next protected optimizer boundary"
                    )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            consumed.extend(pending_indices)
            global_cursor += len(pending_indices)
            pending_indices.clear()
            global_step += 1
            mean_loss = float(
                all_reduce_mean(pending_loss / accumulation, context).item()
            )
            pending_loss.zero_()

            gathered_rng = all_gather_objects(_rng_state(), context)
            rng_states = {str(rank): state for rank, state in enumerate(gathered_rng)}
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()
            fingerprint = _training_state_fingerprint(
                model_state,
                optimizer_state,
                global_step=global_step,
                global_cursor=global_cursor,
                consumed_indices=consumed,
                distributed_rng_states=rng_states,
            )
            primary_error: str | None = None
            if context.is_primary:
                try:
                    _atomic_torch_save(
                        {
                            "schema": CHECKPOINT_SCHEMA,
                            "model": model_state,
                            "optimizer": optimizer_state,
                            "global_step": global_step,
                            "global_cursor": global_cursor,
                            "consumed_indices": consumed,
                            "rehearsal_contract": contract,
                            "distributed_rng_states": rng_states,
                            "mean_loss": mean_loss,
                            "state_fingerprint": fingerprint,
                        },
                        checkpoint,
                    )
                except Exception as exc:  # pragma: no cover - shared filesystem failure
                    primary_error = f"{type(exc).__name__}: {exc}"
            barrier_or_raise(context, primary_error=primary_error)

        gathered_rng = all_gather_objects(_rng_state(), context)
        final_fingerprint = _training_state_fingerprint(
            model.state_dict(),
            optimizer.state_dict(),
            global_step=global_step,
            global_cursor=global_cursor,
            consumed_indices=consumed,
            distributed_rng_states={
                str(rank): state for rank, state in enumerate(gathered_rng)
            },
        )
        fingerprints = all_gather_objects(final_fingerprint, context)
        if len(set(fingerprints)) != 1:
            raise RuntimeError("DDP ranks disagree on final training-state fingerprint")
        if args.expect_fingerprint and final_fingerprint != args.expect_fingerprint:
            raise RuntimeError(
                "interrupted/resumed state differs from uninterrupted reference: "
                f"expected {args.expect_fingerprint}, received {final_fingerprint}"
            )

        report = {
            "schema": REPORT_SCHEMA,
            "global_step": global_step,
            "global_cursor": global_cursor,
            "checkpoint": str(checkpoint),
            "world_size": context.world_size,
            "accumulation": accumulation,
            "state_fingerprint": final_fingerprint,
            "rank": context.rank,
        }
        if context.is_primary:
            if args.report:
                _atomic_json_save(report, Path(args.report).resolve())
            print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        if context is not None:
            destroy_distributed(context)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--fault-after-microsteps", type=int, default=0)
    parser.add_argument("--fault-rank", type=int, default=-1)
    parser.add_argument("--expect-fingerprint", type=str, default="")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1301)
    run_rehearsal(parser.parse_args())


if __name__ == "__main__":
    main()
