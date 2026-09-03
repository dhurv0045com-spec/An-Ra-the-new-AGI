"""Canonical update-stream construction: sampler order to exact microbatches.

One path for every consumer -- production training, canaries, and miniature
proofs alike.  The builder applies the deterministic sampler order over the
frozen pack, selects exactly-full sequences (padding never trains), groups
them into bucket-pure update windows whose real tokens equal the frozen
per-update budget (8x512 = 4x1024 = 2x2048 = 1x4096 = 4096), and reports the
real cursor coordinates each update consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .pack import BUCKETS, MultiPackedShard, PackedSequence, sampler_order


DEFAULT_REAL_TOKENS_PER_UPDATE = 4096


@dataclass(frozen=True, slots=True)
class UpdateWindow:
    """One exact-budget update window with its real cursor coordinates."""

    sequences: tuple[PackedSequence, ...]
    shard_ordinal: int
    sequence_ordinal: int
    real_tokens: int


def build_update_stream(
    shards: list[MultiPackedShard],
    *,
    run_seed: int,
    epoch: int = 0,
    real_tokens_per_update: int = DEFAULT_REAL_TOKENS_PER_UPDATE,
    sequences_per_update: Mapping[int, int] | None = None,
) -> list[UpdateWindow]:
    """Group exactly-full sequences into exact-budget update windows.

    The windows interleave buckets round-robin in ascending bucket order so
    every sequence class appears throughout training, and window cursors
    address the frozen shard list in sampler order.
    """

    if real_tokens_per_update <= 0:
        raise ValueError("real tokens per update must be positive")
    if sequences_per_update is None:
        sequences_per_update = {
            bucket: real_tokens_per_update // bucket
            for bucket in BUCKETS
            if real_tokens_per_update % bucket == 0
        }
    if not sequences_per_update:
        raise ValueError("no bucket divides the requested update size")
    for bucket, count in sequences_per_update.items():
        if count * bucket != real_tokens_per_update:
            raise ValueError(
                f"bucket {bucket} x {count} sequences != {real_tokens_per_update} real tokens"
            )
    order = sampler_order([shard.sha256() for shard in shards], run_seed=run_seed, epoch=epoch)
    pools: dict[int, list[tuple[int, int, PackedSequence]]] = {
        bucket: [] for bucket in sequences_per_update
    }
    for position, shard_index in enumerate(order):
        shard = shards[shard_index]
        if shard.bucket not in pools:
            continue
        for sequence_index, sequence in enumerate(shard.sequences):
            if -1 not in sequence.segment_ids and len(sequence.tokens) == shard.bucket:
                pools[shard.bucket].append((position, sequence_index, sequence))
    streams: list[list[UpdateWindow]] = []
    for bucket in sorted(sequences_per_update):
        need = sequences_per_update[bucket]
        pool = pools[bucket]
        windows = []
        for start in range(0, len(pool) - need + 1, need):
            group = pool[start:start + need]
            end_position, end_sequence, _ = group[-1]
            windows.append(
                UpdateWindow(
                    sequences=tuple(sequence for _, _, sequence in group),
                    shard_ordinal=end_position,
                    sequence_ordinal=end_sequence,
                    real_tokens=real_tokens_per_update,
                )
            )
        streams.append(windows)
    depth = max(len(stream) for stream in streams)
    interleaved: list[UpdateWindow] = []
    for level in range(depth):
        for stream in streams:
            if level < len(stream):
                interleaved.append(stream[level])
    return interleaved


__all__ = [
    "DEFAULT_REAL_TOKENS_PER_UPDATE",
    "UpdateWindow",
    "build_update_stream",
]
