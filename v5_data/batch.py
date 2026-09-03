"""Cursor-driven microbatch assembly over truly packed shards.

A microbatch is a run of consecutive packed sequences in sampler order. The
sampler cursor addresses ``(shard_ordinal, sequence_ordinal)``; this module
walks the packed sequences, returns raw token/segment tuples (framework
conversion stays in the training layer), and produces the exact per-source
real-token ledger the training state machine certifies. Real-token
consumption is cross-checked against ``v5_data.cursor.advance`` so the two
accounting paths cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cursor import advance
from .pack import MultiPackedShard


@dataclass(frozen=True, slots=True)
class MicroBatch:
    """Raw packed microbatch plus exact ledger bookkeeping."""

    tokens: tuple[tuple[int, ...], ...]
    segment_ids: tuple[tuple[int, ...], ...]
    tokens_by_source: dict[str, int]
    consumed_real_tokens: int
    shard_ordinal: int
    sequence_ordinal: int


def microbatch(
    shards: list[MultiPackedShard],
    order: list[int],
    *,
    shard_ordinal: int,
    sequence_ordinal: int,
    sequences: int,
    pad: int,
) -> MicroBatch:
    """Assemble ``sequences`` packed sequences starting at the cursor."""

    if sequences <= 0:
        raise ValueError("microbatch needs at least one sequence")
    if sorted(order) != list(range(len(shards))):
        raise ValueError("sampler order must permute every shard exactly once")
    if not 0 <= shard_ordinal < len(order):
        raise ValueError("shard ordinal is outside the pack")
    flat: list[tuple[int, int]] = []
    for position, shard_index in enumerate(order):
        for sequence_index in range(len(shards[shard_index].sequences)):
            flat.append((position, sequence_index))
    start_candidates = [
        index
        for index, coordinates in enumerate(flat)
        if coordinates == (shard_ordinal, sequence_ordinal)
    ]
    if not start_candidates:
        raise ValueError("cursor coordinates do not address a packed sequence")
    start = start_candidates[0]
    if start + sequences > len(flat):
        raise ValueError("microbatch runs past the end of the pack")
    tokens: list[tuple[int, ...]] = []
    segment_ids: list[tuple[int, ...]] = []
    by_source: dict[str, int] = {}
    for position, sequence_index in flat[start:start + sequences]:
        sequence = shards[order[position]].sequences[sequence_index]
        tokens.append(sequence.tokens)
        segment_ids.append(sequence.segment_ids)
        for index, source in enumerate(sequence.sources):
            count = sum(1 for segment in sequence.segment_ids if segment == index)
            by_source[source] = by_source.get(source, 0) + count
    consumed = sum(
        1
        for row_tokens, row_segments in zip(tokens, segment_ids)
        for _, segment in zip(row_tokens, row_segments)
        if segment >= 0
    )
    (end_shard, end_sequence), cursor_consumed = advance(
        shards,
        order,
        shard_ordinal=shard_ordinal,
        sequence_ordinal=sequence_ordinal,
        sequences=sequences,
        pad=pad,
    )
    by_source_total = sum(by_source.values())
    if consumed != by_source_total or cursor_consumed != by_source_total:
        raise ValueError("microbatch ledger disagrees with cursor consumption accounting")
    return MicroBatch(
        tokens=tuple(tokens),
        segment_ids=tuple(segment_ids),
        tokens_by_source=dict(sorted(by_source.items())),
        consumed_real_tokens=by_source_total,
        shard_ordinal=end_shard,
        sequence_ordinal=end_sequence,
    )


__all__ = ["MicroBatch", "microbatch"]
