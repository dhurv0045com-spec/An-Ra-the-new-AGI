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
class ExactWindow:
    """One exact real-token microstep window with cursor progression."""

    tokens: tuple[tuple[int, ...], ...]
    segment_ids: tuple[tuple[int, ...], ...]
    eligible: tuple[tuple[bool, ...], ...]
    tokens_by_source: dict[str, int]
    real_tokens: int
    row_buckets: tuple[int, ...]
    end_shard_ordinal: int
    end_sequence_ordinal: int
    end_token_offset: int


def build_flat_index(
    shards: list[MultiPackedShard],
    order: list[int],
) -> list[tuple[int, int]]:
    """Precompute the (order position, sequence) walk used by exact windows.

    Memory is bounded by pack size, never by campaign length: long campaigns
    build this once and pass it as ``flat_cache`` instead of rebuilding it
    per microstep window.
    """

    if sorted(order) != list(range(len(shards))):
        raise ValueError("sampler order must permute every shard exactly once")
    flat: list[tuple[int, int]] = []
    for position, shard_index in enumerate(order):
        for sequence_index in range(len(shards[shard_index].sequences)):
            flat.append((position, sequence_index))
    return flat


def exact_window(
    shards: list[MultiPackedShard],
    order: list[int],
    *,
    shard_ordinal: int,
    sequence_ordinal: int,
    token_offset: int,
    real_tokens: int,
    pad: int,
    flat_cache: list[tuple[int, int]] | None = None,
) -> ExactWindow:
    """Take exactly ``real_tokens`` real tokens from the cursor.

    Whole sequences transfer while they fit; the final sequence is cut at
    the exact boundary with an eligibility mask over kept real positions.
    Rows keep native lengths (callers pad to the window maximum); padding
    is never eligible. Sequences may span buckets: the window records the
    per-row bucket mix for audit, and bucket proportions are governed at
    layout time (see ``v5_data.layout``). The end cursor (order position,
    sequence, offset) resumes deterministically with no replay. Raises when
    the pack cannot supply the budget. ``flat_cache`` (from
    ``build_flat_index``) reuses one pack-sized walk across many windows;
    without it the walk is rebuilt per call with identical results.
    """

    if real_tokens <= 0:
        raise ValueError("window needs a positive real-token budget")
    if sorted(order) != list(range(len(shards))):
        raise ValueError("sampler order must permute every shard exactly once")
    if not 0 <= shard_ordinal < len(order):
        raise ValueError("shard ordinal is outside the pack")
    if token_offset < 0:
        raise ValueError("token offset cannot be negative")
    flat = flat_cache if flat_cache is not None else build_flat_index(shards, order)
    starts = [
        index for index, coordinates in enumerate(flat)
        if coordinates == (shard_ordinal, sequence_ordinal)
    ]
    if not starts:
        raise ValueError("cursor coordinates do not address a packed sequence")
    rows_tokens: list[list[int]] = []
    rows_segments: list[list[int]] = []
    rows_eligible: list[list[bool]] = []
    rows_buckets: list[int] = []
    by_source: dict[str, int] = {}
    consumed = 0
    cursor = starts[0]
    pending_offset = token_offset
    end_shard, end_sequence, end_offset = shard_ordinal, sequence_ordinal, token_offset
    while consumed < real_tokens:
        if cursor >= len(flat):
            raise ValueError("window runs past the end of the pack")
        position, sequence_index = flat[cursor]
        sequence = shards[order[position]].sequences[sequence_index]
        rows_buckets.append(len(sequence.tokens))
        real_positions = [
            index for index, token in enumerate(sequence.tokens) if token != pad
        ]
        if pending_offset > len(real_positions):
            raise ValueError("token offset is past the sequence real tokens")
        need = real_tokens - consumed
        take = real_positions[pending_offset:pending_offset + need]
        kept: set[int] = set(take)
        rows_tokens.append(list(sequence.tokens))
        rows_segments.append(list(sequence.segment_ids))
        rows_eligible.append([
            index in kept for index in range(len(sequence.tokens))
        ])
        for index in take:
            segment = sequence.segment_ids[index]
            source = sequence.sources[segment] if 0 <= segment < len(sequence.sources) else None
            if source is None:
                raise ValueError("eligible position lacks source attribution")
            by_source[source] = by_source.get(source, 0) + 1
        consumed += len(take)
        if pending_offset + len(take) < len(real_positions):
            end_shard, end_sequence = position, sequence_index
            end_offset = pending_offset + len(take)
        else:
            following = cursor + 1
            if following < len(flat):
                end_shard, end_sequence = flat[following]
            else:
                end_shard, end_sequence = position, sequence_index + 1
            end_offset = 0
        pending_offset = 0
        cursor += 1
    if consumed != real_tokens or sum(by_source.values()) != real_tokens:
        raise ValueError("window ledger disagrees with the exact budget")
    return ExactWindow(
        tokens=tuple(tuple(row) for row in rows_tokens),
        segment_ids=tuple(tuple(row) for row in rows_segments),
        eligible=tuple(tuple(row) for row in rows_eligible),
        tokens_by_source=dict(sorted(by_source.items())),
        real_tokens=consumed,
        row_buckets=tuple(rows_buckets),
        end_shard_ordinal=end_shard,
        end_sequence_ordinal=end_sequence,
        end_token_offset=end_offset,
    )


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


__all__ = ["ExactWindow", "MicroBatch", "build_flat_index", "exact_window", "microbatch"]
