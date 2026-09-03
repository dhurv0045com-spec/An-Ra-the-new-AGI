"""Sampler cursor arithmetic over immutable pack shards.

Coordinates are ``(shard_ordinal, sequence_ordinal)`` into the frozen shard
list in sampler order. The cursor advances one sequence at a time, rolls into
the next shard, and reports exact real-token consumption for the training
ledger. Callers persist coordinates in ``v5_training.state.CursorState``;
this module never touches training state.
"""

from __future__ import annotations

from .pack import PackedShard


def sequence_count(shards: list[PackedShard], order: list[int]) -> int:
    """Return the total addressable sequences in sampler order."""

    if sorted(order) != list(range(len(shards))):
        raise ValueError("sampler order must permute every shard exactly once")
    return sum(len(shards[index].sequences) for index in order)


def advance(
    shards: list[PackedShard],
    order: list[int],
    *,
    shard_ordinal: int,
    sequence_ordinal: int,
    sequences: int = 1,
    pad: int,
) -> tuple[tuple[int, int], int]:
    """Advance a cursor by whole sequences; return coordinates and real tokens."""

    if sequences <= 0:
        raise ValueError("must advance at least one sequence")
    if not 0 <= shard_ordinal < len(order):
        raise ValueError("shard ordinal is outside the pack")
    flat: list[tuple[int, int]] = [
        (position, sequence)
        for position in range(len(order))
        for sequence in range(len(shards[order[position]].sequences))
    ]
    try:
        cursor = flat.index((shard_ordinal, sequence_ordinal))
    except ValueError:
        raise ValueError("cursor coordinates do not address a packed sequence") from None
    if cursor + sequences > len(flat):
        raise ValueError("advance runs past the end of the pack")
    consumed = 0
    for step in range(sequences):
        position, sequence = flat[cursor + step]
        tokens = shards[order[position]].sequences[sequence]
        consumed += sum(1 for token in tokens if token != pad)
    return flat[cursor + sequences - 1], consumed


__all__ = ["advance", "sequence_count"]
