"""500M campaign milestone logic (pure; Cymek TrainingState is authoritative).

The 500M campaign measures CONSUMED TRAINING TOKENS via Cymek
TrainingState.cumulative_tokens — never update counts, parameter counts,
corpus bytes, or capacity estimates. Milestone checkpoints publish when the
first completed optimizer transaction crosses a threshold; the caller
persists published milestones so a resume can never republish one.
"""

from __future__ import annotations

from typing import Iterable, List

PRIMARY_FINAL_TOKEN_TARGET = 500_000_000
MILESTONES_500M = (50_000_000, 100_000_000, 200_000_000, 350_000_000,
                   500_000_000)


def crossed_milestones(previous_tokens: int, new_tokens: int,
                       milestones: Iterable[int] = MILESTONES_500M
                       ) -> List[int]:
    """Milestones strictly crossed when the consumed-token ledger moves from
    `previous_tokens` to `new_tokens`.

    Token-based (§2): correct for any batch/sequence/accumulation/device/
    packing change. A milestone m counts as crossed iff
    previous_tokens < m <= new_tokens — the first completed transaction
    that reaches m publishes it, and a resume that replays the same ledger
    transition re-derives the same crossing exactly once.
    """
    for name, value in (("previous_tokens", previous_tokens),
                        ("new_tokens", new_tokens)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be int, got {value!r}")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative, got {value}")
    if new_tokens < previous_tokens:
        raise ValueError(
            f"token ledger moved backwards: {new_tokens} < {previous_tokens}")
    return sorted(m for m in milestones
                  if previous_tokens < m <= new_tokens)


def next_milestone(consumed_tokens: int,
                   milestones: Iterable[int] = MILESTONES_500M) -> int | None:
    """The next milestone at or above the current consumed-token ledger."""
    upcoming = sorted(m for m in milestones if m > consumed_tokens)
    return upcoming[0] if upcoming else None


__all__ = ["MILESTONES_500M", "PRIMARY_FINAL_TOKEN_TARGET",
           "crossed_milestones", "next_milestone"]
