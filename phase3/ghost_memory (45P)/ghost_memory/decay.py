"""
45P Ghost State Memory — temporal decay and pruning.

Memories lose weight exponentially with age so stale context fades without
manual deletion. Pruning enforces a hard cap when the store exceeds MAX_MEMORIES.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

ScoreGetter = Callable[[Any], float]
IdGetter = Callable[[Any], int]


def apply_decay(
    created_at: float,
    reference_time: Optional[float] = None,
    half_life_days: float = 30.0,
) -> float:
    """
    Return a multiplicative decay factor in (0, 1] for a memory created at
    `created_at` (Unix seconds).

    Uses half-life decay: factor = 2 ** (-age_days / half_life_days), which is
    equivalent to exp(-ln(2) * age_days / half_life_days). Older memories get
    smaller factors so retrieval scores drop smoothly.
    """
    if half_life_days <= 0:
        return 1.0
    now = reference_time if reference_time is not None else time.time()
    age_seconds = max(0.0, now - float(created_at))
    age_days = age_seconds / 86400.0
    return float(math.pow(2.0, -age_days / float(half_life_days)))


def prune_candidates(
    items: Sequence[Any],
    max_keep: int,
    score_of: ScoreGetter,
    id_of: Optional[IdGetter] = None,
) -> Tuple[List[Any], List[Any]]:
    """
    When len(items) > max_keep, drop the lowest-scoring entries until only
    max_keep remain.

    score_of(item) should combine relevance and decay so pruning removes the
    least useful rows first. Returns (kept, removed) in arbitrary order within
    each list; callers persist the deletion of `removed` by id.

    id_of is optional and only used for documentation clarity in callers;
    pruning is purely by score ordering.
    """
    if len(items) <= max_keep:
        return list(items), []

    scored = [(score_of(it), i, it) for i, it in enumerate(items)]
    scored.sort(key=lambda x: x[0])
    n_drop = len(items) - max_keep
    removed = [t[2] for t in scored[:n_drop]]
    kept = [t[2] for t in scored[n_drop:]]
    return kept, removed
