"""Paired statistical procedures preregistered for E0 model comparisons."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class PairedDelta:
    estimate: float
    lower_95: float
    upper_95: float
    samples: int


def paired_sign_test_pvalue(left: Sequence[bool], right: Sequence[bool]) -> float:
    """Exact two-sided sign test over discordant paired binary outcomes."""

    if len(left) != len(right) or not left:
        raise ValueError("paired nonempty outcomes must have equal length")
    wins = sum(a and not b for a, b in zip(left, right))
    losses = sum(b and not a for a, b in zip(left, right))
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(wins, losses) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_delta(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int,
    resamples: int = 10_000,
) -> PairedDelta:
    if len(left) != len(right) or not left:
        raise ValueError("paired nonempty measurements must have equal length")
    if resamples < 1_000:
        raise ValueError("promotion bootstrap requires at least 1,000 resamples")
    deltas = [a - b for a, b in zip(left, right)]
    estimate = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    draws = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(resamples)
    )
    lower = draws[int(0.025 * resamples)]
    upper = draws[min(resamples - 1, int(0.975 * resamples))]
    return PairedDelta(estimate, lower, upper, resamples)
