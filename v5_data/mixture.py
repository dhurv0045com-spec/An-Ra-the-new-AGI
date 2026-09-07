"""Exact token-budget allocation for the frozen V5 mixture.

Budgets: 65% high-quality natural, 20% code/mathematics/formal, 15%
mechanically verified cognition, over exactly 5,000,000,000 real non-padding
tokens. Cognition families, difficulty shares, and the 20-microstep
supercycle come from the frozen training spec. Allocation uses
largest-remainder rounding so every split sums exactly. The DeficitScheduler
schedules those shares online over committed real-token counters for
campaign execution; Triquetra owns qualification, this module only
schedules, ledgers, and binds identities.
"""

from __future__ import annotations

import hashlib
import json

from v5_contracts.training_spec import build_training_spec


def _frozen() -> dict:
    return build_training_spec()


TOTAL_TOKENS = 5_000_000_000
SLICE_FRACTIONS = {str(name): float(share) for name, share in
                   _frozen()["data"]["mixture_fractions"].items()}

COGNITION_FRACTIONS = {str(name): float(share) for name, share in
                       _frozen()["cognition"]["family_fractions_within_cognition"].items()}

DIFFICULTY_FRACTIONS = {str(name): float(share) for name, share in
                        _frozen()["cognition"]["difficulty_distribution"].items()}

BUCKET_FRACTIONS = {int(bucket): float(share) for bucket, share in
                    _frozen()["packing"]["sequence_buckets"].items()}
SUPERCYCLE = [int(bucket) for bucket in
              _frozen()["packing"]["twenty_microstep_supercycle"]]


def allocate(total: int, fractions: dict[str, float]) -> dict[str, int]:
    """Split an integer budget by fractions with largest-remainder exactness."""

    if total < 0:
        raise ValueError("budget cannot be negative")
    if not fractions or any(fraction < 0 for fraction in fractions.values()):
        raise ValueError("fractions must be nonempty and nonnegative")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to one")
    exact = {name: total * fraction for name, fraction in fractions.items()}
    floored = {name: int(value) for name, value in exact.items()}
    remainder = total - sum(floored.values())
    order = sorted(fractions, key=lambda name: (exact[name] - floored[name], name), reverse=True)
    for index in range(remainder):
        floored[order[index % len(order)]] += 1
    return floored


def slice_allocation() -> dict[str, int]:
    """Allocate the 5B budget across the three top-level slices."""

    return allocate(TOTAL_TOKENS, SLICE_FRACTIONS)


def cognition_allocation() -> dict[str, int]:
    """Allocate the 750M cognition slice across the nine families."""

    return allocate(allocate(TOTAL_TOKENS, SLICE_FRACTIONS)["verified_cognition"], COGNITION_FRACTIONS)


def bucket_plan(supercycle_repeats: int) -> list[int]:
    """Expand the deterministic 20-microstep bucket supercycle."""

    if supercycle_repeats <= 0:
        raise ValueError("supercycle repeats must be positive")
    return SUPERCYCLE * supercycle_repeats


MIXTURE_SCHEDULE_SCHEMA = "anra-v5-mixture-schedule/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


class DeficitScheduler:
    """Largest-deficit greedy scheduler over committed token counters.

    At a committed total T, the next family maximizes
    ``fraction * T - consumed[family]`` (deterministic construction-order
    tie-break). Pure: identical counters always yield the identical choice,
    so offline demand planning and live execution cannot diverge.
    """

    def __init__(self, *, fractions: dict[str, float],
                 order: tuple[str, ...] | None = None) -> None:
        if not fractions or any(value < 0 for value in fractions.values()):
            raise ValueError("scheduler fractions require names and nonnegative shares")
        if abs(sum(fractions.values()) - 1.0) > 1e-9:
            raise ValueError("scheduler fractions must sum to one")
        self.fractions = dict(fractions)
        self.order = tuple(order) if order is not None else tuple(fractions)

    def next(self, *, consumed_total: int,
             consumed: dict[str, int]) -> str:
        """Return the family to schedule next. Pure function of counters."""

        if consumed_total < 0:
            raise ValueError("consumed total cannot be negative")
        best: str | None = None
        best_deficit = 0.0
        first = True
        for name in self.order:
            if name not in self.fractions:
                raise ValueError(f"unknown scheduled family: {name}")
            deficit = self.fractions[name] * consumed_total - consumed.get(name, 0)
            if first or deficit > best_deficit:
                best, best_deficit, first = name, deficit, False
        assert best is not None
        return best


def mixture_schedule_sha256(*, fractions: dict[str, float],
                            allocation: dict[str, int],
                            cell_map_sha256: str | None) -> str:
    """Content identity of the mixture plan actually executed."""

    return hashlib.sha256(_canonical_json(
        {"schema": MIXTURE_SCHEDULE_SCHEMA, "fractions": fractions,
         "allocation": allocation, "cell_map_sha256": cell_map_sha256})).hexdigest()


__all__ = [
    "BUCKET_FRACTIONS",
    "COGNITION_FRACTIONS",
    "DIFFICULTY_FRACTIONS",
    "DeficitScheduler",
    "MIXTURE_SCHEDULE_SCHEMA",
    "SLICE_FRACTIONS",
    "SUPERCYCLE",
    "TOTAL_TOKENS",
    "allocate",
    "bucket_plan",
    "cognition_allocation",
    "mixture_schedule_sha256",
    "slice_allocation",
]
