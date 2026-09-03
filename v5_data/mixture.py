"""Exact token-budget allocation for the frozen V5 mixture.

Budgets: 65% high-quality natural, 20% code/mathematics/formal, 15%
mechanically verified cognition, over exactly 5,000,000,000 real non-padding
tokens. Cognition families, difficulty shares, and the 20-microstep
supercycle come from the frozen training spec. Allocation uses
largest-remainder rounding so every split sums exactly.
"""

from __future__ import annotations


TOTAL_TOKENS = 5_000_000_000
SLICE_FRACTIONS = {"natural": 0.65, "code_math_formal": 0.20, "verified_cognition": 0.15}

COGNITION_FRACTIONS = {
    "identity_copy": 0.08,
    "query_binding": 0.16,
    "semantic_state": 0.16,
    "interference_retrieval": 0.10,
    "relational_composition": 0.20,
    "counterfactual_sensitivity": 0.10,
    "heldout_rule_induction": 0.10,
    "missing_information": 0.05,
    "faithful_realization": 0.05,
}

DIFFICULTY_FRACTIONS = {"easy": 0.34, "medium": 0.355, "hard": 0.305}

BUCKET_FRACTIONS = {512: 0.25, 1024: 0.25, 2048: 0.30, 4096: 0.20}
SUPERCYCLE = [512, 1024, 2048, 4096, 2048, 512, 1024, 2048, 4096, 512,
              1024, 2048, 4096, 2048, 512, 1024, 2048, 4096, 512, 1024]


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


__all__ = [
    "BUCKET_FRACTIONS",
    "COGNITION_FRACTIONS",
    "DIFFICULTY_FRACTIONS",
    "SLICE_FRACTIONS",
    "SUPERCYCLE",
    "TOTAL_TOKENS",
    "allocate",
    "bucket_plan",
    "cognition_allocation",
    "slice_allocation",
]
