"""Readiness v2 status taxonomy (Mission 1/2/3/4/30/31).

Three SEPARATE questions (never conflate):

  SubstrateCapability:      does the model have the computation?
    INSUFFICIENT | WEAK | PARTIAL | STRONG
  ExperimentIdentifiability: can this regime distinguish hypotheses?
    NOT_IDENTIFIABLE | MARGINAL | IDENTIFIABLE
  ResearchReadiness:        may advanced cognition science run here?
    NOT_READY | CALIBRATION_REQUIRED | READY_SCOPED | READY

READY is conservative by design: false-positive readiness wastes entire
research programs; false negatives cost one more calibration run.

Conventions: Wilson 95% CIs on all binary rates (exact, no bootstrap needed
at small N); chance-adjusted reporting (acc, chance, diff) with raw numbers
primary; v0 numeric defaults are weak-substrate-calibrated estimates and are
returned inside receipts so they cannot silently become dogma.
"""

from __future__ import annotations

import math

Z95 = 1.96

# v0 defaults (estimates, echoed in every receipt; see module docstring).
CAP_STRONG_LO = 0.80
CAP_PARTIAL_N_MIN = 30
ORACLE_FLOOR_V2 = 0.40
CEILING_V2 = 0.95


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson 95% interval for a binomial rate. Fail-closed on n=0."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    m = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def chance_report(k: int, n: int, chance: float | None,
                  mode: str = "k_way") -> dict:
    """Chance-aware report with explicit evaluation-mode semantics (Mission 1).

    Modes: RAW_FREE_GENERATION | K_WAY_CANDIDATE_SELECTION |
    CONSTRAINED_CANDIDATE_GENERATION | ORACLE_ASSISTED_GENERATION.
    Only K_WAY_CANDIDATE_SELECTION may carry chance = 1/k. Free generation
    has chance None (null) unless an explicit generative null exists; passing
    a chance for free generation raises instead of silently mixing semantics.
    """
    if mode == "RAW_FREE_GENERATION" and chance is not None:
        raise ValueError("chance must be null for RAW_FREE_GENERATION "
                         "(free text is not forced choice)")
    if mode == "K_WAY_CANDIDATE_SELECTION" and chance is None:
        raise ValueError("K_WAY_CANDIDATE_SELECTION requires chance = 1/k")
    lo, hi = wilson(k, n)
    acc = k / n if n else 0.0
    return {"mode": mode, "acc": round(acc, 4), "n": n,
            "chance": chance, "diff_from_chance": round(acc - chance, 4) if chance is not None else None,
            "wilson95": [round(lo, 4), round(hi, 4)]}


def classify_capability(n: int, raw_k: int, oracle_rate: float | None,
                        legal_best: float | None, chance: float | None,
                        canary_pass: bool | None) -> dict:
    """SubstrateCapability from headroom evidence. canary_pass=False caps at WEAK."""
    raw_lo, raw_hi = wilson(raw_k, n)
    notes: list[str] = []
    if canary_pass is False:
        notes.append("PRIMITIVE_CANARY_FAILED: binding readouts uninterpretable until realization works")
        cap = "WEAK"
    elif raw_lo >= CAP_STRONG_LO:
        cap = "STRONG"
    elif n >= CAP_PARTIAL_N_MIN and raw_lo >= 0.05 and raw_hi <= 0.90:
        cap = "PARTIAL"
    elif raw_hi < 0.05 or (oracle_rate is not None and oracle_rate < ORACLE_FLOOR_V2 and raw_hi < 0.35):
        cap = "INSUFFICIENT"
    else:
        cap = "WEAK"
        notes.append("small-N or borderline: calibrate more before claiming PARTIAL")
    return {"capability": cap, "raw_wilson95": [round(raw_lo, 4), round(raw_hi, 4)],
            "oracle_rate": oracle_rate, "legal_best": legal_best,
            "chance_report": chance_report(raw_k, n, chance,
                                           "K_WAY_CANDIDATE_SELECTION" if chance is not None
                                           else "RAW_FREE_GENERATION"),
            "notes": notes}


def assess_identifiability(n: int, n_failures: int, n_discordant: int,
                           oracle_rate: float | None, chance: float | None,
                           oracle_chance: float | None = None,
                           min_failures: int = 5, min_discordant: int = 5) -> dict:
    """ExperimentIdentifiability with Wilson-aware floor/ceiling checks.

    `chance` applies to K-way selection metrics only (null for free
    generation). `oracle_chance` is an EXPLICIT generative null for the
    oracle ceiling (e.g. 1/k on k-way-shaped binding tasks); it is reported
    as an explicit null, never laundered into generation claims.
    """
    raw_k = n - n_failures
    raw_lo, raw_hi = wilson(raw_k, n)
    flags = {
        "raw_floor": raw_hi < 0.05,
        "raw_ceiling": raw_lo >= CEILING_V2,
        "oracle_floor": oracle_rate is not None and oracle_rate < ORACLE_FLOOR_V2,
        "oracle_near_chance": (oracle_rate is not None and oracle_chance is not None
                               and oracle_rate <= oracle_chance + 0.05),
        "enough_failures": n_failures >= min_failures,
        "enough_repairs": n_discordant >= min_discordant,
    }
    if flags["raw_floor"] or (flags["oracle_floor"] and n_failures >= n - 1):
        status, reason = "NOT_IDENTIFIABLE", "EXPERIMENT_NOT_IDENTIFIABLE: floor substrate"
    elif flags["raw_ceiling"]:
        status, reason = "NOT_IDENTIFIABLE", "CEILING_LIMITED"
    elif not flags["enough_failures"] or not flags["enough_repairs"]:
        status, reason = "MARGINAL", "INTERVENTION_SPARSE at this N"
    elif flags["oracle_floor"]:
        status, reason = "MARGINAL", "ORACLE_LIMITED: weak ceiling weakens attribution"
    else:
        status, reason = "IDENTIFIABLE", "failures and repairs coexist with oracle headroom"
    return {"identifiability": status, "reason": reason, "flags": flags,
            "raw_wilson95": [round(raw_lo, 4), round(raw_hi, 4)]}
