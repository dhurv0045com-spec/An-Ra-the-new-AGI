"""Experiment identifiability: floor/ceiling detection + power (Mission 7/8/29/30).

Answers BEFORE interpretation: can this experiment distinguish its
hypotheses on this checkpoint? Emits RUN / DO_NOT_RUN with reasons.

Threshold defaults are v0, calibrated on the weak V4 substrate. They are
documented estimates, not universal constants.
"""

from __future__ import annotations

import math

FLOOR_ORACLE_MAX = 0.40   # oracle below this -> ORACLE_LIMITED
CEILING_RAW_MIN = 0.95    # raw above this -> CEILING_LIMITED
PARTIAL_RAW_LO = 0.05
PARTIAL_RAW_HI = 0.85
PARTIAL_LIFT_MIN = 0.20   # oracle - raw
MIN_FAILURES = 5
MIN_DISCORDANT = 5


def check_identifiability(n_tasks: int, n_raw_pass: int, n_failures: int,
                          n_oracle_repair: int, n_discordant: int,
                          chance: float = 0.0) -> dict:
    raw = n_raw_pass / max(n_tasks, 1)
    oracle = ((n_tasks - n_failures) + n_oracle_repair) / max(n_tasks, 1)
    flags: dict[str, bool] = {
        "raw_floor": raw < PARTIAL_RAW_LO,
        "raw_ceiling": raw >= CEILING_RAW_MIN,
        "oracle_floor": oracle < FLOOR_ORACLE_MAX,
        "oracle_near_chance": oracle <= chance + 0.05 if chance else False,
        "enough_failures": n_failures >= MIN_FAILURES,
        "enough_repairs": n_discordant >= MIN_DISCORDANT,
    }
    if flags["raw_floor"] and flags["oracle_floor"]:
        decision, reason = "DO_NOT_RUN", ("EXPERIMENT_NOT_IDENTIFIABLE: raw at floor "
                                          "and oracle weak; substrate lacks capability.")
    elif flags["raw_ceiling"]:
        decision, reason = "DO_NOT_RUN", "CEILING_LIMITED: too few failures to diagnose."
    elif not flags["enough_failures"] or not flags["enough_repairs"]:
        decision, reason = "DO_NOT_RUN", "INTERVENTION_SPARSE: pilot shows too few failures/repairs."
    else:
        decision, reason = "RUN", "identifiable: failures and repairs coexist with oracle headroom."
    adequacy = ("FLOOR_LIMITED" if flags["raw_floor"] else
                "CEILING_LIMITED" if flags["raw_ceiling"] else
                "ORACLE_LIMITED" if flags["oracle_floor"] else
                "INTERVENTION_SPARSE" if not flags["enough_repairs"] else "ADEQUATE")
    return {"raw_rate": round(raw, 4), "oracle_rate": round(oracle, 4),
            "flags": flags, "substrate_adequacy": adequacy,
            "decision": decision, "reason": reason}


def required_n_mcnemar(p01: float, p10: float, alpha: float = 0.05,
                       power: float = 0.8) -> int:
    """Approx N for McNemar (normal approx on paired difference; estimate only)."""
    d = p01 - p10
    if d <= 0:
        return -1
    s2 = p01 + p10 - d * d
    z = 1.96 + 0.84  # alpha .05 two-sided, power .8
    n = math.ceil((z * z * s2) / (d * d))
    return max(n, 10)
