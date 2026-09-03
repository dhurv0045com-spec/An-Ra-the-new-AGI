"""Readiness v2 decision pipeline (Mission 5/6/11/12/13/14/17).

STAGE A — CALIBRATE (cheap, small N): locates candidate regions.
  Allowed outcomes: FLOOR | CEILING | CANDIDATE_PARTIAL_REGION | CALIBRATION_UNSTABLE.
  NEVER emits READY.
STAGE B — QUALIFY (frozen protocol, larger N, replication evidence):
  only stage that may emit READY_FOR_BINDING_CAUSAL_RESEARCH (scoped).

READY additionally requires: primitive canaries pass, frontier stable,
legal-intervention headroom > 0, response diversity adequate, power
sufficient, chance-aware QV-lite not below chance. Missing required input ->
READINESS_UNRESOLVED (fail closed, never optimistic).
"""

from __future__ import annotations

from .status import assess_identifiability, classify_capability, wilson


def legal_headroom(raw_rate: float, legal_best: float | None,
                   oracle_rate: float | None) -> dict:
    if legal_best is None or oracle_rate is None:
        return {"status": "UNKNOWN", "legal_gap": None, "oracle_gap": None,
                "note": "missing legal/oracle arm: cannot assess headroom"}
    return {"status": "ORACLE_ELICITABLE_ONLY" if legal_best <= raw_rate + 1e-9 else "CONNECTOR_HEADROOM",
            "legal_gap": round(legal_best - raw_rate, 4),
            "oracle_gap": round(oracle_rate - legal_best, 4)}


def response_diversity(signatures: list[tuple] | None,
                       discord_by_intervention: dict[str, int] | None) -> dict:
    if signatures is None or discord_by_intervention is None:
        return {"status": "UNKNOWN", "note": "missing response profiles"}
    uniq = len(set(signatures))
    strong = sum(1 for v in discord_by_intervention.values() if v >= 3)
    ok = uniq >= 3 and strong >= 2
    return {"status": "ADEQUATE" if ok else "SPARSE",
            "unique_signatures": uniq, "strong_interventions": strong}


def x0_permitted(capability: str, canary_ok: bool | None, legal_repair_lo: float,
                 diversity: str, n: int, regime_frozen: bool) -> dict:
    req = {"capability": capability, "canary_ok": canary_ok,
           "legal_repair_lo": legal_repair_lo, "diversity": diversity,
           "n": n, "regime_frozen": regime_frozen}
    missing = [k for k, v in req.items() if v is None]
    if missing:
        return {"permitted": False, "reason": f"missing: {missing}"}
    ok = (capability in ("PARTIAL", "STRONG") and canary_ok is True
          and legal_repair_lo >= 0.10 and diversity == "ADEQUATE"
          and n >= 50 and regime_frozen is True)
    return {"permitted": ok, "reason": "all X0 gates pass" if ok else f"blocked: {req}"}


def x1_permitted(x0_predictive: bool | None, x0_replicated: bool | None) -> dict:
    if x0_predictive is None or x0_replicated is None:
        return {"permitted": False, "reason": "missing X0 evidence"}
    ok = x0_predictive is True and x0_replicated is True
    return {"permitted": ok, "reason": "X0 predictive + replicated" if ok else "X0 not earned"}


def power_gate(required_n: int, budget_n: int | None) -> dict:
    if required_n < 0:
        return {"status": "NO_EFFECT", "note": "no positive paired effect to power"}
    if budget_n is None:
        return {"status": "UNKNOWN", "note": "budget unstated"}
    return {"status": "SUFFICIENT" if budget_n >= required_n else "INSUFFICIENT_POWER",
            "required_n": required_n, "budget_n": budget_n}


REQUIRED_FOR_READY = ("capability", "identifiability", "canary_ok", "frontier",
                      "legal_best", "diversity", "power", "protocol_sha",
                      "replication_ok", "n")

LEGAL_GAP_MIN = 0.05  # v0 default: legal assistance must add >=5pp over raw


def decide_readiness(stage: str, capability: dict, identifiability: dict,
                     canary_ok: bool | None, frontier_verdict: str | None,
                     legal_gap: float | None, diversity_status: str | None,
                     power_status: str | None, protocol_sha: str | None,
                     replication_ok: bool | None, n: int,
                     qv_lite_vs_chance: float | None = None) -> dict:
    """Fail-closed composition. READY only from stage B with full evidence."""
    provided = {"capability": capability, "identifiability": identifiability,
                "canary_ok": canary_ok, "frontier": frontier_verdict,
                "legal_best": legal_gap, "diversity": diversity_status,
                "power": power_status, "protocol_sha": protocol_sha,
                "replication_ok": replication_ok, "n": n}
    missing = [k for k in REQUIRED_FOR_READY if provided.get(k) is None]
    if stage not in ("calibrate", "qualify"):
        return {"readiness": "READINESS_UNRESOLVED", "reason": f"unknown stage {stage}"}
    if missing:
        return {"readiness": "READINESS_UNRESOLVED",
                "reason": f"missing required inputs (fail closed): {missing}"}
    if stage == "calibrate":
        if identifiability["identifiability"] == "NOT_IDENTIFIABLE":
            return {"readiness": "NOT_READY", "reason": identifiability["reason"]}
        if frontier_verdict == "CALIBRATION_UNSTABLE":
            return {"readiness": "CALIBRATION_REQUIRED",
                    "reason": "frontier unstable at calibration N; widen N before regions"}
        if capability["capability"] in ("PARTIAL", "STRONG"):
            return {"readiness": "CALIBRATION_REQUIRED",
                    "reason": "CANDIDATE_PARTIAL_REGION: freeze protocol + qualify before READY"}
        return {"readiness": "NOT_READY", "reason": f"capability={capability['capability']}"}
    # stage B
    if canary_ok is not True:
        return {"readiness": "NOT_READY", "reason": "PRIMITIVE_CANARY_FAILED or unknown"}
    if frontier_verdict != "STABLE":
        return {"readiness": "CALIBRATION_REQUIRED", "reason": f"frontier={frontier_verdict}"}
    if capability["capability"] not in ("PARTIAL", "STRONG"):
        return {"readiness": "NOT_READY", "reason": f"capability={capability['capability']}"}
    if identifiability["identifiability"] != "IDENTIFIABLE":
        return {"readiness": "NOT_READY", "reason": identifiability["reason"]}
    if legal_gap is not None and legal_gap <= 0:
        return {"readiness": "NOT_READY", "reason": "ORACLE_ELICITABLE_ONLY: no legal headroom"}
    if legal_gap is not None and legal_gap < LEGAL_GAP_MIN:
        return {"readiness": "NOT_READY",
                "reason": "ORACLE_ELICITABLE_ONLY: legal headroom negligible (<5pp)"}
    if diversity_status != "ADEQUATE":
        return {"readiness": "CALIBRATION_REQUIRED", "reason": "response diversity sparse"}
    if power_status == "INSUFFICIENT_POWER":
        return {"readiness": "CALIBRATION_REQUIRED", "reason": "underpowered for primary comparison"}
    if qv_lite_vs_chance is not None and qv_lite_vs_chance < 0:
        return {"readiness": "CALIBRATION_REQUIRED",
                "reason": "QV-lite below chance: no query-conditioned signal to model"}
    if replication_ok is not True:
        return {"readiness": "CALIBRATION_REQUIRED", "reason": "second-seed replication pending"}
    return {"readiness": "READY_SCOPED",
            "reason": "qualified binding regime: PARTIAL capability + identifiable + legal headroom + replicated"}
