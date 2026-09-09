"""CYR-GPU-009 focused one-shot Colab contract.

CYR-GPU-009 exists because real operator calibration showed that CYR-GPU-008's
all-or-nothing minimum campaign could not fit the assigned Colab GPU. No 008
scientific training was executed. 009 copies the useful execution discipline
from Arkenstone Discovery V7: allocate a fixed wall budget, run the highest
information matched unit first, preserve partial evidence, and never reject a
healthy GPU merely because the whole ideal campaign cannot be guaranteed in
advance.

The scientific scope is deliberately narrower: two independent T2 acquisition
parents and one exact same-parent matched retention pair per qualified parent:
HIGH_CONTINUE versus LOW_CONTINUE. This is the cleanest direct replication of
the strongest ARK-007R/ARK-011 retention signal. Transfer, fixed-time, and
hysteretic arms are deferred rather than starving the core matched comparison.
"""
from __future__ import annotations

import math
from statistics import mean
from typing import Any, Mapping

from v5_experiments import cyr_gpu006 as base

CYR9_ID = "CYR-GPU-009"
CYR9_PARENT_SEEDS = (707, 808)
CYR9_ARMS = ("HIGH_CONTINUE", "LOW_CONTINUE")
CYR9_WALL_MINUTES = 165.0
CYR9_PACKAGING_RESERVE_MINUTES = 8.0
CYR9_ACQUISITION_TARGET_TOKENS = 2_000_000
CYR9_CONTINUATION_TARGET_TOKENS = 500_000
CYR9_ACQ_EVAL_INTERVAL_TOKENS = 400_000
CYR9_CONT_EVAL_INTERVAL_TOKENS = 125_000
CYR9_MIN_LAUNCH_MINUTES_PER_PARENT = 45.0

# Reuse the exact V5/data authorities already audited in 006-008.
proxy_registry = base.proxy_registry
assert_proxy_in_registry = base.assert_proxy_in_registry
render_t2_worlds = base.render_t2_worlds
build_data_manifest = base.build_data_manifest
assert_manifest_sha = base.assert_manifest_sha
commutation_audit = base.commutation_audit
build_future_stream = base.build_future_stream
assert_future_tail_equality = base.assert_future_tail_equality
ARKENSTONE_AUDITED_SHA = "5db97c4aaf3ecd1da78f32119f0d979238184dd7"
ARKENSTONE_TECHNIQUE = "Discovery V7 fixed-wall progressive matched-unit scheduling"


def _predicted_seconds(receipt: Mapping[str, Any]) -> float:
    """Estimate the complete two-parent focused campaign from hardware only."""
    if receipt.get("status") != "PASS":
        return math.inf
    tps = float(receipt.get("training_real_tokens_per_sec", 0.0))
    eps = float(receipt.get("generation_examples_per_sec", 0.0))
    if tps <= 0 or eps <= 0:
        return math.inf
    train_tokens = len(CYR9_PARENT_SEEDS) * (
        CYR9_ACQUISITION_TARGET_TOKENS
        + len(CYR9_ARMS) * CYR9_CONTINUATION_TARGET_TOKENS
    )
    acq_events = len(CYR9_PARENT_SEEDS) * math.ceil(
        CYR9_ACQUISITION_TARGET_TOKENS / CYR9_ACQ_EVAL_INTERVAL_TOKENS
    )
    cont_events = len(CYR9_PARENT_SEEDS) * len(CYR9_ARMS) * math.ceil(
        CYR9_CONTINUATION_TARGET_TOKENS / CYR9_CONT_EVAL_INTERVAL_TOKENS
    )
    # Acquisition evaluates 96 controller + 16 train-probe examples. Each
    # retention observation evaluates 96 controller + 112 measurement rows.
    eval_examples = acq_events * 112 + cont_events * 208
    # 8% runtime contingency plus the fixed packaging reserve.
    science = train_tokens / tps + eval_examples / eps
    return science * 1.08 + CYR9_PACKAGING_RESERVE_MINUTES * 60.0


def resolve_from_calibrations(
    calibrations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select a proxy without an all-or-nothing feasibility failure.

    Prefer the largest non-TINY proxy that is predicted to complete the focused
    two-parent experiment inside 165 minutes. If none can, choose the fastest
    passing proxy (normally TINY on free Colab) and let the progressive runner
    return as much matched evidence as the fixed wall allows. Scientific
    outcomes never enter this decision.
    """
    passing = {
        name: receipt
        for name, receipt in calibrations.items()
        if receipt.get("status") == "PASS"
        and float(receipt.get("training_real_tokens_per_sec", 0.0)) > 0
        and float(receipt.get("generation_examples_per_sec", 0.0)) > 0
    }
    if not passing:
        raise ValueError("no calibrated CUDA proxy passed the smoke/calibration gate")

    hard_seconds = CYR9_WALL_MINUTES * 60.0
    chosen = None
    for name in ("MIDI", "MICRO", "RESEARCH_SMALL"):
        if name in passing and _predicted_seconds(passing[name]) <= hard_seconds:
            chosen = name
            break
    if chosen is None:
        # Do not fail on a slow but healthy GPU. Use the proxy with the highest
        # measured useful-token throughput and preserve a strict claim ceiling.
        chosen = max(
            passing,
            key=lambda name: float(passing[name]["training_real_tokens_per_sec"]),
        )

    receipt = passing[chosen]
    predicted = _predicted_seconds(receipt)
    return {
        "schema": "anra-cyr-gpu009-resolved/v1",
        "experiment": CYR9_ID,
        "mode": "full",
        "proxy": chosen,
        "parent_seeds": list(CYR9_PARENT_SEEDS),
        "arms": list(CYR9_ARMS),
        "target_actual_tokens_acquisition": CYR9_ACQUISITION_TARGET_TOKENS,
        "target_actual_tokens_continuation": CYR9_CONTINUATION_TARGET_TOKENS,
        "acquisition_eval_interval_tokens": CYR9_ACQ_EVAL_INTERVAL_TOKENS,
        "continuation_eval_interval_tokens": CYR9_CONT_EVAL_INTERVAL_TOKENS,
        "wall_budget_minutes": CYR9_WALL_MINUTES,
        "packaging_reserve_minutes": CYR9_PACKAGING_RESERVE_MINUTES,
        "minimum_parent_launch_minutes": CYR9_MIN_LAUNCH_MINUTES_PER_PARENT,
        "predicted_complete_seconds": round(predicted, 1),
        "predicted_complete_fit": bool(predicted <= hard_seconds),
        "training_real_tokens_per_sec": float(receipt["training_real_tokens_per_sec"]),
        "generation_examples_per_sec": float(receipt["generation_examples_per_sec"]),
        "claim_ceiling": (
            "TWO_PARENT_FOCUSED_RETENTION_DEVELOPMENT"
            if chosen != "TINY"
            else "TINY_TWO_PARENT_RETENTION_DEVELOPMENT_ONLY"
        ),
        "research_candidate_possible": False,
        "production_promotion_authorized": False,
        "rule": (
            "hardware-only proxy selection; fixed-wall progressive execution; "
            "never fail solely because full ideal scope is predicted not to fit"
        ),
    }


def validate_resolved(resolved: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "experiment", "mode", "proxy", "parent_seeds", "arms",
        "target_actual_tokens_acquisition", "target_actual_tokens_continuation",
        "acquisition_eval_interval_tokens", "continuation_eval_interval_tokens",
        "wall_budget_minutes", "packaging_reserve_minutes",
        "minimum_parent_launch_minutes", "claim_ceiling",
        "research_candidate_possible", "production_promotion_authorized",
    }
    missing = sorted(required - set(resolved))
    if missing:
        raise ValueError(f"CYR-GPU-009 resolution missing {missing}")
    if resolved["experiment"] != CYR9_ID or resolved["mode"] != "full":
        raise ValueError("CYR-GPU-009 resolution identity/mode drift")
    if tuple(int(x) for x in resolved["parent_seeds"]) != CYR9_PARENT_SEEDS:
        raise ValueError("CYR-GPU-009 parent seed drift")
    if tuple(resolved["arms"]) != CYR9_ARMS:
        raise ValueError("CYR-GPU-009 arm drift")
    if float(resolved["wall_budget_minutes"]) > CYR9_WALL_MINUTES:
        raise ValueError("CYR-GPU-009 wall budget exceeds frozen maximum")
    if resolved.get("research_candidate_possible"):
        raise ValueError("focused 009 GPU result cannot be a production research candidate")
    if resolved.get("production_promotion_authorized"):
        raise ValueError("CYR-GPU-009 can never authorize production promotion")
    return dict(resolved)


def decide(parent_runs: list[Mapping[str, Any]], resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Paired replicated retention verdict; incomplete matched units never win."""
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for parent in parent_runs:
        seed = parent.get("seed")
        reasons: list[str] = []
        if parent.get("parent_status") != "G90_CONFIRMED":
            reasons.append("parent not G90_CONFIRMED")
        if not parent.get("parent_equivalence", {}).get("identical", False):
            reasons.append("parent fork bytes not identical")
        if not parent.get("future_tail", {}).get("identical", False):
            reasons.append("future minibatches not identical")
        arms = parent.get("arms", {})
        high = arms.get("HIGH_CONTINUE")
        low = arms.get("LOW_CONTINUE")
        for label, receipt in (("HIGH_CONTINUE", high), ("LOW_CONTINUE", low)):
            if not receipt:
                reasons.append(f"missing {label}")
            elif receipt.get("status") != "COMPLETE":
                reasons.append(f"{label} incomplete")
            elif not receipt.get("redteam_pass", False):
                reasons.append(f"{label} redteam failed")
        if reasons:
            rejected.append({"seed": seed, "reasons": reasons})
            continue
        diff = float(low["retention_ret90"]) - float(high["retention_ret90"])
        valid.append({
            "seed": int(seed),
            "high_ret90": float(high["retention_ret90"]),
            "low_ret90": float(low["retention_ret90"]),
            "low_minus_high": diff,
        })

    if len(valid) >= 2:
        diffs = [row["low_minus_high"] for row in valid]
        if min(diffs) >= 0.05 and mean(diffs) >= 0.10:
            verdict = "REPLICATED_LOW_RETENTION_PROTECTION"
        elif max(diffs) <= 0.0:
            verdict = "REPLICATED_NO_LOW_ADVANTAGE"
        else:
            verdict = "REPLICATED_MIXED_OR_SMALL_EFFECT"
    elif len(valid) == 1:
        verdict = "SINGLE_PARENT_DEVELOPMENT_SIGNAL"
    else:
        verdict = "INCONCLUSIVE_NO_COMPLETE_MATCHED_PAIR"
    return {
        "schema": "anra-cyr-gpu009-decision/v1",
        "experiment": CYR9_ID,
        "verdict": verdict,
        "valid_matched_parents": valid,
        "rejected_parents": rejected,
        "claim_ceiling": resolved["claim_ceiling"],
        "research_candidate": False,
        "production_promotion_authorized": False,
        "pre500m_authorized": False,
        "training_500m_authorized": False,
    }
