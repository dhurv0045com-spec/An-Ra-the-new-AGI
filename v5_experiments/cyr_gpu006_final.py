"""Final unfrozen CYR-GPU-006 contract candidate.

This is the execution authority for the still-unexecuted CYR-GPU-006 campaign.
It incorporates the live Arkenstone Discovery V6 boundary conditions:

* ARK-011 supports state-conditional HIGH->LOW protection on same-task Micro T2.
* ARK-012 does not identify a universal switch threshold.
* ARK-013 shows LOW alone does not solve no-replay cross-task interference.
* ARK-014 shows robust non-arithmetic binding needs order augmentation and its
  HIGH/LOW retention transfer screen was low-event/inconclusive.

The transfer/plasticity stage therefore compares equal-age, equal-exposure
retention states from the SAME acquired parent: HYSTERETIC_HIGH_LOW versus
LOW_CONTINUE. Both are then moved to HIGH LR on the same robust-binding +
fixed-replay stream. This removes the previous pre-continuation-vs-post-
continuation age/exposure confound and directly asks whether the adaptive state
keeps useful plasticity relative to the near-freezing control.
"""
from __future__ import annotations

import math
from statistics import mean
from typing import Any, Mapping

from v5_experiments import cyr_gpu006 as base

CYR6_ID = base.CYR6_ID
CYR6_PARENT_SEEDS = base.CYR6_PARENT_SEEDS
CYR6_ARMS = base.CYR6_ARMS
CYR6_LRS = base.CYR6_LRS
CYR6_HYSTERESIS = base.CYR6_HYSTERESIS
CYR6_MIN_QUALIFIED_PARENTS = base.CYR6_MIN_QUALIFIED_PARENTS
CYR6_MIN_RET90_MEAN_MARGIN = base.CYR6_MIN_RET90_MEAN_MARGIN
CYR6_MIN_PER_PARENT_SUPPORT = base.CYR6_MIN_PER_PARENT_SUPPORT
CYR6_WALL_TARGET_MINUTES = 135.0
CYR6_WALL_HARD_MINUTES = 170.0
CYR6_PACKAGING_RESERVE_MINUTES = 8.0
CYR6_ACQ_MIN_TOKENS = 2_000_000
CYR6_ACQ_TARGET_TOKENS = 4_000_000
CYR6_FORK_MIN_TOKENS = 500_000
CYR6_FORK_TARGET_TOKENS = 2_000_000
CYR6_ACQ_EVAL_INTERVAL_TOKENS = 250_000
CYR6_FORK_EVAL_INTERVAL_TOKENS = 100_000

ARKENSTONE_AUDITED_SHA = "6acd9dcbdd28d00f387ffcd004253a813aca4b66"
ARKENSTONE_DISCOVERY_V6_BUNDLE_SHA256 = (
    "1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15"
)

# Prospective, evidence-derived transfer comparison. Neither arm is selected
# from CYR arithmetic outcomes, so the transfer stage is not post-hoc.
CYR6_TRANSFER_CANDIDATE = "HYSTERETIC_HIGH_LOW"
CYR6_TRANSFER_COMPARATOR = "LOW_CONTINUE"
CYR6_TRANSFER_MODE = "REPLAY_1_IN_10_ROWS"
CYR6_TRANSFER_MIN_PARENTS = 2
CYR6_TRANSFER_TARGET_PARENTS = 3
CYR6_TRANSFER_MIN_TOKENS = 500_000
CYR6_TRANSFER_TARGET_TOKENS = 1_000_000
CYR6_TRANSFER_EVAL_INTERVAL_TOKENS = 100_000
CYR6_TRANSFER_ROBUST_THRESHOLDS = {
    "canonical": 0.90,
    "order_only": 0.85,
    "query_order": 0.85,
}
CYR6_TRANSFER_CONFIRMATIONS = 3
CYR6_TRANSFER_SOURCE_MIN_T2 = 0.90
CYR6_TRANSFER_MAX_PLASTICITY_SLOWDOWN = 0.25
CYR6_TRANSFER_FINAL_TOLERANCE = 0.05
CYR6_TRANSFER_ADVANTAGE_SPEED_FRACTION = 0.10
CYR6_TRANSFER_ADVANTAGE_RETENTION = 0.05

proxy_registry = base.proxy_registry
assert_proxy_in_registry = base.assert_proxy_in_registry
render_t2_worlds = base.render_t2_worlds
build_data_manifest = base.build_data_manifest
assert_manifest_sha = base.assert_manifest_sha
commutation_audit = base.commutation_audit
build_future_stream = base.build_future_stream
assert_future_tail_equality = base.assert_future_tail_equality
fixed_time_switch_point = base.fixed_time_switch_point
lr_for_token = base.lr_for_token
arm_order = base.arm_order
decide_retention = base.decide_campaign
CYR6_TRANSFER_FAMILY = base.CYR6_TRANSFER_FAMILY


def estimate_stage_seconds(
    *,
    training_tokens_per_sec: float,
    eval_examples_per_sec: float,
    acquisition_tokens: int,
    continuation_tokens: int,
    transfer_tokens: int,
    transfer_parents: int = CYR6_TRANSFER_TARGET_PARENTS,
) -> dict[str, float]:
    """Hardware-only stage cost model including candidate-free evaluation."""
    if training_tokens_per_sec <= 0 or eval_examples_per_sec <= 0:
        return {"acquisition": math.inf, "retention": math.inf, "transfer": math.inf}

    acq_train = len(CYR6_PARENT_SEEDS) * acquisition_tokens / training_tokens_per_sec
    acq_events = len(CYR6_PARENT_SEEDS) * math.ceil(
        acquisition_tokens / CYR6_ACQ_EVAL_INTERVAL_TOKENS
    )
    acq_eval = acq_events * (96 + 16) / eval_examples_per_sec

    retention_train = (
        len(CYR6_PARENT_SEEDS)
        * len(CYR6_ARMS)
        * continuation_tokens
        / training_tokens_per_sec
    )
    retention_events = len(CYR6_PARENT_SEEDS) * len(CYR6_ARMS) * math.ceil(
        continuation_tokens / CYR6_FORK_EVAL_INTERVAL_TOKENS
    )
    retention_eval = retention_events * (96 + 112) / eval_examples_per_sec

    # Two full-state transfer arms (adaptive candidate and LOW comparator) per
    # independent source parent. Each eval measures 4 binding variants on 32
    # control worlds plus 112 old-T2 measurement rows. Final sealed adds 4x32
    # binding examples plus another old-T2 measurement.
    transfer_runs = transfer_parents * 2
    transfer_train = transfer_runs * transfer_tokens / training_tokens_per_sec
    transfer_events = transfer_runs * math.ceil(
        transfer_tokens / CYR6_TRANSFER_EVAL_INTERVAL_TOKENS
    )
    transfer_eval = (
        transfer_events * (4 * 32 + 112)
        + transfer_runs * (4 * 32 + 112)
    ) / eval_examples_per_sec

    return {
        "acquisition": acq_train + acq_eval,
        "retention": retention_train + retention_eval,
        "transfer": transfer_train + transfer_eval,
    }


def estimate_wall_seconds(**kwargs: Any) -> float:
    stage = estimate_stage_seconds(**kwargs)
    return sum(stage.values()) + CYR6_PACKAGING_RESERVE_MINUTES * 60.0


def _resolved_for(
    *, name: str, receipt: Mapping[str, Any]
) -> dict[str, Any] | None:
    if receipt.get("status") != "PASS":
        return None
    train_tps = float(receipt["training_real_tokens_per_sec"])
    eval_eps = float(receipt["generation_examples_per_sec"])

    minimum = estimate_wall_seconds(
        training_tokens_per_sec=train_tps,
        eval_examples_per_sec=eval_eps,
        acquisition_tokens=CYR6_ACQ_MIN_TOKENS,
        continuation_tokens=CYR6_FORK_MIN_TOKENS,
        transfer_tokens=CYR6_TRANSFER_MIN_TOKENS,
    )
    if minimum > (CYR6_WALL_HARD_MINUTES - 2.0) * 60.0:
        return None

    lo, hi = 1.0, 4.0
    for _ in range(24):
        factor = (lo + hi) / 2.0
        acq = min(CYR6_ACQ_TARGET_TOKENS, int(CYR6_ACQ_MIN_TOKENS * factor))
        fork = min(CYR6_FORK_TARGET_TOKENS, int(CYR6_FORK_MIN_TOKENS * factor))
        transfer = min(
            CYR6_TRANSFER_TARGET_TOKENS,
            int(CYR6_TRANSFER_MIN_TOKENS * factor),
        )
        predicted = estimate_wall_seconds(
            training_tokens_per_sec=train_tps,
            eval_examples_per_sec=eval_eps,
            acquisition_tokens=acq,
            continuation_tokens=fork,
            transfer_tokens=transfer,
        )
        if predicted <= CYR6_WALL_TARGET_MINUTES * 60.0:
            lo = factor
        else:
            hi = factor

    factor = lo
    acq = min(CYR6_ACQ_TARGET_TOKENS, int(CYR6_ACQ_MIN_TOKENS * factor))
    fork = min(CYR6_FORK_TARGET_TOKENS, int(CYR6_FORK_MIN_TOKENS * factor))
    transfer = min(
        CYR6_TRANSFER_TARGET_TOKENS,
        int(CYR6_TRANSFER_MIN_TOKENS * factor),
    )
    stages = estimate_stage_seconds(
        training_tokens_per_sec=train_tps,
        eval_examples_per_sec=eval_eps,
        acquisition_tokens=acq,
        continuation_tokens=fork,
        transfer_tokens=transfer,
    )
    predicted = sum(stages.values()) + CYR6_PACKAGING_RESERVE_MINUTES * 60.0

    # Stage deadlines use prospective measured costs with a 15% per-stage
    # contingency. If contingencies exceed the hard science window they are
    # scaled proportionally; no scientific outcome enters this calculation.
    raw_budgets = {name_: max(seconds * 1.15, 30.0) for name_, seconds in stages.items()}
    science_limit = (
        CYR6_WALL_HARD_MINUTES - CYR6_PACKAGING_RESERVE_MINUTES
    ) * 60.0
    total_budget = sum(raw_budgets.values())
    scale = min(1.0, science_limit / max(total_budget, 1.0))
    stage_budgets = {name_: round(seconds * scale, 1) for name_, seconds in raw_budgets.items()}

    return {
        "schema": "anra-cyr-gpu006-resolved/final-v1",
        "mode": "full",
        "proxy": name,
        "parents": len(CYR6_PARENT_SEEDS),
        "target_actual_tokens_acquisition": acq,
        "target_actual_tokens_continuation": fork,
        "acquisition_eval_interval_tokens": CYR6_ACQ_EVAL_INTERVAL_TOKENS,
        "continuation_eval_interval_tokens": CYR6_FORK_EVAL_INTERVAL_TOKENS,
        "transfer_enabled": True,
        "transfer_candidate": CYR6_TRANSFER_CANDIDATE,
        "transfer_comparator": CYR6_TRANSFER_COMPARATOR,
        "transfer_mode": CYR6_TRANSFER_MODE,
        "transfer_target_parents": CYR6_TRANSFER_TARGET_PARENTS,
        "transfer_min_parents": CYR6_TRANSFER_MIN_PARENTS,
        "transfer_target_actual_tokens": transfer,
        "transfer_eval_interval_tokens": CYR6_TRANSFER_EVAL_INTERVAL_TOKENS,
        "wall_budget_minutes": CYR6_WALL_HARD_MINUTES,
        "predicted_wall_seconds": round(predicted, 1),
        "predicted_stage_seconds": {k: round(v, 1) for k, v in stages.items()},
        "stage_budgets_seconds": stage_budgets,
        "calibration_proxy": name,
        "training_real_tokens_per_sec": train_tps,
        "generation_examples_per_sec": eval_eps,
        "rule": "hardware-only; loss/accuracy/treatment outcomes forbidden from resolution",
        "arkenstone_audited_sha": ARKENSTONE_AUDITED_SHA,
    }


def resolve_from_calibrations(
    calibrations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Largest real Cymek V5 proxy that preserves the full causal design."""
    for name in ("MIDI", "MICRO", "RESEARCH_SMALL"):
        receipt = calibrations.get(name)
        if receipt:
            resolved = _resolved_for(name=name, receipt=receipt)
            if resolved is not None:
                return resolved
    raise ValueError(
        "no calibrated proxy affords 3 acquisition parents, four matched retention "
        "forks, and the equal-age robust-binding plasticity comparison inside 170 minutes"
    )


def validate_resolved(resolved: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "mode", "proxy", "parents",
        "target_actual_tokens_acquisition", "target_actual_tokens_continuation",
        "acquisition_eval_interval_tokens", "continuation_eval_interval_tokens",
        "transfer_candidate", "transfer_comparator", "transfer_mode",
        "transfer_target_parents", "transfer_min_parents",
        "transfer_target_actual_tokens", "transfer_eval_interval_tokens",
        "wall_budget_minutes", "predicted_wall_seconds", "stage_budgets_seconds",
    }
    missing = sorted(required - set(resolved))
    if missing:
        raise ValueError(f"resolved campaign missing {missing}")
    if resolved["mode"] != "full":
        raise ValueError("CYR-GPU-006 scientific execution requires mode=full")
    if int(resolved["parents"]) != len(CYR6_PARENT_SEEDS):
        raise ValueError("resolver must schedule all three acquisition parents")
    if int(resolved["target_actual_tokens_acquisition"]) < CYR6_ACQ_MIN_TOKENS:
        raise ValueError("acquisition dose below preregistered floor")
    if int(resolved["target_actual_tokens_continuation"]) < CYR6_FORK_MIN_TOKENS:
        raise ValueError("continuation dose below preregistered floor")
    if int(resolved["transfer_target_actual_tokens"]) < CYR6_TRANSFER_MIN_TOKENS:
        raise ValueError("transfer dose below preregistered floor")
    if resolved["transfer_candidate"] != CYR6_TRANSFER_CANDIDATE:
        raise ValueError("transfer candidate drift")
    if resolved["transfer_comparator"] != CYR6_TRANSFER_COMPARATOR:
        raise ValueError("transfer comparator drift")
    if resolved["transfer_mode"] != CYR6_TRANSFER_MODE:
        raise ValueError("transfer mode drift")
    if int(resolved["transfer_min_parents"]) != CYR6_TRANSFER_MIN_PARENTS:
        raise ValueError("transfer replication floor drift")
    if int(resolved["transfer_target_parents"]) != CYR6_TRANSFER_TARGET_PARENTS:
        raise ValueError("transfer target-parent count drift")
    budgets = resolved["stage_budgets_seconds"]
    if set(budgets) != {"acquisition", "retention", "transfer"}:
        raise ValueError("stage budgets must cover acquisition/retention/transfer")
    if any(float(value) <= 0 for value in budgets.values()):
        raise ValueError("stage budgets must be positive")
    wall = float(resolved["wall_budget_minutes"])
    if not 60.0 <= wall <= CYR6_WALL_HARD_MINUTES:
        raise ValueError("resolved wall budget outside frozen limits")
    return dict(resolved)


def final_decision(
    parent_runs: list[Mapping[str, Any]],
    *, transfer: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Combine replicated retention with independently preregistered plasticity evidence."""
    retention = decide_retention(parent_runs)
    transfer_ok = bool(
        transfer
        and transfer.get("status") in {
            "REPLICATED_PLASTICITY_NONINFERIOR",
            "REPLICATED_PLASTICITY_ADVANTAGE",
        }
    )
    research_candidate = bool(
        retention.get("verdict") == "REPLICATED_WINNER"
        and retention.get("winner") == CYR6_TRANSFER_CANDIDATE
        and transfer_ok
    )
    return {
        **retention,
        "schema": "anra-cyr-gpu006-decision/final-v1",
        "transfer": transfer,
        "prospective_transfer_candidate": CYR6_TRANSFER_CANDIDATE,
        "prospective_transfer_comparator": CYR6_TRANSFER_COMPARATOR,
        "research_candidate": research_candidate,
        "production_promotion_authorized": False,
        "claim_limit": (
            "GPU V5-proxy evidence only; TPU semantic confirmation and an explicit "
            "promotion decision remain mandatory"
        ),
    }
