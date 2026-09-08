"""CYR-GPU-006 revised pure contract after live Arkenstone Discovery V6 audit.

This module does not change the experiment identity because CYR-GPU-006 has
not been preregistered or executed. It sharpens the still-unfrozen executable
candidate using validated ARK-011..014 evidence:

* ARK-011 supports state-conditional HIGH->LOW protection at Micro T2.
* ARK-012 does not identify an optimal universal switch threshold.
* ARK-013 shows LOW alone does not solve no-replay cross-task interference.
* ARK-014 repairs non-arithmetic binding with order augmentation, while its
  HIGH/LOW retention screen is low-event and inconclusive.

Consequently the non-arithmetic stage below is a prospective plasticity /
interference test of a prechosen adaptive candidate, not a post-hoc transfer
of whichever arithmetic arm happens to win.
"""
from __future__ import annotations

import math
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

# Prospective transfer/plasticity candidate. This is chosen from ARK-011 prior
# evidence, not from CYR-GPU-006 arithmetic outcomes.
CYR6_TRANSFER_CANDIDATE = "HYSTERETIC_HIGH_LOW"
CYR6_TRANSFER_MIN_PARENTS = 2
CYR6_TRANSFER_MIN_TOKENS = 500_000
CYR6_TRANSFER_TARGET_TOKENS = 1_000_000
CYR6_TRANSFER_EVAL_INTERVAL_TOKENS = 100_000
CYR6_TRANSFER_PRIMARY_MODE = "REPLAY_1_IN_10_ROWS"
CYR6_TRANSFER_OPTIONAL_MODE = "PURE_NEW_SKILL"
CYR6_TRANSFER_ROBUST_THRESHOLDS = {
    "canonical": 0.90,
    "order_only": 0.85,
    "query_order": 0.85,
}
CYR6_TRANSFER_CONFIRMATIONS = 3
CYR6_TRANSFER_MAX_PLASTICITY_SLOWDOWN = 0.25
CYR6_TRANSFER_FINAL_TOLERANCE = 0.05

ARKENSTONE_AUDITED_SHA = "6acd9dcbdd28d00f387ffcd004253a813aca4b66"
ARKENSTONE_DISCOVERY_V6_BUNDLE_SHA256 = (
    "1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15"
)

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


def _transfer_training_runs(*, modes: int) -> int:
    # Two full-state sources (pre-retention parent vs prospective adaptive
    # state) across at least two independent acquisition parents.
    return modes * CYR6_TRANSFER_MIN_PARENTS * 2


def estimate_wall_seconds(
    *,
    training_tokens_per_sec: float,
    eval_examples_per_sec: float,
    acquisition_tokens: int,
    continuation_tokens: int,
    transfer_tokens: int,
    transfer_modes: int,
    parents: int = 3,
    arms: int = 4,
) -> float:
    """Prospective hardware-only wall model including free-generation cost."""
    if training_tokens_per_sec <= 0 or eval_examples_per_sec <= 0:
        return math.inf
    training_tokens = parents * acquisition_tokens + parents * arms * continuation_tokens
    training_tokens += _transfer_training_runs(modes=transfer_modes) * transfer_tokens

    acq_events = parents * math.ceil(acquisition_tokens / CYR6_ACQ_EVAL_INTERVAL_TOKENS)
    fork_events = parents * arms * math.ceil(
        continuation_tokens / CYR6_FORK_EVAL_INTERVAL_TOKENS
    )
    # Arithmetic acquisition: controller 96 + train probe 16.
    eval_examples = acq_events * (96 + 16)
    # Arithmetic continuation: controller 96 + measurement 112.
    eval_examples += fork_events * (96 + 112)

    # Transfer evaluates 4 orthogonal binding variants on 32 control worlds
    # plus 112 T2 measurement rows at every interval. Final sealed measurement
    # is added separately and never controls training.
    transfer_events_per_run = math.ceil(
        transfer_tokens / CYR6_TRANSFER_EVAL_INTERVAL_TOKENS
    )
    transfer_runs = _transfer_training_runs(modes=transfer_modes)
    eval_examples += transfer_runs * transfer_events_per_run * (4 * 32 + 112)
    eval_examples += transfer_runs * (4 * 32 + 48)

    return (
        training_tokens / training_tokens_per_sec
        + eval_examples / eval_examples_per_sec
        + CYR6_PACKAGING_RESERVE_MINUTES * 60.0
    )


def _candidate_resolution(
    *,
    name: str,
    receipt: Mapping[str, Any],
    transfer_modes: tuple[str, ...],
) -> dict[str, Any] | None:
    if receipt.get("status") != "PASS":
        return None
    train_tps = float(receipt["training_real_tokens_per_sec"])
    eval_eps = float(receipt["generation_examples_per_sec"])
    mode_count = len(transfer_modes)
    minimum = estimate_wall_seconds(
        training_tokens_per_sec=train_tps,
        eval_examples_per_sec=eval_eps,
        acquisition_tokens=CYR6_ACQ_MIN_TOKENS,
        continuation_tokens=CYR6_FORK_MIN_TOKENS,
        transfer_tokens=CYR6_TRANSFER_MIN_TOKENS,
        transfer_modes=mode_count,
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
            transfer_modes=mode_count,
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
    predicted = estimate_wall_seconds(
        training_tokens_per_sec=train_tps,
        eval_examples_per_sec=eval_eps,
        acquisition_tokens=acq,
        continuation_tokens=fork,
        transfer_tokens=transfer,
        transfer_modes=mode_count,
    )
    return {
        "schema": "anra-cyr-gpu006-resolved/v2",
        "mode": "full",
        "proxy": name,
        "parents": 3,
        "target_actual_tokens_acquisition": acq,
        "target_actual_tokens_continuation": fork,
        "acquisition_eval_interval_tokens": CYR6_ACQ_EVAL_INTERVAL_TOKENS,
        "continuation_eval_interval_tokens": CYR6_FORK_EVAL_INTERVAL_TOKENS,
        "transfer_enabled": True,
        "transfer_candidate": CYR6_TRANSFER_CANDIDATE,
        "transfer_modes": list(transfer_modes),
        "transfer_target_actual_tokens": transfer,
        "transfer_eval_interval_tokens": CYR6_TRANSFER_EVAL_INTERVAL_TOKENS,
        "wall_budget_minutes": CYR6_WALL_HARD_MINUTES,
        "predicted_wall_seconds": round(predicted, 1),
        "calibration_proxy": name,
        "training_real_tokens_per_sec": train_tps,
        "generation_examples_per_sec": eval_eps,
        "rule": "hardware-only; scientific outcomes forbidden from resolution",
        "arkenstone_audited_sha": ARKENSTONE_AUDITED_SHA,
    }


def resolve_from_calibrations(
    calibrations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Largest proxy first; preserve replication and the primary transfer test.

    If hardware can afford both transfer modes, run the pure-new-skill boundary
    plus the replay condition. Otherwise keep only the replay condition because
    ARK-013 already demonstrated the no-replay failure boundary at Micro scale.
    """
    mode_options = (
        (CYR6_TRANSFER_PRIMARY_MODE, CYR6_TRANSFER_OPTIONAL_MODE),
        (CYR6_TRANSFER_PRIMARY_MODE,),
    )
    for name in ("MIDI", "MICRO", "RESEARCH_SMALL"):
        receipt = calibrations.get(name)
        if not receipt:
            continue
        for modes in mode_options:
            resolved = _candidate_resolution(name=name, receipt=receipt, transfer_modes=modes)
            if resolved is not None:
                return resolved
    raise ValueError(
        "no calibrated proxy affords three parents, four matched retention forks, "
        "and the preregistered robust-binding plasticity test inside the wall budget"
    )


def validate_resolved(resolved: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "mode",
        "proxy",
        "parents",
        "target_actual_tokens_acquisition",
        "target_actual_tokens_continuation",
        "wall_budget_minutes",
        "acquisition_eval_interval_tokens",
        "continuation_eval_interval_tokens",
        "predicted_wall_seconds",
        "transfer_candidate",
        "transfer_modes",
        "transfer_target_actual_tokens",
        "transfer_eval_interval_tokens",
    }
    missing = sorted(required - set(resolved))
    if missing:
        raise ValueError(f"resolved campaign missing {missing}")
    if resolved["mode"] != "full":
        raise ValueError("CYR-GPU-006 scientific execution requires mode=full")
    if int(resolved["parents"]) != len(CYR6_PARENT_SEEDS):
        raise ValueError("resolver must schedule all three preregistered parents")
    if int(resolved["target_actual_tokens_acquisition"]) < CYR6_ACQ_MIN_TOKENS:
        raise ValueError("acquisition dose below preregistered floor")
    if int(resolved["target_actual_tokens_continuation"]) < CYR6_FORK_MIN_TOKENS:
        raise ValueError("continuation dose below preregistered floor")
    if int(resolved["transfer_target_actual_tokens"]) < CYR6_TRANSFER_MIN_TOKENS:
        raise ValueError("transfer dose below preregistered floor")
    if resolved["transfer_candidate"] != CYR6_TRANSFER_CANDIDATE:
        raise ValueError("transfer candidate must be prospectively fixed")
    modes = tuple(resolved["transfer_modes"])
    if CYR6_TRANSFER_PRIMARY_MODE not in modes:
        raise ValueError("replay plasticity transfer mode may not be dropped")
    if any(mode not in {CYR6_TRANSFER_PRIMARY_MODE, CYR6_TRANSFER_OPTIONAL_MODE} for mode in modes):
        raise ValueError("unknown transfer mode")
    wall = float(resolved["wall_budget_minutes"])
    if not 60.0 <= wall <= CYR6_WALL_HARD_MINUTES:
        raise ValueError("resolved wall budget outside preregistered limits")
    return dict(resolved)


def final_decision(
    parent_runs: list[Mapping[str, Any]],
    *,
    transfer: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Retention decision plus a strictly bounded research-candidate flag."""
    retention = base.decide_campaign(parent_runs)
    transfer_supported = bool(
        transfer and transfer.get("status") == "REPLICATED_PLASTICITY_COMPATIBLE"
    )
    research_candidate = bool(
        retention.get("verdict") == "REPLICATED_WINNER"
        and retention.get("winner") == CYR6_TRANSFER_CANDIDATE
        and transfer_supported
    )
    return {
        **retention,
        "schema": "anra-cyr-gpu006-decision/v2",
        "transfer": transfer,
        "prospective_transfer_candidate": CYR6_TRANSFER_CANDIDATE,
        "research_candidate": research_candidate,
        "production_promotion_authorized": False,
        "claim_limit": (
            "GPU V5-proxy evidence only; TPU confirmation and explicit promotion "
            "remain mandatory"
        ),
    }
