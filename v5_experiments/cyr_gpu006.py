"""CYR-GPU-006 pure campaign core.

CYR-GPU-006 supersedes the frozen-but-unexecuted CYR-GPU-005 because the
005 Colab harness discarded the CUDA device/resolver, stopped after the first
qualified parent, and could declare a winner from one subject. 006 keeps the
same scientific question but makes replication and execution semantics part of
the contract.
"""
from __future__ import annotations

import math
from statistics import mean
from typing import Any, Mapping

from v5_experiments import cyr_gpu005 as base

CYR6_ID = "CYR-GPU-006"
CYR6_PARENT_SEEDS = (707, 808, 909)
CYR6_ARMS = base.CYR5_ARMS
CYR6_LRS = base.CYR5_LRS
CYR6_HYSTERESIS = base.CYR5_HYSTERESIS
CYR6_MIN_QUALIFIED_PARENTS = 2
CYR6_MIN_RET90_MEAN_MARGIN = 0.10
CYR6_MIN_PER_PARENT_SUPPORT = 0.05
CYR6_WALL_TARGET_MINUTES = 135.0
CYR6_WALL_HARD_MINUTES = 170.0
CYR6_PACKAGING_RESERVE_MINUTES = 8.0
CYR6_ACQUISITION_STAGE_FRACTION = 0.42
CYR6_CONTINUATION_STAGE_END_FRACTION = 0.90
CYR6_ACQ_MIN_TOKENS = 2_000_000
CYR6_ACQ_TARGET_TOKENS = 4_000_000
CYR6_FORK_MIN_TOKENS = 500_000
CYR6_FORK_TARGET_TOKENS = 2_000_000
CYR6_ACQ_EVAL_INTERVAL_TOKENS = 250_000
CYR6_FORK_EVAL_INTERVAL_TOKENS = 100_000
CYR6_TRANSFER_TOKENS = 250_000
CYR6_TRANSFER_MIN_PARENTS = 2

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
CYR6_TRANSFER_FAMILY = base.CYR5_TRANSFER_FAMILY


def arm_order(parent_index: int) -> tuple[str, ...]:
    """Deterministic rotation so no treatment is always last."""
    arms = list(CYR6_ARMS)
    shift = parent_index % len(arms)
    return tuple(arms[shift:] + arms[:shift])


def validate_resolved(resolved: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "mode", "proxy", "parents",
        "target_actual_tokens_acquisition", "target_actual_tokens_continuation",
        "wall_budget_minutes", "acquisition_eval_interval_tokens",
        "continuation_eval_interval_tokens", "predicted_wall_seconds",
        "predicted_stage_seconds",
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
    wall = float(resolved["wall_budget_minutes"])
    if not (60.0 <= wall <= CYR6_WALL_HARD_MINUTES):
        raise ValueError("resolved wall budget outside preregistered limits")
    science_seconds = (wall - CYR6_PACKAGING_RESERVE_MINUTES) * 60.0
    stage = resolved["predicted_stage_seconds"]
    limits = {
        "acquisition": science_seconds * CYR6_ACQUISITION_STAGE_FRACTION,
        "continuation": science_seconds * (
            CYR6_CONTINUATION_STAGE_END_FRACTION - CYR6_ACQUISITION_STAGE_FRACTION),
        "transfer": science_seconds * (1.0 - CYR6_CONTINUATION_STAGE_END_FRACTION),
    }
    for name, limit in limits.items():
        if float(stage[name]) > limit:
            raise ValueError(
                f"resolved {name} stage predicts {stage[name]:.1f}s but its "
                f"hard window is {limit:.1f}s")
    return dict(resolved)


def estimate_stage_seconds(*, training_tokens_per_sec: float,
                           eval_examples_per_sec: float,
                           acquisition_tokens: int,
                           continuation_tokens: int,
                           parents: int = 3,
                           arms: int = 4,
                           transfer_pairs: int = 2) -> dict[str, float]:
    """Prospective hardware-only runtime model, including generation cost."""
    if training_tokens_per_sec <= 0 or eval_examples_per_sec <= 0:
        return {"acquisition": math.inf, "continuation": math.inf,
                "transfer": math.inf}
    acq_events = parents * math.ceil(
        acquisition_tokens / CYR6_ACQ_EVAL_INTERVAL_TOKENS)
    fork_events = parents * arms * math.ceil(
        continuation_tokens / CYR6_FORK_EVAL_INTERVAL_TOKENS)
    acquisition = (parents * acquisition_tokens / training_tokens_per_sec
                   + acq_events * (96 + 16) / eval_examples_per_sec)
    continuation = (
        parents * arms * continuation_tokens / training_tokens_per_sec
        + fork_events * (96 + 112) / eval_examples_per_sec)
    transfer = (transfer_pairs * 2 * CYR6_TRANSFER_TOKENS
                / training_tokens_per_sec
                + transfer_pairs * 2 * 32 / eval_examples_per_sec)
    return {"acquisition": acquisition, "continuation": continuation,
            "transfer": transfer}


def estimate_wall_seconds(*, training_tokens_per_sec: float,
                          eval_examples_per_sec: float,
                          acquisition_tokens: int,
                          continuation_tokens: int,
                          parents: int = 3,
                          arms: int = 4,
                          transfer_pairs: int = 2) -> float:
    stages = estimate_stage_seconds(
        training_tokens_per_sec=training_tokens_per_sec,
        eval_examples_per_sec=eval_examples_per_sec,
        acquisition_tokens=acquisition_tokens,
        continuation_tokens=continuation_tokens,
        parents=parents, arms=arms, transfer_pairs=transfer_pairs)
    return (sum(stages.values())
            + CYR6_PACKAGING_RESERVE_MINUTES * 60.0)


def _stage_fits(*, stage: Mapping[str, float], wall_minutes: float) -> bool:
    science_seconds = (wall_minutes - CYR6_PACKAGING_RESERVE_MINUTES) * 60.0
    return (
        stage["acquisition"] <= science_seconds * CYR6_ACQUISITION_STAGE_FRACTION
        and stage["continuation"] <= science_seconds * (
            CYR6_CONTINUATION_STAGE_END_FRACTION - CYR6_ACQUISITION_STAGE_FRACTION)
        and stage["transfer"] <= science_seconds * (
            1.0 - CYR6_CONTINUATION_STAGE_END_FRACTION)
    )


def resolve_from_calibrations(calibrations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Choose the largest proxy that affords the full replicated campaign.

    Inputs are strictly hardware/throughput receipts; no losses/accuracies are
    accepted or inspected. The resolver also respects the runner's frozen
    acquisition/continuation/transfer wall windows, so an early stage cannot
    prospectively consume time reserved for later matched arms.
    """
    for name in ("MIDI", "MICRO", "RESEARCH_SMALL"):
        receipt = calibrations.get(name)
        if not receipt or receipt.get("status") != "PASS":
            continue
        train_tps = float(receipt["training_real_tokens_per_sec"])
        eval_eps = float(receipt["generation_examples_per_sec"])
        minimum_stage = estimate_stage_seconds(
            training_tokens_per_sec=train_tps, eval_examples_per_sec=eval_eps,
            acquisition_tokens=CYR6_ACQ_MIN_TOKENS,
            continuation_tokens=CYR6_FORK_MIN_TOKENS)
        if not _stage_fits(stage=minimum_stage,
                           wall_minutes=CYR6_WALL_HARD_MINUTES):
            continue

        # Resolve acquisition and continuation doses independently because
        # their stage costs differ sharply once candidate-free eval is counted.
        acq_lo, acq_hi = CYR6_ACQ_MIN_TOKENS, CYR6_ACQ_TARGET_TOKENS
        for _ in range(24):
            acq_mid = (acq_lo + acq_hi + 1) // 2
            stage = estimate_stage_seconds(
                training_tokens_per_sec=train_tps, eval_examples_per_sec=eval_eps,
                acquisition_tokens=acq_mid,
                continuation_tokens=CYR6_FORK_MIN_TOKENS)
            science = (CYR6_WALL_HARD_MINUTES - CYR6_PACKAGING_RESERVE_MINUTES) * 60.0
            if stage["acquisition"] <= science * CYR6_ACQUISITION_STAGE_FRACTION:
                acq_lo = acq_mid
            else:
                acq_hi = acq_mid - 1
        fork_lo, fork_hi = CYR6_FORK_MIN_TOKENS, CYR6_FORK_TARGET_TOKENS
        for _ in range(24):
            fork_mid = (fork_lo + fork_hi + 1) // 2
            stage = estimate_stage_seconds(
                training_tokens_per_sec=train_tps, eval_examples_per_sec=eval_eps,
                acquisition_tokens=acq_lo, continuation_tokens=fork_mid)
            science = (CYR6_WALL_HARD_MINUTES - CYR6_PACKAGING_RESERVE_MINUTES) * 60.0
            continuation_window = science * (
                CYR6_CONTINUATION_STAGE_END_FRACTION - CYR6_ACQUISITION_STAGE_FRACTION)
            if stage["continuation"] <= continuation_window:
                fork_lo = fork_mid
            else:
                fork_hi = fork_mid - 1
        acq, fork = int(acq_lo), int(fork_lo)
        predicted_stage = estimate_stage_seconds(
            training_tokens_per_sec=train_tps, eval_examples_per_sec=eval_eps,
            acquisition_tokens=acq, continuation_tokens=fork)
        predicted = sum(predicted_stage.values()) + CYR6_PACKAGING_RESERVE_MINUTES * 60.0
        resolved = {
            "schema": "anra-cyr-gpu006-resolved/v1", "mode": "full",
            "proxy": name, "parents": 3,
            "target_actual_tokens_acquisition": acq,
            "target_actual_tokens_continuation": fork,
            "acquisition_eval_interval_tokens": CYR6_ACQ_EVAL_INTERVAL_TOKENS,
            "continuation_eval_interval_tokens": CYR6_FORK_EVAL_INTERVAL_TOKENS,
            "transfer_enabled": True,
            "transfer_target_actual_tokens": CYR6_TRANSFER_TOKENS,
            "wall_budget_minutes": CYR6_WALL_HARD_MINUTES,
            "predicted_wall_seconds": round(predicted, 1),
            "predicted_stage_seconds": {
                key: round(value, 1) for key, value in predicted_stage.items()},
            "calibration_proxy": name,
            "training_real_tokens_per_sec": train_tps,
            "generation_examples_per_sec": eval_eps,
            "rule": "hardware-only; accuracy/loss forbidden; stage-window aware",
        }
        validate_resolved(resolved)
        return resolved
    raise ValueError(
        "no calibrated proxy affords 3 parents + 4 matched forks at the "
        "preregistered minimum dose inside every frozen stage window")


def _valid_parent(parent: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if parent.get("parent_status") != "G90_CONFIRMED":
        reasons.append("parent not G90_CONFIRMED")
    if not parent.get("parent_equivalence", {}).get("identical", False):
        reasons.append("fork parent bytes differ")
    if not parent.get("future_tail", {}).get("identical", False):
        reasons.append("future-tail mismatch")
    arms = parent.get("arms", {})
    for arm in CYR6_ARMS:
        receipt = arms.get(arm)
        if not receipt:
            reasons.append(f"missing {arm}")
            continue
        if receipt.get("status") != "COMPLETE":
            reasons.append(f"{arm} status={receipt.get('status')}")
        if int(receipt.get("actual_real_tokens", 0)) < int(
                receipt.get("target_actual_real_tokens", 1)):
            reasons.append(f"{arm} exposure starved")
        if not receipt.get("redteam_pass", False):
            reasons.append(f"{arm} redteam failed")
    return not reasons, reasons


def decide_campaign(parent_runs: list[Mapping[str, Any]], *,
                    transfer: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Replicated paired-subject verdict; one parent can never produce a win."""
    valid: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for parent in parent_runs:
        ok, reasons = _valid_parent(parent)
        if ok:
            valid.append(parent)
        else:
            rejected.append({"seed": parent.get("seed"), "reasons": reasons})
    if len(valid) < CYR6_MIN_QUALIFIED_PARENTS:
        return {
            "schema": "anra-cyr-gpu006-decision/v1",
            "verdict": "INCONCLUSIVE", "winner": None,
            "qualified_complete_parents": len(valid),
            "minimum_required": CYR6_MIN_QUALIFIED_PARENTS,
            "rejected": rejected,
            "reason": "fewer than two independent contract-valid parent experiments",
            "transfer": transfer,
        }
    scores = {
        arm: [float(parent["arms"][arm]["retention_ret90"]) for parent in valid]
        for arm in CYR6_ARMS
    }
    means = {arm: mean(values) for arm, values in scores.items()}
    ranked = sorted(means, key=means.get, reverse=True)
    candidate = ranked[0]
    pairwise: dict[str, Any] = {}
    supported = True
    for comparator in ranked[1:]:
        diffs = [a - b for a, b in zip(scores[candidate], scores[comparator])]
        mean_diff = mean(diffs)
        support_count = sum(diff >= CYR6_MIN_PER_PARENT_SUPPORT for diff in diffs)
        reversals = sum(diff <= -CYR6_MIN_PER_PARENT_SUPPORT for diff in diffs)
        pairwise[comparator] = {
            "diffs": [round(value, 4) for value in diffs],
            "mean_diff": round(mean_diff, 4),
            "support_count": support_count,
            "reversals": reversals,
        }
        if (mean_diff < CYR6_MIN_RET90_MEAN_MARGIN
                or support_count < CYR6_MIN_QUALIFIED_PARENTS
                or reversals > 0):
            supported = False
    verdict = "REPLICATED_WINNER" if supported else "INCONCLUSIVE"
    transfer_ok = bool(transfer and transfer.get("status") == "REPLICATED_COMPLETE")
    return {
        "schema": "anra-cyr-gpu006-decision/v1",
        "verdict": verdict,
        "winner": candidate if supported else None,
        "qualified_complete_parents": len(valid),
        "parent_seeds": [int(parent["seed"]) for parent in valid],
        "mean_ret90": {arm: round(value, 4) for arm, value in means.items()},
        "per_parent_ret90": {arm: [round(v, 4) for v in values]
                              for arm, values in scores.items()},
        "pairwise": pairwise,
        "minimum_mean_margin": CYR6_MIN_RET90_MEAN_MARGIN,
        "development_candidate_for_postrun_audit": bool(supported and transfer_ok),
        "production_promotion_authorized": False,
        "transfer": transfer,
        "rejected": rejected,
    }
