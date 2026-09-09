"""CYR-GPU-007 hardware-adaptive contract.

007 supersedes CYR-GPU-006 after the operator's real Colab calibration proved
that the frozen 006 full design could not fit the 170 minute hard wall on the
assigned GPU. No scientific outcome was observed. The only new information is
hardware throughput/fit, so 007 preregisters a hardware-only tier resolver.

Scientific priority is preserved:
* same-parent HIGH/LOW/FIXED/HYST matched forks;
* candidate-free sustained G90;
* at least two independent acquisition parents for any replicated claim;
* equal-age/equal-exposure HYST-vs-LOW binding plasticity transfer whenever
  hardware affords it;
* no production/TPU promotion from a GPU result.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from v5_experiments import cyr_gpu006_final as base

CYR7_ID = "CYR-GPU-007"
CYR7_PARENT_SEEDS = (707, 808, 909)
CYR7_ARMS = base.CYR6_ARMS
CYR7_WALL_TARGET_MINUTES = 135.0
CYR7_WALL_HARD_MINUTES = 170.0
CYR7_PACKAGING_RESERVE_MINUTES = 8.0
CYR7_ACQ_MIN_TOKENS = 2_000_000
CYR7_ACQ_TARGET_TOKENS = 4_000_000
CYR7_FORK_MIN_TOKENS = 500_000
CYR7_FORK_TARGET_TOKENS = 2_000_000
CYR7_TRANSFER_MIN_TOKENS = 500_000
CYR7_TRANSFER_TARGET_TOKENS = 1_000_000

# Slow-GPU tiers deliberately evaluate less often than 006. Four or five
# observations still permit the frozen three-confirmation gates/hysteresis.
TIER_SPECS: tuple[dict[str, Any], ...] = (
    {
        "tier": "FULL_3P_TRANSFER3",
        "parents": 3,
        "transfer_enabled": True,
        "transfer_target_parents": 3,
        "transfer_min_parents": 2,
        "acq_eval_interval": 250_000,
        "fork_eval_interval": 100_000,
        "transfer_eval_interval": 100_000,
        "claim_ceiling": "MULTI_PARENT_MULTI_TASK_GPU_DEVELOPMENT",
    },
    {
        "tier": "CORE_2P_TRANSFER2",
        "parents": 2,
        "transfer_enabled": True,
        "transfer_target_parents": 2,
        "transfer_min_parents": 2,
        "acq_eval_interval": 400_000,
        "fork_eval_interval": 125_000,
        "transfer_eval_interval": 125_000,
        "claim_ceiling": "TWO_PARENT_MULTI_TASK_GPU_DEVELOPMENT",
    },
    {
        "tier": "RETENTION_2P_ONLY",
        "parents": 2,
        "transfer_enabled": False,
        "transfer_target_parents": 0,
        "transfer_min_parents": 0,
        "acq_eval_interval": 400_000,
        "fork_eval_interval": 125_000,
        "transfer_eval_interval": 125_000,
        "claim_ceiling": "TWO_PARENT_SAME_TASK_RETENTION_ONLY",
    },
)

# Scientific scale preference. TINY is a last-resort development tier; a
# positive TINY result can never become a research candidate for production.
PROXY_ORDER = ("MIDI", "MICRO", "RESEARCH_SMALL", "TINY")

# Re-export immutable 006 scientific/data authorities.
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
decide_retention = base.decide_retention
ARKENSTONE_AUDITED_SHA = base.ARKENSTONE_AUDITED_SHA
ARKENSTONE_DISCOVERY_V6_BUNDLE_SHA256 = base.ARKENSTONE_DISCOVERY_V6_BUNDLE_SHA256
CYR7_TRANSFER_CANDIDATE = base.CYR6_TRANSFER_CANDIDATE
CYR7_TRANSFER_COMPARATOR = base.CYR6_TRANSFER_COMPARATOR
CYR7_TRANSFER_MODE = base.CYR6_TRANSFER_MODE
CYR7_TRANSFER_SOURCE_MIN_T2 = base.CYR6_TRANSFER_SOURCE_MIN_T2
CYR7_TRANSFER_MAX_PLASTICITY_SLOWDOWN = base.CYR6_TRANSFER_MAX_PLASTICITY_SLOWDOWN
CYR7_TRANSFER_FINAL_TOLERANCE = base.CYR6_TRANSFER_FINAL_TOLERANCE
CYR7_TRANSFER_ADVANTAGE_SPEED_FRACTION = base.CYR6_TRANSFER_ADVANTAGE_SPEED_FRACTION
CYR7_TRANSFER_ADVANTAGE_RETENTION = base.CYR6_TRANSFER_ADVANTAGE_RETENTION


def _stage_seconds(
    *,
    training_tokens_per_sec: float,
    eval_examples_per_sec: float,
    parents: int,
    transfer_enabled: bool,
    transfer_parents: int,
    acquisition_tokens: int,
    continuation_tokens: int,
    transfer_tokens: int,
    acq_eval_interval: int,
    fork_eval_interval: int,
    transfer_eval_interval: int,
) -> dict[str, float]:
    if training_tokens_per_sec <= 0 or eval_examples_per_sec <= 0:
        return {"acquisition": math.inf, "retention": math.inf, "transfer": math.inf}
    acq_events = parents * math.ceil(acquisition_tokens / acq_eval_interval)
    acquisition = (
        parents * acquisition_tokens / training_tokens_per_sec
        + acq_events * (96 + 16) / eval_examples_per_sec
    )
    retention_events = parents * len(CYR7_ARMS) * math.ceil(
        continuation_tokens / fork_eval_interval
    )
    retention = (
        parents * len(CYR7_ARMS) * continuation_tokens / training_tokens_per_sec
        + retention_events * (96 + 112) / eval_examples_per_sec
    )
    transfer = 0.0
    if transfer_enabled:
        runs = transfer_parents * 2
        events = runs * math.ceil(transfer_tokens / transfer_eval_interval)
        transfer = (
            runs * transfer_tokens / training_tokens_per_sec
            + (events * (4 * 32 + 112) + runs * (4 * 32 + 112))
            / eval_examples_per_sec
        )
    return {"acquisition": acquisition, "retention": retention, "transfer": transfer}


def _cost_with_contingency(stages: Mapping[str, float]) -> float:
    science = sum(max(float(value) * 1.12, 15.0 if float(value) > 0 else 0.0)
                  for value in stages.values())
    return science + CYR7_PACKAGING_RESERVE_MINUTES * 60.0


def _fit(
    *, name: str, receipt: Mapping[str, Any], tier: Mapping[str, Any]
) -> dict[str, Any] | None:
    if receipt.get("status") != "PASS":
        return None
    train_tps = float(receipt["training_real_tokens_per_sec"])
    eval_eps = float(receipt["generation_examples_per_sec"])

    def stages_for(factor: float) -> tuple[dict[str, float], int, int, int]:
        acq = min(CYR7_ACQ_TARGET_TOKENS, int(CYR7_ACQ_MIN_TOKENS * factor))
        fork = min(CYR7_FORK_TARGET_TOKENS, int(CYR7_FORK_MIN_TOKENS * factor))
        transfer = min(CYR7_TRANSFER_TARGET_TOKENS,
                       int(CYR7_TRANSFER_MIN_TOKENS * factor))
        stages = _stage_seconds(
            training_tokens_per_sec=train_tps,
            eval_examples_per_sec=eval_eps,
            parents=int(tier["parents"]),
            transfer_enabled=bool(tier["transfer_enabled"]),
            transfer_parents=int(tier["transfer_target_parents"]),
            acquisition_tokens=acq,
            continuation_tokens=fork,
            transfer_tokens=transfer,
            acq_eval_interval=int(tier["acq_eval_interval"]),
            fork_eval_interval=int(tier["fork_eval_interval"]),
            transfer_eval_interval=int(tier["transfer_eval_interval"]),
        )
        return stages, acq, fork, transfer

    minimum, *_ = stages_for(1.0)
    if _cost_with_contingency(minimum) > CYR7_WALL_HARD_MINUTES * 60.0:
        return None

    # Increase dose prospectively until the measured hardware fills ~135 min;
    # outcomes never enter this search.
    lo, hi = 1.0, 4.0
    for _ in range(24):
        mid = (lo + hi) / 2.0
        stages, *_ = stages_for(mid)
        if _cost_with_contingency(stages) <= CYR7_WALL_TARGET_MINUTES * 60.0:
            lo = mid
        else:
            hi = mid
    stages, acq, fork, transfer = stages_for(lo)
    raw_budgets = {
        key: (round(max(value * 1.12, 15.0), 1) if value > 0 else 1.0)
        for key, value in stages.items()
    }
    predicted = sum(stages.values()) + CYR7_PACKAGING_RESERVE_MINUTES * 60.0
    selected_seeds = list(CYR7_PARENT_SEEDS[: int(tier["parents"])])
    claim_ceiling = str(tier["claim_ceiling"])
    if name == "TINY":
        claim_ceiling = "TINY_PROXY_DEVELOPMENT_ONLY"
    return {
        "schema": "anra-cyr-gpu007-resolved/v1",
        "mode": "full",
        "tier": tier["tier"],
        "proxy": name,
        "parents": len(selected_seeds),
        "parent_seeds": selected_seeds,
        "arms": list(CYR7_ARMS),
        "target_actual_tokens_acquisition": acq,
        "target_actual_tokens_continuation": fork,
        "acquisition_eval_interval_tokens": int(tier["acq_eval_interval"]),
        "continuation_eval_interval_tokens": int(tier["fork_eval_interval"]),
        "transfer_enabled": bool(tier["transfer_enabled"]),
        "transfer_candidate": CYR7_TRANSFER_CANDIDATE,
        "transfer_comparator": CYR7_TRANSFER_COMPARATOR,
        "transfer_mode": CYR7_TRANSFER_MODE,
        "transfer_target_parents": int(tier["transfer_target_parents"]),
        "transfer_min_parents": int(tier["transfer_min_parents"]),
        "transfer_target_actual_tokens": transfer,
        "transfer_eval_interval_tokens": int(tier["transfer_eval_interval"]),
        "wall_budget_minutes": CYR7_WALL_HARD_MINUTES,
        "predicted_wall_seconds": round(predicted, 1),
        "predicted_stage_seconds": {k: round(v, 1) for k, v in stages.items()},
        "stage_budgets_seconds": raw_budgets,
        "training_real_tokens_per_sec": train_tps,
        "generation_examples_per_sec": eval_eps,
        "claim_ceiling": claim_ceiling,
        "research_candidate_possible": bool(tier["transfer_enabled"] and name != "TINY"),
        "rule": "hardware-only tier/proxy/dose resolution; scientific outcomes forbidden",
        "arkenstone_audited_sha": ARKENSTONE_AUDITED_SHA,
    }


def resolve_from_calibrations(
    calibrations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose the strongest preregistered tier that the actual GPU can afford."""
    # Causal scope outranks parameter scale. Within a tier choose the largest
    # calibrated proxy. TINY is last-resort and has a strict claim ceiling.
    for tier in TIER_SPECS:
        for name in PROXY_ORDER:
            receipt = calibrations.get(name)
            if receipt:
                resolved = _fit(name=name, receipt=receipt, tier=tier)
                if resolved is not None:
                    return resolved
    diagnostics = {
        name: {
            "status": receipt.get("status"),
            "training_real_tokens_per_sec": receipt.get("training_real_tokens_per_sec"),
            "generation_examples_per_sec": receipt.get("generation_examples_per_sec"),
            "peak_vram_gb": receipt.get("peak_vram_gb"),
        }
        for name, receipt in calibrations.items()
    }
    raise ValueError(
        "assigned GPU cannot afford even the preregistered two-parent matched-fork "
        f"minimum inside 170 minutes; calibration={diagnostics}"
    )


def validate_resolved(resolved: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "mode", "tier", "proxy", "parents", "parent_seeds", "arms",
        "target_actual_tokens_acquisition", "target_actual_tokens_continuation",
        "acquisition_eval_interval_tokens", "continuation_eval_interval_tokens",
        "transfer_enabled", "transfer_candidate", "transfer_comparator", "transfer_mode",
        "transfer_target_parents", "transfer_min_parents", "transfer_target_actual_tokens",
        "transfer_eval_interval_tokens", "wall_budget_minutes", "predicted_wall_seconds",
        "predicted_stage_seconds", "stage_budgets_seconds", "claim_ceiling",
        "research_candidate_possible",
    }
    missing = sorted(required - set(resolved))
    if missing:
        raise ValueError(f"resolved campaign missing {missing}")
    if resolved["mode"] != "full":
        raise ValueError("CYR-GPU-007 scientific execution requires mode=full")
    parents = int(resolved["parents"])
    if parents not in (2, 3):
        raise ValueError("007 requires two or three independent parents")
    seeds = tuple(int(x) for x in resolved["parent_seeds"])
    if seeds != CYR7_PARENT_SEEDS[:parents]:
        raise ValueError("parent seed subset/order drift")
    if tuple(resolved["arms"]) != tuple(CYR7_ARMS):
        raise ValueError("matched fork arm set drift")
    if int(resolved["target_actual_tokens_acquisition"]) < CYR7_ACQ_MIN_TOKENS:
        raise ValueError("acquisition dose below floor")
    if int(resolved["target_actual_tokens_continuation"]) < CYR7_FORK_MIN_TOKENS:
        raise ValueError("continuation dose below floor")
    if bool(resolved["transfer_enabled"]):
        if int(resolved["transfer_min_parents"]) < 2:
            raise ValueError("transfer replication floor below two")
        if int(resolved["transfer_target_parents"]) > parents:
            raise ValueError("transfer parents exceed acquisition parents")
        if int(resolved["transfer_target_actual_tokens"]) < CYR7_TRANSFER_MIN_TOKENS:
            raise ValueError("transfer dose below floor")
    else:
        if int(resolved["transfer_min_parents"]) != 0 or int(resolved["transfer_target_parents"]) != 0:
            raise ValueError("retention-only tier must not reserve transfer parents")
        if resolved["research_candidate_possible"]:
            raise ValueError("retention-only tier cannot produce a research candidate")
    if resolved["proxy"] == "TINY" and resolved["research_candidate_possible"]:
        raise ValueError("TINY proxy cannot produce a production-facing research candidate")
    wall = float(resolved["wall_budget_minutes"])
    if not 60.0 <= wall <= CYR7_WALL_HARD_MINUTES:
        raise ValueError("resolved wall budget outside contract")
    budgets = resolved["stage_budgets_seconds"]
    if set(budgets) != {"acquisition", "retention", "transfer"}:
        raise ValueError("stage budget keys drift")
    return dict(resolved)


def final_decision(
    parent_runs: list[Mapping[str, Any]],
    *, transfer: Mapping[str, Any] | None,
    resolved: Mapping[str, Any],
) -> dict[str, Any]:
    result = base.final_decision(parent_runs, transfer=transfer)
    possible = bool(resolved.get("research_candidate_possible", False))
    if not possible:
        result["research_candidate"] = False
    result["schema"] = "anra-cyr-gpu007-decision/v1"
    result["experiment"] = CYR7_ID
    result["tier"] = resolved["tier"]
    result["proxy"] = resolved["proxy"]
    result["claim_ceiling"] = resolved["claim_ceiling"]
    result["research_candidate_possible"] = possible
    result["production_promotion_authorized"] = False
    return result
