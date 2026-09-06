"""PRE500M — fail-closed readiness certification for the 500M-token campaign.

This module does NOT train. It validates that every hard requirement for a
reliable, resumable, measurable 500,000,000-token Cymek campaign is met and
emits NEXT_500M_DECISION.json (ready_for_500m_training=true ONLY if every
requirement passes; otherwise precise blocking reasons).

Pure estimators and the decision builder are module-level and unit-tested;
device-bound measurements (representative production throughput) are an
injected seam so the decision logic is testable and the honest DATA_NOT_READY
state is expressible without a TPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from citadel_tpu.milestones import MILESTONES_500M, PRIMARY_FINAL_TOKEN_TARGET

CAMPAIGN_ID = "cymek-500m-v1"
TARGET_TOKENS = PRIMARY_FINAL_TOKEN_TARGET

# Cymek production model identity (v5_contracts/model_spec.V5A_250M at the
# pinned lineage; exact receipt total enforced by Cymek's assert_receipt).
PRODUCTION_MODEL = {
    "name": "V5A_250M",
    "parameter_count": 250_216_960,
    "layers": 26, "width": 896, "query_heads": 14, "kv_heads": 7,
    "head_dimension": 64, "ffn_width": 2368, "vocabulary_size": 24_576,
    "context_length": 4096, "precision": "bf16-compute/fp32-master",
}

# Cymek WSD schedule in TOKEN space (v5_contracts/training_spec at the pin:
# warmup 0->50M linear to 3e-4, stable 50M->4.5B at 3e-4). Retained as-is:
# already 500M-compatible because it is token-indexed.
LR_SCHEDULE = {"kind": "WSD-token-space", "warmup_tokens": 50_000_000,
               "peak_lr": 3e-4, "stable_until_tokens": 4_500_000_000,
               "final_lr_at_500m": 3e-4}

MILESTONES = MILESTONES_500M
EVAL_TOKEN_POINTS = (0, 10_000_000, 25_000_000, 50_000_000, 100_000_000,
                     200_000_000, 350_000_000, 500_000_000)
RECOVERY_CHECKPOINT_INTERVAL_MINUTES = 30

DATA_READINESS_STATES = ("DECLARED", "MATERIALIZED", "VERIFIED", "QUALIFIED",
                         "RUNNABLE")


def lr_schedule_table() -> list[dict[str, Any]]:
    """Token-based schedule table at the §19 checkpoints (pure)."""
    points = (0, 1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000,
              100_000_000, 200_000_000, 350_000_000, TARGET_TOKENS)
    warmup = LR_SCHEDULE["warmup_tokens"]
    peak = LR_SCHEDULE["peak_lr"]
    table = []
    for tokens in points:
        lr = peak if tokens >= warmup else peak * (tokens / warmup)
        table.append({"tokens": tokens, "learning_rate": lr})
    return table


def storage_estimate_gb(model_params: int = PRODUCTION_MODEL["parameter_count"],
                        recovery_checkpoints: int = 4,
                        milestone_checkpoints: int = 5) -> dict[str, Any]:
    """§18 storage feasibility (pure arithmetic; fp32 master weights).

    Per checkpoint: model 4 B/param + AdamW moments 8 B/param
    (+ 4 B/param gradients in the resident set, not per checkpoint file).
    """
    model_gb = model_params * 4 / 1024 ** 3
    optimizer_gb = model_params * 8 / 1024 ** 3
    per_checkpoint_gb = model_gb + optimizer_gb
    recovery_gb = per_checkpoint_gb * recovery_checkpoints
    milestone_gb = per_checkpoint_gb * milestone_checkpoints
    peak_local_gb = per_checkpoint_gb * 2 + recovery_gb  # staging + rotation
    persistent_gb = recovery_gb + milestone_gb
    return {
        "per_checkpoint_gb": round(per_checkpoint_gb, 3),
        "recovery_rotation_gb": round(recovery_gb, 3),
        "scientific_milestones_gb": round(milestone_gb, 3),
        "PEAK_LOCAL_STORAGE_GB": round(peak_local_gb, 3),
        "PERSISTENT_STORAGE_GB": round(persistent_gb, 3),
        "recovery_checkpoints": recovery_checkpoints,
        "milestone_checkpoints": milestone_checkpoints,
    }


def data_readiness(*, corpus_materialized: bool = False,
                   unique_runnable_train_tokens: int | None = None,
                   mixture_scheduled_tokens: dict[str, int] | None = None,
                   mixture_available_unique_tokens: dict[str, int] | None = None,
                   tokenizer_artifact_sha: str | None = None,
                   ) -> dict[str, Any]:
    """Hard data gate (§5/§6/§8): RUNNABLE requires materialized bytes,
    a frozen tokenizer identity, and per-source unique supply covering the
    scheduled 500M allocation without pathological replay. Never fakes
    readiness."""
    scheduled = mixture_scheduled_tokens or {}
    available = mixture_available_unique_tokens or {}
    scheduled_total = sum(scheduled.values())
    blockers: list[str] = []
    state = "RUNNABLE"
    if not corpus_materialized:
        blockers.append("production corpus not MATERIALIZED: no manifest-"
                        "bound bytes exist for the 500M campaign")
        state = "DECLARED"
    if not tokenizer_artifact_sha:
        blockers.append("tokenizer artifact identity not frozen "
                        "(provisional 24,576 BPE is not a production artifact)")
    if scheduled_total != TARGET_TOKENS:
        blockers.append(f"mixture scheduled tokens {scheduled_total} != "
                        f"{TARGET_TOKENS}")
    replay_over = {}
    for source, need in sorted(scheduled.items()):
        have = available.get(source, 0)
        if have < need:
            factor = round(need / max(have, 1), 2)
            replay_over[source] = factor
            blockers.append(f"source {source}: unique supply {have} < scheduled "
                            f"{need} (replay {factor}x)")
    if replay_over:
        state = "REPLAY_REQUIRED" if state == "RUNNABLE" else state
    if blockers:
        state = "DATA_NOT_READY"
    return {"state": state,
            "UNIQUE_RUNNABLE_TRAIN_TOKENS": unique_runnable_train_tokens,
            "scheduled_tokens_by_source": scheduled,
            "available_unique_tokens_by_source": available,
            "replay_factor_by_source": replay_over,
            "blockers": blockers}


def build_next_500m_decision(**parts: Any) -> dict[str, Any]:
    """NEXT_500M_DECISION builder (pure, fail-closed per §26).

    ready_for_500m_training is true ONLY if every hard requirement passes:
    canonical Cymek lineage, model identity, campaign spec, milestone logic
    (self-verified), token-based LR schedule, DATA RUNNABLE, static-shape
    fit evidence, representative measured throughput, storage feasibility,
    checkpoint/resume certification, evaluation hooks, and no stop-gate
    blockers. Any missing/false input yields ready=false + precise reasons.
    """
    import math as _math

    blocking: list[str] = []
    target = parts.get("target_tokens", TARGET_TOKENS)
    if target != TARGET_TOKENS:
        blocking.append(f"target tokens {target!r} != {TARGET_TOKENS}")
    lineage_sha = parts.get("canonical_cymek_sha")
    if lineage_sha != parts.get("runtime_pin_sha"):
        blocking.append("Cymek runtime is not the canonical pinned lineage")
    model = parts.get("model") or {}
    if model.get("parameter_count") != PRODUCTION_MODEL["parameter_count"]:
        blocking.append("production model identity is not the audited "
                        "V5A_250M receipt (250,216,960)")
    if not parts.get("campaign_spec_sha256"):
        blocking.append("campaign spec identity missing")
    data = parts.get("data") or {}
    if data.get("state") != "RUNNABLE":
        blocking.append("DATA_NOT_READY: " +
                        "; ".join(data.get("blockers", []))[:200])
    for requirement, label in (
            ("milestone_logic_verified", "milestone crossing logic unverified"),
            ("lr_schedule_token_based", "LR schedule is not token-based"),
            ("static_shape_fit", "static-shape fit unproven"),
            ("checkpoint_transaction_certified",
             "checkpoint transaction uncertified"),
            ("exact_resume_certified", "exact-resume uncertified"),
            ("fresh_runtime_resume_certified",
             "fresh-runtime resume uncertified"),
            ("evaluation_hooks_wired", "evaluation hooks not wired"),
            ("storage_feasible", "storage infeasible")):
        if parts.get(requirement) is not True:
            blocking.append(label)
    rate = parts.get("estimated_tokens_per_second")
    if not (isinstance(rate, (int, float)) and not isinstance(rate, bool)
            and _math.isfinite(float(rate)) and float(rate) > 0):
        blocking.append("no representative measured production throughput "
                        f"(got {rate!r}); PRE500M requires a real-rate "
                        "certification on the production model/shape/data")
    hours = None
    if isinstance(rate, (int, float)) and rate and rate > 0:
        hours = round(TARGET_TOKENS / float(rate) / 3600, 2)
    stop_gates = parts.get("stop_gates_at") or []
    for milestone in (50_000_000, 100_000_000, 200_000_000):
        if milestone not in stop_gates:
            blocking.append(f"missing go/no-go stop gate at {milestone}")
    return {
        "campaign_id": CAMPAIGN_ID,
        "target_tokens": TARGET_TOKENS,
        "ready_for_500m_training": not blocking,
        "blocking_reasons": blocking,
        "canonical_cymek_sha": lineage_sha,
        "model": model,
        "lr_schedule": LR_SCHEDULE,
        "estimated_tokens_per_second": rate,
        "estimated_hours_500m": hours,
        "milestones": list(MILESTONES),
        "evaluation_token_points": list(EVAL_TOKEN_POINTS),
        "recommended_recovery_checkpoint_interval_minutes":
            RECOVERY_CHECKPOINT_INTERVAL_MINUTES,
        "storage": parts.get("storage"),
        "data_state": data.get("state"),
    }


__all__ = ["CAMPAIGN_ID", "DATA_READINESS_STATES", "EVAL_TOKEN_POINTS",
           "LR_SCHEDULE", "MILESTONES", "PRODUCTION_MODEL",
           "RECOVERY_CHECKPOINT_INTERVAL_MINUTES", "TARGET_TOKENS",
           "build_next_500m_decision", "data_readiness", "lr_schedule_table",
           "storage_estimate_gb"]
