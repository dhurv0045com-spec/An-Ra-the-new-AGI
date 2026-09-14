"""FORMATION-MUX-001 Science-S5 protocol binding.

S5 leaves S4 causal questions/endpoints unchanged. It binds the expanded
60k-row training surface and more durable checkpoint cadence requested before
any official outcomes were observed.
"""
from __future__ import annotations
import hashlib, json
from typing import Any
from v5_experiments.formation_mux_protocol_v4 import *  # noqa: F401,F403
from v5_experiments.formation_mux_surface_v5 import PER_FAMILY_COUNTS, PUBLIC_SCHEMA

AMENDMENT = "AMENDMENT_1D_PREEXECUTION_LONG_RUN_DATA_CUSTODY"
AMENDMENT_CHAIN = (
    "AMENDMENT_1_PREEXECUTION",
    "AMENDMENT_1B_PREEXECUTION_CLIP_ISOLATION",
    "AMENDMENT_1C_PREEXECUTION_SEALED_FIREWALL",
    AMENDMENT,
)

# Scientific exposure is unchanged. Only durability cadence changes.
A_CHECKPOINT_EVERY_UPDATES = 200
B_CHECKPOINT_EVERY_TOKENS = 10_000
PROGRESS_SNAPSHOT_AT_EVERY_CHECKPOINT = True


def protocol_payload(experiment: str) -> dict[str, Any]:
    if experiment not in EXPERIMENTS:
        raise ValueError(experiment)
    return {
        "campaign": CAMPAIGN,
        "amendment_chain": list(AMENDMENT_CHAIN),
        "experiment": experiment,
        "arms": list(ARMS_A if experiment == EXPERIMENT_A else ARMS_B),
        "seed_bundles": list(SEED_BUNDLES),
        "b_namespace_xor": B_NAMESPACE_XOR,
        "batch_rows": BATCH_ROWS,
        "lr": LR,
        "max_generation_tokens": MAX_GENERATION_TOKENS,
        "surface_counts_per_family": dict(PER_FAMILY_COUNTS),
        "public_surface_schema": PUBLIC_SCHEMA,
        "sealed_firewall": SEALED_FIREWALL,
        "primary_metric": "identity formation AUC (candidate-free exact+valid-EOS)",
        "sealed_endpoint": "identity exact+valid-EOS",
        "contrasts": [list(c) for c in (CONTRASTS_A if experiment == EXPERIMENT_A else (CONTRAST_B,))],
        "clip_semantics": "whole-model global clip; M2 frozen extra-row gradients retained through clip but excluded from parameter update/state/decay",
        "A": ({
            "updates": A_UPDATES,
            "eligible_from_update": A_ELIGIBLE_FROM_UPDATE,
            "eval_every_updates": A_EVAL_EVERY_UPDATES,
            "checkpoint_every_updates": A_CHECKPOINT_EVERY_UPDATES,
        } if experiment == EXPERIMENT_A else None),
        "B": ({
            "processed_token_budget": B_PROCESSED_TOKEN_BUDGET,
            "eligible_from_tokens": B_ELIGIBLE_FROM_TOKENS,
            "eval_every_tokens": B_EVAL_EVERY_TOKENS,
            "checkpoint_every_tokens": B_CHECKPOINT_EVERY_TOKENS,
            "exposure_mismatch_tolerance": B_EXPOSURE_MISMATCH_TOLERANCE,
        } if experiment == EXPERIMENT_B else None),
        "checkpoint_policy": {
            "exact_resume": True,
            "progress_snapshot_at_every_checkpoint": PROGRESS_SNAPSHOT_AT_EVERY_CHECKPOINT,
            "campaign_may_span_multiple_sessions": True,
            "science_wall_time_stop": False,
        },
        "thresholds": {
            "formation_auc": FORMATION_AUC_THRESHOLD,
            "sealed_endpoint": ENDPOINT_GAP_THRESHOLD,
            "sign_consistency": SIGN_CONSISTENCY,
        },
    }


def protocol_sha(experiment: str) -> str:
    return hashlib.sha256(json.dumps(
        protocol_payload(experiment), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
