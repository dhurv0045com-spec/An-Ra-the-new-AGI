"""FORMATION-MUX-001 Science-S4 protocol binding.

Science S4 keeps every S3 treatment and endpoint, adds the prospective sealed
custody repair (Amendment 1C), and makes the protocol hash explicitly bind all
material formation/evaluation cadence constants used by the runner.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from v5_experiments.formation_mux_protocol_v3 import *  # noqa: F401,F403
from anra_v5 import formation_mux_model_v3 as fxm
from v5_experiments.formation_mux_surface_v2 import PER_FAMILY_COUNTS
from v5_experiments.formation_mux_surface_v4 import PUBLIC_SCHEMA

AMENDMENT = "AMENDMENT_1C_PREEXECUTION_SEALED_FIREWALL"
AMENDMENT_CHAIN = (
    "AMENDMENT_1_PREEXECUTION",
    "AMENDMENT_1B_PREEXECUTION_CLIP_ISOLATION",
    AMENDMENT,
)
ARMS_A = fxm.ARMS
SEALED_FIREWALL = (
    "workers receive training+development public manifest only; raw sealed rows "
    "are not persisted, are regenerated only after DEVELOPMENT_COMPLETE and "
    "SEALED_MARKER=STARTED, must match precommitted SHA-256, and are never "
    "written to partial-session output or result bundles"
)


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
        "contrasts": [
            list(c)
            for c in (
                CONTRASTS_A if experiment == EXPERIMENT_A else (CONTRAST_B,)
            )
        ],
        "clip_semantics": (
            "whole-model global clip; M2 frozen extra-row gradients retained "
            "through clip but excluded from parameter update/state/decay"
        ),
        "A": (
            {
                "updates": A_UPDATES,
                "eligible_from_update": A_ELIGIBLE_FROM_UPDATE,
                "eval_every_updates": A_EVAL_EVERY_UPDATES,
                "checkpoint_every_updates": A_CHECKPOINT_EVERY_UPDATES,
            }
            if experiment == EXPERIMENT_A
            else None
        ),
        "B": (
            {
                "processed_token_budget": B_PROCESSED_TOKEN_BUDGET,
                "eligible_from_tokens": B_ELIGIBLE_FROM_TOKENS,
                "eval_every_tokens": B_EVAL_EVERY_TOKENS,
                "checkpoint_every_tokens": B_CHECKPOINT_EVERY_TOKENS,
                "exposure_mismatch_tolerance": B_EXPOSURE_MISMATCH_TOLERANCE,
            }
            if experiment == EXPERIMENT_B
            else None
        ),
        "thresholds": {
            "formation_auc": FORMATION_AUC_THRESHOLD,
            "sealed_endpoint": ENDPOINT_GAP_THRESHOLD,
            "sign_consistency": SIGN_CONSISTENCY,
        },
    }


def protocol_sha(experiment: str) -> str:
    return hashlib.sha256(
        json.dumps(
            protocol_payload(experiment),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
