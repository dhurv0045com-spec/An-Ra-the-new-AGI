"""FORMATION-MUX-001 Amendment-1B protocol binding.

Extends v2 with corrected causal interpretation: frozen extra-row gradients
remain in the whole-model global clip; the direct M1->M2 intervention is
optimizer/update eligibility, not clip participation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from v5_experiments.formation_mux_protocol_v2 import *  # noqa: F401,F403
from anra_v5 import formation_mux_model_v3 as fxm

AMENDMENT = "AMENDMENT_1B_PREEXECUTION_CLIP_ISOLATION"
ARMS_A = fxm.ARMS


def causal_variable_matrix() -> list[dict[str, Any]]:
    def row(arm: str, denominator: int, update_extra: bool, state_extra: bool, decay_extra: float):
        return {
            "arm": arm,
            "physical_row_count": fxm.PHYSICAL_VOCAB,
            "shared_rows": "0..4095",
            "extra_rows": "4096..24575",
            "training_denominator_classes": denominator,
            "extra_row_parameter_updates": update_extra,
            "extra_row_optimizer_state_evolution": state_extra,
            "extra_row_weight_decay": decay_extra,
            "global_clip_scope": "whole live model gradient; frozen extra-row gradients retained when denominator includes them",
            "tying": "tied embedding/output",
            "raw_parameter_count": "identical across arms",
            "initialization": "byte-identical per matched seed",
        }

    return [
        row("M0_STANDARD", 24576, True, True, 0.1),
        row("M1_EXTRA_NO_DECAY", 24576, True, True, 0.0),
        row("M2_EXTRA_FROZEN", 24576, False, False, 0.0),
        row("M3_EXTRA_FROZEN_MASKED", 4096, False, False, 0.0),
    ]


def assert_contrast_isolation() -> None:
    matrix = {r["arm"]: r for r in causal_variable_matrix()}
    ignore = {
        "arm",
        "physical_row_count",
        "shared_rows",
        "extra_rows",
        "global_clip_scope",
        "tying",
        "raw_parameter_count",
        "initialization",
    }
    expected = {
        "extra-row weight decay": {"extra_row_weight_decay"},
        "extra-row trainability/update evolution": {
            "extra_row_parameter_updates",
            "extra_row_optimizer_state_evolution",
        },
        "extra-row denominator participation": {"training_denominator_classes"},
    }
    for higher, lower, label in CONTRASTS_A:
        diffs = {
            k for k in matrix[higher]
            if k not in ignore and matrix[higher][k] != matrix[lower][k]
        }
        if diffs != expected[label]:
            raise RuntimeError(
                f"causal contrast {label} invalid: observed {sorted(diffs)}, expected {sorted(expected[label])}"
            )


def protocol_payload(experiment: str) -> dict[str, Any]:
    if experiment not in EXPERIMENTS:
        raise ValueError(experiment)
    return {
        "campaign": CAMPAIGN,
        "amendment": AMENDMENT,
        "experiment": experiment,
        "arms": list(ARMS_A if experiment == EXPERIMENT_A else ARMS_B),
        "seed_bundles": list(SEED_BUNDLES),
        "b_namespace_xor": B_NAMESPACE_XOR,
        "batch_rows": BATCH_ROWS,
        "lr": LR,
        "primary_metric": "identity formation AUC (candidate-free exact+valid-EOS)",
        "sealed_endpoint": "identity exact+valid-EOS",
        "contrasts": [list(c) for c in (CONTRASTS_A if experiment == EXPERIMENT_A else (CONTRAST_B,))],
        "A_updates": A_UPDATES if experiment == EXPERIMENT_A else None,
        "B_processed_token_budget": B_PROCESSED_TOKEN_BUDGET if experiment == EXPERIMENT_B else None,
        "B_exposure_tolerance": B_EXPOSURE_MISMATCH_TOLERANCE if experiment == EXPERIMENT_B else None,
        "clip_semantics": "whole-model global clip; M2 frozen extra-row gradients retained through clip",
        "thresholds": {
            "formation_auc": FORMATION_AUC_THRESHOLD,
            "sealed_endpoint": ENDPOINT_GAP_THRESHOLD,
            "sign_consistency": SIGN_CONSISTENCY,
        },
    }


def protocol_sha(experiment: str) -> str:
    return hashlib.sha256(
        json.dumps(protocol_payload(experiment), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
