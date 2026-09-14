"""FORMATION-MUX-001 Amendment-1 protocol.

Two independent experiments share execution only. No statistics are pooled.
All thresholds and outcome mappings are prospective.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from anra_v5 import formation_mux_model_v2 as fxm

CAMPAIGN = "FORMATION-MUX-001"
AMENDMENT = "AMENDMENT_1_PREEXECUTION"
EXPERIMENT_A = "CS-MECH-002"
EXPERIMENT_B = "REP-FORM-003A"
EXPERIMENTS = (EXPERIMENT_A, EXPERIMENT_B)
ARMS_A = fxm.ARMS
ARMS_B = ("R0_PRODUCTION_BPE", "R1_ISOMORPHIC_RENDERING")

SEED_BUNDLES = (73011, 73012, 73013, 73014)
B_NAMESPACE_XOR = 0x5EED0001
CALIBRATION_SEEDS = (99101, 99102, 99103, 99104)

A_UPDATES = 2_000
A_ELIGIBLE_FROM_UPDATE = 600
A_EVAL_EVERY_UPDATES = 100
A_CHECKPOINT_EVERY_UPDATES = 250

# REP-FORM is matched by actual processed non-padding token positions, not
# equal optimizer-step count. Full examples are indivisible, so each arm may
# overshoot by at most one batch; relative endpoint mismatch must stay <=0.1%.
B_PROCESSED_TOKEN_BUDGET = 500_000
B_ELIGIBLE_FROM_TOKENS = 150_000
B_EVAL_EVERY_TOKENS = 25_000
B_CHECKPOINT_EVERY_TOKENS = 50_000
B_EXPOSURE_MISMATCH_TOLERANCE = 0.001

BATCH_ROWS = 16
LR = 1e-3
MAX_GENERATION_TOKENS = 32

FORMATION_AUC_THRESHOLD = 0.05
ENDPOINT_GAP_THRESHOLD = 0.05
SIGN_CONSISTENCY = 3

WALL_BUDGET_MINUTES = 630.0
PARTITION_THRESHOLD_MINUTES = 630.0

CONTRASTS_A = (
    ("M0_STANDARD", "M1_EXTRA_NO_DECAY", "extra-row weight decay"),
    ("M1_EXTRA_NO_DECAY", "M2_EXTRA_FROZEN", "extra-row trainability/update evolution"),
    ("M2_EXTRA_FROZEN", "M3_EXTRA_FROZEN_MASKED", "extra-row denominator participation"),
)
CONTRAST_B = (
    "R0_PRODUCTION_BPE",
    "R1_ISOMORPHIC_RENDERING",
    "production segmentation vs isomorphic rendering",
)

VERDICTS = (
    "SUCCESS",
    "NULL",
    "REVERSE_EFFECT",
    "PARTIAL_OR_INTERACTION",
    "INCONCLUSIVE",
    "INCONCLUSIVE_EXPOSURE_MISMATCH",
    "ENGINEERING_FAILURE",
)
SEALED_STATES = ("NOT_CONSUMED", "STARTED", "COMPLETE")

NEXT_ACTION = {
    "SUCCESS": "record the supported contrast; advance only the specific architecture consequence it licenses",
    "NULL": "stop this mechanism line; retain the conservative working choice and move to the next roadmap item",
    "REVERSE_EFFECT": "record the reversed effect as a full-value result; do not relabel it as success",
    "PARTIAL_OR_INTERACTION": "do not promote an arm; isolate only the smallest remaining interaction if decision value justifies it",
    "INCONCLUSIVE": "preserve evidence and complete the frozen protocol without changing science",
    "INCONCLUSIVE_EXPOSURE_MISMATCH": "do not interpret the rendering contrast; repair execution so the frozen exposure tolerance is met",
    "ENGINEERING_FAILURE": "repair engineering only; keep the scientific protocol frozen",
}


def b_seed(bundle_seed: int) -> int:
    return int(bundle_seed) ^ B_NAMESPACE_XOR


def queue_assignment() -> dict[str, list[dict[str, Any]]]:
    """Matched seed bundles remain local to one physical T4."""

    queues: dict[str, list[dict[str, Any]]] = {"GPU0": [], "GPU1": []}
    for exp, arms in ((EXPERIMENT_A, ARMS_A), (EXPERIMENT_B, ARMS_B)):
        for index, bundle in enumerate(SEED_BUNDLES):
            gpu = f"GPU{index % 2}"
            for arm in arms:
                queues[gpu].append(
                    {
                        "experiment": exp,
                        "seed_bundle": bundle,
                        "seed_bundle_label": f"S{index + 1}",
                        "arm": arm,
                    }
                )
    return queues


def total_official_arms() -> int:
    return len(SEED_BUNDLES) * (len(ARMS_A) + len(ARMS_B))


def causal_variable_matrix() -> list[dict[str, Any]]:
    def row(arm: str, denominator: str, trainable: bool, state: bool, decay: float):
        return {
            "arm": arm,
            "physical_row_count": fxm.PHYSICAL_VOCAB,
            "shared_rows": "0..4095",
            "extra_rows": "4096..24575",
            "denominator_membership_training": denominator,
            "extra_row_trainability": trainable,
            "extra_row_gradient_receipt_after_mask": trainable,
            "extra_row_optimizer_state_evolution": state,
            "extra_row_weight_decay": decay,
            "tying": "tied embedding/output",
            "raw_parameter_count": "identical across arms",
            "effective_output_classes_training": (
                "4096" if arm == "M3_EXTRA_FROZEN_MASKED" else "24576"
            ),
            "effective_output_classes_eval": "24576",
            "shared_row_treatment": "row-aware AdamW wd=0.1 after global clip",
            "initialization": "byte-identical per matched seed",
        }

    return [
        row("M0_STANDARD", "all 24576 rows", True, True, 0.1),
        row("M1_EXTRA_NO_DECAY", "all 24576 rows", True, True, 0.0),
        row("M2_EXTRA_FROZEN", "all 24576 rows", False, False, 0.0),
        row("M3_EXTRA_FROZEN_MASKED", "shared 4096 rows", False, False, 0.0),
    ]


def assert_contrast_isolation() -> None:
    matrix = {r["arm"]: r for r in causal_variable_matrix()}
    ignore = {
        "arm",
        "physical_row_count",
        "shared_rows",
        "extra_rows",
        "tying",
        "raw_parameter_count",
        "shared_row_treatment",
        "initialization",
        "effective_output_classes_eval",
    }
    allowed = {
        "extra-row weight decay": {"extra_row_weight_decay"},
        "extra-row trainability/update evolution": {
            "extra_row_trainability",
            "extra_row_gradient_receipt_after_mask",
            "extra_row_optimizer_state_evolution",
        },
        "extra-row denominator participation": {
            "denominator_membership_training",
            "effective_output_classes_training",
        },
    }
    for higher, lower, label in CONTRASTS_A:
        diffs = {
            k for k in matrix[higher]
            if k not in ignore and matrix[higher][k] != matrix[lower][k]
        }
        if diffs != allowed[label]:
            raise RuntimeError(
                f"causal contrast {label} invalid: observed {sorted(diffs)}, "
                f"expected {sorted(allowed[label])}"
            )


def paired_verdict(
    *,
    formation_auc_deltas: Mapping[int, float],
    sealed_endpoint_gaps: Mapping[int, float],
    threshold: float = FORMATION_AUC_THRESHOLD,
    endpoint_threshold: float = ENDPOINT_GAP_THRESHOLD,
    required_signs: int = SIGN_CONSISTENCY,
) -> dict[str, Any]:
    bundles = list(SEED_BUNDLES)
    if any(b not in formation_auc_deltas or b not in sealed_endpoint_gaps for b in bundles):
        return {
            "verdict": "INCONCLUSIVE",
            "reason": "not all four matched seed bundles have both development AUC and sealed endpoint data",
        }
    mean_auc = sum(float(formation_auc_deltas[b]) for b in bundles) / len(bundles)
    mean_endpoint = sum(float(sealed_endpoint_gaps[b]) for b in bundles) / len(bundles)
    signs = [
        1 if formation_auc_deltas[b] > 0 else (-1 if formation_auc_deltas[b] < 0 else 0)
        for b in bundles
    ]
    pos, neg = signs.count(1), signs.count(-1)
    sign = 1 if pos >= required_signs else (-1 if neg >= required_signs else 0)
    endpoint_signs = [
        1 if sealed_endpoint_gaps[b] > 0 else (-1 if sealed_endpoint_gaps[b] < 0 else 0)
        for b in bundles
    ]
    endpoint_consistent = (
        endpoint_signs.count(sign) >= required_signs if sign != 0 else False
    )
    if sign != 0 and abs(mean_auc) >= threshold and abs(mean_endpoint) >= endpoint_threshold and endpoint_consistent:
        verdict = "SUCCESS" if sign > 0 else "REVERSE_EFFECT"
    elif abs(mean_auc) < threshold and abs(mean_endpoint) < endpoint_threshold:
        verdict = "NULL"
    else:
        verdict = "PARTIAL_OR_INTERACTION"
    return {
        "verdict": verdict,
        "mean_development_formation_auc_delta": round(mean_auc, 6),
        "mean_sealed_endpoint_gap": round(mean_endpoint, 6),
        "development_signs": signs,
        "sealed_endpoint_signs": endpoint_signs,
        "thresholds": {
            "formation_auc": threshold,
            "sealed_endpoint": endpoint_threshold,
            "sign_consistency": required_signs,
        },
        "next_action": NEXT_ACTION[verdict],
    }


def aggregate_experiment_verdict(contrast_verdicts: list[str]) -> str:
    """Prospective experiment-level rollup; contrast-level verdicts remain authoritative."""

    if not contrast_verdicts:
        return "INCONCLUSIVE"
    if any(v in ("INCONCLUSIVE", "INCONCLUSIVE_EXPOSURE_MISMATCH", "ENGINEERING_FAILURE") for v in contrast_verdicts):
        return "INCONCLUSIVE"
    if all(v == "NULL" for v in contrast_verdicts):
        return "NULL"
    non_null = [v for v in contrast_verdicts if v != "NULL"]
    if len(non_null) == 1:
        return non_null[0]
    if len(set(non_null)) == 1:
        return non_null[0]
    return "PARTIAL_OR_INTERACTION"


def exposure_mismatch(a_tokens: int, b_tokens: int) -> float:
    denom = max(int(a_tokens), int(b_tokens), 1)
    return abs(int(a_tokens) - int(b_tokens)) / denom


def claim_ceiling(experiment: str, verdict: str) -> str:
    scope = (
        "No production-vocabulary optimality, PRE500M, 250M/500M authorization, "
        "general cognition, or AGI claim follows from this campaign."
    )
    if experiment == EXPERIMENT_A:
        base = "fixed physical V24576 on the qualified 8L/256w mechanism surface"
    elif experiment == EXPERIMENT_B:
        base = "fixed physical V24576 on matched latent worlds under two renderings"
    else:
        raise ValueError(experiment)
    return f"{base}; verdict {verdict}. {scope}"


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
