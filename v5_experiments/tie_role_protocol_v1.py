"""Prospective TIE-ROLE frontier extension for FORMATION-MUX-001.

Two independent experiments share execution only:
- TIE-ROLE-001: latent controlled 2x2 gradient-role factorial.
- TIE-ROLE-XFER-001: production-BPE transfer of the balanced treatment.

Science S5 remains frozen and authoritative for its own verdicts.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

CAMPAIGN = "FORMATION-MUX-001"
EXTENSION = "TIE-ROLE-FRONTIER-001"
EXPERIMENT_A = "TIE-ROLE-001"
EXPERIMENT_B = "TIE-ROLE-XFER-001"
EXPERIMENTS = (EXPERIMENT_A, EXPERIMENT_B)

ARMS_A = (
    "T0_CANONICAL",
    "T1_INPUT_X4",
    "T2_OUTPUT_X025",
    "T3_BALANCED_X4_X025",
)
ARMS_B = (
    "X0_CANONICAL_R0",
    "X1_BALANCED_R0",
)

SEED_BUNDLES = (73011, 73012, 73013, 73014)
B_NAMESPACE_XOR = 0x71E50101
CALIBRATION_SEEDS = (99211, 99212, 99213, 99214)

A_UPDATES = 2_000
A_ELIGIBLE_FROM_UPDATE = 600
A_EVAL_EVERY_UPDATES = 100
A_CHECKPOINT_EVERY_UPDATES = 200

B_PROCESSED_TOKEN_BUDGET = 500_000
B_ELIGIBLE_FROM_TOKENS = 150_000
B_EVAL_EVERY_TOKENS = 25_000
B_CHECKPOINT_EVERY_TOKENS = 10_000
B_EXPOSURE_MISMATCH_TOLERANCE = 0.001

BATCH_ROWS = 16
LR = 1e-3
MAX_GENERATION_TOKENS = 32
FORMATION_AUC_THRESHOLD = 0.05
ENDPOINT_GAP_THRESHOLD = 0.05
SIGN_CONSISTENCY = 3

PRIMARY_CONTRAST_A = (
    "T3_BALANCED_X4_X025",
    "T0_CANONICAL",
    "role-balanced tied-gradient treatment",
)
SECONDARY_CONTRASTS_A = (
    ("T1_INPUT_X4", "T0_CANONICAL", "input-path gradient amplification"),
    ("T2_OUTPUT_X025", "T0_CANONICAL", "output-path gradient attenuation"),
)
CONTRASTS_A = (PRIMARY_CONTRAST_A, *SECONDARY_CONTRASTS_A)
CONTRAST_B = (
    "X1_BALANCED_R0",
    "X0_CANONICAL_R0",
    "role-balanced treatment transfer under production BPE",
)

NEXT_ACTION = {
    "SUCCESS": "advance only the preregistered tied-role consequence; require transfer and mechanism consistency before Core promotion",
    "NULL": "stop tied-role gradient balancing as a Core mechanism",
    "REVERSE_EFFECT": "record reversed effect and retain canonical tied-gradient treatment",
    "PARTIAL_OR_INTERACTION": "do not promote; isolate only the smallest remaining representation/tying interaction",
    "INCONCLUSIVE": "preserve evidence and complete the frozen extension",
    "INCONCLUSIVE_EXPOSURE_MISMATCH": "repair execution only; do not interpret transfer",
    "ENGINEERING_FAILURE": "repair engineering only; scientific protocol remains frozen",
}


def assert_frontier_launch_allowed() -> None:
    raise RuntimeError(
        "TIE_ROLE_PILOT_NO_GO: TIE-ROLE-FRONTIER-001 launch blocked by the "
        "user-reported TIE-ROLE-PILOT-001 decision: DO NOT RUN FULL TIE-ROLE. "
        "Explicit review and authorization are required to reopen the campaign."
    )


def b_seed(bundle_seed: int) -> int:
    return int(bundle_seed) ^ B_NAMESPACE_XOR


def gradient_scales(arm: str) -> tuple[float, float]:
    table = {
        "T0_CANONICAL": (1.0, 1.0),
        "T1_INPUT_X4": (4.0, 1.0),
        "T2_OUTPUT_X025": (1.0, 0.25),
        "T3_BALANCED_X4_X025": (4.0, 0.25),
        "X0_CANONICAL_R0": (1.0, 1.0),
        "X1_BALANCED_R0": (4.0, 0.25),
    }
    if arm not in table:
        raise ValueError(f"unknown tie-role arm {arm}")
    return table[arm]


def queue_assignment() -> dict[str, list[dict[str, Any]]]:
    queues: dict[str, list[dict[str, Any]]] = {"GPU0": [], "GPU1": []}
    for experiment, arms in ((EXPERIMENT_A, ARMS_A), (EXPERIMENT_B, ARMS_B)):
        for index, bundle in enumerate(SEED_BUNDLES):
            gpu = f"GPU{index % 2}"
            for arm in arms:
                queues[gpu].append({
                    "experiment": experiment,
                    "seed_bundle": bundle,
                    "seed_bundle_label": f"S{index + 1}",
                    "arm": arm,
                })
    return queues


def total_official_arms() -> int:
    return len(SEED_BUNDLES) * (len(ARMS_A) + len(ARMS_B))


def exposure_mismatch(a_tokens: int, b_tokens: int) -> float:
    denom = max(int(a_tokens), int(b_tokens), 1)
    return abs(int(a_tokens) - int(b_tokens)) / denom


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
        return {"verdict": "INCONCLUSIVE", "reason": "missing matched seed evidence"}
    mean_auc = sum(float(formation_auc_deltas[b]) for b in bundles) / len(bundles)
    mean_endpoint = sum(float(sealed_endpoint_gaps[b]) for b in bundles) / len(bundles)
    signs = [1 if formation_auc_deltas[b] > 0 else (-1 if formation_auc_deltas[b] < 0 else 0) for b in bundles]
    endpoint_signs = [1 if sealed_endpoint_gaps[b] > 0 else (-1 if sealed_endpoint_gaps[b] < 0 else 0) for b in bundles]
    pos, neg = signs.count(1), signs.count(-1)
    sign = 1 if pos >= required_signs else (-1 if neg >= required_signs else 0)
    endpoint_consistent = endpoint_signs.count(sign) >= required_signs if sign else False
    if sign and abs(mean_auc) >= threshold and abs(mean_endpoint) >= endpoint_threshold and endpoint_consistent:
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


def protocol_payload(experiment: str) -> dict[str, Any]:
    if experiment not in EXPERIMENTS:
        raise ValueError(experiment)
    return {
        "campaign": CAMPAIGN,
        "extension": EXTENSION,
        "experiment": experiment,
        "arms": list(ARMS_A if experiment == EXPERIMENT_A else ARMS_B),
        "gradient_scales": {
            arm: list(gradient_scales(arm))
            for arm in (ARMS_A if experiment == EXPERIMENT_A else ARMS_B)
        },
        "seed_bundles": list(SEED_BUNDLES),
        "b_namespace_xor": B_NAMESPACE_XOR,
        "batch_rows": BATCH_ROWS,
        "lr": LR,
        "max_generation_tokens": MAX_GENERATION_TOKENS,
        "primary_metric": "identity formation AUC (candidate-free exact+valid-EOS)",
        "sealed_endpoint": "identity exact+valid-EOS",
        "contrasts": [list(c) for c in (CONTRASTS_A if experiment == EXPERIMENT_A else (CONTRAST_B,))],
        "A": {
            "updates": A_UPDATES,
            "eligible_from_update": A_ELIGIBLE_FROM_UPDATE,
            "eval_every_updates": A_EVAL_EVERY_UPDATES,
            "checkpoint_every_updates": A_CHECKPOINT_EVERY_UPDATES,
        } if experiment == EXPERIMENT_A else None,
        "B": {
            "processed_token_budget": B_PROCESSED_TOKEN_BUDGET,
            "eligible_from_tokens": B_ELIGIBLE_FROM_TOKENS,
            "eval_every_tokens": B_EVAL_EVERY_TOKENS,
            "checkpoint_every_tokens": B_CHECKPOINT_EVERY_TOKENS,
            "exposure_mismatch_tolerance": B_EXPOSURE_MISMATCH_TOLERANCE,
        } if experiment == EXPERIMENT_B else None,
        "thresholds": {
            "formation_auc": FORMATION_AUC_THRESHOLD,
            "sealed_endpoint": ENDPOINT_GAP_THRESHOLD,
            "sign_consistency": SIGN_CONSISTENCY,
        },
        "forward_equivalent_tied_parameterization": True,
        "science_s5_unchanged": True,
    }


def protocol_sha(experiment: str) -> str:
    return hashlib.sha256(json.dumps(protocol_payload(experiment), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
