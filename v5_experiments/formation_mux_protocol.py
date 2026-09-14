"""FORMATION-MUX-001 protocol: pure constants, verdict logic, firewall.

Two scientifically independent experiments share one Kaggle campaign:

    CS-MECH-002    4 arms x 4 matched seed bundles = 16 official arms
    REP-FORM-003A  2 renderings x the same 4 matched seed bundles = 8 arms

Nothing is pooled across experiments. All thresholds, verdict mappings,
claim ceilings, and sealed rules below are fixed BEFORE execution.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from anra_v5 import formation_mux_model as fxm

CAMPAIGN = "FORMATION-MUX-001"
EXPERIMENT_A = "CS-MECH-002"
EXPERIMENT_B = "REP-FORM-003A"
EXPERIMENTS = (EXPERIMENT_A, EXPERIMENT_B)

ARMS_A = fxm.ARMS
ARMS_B = ("R0_PRODUCTION_BPE", "R1_ISOMORPHIC_RENDERING")

# 4 prospectively declared matched seed bundles; Experiment B derives its own
# RNG namespace by a fixed XOR so A can never perturb B randomness.
SEED_BUNDLES = (73011, 73012, 73013, 73014)
B_NAMESPACE_XOR = 0x5EED0001

UPDATES = 2_000
BATCH_ROWS = 16
EVAL_EVERY = 100
CHECKPOINT_EVERY = 250
LR = 1e-3
WALL_BUDGET_MINUTES = 630.0          # Kaggle 12h ceiling minus packaging margin
PARTITION_THRESHOLD_MINUTES = 630.0  # hard rule: stop above 10.5 h
SCIENCE_DEADLINE_MARGIN_MINUTES = 30.0

FORMATION_AUC_THRESHOLD = 0.05
ENDPOINT_GAP_THRESHOLD = 0.05
SIGN_CONSISTENCY = 3                 # of 4 matched seeds
CONTRASTS_A = (("M0_STANDARD", "M1_EXTRA_NO_DECAY", "extra-row weight decay"),
               ("M1_EXTRA_NO_DECAY", "M2_EXTRA_FROZEN", "extra-row trainability"),
               ("M2_EXTRA_FROZEN", "M3_EXTRA_FROZEN_MASKED",
                "extra-row denominator participation"))
CONTRAST_B = ("R0_PRODUCTION_BPE", "R1_ISOMORPHIC_RENDERING",
              "production segmentation vs isomorphic rendering")

VERDICTS = ("SUCCESS", "NULL", "REVERSE_EFFECT", "PARTIAL_OR_INTERACTION",
            "INCONCLUSIVE", "ENGINEERING_FAILURE")

SEALED_STATES = ("NOT_CONSUMED", "STARTED", "COMPLETE")


def b_seed(bundle_seed: int) -> int:
    return bundle_seed ^ B_NAMESPACE_XOR


def queue_assignment() -> dict[str, list[dict[str, Any]]]:
    """Deterministic dual-worker queue. One matched seed bundle = one GPU
    for ALL of its arms; bundles alternate physical workers; within a
    bundle the arms run serially. Experiment B bundles follow A's bundles
    on the same worker so each worker owns whole bundles."""

    queues: dict[str, list[dict[str, Any]]] = {"GPU0": [], "GPU1": []}
    plan: list[tuple[str, str, int, tuple[str, ...], str]] = []
    for index, bundle in enumerate(SEED_BUNDLES):
        plan.append((EXPERIMENT_A, f"S{index + 1}", bundle, ARMS_A,
                     f"GPU{index % 2}"))
    for index, bundle in enumerate(SEED_BUNDLES):
        plan.append((EXPERIMENT_B, f"S{index + 1}", bundle, ARMS_B,
                     f"GPU{index % 2}"))
    for experiment, label, bundle, arms, gpu in plan:
        for arm in arms:
            queues[gpu].append({
                "experiment": experiment, "seed_bundle_label": label,
                "seed_bundle": bundle, "arm": arm})
    return queues


def total_official_arms() -> int:
    return len(SEED_BUNDLES) * len(ARMS_A) + len(SEED_BUNDLES) * len(ARMS_B)


def causal_variable_matrix() -> list[dict[str, Any]]:
    """Every causal variable per arm; contrasts may differ in exactly one
    variable block (enforced by tests)."""

    def row(arm, denominator, trainability, grad_receipt, state_evolution,
            decay_extra):
        return {"arm": arm,
                "physical_row_count": fxm.PHYSICAL_VOCAB,
                "denominator_membership_training": denominator,
                "extra_row_trainability": trainability,
                "extra_row_gradient_receipt": grad_receipt,
                "extra_row_optimizer_state_evolution": state_evolution,
                "extra_row_weight_decay": decay_extra,
                "tying": "tied full-softmax embedding/output",
                "raw_parameter_count": "identical across arms (fixed V24576)",
                "effective_output_classes_training": (
                    "24576" if not arm.endswith("MASKED") else "128"),
                "effective_output_classes_eval": "24576",
                "shared_row_treatment": "canonical AdamW wd=0.1, lr=1e-3",
                "initialization": "byte-identical per matched seed"}

    return [
        row("M0_STANDARD", "all 24576 rows", True, True, True, 0.1),
        row("M1_EXTRA_NO_DECAY", "all 24576 rows", True, True, True, 0.0),
        row("M2_EXTRA_FROZEN", "all 24576 rows", False, False, False, 0.0),
        row("M3_EXTRA_FROZEN_MASKED", "active rows only", False, False,
            False, 0.0),
    ]


def assert_contrast_isolation() -> None:
    """Each preregistered contrast must differ in exactly its declared
    variable block; otherwise the design is invalid before execution."""

    matrix = {row["arm"]: row for row in causal_variable_matrix()}
    shared = ("arm", "physical_row_count", "tying", "raw_parameter_count",
              "shared_row_treatment", "initialization")
    for higher, lower, label in CONTRASTS_A:
        diffs = {key for key in matrix[higher]
                 if key not in shared and
                 matrix[higher][key] != matrix[lower][key]}
        allowed = {
            "extra-row weight decay": {
                "extra_row_weight_decay"},
            "extra-row trainability": {
                "extra_row_trainability", "extra_row_gradient_receipt",
                "extra_row_optimizer_state_evolution",
                "extra_row_weight_decay"},
            "extra-row denominator participation": {
                "denominator_membership_training",
                "effective_output_classes_training"},
        }[label]
        if not diffs <= allowed:
            raise ValueError(
                f"contrast {higher}-{lower} ({label}) leaks variables: "
                f"{sorted(diffs - allowed)}")


def paired_verdict(*, deltas: Mapping[int, float],
                   endpoint_gaps: Mapping[int, float],
                   threshold: float = FORMATION_AUC_THRESHOLD,
                   endpoint_threshold: float = ENDPOINT_GAP_THRESHOLD,
                   required_signs: int = SIGN_CONSISTENCY,
                   ) -> dict[str, Any]:
    """Prospective paired verdict over matched seed bundles.

    deltas: per-bundle (higher-arm AUC - lower-arm AUC); endpoint_gaps:
    per-bundle (higher-arm endpoint - lower-arm endpoint). NULL and
    REVERSE_EFFECT are full-value outcomes; nothing here requires a
    positive result."""

    complete = sorted(deltas)
    if len(complete) < len(SEED_BUNDLES):
        return {"verdict": "INCONCLUSIVE",
                "reason": (f"only {len(complete)}/{len(SEED_BUNDLES)} matched "
                           "bundles complete; surviving arms are not pairs"),
                "deltas": dict(deltas), "endpoint_gaps": dict(endpoint_gaps)}
    signs = [1 if deltas[b] > 0 else (-1 if deltas[b] < 0 else 0)
             for b in complete]
    majority = max(set(signs), key=signs.count)
    consistent = signs.count(majority) >= required_signs
    mean_delta = sum(deltas[b] for b in complete) / len(complete)
    mean_gap = sum(endpoint_gaps[b] for b in complete) / len(complete)
    endpoint_ok = abs(mean_gap) >= endpoint_threshold
    if consistent and majority != 0 and abs(mean_delta) >= threshold and endpoint_ok:
        verdict = "SUCCESS" if majority > 0 else "REVERSE_EFFECT"
    elif consistent and majority != 0 and abs(mean_delta) >= threshold and not endpoint_ok:
        verdict = "PARTIAL_OR_INTERACTION"
    elif abs(mean_delta) < threshold and abs(mean_gap) < endpoint_threshold:
        verdict = "NULL"
    else:
        verdict = "PARTIAL_OR_INTERACTION"
    return {"verdict": verdict, "mean_delta": round(mean_delta, 6),
            "mean_endpoint_gap": round(mean_gap, 6),
            "sign_consistency": f"{signs.count(majority)}/{len(signs)}",
            "thresholds": {"formation_auc": threshold,
                           "endpoint_gap": endpoint_threshold,
                           "sign_consistency": required_signs},
            "deltas": {str(b): round(deltas[b], 6) for b in complete},
            "endpoint_gaps": {str(b): round(endpoint_gaps[b], 6)
                              for b in complete},
            "next_action": NEXT_ACTION[verdict]}


NEXT_ACTION = {
    "SUCCESS": ("the contrast's mechanism is supported; record the effect and "
                "advance the specific architecture consequence it licenses"),
    "NULL": ("mechanism not supported; STOP this mechanism line; keep the "
             "conservative working choice and move to the next roadmap item "
             "(no follow-on experiment is created merely because the answer "
             "was null)"),
    "REVERSE_EFFECT": ("record the reversed effect with the same claim "
                       "ceiling discipline; it is a full-value result"),
    "PARTIAL_OR_INTERACTION": ("treat as interaction evidence; do not promote "
                               "any arm; design the smallest follow-up that "
                               "separates the interaction"),
    "INCONCLUSIVE": ("do not reinterpret; restore wall/data budget and rerun "
                     "the frozen protocol or extend execution partitioning"),
    "ENGINEERING_FAILURE": ("preserve failure evidence; repair engineering "
                            "only; scientific protocol unchanged"),
}


def claim_ceiling(experiment: str, verdict: str) -> str:
    base = {
        EXPERIMENT_A: ("at fixed physical V24576 on the qualified 8L/256w "
                       "development geometry and the CS-transfer latent "
                       "identity/copy surface"),
        EXPERIMENT_B: ("at fixed physical V24576 on the same latent worlds "
                       "under exactly two renderings"),
    }[experiment]
    scope = ("No claim about production vocab optimality, PRE500M, 250M/500M "
             "authorization, general cognition, or architectures beyond the "
             "declared arms follows from any verdict here.")
    return f"{base}; verdict {verdict}. {scope}"


def protocol_sha(experiment: str) -> str:
    payload = {"experiment": experiment, "arms": ARMS_A if experiment ==
               EXPERIMENT_A else ARMS_B,
               "seed_bundles": SEED_BUNDLES, "updates": UPDATES,
               "batch_rows": BATCH_ROWS, "lr": LR,
               "contrasts": [list(c) for c in
                             (CONTRASTS_A if experiment == EXPERIMENT_A
                              else [CONTRAST_B])],
               "thresholds": {"formation_auc": FORMATION_AUC_THRESHOLD,
                              "endpoint_gap": ENDPOINT_GAP_THRESHOLD,
                              "sign_consistency": SIGN_CONSISTENCY},
               "b_namespace_xor": B_NAMESPACE_XOR}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def preregistration(experiment: str) -> dict[str, Any]:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment}")
    return {
        "schema": "anra.formation-mux-preregistration/v1",
        "campaign": CAMPAIGN, "experiment": experiment,
        "arms": list(ARMS_A if experiment == EXPERIMENT_A else ARMS_B),
        "seed_bundles": list(SEED_BUNDLES),
        "seed_namespace_xor": B_NAMESPACE_XOR if experiment == EXPERIMENT_B else 0,
        "updates_per_arm": UPDATES, "batch_rows": BATCH_ROWS,
        "eval_every": EVAL_EVERY, "checkpoint_every": CHECKPOINT_EVERY,
        "lr": LR,
        "primary_metric": "identity formation AUC (candidate-free generation)",
        "endpoint_metric": "identity exact + valid EOS at the fixed endpoint",
        "contrasts": [list(c) for c in
                      (CONTRASTS_A if experiment == EXPERIMENT_A else [CONTRAST_B])],
        "causal_variable_matrix": (causal_variable_matrix()
                                   if experiment == EXPERIMENT_A else None),
        "verdicts": list(VERDICTS),
        "thresholds": {"formation_auc": FORMATION_AUC_THRESHOLD,
                       "endpoint_gap": ENDPOINT_GAP_THRESHOLD,
                       "sign_consistency_required": SIGN_CONSISTENCY},
        "next_action": dict(NEXT_ACTION),
        "sealed_policy": ("per-experiment sealed identity; consumed once after "
                          "the development protocol is frozen; markers "
                          "NOT_CONSUMED/STARTED/COMPLETE; STARTED-without-result "
                          "fails closed"),
        "protocol_sha256": protocol_sha(experiment),
        "claim_ceiling_rule": "see claim_ceiling(); no large-scale authorization",
    }
