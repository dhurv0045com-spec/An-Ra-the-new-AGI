"""Registry and fail-closed result gate for the seven non-critical moonshots."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MoonshotPilot:
    moonshot_id: str
    title: str
    required_metrics: tuple[str, ...]
    minimums: Mapping[str, float]
    maximums: Mapping[str, float] | None = None


MOONSHOT_PILOTS = (
    MoonshotPilot(
        "m1",
        "attention_ssm_hybrid",
        (
            "short_context_ratio",
            "long_context_speedup",
            "model_parameters",
            "seed_count",
        ),
        {
            "short_context_ratio": 0.98,
            "long_context_speedup": 1.5,
            "model_parameters": 150_000_000,
            "seed_count": 3,
        },
        {"model_parameters": 150_000_000},
    ),
    MoonshotPilot(
        "m2",
        "vision_encoder_projector",
        (
            "reconstruction_mse_improvement",
            "contrastive_recall_at_1",
            "vision_qa_accuracy",
            "heldout_pairs",
            "qa_items",
        ),
        {
            "reconstruction_mse_improvement": 0.30,
            "contrastive_recall_at_1": 0.40,
            "vision_qa_accuracy": 0.60,
            "heldout_pairs": 5_000,
            "qa_items": 200,
        },
    ),
    MoonshotPilot(
        "m3",
        "latent_reasoning",
        (
            "reasoning_score_ratio",
            "inference_flops_ratio",
            "model_parameters",
            "seed_count",
        ),
        {
            "reasoning_score_ratio": 1.15,
            "inference_flops_ratio": 0.0,
            "model_parameters": 150_000_000,
            "seed_count": 3,
        },
        {"inference_flops_ratio": 1.0, "model_parameters": 150_000_000},
    ),
    MoonshotPilot(
        "m4",
        "world_model_calibration",
        (
            "calibration_error",
            "action_success",
            "simulation_baseline_gain",
            "digital_top1_accuracy",
            "digital_majority_baseline_gain",
        ),
        {
            "action_success": 0.70,
            "simulation_baseline_gain": 1e-9,
            "digital_top1_accuracy": 0.65,
            "digital_majority_baseline_gain": 1e-9,
        },
        {"calibration_error": 0.10},
    ),
    MoonshotPilot(
        "m5",
        "trained_retriever_head",
        ("training_pairs", "recall_at_5_gain"),
        {"training_pairs": 20_000, "recall_at_5_gain": 0.10},
    ),
    MoonshotPilot(
        "m6",
        "formal_proof_expansion",
        ("proof_cases", "deterministic_pass_rate"),
        {"proof_cases": 100, "deterministic_pass_rate": 0.95},
    ),
    MoonshotPilot(
        "m7",
        "proposal_only_development",
        (
            "merged_human_approved_prs",
            "signed_gate_records",
            "reverted_prs",
            "unauthorized_apply_count",
        ),
        {"merged_human_approved_prs": 10, "signed_gate_records": 10},
        {"reverted_prs": 0.0, "unauthorized_apply_count": 0.0},
    ),
)


def evaluate_moonshot_pilot(pilot_id: str, metrics: Mapping[str, float]) -> dict[str, object]:
    pilot = next((item for item in MOONSHOT_PILOTS if item.moonshot_id == pilot_id), None)
    if pilot is None:
        raise KeyError(pilot_id)
    gates = {}
    for metric in pilot.required_metrics:
        value = float(metrics.get(metric, float("nan")))
        maximums = pilot.maximums or {}
        lower_bound = float(pilot.minimums.get(metric, float("-inf")))
        upper_bound = float(maximums.get(metric, float("inf")))
        gates[metric] = math.isfinite(value) and lower_bound <= value <= upper_bound
    return {
        "pilot": asdict(pilot),
        "metrics": dict(metrics),
        "gates": gates,
        "passed": bool(gates) and all(gates.values()),
    }
