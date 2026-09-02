"""P35 to M102 Scale-Transfer Contract and Promotion Criteria."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .evaluator import EvaluationSummary


@dataclass(frozen=True, slots=True)
class TransferContract:
    schema: str
    min_ood_effect_size_delta: float
    min_bootstrap_ci_lower_95: float
    max_sign_test_p_value: float
    min_query_sensitivity_rate: float
    min_invariance_stable_rate: float
    min_natural_analogue_transfer_gain: float
    min_worst_family_accuracy_floor: float
    min_raw_core_unassisted_gain: float
    max_substrate_regression_fraction: float
    two_seed_replication_max_gap: float


STANDARD_P35_TO_M102_CONTRACT = TransferContract(
    schema="senora-m102-transfer-contract/v1",
    min_ood_effect_size_delta=0.25,
    min_bootstrap_ci_lower_95=0.15,
    max_sign_test_p_value=0.005,
    min_query_sensitivity_rate=0.80,
    min_invariance_stable_rate=0.85,
    min_natural_analogue_transfer_gain=0.15,
    min_worst_family_accuracy_floor=0.40,
    min_raw_core_unassisted_gain=0.15,
    max_substrate_regression_fraction=0.03,
    two_seed_replication_max_gap=0.03,
)


@dataclass(frozen=True, slots=True)
class TransferDecision:
    authorized: bool
    status: str
    checks: dict[str, bool]
    blockers: list[str]
    candidate_summary: dict[str, Any]


def evaluate_transfer_decision(
    candidate_eval: EvaluationSummary,
    control_eval: EvaluationSummary,
    *,
    substrate_regression_fraction: float,
    seed2_candidate_eval: EvaluationSummary | None = None,
    contract: TransferContract = STANDARD_P35_TO_M102_CONTRACT,
) -> TransferDecision:
    """Evaluate whether candidate P35 results scientifically warrant scaling to ~102M."""

    checks: dict[str, bool] = {}
    blockers: list[str] = []

    # 1. Fresh OOD effect size
    ood_delta = candidate_eval.raw_core_accuracy - control_eval.raw_core_accuracy
    checks["ood_effect_size"] = ood_delta >= contract.min_ood_effect_size_delta
    if not checks["ood_effect_size"]:
        blockers.append(
            f"OOD effect size delta {ood_delta:.3f} is below required threshold {contract.min_ood_effect_size_delta:.3f}"
        )

    # 2. Query-swap sensitivity flip rate
    checks["query_sensitivity"] = candidate_eval.pair_sensitivity_flip_rate >= contract.min_query_sensitivity_rate
    if not checks["query_sensitivity"]:
        blockers.append(
            f"Query-swap sensitivity rate {candidate_eval.pair_sensitivity_flip_rate:.3f} "
            f"is below threshold {contract.min_query_sensitivity_rate:.3f}"
        )

    # 3. Invariance stability rate
    checks["invariance_stability"] = candidate_eval.pair_invariance_stable_rate >= contract.min_invariance_stable_rate
    if not checks["invariance_stability"]:
        blockers.append(
            f"Invariance stability rate {candidate_eval.pair_invariance_stable_rate:.3f} "
            f"is below threshold {contract.min_invariance_stable_rate:.3f}"
        )

    # 4. Natural analogue transfer gain
    natural_delta = candidate_eval.natural_analogue_macro_accuracy - control_eval.natural_analogue_macro_accuracy
    checks["natural_analogue_transfer"] = natural_delta >= contract.min_natural_analogue_transfer_gain
    if not checks["natural_analogue_transfer"]:
        blockers.append(
            f"Natural analogue transfer gain {natural_delta:.3f} is below threshold {contract.min_natural_analogue_transfer_gain:.3f}"
        )

    # 5. Worst family floor
    worst_family_acc = min(candidate_eval.family_accuracies.values()) if candidate_eval.family_accuracies else 0.0
    checks["worst_family_floor"] = worst_family_acc >= contract.min_worst_family_accuracy_floor
    if not checks["worst_family_floor"]:
        blockers.append(
            f"Worst-family accuracy {worst_family_acc:.3f} is below floor {contract.min_worst_family_accuracy_floor:.3f}"
        )

    # 6. Raw Core unassisted requirement (effect must exist in unassisted raw generation)
    checks["raw_core_unassisted"] = candidate_eval.raw_core_accuracy >= contract.min_raw_core_unassisted_gain
    if not checks["raw_core_unassisted"]:
        blockers.append(
            f"Raw core unassisted accuracy {candidate_eval.raw_core_accuracy:.3f} "
            f"is below threshold {contract.min_raw_core_unassisted_gain:.3f}"
        )

    # 7. Substrate loss retention
    checks["substrate_retention"] = substrate_regression_fraction <= contract.max_substrate_regression_fraction
    if not checks["substrate_retention"]:
        blockers.append(
            f"General language substrate regression {substrate_regression_fraction * 100:.2f}% "
            f"exceeds 3.0% ceiling ({contract.max_substrate_regression_fraction * 100:.2f}%)"
        )

    # 8. Two-seed fresh replication
    if seed2_candidate_eval is None:
        checks["two_seed_replication"] = False
        blockers.append("M102 authorization requires a second replicated seed for the winning P35 arm.")
    else:
        gap = abs(candidate_eval.raw_core_accuracy - seed2_candidate_eval.raw_core_accuracy)
        checks["two_seed_replication"] = gap <= contract.two_seed_replication_max_gap
        if not checks["two_seed_replication"]:
            blockers.append(
                f"Two-seed replication gap {gap:.3f} exceeds maximum tolerated variation {contract.two_seed_replication_max_gap:.3f}"
            )

    authorized = len(blockers) == 0
    status = "AUTHORIZED_FOR_M102" if authorized else "M102_SCALE_BLOCKED"

    return TransferDecision(
        authorized=authorized,
        status=status,
        checks=checks,
        blockers=blockers,
        candidate_summary={
            "raw_core_accuracy": candidate_eval.raw_core_accuracy,
            "ood_delta": ood_delta,
            "natural_analogue_delta": natural_delta,
            "worst_family_accuracy": worst_family_acc,
            "query_sensitivity_flip_rate": candidate_eval.pair_sensitivity_flip_rate,
            "invariance_stable_rate": candidate_eval.pair_invariance_stable_rate,
            "substrate_regression_fraction": substrate_regression_fraction,
        },
    )