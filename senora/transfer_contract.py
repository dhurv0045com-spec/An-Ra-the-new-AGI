"""P35 to M102 Scale-Transfer Contract, Causal Effect Replication, and Statistical Decision Protocol.

Enforces:
1. Raw Core Gain defined strictly as causal treatment effect (candidate - matched_control).
2. Replication evaluates treatment effect replication across seeds (effect_seed1 vs effect_seed2),
   not candidate absolute accuracy.
3. Full statistical test execution: paired sign test p-value and 10,000-resample bootstrap 95% CI.
4. Prospective power analysis to justify detectability of promotion thresholds.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .evaluator import EvaluationSummary


@dataclass(frozen=True, slots=True)
class ProspectivePowerReceipt:
    sample_size: int
    alpha: float
    power: float
    standard_error_at_parity: float
    minimum_detectable_effect_size: float
    threshold_detectable: bool
    rationale: str


def compute_prospective_power(
    sample_size: int = 240,
    alpha: float = 0.005,
    power: float = 0.80,
    target_effect: float = 0.25,
) -> ProspectivePowerReceipt:
    """Perform prospective statistical power calculation for paired difference testing."""
    # Under null hypothesis p1 = p2 = 0.5, variance of paired difference is bounded by 0.5 / N
    # Z_(alpha/2) for alpha=0.005 is approx 2.807; Z_beta for power=0.80 is 0.842
    z_alpha = 2.807  # two-sided alpha=0.005
    z_beta = 0.842   # 80% power
    se = math.sqrt(0.5 / max(sample_size, 1))
    mdes = (z_alpha + z_beta) * se

    detectable = target_effect >= mdes
    return ProspectivePowerReceipt(
        sample_size=sample_size,
        alpha=alpha,
        power=power,
        standard_error_at_parity=round(se, 4),
        minimum_detectable_effect_size=round(mdes, 4),
        threshold_detectable=detectable,
        rationale=(
            f"At sample size N={sample_size} and two-sided alpha={alpha}, the minimum detectable "
            f"effect size (MDES) is {mdes:.3f}. Target threshold {target_effect:.3f} is "
            f"{'DETECTABLE' if detectable else 'UNDETECTABLE'}."
        ),
    )


@dataclass(frozen=True, slots=True)
class StatisticalTestResults:
    treatment_effect_delta: float
    bootstrap_ci_lower_95: float
    bootstrap_ci_upper_95: float
    sign_test_p_value: float
    concordant_wins: int
    concordant_losses: int
    ties: int


def calculate_paired_statistics(
    candidate_outcomes: Sequence[bool],
    control_outcomes: Sequence[bool],
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> StatisticalTestResults:
    """Calculate paired bootstrap confidence interval and paired sign test on discordant pairs."""
    n = len(candidate_outcomes)
    if n != len(control_outcomes):
        raise ValueError("candidate and control outcome sequences must have identical lengths")

    diffs = [1.0 if c and not k else (-1.0 if not c and k else 0.0) for c, k in zip(candidate_outcomes, control_outcomes)]
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    ties = sum(1 for d in diffs if d == 0)

    mean_delta = sum(diffs) / max(n, 1)

    # Paired sign test on discordant pairs (binomial test with p=0.5)
    discordant = wins + losses
    if discordant == 0:
        p_val = 1.0
    else:
        # Exact two-sided binomial p-value
        k = min(wins, losses)
        # sum_{i=0}^k binom(discordant, i) * 0.5^discordant
        cumulative = sum(math.comb(discordant, i) for i in range(k + 1)) * (0.5 ** discordant)
        p_val = min(1.0, 2.0 * cumulative)

    # Paired bootstrap CI
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()

    lower_idx = int(0.025 * resamples)
    upper_idx = int(0.975 * resamples)

    return StatisticalTestResults(
        treatment_effect_delta=round(mean_delta, 4),
        bootstrap_ci_lower_95=round(boot_means[lower_idx], 4),
        bootstrap_ci_upper_95=round(boot_means[upper_idx], 4),
        sign_test_p_value=round(p_val, 6),
        concordant_wins=wins,
        concordant_losses=losses,
        ties=ties,
    )


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
    min_raw_core_treatment_gain: float
    max_substrate_regression_fraction: float
    two_seed_replication_max_gap: float


STANDARD_P35_TO_M102_CONTRACT = TransferContract(
    schema="senora-m102-transfer-contract/v2",
    min_ood_effect_size_delta=0.25,
    min_bootstrap_ci_lower_95=0.15,
    max_sign_test_p_value=0.005,
    min_query_sensitivity_rate=0.80,
    min_invariance_stable_rate=0.85,
    min_natural_analogue_transfer_gain=0.15,
    min_worst_family_accuracy_floor=0.40,
    min_raw_core_treatment_gain=0.15,
    max_substrate_regression_fraction=0.03,
    two_seed_replication_max_gap=0.05,
)


@dataclass(frozen=True, slots=True)
class TransferDecision:
    authorized: bool
    status: str
    checks: dict[str, bool]
    blockers: list[str]
    statistics: dict[str, Any]
    candidate_summary: dict[str, Any]


def evaluate_transfer_decision(
    candidate_eval: EvaluationSummary,
    control_eval: EvaluationSummary,
    *,
    substrate_regression_fraction: float,
    paired_statistics: StatisticalTestResults | None = None,
    seed2_candidate_eval: EvaluationSummary | None = None,
    seed2_control_eval: EvaluationSummary | None = None,
    contract: TransferContract = STANDARD_P35_TO_M102_CONTRACT,
) -> TransferDecision:
    """Evaluate whether candidate P35 results scientifically warrant scaling to ~102M."""
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    # 1. Causal Raw Core Treatment Gain: Delta_raw = Candidate - Control
    raw_core_gain = candidate_eval.raw_core_accuracy - control_eval.raw_core_accuracy
    checks["raw_core_treatment_gain"] = raw_core_gain >= contract.min_raw_core_treatment_gain
    if not checks["raw_core_treatment_gain"]:
        blockers.append(
            f"Raw core treatment gain {raw_core_gain:.3f} is below required threshold {contract.min_raw_core_treatment_gain:.3f}"
        )

    # 2. OOD effect size
    checks["ood_effect_size"] = raw_core_gain >= contract.min_ood_effect_size_delta
    if not checks["ood_effect_size"]:
        blockers.append(
            f"OOD effect size delta {raw_core_gain:.3f} is below required threshold {contract.min_ood_effect_size_delta:.3f}"
        )

    # 3. Statistical Confidence & Sign Test
    if paired_statistics is not None:
        checks["bootstrap_ci_lower"] = paired_statistics.bootstrap_ci_lower_95 >= contract.min_bootstrap_ci_lower_95
        if not checks["bootstrap_ci_lower"]:
            blockers.append(
                f"Bootstrap 95% CI lower bound {paired_statistics.bootstrap_ci_lower_95:.3f} "
                f"is below required floor {contract.min_bootstrap_ci_lower_95:.3f}"
            )

        checks["sign_test_p_value"] = paired_statistics.sign_test_p_value <= contract.max_sign_test_p_value
        if not checks["sign_test_p_value"]:
            blockers.append(
                f"Paired sign test p-value {paired_statistics.sign_test_p_value:.6f} "
                f"exceeds significance alpha {contract.max_sign_test_p_value:.6f}"
            )
    else:
        checks["bootstrap_ci_lower"] = False
        checks["sign_test_p_value"] = False
        blockers.append("Paired statistical analysis (bootstrap CI and sign test) is required for transfer decision.")

    # 4. Query-swap sensitivity flip rate
    checks["query_sensitivity"] = candidate_eval.pair_sensitivity_flip_rate >= contract.min_query_sensitivity_rate
    if not checks["query_sensitivity"]:
        blockers.append(
            f"Query-swap sensitivity rate {candidate_eval.pair_sensitivity_flip_rate:.3f} "
            f"is below threshold {contract.min_query_sensitivity_rate:.3f}"
        )

    # 5. Invariance stability rate
    checks["invariance_stability"] = candidate_eval.pair_invariance_stable_rate >= contract.min_invariance_stable_rate
    if not checks["invariance_stability"]:
        blockers.append(
            f"Invariance stability rate {candidate_eval.pair_invariance_stable_rate:.3f} "
            f"is below threshold {contract.min_invariance_stable_rate:.3f}"
        )

    # 6. Natural analogue transfer gain (causal delta over control)
    natural_delta = candidate_eval.natural_analogue_macro_accuracy - control_eval.natural_analogue_macro_accuracy
    checks["natural_analogue_transfer"] = natural_delta >= contract.min_natural_analogue_transfer_gain
    if not checks["natural_analogue_transfer"]:
        blockers.append(
            f"Natural analogue transfer gain {natural_delta:.3f} is below threshold {contract.min_natural_analogue_transfer_gain:.3f}"
        )

    # 7. Worst family floor
    worst_family_acc = min(candidate_eval.family_accuracies.values()) if candidate_eval.family_accuracies else 0.0
    checks["worst_family_floor"] = worst_family_acc >= contract.min_worst_family_accuracy_floor
    if not checks["worst_family_floor"]:
        blockers.append(
            f"Worst-family accuracy {worst_family_acc:.3f} is below floor {contract.min_worst_family_accuracy_floor:.3f}"
        )

    # 8. Substrate loss retention
    checks["substrate_retention"] = substrate_regression_fraction <= contract.max_substrate_regression_fraction
    if not checks["substrate_retention"]:
        blockers.append(
            f"General language substrate regression {substrate_regression_fraction * 100:.2f}% "
            f"exceeds 3.0% ceiling ({contract.max_substrate_regression_fraction * 100:.2f}%)"
        )

    # 9. Two-seed causal treatment effect replication
    if seed2_candidate_eval is None or seed2_control_eval is None:
        checks["two_seed_replication"] = False
        blockers.append("M102 authorization requires a second replicated seed with matched candidate and control.")
    else:
        effect_seed1 = raw_core_gain
        effect_seed2 = seed2_candidate_eval.raw_core_accuracy - seed2_control_eval.raw_core_accuracy
        gap = abs(effect_seed1 - effect_seed2)
        direction_replicates = effect_seed1 > 0 and effect_seed2 > 0
        checks["two_seed_replication"] = direction_replicates and (gap <= contract.two_seed_replication_max_gap)
        if not checks["two_seed_replication"]:
            blockers.append(
                f"Two-seed causal treatment effect replication failed: seed1={effect_seed1:.3f}, "
                f"seed2={effect_seed2:.3f}, gap={gap:.3f} (tolerated: {contract.two_seed_replication_max_gap:.3f})"
            )

    authorized = len(blockers) == 0
    status = "AUTHORIZED_FOR_M102" if authorized else "M102_SCALE_BLOCKED"

    return TransferDecision(
        authorized=authorized,
        status=status,
        checks=checks,
        blockers=blockers,
        statistics=asdict(paired_statistics) if paired_statistics else {},
        candidate_summary={
            "raw_core_accuracy": candidate_eval.raw_core_accuracy,
            "control_raw_core_accuracy": control_eval.raw_core_accuracy,
            "raw_core_treatment_gain": raw_core_gain,
            "natural_analogue_delta": natural_delta,
            "worst_family_accuracy": worst_family_acc,
            "query_sensitivity_flip_rate": candidate_eval.pair_sensitivity_flip_rate,
            "invariance_stable_rate": candidate_eval.pair_invariance_stable_rate,
            "substrate_regression_fraction": substrate_regression_fraction,
        },
    )