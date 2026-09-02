"""Unit tests for senora.transfer_contract."""

from __future__ import annotations

import unittest

from senora.evaluator import EvaluationSummary
from senora.transfer_contract import (
    STANDARD_P35_TO_M102_CONTRACT,
    StatisticalTestResults,
    calculate_paired_statistics,
    compute_prospective_power,
    evaluate_transfer_decision,
)


def _make_eval_summary(
    raw_core: float,
    natural_analogue: float = 0.50,
    sensitivity: float = 0.85,
    invariance: float = 0.90,
    worst_family: float = 0.50,
) -> EvaluationSummary:
    return EvaluationSummary(
        schema="senora-evaluation-summary/v1",
        suite_split="development",
        case_count=100,
        raw_core_accuracy=raw_core,
        constrained_accuracy=raw_core,
        assisted_accuracy=raw_core,
        intervention_dependence_rate=0.0,
        assistance_harm_rate=0.0,
        family_accuracies={"family_a": worst_family, "family_b": raw_core},
        difficulty_curves={"all": {1: raw_core}},
        pair_sensitivity_flip_rate=sensitivity,
        pair_invariance_stable_rate=invariance,
        natural_analogue_macro_accuracy=natural_analogue,
        candidate_scoring_status="BLOCKED_BY_SCORER_FIREWALL",
        candidate_selection_accuracy=None,
    )


class TestSenoraTransferContract(unittest.TestCase):
    def test_prospective_power_calculation(self) -> None:
        receipt = compute_prospective_power(sample_size=240, alpha=0.005, power=0.80, target_effect=0.25)
        self.assertEqual(receipt.sample_size, 240)
        self.assertTrue(receipt.threshold_detectable)
        self.assertLess(receipt.minimum_detectable_effect_size, 0.25)

    def test_calculate_paired_statistics(self) -> None:
        # Candidate wins 40 times where control lost, ties on 60
        cand_outcomes = [True] * 40 + [True] * 30 + [False] * 30
        ctrl_outcomes = [False] * 40 + [True] * 30 + [False] * 30

        stats = calculate_paired_statistics(cand_outcomes, ctrl_outcomes, resamples=1000)
        self.assertEqual(stats.concordant_wins, 40)
        self.assertEqual(stats.concordant_losses, 0)
        self.assertEqual(stats.ties, 60)
        self.assertEqual(stats.treatment_effect_delta, 0.40)
        self.assertGreater(stats.bootstrap_ci_lower_95, 0.20)
        self.assertLess(stats.sign_test_p_value, 0.0001)

    def test_transfer_decision_requires_replicated_causal_effect(self) -> None:
        ctrl1 = _make_eval_summary(raw_core=0.20, natural_analogue=0.20)
        cand1 = _make_eval_summary(raw_core=0.55, natural_analogue=0.45)  # treatment effect = +0.35

        stats = StatisticalTestResults(
            treatment_effect_delta=0.35,
            bootstrap_ci_lower_95=0.25,
            bootstrap_ci_upper_95=0.45,
            sign_test_p_value=0.00001,
            concordant_wins=35,
            concordant_losses=0,
            ties=65,
        )

        # 1. Missing seed 2 fails
        decision_no_seed2 = evaluate_transfer_decision(
            cand1, ctrl1, substrate_regression_fraction=0.01, paired_statistics=stats
        )
        self.assertFalse(decision_no_seed2.authorized)
        self.assertFalse(decision_no_seed2.checks["two_seed_replication"])

        # 2. Seed 2 replicates treatment effect (+0.35 vs +0.33) -> Authorize
        ctrl2 = _make_eval_summary(raw_core=0.22, natural_analogue=0.20)
        cand2 = _make_eval_summary(raw_core=0.55, natural_analogue=0.45)  # treatment effect = +0.33

        decision_replicated = evaluate_transfer_decision(
            cand1,
            ctrl1,
            substrate_regression_fraction=0.01,
            paired_statistics=stats,
            seed2_candidate_eval=cand2,
            seed2_control_eval=ctrl2,
        )
        self.assertTrue(decision_replicated.authorized)
        self.assertEqual(decision_replicated.status, "AUTHORIZED_FOR_M102")

        # 3. Seed 2 effect does not replicate (candidate2 == control2) -> Fails
        cand2_flat = _make_eval_summary(raw_core=0.22)
        decision_failed = evaluate_transfer_decision(
            cand1,
            ctrl1,
            substrate_regression_fraction=0.01,
            paired_statistics=stats,
            seed2_candidate_eval=cand2_flat,
            seed2_control_eval=ctrl2,
        )
        self.assertFalse(decision_failed.authorized)


if __name__ == "__main__":
    unittest.main()