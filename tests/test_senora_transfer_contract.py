"""Unit tests for senora.transfer_contract."""

from __future__ import annotations

import unittest

from senora.evaluator import EvaluationSummary
from senora.transfer_contract import evaluate_transfer_decision


class TestTransferContract(unittest.TestCase):
    def setUp(self) -> None:
        self.control_eval = EvaluationSummary(
            schema="senora-evaluation-summary/v1",
            suite_split="fresh",
            case_count=100,
            raw_core_accuracy=0.20,
            constrained_accuracy=0.30,
            assisted_accuracy=None,
            intervention_dependence_rate=None,
            assistance_harm_rate=None,
            family_accuracies={"binding": 0.20, "state": 0.20},
            difficulty_curves={},
            pair_sensitivity_flip_rate=0.40,
            pair_invariance_stable_rate=0.70,
            natural_analogue_macro_accuracy=0.20,
            candidate_scoring_status="NOT_EVALUATED",
            candidate_selection_accuracy=None,
        )

    def test_transfer_decision_authorized_when_all_criteria_met(self) -> None:
        candidate_eval = EvaluationSummary(
            schema="senora-evaluation-summary/v1",
            suite_split="fresh",
            case_count=100,
            raw_core_accuracy=0.70,  # delta = +0.50 (>= 0.25)
            constrained_accuracy=0.75,
            assisted_accuracy=None,
            intervention_dependence_rate=None,
            assistance_harm_rate=None,
            family_accuracies={"binding": 0.70, "state": 0.65},  # min >= 0.40
            difficulty_curves={},
            pair_sensitivity_flip_rate=0.88,  # >= 0.80
            pair_invariance_stable_rate=0.92,  # >= 0.85
            natural_analogue_macro_accuracy=0.60,  # delta = +0.40 (>= 0.15)
            candidate_scoring_status="NOT_EVALUATED",
            candidate_selection_accuracy=None,
        )
        seed2_eval = EvaluationSummary(
            schema="senora-evaluation-summary/v1",
            suite_split="fresh",
            case_count=100,
            raw_core_accuracy=0.69,  # gap = 0.01 (<= 0.03)
            constrained_accuracy=0.74,
            assisted_accuracy=None,
            intervention_dependence_rate=None,
            assistance_harm_rate=None,
            family_accuracies={"binding": 0.68, "state": 0.64},
            difficulty_curves={},
            pair_sensitivity_flip_rate=0.87,
            pair_invariance_stable_rate=0.91,
            natural_analogue_macro_accuracy=0.59,
            candidate_scoring_status="NOT_EVALUATED",
            candidate_selection_accuracy=None,
        )
        decision = evaluate_transfer_decision(
            candidate_eval,
            self.control_eval,
            substrate_regression_fraction=0.015,  # 1.5% regression (<= 3.0%)
            seed2_candidate_eval=seed2_eval,
        )
        self.assertTrue(decision.authorized)
        self.assertEqual(decision.status, "AUTHORIZED_FOR_M102")
        self.assertEqual(len(decision.blockers), 0)

    def test_transfer_decision_blocked_on_substrate_loss_regression(self) -> None:
        candidate_eval = EvaluationSummary(
            schema="senora-evaluation-summary/v1",
            suite_split="fresh",
            case_count=100,
            raw_core_accuracy=0.70,
            constrained_accuracy=0.75,
            assisted_accuracy=None,
            intervention_dependence_rate=None,
            assistance_harm_rate=None,
            family_accuracies={"binding": 0.70, "state": 0.65},
            difficulty_curves={},
            pair_sensitivity_flip_rate=0.88,
            pair_invariance_stable_rate=0.92,
            natural_analogue_macro_accuracy=0.60,
            candidate_scoring_status="NOT_EVALUATED",
            candidate_selection_accuracy=None,
        )
        decision = evaluate_transfer_decision(
            candidate_eval,
            self.control_eval,
            substrate_regression_fraction=0.045,  # 4.5% regression (> 3.0%)
            seed2_candidate_eval=candidate_eval,
        )
        self.assertFalse(decision.authorized)
        self.assertEqual(decision.status, "M102_SCALE_BLOCKED")
        self.assertTrue(any("substrate regression" in b for b in decision.blockers))


if __name__ == "__main__":
    unittest.main()