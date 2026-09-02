"""Unit tests for senora.evaluator."""

from __future__ import annotations

import unittest

from e0_cognition.evaluation_generators import build_evaluation_suite, Split
from senora.evaluator import CasePrediction, SenoraEvaluator


class TestSenoraEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.suite = build_evaluation_suite(Split.DEVELOPMENT, seed=101, groups_per_family=1)

    def test_evaluator_all_correct_predictions(self) -> None:
        evaluator = SenoraEvaluator(self.suite, scorer_firewall_status="FAIL_DEVELOPMENT_POLICY")
        predictions = [
            CasePrediction(
                case_id=case.case_id,
                raw_output=case.answer,
                constrained_output=case.answer,
                assisted_output=case.answer,
                candidate_logprobs={cand: (1.0 if cand == case.answer else 0.0) for cand in case.candidates},
            )
            for case in self.suite.cases
        ]
        summary = evaluator.evaluate_predictions(predictions)
        self.assertEqual(summary.raw_core_accuracy, 1.0)
        self.assertEqual(summary.constrained_accuracy, 1.0)
        self.assertEqual(summary.assisted_accuracy, 1.0)
        self.assertEqual(summary.intervention_dependence_rate, 0.0)
        self.assertEqual(summary.assistance_harm_rate, 0.0)
        self.assertEqual(summary.natural_analogue_macro_accuracy, 1.0)

        # Candidate scoring must be blocked by scorer firewall
        self.assertIn("BLOCKED_BY_SCORER_FIREWALL", summary.candidate_scoring_status)
        self.assertIsNone(summary.candidate_selection_accuracy)

    def test_evaluator_with_certified_scorer_firewall(self) -> None:
        evaluator = SenoraEvaluator(self.suite, scorer_firewall_status="PASSED")
        predictions = [
            CasePrediction(
                case_id=case.case_id,
                raw_output=case.answer,
                constrained_output=case.answer,
                assisted_output=case.answer,
                candidate_logprobs={cand: (1.0 if cand == case.answer else 0.0) for cand in case.candidates},
            )
            for case in self.suite.cases
        ]
        summary = evaluator.evaluate_predictions(predictions)
        self.assertEqual(summary.candidate_scoring_status, "CERTIFIED_CANDIDATE_SCORING")
        self.assertEqual(summary.candidate_selection_accuracy, 1.0)

    def test_intervention_dependence_detected(self) -> None:
        evaluator = SenoraEvaluator(self.suite)
        # Raw outputs are wrong, but assisted outputs are correct
        predictions = [
            CasePrediction(
                case_id=case.case_id,
                raw_output="WRONG_ANSWER",
                constrained_output="WRONG_ANSWER",
                assisted_output=case.answer,
            )
            for case in self.suite.cases
        ]
        summary = evaluator.evaluate_predictions(predictions)
        self.assertEqual(summary.raw_core_accuracy, 0.0)
        self.assertEqual(summary.assisted_accuracy, 1.0)
        self.assertEqual(summary.intervention_dependence_rate, 1.0)
        self.assertEqual(summary.assistance_harm_rate, 0.0)


if __name__ == "__main__":
    unittest.main()