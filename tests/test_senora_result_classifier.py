"""Unit tests for senora.result_classifier."""

from __future__ import annotations

import unittest

from senora.evaluator import EvaluationSummary
from senora.result_classifier import P35ResultCategory, classify_p35_a_results
from senora.transfer_contract import StatisticalTestResults


def _make_eval(
    raw_core: float,
    natural: float = 0.50,
    sensitivity: float = 0.85,
    worst_fam: float = 0.50,
    scoring_status: str = "BLOCKED_BY_SCORER_FIREWALL",
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
        family_accuracies={"fam_a": worst_fam, "fam_b": raw_core},
        difficulty_curves={"all": {1: raw_core}},
        pair_sensitivity_flip_rate=sensitivity,
        pair_invariance_stable_rate=0.90,
        natural_analogue_macro_accuracy=natural,
        candidate_scoring_status=scoring_status,
        candidate_selection_accuracy=None,
    )


class TestSenoraResultClassifier(unittest.TestCase):
    def test_classify_no_effect(self) -> None:
        ctrl = _make_eval(raw_core=0.30)
        cand = _make_eval(raw_core=0.32)  # delta = 0.02
        res = classify_p35_a_results(cand, ctrl, substrate_regression_fraction=0.01)
        self.assertEqual(res.category, P35ResultCategory.NO_EFFECT)
        self.assertIn("FALSIFY_DATA_MIXTURE_HYPOTHESIS", res.precommitted_next_action)

    def test_classify_substrate_tradeoff(self) -> None:
        ctrl = _make_eval(raw_core=0.20)
        cand = _make_eval(raw_core=0.60)  # delta = 0.40
        # Substrate regressed 4.5% > 3.0%
        res = classify_p35_a_results(cand, ctrl, substrate_regression_fraction=0.045)
        self.assertEqual(res.category, P35ResultCategory.SUBSTRATE_TRADEOFF)
        self.assertIn("ADJUST_CURRICULUM_MIXTURE", res.precommitted_next_action)

    def test_classify_family_collapse(self) -> None:
        ctrl = _make_eval(raw_core=0.20, worst_fam=0.30)
        cand = _make_eval(raw_core=0.60, worst_fam=0.15)  # fam_a collapsed to 15%
        res = classify_p35_a_results(cand, ctrl, substrate_regression_fraction=0.01)
        self.assertEqual(res.category, P35ResultCategory.FAMILY_COLLAPSE)
        self.assertIn("DEBUG_FAILING_FAMILY_GENERATOR", res.precommitted_next_action)

    def test_classify_synthetic_only(self) -> None:
        ctrl = _make_eval(raw_core=0.20, natural=0.20)
        cand = _make_eval(raw_core=0.55, natural=0.21)  # raw_core +35%, but natural +1%
        res = classify_p35_a_results(cand, ctrl, substrate_regression_fraction=0.01)
        self.assertEqual(res.category, P35ResultCategory.SYNTHETIC_ONLY)
        self.assertIn("HALT_OBJECTIVE_WORK_AND_REDESIGN_DATA", res.precommitted_next_action)

    def test_classify_realization_only(self) -> None:
        ctrl = _make_eval(raw_core=0.20, sensitivity=0.45)
        cand = _make_eval(raw_core=0.55, sensitivity=0.45)  # sensitivity <= 50%
        res = classify_p35_a_results(cand, ctrl, substrate_regression_fraction=0.01)
        self.assertEqual(res.category, P35ResultCategory.REALIZATION_ONLY)
        self.assertIn("INVESTIGATE_SELECTION_BOTTLENECK", res.precommitted_next_action)

    def test_classify_seed_unstable(self) -> None:
        ctrl1 = _make_eval(raw_core=0.20, natural=0.20)
        cand1 = _make_eval(raw_core=0.60, natural=0.45)  # Seed 1: +0.40
        ctrl2 = _make_eval(raw_core=0.20, natural=0.20)
        cand2 = _make_eval(raw_core=0.24, natural=0.22)  # Seed 2: +0.04
        res = classify_p35_a_results(
            cand1, ctrl1, substrate_regression_fraction=0.01,
            seed2_treatment_eval=cand2, seed2_control_eval=ctrl2,
        )
        self.assertEqual(res.category, P35ResultCategory.SEED_UNSTABLE)
        self.assertIn("INVESTIGATE_OPTIMIZATION_VARIANCE", res.precommitted_next_action)

    def test_classify_robust_positive_replicated(self) -> None:
        ctrl1 = _make_eval(raw_core=0.20, natural=0.20)
        cand1 = _make_eval(raw_core=0.55, natural=0.45)  # Seed 1: +0.35
        ctrl2 = _make_eval(raw_core=0.20, natural=0.20)
        cand2 = _make_eval(raw_core=0.53, natural=0.43)  # Seed 2: +0.33
        res = classify_p35_a_results(
            cand1, ctrl1, substrate_regression_fraction=0.01,
            seed2_treatment_eval=cand2, seed2_control_eval=ctrl2,
        )
        self.assertEqual(res.category, P35ResultCategory.ROBUST_POSITIVE)
        self.assertTrue(res.two_seed_replicated)
        self.assertIn("AUTHORIZE_P35_B", res.precommitted_next_action)


if __name__ == "__main__":
    unittest.main()