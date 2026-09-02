from __future__ import annotations

import unittest

from e2_architecture.scoring_policy_tournament import (
    _equivalence,
    _holm_decisions,
    _student_t_cdf,
    _synthetic_checks,
)


class E2ScoringPolicyTournamentTests(unittest.TestCase):
    def test_student_t_reference_points(self) -> None:
        self.assertAlmostEqual(_student_t_cdf(0.0, 4), 0.5, places=12)
        self.assertAlmostEqual(_student_t_cdf(2.131846786, 4), 0.95, places=8)
        self.assertAlmostEqual(_student_t_cdf(-2.131846786, 4), 0.05, places=8)

    def test_equivalence_is_clustered_and_requires_every_seed(self) -> None:
        centered = _equivalence([1 / 3] * 5)
        self.assertTrue(centered["inside_equivalence_margin"])
        self.assertTrue(centered["every_seed_inside_margin"])
        self.assertEqual(centered["tost_p_value"], 0.0)
        outlier = _equivalence([1 / 3, 1 / 3, 1 / 3, 1 / 3, 0.5])
        self.assertFalse(outlier["every_seed_inside_margin"])
        self.assertFalse(outlier["inside_equivalence_margin"])

    def test_holm_is_step_down_and_never_skips_a_failure(self) -> None:
        decisions = _holm_decisions({"a": 0.001, "b": 0.006, "c": 0.007})
        self.assertTrue(decisions["a"])
        self.assertFalse(decisions["b"])
        self.assertFalse(decisions["c"])

    def test_interventions_preserve_valid_logprob_domain_and_recover(self) -> None:
        checks = _synthetic_checks()
        self.assertEqual(checks["injection_recovery"], 1.0)
        self.assertEqual(checks["swap_recovery"], 1.0)
        # All three target roles must be exercised, and the rotation gates
        # must catch a deliberately position-biased selector (vacuous-gate tripwire).
        self.assertTrue(checks["all_three_roles_injected"])
        self.assertEqual(checks["position_bias_negative_control_caught"], 1.0)


if __name__ == "__main__":
    unittest.main()
