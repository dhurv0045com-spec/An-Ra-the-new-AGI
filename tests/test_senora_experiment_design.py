"""Unit tests for senora.experiment_design."""

from __future__ import annotations

import unittest

from senora.experiment_design import build_p35_cms1_plan
from senora.model import EXPECTED_P35_PARAMETER_COUNT


class TestSenoraExperimentDesign(unittest.TestCase):
    def test_plan_sequential_structure_and_arms(self) -> None:
        plan = build_p35_cms1_plan()
        self.assertEqual(plan.experiment_id, "P35-CMS-1")
        self.assertEqual(len(plan.arms), 3)

        # Arm 0: Phase P35-A, Control Substrate (0% cognition)
        arm0 = plan.arms[0]
        self.assertEqual(arm0["name"], "control-substrate-00")
        self.assertEqual(arm0["phase"], "P35-A")
        self.assertEqual(arm0["cognition_fraction"], 0.0)
        self.assertAlmostEqual(arm0["natural_fraction"], 65.0 / 85.0, places=4)
        self.assertAlmostEqual(arm0["code_fraction"], 20.0 / 85.0, places=4)
        self.assertEqual(arm0["matching_basis"], "FLOP_MATCHED")

        # Arm 1: Phase P35-A, 15% Cognition CE
        arm1 = plan.arms[1]
        self.assertEqual(arm1["name"], "cognition-mixture-15-ce")
        self.assertEqual(arm1["phase"], "P35-A")
        self.assertEqual(arm1["cognition_fraction"], 0.15)
        self.assertAlmostEqual(arm1["natural_fraction"], 0.65, places=5)
        self.assertAlmostEqual(arm1["code_fraction"], 0.20, places=5)
        self.assertEqual(arm1["matching_basis"], "FLOP_MATCHED")

        # Arm 2: Phase P35-B, 15% Cognition CE + Query-Swap
        arm2 = plan.arms[2]
        self.assertEqual(arm2["name"], "cognition-mixture-15-qswap")
        self.assertEqual(arm2["phase"], "P35-B")
        self.assertEqual(arm2["query_swap_lambda"], 0.10)
        self.assertEqual(arm2["matching_basis"], "TOKEN_MATCHED")

    def test_exact_flop_accounting(self) -> None:
        plan = build_p35_cms1_plan()
        expected_param_count = EXPECTED_P35_PARAMETER_COUNT  # 35,411,328
        expected_flops = 6 * expected_param_count * 50_000_000

        self.assertEqual(plan.arms[0]["idealized_6nd_flops"], expected_flops)
        self.assertEqual(plan.arms[1]["idealized_6nd_flops"], expected_flops)
        self.assertEqual(plan.arms[0]["idealized_6nd_flops"], 10_623_398_400_000_000)

    def test_development_vs_prospective_separation(self) -> None:
        plan = build_p35_cms1_plan()
        self.assertIn("Split.DEVELOPMENT", plan.development_evaluation_suite)
        self.assertIn("Split.FRESH", plan.prospective_confirmation_suite)


if __name__ == "__main__":
    unittest.main()