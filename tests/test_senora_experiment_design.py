"""Unit tests for senora.experiment_design."""

from __future__ import annotations

import unittest

from senora.experiment_design import P35_MODEL_SPEC, build_p35_cms1_plan


class TestExperimentDesign(unittest.TestCase):
    def test_p35_model_spec_parameters(self) -> None:
        receipt = P35_MODEL_SPEC.parameter_receipt()
        # 35,411,328 exact parameters under 2:1 GQA with affine QK-norm
        self.assertEqual(receipt.total, 35_411_328)
        self.assertEqual(P35_MODEL_SPEC.layers, 16)
        self.assertEqual(P35_MODEL_SPEC.width, 384)
        self.assertEqual(P35_MODEL_SPEC.query_heads, 6)
        self.assertEqual(P35_MODEL_SPEC.kv_heads, 3)  # 2:1 GQA

    def test_p35_cms1_plan_structure_and_matching(self) -> None:
        plan = build_p35_cms1_plan()
        self.assertEqual(plan.experiment_id, "P35-CMS-1")
        self.assertEqual(len(plan.arms), 3)

        arm_names = [arm["name"] for arm in plan.arms]
        self.assertEqual(arm_names, [
            "control-substrate-00",
            "cognition-mixture-15-ce",
            "cognition-mixture-15-qswap",
        ])

        # Verify exact token and FLOP matching across all arms
        token_budgets = {arm["token_budget"] for arm in plan.arms}
        flop_budgets = {arm["idealized_6nd_flops"] for arm in plan.arms}
        self.assertEqual(len(token_budgets), 1)
        self.assertEqual(len(flop_budgets), 1)
        self.assertEqual(token_budgets.pop(), 50_000_000)

        digest = plan.sha256()
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()