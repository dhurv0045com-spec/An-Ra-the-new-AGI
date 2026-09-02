"""Unit tests for senora.cost_model."""

from __future__ import annotations

import unittest

from senora.cost_model import compute_arm_cost, compute_p35_cms1_budget


class TestSenoraCostModel(unittest.TestCase):
    def test_arm_cost_computation(self) -> None:
        cost = compute_arm_cost(
            arm_name="control-substrate-00",
            phase="P35-A",
            token_budget=50_000_000,
            parameters=35_411_328,
        )
        self.assertEqual(cost.theoretical_6nd_flops, 10_623_398_400_000_000)
        self.assertEqual(cost.effective_training_flops, 10_623_398_400_000_000)
        self.assertAlmostEqual(cost.gpu_hours_median, 0.93, delta=0.05)

    def test_experiment_budget(self) -> None:
        budget = compute_p35_cms1_budget(seeds_per_arm=1)
        self.assertEqual(len(budget.arms), 3)
        self.assertEqual(budget.total_token_budget_per_seed, 150_000_000)
        self.assertGreater(budget.total_effective_flops_per_seed, 30_000_000_000_000_000)
        self.assertLess(budget.total_gpu_hours_optimistic, 2.5)


if __name__ == "__main__":
    unittest.main()