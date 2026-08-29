from __future__ import annotations

import unittest

from e3_data_objective.plan import SCREEN_TOKENS, build_plan


class E3DataObjectiveTests(unittest.TestCase):
    def test_phase_a_mixtures_have_exact_tokens_and_fixed_non_cognition_ratio(self) -> None:
        plan = build_plan()
        allocations = [arm.mixture.token_allocation(SCREEN_TOKENS) for arm in plan.mixture_arms]
        self.assertEqual(
            allocations,
            [
                {"natural": 145_294_118, "code_math_formal": 44_705_882, "verified_cognition": 10_000_000},
                {"natural": 130_000_000, "code_math_formal": 40_000_000, "verified_cognition": 30_000_000},
                {"natural": 107_058_824, "code_math_formal": 32_941_176, "verified_cognition": 60_000_000},
            ],
        )
        self.assertTrue(all(sum(row.values()) == SCREEN_TOKENS for row in allocations))

    def test_phase_b_has_only_ce_and_preregistered_query_swap_weights(self) -> None:
        plan = build_plan()
        self.assertEqual(
            tuple(arm.query_swap_lambda for arm in plan.objective_arms),
            (0.0, 0.05, 0.15),
        )
        self.assertTrue(
            all(arm.scope == "mechanically-verified-query-swap-pairs-only" for arm in plan.objective_arms)
        )

    def test_plan_status_transitions_fail_closed(self) -> None:
        self.assertEqual(build_plan().status(), "BLOCKED_UPSTREAM_INPUTS")
        digest = "b" * 64
        ready_a = build_plan(
            tokenizer_sha256=digest,
            corpus_manifest_sha256=digest,
            generator_sha256=digest,
            e2_winner_sha256=digest,
            model_constructor_sha256=digest,
            raw_byte_budget=400_000_000,
        )
        self.assertEqual(ready_a.status(), "READY_FOR_PHASE_A_MIXTURE_SCREEN")
        ready_b = build_plan(
            tokenizer_sha256=digest,
            corpus_manifest_sha256=digest,
            generator_sha256=digest,
            e2_winner_sha256=digest,
            model_constructor_sha256=digest,
            raw_byte_budget=400_000_000,
            selected_mixture="cognition-15-ce",
            selected_neighbor="cognition-05-ce",
        )
        self.assertEqual(ready_b.status(), "READY_FOR_PHASE_B_OBJECTIVE_SCREEN")

    def test_phase_b_rejects_nonadjacent_or_partial_selection(self) -> None:
        with self.assertRaises(ValueError):
            build_plan(selected_mixture="cognition-15-ce")
        with self.assertRaises(ValueError):
            build_plan(
                selected_mixture="cognition-05-ce",
                selected_neighbor="cognition-30-ce",
            )

    def test_trace_arm_requires_hashed_transfer_failure(self) -> None:
        with self.assertRaises(ValueError):
            build_plan(trace_arm_enabled=True)
        plan = build_plan(
            trace_arm_enabled=True,
            trace_trigger_receipt_sha256="c" * 64,
        )
        self.assertTrue(plan.trace_arm_enabled)
        self.assertEqual(plan.as_dict()["trace_arm"]["required_evaluation"], "trace-free")

    def test_promotion_gates_reject_narrow_or_assisted_only_wins(self) -> None:
        gates = build_plan().as_dict()["promotion_gates"]
        self.assertEqual(gates["maximum_substrate_loss_regression_fraction"], 0.03)
        self.assertTrue(gates["natural_analogue_transfer_required"])
        self.assertTrue(gates["candidate_free_improvement_required"])
        self.assertTrue(gates["assisted_only_gain_rejected"])


if __name__ == "__main__":
    unittest.main()
