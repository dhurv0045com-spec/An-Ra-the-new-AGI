from __future__ import annotations

import dataclasses
import json
import re
import unittest
from pathlib import Path

from e0_cognition.baselines import evaluate_all_baselines
from e0_cognition.certify import build_development_certificate
from e0_cognition.contracts import PairKind, Split, assert_split_disjoint
from e0_cognition.evaluation_generators import build_evaluation_suite
from e0_cognition.metrics import (
    measure_assistance,
    measure_pair_behavior,
    measure_realization,
    measure_selection,
    query_conditioning_lift,
    selection_eligible,
)
from e0_cognition.reference_solvers import assert_reference_solver_agreement
from e0_cognition.statistics import (
    approximate_two_proportion_n_per_arm,
    uniform_candidate_chance,
    wilson_interval,
)
from e0_cognition.training_generators import (
    assert_training_eval_disjoint,
    build_training_examples,
)


class E0ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dev = build_evaluation_suite(Split.DEVELOPMENT, seed=101, groups_per_family=4)

    def test_suite_is_valid_and_deterministic(self) -> None:
        self.dev.assert_valid()
        again = build_evaluation_suite(Split.DEVELOPMENT, seed=101, groups_per_family=4)
        self.assertEqual(self.dev.sha256(), again.sha256())
        self.assertEqual(len(self.dev.cases), 92)
        self.assertEqual(len(self.dev.pairs), 28)

    def test_counterfactual_pair_kinds_are_present(self) -> None:
        self.assertEqual({pair.kind for pair in self.dev.pairs}, set(PairKind))

    def test_tampered_pair_fails_mechanical_contract(self) -> None:
        pair = next(pair for pair in self.dev.pairs if pair.kind is PairKind.QUERY_SWAP)
        bad_changed = dataclasses.replace(pair.changed, facts=pair.changed.facts + ("extra",))
        with self.assertRaises(AssertionError):
            dataclasses.replace(pair, changed=bad_changed).assert_contract()

    def test_candidates_are_fixed_across_every_pair(self) -> None:
        for pair in self.dev.pairs:
            self.assertEqual(pair.base.candidates, pair.changed.candidates)

    def test_model_view_excludes_truth_and_candidates(self) -> None:
        view = self.dev.cases[0].model_view()
        self.assertEqual(set(view), {"context", "query", "prompt"})
        self.assertNotIn("answer", view)
        self.assertNotIn("hidden", view)
        self.assertNotIn("candidates", view)
        self.assertNotIn("surface_axes", view)

    def test_context_position_and_output_axes_are_covered(self) -> None:
        axes = self.dev.surface_axis_histograms()
        self.assertTrue(
            {"front", "middle", "back", "distributed", "answer-absent"}.issubset(
                axes["relevant_position"]
            )
        )
        self.assertGreaterEqual(len(axes["answer_format"]), 5)

    def test_split_vocabularies_are_disjoint(self) -> None:
        sealed = build_evaluation_suite(Split.SEALED, seed=202, groups_per_family=2)
        fresh = build_evaluation_suite(Split.FRESH, seed=303, groups_per_family=2)
        assert_split_disjoint((self.dev, sealed, fresh))

    def test_independent_surface_solver_agrees(self) -> None:
        assert_reference_solver_agreement(self.dev)

    def test_generator_contracts_across_many_seeds(self) -> None:
        for seed in range(20):
            suite = build_evaluation_suite(Split.DEVELOPMENT, seed=10_000 + seed, groups_per_family=1)
            suite.assert_valid()
            assert_reference_solver_agreement(suite)

    def test_sealed_seed_has_no_zero_default(self) -> None:
        with self.assertRaises(ValueError):
            build_evaluation_suite(Split.SEALED, seed=0)

    def test_training_and_evaluation_namespaces_are_disjoint(self) -> None:
        training = build_training_examples(seed=404, count=32)
        assert_training_eval_disjoint(training, {case.template_id for case in self.dev.cases})
        self.assertTrue(all("answer" not in example.model_view() for example in training))

    def test_non_neural_baselines_are_reported(self) -> None:
        results = evaluate_all_baselines(self.dev)
        self.assertEqual(
            set(results),
            {
                "deterministic_random", "first_candidate", "last_candidate", "lexical_overlap",
                "latest_fact", "nearest_position", "bag_of_words", "broken_state_tracker",
                "fixed_reverse_rule", "fixed_identity_rule", "fixed_repeat_left_rule",
                "fixed_repeat_right_rule", "direct_retrieval_control", "full_truth_oracle",
            },
        )
        self.assertLess(results["deterministic_random"]["accuracy"], 0.5)
        self.assertEqual(results["full_truth_oracle"]["accuracy"], 1.0)
        self.assertLess(results["broken_state_tracker"]["by_family"]["state_overwrite"], 0.25)

    def test_state_semantics_are_not_serialization_order(self) -> None:
        state_cases = [
            case
            for case in self.dev.cases
            if case.family in {"state_overwrite", "natural_state_analogue"}
        ]
        self.assertEqual({dict(case.surface_axes)["state_query"] for case in state_cases},
                         {"latest", "intermediate", "rollback", "precedence"})
        self.assertEqual(
            {dict(case.surface_axes)["query_time_relation"] for case in state_cases},
            {"between-events", "after-events"},
        )
        self.assertEqual({dict(case.surface_axes)["variable_interleaving"] for case in state_cases},
                         {"two-variable"})
        self.assertTrue(all(len(case.hidden.relevant_fact_indices) >= 1 for case in state_cases))
        for case in state_cases:
            query_time = int(re.search(r"(?:time|minute) (\d+)", case.query).group(1))
            event_times = {
                int(match.group(1))
                for fact in case.facts
                for match in [re.search(r"(?:time=|minute )(\d+)", fact)]
                if match
            }
            self.assertNotIn(query_time, event_times)

    def test_rule_structures_are_multiple_and_split_held_out(self) -> None:
        dev_structures = {
            dict(case.surface_axes)["rule_structure"]
            for case in self.dev.cases
            if case.family == "rule_induction"
        }
        self.assertGreaterEqual(len(dev_structures), 4)
        sealed = build_evaluation_suite(Split.SEALED, seed=202, groups_per_family=4)
        fresh = build_evaluation_suite(Split.FRESH, seed=303, groups_per_family=4)
        for other in (sealed, fresh):
            other_structures = {
                dict(case.surface_axes)["rule_structure"]
                for case in other.cases
                if case.family == "rule_induction"
            }
            self.assertTrue(dev_structures.isdisjoint(other_structures))

    def test_pair_sensitivity_invariance_and_assistance_are_separate(self) -> None:
        predictions = {case.case_id: case.answer for case in self.dev.cases}
        pair = measure_pair_behavior(self.dev, predictions)
        self.assertGreater(pair.sensitivity_total, 0)
        self.assertGreater(pair.invariance_total, 0)
        self.assertEqual(pair.sensitivity_both_correct, pair.sensitivity_total)
        self.assertEqual(pair.invariance_stable, pair.invariance_total)
        assistance = measure_assistance("wrong", "right", "right")
        self.assertTrue(assistance.intervention_dependence)
        self.assertFalse(assistance.assistance_harm)

    def test_copy_controls_are_realization_only(self) -> None:
        copies = [case for case in self.dev.cases if case.family == "exact_contextual_copy"]
        self.assertTrue(copies)
        self.assertTrue(all(not selection_eligible(case) for case in copies))

    def test_representation_selection_realization_metrics_are_separate(self) -> None:
        selected = measure_selection({"A": -2.0, "B": -0.5, "C": -1.0}, "C")
        self.assertEqual(selected.rank, 2)
        self.assertAlmostEqual(selected.margin, -0.5)
        self.assertAlmostEqual(query_conditioning_lift({"C": 1.5}, {"C": 0.25}, "C"), 1.25)
        realized = measure_realization(
            "C.", "C", "C", unassisted_selection_correct=False
        )
        self.assertFalse(realized.raw_exact)
        self.assertTrue(realized.constrained_exact)
        self.assertIsNone(realized.conditional_realization)
        selected_realization = measure_realization(
            "C", "C", "C", unassisted_selection_correct=True
        )
        self.assertEqual(selected_realization.conditional_realization, 1.0)

    def test_development_certificate_passes_without_claiming_model_quality(self) -> None:
        certificate = build_development_certificate(seed=505, groups_per_family=16)
        self.assertEqual(certificate["status"], "PASS")
        self.assertIn("not a V5 model result", certificate["scope"])
        self.assertFalse(certificate["sealed_policy"]["seed_in_repository"])

    def test_state_shortcut_gate_covers_lexical_and_position_heuristics(self) -> None:
        certificate = build_development_certificate(seed=88, groups_per_family=16)
        self.assertEqual(certificate["status"], "PASS")
        audit = certificate["shortcut_audit"]["state_heuristics"]
        self.assertEqual(
            set(audit),
            {
                "first_candidate",
                "last_candidate",
                "latest_fact",
                "nearest_position",
                "lexical_overlap",
                "bag_of_words",
            },
        )
        for name, result in audit.items():
            if name in {"latest_fact", "nearest_position"}:
                self.assertEqual(result["null_method"], "analytic-random-serialization")
            else:
                self.assertEqual(result["null_method"], "casewise-uniform-candidate")
            self.assertLessEqual(result["accuracy"], result["calibrated_chance"] + 0.10)

    def test_shortcut_repair_receipt_matches_canonical_certificate(self) -> None:
        root = Path(__file__).parents[1]
        receipt = json.loads(
            (root / "artifacts/e0/shortcut_repair_receipt.json").read_text(encoding="utf-8")
        )
        certificate = json.loads(
            (root / "artifacts/e0/development_certificate.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["schema"], "esoes-e0-shortcut-repair/v2")
        self.assertEqual(receipt["after"]["suite_sha256"], certificate["suite"]["sha256"])
        self.assertEqual(
            receipt["after"]["generator_version"], certificate["suite"]["generator_version"]
        )
        self.assertEqual(
            receipt["after"]["state_casewise_chance"],
            certificate["shortcut_audit"]["state_heuristics"]["bag_of_words"]["chance"],
        )
        for name, accuracy in receipt["after"]["state_heuristics"].items():
            self.assertEqual(
                accuracy,
                certificate["shortcut_audit"]["state_heuristics"][name]["accuracy"],
            )
        self.assertGreater(
            receipt["false_green"]["bag_of_words_pooled_state_accuracy"], 0.8
        )

    def test_statistical_calibration_is_explicit(self) -> None:
        chance = uniform_candidate_chance(self.dev)
        self.assertGreater(chance, 0.0)
        self.assertLess(chance, 1.0)
        lower, upper = wilson_interval(50, 100)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)
        self.assertGreater(approximate_two_proportion_n_per_arm(0.25, 0.35), 100)


if __name__ == "__main__":
    unittest.main()
