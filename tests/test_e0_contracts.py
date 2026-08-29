from __future__ import annotations

import dataclasses
import unittest

from e0_cognition.baselines import evaluate_all_baselines
from e0_cognition.certify import build_development_certificate
from e0_cognition.contracts import PairKind, Split, assert_split_disjoint
from e0_cognition.evaluation_generators import build_evaluation_suite
from e0_cognition.metrics import measure_realization, measure_selection, query_conditioning_lift
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
        self.assertEqual(len(self.dev.cases), 88)
        self.assertEqual(len(self.dev.pairs), 24)

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
                "direct_retrieval_control", "full_truth_oracle",
            },
        )
        self.assertLess(results["deterministic_random"]["accuracy"], 0.5)
        self.assertEqual(results["full_truth_oracle"]["accuracy"], 1.0)
        self.assertLess(results["broken_state_tracker"]["by_family"]["state_overwrite"], 0.25)

    def test_representation_selection_realization_metrics_are_separate(self) -> None:
        selected = measure_selection({"A": -2.0, "B": -0.5, "C": -1.0}, "C")
        self.assertEqual(selected.rank, 2)
        self.assertAlmostEqual(selected.margin, -0.5)
        self.assertAlmostEqual(query_conditioning_lift({"C": 1.5}, {"C": 0.25}, "C"), 1.25)
        realized = measure_realization("C.", "C", "C")
        self.assertFalse(realized.raw_exact)
        self.assertTrue(realized.constrained_exact)
        self.assertEqual(realized.conditional_realization, 0.0)

    def test_development_certificate_passes_without_claiming_model_quality(self) -> None:
        certificate = build_development_certificate(seed=505, groups_per_family=16)
        self.assertEqual(certificate["status"], "PASS")
        self.assertIn("not a V5 model result", certificate["scope"])
        self.assertFalse(certificate["sealed_policy"]["seed_in_repository"])

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
