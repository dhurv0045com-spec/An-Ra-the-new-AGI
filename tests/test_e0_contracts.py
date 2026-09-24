from __future__ import annotations

import dataclasses
import json
import re
import unittest
from collections import Counter
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
    TRAINING_COGNITION_FAMILIES,
    assert_training_eval_disjoint,
    build_training_examples,
)
from v5_training.production_entry import frozen_cognition_fractions


class E0ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.groups_per_family = 4
        cls.dev = build_evaluation_suite(
            Split.DEVELOPMENT, seed=101, groups_per_family=cls.groups_per_family
        )

    def test_suite_is_valid_and_deterministic(self) -> None:
        self.dev.assert_valid()
        again = build_evaluation_suite(Split.DEVELOPMENT, seed=101, groups_per_family=4)
        self.assertEqual(self.dev.sha256(), again.sha256())
        self.assertEqual(len(self.dev.cases), 140)
        self.assertEqual(len(self.dev.pairs), 52)

    def test_interference_retrieval_covers_the_frozen_dose_and_position_grid(self) -> None:
        expected = {
            (0, 1),
            (2, 1), (2, 2), (2, 3),
            (4, 1), (4, 2), (4, 3), (4, 4),
            (8, 1), (8, 2), (8, 3), (8, 4),
            (16, 1), (16, 2), (16, 3), (16, 4),
            (32, 1), (32, 2), (32, 3), (32, 4),
        }
        cases = [case for case in self.dev.cases if case.family == "interference_retrieval"]
        observed = Counter(
            (
                dict(case.difficulty)["distractors"],
                dict(case.difficulty)["context_position_quartile"],
            )
            for case in cases
        )

        self.assertEqual(set(observed), expected)
        self.assertEqual(set(observed.values()), {2})
        self.assertEqual({dict(case.difficulty)["hops"] for case in cases}, {2})
        self.assertEqual(
            sum(pair.base.family == "interference_retrieval" for pair in self.dev.pairs),
            len(expected),
        )

    def test_faithful_realization_checks_exact_payload_revision_surface(self) -> None:
        cases = [case for case in self.dev.cases if case.family == "faithful_realization"]
        pairs = [pair for pair in self.dev.pairs if pair.base.family == "faithful_realization"]
        self.assertEqual(len(cases), 2 * self.groups_per_family)
        self.assertEqual(len(pairs), self.groups_per_family)
        for case in cases:
            self.assertRegex(case.answer, r"^payload=[A-Z0-9-]+; revision=[1-8]$")
            self.assertIn("payload=<code>; revision=<number>", case.query)
            self.assertIn(case.answer, case.candidates)
            self.assertEqual(dict(case.surface_axes)["realization_format"], "payload-revision-v1")
            self.assertEqual(set(case.model_view()), {"context", "query", "prompt"})
        for pair in pairs:
            self.assertEqual(pair.kind, PairKind.RELEVANT_FACT_SWAP)
            pair.assert_contract()

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

    def test_training_generator_covers_frozen_cognition_contract(self) -> None:
        fractions = frozen_cognition_fractions()
        training = build_training_examples(
            seed=2409, count=270, family_fractions=fractions
        )
        self.assertEqual(len(training), 270)
        self.assertEqual({example.family for example in training}, set(TRAINING_COGNITION_FAMILIES))
        self.assertEqual(
            {example.template_id for example in training},
            {f"train.causal.{family}" for family in TRAINING_COGNITION_FAMILIES},
        )
        self.assertEqual(
            sum(fractions.values()),
            1.0,
        )
        for family in TRAINING_COGNITION_FAMILIES:
            family_examples = [example for example in training if example.family == family]
            surfaces = {example.surface for example in family_examples}
            self.assertTrue({"natural", "semi_natural"}.issubset(surfaces))
            self.assertGreaterEqual(
                sum(example.surface in {"natural", "semi_natural"} for example in family_examples),
                (len(family_examples) + 3) // 4,
            )
        band_counts = {
            band: sum(example.difficulty_band == band for example in training)
            for band in ("easy", "medium", "hard")
        }
        self.assertEqual(sum(band_counts.values()), len(training))
        self.assertTrue(all(value > 0 for value in band_counts.values()))
        by_family = {
            family: [example for example in training if example.family == family]
            for family in TRAINING_COGNITION_FAMILIES
        }
        difficulty_of = lambda example: dict(example.difficulty)
        self.assertTrue(
            {difficulty_of(example)["cardinality"] for example in by_family["query_binding"]}
            >= {2, 4, 8, 16}
        )
        self.assertTrue(
            {difficulty_of(example)["distractors"] for example in by_family["interference_retrieval"]}
            >= {0, 2, 4, 8, 16, 32}
        )
        expected_interference_grid = {(0, 1)} | {
            (2, quartile) for quartile in (1, 2, 3)
        } | {
            (dose, quartile)
            for dose in (4, 8, 16, 32)
            for quartile in (1, 2, 3, 4)
        }
        self.assertEqual(
            {
                (difficulty_of(example)["distractors"], difficulty_of(example)["context_position_quartile"])
                for example in by_family["interference_retrieval"]
            },
            expected_interference_grid,
        )
        self.assertEqual(
            {difficulty_of(example)["context_position_quartile"] for example in by_family["interference_retrieval"]},
            {1, 2, 3, 4},
        )
        state_examples = by_family["semantic_state"]
        self.assertEqual(
            {dict(example.surface_axes)["state_query"] for example in state_examples},
            {"latest", "intermediate", "rollback", "precedence"},
        )
        self.assertTrue(
            {difficulty_of(example)["state_variables"] for example in by_family["semantic_state"]}
            >= {1, 2, 4}
        )
        self.assertTrue(
            {difficulty_of(example)["state_updates"] for example in by_family["semantic_state"]}
            >= {2, 4, 8}
        )
        self.assertTrue(
            {difficulty_of(example)["hops"] for example in by_family["relational_composition"]}
            >= {1, 2, 3}
        )
        self.assertTrue(
            {difficulty_of(example)["rule_demonstrations"] for example in by_family["heldout_rule_induction"]}
            >= {2, 4, 8}
        )
        for example in training:
            self.assertEqual(set(example.model_view()), {"context", "query"})
            self.assertNotIn(example.family, example.model_view()["query"])
            self.assertNotIn("difficulty_level", example.model_view()["context"])
            self.assertTrue(example.answer)
            self.assertTrue(example.counterfactual_answer)
            self.assertNotIn("causal_graph", example.model_view())
            self.assertNotIn("counterfactual_answer", example.model_view())
            self.assertNotIn("relevant_variables", example.model_view())
        assert_training_eval_disjoint(
            training, {case.template_id for case in self.dev.cases}
        )

    def test_training_family_targets_match_their_executable_evidence(self) -> None:
        training = build_training_examples(
            seed=90210, count=270, family_fractions=frozen_cognition_fractions()
        )
        for example in training:
            with self.subTest(family=example.family, example=example.example_id):
                context = example.model_view()["context"]
                if example.family == "identity_copy":
                    self.assertIn(example.answer, context)
                elif example.family == "query_binding":
                    entity = re.search(r"account-(R\d+)", example.query).group(1)
                    bindings = {
                        subject: obj
                        for subject, relation, obj in example.causal_graph
                        if relation == "has-payload"
                    }
                    self.assertEqual(bindings[f"account-{entity}"], example.answer)
                elif example.family == "semantic_state":
                    entity = example.relevant_variables[0]
                    cutoff = int(re.findall(r"minute (\d+)", example.query)[-1])
                    events = []
                    for subject, relation, obj in example.causal_graph:
                        match = re.fullmatch(
                            re.escape(entity) + r"@minute-(\d+)-priority-(\d+)", subject
                        )
                        if not match:
                            continue
                        minute, priority = map(int, match.groups())
                        rollback = re.fullmatch(r"rolls-back-to-minute-(\d+)", relation)
                        events.append(
                            (minute, priority, obj, int(rollback.group(1)) if rollback else None)
                        )

                    def value_at(time):
                        eligible = sorted(
                            (event for event in events if event[0] <= time),
                            key=lambda event: (event[0], event[1]),
                        )
                        self.assertTrue(eligible)
                        _minute, _priority, value, rollback_time = eligible[-1]
                        return value_at(rollback_time) if rollback_time is not None else value

                    self.assertEqual(value_at(cutoff), example.answer)
                    query_kind = dict(example.surface_axes)["state_query"]
                    self.assertIn(query_kind, {"latest", "intermediate", "rollback", "precedence"})
                    if query_kind == "rollback":
                        self.assertTrue(any(rollback is not None for *_prefix, rollback in events))
                    if query_kind == "precedence":
                        self.assertTrue(
                            any(sum(event[0] == minute for event in events) > 1 for minute in {e[0] for e in events})
                        )
                elif example.family == "interference_retrieval":
                    subject = re.search(r"(shipment-R\d+-north)", example.query).group(1)
                    retrieval = {
                        source: obj
                        for source, relation, obj in example.causal_graph
                        if relation == "contains-payload"
                    }
                    self.assertEqual(retrieval[subject], example.answer)
                    shadow = subject.removesuffix("-north") + "-south"
                    expected_counterfactual = retrieval.get(shadow, "<MISSING>")
                    self.assertEqual(example.counterfactual_answer, expected_counterfactual)
                elif example.family == "relational_composition":
                    relations = {
                        (subject, relation): obj
                        for subject, relation, obj in example.causal_graph
                    }
                    start = example.causal_graph[0][0]
                    current = start
                    while (current, "routes-to") in relations:
                        current = relations[(current, "routes-to")]
                    self.assertEqual(relations[(current, "stores-payload")], example.answer)
                elif example.family == "counterfactual_sensitivity":
                    self.assertNotEqual(example.answer, example.counterfactual_answer)
                    self.assertIn((example.relevant_variables[0], "intervention-sets-value", example.answer), example.causal_graph)
                elif example.family == "heldout_rule_induction":
                    relation = example.causal_graph[0][1]
                    step = int(relation.removeprefix("latent-add-").removesuffix("-mod-10"))
                    heldout_digit = int(example.query.split("Apply the demonstrated rule to ", 1)[1].split(".", 1)[0])
                    self.assertEqual(example.answer, str((heldout_digit + step) % 10))
                    self.assertNotIn((str(heldout_digit), relation, example.answer), example.causal_graph)
                elif example.family == "missing_information":
                    self.assertEqual(example.answer, "<MISSING>")
                    absent_key = example.relevant_variables[0]
                    self.assertFalse(any(absent_key in (subject, obj) for subject, _relation, obj in example.causal_graph))
                elif example.family == "faithful_realization":
                    self.assertRegex(example.answer, r"^payload=TX\d{6}; revision=\d+$")
                else:
                    self.fail(f"unrecognized cognition family {example.family}")

    def test_training_generator_refuses_a_count_that_cannot_cover_every_family(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 9"):
            build_training_examples(seed=9, count=8)

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
                self.assertEqual(result["null_method"], "hypothetical-random-serialization")
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
