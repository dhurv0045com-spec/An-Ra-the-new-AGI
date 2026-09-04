"""Binding generator v2: contract, controls, suite, qualification."""

from __future__ import annotations

import unittest

from e0_cognition.binding_v2 import (
    FRESH_GRAMMARS,
    GRAMMARS,
    generate_group,
    interference_pair_control,
    pair_destroyed_control,
    queried_entity,
    truth_solver,
    value_swap_control,
)
from e0_cognition.shortcut_suite import (
    FrequencyPrior,
    TemplatePrior,
    pair_sensitivity,
    predict_bag_of_words,
    qualify_pairs,
    run_suite,
)


def _clean(seed=11, groups=range(6), cardinality=3, split="development"):
    cases, aux = [], None
    out = []
    for gi in groups:
        group_cases, aux = generate_group(
            seed=seed, group_index=gi, cardinality=cardinality, split=split, mode="clean")
        out.extend(group_cases)
    return out, aux


def _interference(seed=21, groups=range(6), cardinality=3, split="development"):
    out = []
    histories = {}
    for gi in groups:
        group_cases, aux = generate_group(
            seed=seed, group_index=gi, cardinality=cardinality, split=split,
            mode="interference")
        out.extend(group_cases)
        histories.update(aux["histories"])
    return out, histories


def _task(case, **extra):
    task = {
        "case_id": case.case_id, "facts": list(case.facts),
        "facts_text": "\n".join(case.facts), "query": case.query,
        "candidates": list(case.candidates), "gold": case.gold,
        "cluster_id": case.cluster_id, "grammar": case.grammar,
    }
    task.update(extra)
    return task


class BindingV2ContractTests(unittest.TestCase):
    def test_deterministic_and_truth_perfect(self) -> None:
        first, aux1 = generate_group(seed=11, group_index=2, cardinality=3, split="development")
        second, _aux2 = generate_group(seed=11, group_index=2, cardinality=3, split="development")
        self.assertEqual([c.case_id for c in first], [c.case_id for c in second])
        pairing = aux1["pairing"]
        for case in first:
            self.assertEqual(truth_solver(case, pairing=pairing), case.gold)
            self.assertEqual(case.split, "development")

    def test_controls_change_gold_with_ancestry(self) -> None:
        cases, aux = generate_group(seed=11, group_index=0, cardinality=3, split="development")
        pairing = aux["pairing"]
        case = cases[0]
        for control in (
            pair_destroyed_control(case, pairing=pairing),
            value_swap_control(case, pairing=pairing),
        ):
            self.assertNotEqual(control.gold, case.gold)
            self.assertIn(control.gold, control.candidates)
            self.assertEqual(control.control_of, case.cluster_id)
            self.assertNotEqual(control.cluster_id, case.cluster_id)

    def test_query_groups_cover_all_entities(self) -> None:
        cases, aux = generate_group(seed=11, group_index=1, cardinality=4, split="development")
        pairing = aux["pairing"]
        queried = sorted(queried_entity(case, pairing) for case in cases)
        self.assertEqual(queried, sorted(entity for entity, _ in pairing))

    def test_candidates_fully_attested(self) -> None:
        cases, _aux = generate_group(seed=11, group_index=0, cardinality=3, split="development")
        for case in cases:
            context = "\n".join(case.facts)
            for candidate in case.candidates:
                self.assertIn(candidate, context)

    def test_positions_span_context(self) -> None:
        thirds = set()
        for gi in range(9):
            cases, _aux = generate_group(seed=11, group_index=gi, cardinality=3, split="development")
            for case in cases:
                thirds.add(case.target_position // 1)
        self.assertGreaterEqual(len(thirds), 2)

    def test_lexicon_splits_disjoint(self) -> None:
        train_cases, _ = generate_group(seed=11, group_index=0, cardinality=4, split="training")
        fresh_cases, _ = generate_group(
            seed=11, group_index=0, cardinality=4, split="fresh", structural_fresh=True)
        train_vocab = set(word for case in train_cases for word in " ".join(case.facts).split())
        fresh_vocab = set(word for case in fresh_cases for word in " ".join(case.facts).split())
        train_entities = {w for w in train_vocab if w.startswith("EN-")}
        fresh_entities = {w for w in fresh_vocab if w.startswith("EN-")}
        self.assertFalse(train_entities & fresh_entities)
        self.assertTrue(all(c.grammar in FRESH_GRAMMARS for c in fresh_cases))
        self.assertTrue(all(c.grammar not in FRESH_GRAMMARS for c in train_cases))


class InterferenceQualificationTests(unittest.TestCase):
    def _cohorts(self, seed=21, fit_groups=20, eval_groups=20):
        fit_cases, fit_hist = [], {}
        for gi in range(fit_groups):
            cases, aux = generate_group(seed=seed, group_index=gi, cardinality=3,
                                        split="training", mode="interference")
            fit_cases.extend(cases)
            fit_hist.update(aux["histories"])
        eval_cases, eval_hist = [], {}
        for gi in range(fit_groups, fit_groups + eval_groups):
            cases, aux = generate_group(seed=seed, group_index=gi, cardinality=3,
                                        split="development", mode="interference")
            eval_cases.extend(cases)
            eval_hist.update(aux["histories"])
        fit = [_task(c) for c in fit_cases]
        ev = [_task(c) for c in eval_cases]
        pairs = []
        for case in eval_cases:
            control = interference_pair_control(case, histories=eval_hist)
            base = _task(case)
            pairs.append((base, dict(
                base, case_id=control.case_id, facts=list(control.facts),
                facts_text="\n".join(control.facts), gold=control.gold,
                cluster_id=control.cluster_id)))
        return fit, ev, pairs

    def test_interference_pair_qualification(self) -> None:
        from e0_cognition.shortcut_suite import BASELINES, CentroidProbe, FrequencyPrior, TemplatePrior

        fit, ev, pairs = self._cohorts()
        results = run_suite(fit, ev)
        self.assertEqual(results["truth_solver"]["accuracy"], 1.0)
        accuracies, counts = {}, {}
        for name in sorted(BASELINES):
            if name == "truth_solver":
                continue
            stats = pair_sensitivity(BASELINES[name], pairs)
            accuracies[name] = stats["pair_accuracy"]
            counts[name] = len(pairs)
        value_prior = FrequencyPrior("value")
        value_prior.fit(fit)
        template_prior = TemplatePrior()
        template_prior.fit(fit)
        probe = CentroidProbe()
        try:
            probe.fit(fit)
            fitted = {"value_frequency": value_prior.predict, "surface_template": template_prior.predict,
                      "linear_centroid": probe.predict}
        except ValueError:
            fitted = {"value_frequency": value_prior.predict, "surface_template": template_prior.predict}
        for name, predictor in fitted.items():
            stats = pair_sensitivity(predictor, pairs)
            accuracies[name] = stats["pair_accuracy"]
            counts[name] = len(pairs)
        qual = qualify_pairs(accuracies, counts, null_ceiling=0.25, max_excess=0.10,
                             truth_pair_accuracy=1.0)
        self.assertEqual(qual["verdict"], "GENERATOR_QUALIFIED")

    def test_clean_mode_stays_calibration_only(self) -> None:
        cases, aux = generate_group(seed=11, group_index=0, cardinality=4, split="development")
        pairing = aux["pairing"]
        tasks = [_task(c) for c in cases]
        results = run_suite(tasks, tasks)
        # Sentence co-occurrence solves clean binding: raw accuracy gates
        # must never use this tier for selection.
        self.assertEqual(results["bag_of_words"]["accuracy"], 1.0)
        self.assertEqual(results["truth_solver"]["accuracy"], 1.0)


class TokenizerInteractionTests(unittest.TestCase):
    def test_nonce_fragmentation_is_bounded_and_uniform(self) -> None:
        try:
            from tokenizers import Tokenizer  # noqa: F401
        except ImportError:
            self.skipTest("tokenizers package is not installed")
        import hashlib
        from pathlib import Path as _Path

        from v5_tokenizer.artifact import load_frozen

        root = _Path(__file__).resolve().parents[1]
        tokenizer = load_frozen(
            root / "artifacts/e1/local_tournament/tokenizer-24576.json.gz",
            expected_sha256="97e12db63b343312e5e4abc37df9ef4b01fcb1faba792a6420a4c1b15d0a7fbc",
            vocabulary_size=24576,
            trainer_config_sha256=hashlib.sha256(
                (root / "v5_tokenizer/legacy_24k_trainer_record.json").read_bytes()
            ).hexdigest(),
            corpus_manifest_sha256="eb1f0dbac64524ff4dc589c0292af6dc4c3803f48f8fe0af0a77684fea26fc67",
        )
        entity_lengths, value_lengths = [], []
        for group_index in range(10):
            _cases, aux = generate_group(
                seed=31, group_index=group_index, cardinality=4, split="development")
            for entity, value in aux["pairing"]:
                entity_lengths.append(len(tokenizer.encode(entity)))
                value_lengths.append(len(tokenizer.encode(value)))
        self.assertTrue(all(length <= 6 for length in entity_lengths + value_lengths))
        self.assertLessEqual(max(value_lengths) - min(value_lengths), 3)


class TransferSetTests(unittest.TestCase):
    def test_transfer_is_dev_only_and_well_formed(self) -> None:
        from e0_cognition.binding_transfer import TRANSFER_SPLIT, as_tasks

        tasks = as_tasks()
        self.assertGreaterEqual(len(tasks), 10)
        self.assertEqual(TRANSFER_SPLIT, "development")
        for task in tasks:
            self.assertIn(task["gold"], task["candidates"])
            self.assertTrue(task["facts"] and task["query"])


if __name__ == "__main__":
    unittest.main()
