"""B09 focused tests: recomputed scoring, paired metrics, retention,
promotion and pool-separated storage."""
import tempfile
import unittest

from bramastra_lab.research.evaluation.scoring import (
    EvaluationError,
    PromotionConfig,
    RawOutcome,
    brier_score,
    complete_answer_metrics,
    decide_promotion,
    family_metrics,
    paired_goal_metrics,
    retention_report,
)
from bramastra_lab.research.evaluation.store import StoreError


def outcome(outcome_id: str, *, prediction="4", stopped=True, label="4", family="math",
            pool="measurement", cost=1.0, pair=None, role=None, confidence=None,
            split="development"):
    return RawOutcome(outcome_id=outcome_id, pool=pool, split=split, family=family,
                      task_semantic_id=f"task-{outcome_id}", prediction=prediction,
                      stopped_on_eos=stopped, label=label, cost=cost,
                      pair_group_id=pair, role=role, confidence=confidence)


class CompleteAnswerTests(unittest.TestCase):
    def test_exact_and_eos_recomputed(self) -> None:
        metrics = complete_answer_metrics([outcome("a"), outcome("b", prediction="5")])
        self.assertAlmostEqual(metrics["complete_answer_rate"], 0.5)
        self.assertAlmostEqual(metrics["exact_answer_rate"], 0.5)
        self.assertTrue(metrics["recomputed"])

    def test_wrong_eos_cannot_pass(self) -> None:
        # Exact text but stopping invalid: must not pass complete-answer.
        metrics = complete_answer_metrics([outcome("a", stopped=False)])
        self.assertAlmostEqual(metrics["complete_answer_rate"], 0.0)
        self.assertAlmostEqual(metrics["exact_answer_rate"], 1.0)
        self.assertEqual(metrics["exact_but_stopping_invalid"], 1)

    def test_no_success_field_exists_to_forge(self) -> None:
        # Even a hand-forged outcome cannot inject correctness: the raw
        # record carries no success/bool field at all.
        fields = set(outcome("a").to_dict())
        self.assertNotIn("success", fields)
        self.assertNotIn("correct", fields)

    def test_empty_outcomes_reject(self) -> None:
        with self.assertRaises(EvaluationError):
            complete_answer_metrics([])


class PairedMetricsTests(unittest.TestCase):
    def test_pair_metrics_computed(self) -> None:
        pairs = [
            outcome("p1a", prediction="4", pair="g1", role="primary"),
            outcome("p1b", prediction="5", pair="g1", role="swapped", label="6"),
            outcome("p2a", prediction="6", pair="g2", role="primary", label="6"),
            outcome("p2b", prediction="6", pair="g2", role="swapped", label="6"),
        ]
        metrics = paired_goal_metrics(pairs)
        self.assertEqual(metrics["pairs"], 2)
        self.assertAlmostEqual(metrics["both_correct_rate"], 0.5)
        # p1 predicts "4"/"5" (different), p2 predicts "6"/"6" (same).
        self.assertAlmostEqual(metrics["same_answer_rate"], 0.5)
        self.assertAlmostEqual(metrics["primary_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["swapped_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["goal_swap_gap"], 0.5)

    def test_missing_pair_member_rejects(self) -> None:
        with self.assertRaises(EvaluationError):
            paired_goal_metrics([outcome("p1a", pair="g1", role="primary")])

    def test_duplicate_roles_reject(self) -> None:
        with self.assertRaises(EvaluationError):
            paired_goal_metrics([
                outcome("p1a", pair="g1", role="primary"),
                outcome("p1b", pair="g1", role="primary"),
            ])

    def test_unpaired_outcomes_are_ignored_by_paired_metrics(self) -> None:
        with self.assertRaises(EvaluationError):
            paired_goal_metrics([outcome("solo")])


class RetentionTests(unittest.TestCase):
    def test_family_regression_cannot_hide(self) -> None:
        reference = {"math": {"complete_answer_rate": 0.9, "count": 100},
                     "logic": {"complete_answer_rate": 0.9, "count": 100},
                     "vision": {"complete_answer_rate": 0.9, "count": 100}}
        candidate = {"math": {"complete_answer_rate": 1.0, "count": 100},
                     "logic": {"complete_answer_rate": 1.0, "count": 100},
                     "vision": {"complete_answer_rate": 0.8, "count": 100}}
        report = retention_report(reference, candidate, regression_margin=0.02)
        self.assertEqual(report["worst_family"], "vision")
        self.assertEqual(report["regressed_families"], ["vision"])
        # The mean improved, yet the regression is still flagged.
        self.assertGreater(report["mean_delta"], 0.0)

    def test_margin_boundary(self) -> None:
        reference = {"f": {"complete_answer_rate": 0.5, "count": 10}}
        at_margin = {"f": {"complete_answer_rate": 0.48, "count": 10}}
        just_inside = {"f": {"complete_answer_rate": 0.485, "count": 10}}
        self.assertEqual(
            retention_report(reference, at_margin, regression_margin=0.02)
            ["regressed_families"], ["f"])
        self.assertEqual(
            retention_report(reference, just_inside, regression_margin=0.02)
            ["regressed_families"], [])


class PromotionTests(unittest.TestCase):
    @staticmethod
    def build_inputs(*, cand_primary=0.9, ref_primary=0.8, logic_delta=0.0,
                     pairs=8, both_correct=0.9, evidence=True, sealed_fresh=True):
        reference_metrics = {"complete_answer_rate": ref_primary}
        candidate_metrics = {"complete_answer_rate": cand_primary}
        reference_family = {"math": {"complete_answer_rate": ref_primary, "count": 10},
                            "logic": {"complete_answer_rate": 0.9, "count": 10}}
        candidate_family = {
            "math": {"complete_answer_rate": cand_primary, "count": 10},
            "logic": {"complete_answer_rate": 0.9 + logic_delta, "count": 10}}
        paired = {"pairs": pairs, "both_correct_rate": both_correct,
                  "goal_swap_gap": 0.4}
        decision = decide_promotion(
            reference_metrics=reference_metrics, candidate_metrics=candidate_metrics,
            reference_family=reference_family, candidate_family=candidate_family,
            paired=paired, config=PromotionConfig(), evidence_complete=evidence,
            sealed_fresh=sealed_fresh)
        return decision

    def test_accept_on_clear_evidence(self) -> None:
        decision = self.build_inputs()
        self.assertEqual(decision["decision"], "accept")
        self.assertEqual(decision["reasons"], [])

    def test_default_is_no_promotion_without_evidence(self) -> None:
        decision = self.build_inputs(cand_primary=0.80, evidence=False)
        self.assertEqual(decision["decision"], "no_promotion")
        self.assertIn("evidence_incomplete", decision["reasons"])

    def test_small_improvement_does_not_promote(self) -> None:
        decision = self.build_inputs(cand_primary=0.82)
        self.assertIn("primary_margin_not_met", decision["reasons"][0])

    def test_family_regression_blocks_promotion(self) -> None:
        decision = self.build_inputs(logic_delta=-0.4)
        self.assertEqual(decision["decision"], "no_promotion")
        self.assertTrue(any(r.startswith("family_regression") for r in decision["reasons"]))

    def test_sealed_reuse_blocks_acceptance(self) -> None:
        decision = self.build_inputs(sealed_fresh=False)
        self.assertEqual(decision["decision"], "no_promotion")
        self.assertIn("sealed_pool_reuse_detected", decision["reasons"])


class BrierTests(unittest.TestCase):
    def test_brier_computed_when_confidence_declared(self) -> None:
        outcomes = [
            outcome("a", confidence=1.0, prediction="4"),
            outcome("b", confidence=0.0, prediction="4"),
        ]
        report = brier_score(outcomes)
        self.assertAlmostEqual(report["brier"], 0.5)

    def test_brier_absent_without_confidence(self) -> None:
        self.assertIsNone(brier_score([outcome("a")]))


class StoreTests(unittest.TestCase):
    def test_pool_separation_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from bramastra_lab.research.evaluation.store import EvaluationStore

            store = EvaluationStore(tmp)
            store.record([outcome("m1")], pool="measurement", run_id="r1")
            store.record([outcome("s1", pool="sealed")], pool="sealed", run_id="r1")
            measurement_ids = {o.outcome_id for o in store.read("measurement")}
            sealed_ids = {o.outcome_id for o in store.read("sealed")}
            self.assertEqual(measurement_ids, {"m1"})
            self.assertEqual(sealed_ids, {"s1"})
            with self.assertRaises(StoreError):
                store.record([outcome("c1")], pool="controller", run_id="r1")
            with self.assertRaises(StoreError):
                store.read("controller")

    def test_sealed_reuse_is_logged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from bramastra_lab.research.evaluation.store import EvaluationStore

            store = EvaluationStore(tmp)
            self.assertTrue(store.sealed_freshness()["fresh"])
            store.record([outcome("s1", pool="sealed")], pool="sealed", run_id="run-1")
            store.record([outcome("s1", pool="sealed")], pool="sealed", run_id="run-2")
            freshness = store.sealed_freshness()
            self.assertFalse(freshness["fresh"])
            self.assertEqual(freshness["previous_uses"], 2)

    def test_empty_record_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from bramastra_lab.research.evaluation.store import EvaluationStore

            store = EvaluationStore(tmp)
            with self.assertRaises(StoreError):
                store.record([], pool="measurement", run_id="r1")


if __name__ == "__main__":
    unittest.main()
