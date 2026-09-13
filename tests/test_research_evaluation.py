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
            split="development", task_semantic_id=None, case_id=None):
    return RawOutcome(outcome_id=outcome_id, pool=pool, split=split, family=family,
                      task_semantic_id=task_semantic_id or f"task-{outcome_id}",
                      prediction=prediction,
                      stopped_on_eos=stopped, label=label, cost=cost,
                      pair_group_id=pair, role=role, confidence=confidence,
                      case_id=case_id or outcome_id)


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

    def test_case_id_required(self) -> None:
        with self.assertRaises(EvaluationError):
            RawOutcome("a", "measurement", "development", "math", "w",
                       "4", True, "4", 0.0, case_id="")

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
    def build_bundle(*, cand_primary=0.9, ref_primary=0.8, logic_delta=0.0,
                     pairs=8, both_correct=0.9, sealed_fresh=True,
                     paired_goal=True, requires_uncertainty=False,
                     clustered=None, evidence_complete=True):
        from bramastra_lab.research.evaluation.scoring import (EvidenceBundle,
                                                               EvaluationProtocol)

        protocol = EvaluationProtocol(protocol_id="test-protocol",
                                      paired_goal=paired_goal,
                                      requires_uncertainty=requires_uncertainty)
        paired = {"pairs": pairs, "both_correct_rate": both_correct,
                  "goal_swap_gap": 0.4} if paired_goal else None
        return EvidenceBundle(
            parent_identity="parent-1", child_identity="child-1",
            protocol=protocol, pool="confirmation", data_identity="data-1",
            reference_metrics={"complete_answer_rate": ref_primary},
            candidate_metrics={"complete_answer_rate": cand_primary},
            reference_family={"math": {"complete_answer_rate": ref_primary, "count": 10},
                              "logic": {"complete_answer_rate": 0.9, "count": 10}},
            candidate_family={"math": {"complete_answer_rate": cand_primary, "count": 10},
                              "logic": {"complete_answer_rate": 0.9 + logic_delta,
                                        "count": 10}},
            paired=paired,
            clustered_uncertainty=clustered,
            sealed_fresh=sealed_fresh)

    def test_accept_on_clear_evidence(self) -> None:
        from bramastra_lab.research.evaluation.scoring import decide_promotion

        decision = decide_promotion(self.build_bundle(), PromotionConfig())
        self.assertEqual(decision["decision"], "accept")
        self.assertEqual(decision["reasons"], [])

    def test_missing_pair_receipt_is_insufficient_under_paired_protocol(self) -> None:
        from bramastra_lab.research.evaluation.scoring import (decide_promotion,
                                                               EvidenceBundle)

        bundle = self.build_bundle()
        broken = EvidenceBundle(**{**bundle.__dict__, "paired": None})
        decision = decide_promotion(broken, PromotionConfig())
        self.assertEqual(decision["decision"], "no_promotion")
        self.assertIn("missing_required_pair_receipt", decision["reasons"])

    def test_missing_uncertainty_receipt_when_required(self) -> None:
        from bramastra_lab.research.evaluation.scoring import decide_promotion

        bundle = self.build_bundle(requires_uncertainty=True, clustered=None)
        decision = decide_promotion(bundle, PromotionConfig())
        self.assertIn("missing_required_uncertainty_receipt", decision["reasons"])

    def test_pair_criteria_not_applicable_recorded(self) -> None:
        from bramastra_lab.research.evaluation.scoring import decide_promotion

        bundle = self.build_bundle(paired_goal=False)
        decision = decide_promotion(bundle, PromotionConfig())
        self.assertEqual(decision["pair_criteria"], "not_applicable_for_this_protocol")
        self.assertNotIn("insufficient_pairs_for_goal_check", decision["reasons"])

    def test_default_is_no_promotion_without_evidence(self) -> None:
        from bramastra_lab.research.evaluation.scoring import decide_promotion

        decision = decide_promotion(self.build_bundle(cand_primary=0.80), PromotionConfig())
        self.assertEqual(decision["decision"], "no_promotion")
        self.assertIn("primary_margin_not_met", decision["reasons"][0])

    def test_small_improvement_does_not_promote(self) -> None:
        from bramastra_lab.research.evaluation.scoring import decide_promotion

        decision = decide_promotion(self.build_bundle(cand_primary=0.82), PromotionConfig())
        self.assertIn("primary_margin_not_met", decision["reasons"][0])

    def test_family_regression_blocks_promotion(self) -> None:
        from bramastra_lab.research.evaluation.scoring import decide_promotion

        decision = decide_promotion(self.build_bundle(logic_delta=-0.4), PromotionConfig())
        self.assertEqual(decision["decision"], "no_promotion")
        self.assertTrue(any(r.startswith("family_regression") for r in decision["reasons"]))

    def test_sealed_reuse_blocks_acceptance(self) -> None:
        from bramastra_lab.research.evaluation.scoring import decide_promotion

        decision = decide_promotion(self.build_bundle(sealed_fresh=False), PromotionConfig())
        self.assertEqual(decision["decision"], "no_promotion")
        self.assertIn("sealed_pool_reuse_detected", decision["reasons"])

    def test_parent_child_must_differ(self) -> None:
        from bramastra_lab.research.evaluation.scoring import (decide_promotion,
                                                               EvidenceBundle)

        bundle = self.build_bundle()
        with self.assertRaises(Exception):
            EvidenceBundle(**{**bundle.__dict__,
                              "parent_identity": "same", "child_identity": "same"})

    def test_uncertainty_underpowered_blocks_accept(self) -> None:
        from bramastra_lab.research.evaluation.scoring import decide_promotion

        bundle = self.build_bundle(requires_uncertainty=True,
                                   clustered={"clusters": 1, "ci_includes_zero": False})
        decision = decide_promotion(bundle, PromotionConfig())
        self.assertTrue(any(r.startswith("underpowered") for r in decision["reasons"]))


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


class SustainedGateTests(unittest.TestCase):
    def test_gate_requires_consecutive_confirmation(self) -> None:
        from bramastra_lab.research.evaluation.scoring import sustained_gate

        series = [(100, 0.95), (150, 0.5), (200, 0.96), (250, 0.97), (300, 0.98)]
        gate = sustained_gate(series, threshold=0.9, consecutive=3)
        self.assertEqual(gate["onset_update"], 200)
        self.assertEqual(gate["confirmed_update"], 300)
        self.assertTrue(gate["sustained_confirmed"])
        self.assertTrue(gate["peak_claims_forbidden"])
        self.assertEqual(gate["max_score_for_audit_only"], 0.98)

    def test_streak_interrupted_by_dip_resets(self) -> None:
        from bramastra_lab.research.evaluation.scoring import sustained_gate

        series = [(100, 0.95), (150, 0.95), (200, 0.5), (250, 0.95), (300, 0.95),
                  (350, 0.95)]
        gate = sustained_gate(series, threshold=0.9, consecutive=3)
        self.assertEqual(gate["onset_update"], 250)
        self.assertEqual(gate["confirmed_update"], 350)

    def test_run_ending_before_confirmation_is_not_confirmed(self) -> None:
        from bramastra_lab.research.evaluation.scoring import sustained_gate

        gate = sustained_gate([(100, 0.95), (150, 0.96)], threshold=0.9, consecutive=3)
        self.assertFalse(gate["sustained_confirmed"])
        self.assertIsNone(gate["confirmed_update"])
        # Formation AUC is the integral, not the peak.
        self.assertAlmostEqual(gate["formation_auc"], 0.955)

    def test_empty_series_rejects(self) -> None:
        from bramastra_lab.research.evaluation.scoring import sustained_gate

        with self.assertRaises(EvaluationError):
            sustained_gate([], threshold=0.9)


class ClusteredBootstrapTests(unittest.TestCase):
    @staticmethod
    def build_paired_worlds(n_clusters=4, items_per=4, *, ref_correct=True,
                            cand_correct=True):
        refs, cands = [], []
        for cluster in range(n_clusters):
            for item in range(items_per):
                task = f"task-{cluster}"
                case = f"case-{cluster}-{item}"
                refs.append(outcome(f"r{cluster}-{item}", label="4",
                                    prediction="4" if ref_correct else "6",
                                    task_semantic_id=task, case_id=case))
                cands.append(outcome(f"c{cluster}-{item}", label="4",
                                     prediction="4" if cand_correct else "6",
                                     task_semantic_id=task, case_id=case))
        return refs, cands

    def test_delta_and_clusters(self) -> None:
        from bramastra_lab.research.evaluation.scoring import clustered_bootstrap_delta

        refs, cands = self.build_paired_worlds(ref_correct=False, cand_correct=True)
        report = clustered_bootstrap_delta(refs, cands, iterations=200, seed=1)
        self.assertEqual(report["clusters"], 4)
        self.assertAlmostEqual(report["delta"], 1.0)
        self.assertFalse(report["ci_includes_zero"])

    def test_unmatched_cases_reject(self) -> None:
        from bramastra_lab.research.evaluation.scoring import clustered_bootstrap_delta

        refs, cands = self.build_paired_worlds()
        extra = [outcome("extra", task_semantic_id="task-0", case_id="case-other")]
        with self.assertRaises(EvaluationError):
            clustered_bootstrap_delta(refs, cands + extra)

    def test_different_cases_same_world_reject(self) -> None:
        """The chief probe's defect: different cases in one world cannot pair."""
        from bramastra_lab.research.evaluation.scoring import clustered_bootstrap_delta

        refs = [outcome("a", label="a", prediction="wrong", task_semantic_id="world",
                        case_id="different-task-a")]
        cands = [outcome("b", label="b", prediction="b", task_semantic_id="world",
                         case_id="different-task-b")]
        with self.assertRaises(EvaluationError):
            clustered_bootstrap_delta(refs, cands, iterations=10)

    def test_mismatched_labels_within_case_reject(self) -> None:
        from bramastra_lab.research.evaluation.scoring import clustered_bootstrap_delta

        refs = [outcome("a", label="4", task_semantic_id="w", case_id="c1")]
        cands = [outcome("b", label="5", task_semantic_id="w", case_id="c1")]
        with self.assertRaises(EvaluationError) as caught:
            clustered_bootstrap_delta(refs, cands)
        self.assertIn("label", str(caught.exception))

    def test_underpowered_promotion_is_refused(self) -> None:
        from bramastra_lab.research.evaluation.scoring import decide_promotion

        refs, cands = self.build_paired_worlds(n_clusters=1, ref_correct=False,
                                               cand_correct=True)
        bundle = PromotionTests.build_bundle(requires_uncertainty=True,
                                             clustered={"clusters": 1,
                                                        "ci_includes_zero": False})
        decision = decide_promotion(bundle, PromotionConfig())
        self.assertEqual(decision["decision"], "no_promotion")
        self.assertTrue(any(r.startswith("underpowered") for r in decision["reasons"]))


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
