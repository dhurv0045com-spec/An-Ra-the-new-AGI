from __future__ import annotations

import hashlib
import unittest

from e0_cognition.contracts import Split
from e0_cognition.evaluation_generators import build_evaluation_suite
from e0_cognition.results import CaseOutcome, ConditionOutcome, EvaluationRun, ReplicationBundle
from e0_cognition.metrics import selection_eligible


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _run(suite, *, checkpoint: str = "checkpoint", evaluator: str = "evaluator") -> EvaluationRun:
    outcomes = tuple(
        CaseOutcome(
            case.case_id,
            ConditionOutcome(True if selection_eligible(case) else None, True),
            constrained=ConditionOutcome(True if selection_eligible(case) else None, True),
            assisted=ConditionOutcome(True if selection_eligible(case) else None, True),
        )
        for case in suite.cases
    )
    return EvaluationRun(
        suite.split,
        suite.sha256(),
        _sha(checkpoint),
        _sha(evaluator),
        outcomes,
    )


class E0ResultContractTests(unittest.TestCase):
    def test_raw_and_assisted_summary_keeps_intervention_dependence_separate(self) -> None:
        suite = build_evaluation_suite(Split.DEVELOPMENT, seed=811, groups_per_family=2)
        run = _run(suite)
        summary = run.summary(suite)
        self.assertEqual(summary["raw_core"]["realization_accuracy"], 1.0)
        self.assertEqual(summary["raw_core"]["conditional_realization_accuracy"], 1.0)
        self.assertEqual(
            summary["raw_core"]["conditional_realization_cases"],
            summary["raw_core"]["selection_cases"],
        )
        self.assertEqual(summary["assisted"]["selection_accuracy"], 1.0)
        self.assertEqual(summary["intervention_dependence"]["raw_failures_repaired"], 0)

    def test_conditional_realization_excludes_incorrect_unassisted_selection(self) -> None:
        suite = build_evaluation_suite(Split.DEVELOPMENT, seed=816, groups_per_family=1)
        eligible = [case for case in suite.cases if selection_eligible(case)]
        excluded = eligible[0].case_id
        outcomes = tuple(
            CaseOutcome(
                case.case_id,
                ConditionOutcome(
                    False if case.case_id == excluded else True if selection_eligible(case) else None,
                    True,
                ),
            )
            for case in suite.cases
        )
        run = EvaluationRun(suite.split, suite.sha256(), _sha("conditional"), _sha("eval"), outcomes)
        summary = run.summary(suite)["raw_core"]
        self.assertEqual(summary["conditional_realization_cases"], summary["selection_cases"] - 1)
        self.assertEqual(summary["conditional_realization_accuracy"], 1.0)

    def test_copy_case_cannot_supply_selection_outcome(self) -> None:
        suite = build_evaluation_suite(Split.DEVELOPMENT, seed=812, groups_per_family=1)
        copy = next(case for case in suite.cases if case.family == "exact_contextual_copy")
        bad = CaseOutcome(copy.case_id, ConditionOutcome(True, True))
        outcomes = tuple(
            bad if case.case_id == copy.case_id else CaseOutcome(
                case.case_id,
                ConditionOutcome(True if selection_eligible(case) else None, True),
            )
            for case in suite.cases
        )
        run = EvaluationRun(suite.split, suite.sha256(), _sha("c"), _sha("e"), outcomes)
        with self.assertRaises(ValueError):
            run.assert_valid(suite)

    def test_replication_requires_three_distinct_fixture_hashes(self) -> None:
        development = build_evaluation_suite(Split.DEVELOPMENT, seed=813, groups_per_family=1)
        sealed = build_evaluation_suite(Split.SEALED, seed=814, groups_per_family=1)
        fresh = build_evaluation_suite(Split.FRESH, seed=815, groups_per_family=1)
        bundle = ReplicationBundle(_run(development), _run(sealed), _run(fresh))
        bundle.assert_valid(development, sealed, fresh)
        self.assertEqual(len(bundle.sha256()), 64)


if __name__ == "__main__":
    unittest.main()
