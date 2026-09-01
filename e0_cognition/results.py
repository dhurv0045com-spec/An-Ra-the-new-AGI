"""Raw-Core, assisted, and replication result contracts for E0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .contracts import CausalCase, EvaluationSuite, Split
from .metrics import selection_eligible
from .preregistration import protocol_sha256


@dataclass(frozen=True, slots=True)
class ConditionOutcome:
    """One condition's observable result; selection is absent for copy controls."""

    selection_correct: bool | None
    realization_correct: bool


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    case_id: str
    raw_core: ConditionOutcome
    constrained: ConditionOutcome | None = None
    assisted: ConditionOutcome | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    split: Split
    suite_sha256: str
    checkpoint_sha256: str
    evaluator_sha256: str
    outcomes: tuple[CaseOutcome, ...]
    protocol_sha256: str = protocol_sha256()

    def assert_valid(self, suite: EvaluationSuite) -> None:
        suite.assert_valid()
        if self.split != suite.split or self.suite_sha256 != suite.sha256():
            raise ValueError("evaluation run is bound to a different suite")
        if len(self.checkpoint_sha256) != 64 or len(self.evaluator_sha256) != 64:
            raise ValueError("evaluation identities must be SHA-256")
        if self.protocol_sha256 != protocol_sha256():
            raise ValueError("evaluation protocol identity mismatch")
        by_id = {outcome.case_id: outcome for outcome in self.outcomes}
        expected = {case.case_id for case in suite.cases}
        if set(by_id) != expected:
            raise ValueError("evaluation outcomes must cover the suite exactly once")
        for case in suite.cases:
            outcome = by_id[case.case_id]
            expected_selection = selection_eligible(case)
            if expected_selection != (outcome.raw_core.selection_correct is not None):
                raise ValueError("copy/realization controls cannot enter selection metrics")
            for assisted in (outcome.constrained, outcome.assisted):
                if assisted is not None and expected_selection != (assisted.selection_correct is not None):
                    raise ValueError("assisted selection eligibility mismatch")

    def summary(self, suite: EvaluationSuite) -> dict[str, object]:
        self.assert_valid(suite)
        outcomes = {outcome.case_id: outcome for outcome in self.outcomes}
        def condition_summary(name: str) -> dict[str, object]:
            condition = [getattr(outcomes[case.case_id], name) for case in suite.cases]
            present = [result for result in condition if result is not None]
            selection = [result.selection_correct for result in present if result.selection_correct is not None]
            realization = [result.realization_correct for result in present]
            conditional_realization = [
                result.realization_correct
                for result in present
                if result.selection_correct is True
            ]
            return {
                "selection_accuracy": sum(selection) / len(selection) if selection else None,
                "selection_cases": len(selection),
                "realization_accuracy": sum(realization) / len(realization) if realization else None,
                "realization_cases": len(realization),
                "conditional_realization_accuracy": (
                    sum(conditional_realization) / len(conditional_realization)
                    if conditional_realization else None
                ),
                "conditional_realization_cases": len(conditional_realization),
            }

        raw = condition_summary("raw_core")
        summary: dict[str, object] = {"raw_core": raw}
        for name in ("constrained", "assisted"):
            if any(getattr(outcomes[case.case_id], name) is not None for case in suite.cases):
                summary[name] = condition_summary(name)
        assisted = [outcomes[case.case_id].assisted for case in suite.cases]
        paired = [
            (outcomes[case.case_id].raw_core.realization_correct, result.realization_correct)
            for case, result in zip(suite.cases, assisted)
            if result is not None
        ]
        summary["intervention_dependence"] = {
            "paired_cases": len(paired),
            "raw_failures_repaired": sum((not raw_ok) and assisted_ok for raw_ok, assisted_ok in paired),
            "raw_successes_harmed": sum(raw_ok and (not assisted_ok) for raw_ok, assisted_ok in paired),
        }
        return summary


@dataclass(frozen=True, slots=True)
class ReplicationBundle:
    development: EvaluationRun
    sealed: EvaluationRun
    fresh: EvaluationRun

    def assert_valid(
        self, development_suite: EvaluationSuite, sealed_suite: EvaluationSuite, fresh_suite: EvaluationSuite
    ) -> None:
        runs = ((self.development, development_suite), (self.sealed, sealed_suite), (self.fresh, fresh_suite))
        for run, suite in runs:
            run.assert_valid(suite)
        if len({self.development.checkpoint_sha256, self.sealed.checkpoint_sha256, self.fresh.checkpoint_sha256}) != 1:
            raise ValueError("replication must evaluate one checkpoint across splits")
        if len({self.development.evaluator_sha256, self.sealed.evaluator_sha256, self.fresh.evaluator_sha256}) != 1:
            raise ValueError("replication must use one evaluator build")
        if len({run.suite_sha256 for run, _ in runs}) != 3:
            raise ValueError("development, sealed, and fresh fixtures must be distinct")

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()
