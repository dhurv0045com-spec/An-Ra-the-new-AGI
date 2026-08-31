from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

from e0_cognition.scoring_certification import CandidateTrace
from e2_architecture.scoring_policy import (
    CandidateEvidence,
    Policy,
    build_preregistration,
    score_contextual_calibration,
    score_independent_policy,
    select,
)


def _evidence(candidate: str, target: float, neutral: float) -> CandidateEvidence:
    trace = CandidateTrace((1,), (target,))
    references = tuple(CandidateTrace((1,), (neutral,)) for _ in range(4))
    return CandidateEvidence(candidate, trace, references)


class E2ScoringPolicyTests(unittest.TestCase):
    def test_independent_policy_formulas_are_exact(self) -> None:
        evidence = CandidateEvidence(
            "four",
            CandidateTrace((1, 2), (-2.0, -2.0)),
            tuple(CandidateTrace((1, 2), (-3.0, -3.0)) for _ in range(4)),
        )
        self.assertEqual(score_independent_policy(evidence, Policy.SUM), -4.0)
        self.assertEqual(score_independent_policy(evidence, Policy.TOKEN_MEAN), -2.0)
        self.assertEqual(score_independent_policy(evidence, Policy.BYTE_MEAN), -1.0)
        self.assertEqual(score_independent_policy(evidence, Policy.DOMAIN_PMI), 2.0)

    def test_contextual_calibration_uses_log_space_and_recovers_swap(self) -> None:
        left = _evidence("left", -1.0, -2.0)
        right = _evidence("right", -2.0, -2.0)
        self.assertEqual(select(score_contextual_calibration((left, right))), "left")
        swapped = (replace(left, target=CandidateTrace((1,), (-2.0,))), replace(right, target=CandidateTrace((1,), (-1.0,))))
        self.assertEqual(select(score_contextual_calibration(swapped)), "right")
        huge = (_evidence("a", -10_000.0, -10_001.0), _evidence("b", -10_001.0, -10_001.0))
        self.assertTrue(all(value == value for value in score_contextual_calibration(huge).values()))

    def test_mismatched_neutral_tokenization_and_ties_fail_closed(self) -> None:
        evidence = _evidence("x", -1.0, -2.0)
        bad = replace(
            evidence,
            neutral=(CandidateTrace((2,), (-2.0,)), *evidence.neutral[1:]),
        )
        with self.assertRaises(ValueError):
            score_independent_policy(bad, Policy.DOMAIN_PMI)
        with self.assertRaises(ValueError):
            select({"a": 0.0, "b": 0.0})

    def test_committed_preregistration_is_outcome_free_and_source_bound(self) -> None:
        root = Path(__file__).resolve().parents[1]
        plan = json.loads(
            (root / "artifacts/e2/scoring_policy_preregistration.json").read_text(encoding="utf-8")
        )
        self.assertEqual(plan, build_preregistration())
        self.assertEqual(plan["status"], "PREREGISTERED_NO_RESULTS")
        self.assertEqual(plan["fixtures"]["independent_candidate_triplets"], 256)
        self.assertEqual(plan["neutral_contexts"]["panels"], 2)
        self.assertNotIn("rows", plan)
        self.assertNotIn("selected_policy", plan)
        normalized = (root / "e2_architecture/scoring_policy.py").read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertEqual(
            plan["implementation_sha256"],
            hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
