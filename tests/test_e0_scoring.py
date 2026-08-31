from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

from e0_cognition.scoring_certification import (
    CandidateTrace,
    DeterministicControlAdapter,
    PositionControlAdapter,
    ScoreMode,
    aggregate_candidate_log_likelihood,
    bias_profile,
    build_bias_probe_cases,
    build_scoring_certificate,
    implementation_sha256,
    predict_case,
    score_case,
)


class E0ScoringCertificationTests(unittest.TestCase):
    def test_all_aggregation_modes_use_candidate_suffix_only(self) -> None:
        trace = CandidateTrace((11, 12), (-2.0, -2.0))
        self.assertEqual(aggregate_candidate_log_likelihood(trace, "four", ScoreMode.SUM), -4.0)
        self.assertEqual(
            aggregate_candidate_log_likelihood(trace, "four", ScoreMode.TOKEN_NORMALIZED),
            -2.0,
        )
        self.assertEqual(
            aggregate_candidate_log_likelihood(trace, "four", ScoreMode.BYTE_NORMALIZED),
            -1.0,
        )

    def test_invalid_candidate_traces_fail_closed(self) -> None:
        invalid = (
            CandidateTrace((), ()),
            CandidateTrace((1,), (-1.0, -2.0)),
            CandidateTrace((-1,), (-1.0,)),
            CandidateTrace((1,), (math.nan,)),
            CandidateTrace((1,), (0.25,)),
        )
        for trace in invalid:
            with self.assertRaises(ValueError):
                trace.assert_valid()

    def test_adapter_never_receives_hidden_answer(self) -> None:
        case = build_bias_probe_cases(groups=1)[0]

        class SpyAdapter:
            identity_sha256 = "a" * 64

            def __init__(self) -> None:
                self.keys: set[str] | None = None

            def trace(self, model_view, candidate, candidate_position):
                del candidate, candidate_position
                self.keys = set(model_view)
                return CandidateTrace((1,), (-1.0,))

        adapter = SpyAdapter()
        score_case(case, adapter, ScoreMode.SUM)
        self.assertEqual(adapter.keys, {"context", "query", "prompt"})

    def test_adapter_identity_must_be_a_lowercase_sha256(self) -> None:
        case = build_bias_probe_cases(groups=1)[0]

        class UnboundAdapter:
            identity_sha256 = "not-a-hash"

            def trace(self, model_view, candidate, candidate_position):
                del model_view, candidate, candidate_position
                return CandidateTrace((1,), (-1.0,))

        with self.assertRaises(ValueError):
            score_case(case, UnboundAdapter(), ScoreMode.SUM)

    def test_deterministic_oracle_and_random_controls(self) -> None:
        cases = build_bias_probe_cases(groups=4)
        token_ids = {
            "x": (101, 102, 103, 104),
            "medium": (201,),
            "the-longest-candidate": (777, 301),
        }
        targets = {
            hashlib.sha256(
                json.dumps(
                    case.model_view(), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(): case.answer
            for case in cases
        }
        oracle = DeterministicControlAdapter(
            policy="target", token_ids=token_ids, targets_by_prompt=targets
        )
        random_a = DeterministicControlAdapter(
            policy="random_token_logits", token_ids=token_ids, seed=7
        )
        random_b = DeterministicControlAdapter(
            policy="random_token_logits", token_ids=token_ids, seed=7
        )
        for mode in ScoreMode:
            self.assertTrue(
                all(predict_case(score_case(case, oracle, mode)) == case.answer for case in cases)
            )
            left = [score_case(case, random_a, mode) for case in cases]
            right = [score_case(case, random_b, mode) for case in cases]
            self.assertEqual(left, right)
            self.assertEqual(random_a.identity_sha256, random_b.identity_sha256)

    def test_position_bias_and_rotation_instability_are_detected(self) -> None:
        cases = build_bias_probe_cases(groups=8)
        adapter = PositionControlAdapter(
            {
                "x": (101, 102, 103, 104),
                "medium": (201,),
                "the-longest-candidate": (777, 301),
            }
        )
        profile = bias_profile(cases, adapter, ScoreMode.TOKEN_NORMALIZED)
        self.assertEqual(profile.first_position_rate, 1.0)
        self.assertEqual(profile.rotation_stability_rate, 0.0)

    def test_receipt_is_source_bound_and_fail_closed(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        receipt_path = repository / "artifacts/e0/scoring_adapter_certificate.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["implementation_sha256"], implementation_sha256())
        self.assertEqual(receipt["status"], "CONTRACT_PASS_DEVICE_PENDING")
        self.assertTrue(all(receipt["checks"].values()))
        self.assertFalse(receipt["promotion_authorized"])
        self.assertTrue(receipt["fail_closed"])
        self.assertIsNone(receipt["production_scoring_mode"])
        self.assertEqual(receipt["random_weight_p35_device_evidence"]["status"], "PENDING")

    def test_freshly_built_certificate_matches_contract(self) -> None:
        receipt = build_scoring_certificate()
        self.assertEqual(receipt["status"], "CONTRACT_PASS_DEVICE_PENDING")
        self.assertTrue(all(receipt["checks"].values()))
        self.assertEqual(set(receipt["score_modes"]), {mode.value for mode in ScoreMode})
        self.assertEqual(receipt["pair_metrics"]["sensitivity_correct_flip"], receipt["pair_metrics"]["sensitivity_total"])
        self.assertEqual(receipt["pair_metrics"]["invariance_stable"], receipt["pair_metrics"]["invariance_total"])


if __name__ == "__main__":
    unittest.main()
