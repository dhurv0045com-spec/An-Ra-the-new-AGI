from __future__ import annotations

import copy
import dataclasses
import hashlib
import unittest

from e1_tokenizer.audit import audit_receipt
from e1_tokenizer.compare import pareto_front
from e1_tokenizer.probes import PROBES
from e1_tokenizer.tournament import CANDIDATE_VOCABULARIES, build_plan, probe_manifest_sha256


def candidate_receipt(name: str, vocabulary_size: int, artifact_sha256: str) -> dict[str, object]:
    return {
        "schema": "esoes-e1-candidate-encoding/v1",
        "tokenizer_name": name,
        "vocabulary_size": vocabulary_size,
        "artifact_sha256": artifact_sha256,
        "unknown_token_id": None,
        "encodings": [
            {
                "probe_id": probe.probe_id,
                "token_ids": list(probe.text.encode("utf-8")),
                "decoded_text": probe.text,
            }
            for probe in PROBES
        ],
    }


class E1TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sha = hashlib.sha256(b"fake-tokenizer-artifact").hexdigest()

    def test_identity_preserving_candidate_passes_static_audit(self) -> None:
        report = audit_receipt(candidate_receipt("byte-canary", 16_384, self.sha), artifact_sha256=self.sha)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["checks"]["identity_roundtrip"])
        self.assertEqual(report["metrics"]["tokens_per_byte"], 1.0)

    def test_artifact_hash_mismatch_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            audit_receipt(candidate_receipt("bad", 16_384, self.sha), artifact_sha256="0" * 64)

    def test_unknown_token_is_detected(self) -> None:
        receipt = candidate_receipt("unknown", 16_384, self.sha)
        receipt["unknown_token_id"] = 32
        report = audit_receipt(receipt, artifact_sha256=self.sha)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["checks"]["zero_unknowns"])

    def test_pareto_front_does_not_hide_weighting(self) -> None:
        small = audit_receipt(candidate_receipt("small", 16_384, self.sha), artifact_sha256=self.sha)
        efficient = copy.deepcopy(small)
        efficient["candidate"] = "efficient"
        efficient["vocabulary_size"] = 32_768
        efficient["metrics"]["tokens_per_byte"] = 0.7
        dominated = copy.deepcopy(small)
        dominated["candidate"] = "dominated"
        dominated["vocabulary_size"] = 32_768
        self.assertEqual(pareto_front([small, efficient, dominated]), ["efficient", "small"])

    def test_tournament_is_exactly_matched_and_pending_external_inputs(self) -> None:
        plan = build_plan(raw_byte_budget=1234, training_flops_budget=5678)
        plan.assert_valid()
        self.assertEqual(plan.candidate_vocabulary_sizes, CANDIDATE_VOCABULARIES)
        self.assertEqual(plan.status(), "BLOCKED_EXTERNAL_CORPUS")
        self.assertEqual(len(probe_manifest_sha256()), 64)

    def test_tournament_rejects_unmatched_arm_budget(self) -> None:
        plan = build_plan()
        bad = dataclasses.replace(
            plan,
            arms=(dataclasses.replace(plan.arms[0], matched_raw_bytes=1), *plan.arms[1:]),
        )
        with self.assertRaises(ValueError):
            bad.assert_valid()


if __name__ == "__main__":
    unittest.main()
