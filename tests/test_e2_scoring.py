from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

from e0_cognition.scoring_certification import ScoreMode
from e2_architecture.scoring_benchmark import (
    PARITY_MAX_ABSOLUTE_ERROR,
    PARITY_RELATIVE_RMS_ERROR,
    ScoringConfig,
    VOCABULARIES,
    _middle_arm,
    build_null_cases,
    compare_receipts,
)


class E2ScoringBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = Path(__file__).resolve().parents[1]
        root = cls.repository / "artifacts/e2"
        cls.cpu = json.loads((root / "local_cpu_scoring_null.json").read_text(encoding="utf-8"))
        cls.cuda = json.loads((root / "local_cuda_scoring_null.json").read_text(encoding="utf-8"))
        cls.parity = json.loads(
            (root / "local_cpu_cuda_scoring_parity.json").read_text(encoding="utf-8")
        )

    def test_null_cases_balance_position_rotations(self) -> None:
        cases = build_null_cases()
        self.assertEqual(len(cases), 18)
        groups: dict[str, list[object]] = {}
        for case in cases:
            groups.setdefault(dict(case.provenance)["rotation_group"], []).append(case)
        self.assertEqual(len(groups), 6)
        for values in groups.values():
            self.assertEqual(len(values), 3)
            self.assertEqual({case.answer for case in values}, {values[0].answer})
            self.assertEqual({case.candidates[0] for case in values}, set(values[0].candidates))

    def test_resized_middle_models_make_parameter_difference_explicit(self) -> None:
        totals = [_middle_arm(vocabulary).model.parameter_receipt().total for vocabulary in VOCABULARIES]
        self.assertEqual(totals, sorted(totals))
        self.assertEqual(len(set(totals)), 3)

    def test_scoring_config_fails_closed(self) -> None:
        for config in (
            ScoringConfig(device="tpu"),
            ScoringConfig(device="cpu", seed=-1),
            ScoringConfig(device="cpu", batch_size=0),
        ):
            with self.assertRaises(ValueError):
                config.assert_valid()

    def test_device_receipts_are_source_and_artifact_bound(self) -> None:
        source_hash = hashlib.sha256(
            (self.repository / "e2_architecture/scoring_benchmark.py").read_bytes()
        ).hexdigest()
        constructor_hash = hashlib.sha256(
            (self.repository / "e2_architecture/block_benchmark.py").read_bytes()
        ).hexdigest()
        for receipt, device in ((self.cpu, "cpu"), (self.cuda, "cuda")):
            self.assertEqual(receipt["status"], "PASS_LOCAL_NULL_DEVICE")
            self.assertEqual(receipt["implementation_sha256"], source_hash)
            self.assertEqual(receipt["model_constructor_sha256"], constructor_hash)
            self.assertEqual(receipt["config"]["device"], device)
            self.assertTrue(all(receipt["checks"].values()))
            self.assertFalse(receipt["promotion_authorized"])
            self.assertIsNone(receipt["production_scoring_mode"])
            self.assertEqual([row["vocabulary_size"] for row in receipt["rows"]], list(VOCABULARIES))
            for row in receipt["rows"]:
                artifact = self.repository / "artifacts/e1/local_tournament" / row["tokenizer_artifact"]
                self.assertEqual(row["tokenizer_sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest())
                self.assertTrue(row["finite"])
                self.assertTrue(row["roundtrip_and_trace_coverage"])
                self.assertEqual(set(row["bias_by_mode"]), {mode.value for mode in ScoreMode})

    def test_cpu_cuda_parity_receipt_recomputes(self) -> None:
        rebuilt = compare_receipts(self.cpu, self.cuda)
        self.assertEqual(rebuilt["status"], "PASS_LOCAL_CPU_CUDA_NULL_PARITY")
        self.assertEqual(rebuilt["metrics"], self.parity["metrics"])
        self.assertTrue(all(self.parity["checks"].values()))
        self.assertLessEqual(
            self.parity["metrics"]["maximum_absolute_error"], PARITY_MAX_ABSOLUTE_ERROR
        )
        self.assertLessEqual(
            self.parity["metrics"]["relative_rms_error"], PARITY_RELATIVE_RMS_ERROR
        )
        self.assertEqual(self.parity["metrics"]["prediction_mismatches"], 0)

    def test_all_committed_scores_are_finite(self) -> None:
        for receipt in (self.cpu, self.cuda):
            for row in receipt["rows"]:
                for trace in row["candidate_tokenizations"]:
                    self.assertTrue(all(math.isfinite(value) for value in trace["token_logprobs"]))


if __name__ == "__main__":
    unittest.main()
