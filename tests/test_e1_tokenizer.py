from __future__ import annotations

import copy
import dataclasses
import gzip
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from e1_tokenizer.audit import audit_receipt
from e1_tokenizer.compare import pareto_front
from e1_tokenizer.local_tournament import SourceSpec, build_records, parse_source
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

    def test_local_corpus_split_is_deterministic_and_duplicate_safe(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_text("alpha\nbeta\nalpha\ngamma\ndelta\nepsilon\n", encoding="utf-8")
            source = SourceSpec("fixture", "natural", path)
            train_a, eval_a, manifest_a = build_records([source], holdout_modulus=2)
            train_b, eval_b, manifest_b = build_records([source], holdout_modulus=2)
            self.assertEqual(train_a, train_b)
            self.assertEqual(eval_a, eval_b)
            self.assertEqual(manifest_a, manifest_b)
            train_hashes = {row.text_sha256 for row in train_a}
            eval_hashes = {row.text_sha256 for row in eval_a}
            self.assertFalse(train_hashes & eval_hashes)
            self.assertTrue(manifest_a["holdout"]["duplicate_text_cannot_cross_splits"])

    def test_local_source_parser_preserves_windows_drive_colon(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("evidence", encoding="utf-8")
            source = parse_source(f"sample::formal::{path}")
            self.assertEqual(source.label, "sample")
            self.assertEqual(source.domain, "formal")
            self.assertEqual(source.path, path.resolve())

    def test_source_root_name_is_not_treated_as_an_excluded_descendant(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / ".codex-worktrees"
            root.mkdir()
            (root / "evidence.md").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            train, evaluation, _ = build_records(
                [SourceSpec("fixture", "docs", root)], holdout_modulus=2
            )
            self.assertTrue(train)
            self.assertTrue(evaluation)

    def test_local_v4_evidence_receipts_are_self_consistent(self) -> None:
        root = Path(__file__).parents[1]
        baseline = json.loads(
            (root / "artifacts/e1/v4_32k_baseline_audit.json").read_text(encoding="utf-8")
        )
        proxy = json.loads(
            (root / "artifacts/e1/v4_prefix_truncation_proxy.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(baseline["status"], "PASS")
        self.assertEqual(
            baseline["artifact_sha256"], proxy["source_artifact_sha256"]
        )
        self.assertAlmostEqual(
            baseline["metrics"]["tokens_per_byte"],
            baseline["metrics"]["total_tokens"] / baseline["metrics"]["total_utf8_bytes"],
        )
        rows = {row["vocabulary_size"]: row for row in proxy["repository_proxy"]["rows"]}
        self.assertEqual(set(rows), set(CANDIDATE_VOCABULARIES))
        for vocabulary_size, row in rows.items():
            self.assertEqual(row["embedding_parameters_at_width_896"], vocabulary_size * 896)
            self.assertAlmostEqual(
                row["tokens_per_byte"],
                row["tokens"] / proxy["repository_proxy"]["utf8_bytes"],
            )
        self.assertLess(rows[24_576]["inflation_vs_32k_pct"], 2.0)
        self.assertGreater(rows[16_384]["inflation_vs_32k_pct"], 4.0)

    def test_local_independent_candidate_artifacts_match_receipt(self) -> None:
        root = Path(__file__).parents[1]
        tournament = root / "artifacts/e1/local_tournament"
        result = json.loads((tournament / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "DEVELOPMENT_STATIC_PASS")
        for row in result["candidate_rows"]:
            artifact = tournament / row["artifact"]
            self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), row["artifact_sha256"])
            payload = json.loads(gzip.decompress(artifact.read_bytes()).decode("utf-8"))
            self.assertEqual(len(payload["model"]["vocab"]), row["vocabulary_size"])
            self.assertTrue(row["compressed_artifact_reload_pass"])
            self.assertEqual(row["evaluation"]["unexpected_unknown_token_occurrences"], 0)
        original = tournament / "tokenizer-24576.json.gz"
        replica = tournament / "tokenizer-24576-replica.json.gz"
        self.assertEqual(original.read_bytes(), replica.read_bytes())
        self.assertTrue(result["determinism"]["byte_identical"])


if __name__ == "__main__":
    unittest.main()
