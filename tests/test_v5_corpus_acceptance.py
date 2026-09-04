"""Corpus acceptance and P35-A readiness gates (M30-M34)."""

from __future__ import annotations

import unittest

from v5_data.qualify import CENTER_5B_MIXTURE, SourceArtifact, qualify_dataset
from v5_experiments.p35a_readiness import P35_PARAMETERS, evaluate_p35a_readiness


def _manifest(total=1000, families=None, scan="a" * 64, tokenizer="b" * 64):
    families = families or {"natural": 650, "code_math_formal": 200, "verified_cognition": 150}
    return {
        "sources": [
            {"raw_sha256": "c" * 64, "split": "training"},
            {"raw_sha256": "d" * 64, "split": "training"},
        ],
        "contamination_scan_sha256": scan,
        "tokenizer_sha256": tokenizer,
        "tokens_by_family": families,
        "total_tokens": total,
    }


def _receipt():
    return {"artifact": {"sha256": "b" * 64}}


class SourceArtifactTests(unittest.TestCase):
    def test_valid_artifact(self) -> None:
        SourceArtifact(
            schema="anra-v5-source-artifact/v1", source_id="s1",
            artifact_sha256="a" * 64, format="jsonl", document_count=10,
            source_class="natural", provenance="test", quality_status="accepted",
            license="permissive", cluster_dedup_status="deduped",
        ).assert_valid()

    def test_rejects_gaps(self) -> None:
        with self.assertRaises(ValueError):
            SourceArtifact(
                schema="anra-v5-source-artifact/v1", source_id="",
                artifact_sha256="a" * 64, format="jsonl", document_count=10,
                source_class="natural", provenance="test", quality_status="accepted",
                license="permissive", cluster_dedup_status="deduped",
            ).assert_valid()


class QualifyDatasetTests(unittest.TestCase):
    def _qualify(self, **overrides):
        params: dict[str, object] = {
            "data_manifest": _manifest(),
            "manifest_audit": {
                "processed_document_sha256": {"d1": "e" * 64},
                "exact_duplicate_drops": {},
            },
            "tokenizer_receipt": _receipt(),
            "expected_tokenizer_sha256": "b" * 64,
            "family_to_slice": {
                "natural": "natural", "code_math_formal": "code_math_formal",
                "verified_cognition": "verified_cognition",
            },
            "mixture_targets": {"natural": 650, "code_math_formal": 200, "verified_cognition": 150},
            "required_cognition_families": [],
            "generator_qualifications": {},
        }
        params.update(overrides)
        return qualify_dataset(**params)  # type: ignore[arg-type]

    def test_qualified_dataset_passes(self) -> None:
        receipt = self._qualify()
        self.assertEqual(receipt["status"], "DATASET_QUALIFIED")
        self.assertEqual(receipt["blockers"], [])

    def test_insufficient_tokens_block(self) -> None:
        receipt = self._qualify(
            mixture_targets={"natural": 650000, "code_math_formal": 200, "verified_cognition": 150}
        )
        self.assertEqual(receipt["status"], "BLOCKED_BY_DATASET")
        self.assertTrue(any("unique qualified tokens" in blocker for blocker in receipt["blockers"]))

    def test_unqualified_cognition_blocks(self) -> None:
        receipt = self._qualify(
            required_cognition_families=["query_binding"],
            generator_qualifications={"query_binding": "GENERATOR_NOT_QUALIFIED"},
        )
        self.assertEqual(receipt["status"], "BLOCKED_BY_DATASET")

    def test_wrong_tokenizer_blocks(self) -> None:
        receipt = self._qualify(expected_tokenizer_sha256="f" * 64)
        self.assertEqual(receipt["status"], "BLOCKED_BY_DATASET")


class P35AReadinessTests(unittest.TestCase):
    def test_recipe_is_exact(self) -> None:
        from v5_experiments.p35a_readiness import check_recipe

        self.assertTrue(check_recipe()["pass"])
        self.assertEqual(P35_PARAMETERS, 35_411_328)

    def test_live_tree_is_blocked_by_dataset(self) -> None:
        from pathlib import Path

        receipt = evaluate_p35a_readiness(Path(__file__).resolve().parents[1])
        self.assertEqual(receipt["verdict"], "BLOCKED")
        self.assertIn("qualified_dataset", receipt["blocked_by"])
        self.assertIn(receipt["gates"]["exact_p35_architecture"]["pass"], (True,))
        self.assertEqual(receipt["gates"]["compute"]["pass"], True)

    def test_center_mixture_sums_to_5b(self) -> None:
        self.assertEqual(sum(CENTER_5B_MIXTURE.values()), 5_000_000_000)


if __name__ == "__main__":
    unittest.main()
