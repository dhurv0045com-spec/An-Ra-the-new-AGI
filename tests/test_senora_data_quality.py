"""Unit tests for senora.data_quality."""

from __future__ import annotations

import unittest

from e0_cognition.evaluation_generators import Split, build_evaluation_suite
from senora.data_quality import audit_cognition_corpus, compute_structural_signature, validate_cognition_case


class TestSenoraDataQuality(unittest.TestCase):
    def test_audit_cognition_corpus_real_suite(self) -> None:
        suite = build_evaluation_suite(Split.DEVELOPMENT, seed=101, groups_per_family=1)
        receipt = audit_cognition_corpus(suite.cases, corpus_manifest_sha256="a" * 64)

        self.assertEqual(receipt.status, "PASS_DATA_QUALITY_CERTIFIED")
        self.assertTrue(receipt.all_items_verified)
        self.assertTrue(receipt.leak_free)
        self.assertTrue(receipt.namespace_certified)
        self.assertGreaterEqual(receipt.diversity.unique_families_count, 5)
        self.assertGreater(receipt.diversity.template_entropy, 1.0)
        self.assertEqual(receipt.diversity.exact_duplicate_rate, 0.0)

    def test_structural_signature_invariance(self) -> None:
        sig1 = compute_structural_signature("If Alice goes home then Bob stays", "binding", 1)
        sig2 = compute_structural_signature("If Charlie goes home then David stays", "binding", 1)
        self.assertEqual(sig1, sig2)  # Identical structural topology despite entity permutation


if __name__ == "__main__":
    unittest.main()