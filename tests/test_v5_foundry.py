"""Data foundry unit tests: registry, cleaning, dedup, contamination, mixture."""

from __future__ import annotations

import unittest

from v5_data.accounting import account_source, plan_mixture
from v5_data.contamination import (
    check_cluster_ancestry,
    check_exact_hashes,
    check_fingerprints,
    check_ngram_overlap,
    check_normalized_exact,
    scan_all,
)
from v5_data.near_dedup import (
    DedupCluster,
    cluster_near_duplicates,
    minhash_signature,
    minhash_similarity,
)
from v5_data.normalize import NORMALIZE_VERSION, normalize_text
from v5_data.quality import QUALITY_VERSION, judge
from v5_data.registry import DataSource, DataSourceRegistry

try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def _source(source_id="s1", status="DISCOVERED", history=("DISCOVERED",)):
    return DataSource(
        source_id=source_id, version="2026-09-04", source_class="natural",
        artifact_sha256="a" * 64, format="parquet", document_count=10,
        byte_count=1000, provenance="huggingface-hub", license="O-DC-BY-1.0",
        acquisition_method="acquire.py resolve_huggingface", quality_tier="ungraded",
        language_domain="en/general", processing_policy="minimal-v1",
        status=status, status_history=history,
    )


class RegistryTests(unittest.TestCase):
    def test_lifecycle_forward_and_history(self) -> None:
        registry = DataSourceRegistry(schema="anra-v5-source-registry/v1", sources=())
        registry = registry.with_source(_source())
        registry = registry.with_transition("s1", "ACQUIRED")
        registry = registry.with_transition("s1", "IDENTITY_VERIFIED")
        self.assertEqual(len(registry.sha256()), 64)
        with self.assertRaises(ValueError):
            registry.with_transition("s1", "DISCOVERED")
        with self.assertRaises(ValueError):
            registry.with_source(_source())
        with self.assertRaises(ValueError):
            registry.with_transition("ghost", "ACQUIRED")

    def test_rejects_bad_class_and_round_trip(self) -> None:
        import dataclasses

        with self.assertRaises(ValueError):
            dataclasses.replace(_source(), source_class="vibes").assert_valid()
        registry = DataSourceRegistry(
            schema="anra-v5-source-registry/v1", sources=(_source(),)
        )
        import json

        clone = DataSourceRegistry.from_dict(json.loads(json.dumps({
            "schema": registry.schema,
            "sources": [{
                "source_id": "s1", "version": "2026-09-04", "source_class": "natural",
                "artifact_sha256": "a" * 64, "format": "parquet", "document_count": 10,
                "byte_count": 1000, "provenance": "huggingface-hub", "license": "O-DC-BY-1.0",
                "acquisition_method": "acquire.py resolve_huggingface", "quality_tier": "ungraded",
                "language_domain": "en/general", "processing_policy": "minimal-v1",
                "status": "DISCOVERED", "status_history": ["DISCOVERED"],
            }],
        })))
        self.assertEqual(clone.sha256(), registry.sha256())


class CleanTests(unittest.TestCase):
    def test_normalize_records_transforms(self) -> None:
        doc = normalize_text("d", "  hello\r\nworld  ")
        self.assertEqual(doc.text, "hello\nworld")
        self.assertIn("canonical_newlines", doc.applied)
        self.assertEqual(doc.normalize_version, NORMALIZE_VERSION)

    def test_quality_verdicts(self) -> None:
        single_signal = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" * 5
        verdict = judge("g", single_signal)
        self.assertEqual(verdict.decision, "QUARANTINE")
        self.assertIn("extreme_character_repetition", verdict.reasons)
        triple = (
            "BUY NOW " * 40 + "\n" + "\n".join(f"https://spam.example/{i}" for i in range(12))
            + "\n<div>click here</div>" * 12
        )
        verdict = judge("t", triple * 3)
        self.assertEqual(verdict.decision, "DROP")
        clean = (
            "The transformer architecture changed natural language processing. "
            "Researchers found that attention mechanisms capture long-range "
            "dependencies effectively across many benchmarks and domains."
        )
        verdict = judge("c", clean * 4)
        self.assertEqual(verdict.decision, "KEEP")
        self.assertEqual(verdict.quality_version, QUALITY_VERSION)

    def test_domain_tags(self) -> None:
        code = judge("code", "def train_step(state, batch):\n    return state.advance(batch)\n" * 6)
        self.assertEqual(code.domain, "code")
        dialogue = judge("dlg", "USER: hello there friend\nASSISTANT: hi, how can I help you today?\n" * 6)
        self.assertEqual(dialogue.domain, "dialogue")


class NearDedupTests(unittest.TestCase):
    def test_self_similarity_and_separation(self) -> None:
        text = "the quick brown fox jumps over the lazy dog near the river bank"
        self.assertEqual(minhash_similarity(minhash_signature(text), minhash_signature(text)), 1.0)
        other = "quantum chromodynamics describes strong interactions of quarks and gluons"
        self.assertLess(minhash_similarity(minhash_signature(text), minhash_signature(other)), 0.3)

    def test_clusters_near_duplicates(self) -> None:
        base = "the council approved the new transit budget after three hours of debate yesterday"
        near = "the council approved the new transit budget after three hours of debate today"
        far = "photosynthesis converts light energy into chemical energy in green plants daily"
        clusters = cluster_near_duplicates({"a": base, "b": near, "c": far}, threshold=0.5)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(set(clusters[0].members), {"a", "b"})
        clusters[0].assert_valid()
        with self.assertRaises(ValueError):
            DedupCluster("x", "fuzzy", "a", ("a",), ("s",), 0.5).assert_valid()


class ContaminationTests(unittest.TestCase):
    def test_layers_detect_and_clean_passes(self) -> None:
        train = {"d1": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"}
        bench = {"b1": "completely different words about other topics entirely here now"}
        self.assertEqual(check_exact_hashes({"d1": "a" * 64}, {"b1": "b" * 64}), [])
        self.assertEqual(check_normalized_exact(train, bench), [])
        self.assertEqual(check_ngram_overlap(train, bench, ngram_order=8), [])
        self.assertEqual(check_fingerprints(train, bench), [])
        self.assertEqual(check_cluster_ancestry({"d1": "cluster-1"}, {"b1": "cluster-2"}), [])

    def test_exact_and_ancestry_hit(self) -> None:
        self.assertTrue(check_exact_hashes({"d1": "a" * 64}, {"b1": "a" * 64}))
        self.assertTrue(check_cluster_ancestry({"d1": "c1"}, {"b1": "c1"}))

    def test_scan_all_receipt(self) -> None:
        receipt = scan_all(
            train_texts={"d1": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"},
            train_shas={"d1": "a" * 64},
            train_clusters={"d1": "cluster-1"},
            benchmarks={"b1": "completely different words about other topics entirely here now"},
            benchmark_shas={"b1": "b" * 64},
            eval_clusters={"b1": "cluster-2"},
            ngram_order=8,
        )
        self.assertEqual(receipt["status"], "CLEAN")
        self.assertEqual(len(receipt["sha256"]), 64)


class AccountingTests(unittest.TestCase):
    def test_source_accounting_is_exact(self) -> None:
        accounting = account_source(
            source_id="s", source_class="natural",
            documents=[("d1", "ab", "ab"), ("d2", "cdef", "cdef")],
            encode=len,
            split_of={"d1": "TRAIN", "d2": "DEV"},
        )
        self.assertEqual(
            (accounting.raw_tokens, accounting.unique_tokens, accounting.train_tokens, accounting.dev_tokens),
            (6, 6, 2, 4),
        )
        with self.assertRaises(ValueError):
            account_source(
                source_id="s", source_class="natural",
                documents=[("d1", "ab", "ab"), ("d1", "ab", "ab")],
                encode=len, split_of={"d1": "TRAIN"},
            )

    def test_mixture_feasible_and_capped(self) -> None:
        feasible = plan_mixture(
            {"natural": 1000, "code": 500}, token_budget=1000,
            target_proportions={"natural": 0.65, "code": 0.35}, max_reuse=2.0,
        )
        self.assertEqual(feasible["status"], "FEASIBLE")
        self.assertEqual(
            sum(feasible["planned_consumed"].values()), 1000  # type: ignore[union-attr]
        )
        blocked = plan_mixture(
            {"natural": 100}, token_budget=1000,
            target_proportions={"natural": 0.65, "code": 0.35}, max_reuse=1.0,
        )
        self.assertEqual(blocked["status"], "INFEASIBLE")
        self.assertIn("code", blocked["shortfalls"])  # type: ignore[union-attr]
        with self.assertRaises(ValueError):
            plan_mixture({"natural": 1}, token_budget=0, target_proportions={"natural": 1.0}, max_reuse=1.0)


@unittest.skipIf(not HAS_PYARROW, "pyarrow is not installed")
class ReaderTests(unittest.TestCase):
    def test_readers_round_trip_tiny_parquet(self) -> None:
        import tempfile
        from pathlib import Path

        import pyarrow as pa
        import pyarrow.parquet as pq

        from v5_data.readers import read_fineweb_edu, read_finemath, read_smoltalk

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(pa.table({
                "text": ["hello world this is a test document here"],
                "id": ["r1"], "url": ["http://example.test/x"],
            }), root / "fw.parquet")
            rows = list(read_fineweb_edu(root / "fw.parquet", source_id="fw"))
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0].local_id.startswith("fineweb:"))
            pq.write_table(pa.table({
                "text": ["Let x be an integer greater than zero today"],
                "url": ["http://example.test/y"],
            }), root / "fm.parquet")
            rows = list(read_finemath(root / "fm.parquet", source_id="fm"))
            self.assertEqual(len(rows), 1)
            pq.write_table(pa.table({
                "messages": [[{"role": "user", "content": "hi there friend"},
                              {"role": "assistant", "content": "hello to you too"}]],
                "source": ["test"],
            }), root / "st.parquet")
            rows = list(read_smoltalk(root / "st.parquet", source_id="st"))
            self.assertEqual(len(rows), 1)
            self.assertIn("USER:", rows[0].text)


if __name__ == "__main__":
    unittest.main()
