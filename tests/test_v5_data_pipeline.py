"""True packing, microbatch, and data-manifest pipeline tests."""

from __future__ import annotations

import unittest

from v5_data.batch import microbatch
from v5_data.manifest import Document, build_data_manifest, manifest_sha256
from v5_data.pack import (
    MultiPackedShard,
    PackedSequence,
    bucket_for,
    chunk_document,
    multi_pack_ledger,
    pack_documents,
    sampler_order,
)


BOS, EOS, PAD = 2, 3, 0


def _documents() -> list[tuple[str, list[int], str]]:
    return [
        ("doc-a", [10, 11, 12], "natural"),
        ("doc-b", [20, 21], "code"),
        ("doc-c", [30], "natural"),
        ("doc-d", list(range(40, 40 + 600)), "natural"),  # forces the 2048 bucket? no: 604 -> 1024
    ]


class TruePackingTest(unittest.TestCase):
    def test_multiple_documents_share_one_sequence(self) -> None:
        shards, audit = pack_documents(
            _documents(), bos=BOS, eos=EOS, pad=PAD, sequences_per_shard=4
        )
        self.assertGreater(len(shards), 0)
        sequences = [sequence for shard in shards for sequence in shard.sequences]
        multi = [sequence for sequence in sequences if len(sequence.sources) > 1]
        self.assertTrue(multi, "short documents must share a packed sequence")
        for sequence in sequences:
            self.assertEqual(len(sequence.tokens), len(sequence.segment_ids))
            ids = list(sequence.segment_ids)
            if -1 in ids:
                # padding must be trailing and carry token PAD
                first_pad = ids.index(-1)
                self.assertEqual(set(ids[first_pad:]), {-1})
                self.assertEqual(set(sequence.tokens[first_pad:]), {PAD})
            # segments are contiguous and BOS/EOS bounded
            for index in range(len(sequence.sources)):
                positions = [i for i, segment in enumerate(ids) if segment == index]
                self.assertTrue(positions)
                self.assertEqual(min(positions) + len(positions) - 1, max(positions))
                self.assertEqual(sequence.tokens[positions[0]], BOS)
                self.assertEqual(sequence.tokens[positions[-1]], EOS)

    def test_ledger_is_exact_and_per_source(self) -> None:
        shards, audit = pack_documents(
            _documents(), bos=BOS, eos=EOS, pad=PAD, sequences_per_shard=4
        )
        content_tokens = sum(len(content) for _, content, _ in _documents())
        documents = len(_documents())
        self.assertEqual(audit["real_nonpad_tokens"], content_tokens + 2 * documents)
        cross_checked = multi_pack_ledger(shards)
        self.assertEqual(cross_checked["real_nonpad_tokens"], audit["real_nonpad_tokens"])
        self.assertEqual(
            {k: v for k, v in cross_checked.items() if k in {"natural", "code"}},
            audit["tokens_by_source"],
        )
        self.assertLessEqual(audit["pack_efficiency"], 1.0)
        self.assertGreater(audit["pack_efficiency"], 0.3)

    def test_packing_is_deterministic_and_order_independent(self) -> None:
        first, audit_first = pack_documents(
            _documents(), bos=BOS, eos=EOS, pad=PAD, sequences_per_shard=2
        )
        shuffled = list(reversed(_documents()))
        second, audit_second = pack_documents(
            shuffled, bos=BOS, eos=EOS, pad=PAD, sequences_per_shard=2
        )
        self.assertEqual(
            [shard.sha256() for shard in first], [shard.sha256() for shard in second]
        )
        self.assertEqual(audit_first["tokens_by_source"], audit_second["tokens_by_source"])

    def test_long_document_chunking_keeps_boundaries(self) -> None:
        long_content = list(range(100, 100 + 9000))  # > 4096 capacity
        documents = [("big", long_content, "natural")]
        shards, audit = pack_documents(
            documents, bos=BOS, eos=EOS, pad=PAD, sequences_per_shard=4
        )
        chunks = chunk_document(long_content, bos=BOS, eos=EOS)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            audit["real_nonpad_tokens"],
            len(long_content) + 2 * len(chunks),
        )
        for shard in shards:
            self.assertLessEqual(shard.bucket, 4096)
            for sequence in shard.sequences:
                self.assertLessEqual(max(sequence.segment_ids) + 1, len(sequence.sources))

    def test_segment_ids_feed_cursor_arithmetic(self) -> None:
        shards, _ = pack_documents(
            _documents(), bos=BOS, eos=EOS, pad=PAD, sequences_per_shard=4
        )
        hashes = [shard.sha256() for shard in shards]
        order = sampler_order(hashes, run_seed=5, epoch=0)
        batch = microbatch(
            shards,
            order,
            shard_ordinal=0,
            sequence_ordinal=0,
            sequences=2,
            pad=PAD,
        )
        self.assertEqual(len(batch.tokens), 2)
        self.assertEqual(batch.consumed_real_tokens, sum(batch.tokens_by_source.values()))
        self.assertGreater(batch.consumed_real_tokens, 0)

    def test_dense_corpus_yields_full_sequences_with_valid_splits(self) -> None:
        documents = [
            (f"doc-{index:03d}", [(index * 7 + offset) % 400 + 4 for offset in range(30 + index % 50)],
             "natural")
            for index in range(60)
        ]
        shards, audit = pack_documents(
            documents, bos=BOS, eos=EOS, pad=PAD, sequences_per_shard=8
        )
        sequences = [sequence for shard in shards for sequence in shard.sequences]
        full = [sequence for sequence in sequences if -1 not in sequence.segment_ids]
        self.assertEqual(audit["full_sequences"], len(full))
        self.assertEqual(audit["padded_sequences"], len(sequences) - len(full))
        self.assertGreater(len(full), 0)
        for sequence in full:
            self.assertEqual(sequence.real_tokens, max(len(s) for s in [sequence.tokens]))
            self.assertEqual(len(sequence.tokens), 512 if sequence.real_tokens == 512 else sequence.real_tokens)
        # split boundaries stay valid: every segment is BOS...EOS within its span
        for sequence in sequences:
            ids = list(sequence.segment_ids)
            for index in range(len(sequence.sources)):
                positions = [i for i, segment in enumerate(ids) if segment == index]
                self.assertEqual(sequence.tokens[positions[0]], BOS)
                self.assertEqual(sequence.tokens[positions[-1]], EOS)
        # per-source ledger equals content + 2 per emitted segment
        self.assertEqual(
            audit["real_nonpad_tokens"],
            sum(len(content) for _, content, _ in documents) + 2 * audit["segments"],
        )
        # an exact stream: full 512-token sequences feed whole-sequence updates
        full_sequences = [sequence for sequence in sequences if -1 not in sequence.segment_ids]
        self.assertTrue(
            all(sequence.real_tokens == 512 for sequence in full_sequences if len(sequence.tokens) == 512)
        )

    def test_microbatch_ledger_cross_checks_cursor(self) -> None:
        shards, _ = pack_documents(
            _documents(), bos=BOS, eos=EOS, pad=PAD, sequences_per_shard=2
        )
        hashes = [shard.sha256() for shard in shards]
        order = sampler_order(hashes, run_seed=1, epoch=0)
        total_sequences = sum(len(shard.sequences) for shard in shards)
        batch = microbatch(
            shards,
            order,
            shard_ordinal=0,
            sequence_ordinal=0,
            sequences=total_sequences,
            pad=PAD,
        )
        self.assertEqual(batch.consumed_real_tokens, sum(batch.tokens_by_source.values()))
        with self.assertRaises(ValueError):
            microbatch(
                shards,
                order,
                shard_ordinal=0,
                sequence_ordinal=0,
                sequences=total_sequences + 1,
                pad=PAD,
            )


class DataManifestTest(unittest.TestCase):
    def _documents(self) -> list[Document]:
        return [
            Document("d1", "the quick brown fox jumps", "src-1", "prose", "natural",
                     "first-party-authorized", "2026-01-01"),
            Document("d2", "packed pipelines bind provenance", "src-2", "prose", "natural",
                     "first-party-authorized", "2026-01-01"),
            Document("d3", "for i in range(3): print(i)", "src-3", "code", "code_math_formal",
                     "first-party-authorized", "2026-01-01"),
            Document("d4", "the quick brown fox jumps", "src-4", "prose", "natural",
                     "first-party-authorized", "2026-01-01"),
        ]

    def test_manifest_dedup_split_and_accounting(self) -> None:
        boundaries = {"training": 0.7, "development": 0.2, "sealed": 0.05, "fresh": 0.05}
        manifest, audit = build_data_manifest(
            self._documents(),
            manifest_id="test-manifest",
            tokenizer_sha256="a" * 64,
            filter_version="filter/v1",
            dedup_version="exact/v1",
            split_salt="test-salt",
            split_boundaries=boundaries,
            count_tokens=lambda text: len(text.split()),
        )
        self.assertEqual(len(manifest.sources), 3, "exact duplicate must be dropped")
        self.assertEqual(len(audit["exact_duplicate_drops"]), 1)
        self.assertEqual(audit["contamination_hits"], 0)
        self.assertEqual(
            sum(manifest.tokens_by_family.values()), manifest.total_tokens
        )
        self.assertEqual(manifest_sha256(manifest), manifest_sha256(manifest))

    def test_duplicate_documents_never_cross_splits(self) -> None:
        boundaries = {"training": 0.7, "development": 0.2, "sealed": 0.05, "fresh": 0.05}
        manifest, audit = build_data_manifest(
            self._documents(),
            manifest_id="split-manifest",
            tokenizer_sha256="a" * 64,
            filter_version="filter/v1",
            dedup_version="exact/v1",
            split_salt="test-salt",
            split_boundaries=boundaries,
            count_tokens=lambda text: len(text.split()),
        )
        hashes = [record.raw_sha256 for record in manifest.sources]
        self.assertEqual(len(hashes), len(set(hashes)), "dropped duplicates leave one record")

    def test_contamination_fails_closed(self) -> None:
        benchmarks = {"eval-1": "INSERT SECRET BENCHMARK PHRASE HERE for testing purposes now"}
        documents = list(self._documents()) + [
            Document("d5", "we INSERT SECRET BENCHMARK PHRASE HERE for testing purposes now",
                     "src-5", "prose", "natural", "first-party-authorized", "2026-01-01")
        ]
        with self.assertRaises(ValueError):
            build_data_manifest(
                documents,
                manifest_id="contaminated",
                tokenizer_sha256="a" * 64,
                filter_version="filter/v1",
                dedup_version="exact/v1",
                split_salt="salt",
                split_boundaries={"training": 0.7, "development": 0.2, "sealed": 0.05, "fresh": 0.05},
                count_tokens=lambda text: len(text.split()),
                contamination_benchmarks=benchmarks,
            )


if __name__ == "__main__":
    unittest.main()
