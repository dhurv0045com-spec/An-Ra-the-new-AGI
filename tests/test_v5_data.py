from __future__ import annotations

import unittest

from v5_data.cursor import advance, sequence_count
from v5_data.mixture import (
    allocate,
    cognition_allocation,
    slice_allocation,
)
from v5_data.pack import build_shards, pack_ledger, sampler_order
from v5_data.split import (
    SPLITS,
    assign_split,
    exact_clusters,
    scan_contamination,
)


class MixtureTests(unittest.TestCase):
    def test_slice_allocation_is_exact_5b(self) -> None:
        allocation = slice_allocation()
        self.assertEqual(sum(allocation.values()), 5_000_000_000)
        self.assertEqual(allocation["natural"], 3_250_000_000)
        self.assertEqual(allocation["code_math_formal"], 1_000_000_000)
        self.assertEqual(allocation["verified_cognition"], 750_000_000)

    def test_cognition_allocation_is_exact_750m(self) -> None:
        allocation = cognition_allocation()
        self.assertEqual(sum(allocation.values()), 750_000_000)
        self.assertEqual(allocation["relational_composition"], 150_000_000)

    def test_allocate_is_exact_and_rejects_bad_fractions(self) -> None:
        self.assertEqual(
            allocate(10, {"a": 0.34, "b": 0.355, "c": 0.305}), {"a": 3, "b": 4, "c": 3}
        )
        with self.assertRaises(ValueError):
            allocate(10, {"a": 0.5, "b": 0.4})
        with self.assertRaises(ValueError):
            allocate(-1, {"a": 1.0})


class SplitTests(unittest.TestCase):
    def test_assignment_is_deterministic_and_bounded(self) -> None:
        boundaries = {"training": 0.8, "development": 0.1, "sealed": 0.05, "fresh": 0.05}
        first = assign_split("cluster-1", salt="v5", boundaries=boundaries)
        self.assertIn(first, SPLITS)
        self.assertEqual(first, assign_split("cluster-1", salt="v5", boundaries=boundaries))
        seen = {
            assign_split(f"cluster-{index}", salt="v5", boundaries=boundaries)
            for index in range(500)
        }
        self.assertEqual(seen, set(SPLITS))

    def test_assignment_rejects_bad_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            assign_split("k", salt="s", boundaries={"training": 1.0})
        with self.assertRaises(ValueError):
            assign_split("", salt="s", boundaries={s: 0.25 for s in SPLITS})

    def test_exact_clusters_group_duplicates(self) -> None:
        clusters = exact_clusters({"a": "ab" * 32, "b": "ab" * 32, "c": "cd" * 32})
        self.assertEqual(clusters, {"ab" * 32: ["a", "b"], "cd" * 32: ["c"]})

    def test_contamination_scan_finds_verbatim_ngrams(self) -> None:
        documents = {"d1": "the quick brown fox jumps over the lazy dog today"}
        benchmarks = {"b1": "quick brown fox jumps over the lazy dog"}
        hits = scan_contamination(documents, benchmarks, ngram_order=4)
        self.assertTrue(hits)
        self.assertEqual(hits[0].benchmark_id, "b1")
        clean = scan_contamination({"d1": "entirely unrelated prose here"}, benchmarks, ngram_order=4)
        self.assertEqual(clean, [])


class PackTests(unittest.TestCase):
    def _docs(self):
        return [
            ("doc-b", [10, 11, 12], "natural"),
            ("doc-a", [20, 21], "code"),
            ("doc-c", list(range(100, 600)), "natural"),
        ]

    def test_pack_is_deterministic_and_ledger_exact(self) -> None:
        first = build_shards(self._docs(), bos=2, eos=3, pad=0, sequences_per_shard=2)
        second = build_shards(list(reversed(self._docs())), bos=2, eos=3, pad=0, sequences_per_shard=2)
        self.assertEqual([s.sha256() for s in first], [s.sha256() for s in second])
        ledger = pack_ledger(first, pad=0)
        self.assertEqual(ledger, {"real_nonpad_tokens": 511, "shards": 2})
        with self.assertRaises(ValueError):
            build_shards([("d", [1], "n"), ("d", [2], "n")], bos=2, eos=3, pad=0, sequences_per_shard=2)

    def test_sampler_order_is_deterministic_and_complete(self) -> None:
        shards = build_shards(self._docs(), bos=2, eos=3, pad=0, sequences_per_shard=1)
        hashes = [s.sha256() for s in shards]
        first = sampler_order(hashes, run_seed=7, epoch=0)
        self.assertEqual(sorted(first), list(range(len(shards))))
        self.assertEqual(first, sampler_order(list(hashes), run_seed=7, epoch=0))
        distinct = {
            tuple(sampler_order(list(hashes), run_seed=seed, epoch=0)) for seed in range(16)
        }
        self.assertGreater(len(distinct), 1)

    def test_cursor_advance_counts_real_tokens(self) -> None:
        shards = build_shards(self._docs(), bos=2, eos=3, pad=0, sequences_per_shard=2)
        order = sampler_order([s.sha256() for s in shards], run_seed=7, epoch=0)
        total_sequences = sequence_count(shards, order)
        self.assertGreater(total_sequences, 0)
        (position, _sequence), consumed = advance(
            shards, order, shard_ordinal=0, sequence_ordinal=0, sequences=1, pad=0
        )
        first_shard = shards[order[0]]
        expected = sum(1 for token in first_shard.sequences[0] if token != 0)
        self.assertEqual(consumed, expected)
        self.assertEqual((position, 0), (0, 0))
        with self.assertRaises(ValueError):
            advance(shards, order, shard_ordinal=99, sequence_ordinal=0, sequences=1, pad=0)


if __name__ == "__main__":
    unittest.main()
