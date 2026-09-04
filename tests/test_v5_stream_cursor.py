"""Authoritative stream cursor: determinism, exactness, resume, red-team.

Mechanism tests use a small synthetic pack (labeled as such); the M3/M4
integration proof runs the same code over the real frozen corpus.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None

from v5_data.pack import pack_documents, sampler_order
from v5_data.stream import (
    SamplerSpec,
    StreamCursor,
    assert_cursor_advances,
    audit_epoch,
    next_window,
    take_slices,
)


def _pack():
    documents = [
        (f"doc-{index}", [10 + index, 20 + index, 30 + index, 40 + index], "test")
        for index in range(6)
    ]
    shards, _ledger = pack_documents(
        documents, bos=2, eos=3, pad=0, sequences_per_shard=2
    )
    manifest = [(shard.shard_id, shard.sha256()) for shard in shards]
    import hashlib
    import json

    manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    return shards, manifest, manifest_sha


def _spec(manifest_sha, seed=11, per_update=8):
    return SamplerSpec(
        schema="anra-v5-sampler-spec/v1",
        pack_manifest_sha256=manifest_sha,
        run_seed=seed,
        tokens_per_update=per_update,
        buckets=(),
    )


def _cursor(manifest_sha, spec, epoch=0):
    return StreamCursor(
        schema="anra-v5-stream-cursor/v1",
        pack_manifest_sha256=manifest_sha,
        sampler_spec_sha256=spec.sha256(),
        epoch=epoch,
        sampler_position=0,
        token_offset=0,
        cumulative_real_tokens=0,
    )


class CursorContractTests(unittest.TestCase):
    def test_spec_and_cursor_reject_garbage(self) -> None:
        _shards, _manifest, manifest_sha = _pack()
        with self.assertRaises(ValueError):
            _spec(manifest_sha, per_update=0).assert_valid()
        with self.assertRaises(ValueError):
            StreamCursor(
                schema="wrong", pack_manifest_sha256=manifest_sha,
                sampler_spec_sha256="0" * 64, epoch=0, sampler_position=0,
                token_offset=0, cumulative_real_tokens=0,
            ).assert_valid()

    def test_next_window_is_deterministic_and_exact(self) -> None:
        shards, manifest, manifest_sha = _pack()
        spec = SamplerSpec(
            schema="anra-v5-sampler-spec/v1", pack_manifest_sha256=manifest_sha,
            run_seed=11, tokens_per_update=8, buckets=(512,),
        )
        first, second_cursor = next_window(
            shards, spec, _cursor(manifest_sha, spec),
            pack_manifest_sha256=manifest_sha, shard_idents=manifest,
        )
        again, again_cursor = next_window(
            shards, spec, _cursor(manifest_sha, spec),
            pack_manifest_sha256=manifest_sha, shard_idents=manifest,
        )
        self.assertEqual(first, again)
        self.assertEqual(second_cursor, again_cursor)
        total = sum(take.length_real_tokens for take in first)
        self.assertEqual(total, 8)
        self.assertEqual(second_cursor.cumulative_real_tokens, 8)

    def test_window_rebuilds_without_manifest_idents(self) -> None:
        shards, manifest, manifest_sha = _pack()
        spec = SamplerSpec(
            schema="anra-v5-sampler-spec/v1", pack_manifest_sha256=manifest_sha,
            run_seed=11, tokens_per_update=8, buckets=(512,),
        )
        with_idents, _ = next_window(
            shards, spec, _cursor(manifest_sha, spec),
            pack_manifest_sha256=manifest_sha, shard_idents=manifest,
        )
        without, _ = next_window(
            shards, spec, _cursor(manifest_sha, spec),
            pack_manifest_sha256=manifest_sha, shard_idents=None,
        )
        self.assertEqual(with_idents, without)

    def test_take_slices_resolve_padding(self) -> None:
        shards, manifest, manifest_sha = _pack()
        sequence = shards[0].sequences[0]
        tokens, segids = take_slices(sequence, 0, 2)
        self.assertEqual(len(tokens), 2)
        self.assertTrue(all(token != 0 for token in tokens))
        with self.assertRaises(ValueError):
            take_slices(sequence, 0, 10**9)

    def test_audit_reports_single_coverage(self) -> None:
        shards, manifest, manifest_sha = _pack()
        spec = SamplerSpec(
            schema="anra-v5-sampler-spec/v1", pack_manifest_sha256=manifest_sha,
            run_seed=11, tokens_per_update=8, buckets=(512,),
        )
        receipt = audit_epoch(
            shards, spec, pack_manifest_sha256=manifest_sha,
            shard_idents=manifest, epoch=0, seeds=(12, 13),
        )
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(receipt["single_coverage"])
        self.assertTrue(
            all(variant["valid_permutation"] for variant in receipt["seed_variants_differ"])
        )


class RealCorpusAuditTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_audit_passes_over_real_tokenizer_pack(self) -> None:
        import hashlib
        import json

        try:
            from v5_tokenizer.artifact import load_frozen
        except ImportError:
            self.skipTest("tokenizer artifact loader unavailable")
        try:
            from tokenizers import Tokenizer  # noqa: F401
        except ImportError:
            self.skipTest("tokenizers package is not installed")
        from pathlib import Path as _Path

        from v5_data.pack import pack_documents as _pack_documents

        root = _Path(__file__).resolve().parents[1]
        tokenizer = load_frozen(
            root / "artifacts/e1/local_tournament/tokenizer-24576.json.gz",
            expected_sha256="97e12db63b343312e5e4abc37df9ef4b01fcb1faba792a6420a4c1b15d0a7fbc",
            vocabulary_size=24576,
            trainer_config_sha256=hashlib.sha256(
                (root / "v5_tokenizer/legacy_24k_trainer_record.json").read_bytes()
            ).hexdigest(),
            corpus_manifest_sha256="eb1f0dbac64524ff4dc589c0292af6dc4c3803f48f8fe0af0a77684fea26fc67",
        )
        files = ["v5_contracts/model_spec.py", "v5_training/state.py", "blueprint/EXECUTION.md"]
        documents = [
            (f"real-{index}", tokenizer.encode((root / name).read_text(encoding="utf-8")), "natural")
            for index, name in enumerate(files)
        ]
        shards, _ledger = _pack_documents(documents, bos=2, eos=3, pad=0, sequences_per_shard=2)
        manifest = [(shard.shard_id, shard.sha256()) for shard in shards]
        manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        spec = SamplerSpec(
            schema="anra-v5-sampler-spec/v1", pack_manifest_sha256=manifest_sha,
            run_seed=21, tokens_per_update=1024, buckets=(512,),
        )
        receipt = audit_epoch(
            shards, spec, pack_manifest_sha256=manifest_sha,
            shard_idents=manifest, epoch=0, seeds=(22, 23),
        )
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(receipt["single_coverage"])


class CursorRedTeamTests(unittest.TestCase):
    def _advanced(self):
        shards, manifest, manifest_sha = _pack()
        spec = SamplerSpec(
            schema="anra-v5-sampler-spec/v1", pack_manifest_sha256=manifest_sha,
            run_seed=11, tokens_per_update=8, buckets=(512,),
        )
        before = _cursor(manifest_sha, spec)
        _, after = next_window(
            shards, spec, before,
            pack_manifest_sha256=manifest_sha, shard_idents=manifest,
        )
        return shards, manifest, manifest_sha, spec, before, after

    def test_advance_accepts_genuine_progress(self) -> None:
        _, _, _, _, before, after = self._advanced()
        assert_cursor_advances(before, after, tokens_per_update=8)

    def test_rollback_reuse_and_jump_rejected(self) -> None:
        shards, manifest, manifest_sha, spec, before, after = self._advanced()
        with self.assertRaises(ValueError):
            assert_cursor_advances(after, before, tokens_per_update=8)
        with self.assertRaises(ValueError):
            assert_cursor_advances(before, before, tokens_per_update=8)
        _, second = next_window(
            shards, spec, after,
            pack_manifest_sha256=manifest_sha, shard_idents=manifest,
        )
        with self.assertRaises(ValueError):
            assert_cursor_advances(before, second, tokens_per_update=8)

    def test_wrong_epoch_seed_pack_spec_rejected(self) -> None:
        shards, manifest, manifest_sha, spec, before, after = self._advanced()
        other_pack = StreamCursor(
            schema="anra-v5-stream-cursor/v1", pack_manifest_sha256="f" * 64,
            sampler_spec_sha256=spec.sha256(), epoch=after.epoch,
            sampler_position=after.sampler_position, token_offset=after.token_offset,
            cumulative_real_tokens=after.cumulative_real_tokens,
        )
        with self.assertRaises(ValueError):
            assert_cursor_advances(before, other_pack, tokens_per_update=8)
        with self.assertRaises(ValueError):
            next_window(
                shards, spec, before, pack_manifest_sha256="f" * 64,
                shard_idents=manifest,
            )
        other_spec = SamplerSpec(
            schema="anra-v5-sampler-spec/v1", pack_manifest_sha256=manifest_sha,
            run_seed=99, tokens_per_update=8, buckets=(512,),
        )
        with self.assertRaises(ValueError):
            next_window(
                shards, other_spec, before,
                pack_manifest_sha256=manifest_sha, shard_idents=manifest,
            )

    def test_ledger_mismatch_rejected(self) -> None:
        _, _, _, _, before, after = self._advanced()
        with self.assertRaises(ValueError):
            assert_cursor_advances(before, after, tokens_per_update=7)

    def test_empty_buckets_rejected(self) -> None:
        _shards, _manifest, manifest_sha = _pack()
        with self.assertRaises(ValueError):
            SamplerSpec(
                schema="anra-v5-sampler-spec/v1", pack_manifest_sha256=manifest_sha,
                run_seed=11, tokens_per_update=8, buckets=(),
            ).assert_valid()


if __name__ == "__main__":
    unittest.main()
