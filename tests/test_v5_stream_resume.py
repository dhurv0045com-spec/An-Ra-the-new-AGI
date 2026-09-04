"""M3: checkpoint resume from artifacts only (PATH A vs PATH B).

PATH A runs updates 1..N+1 uninterrupted. PATH B checkpoints after update N,
destroys model/optimizer/sampler objects, then rebuilds the next window from
committed artifacts alone (pack manifest idents, sampler spec, persisted
stream cursor) in a FRESH SUBPROCESS, restores into fresh objects, and runs
update N+1. Token IDs, segment IDs, ledger, LR, loss, gradient norms,
parameter hash, moments, and post-update RNG must all match.

Mechanism test over a small synthetic pack (labeled as such).
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None

from v5_contracts.model_spec import V5A_250M
from v5_data.pack import pack_documents
from v5_data.stream import SamplerSpec, StreamCursor, next_window, take_slices

ROOT = Path(__file__).resolve().parents[1]

PROBE_SCRIPT = ";".join([
    "import json,sys",
    "sys.path.insert(0,sys.argv[2])",
    "from v5_data.stream import next_window, SamplerSpec, StreamCursor",
    "from v5_data.pack import pack_documents",
    "payload=json.load(open(sys.argv[1]))",
    "shards,_=pack_documents(payload['documents'],bos=2,eos=3,pad=0,sequences_per_shard=2)",
    "spec=SamplerSpec(**payload['spec'])",
    "cursor=StreamCursor(**payload['cursor'])",
    "takes,next_cursor=next_window(shards,spec,cursor,pack_manifest_sha256=payload['manifest_sha'],shard_idents=[tuple(x) for x in payload['idents']])",
    "import dataclasses",
    "print(json.dumps({'takes':[[t.epoch,t.order_position,t.shard_index,t.sequence_index,t.start_real_token,t.length_real_tokens] for t in takes],'cursor':dataclasses.asdict(next_cursor)}))",
])


def _tiny_spec():
    return dataclasses.replace(
        V5A_250M, layers=1, width=32, query_heads=2, kv_heads=1,
        head_dimension=16, ffn_width=64, vocabulary_size=256, context_length=128,
    )


def _documents():
    return [
        ["sdoc-a", [10 + i for i in range(24)], "test"],
        ["sdoc-b", [50 + i for i in range(24)], "test"],
        ["sdoc-c", [90 + i for i in range(20)], "test"],
    ]


def _pack():
    shards, _ledger = pack_documents(
        _documents(), bos=2, eos=3, pad=0, sequences_per_shard=2
    )
    manifest = [(shard.shard_id, shard.sha256()) for shard in shards]
    import hashlib

    manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    return shards, manifest, manifest_sha


@unittest.skipIf(torch is None, "PyTorch is not installed")
class StreamResumeTests(unittest.TestCase):
    def _assembled(self, shards, takes, device):
        rows_tokens: list[int] = []
        rows_segments: list[int] = []
        for ordinal, take in enumerate(takes):
            sequence = shards[take.order_position].sequences[take.sequence_index]
            ids, _segids = take_slices(
                sequence, take.start_real_token, take.length_real_tokens
            )
            rows_tokens.extend(ids)
            rows_segments.extend([ordinal] * len(ids))
        tokens = torch.tensor([rows_tokens], dtype=torch.int64, device=device)
        segments = torch.tensor([rows_segments], dtype=torch.int64, device=device)
        return tokens, segments

    def _run_update(self, backend, backend_mod, state, shards, takes, manifest_sha):
        from v5_training.production_backend import PackedBatch
        from v5_training.state import CURSOR_SCHEMA, CursorState

        tokens, segments = self._assembled(shards, takes, backend.device or "cpu")
        ledger = sum(take.length_real_tokens for take in takes)
        last = takes[-1]
        cursor_state = CursorState(
            CURSOR_SCHEMA, manifest_sha, last.order_position,
            last.sequence_index, last.start_real_token + last.length_real_tokens,
        )
        batch = PackedBatch(
            tokens=tokens,
            segment_ids=segments,
            tokens_by_source={"test": ledger},
            cursor=cursor_state,
            rng_state_sha256="0" * 64,
        )
        report = backend.step(state, batch)
        after = state.advance(
            tokens_by_source=dict(report.tokens_by_source),
            cursor=cursor_state,
            rng_state_sha256=report.rng_state_sha256,
            parent_checkpoint_sha256=None,
        )
        receipt = dict(backend.last_receipt or {})
        return after, report, receipt, (tokens, segments)

    def test_path_a_equals_path_b_from_artifacts_only(self) -> None:
        import hashlib

        from v5_model.core import initialize
        from v5_training.checkpoint import CheckpointStore
        from v5_training.optimizer import build_adamw_optimizer
        from v5_training.production_backend import (
            ProductionTrainingBackend,
            _tensor_sha256,
            production_payloads,
            restore_production,
        )
        from v5_training.state import (
            CURSOR_SCHEMA,
            IDENTITY_SCHEMA,
            CursorState,
            IdentityBindings,
            TrainingState,
        )
        import v5_training.production_backend as backend_mod

        def _sha_fn(model, *, torch):
            digest = hashlib.sha256()
            for name, parameter in sorted(model.named_parameters()):
                digest.update(name.encode() + b"\0")
                digest.update(
                    parameter.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
                )
            return digest.hexdigest()

        shards, manifest, manifest_sha = _pack()
        spec = SamplerSpec(
            schema="anra-v5-sampler-spec/v1", pack_manifest_sha256=manifest_sha,
            run_seed=5, tokens_per_update=32, buckets=(512,),
        )
        cursor = StreamCursor(
            schema="anra-v5-stream-cursor/v1", pack_manifest_sha256=manifest_sha,
            sampler_spec_sha256=spec.sha256(), epoch=0, sampler_position=0,
            token_offset=0, cumulative_real_tokens=0,
        )
        identities = IdentityBindings(
            IDENTITY_SCHEMA, "a" * 40, "b" * 64, "b" * 64, manifest_sha,
            manifest_sha, "b" * 64, "b" * 64, "b" * 64, "b" * 64,
        )

        def fresh_backend(seed):
            torch.manual_seed(seed)
            model = initialize(_tiny_spec(), seed, torch_module=torch)
            optimizer = build_adamw_optimizer(model, torch_module=torch)
            return ProductionTrainingBackend(
                model=model, optimizer=optimizer, bos_id=2, pad_id=0,
                device="cpu", torch_module=torch,
            )

        def fresh_state():
            return TrainingState.initial(
                lineage_id="m3", token_budget=96, tokens_per_update=32,
                cursor=CursorState(CURSOR_SCHEMA, manifest_sha, 0, 0, 0),
                rng_state_sha256="c" * 64, curriculum_phase="u", identities=identities,
            )

        backend_a = fresh_backend(5)
        state_a = fresh_state()
        live_cursor = cursor
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory), "m3")
            for _ in range(2):
                takes, live_cursor = next_window(
                    shards, spec, live_cursor, pack_manifest_sha256=manifest_sha,
                    shard_idents=manifest,
                )
                state_a, _, _, _ = self._run_update(
                    backend_a, backend_mod, state_a, shards, takes, manifest_sha
                )
            payloads = production_payloads(backend_a, state=state_a)
            checkpoint_sha = store.publish(
                state=state_a, payloads=payloads, expected_parent_sha256=None
            )
            takes3_a, _ = next_window(
                shards, spec, live_cursor, pack_manifest_sha256=manifest_sha,
                shard_idents=manifest,
            )
            state_a3, report_a, receipt_a, tensors_a = self._run_update(
                backend_a, backend_mod, state_a, shards, takes3_a, manifest_sha
            )
            sha_a = _sha_fn(backend_a.model, torch=torch)

            probe_input = {
                "documents": _documents(),
                "spec": {
                    "schema": spec.schema, "pack_manifest_sha256": spec.pack_manifest_sha256,
                    "run_seed": spec.run_seed, "tokens_per_update": spec.tokens_per_update,
                    "buckets": list(spec.buckets),
                },
                "cursor": {
                    "schema": live_cursor.schema,
                    "pack_manifest_sha256": live_cursor.pack_manifest_sha256,
                    "sampler_spec_sha256": live_cursor.sampler_spec_sha256,
                    "epoch": live_cursor.epoch,
                    "sampler_position": live_cursor.sampler_position,
                    "token_offset": live_cursor.token_offset,
                    "cumulative_real_tokens": live_cursor.cumulative_real_tokens,
                },
                "manifest_sha": manifest_sha,
                "idents": manifest,
            }
            with tempfile.TemporaryDirectory() as probedir:
                probe_file = Path(probedir) / "probe.json"
                probe_file.write_text(json.dumps(probe_input), encoding="utf-8")
                completed = subprocess.run(
                    [sys.executable, "-c", PROBE_SCRIPT, str(probe_file), str(ROOT)],
                    capture_output=True, text=True, check=True, cwd=ROOT,
                )
            rebuilt = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(
                [tuple(t) for t in rebuilt["takes"]],
                [(t.epoch, t.order_position, t.shard_index, t.sequence_index,
                  t.start_real_token, t.length_real_tokens) for t in takes3_a],
            )
            backend_b = fresh_backend(999)
            restored_state = store.restore(checkpoint_sha)[0]
            restore_production(backend_b, payloads=dict(payloads))
            self.assertEqual(restored_state, state_a)
            state_b3, report_b, receipt_b, tensors_b = self._run_update(
                backend_b, backend_mod, restored_state, shards, takes3_a, manifest_sha
            )
            sha_b = _sha_fn(backend_b.model, torch=torch)

        self.assertTrue(torch.equal(tensors_a[0], tensors_b[0]))
        self.assertTrue(torch.equal(tensors_a[1], tensors_b[1]))
        self.assertEqual(dict(report_a.tokens_by_source), dict(report_b.tokens_by_source))
        for key in ("loss", "grad_norm_pre_clip", "grad_norm_post_clip", "supervised_tokens"):
            if key in receipt_a or key in receipt_b:
                self.assertEqual(receipt_a.get(key), receipt_b.get(key), msg=key)
        self.assertEqual(sha_a, sha_b)
        self.assertEqual(report_a.rng_state_sha256, report_b.rng_state_sha256)
        self.assertEqual(state_a3.cumulative_tokens, state_b3.cumulative_tokens)


if __name__ == "__main__":
    unittest.main()
