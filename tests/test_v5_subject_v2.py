"""Subject manifest V2: explicit custody, real artifact verification."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None

from v5_contracts.model_spec import V5A_250M
from v5_registry.subject_v2 import (
    SUBJECT_SCHEMA_V2,
    CoreSubjectManifestV2,
    verify_subject_artifacts,
)


def _tiny_spec():
    return dataclasses.replace(
        V5A_250M, layers=1, width=32, query_heads=2, kv_heads=1,
        head_dimension=16, ffn_width=64, vocabulary_size=256, context_length=64,
    )


def _payloads(model, optimizer, state):
    from v5_training.checkpoint import _canonical_json as store_json

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_bytes = buffer.getvalue()
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    optimizer_bytes = buffer.getvalue()
    return {
        "model.bin": model_bytes,
        "optimizer.bin": optimizer_bytes,
        "scheduler.json": store_json({"schedule_tokens": state.schedule_tokens}),
        "rng.bin": store_json({"cpu": "00"}),
        "cursor.json": store_json(__import__("dataclasses").asdict(state.cursor)),
        "ledger.json": store_json(dict(state.tokens_by_source)),
        "training_state.json": store_json(state.canonical()),
    }


def _manifest(**overrides):
    fields: dict[str, object] = {
        "checkpoint_object_sha256": "a" * 64,
        "checkpoint_manifest_sha256": "a" * 64,
        "model_payload_sha256": "b" * 64,
        "optimizer_payload_sha256": "c" * 64,
        "parameter_sha256": "d" * 64,
        "training_state_sha256": "e" * 64,
        "model_spec_sha256": "f" * 64,
        "tokenizer_artifact_sha256": "0" * 64,
        "tokenizer_identity_sha256": "1" * 64,
        "training_spec_sha256": "2" * 64,
        "data_manifest_sha256": "3" * 64,
        "pack_manifest_sha256": "4" * 64,
        "optimizer_spec_sha256": "5" * 64,
        "schedule_spec_sha256": "6" * 64,
        "curriculum_spec_sha256": "7" * 64,
        "source_commit": "8" * 40,
        "parent_checkpoint_sha256": None,
        "global_update": 1,
        "cumulative_training_tokens": 32,
        "stage": "canary",
        "seed": 7,
        "custody": "local-fsync",
        "creation_receipt_sha256": "9" * 64,
    }
    fields.update(overrides)
    return CoreSubjectManifestV2.create(**fields)


class SubjectV2Tests(unittest.TestCase):
    def test_object_and_manifest_identity_must_agree(self) -> None:
        with self.assertRaises(ValueError):
            _manifest(checkpoint_manifest_sha256="f" * 64)

    def test_round_trip_and_reject_unknown_fields(self) -> None:
        manifest = _manifest()
        clone = CoreSubjectManifestV2.from_dict(json.loads(json.dumps(manifest.canonical())))
        self.assertEqual(clone.sha256(), manifest.sha256())
        bad = dict(manifest.canonical())
        bad["extra"] = 1
        with self.assertRaises(ValueError):
            CoreSubjectManifestV2.from_dict(bad)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SubjectVerificationTests(unittest.TestCase):
    def test_verify_passes_on_real_checkpoint(self) -> None:
        from v5_model.core import initialize
        from v5_training.checkpoint import CheckpointStore
        from v5_training.optimizer import build_adamw_optimizer
        from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState

        torch.manual_seed(0)
        spec = _tiny_spec()
        model = initialize(spec, 0, torch_module=torch)
        optimizer = build_adamw_optimizer(model, torch_module=torch)
        identities = IdentityBindings(IDENTITY_SCHEMA, "a" * 40, *["b" * 64] * 8)
        state = TrainingState.initial(
            lineage_id="v2test", token_budget=32, tokens_per_update=32,
            cursor=CursorState(CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, 0, 0),
            rng_state_sha256="c" * 64, curriculum_phase="u", identities=identities,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_path = root / "tokenizer.bin"
            tokenizer_path.write_bytes(b"fake-tokenizer-bytes")
            store = CheckpointStore(root / "checkpoints", "v2test")
            state1 = state.advance(
                tokens_by_source={"t": 32},
                cursor=CursorState(CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, 1, 0),
                rng_state_sha256="d" * 64,
                parent_checkpoint_sha256=None,
            )
            payloads = _payloads(model, optimizer, state1)
            checkpoint_sha = store.publish(
                state=state1, payloads=payloads, expected_parent_sha256=None
            )
            digest = hashlib.sha256()
            for name, parameter in sorted(model.named_parameters()):
                digest.update(name.encode() + b"\0")
                digest.update(
                    parameter.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
                )
            manifest = _manifest(
                checkpoint_object_sha256=checkpoint_sha,
                checkpoint_manifest_sha256=checkpoint_sha,
                model_payload_sha256=hashlib.sha256(payloads["model.bin"]).hexdigest(),
                optimizer_payload_sha256=hashlib.sha256(payloads["optimizer.bin"]).hexdigest(),
                parameter_sha256=digest.hexdigest(),
                training_state_sha256=hashlib.sha256(payloads["training_state.json"]).hexdigest(),
                model_spec_sha256=spec.sha256(),
                tokenizer_artifact_sha256=hashlib.sha256(b"fake-tokenizer-bytes").hexdigest(),
                source_commit="a" * 40,
            )
            receipt = verify_subject_artifacts(
                manifest, checkpoint_root=root / "checkpoints", lineage_id="v2test",
                tokenizer_artifact_path=tokenizer_path, model_spec=spec,
                torch_module=torch,
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertTrue(all(receipt["checks"].values()))

    def test_tampered_payload_hash_fails_verification(self) -> None:
        import dataclasses as _dataclasses

        from v5_model.core import initialize
        from v5_training.checkpoint import CheckpointStore
        from v5_training.optimizer import build_adamw_optimizer
        from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState

        torch.manual_seed(0)
        spec = _tiny_spec()
        model = initialize(spec, 0, torch_module=torch)
        optimizer = build_adamw_optimizer(model, torch_module=torch)
        identities = IdentityBindings(IDENTITY_SCHEMA, "a" * 40, *["b" * 64] * 8)
        state = TrainingState.initial(
            lineage_id="v2test", token_budget=32, tokens_per_update=32,
            cursor=CursorState(CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, 0, 0),
            rng_state_sha256="c" * 64, curriculum_phase="u", identities=identities,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_path = root / "tokenizer.bin"
            tokenizer_path.write_bytes(b"fake-tokenizer-bytes")
            store = CheckpointStore(root / "checkpoints", "v2test")
            state1 = state.advance(
                tokens_by_source={"t": 32},
                cursor=CursorState(CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, 1, 0),
                rng_state_sha256="d" * 64,
                parent_checkpoint_sha256=None,
            )
            payloads = _payloads(model, optimizer, state1)
            checkpoint_sha = store.publish(
                state=state1, payloads=payloads, expected_parent_sha256=None
            )
            digest = hashlib.sha256()
            for name, parameter in sorted(model.named_parameters()):
                digest.update(name.encode() + b"\0")
                digest.update(
                    parameter.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
                )
            good = _manifest(
                checkpoint_object_sha256=checkpoint_sha,
                checkpoint_manifest_sha256=checkpoint_sha,
                model_payload_sha256=hashlib.sha256(payloads["model.bin"]).hexdigest(),
                optimizer_payload_sha256=hashlib.sha256(payloads["optimizer.bin"]).hexdigest(),
                parameter_sha256=digest.hexdigest(),
                training_state_sha256=hashlib.sha256(payloads["training_state.json"]).hexdigest(),
                model_spec_sha256=spec.sha256(),
                tokenizer_artifact_sha256=hashlib.sha256(b"fake-tokenizer-bytes").hexdigest(),
                source_commit="a" * 40,
            )
            bad = _dataclasses.replace(good, model_payload_sha256="f" * 64)
            receipt = verify_subject_artifacts(
                bad, checkpoint_root=root / "checkpoints", lineage_id="v2test",
                tokenizer_artifact_path=tokenizer_path, model_spec=spec,
                torch_module=torch,
            )
            self.assertEqual(receipt["status"], "FAIL")
            self.assertFalse(receipt["checks"]["model_payload"])


if __name__ == "__main__":
    unittest.main()
