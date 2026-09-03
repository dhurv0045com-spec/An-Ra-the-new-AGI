"""Registry, subject handshake, capability, and claim tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v5_registry.capability import (
    CapabilityRegistry,
    ClaimRegistry,
)
from v5_registry.registry import CheckpointRegistry
from v5_registry.subject import CoreSubjectManifest, triquetra_validation


def _manifest(**overrides) -> CoreSubjectManifest:
    fields = dict(
        checkpoint_sha256="a" * 64,
        checkpoint_file_sha256="a" * 64,
        parameter_sha256="b" * 64,
        model_spec_sha256="c" * 64,
        tokenizer_artifact_sha256="d" * 64,
        tokenizer_identity_sha256="e" * 64,
        training_spec_sha256="1" * 64,
        data_manifest_sha256="2" * 64,
        pack_manifest_sha256="3" * 64,
        optimizer_spec_sha256="4" * 64,
        schedule_spec_sha256="5" * 64,
        curriculum_spec_sha256="6" * 64,
        source_commit="0123456789abcdef0123456789abcdef01234567",
        parent_checkpoint_sha256=None,
        global_update=3,
        cumulative_training_tokens=12_288,
        stage="SOFTWARE_MINIATURE",
        seed=7,
        custody="local-ephemeral-checkpoint-store",
        creation_receipt_sha256="7" * 64,
    )
    fields.update(overrides)
    return CoreSubjectManifest.create(**fields)


class SubjectTest(unittest.TestCase):
    def test_manifest_validates_and_hashes(self) -> None:
        manifest = _manifest()
        manifest.assert_valid()
        self.assertEqual(manifest.sha256(), manifest.sha256())
        validation = triquetra_validation(manifest.canonical())
        self.assertTrue(validation["valid"], validation)

    def test_handshake_matches_triquetra_required_fields(self) -> None:
        """The handshake contract must reproduce Triquetra's validator set."""

        from v5_registry.subject import TRIQUETRA_REQUIRED_FIELDS

        expected = {
            "schema", "checkpoint_file_sha256", "parameter_sha256",
            "model_spec_sha256", "tokenizer_artifact_sha256",
            "tokenizer_identity_sha256", "training_spec_sha256",
            "data_manifest_sha256", "pack_manifest_sha256",
            "source_commit", "cumulative_training_tokens",
            "global_update", "stage", "seed",
        }
        self.assertEqual(set(TRIQUETRA_REQUIRED_FIELDS), expected)

    def test_placeholder_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _manifest(training_spec_sha256="PENDING")

    def test_checkpoint_identity_disagreement_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _manifest(checkpoint_file_sha256="f" * 64)

    def test_roundtrip(self) -> None:
        manifest = _manifest()
        restored = CoreSubjectManifest.from_dict(manifest.canonical())
        self.assertEqual(manifest.sha256(), restored.sha256())


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = CheckpointRegistry(Path(tempfile.mkdtemp()) / "registry")
        self.identity = self.registry.register(_manifest())

    def test_register_is_content_addressed(self) -> None:
        again = self.registry.register(_manifest())
        self.assertEqual(self.identity, again)
        self.assertEqual(len(self.registry.identities()), 1)

    def test_collision_rejected(self) -> None:
        # an entry whose file content disagrees with its content address
        # cannot silently replace the registered subject
        manifest = _manifest()
        identity = self.registry.register(manifest)
        path = self.registry._entry_path(identity)
        forged = dict(json.loads(path.read_text("utf-8")))
        forged["manifest"] = _manifest(parameter_sha256="9" * 64).canonical()
        path.write_text(json.dumps(forged), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.registry.register(manifest)

    def test_lifecycle_transitions(self) -> None:
        self.assertEqual(self.registry.status(self.identity), "CREATED")
        self.registry.transition(self.identity, to="IDENTITY_VERIFIED")
        self.registry.transition(self.identity, to="TRAINING_COMPLETE")
        self.assertEqual(self.registry.status(self.identity), "TRAINING_COMPLETE")
        # re-asserting an earlier stage is a forward-only no-op (canary
        # re-runs are idempotent); regressed states are unreachable
        self.registry.transition(self.identity, to="IDENTITY_VERIFIED")
        self.assertEqual(self.registry.status(self.identity), "TRAINING_COMPLETE")
        # skipping forward is always invalid
        early = self.registry.register(_manifest(checkpoint_sha256="c" * 64, checkpoint_file_sha256="c" * 64))
        with self.assertRaises(ValueError):
            self.registry.transition(early, to="PROMOTED")

    def test_attach_evaluation_advances_state(self) -> None:
        self.registry.transition(self.identity, to="IDENTITY_VERIFIED")
        self.registry.transition(self.identity, to="TRAINING_COMPLETE")
        state = self.registry.attach_evaluation(
            self.identity, evaluation_receipt_sha256="e" * 64
        )
        self.assertEqual(state, "DEV_EVALUATED")

    def test_lineage_dag(self) -> None:
        parent = _manifest()
        parent_sha = self.registry.register(parent)
        child = self.registry.register(
            _manifest(
                checkpoint_sha256="b" * 64,
                checkpoint_file_sha256="b" * 64,
                parent_checkpoint_sha256=parent.checkpoint_sha256,
                global_update=5,
                cumulative_training_tokens=20_480,
            )
        )
        children = self.registry.children_of(parent.checkpoint_sha256)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]["checkpoint_sha256"], "b" * 64)
        ancestry = self.registry.ancestry(child)
        self.assertEqual(len(ancestry), 1)
        self.assertEqual(ancestry[0]["checkpoint_sha256"], "a" * 64)

    def test_tampered_entry_rejected(self) -> None:
        path = self.registry._entry_path(self.identity)
        entry = json.loads(path.read_text("utf-8"))
        entry["manifest"]["seed"] = 99
        path.write_text(json.dumps(entry), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.registry.status(self.identity)


class CapabilityAndClaimTest(unittest.TestCase):
    def test_capability_profile_is_receipt_bound(self) -> None:
        root = Path(tempfile.mkdtemp())
        registry = CapabilityRegistry(root)
        registry.record(
            subject_manifest_sha256="a" * 64,
            family="query_binding",
            status="UNKNOWN",
            operation="ADDRESS",
            receipt_sha256="b" * 64,
        )
        profile = registry.profile("a" * 64)
        self.assertEqual(profile["query_binding"]["status"], "UNKNOWN")
        with self.assertRaises(ValueError):
            registry.record(
                subject_manifest_sha256="a" * 64,
                family="not_a_family",
                status="ROBUST",
                operation="ADDRESS",
                receipt_sha256="b" * 64,
            )
        with self.assertRaises(ValueError):
            registry.record(
                subject_manifest_sha256="a" * 64,
                family="query_binding",
                status="NATIVE_ROBUST",
                operation="ADDRESS",
                receipt_sha256="not-a-hash",
            )

    def test_claims_carry_receipts(self) -> None:
        registry = ClaimRegistry(Path(tempfile.mkdtemp()))
        sha = registry.register(
            claim_id="C1",
            text="miniature proves the production path",
            scope="SOFTWARE_MINIATURE",
            status="LOCAL_CANARY",
            supporting_receipts=["a" * 64],
        )
        self.assertEqual(len(sha), 64)
        with self.assertRaises(ValueError):
            registry.register(
                claim_id="C1",
                text="duplicate",
                scope="x",
                status="HYPOTHESIS",
                supporting_receipts=[],
            )
        with self.assertRaises(ValueError):
            registry.register(
                claim_id="C2",
                text="bad status",
                scope="x",
                status="TRUE",
                supporting_receipts=[],
            )


if __name__ == "__main__":
    unittest.main()
