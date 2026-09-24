"""One-shot Phase-One custody claims fail closed under retries and races."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from signac_100m.custody import (
    CustodyClaim,
    CustodyClaimInProgress,
    CustodyError,
    CustodyPlan,
    CustodyReceipt,
    PhaseOneCustody,
)
from v5_registry.registry import CheckpointRegistry
from v5_registry.subject import CoreSubjectManifest


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _registered_dev_subject(
    registry: CheckpointRegistry, *, seed: int, label: str, recipe_label: str = "recipe-a"
) -> tuple[CoreSubjectManifest, str]:
    subject = CoreSubjectManifest.create(
        checkpoint_sha256=_hash(f"{label}-checkpoint"),
        checkpoint_file_sha256=_hash(f"{label}-checkpoint"),
        parameter_sha256=_hash(f"{label}-parameters"),
        model_spec_sha256=_hash("model"),
        tokenizer_artifact_sha256=_hash("tokenizer-artifact"),
        tokenizer_identity_sha256=_hash("tokenizer-identity"),
        training_spec_sha256=_hash(f"{recipe_label}-training"),
        data_manifest_sha256=_hash(f"{recipe_label}-training-data"),
        pack_manifest_sha256=_hash(f"{recipe_label}-pack"),
        optimizer_spec_sha256=_hash(f"{recipe_label}-optimizer"),
        schedule_spec_sha256=_hash(f"{recipe_label}-schedule"),
        curriculum_spec_sha256=_hash(f"{recipe_label}-curriculum"),
        source_commit="0123456789abcdef0123456789abcdef01234567",
        parent_checkpoint_sha256=None,
        global_update=2,
        cumulative_training_tokens=8192,
        stage="PHASE1_TEST",
        seed=seed,
        custody="local-test-only",
        creation_receipt_sha256=_hash(f"{label}-creation"),
        source_tree_sha256=_hash("source-tree"),
    )
    identity = registry.register(subject)
    registry.transition(identity, to="IDENTITY_VERIFIED")
    registry.transition(identity, to="TRAINING_COMPLETE")
    dev_receipt = _hash(f"{label}-development-evaluation")
    registry.attach_evaluation(identity, evaluation_receipt_sha256=dev_receipt)
    return subject, dev_receipt


def _plan(
    subject: CoreSubjectManifest,
    dev_receipt: str,
    *,
    split: str = "sealed",
    label: str = "sealed-1",
    dataset: str | None = None,
    fixture: str | None = None,
    evaluation_seed: int = 73,
    clusters: tuple[str, ...] | None = None,
    predecessor: str | None = None,
) -> CustodyPlan:
    return CustodyPlan(
        split=split,
        dataset_manifest_sha256=_hash(dataset or f"{label}-dataset"),
        fixture_sha256=_hash(f"{fixture or label}-fixture"),
        protocol_sha256=_hash(f"{label}-protocol"),
        protocol_contract_sha256=_hash("phase1-frozen-protocol-contract"),
        evaluator_sha256=_hash("frozen-evaluator"),
        subject_manifest_sha256=subject.sha256(),
        source_tree_sha256=_hash("source-tree"),
        training_seed=subject.seed,
        evaluation_seed=evaluation_seed,
        development_receipt_sha256=dev_receipt,
        cluster_sha256s=clusters or (_hash(f"{label}-cluster-a"), _hash(f"{label}-cluster-b")),
        round_id=label,
        predecessor_receipt_sha256=predecessor,
    )


def _complete(custody: PhaseOneCustody, claim: CustodyClaim) -> CustodyReceipt:
    return custody.complete(
        claim,
        outcome="PASS",
        evaluation_receipt_sha256=_hash("evaluation-result"),
        evidence_artifact_sha256=_hash("task-evidence-artifact"),
        custody_attestation_sha256=_hash("custodian-attestation"),
        independent_review_sha256=_hash("independent-review"),
    )


class CustodyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry = CheckpointRegistry(self.root / "registry")
        self.subject, self.dev_receipt = _registered_dev_subject(
            self.registry, seed=11, label="subject-a"
        )
        self.custody = PhaseOneCustody(self.root / "custody", self.registry)
        self.plan = _plan(self.subject, self.dev_receipt)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_claim_and_terminal_retry_return_existing_receipt(self) -> None:
        claimed = self.custody.claim(self.plan)
        self.assertIsInstance(claimed, CustodyClaim)
        receipt = _complete(self.custody, claimed)

        replay = self.custody.claim(self.plan)
        self.assertIsInstance(replay, CustodyReceipt)
        self.assertEqual(replay.sha256(), receipt.sha256())
        self.assertEqual(replay.outcome, "PASS")

    def test_passing_sealed_receipt_advances_registry_but_not_promotion(self) -> None:
        claim = self.custody.claim(self.plan)
        receipt = _complete(self.custody, claim)

        state = self.registry.attach_sealed_confirmation(
            self.subject.sha256(), custody_receipt=receipt.as_record()
        )
        self.assertEqual(state, "SEALED_EVALUATED")
        self.assertEqual(
            self.registry.attach_sealed_confirmation(
                self.subject.sha256(), custody_receipt=receipt.as_record()
            ),
            "SEALED_EVALUATED",
        )
        with self.assertRaisesRegex(ValueError, "verified Phase-One custody receipt"):
            self.registry.transition(self.subject.sha256(), to="PROMOTED")

    def test_registry_rejects_tampered_custody_receipt_before_transition(self) -> None:
        claim = self.custody.claim(self.plan)
        receipt = _complete(self.custody, claim)
        tampered = receipt.as_record()
        tampered["plan"] = {**tampered["plan"], "source_tree_sha256": _hash("changed-tree")}

        with self.assertRaisesRegex(ValueError, "receipt hash mismatch"):
            self.registry.attach_sealed_confirmation(
                self.subject.sha256(), custody_receipt=tampered
            )
        self.assertEqual(self.registry.status(self.subject.sha256()), "DEV_EVALUATED")

    def test_pending_claim_cannot_be_reopened(self) -> None:
        self.custody.claim(self.plan)
        with self.assertRaises(CustodyClaimInProgress):
            self.custody.claim(self.plan)

    def test_changed_plan_cannot_reuse_a_claimed_dataset(self) -> None:
        self.custody.claim(self.plan)
        changed = _plan(
            self.subject,
            self.dev_receipt,
            label="changed-protocol",
            dataset="sealed-1-dataset",
        )
        with self.assertRaisesRegex(CustodyError, "already reserved"):
            self.custody.claim(changed)

    def test_fixture_identity_cannot_be_rewrapped_under_new_manifest(self) -> None:
        self.custody.claim(self.plan)
        changed_manifest = _plan(
            self.subject,
            self.dev_receipt,
            label="rewrapped-fixture",
            dataset="different-dataset-manifest",
            fixture="sealed-1",
        )
        with self.assertRaisesRegex(CustodyError, "fixture identity is already reserved"):
            self.custody.claim(changed_manifest)

    def test_source_clusters_cannot_be_reused_under_new_surface_hashes(self) -> None:
        self.custody.claim(self.plan)
        changed_surface = _plan(
            self.subject,
            self.dev_receipt,
            label="repacked-same-clusters",
            dataset="new-dataset-manifest",
            fixture="new-fixture",
            clusters=self.plan.cluster_sha256s,
        )
        with self.assertRaisesRegex(CustodyError, "cluster identity is already reserved"):
            self.custody.claim(changed_surface)

    def test_failed_attempt_is_terminal_and_replayed_without_new_claim(self) -> None:
        claim = self.custody.claim(self.plan)
        failed = self.custody.fail(claim, failure_receipt_sha256=_hash("partial-attempt"))

        replay = self.custody.claim(self.plan)
        self.assertIsInstance(replay, CustodyReceipt)
        self.assertEqual(replay.sha256(), failed.sha256())
        self.assertEqual(replay.outcome, "FAILED")
        with self.assertRaises(CustodyError):
            _complete(self.custody, claim)

    def test_same_dataset_and_fixture_cannot_be_relabelled_across_splits(self) -> None:
        sealed_claim = self.custody.claim(self.plan)
        sealed_receipt = _complete(self.custody, sealed_claim)
        relabelled = _plan(
            self.subject,
            self.dev_receipt,
            split="fresh",
            label="fresh-round",
            dataset="sealed-1-dataset",
            fixture="sealed-1",
            evaluation_seed=88,
            predecessor=sealed_receipt.sha256(),
        )
        with self.assertRaisesRegex(CustodyError, "different frozen plan"):
            self.custody.claim(relabelled)

    def test_claim_requires_attached_development_receipt(self) -> None:
        wrong_receipt = _plan(
            self.subject,
            _hash("unattached-development-receipt"),
            label="wrong-dev-receipt",
        )
        with self.assertRaisesRegex(CustodyError, "does not name a development receipt"):
            self.custody.claim(wrong_receipt)

    def test_fresh_requires_new_subject_seed_surface_and_disjoint_clusters(self) -> None:
        sealed_claim = self.custody.claim(self.plan)
        sealed_receipt = _complete(self.custody, sealed_claim)
        fresh_subject, fresh_dev_receipt = _registered_dev_subject(
            self.registry, seed=12, label="subject-b"
        )
        fresh_plan = _plan(
            fresh_subject,
            fresh_dev_receipt,
            split="fresh",
            label="fresh-1",
            evaluation_seed=74,
            predecessor=sealed_receipt.sha256(),
        )
        fresh_claim = self.custody.claim(fresh_plan)
        self.assertIsInstance(fresh_claim, CustodyClaim)

        overlapping = _plan(
            fresh_subject,
            fresh_dev_receipt,
            split="fresh",
            label="fresh-overlap",
            evaluation_seed=75,
            clusters=(self.plan.cluster_sha256s[0], _hash("new-fresh-cluster")),
            predecessor=sealed_receipt.sha256(),
        )
        with self.assertRaisesRegex(CustodyError, "share source clusters"):
            self.custody.claim(overlapping)

    def test_fresh_replication_requires_the_same_frozen_training_recipe(self) -> None:
        sealed_claim = self.custody.claim(self.plan)
        sealed_receipt = _complete(self.custody, sealed_claim)
        changed_recipe_subject, changed_recipe_dev = _registered_dev_subject(
            self.registry, seed=12, label="subject-recipe-change", recipe_label="recipe-b"
        )
        changed_recipe = _plan(
            changed_recipe_subject,
            changed_recipe_dev,
            split="fresh",
            label="fresh-recipe-change",
            evaluation_seed=74,
            predecessor=sealed_receipt.sha256(),
        )
        with self.assertRaisesRegex(CustodyError, "training recipe differs"):
            self.custody.claim(changed_recipe)

    def test_claim_requires_dev_evaluated_lifecycle(self) -> None:
        other = CoreSubjectManifest.from_dict(
            {
                **self.subject.canonical(),
                "checkpoint_sha256": _hash("untrained-checkpoint"),
                "checkpoint_file_sha256": _hash("untrained-checkpoint"),
            }
        )
        identity = self.registry.register(other)
        untrained_plan = _plan(other, _hash("receipt"), label="untrained")
        with self.assertRaisesRegex(CustodyError, "requires a DEV_EVALUATED subject"):
            self.custody.claim(untrained_plan)
        self.assertEqual(self.registry.status(identity), "CREATED")

    def test_concurrent_claim_allows_exactly_one_consumer(self) -> None:
        def attempt() -> object:
            try:
                return self.custody.claim(self.plan)
            except CustodyError as error:
                return error

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _index: attempt(), range(6)))
        claims = [result for result in results if isinstance(result, CustodyClaim)]
        self.assertEqual(len(claims), 1)
        self.assertEqual(sum(isinstance(result, CustodyError) for result in results), 5)


if __name__ == "__main__":
    unittest.main()
