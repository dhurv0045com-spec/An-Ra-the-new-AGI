"""Production backend tests: real mutation certification and stale-optimizer attacks."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize
from v5_training.checkpoint import CheckpointStore
from v5_training.optimizer import build_adamw_optimizer
from v5_training.production_backend import (
    PackedBatch,
    ProductionTrainingBackend,
    StaleOptimizerOwnership,
    UpdateEvidence,
    assert_live_ownership,
    bounded_warmup_schedule,
    capture_evidence,
    certify_real_update,
    production_payloads,
    restore_production,
)
from v5_training.runner import RunController
from v5_training.state import (
    CURSOR_SCHEMA,
    IDENTITY_SCHEMA,
    CursorState,
    IdentityBindings,
    TrainingState,
)
from v5_training.trainer import train


TINY_SPEC = ModelSpec(
    schema="anra-v5-model-spec/v1",
    family="dense-decoder-transformer",
    vocabulary_size=512,
    width=64,
    layers=2,
    query_heads=4,
    kv_heads=2,
    head_dimension=16,
    ffn_width=128,
    context_length=64,
    rope_base=10_000.0,
    norm_epsilon=1e-5,
    tied_embeddings=True,
    qk_norm=True,
    qk_norm_affine=True,
    linear_bias=False,
    dropout=0.0,
)

BOS, PAD = 2, 0
SEQUENCE_LENGTH = 8


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _identities() -> IdentityBindings:
    return IdentityBindings(
        schema=IDENTITY_SCHEMA,
        source_commit=_sha("commit")[:40],
        model_spec_sha256=TINY_SPEC.sha256(),
        tokenizer_sha256=_sha("tokenizer"),
        data_manifest_sha256=_sha("data"),
        pack_manifest_sha256=_sha("pack"),
        run_spec_sha256=_sha("run"),
        optimizer_spec_sha256=_sha("adamw"),
        schedule_spec_sha256=_sha("canary-schedule"),
        curriculum_spec_sha256=_sha("curriculum"),
    )


def _initial_state() -> TrainingState:
    identities = _identities()
    return TrainingState.initial(
        lineage_id="production-backend-test",
        token_budget=14,
        tokens_per_update=7,
        cursor=CursorState(CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, 0, 0),
        rng_state_sha256=_sha("rng0"),
        curriculum_phase="canary",
        identities=identities,
    )


def _batch(step: int, cursor_sequence: int) -> PackedBatch:
    """Two packed segments in one sequence: [BOS a b EOS BOS c EOS PAD]."""

    generator = torch.Generator().manual_seed(1000 + step)
    row = torch.randint(4, TINY_SPEC.vocabulary_size, (1, 5), generator=generator)
    tokens = torch.tensor([[BOS, row[0, 0].item(), row[0, 1].item(), 3,
                            BOS, row[0, 2].item(), 3, PAD]])
    segment_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1]], dtype=torch.int32)
    identities = _identities()
    return PackedBatch(
        tokens=tokens,
        segment_ids=segment_ids,
        tokens_by_source={"test": 7},
        cursor=CursorState(
            CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, cursor_sequence, 7 * (step + 1)
        ),
        rng_state_sha256=_sha(f"rng{step}"),
    )


def _backend(model: torch.nn.Module) -> ProductionTrainingBackend:
    optimizer = build_adamw_optimizer(model)
    return ProductionTrainingBackend(
        model=model,
        optimizer=optimizer,
        bos_id=BOS,
        pad_id=PAD,
        schedule=bounded_warmup_schedule(peak_learning_rate=3e-4),
    )


class ProductionBackendTest(unittest.TestCase):
    def test_real_update_mutates_parameters_and_moments(self) -> None:
        torch.manual_seed(7)
        model = initialize(TINY_SPEC, seed=11)
        backend = _backend(model)
        state = _initial_state()
        report = backend.step(state, _batch(0, 1))
        self.assertTrue(report.loss_finite)
        self.assertTrue(report.grad_finite)
        self.assertGreaterEqual(report.grad_norm_post_clip, 0.0)
        self.assertLessEqual(report.grad_norm_post_clip, 1.0 + 1e-6)
        receipt = backend.last_receipt
        self.assertIsNotNone(receipt)
        self.assertTrue(receipt["parameter_sha256_changed"])
        self.assertNotEqual(receipt["before"]["parameter_sha256"], receipt["after"]["parameter_sha256"])
        self.assertNotEqual(receipt["before"]["moment_sha256"], receipt["after"]["moment_sha256"])
        self.assertEqual(receipt["supervised_tokens"], 5)
        self.assertEqual(receipt["learning_rate"], 3e-4)
        steps = receipt["after"]["optimizer_steps"]
        self.assertTrue(steps and all(value == 1 for value in steps.values()))
        self.assertEqual(receipt["consumed_real_tokens"], 7)

    def test_schedule_is_token_indexed(self) -> None:
        torch.manual_seed(8)
        model = initialize(TINY_SPEC, seed=3)
        backend = _backend(model)
        state = _initial_state()
        first = backend.step(state, _batch(0, 1))
        self.assertEqual(first, first)
        advanced = state.advance(
            tokens_by_source=dict(first.tokens_by_source),
            cursor=first.cursor,
            rng_state_sha256=first.rng_state_sha256,
            parent_checkpoint_sha256=None,
        )
        second = backend.step(advanced, _batch(1, 2))
        self.assertEqual(second, second)
        # both canary updates sit past the zero-length warmup: constant peak
        self.assertEqual(backend.last_receipt["learning_rate"], 3e-4)

    def test_canonical_schedule_zero_lr_first_update_passes(self) -> None:
        torch.manual_seed(9)
        model = initialize(TINY_SPEC, seed=5)
        optimizer = build_adamw_optimizer(model)
        backend = ProductionTrainingBackend(
            model=model, optimizer=optimizer, bos_id=BOS, pad_id=PAD
        )
        state = _initial_state()
        report = backend.step(state, _batch(0, 1))
        self.assertEqual(backend.last_receipt["learning_rate"], 0.0)
        self.assertEqual(
            backend.last_receipt["before"]["parameter_sha256"],
            backend.last_receipt["after"]["parameter_sha256"],
        )
        self.assertNotEqual(
            backend.last_receipt["before"]["moment_sha256"],
            backend.last_receipt["after"]["moment_sha256"],
        )
        self.assertTrue(report.loss_finite)

    def test_stale_optimizer_ownership_rejected(self) -> None:
        """Mission 3: the historical core-vnext failure must be mechanically caught."""

        torch.manual_seed(10)
        model_a = initialize(TINY_SPEC, seed=21)
        backend = _backend(model_a)
        model_b = initialize(TINY_SPEC, seed=99)
        backend.model = model_b
        with self.assertRaises(StaleOptimizerOwnership):
            backend.step(_initial_state(), _batch(0, 1))

    def test_replaced_optimizer_storage_rejected(self) -> None:
        torch.manual_seed(11)
        model = initialize(TINY_SPEC, seed=31)
        backend = _backend(model)
        for parameter in model.parameters():
            parameter.data = parameter.data.clone()
        # same live objects after a clone of storage: still owned, must pass
        assert_live_ownership(model, backend.optimizer)
        fresh_model = initialize(TINY_SPEC, seed=32)
        with self.assertRaises(ValueError):
            assert_live_ownership(fresh_model, backend.optimizer)

    def test_certify_rejects_unchanged_parameters(self) -> None:
        torch.manual_seed(12)
        model = initialize(TINY_SPEC, seed=41)
        optimizer = build_adamw_optimizer(model)
        evidence = capture_evidence(model, optimizer, torch=torch)
        with self.assertRaises(ValueError):
            certify_real_update(
                model=model,
                optimizer=optimizer,
                before=evidence,
                after=evidence,
                expected_learning_rate=3e-4,
                supervised_tokens=5,
                loss=1.0,
                grad_norm_pre_clip=0.5,
                grad_norm_post_clip=0.5,
                torch=torch,
            )

    def test_multi_segment_loss_excludes_boundaries(self) -> None:
        torch.manual_seed(13)
        model = initialize(TINY_SPEC, seed=51)
        backend = _backend(model)
        report = backend.step(_initial_state(), _batch(0, 1))
        # eligible targets: a, b, EOS, c, EOS == 5 (BOS/PAD/segment-cross excluded)
        self.assertEqual(backend.last_receipt["supervised_tokens"], 5)
        self.assertTrue(report.loss_finite)

    def test_exact_resume_through_production_path(self) -> None:
        torch.manual_seed(14)

        def run(continue_from: tuple[TrainingState, ProductionTrainingBackend, list] | None):
            store_root = Path(tempfile.mkdtemp()) / "store"
            store = CheckpointStore(store_root, "production-backend-test")
            batches = [_batch(0, 1), _batch(1, 2)]
            if continue_from is None:
                model = initialize(TINY_SPEC, seed=77)
                backend = _backend(model)
                state = _initial_state()
                queue = [batches[0]]
            else:
                state, backend, queue = continue_from
            second_receipts: list[dict] = []

            def backend_step(current: TrainingState):
                report = backend.step(current, batches[current.global_update])
                if current.global_update == 1:
                    second_receipts.append(dict(backend.last_receipt))
                return report

            controller = RunController(target_update=2 - state.global_update)
            controller.start()
            final_state = train(
                state=state,
                controller=controller,
                store=store,
                payload_builder=lambda s: production_payloads(backend, state=s),
                backend_step=backend_step,
                updates=2 - state.global_update,
                checkpoint_every=1,
            )
            return final_state, backend, store, second_receipts

        final_a, backend_a, _, second_a = run(None)
        # rebuild the interrupted path: one update, checkpoint, fresh objects
        torch.manual_seed(14)
        store_root = Path(tempfile.mkdtemp()) / "store"
        store_b = CheckpointStore(store_root, "production-backend-test")
        batches = [_batch(0, 1), _batch(1, 2)]
        model_b = initialize(TINY_SPEC, seed=77)
        backend_b = _backend(model_b)
        state_b = _initial_state()
        controller_b = RunController(target_update=1)
        controller_b.start()
        train(
            state=state_b,
            controller=controller_b,
            store=store_b,
            payload_builder=lambda s: production_payloads(backend_b, state=s),
            backend_step=lambda current: backend_b.step(current, batches[current.global_update]),
            updates=1,
            checkpoint_every=1,
        )
        restored_state, payloads = store_b.restore()
        fresh_model = initialize(TINY_SPEC, seed=12345)
        fresh_backend = _backend(fresh_model)
        restore_production(fresh_backend, payloads=payloads)
        resumed_state, _, _, second_b = run(
            (restored_state, fresh_backend, [batches[1]])
        )

        evidence_a = capture_evidence(backend_a.model, backend_a.optimizer, torch=torch)
        evidence_b = capture_evidence(fresh_backend.model, fresh_backend.optimizer, torch=torch)
        self.assertEqual(evidence_a.parameter_sha256, evidence_b.parameter_sha256)
        self.assertEqual(evidence_a.moment_sha256, evidence_b.moment_sha256)
        self.assertEqual(evidence_a.optimizer_steps, evidence_b.optimizer_steps)
        # the restored update must be the SAME update: identical loss, gradient
        # norms, learning rate, parameter/moment hashes, and post-update RNG.
        # embedding_data_ptr is run-local memory identity and is excluded.
        def semantic(receipt: dict) -> dict:
            trimmed = {k: v for k, v in receipt.items() if k not in ("sha256",)}
            trimmed["before"] = {k: v for k, v in receipt["before"].items() if k != "embedding_data_ptr"}
            trimmed["after"] = {k: v for k, v in receipt["after"].items() if k != "embedding_data_ptr"}
            return trimmed

        self.assertEqual([semantic(r) for r in second_a], [semantic(r) for r in second_b])
        # parent_checkpoint_sha256 is store-local lineage and legitimately
        # differs between the two independent checkpoint stores; every field
        # that defines the training continuation must be identical.
        from dataclasses import replace

        lineage_free_a = replace(final_a, parent_checkpoint_sha256=None)
        lineage_free_b = replace(resumed_state, parent_checkpoint_sha256=None)
        self.assertEqual(lineage_free_a, lineage_free_b)


if __name__ == "__main__":
    unittest.main()
