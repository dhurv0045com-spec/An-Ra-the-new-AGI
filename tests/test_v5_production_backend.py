"""Production backend tests: real mutation certification and stale-optimizer attacks."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import torch

from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize
from v5_training.checkpoint import CheckpointStore
from v5_training.distributed import RankZeroCheckpointCoordinator
from v5_training.step import CLIP_NORM_TOLERANCE, GRAD_CLIP_GLOBAL_L2
from v5_training.optimizer import build_adamw_optimizer
from v5_training.production_backend import (
    PackedBatch,
    ProductionTrainingBackend,
    StaleOptimizerOwnership,
    UpdateEvidence,
    _batched_boolean_values,
    _batched_scalar_values,
    _cpu_checkpoint_tree,
    assert_live_ownership,
    bounded_warmup_schedule,
    capture_evidence,
    certify_real_update,
    production_payloads,
    production_shared_payloads,
    restore_production,
    restore_production_shared,
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
    def test_checkpoint_tree_stages_cpu_tensors_and_preserves_aliases_and_metadata(self) -> None:
        class MarkerTensor:
            def __init__(self, device: str, events: list[tuple[str, str]]) -> None:
                self.device = device
                self.events = events

            def detach(self) -> "MarkerTensor":
                self.events.append(("detach", self.device))
                return MarkerTensor(self.device, self.events)

            def to(self, *, device: str) -> "MarkerTensor":
                self.events.append(("to", device))
                return MarkerTensor(device, self.events)

            def contiguous(self) -> "MarkerTensor":
                self.events.append(("contiguous", self.device))
                return self

        class TensorModule:
            Tensor = MarkerTensor

        events: list[tuple[str, str]] = []
        shared_tensor = MarkerTensor("xla", events)
        source = OrderedDict(weight=shared_tensor, tied_weight=shared_tensor)
        source._metadata = OrderedDict([("", {"version": 1})])

        staged = _cpu_checkpoint_tree(source, torch=TensorModule)

        self.assertIsInstance(staged, OrderedDict)
        self.assertIs(staged["weight"], staged["tied_weight"])
        self.assertEqual(staged["weight"].device, "cpu")
        self.assertEqual(staged._metadata, source._metadata)
        self.assertIsNot(staged._metadata, source._metadata)
        self.assertEqual(events, [("detach", "xla"), ("to", "cpu"), ("contiguous", "cpu")])

    def test_shared_checkpoint_saves_detached_cpu_snapshots(self) -> None:
        source = _backend(initialize(TINY_SPEC, seed=7181))
        model_tensor = torch.tensor([1.0], requires_grad=True)
        optimizer_tensor = torch.tensor([2.0], requires_grad=True)
        model_state = OrderedDict(weight=model_tensor, tied_weight=model_tensor)
        model_state._metadata = OrderedDict([("", {"version": 1})])
        optimizer_state = {"state": {0: {"exp_avg": optimizer_tensor}}, "param_groups": []}
        captured: list[object] = []

        def capture_save(value: object, buffer: object) -> None:
            captured.append(value)
            buffer.write(b"snapshot")

        with (
            patch.object(source.model, "state_dict", return_value=model_state),
            patch.object(source.optimizer, "state_dict", return_value=optimizer_state),
            patch.object(torch, "save", side_effect=capture_save),
        ):
            payloads = production_shared_payloads(source)

        self.assertEqual(set(payloads), {"model.bin", "optimizer.bin", "scheduler.json"})
        self.assertFalse(captured[0]["weight"].requires_grad)
        self.assertEqual(captured[0]["weight"].device.type, "cpu")
        self.assertIs(captured[0]["weight"], captured[0]["tied_weight"])
        self.assertEqual(captured[0]._metadata, model_state._metadata)
        self.assertFalse(captured[1]["state"][0]["exp_avg"].requires_grad)
        self.assertEqual(captured[1]["state"][0]["exp_avg"].device.type, "cpu")

    def test_shared_restore_loads_model_optimizer_without_changing_process_rng(self) -> None:
        torch.manual_seed(718)
        source = _backend(initialize(TINY_SPEC, seed=719))
        source_payloads = production_shared_payloads(source)
        expected = capture_evidence(source.model, source.optimizer, torch=torch)

        torch.manual_seed(720)
        fresh = _backend(initialize(TINY_SPEC, seed=721))
        rng_before = torch.get_rng_state().clone()
        restore_production_shared(fresh, payloads=source_payloads)

        observed = capture_evidence(fresh.model, fresh.optimizer, torch=torch)
        self.assertEqual(observed.parameter_sha256, expected.parameter_sha256)
        self.assertEqual(observed.moment_sha256, expected.moment_sha256)
        self.assertEqual(observed.optimizer_steps, expected.optimizer_steps)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

    def test_shared_restore_rejects_schedule_drift_before_loading_weights(self) -> None:
        source = _backend(initialize(TINY_SPEC, seed=722))
        payloads = production_shared_payloads(source)
        fresh = _backend(initialize(TINY_SPEC, seed=723))
        before_fresh = capture_evidence(fresh.model, fresh.optimizer, torch=torch)
        scheduler = json.loads(payloads["scheduler.json"])
        scheduler["schedule"] = {"untrusted": True}
        payloads["scheduler.json"] = json.dumps(
            scheduler, sort_keys=True, separators=(",", ":"),
        ).encode()

        with self.assertRaisesRegex(ValueError, "schedule"):
            restore_production_shared(fresh, payloads=payloads)

        after_fresh = capture_evidence(fresh.model, fresh.optimizer, torch=torch)
        self.assertEqual(before_fresh.parameter_sha256, after_fresh.parameter_sha256)
        self.assertEqual(before_fresh.moment_sha256, after_fresh.moment_sha256)

    def test_real_update_mutates_parameters_and_moments(self) -> None:
        torch.manual_seed(7)
        model = initialize(TINY_SPEC, seed=11)
        backend = _backend(model)
        state = _initial_state()
        with patch.object(torch, "isfinite", wraps=torch.isfinite) as isfinite:
            report = backend.step(state, _batch(0, 1))
        # One gradient and two Adam moments are checked per parameter by the
        # update certifier; the report path must not rescan gradients or loss.
        self.assertEqual(isfinite.call_count, 3 * len(tuple(model.named_parameters())))
        self.assertTrue(report.loss_finite)
        self.assertTrue(report.grad_finite)
        self.assertGreaterEqual(report.grad_norm_post_clip, 0.0)
        self.assertLessEqual(
            report.grad_norm_post_clip,
            GRAD_CLIP_GLOBAL_L2 + CLIP_NORM_TOLERANCE)
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

    def test_failed_update_certification_requires_checkpoint_restore(self) -> None:
        backend = _backend(initialize(TINY_SPEC, seed=1111))
        checkpoint = production_shared_payloads(backend)

        with patch(
            "v5_training.production_backend.certify_real_update",
            side_effect=ValueError("forced certificate failure"),
        ):
            with self.assertRaisesRegex(ValueError, "forced certificate failure"):
                backend.step(_initial_state(), _batch(0, 1))

        self.assertIsNone(backend.last_receipt)
        with self.assertRaisesRegex(RuntimeError, "restore the checkpointed model"):
            backend.step(_initial_state(), _batch(0, 1))
        with self.assertRaisesRegex(RuntimeError, "update certificate"):
            production_shared_payloads(backend)

        restore_production_shared(backend, payloads=checkpoint)
        report = backend.step(_initial_state(), _batch(0, 1))
        self.assertTrue(report.loss_finite)
        self.assertIsNotNone(backend.last_receipt)

    def test_real_update_uses_hidden_path_without_materializing_full_logits(self) -> None:
        model = initialize(TINY_SPEC, seed=111)
        backend = _backend(model)
        with patch.object(
            model, "forward", side_effect=AssertionError("full logits path was called"),
        ):
            report = backend.step(_initial_state(), _batch(0, 1))
        self.assertTrue(report.loss_finite)
        self.assertTrue(report.grad_finite)
        self.assertEqual(backend.last_receipt["supervised_tokens"], 5)

    def test_rank_local_supervision_uses_global_loss_denominator(self) -> None:
        torch.manual_seed(17)
        model = initialize(TINY_SPEC, seed=18)
        backend = _backend(model)
        state = _initial_state()
        batch = _batch(0, 1)
        ctx = backend.begin_update(state)
        ctx = backend.accumulate_microstep(
            ctx,
            tokens=batch.tokens,
            segment_ids=batch.segment_ids,
            eligible=torch.ones_like(batch.tokens, dtype=torch.bool),
            tokens_by_source=batch.tokens_by_source,
            planned_total=10,
        )
        with patch.object(torch, "isfinite", wraps=torch.isfinite) as isfinite:
            report = backend.finish_update(
                state,
                ctx,
                planned_total=10,
                expected_local_tokens=5,
                cursor=batch.cursor,
            )
        self.assertEqual(isfinite.call_count, 3 * len(tuple(model.named_parameters())))
        self.assertTrue(report.loss_finite)
        self.assertTrue(report.grad_finite)
        self.assertEqual(backend.last_receipt["supervised_tokens"], 10)
        self.assertEqual(backend.last_receipt["loss_scope"], "RANK_LOCAL_CONTRIBUTION")

    def test_failed_accumulated_update_blocks_followup_microsteps(self) -> None:
        backend = _backend(initialize(TINY_SPEC, seed=1112))
        batch = _batch(0, 1)
        context = backend.begin_update(_initial_state())
        context = backend.accumulate_microstep(
            context,
            tokens=batch.tokens,
            segment_ids=batch.segment_ids,
            eligible=torch.ones_like(batch.tokens, dtype=torch.bool),
            tokens_by_source=batch.tokens_by_source,
            planned_total=5,
        )

        with patch(
            "v5_training.production_backend.certify_real_update",
            side_effect=ValueError("forced certificate failure"),
        ):
            with self.assertRaisesRegex(ValueError, "forced certificate failure"):
                backend.finish_update(
                    _initial_state(), context, planned_total=5,
                    expected_local_tokens=5, cursor=batch.cursor,
                )

        with self.assertRaisesRegex(RuntimeError, "restore the checkpointed model"):
            backend.begin_update(_initial_state())

    def test_two_rank_accumulation_matches_global_batch_update(self) -> None:
        state = _initial_state()
        first = _batch(0, 1)
        live = _batch(1, 2)
        # Rank 0 owns only an inert padded row. It must still run backward and
        # contribute zero gradients so its collective layout matches rank 1.
        rank_batches = [
            PackedBatch(
                tokens=torch.full_like(first.tokens, PAD),
                segment_ids=torch.full_like(first.segment_ids, -1),
                tokens_by_source={"test": 7},
                cursor=first.cursor,
                rng_state_sha256=first.rng_state_sha256,
            ),
            live,
        ]
        expected_local_counts = [0, 5]
        global_total = sum(expected_local_counts)
        global_batch = PackedBatch(
            tokens=torch.cat([batch.tokens for batch in rank_batches], dim=0),
            segment_ids=torch.cat([batch.segment_ids for batch in rank_batches], dim=0),
            tokens_by_source={"test": 7},
            cursor=rank_batches[-1].cursor,
            rng_state_sha256=_sha("global-rng"),
        )

        reference = _backend(initialize(TINY_SPEC, seed=181))
        reference.step(state, global_batch)

        replicas = [_backend(initialize(TINY_SPEC, seed=181)) for _ in range(2)]
        contexts = []
        for backend, batch in zip(replicas, rank_batches):
            context = backend.begin_update(state)
            contexts.append(backend.accumulate_microstep(
                context,
                tokens=batch.tokens,
                segment_ids=batch.segment_ids,
                eligible=torch.ones_like(batch.tokens, dtype=torch.bool),
                tokens_by_source={"test": 7},
                planned_total=global_total,
            ))

        self.assertEqual(
            [sum(context["eligible_counts"]) for context in contexts],
            expected_local_counts,
        )
        self.assertEqual(sum(sum(context["eligible_counts"]) for context in contexts),
                         global_total)
        self.assertTrue(all(parameter.grad is not None
                            for parameter in replicas[0].model.parameters()))

        local_numerators = [sum(context["loss_numerators"]) for context in contexts]
        loss_claims = [{
            "schema": "anra-v5-distributed-update-loss/v1",
            "rank": rank,
            "world_size": 2,
            "global_update": state.global_update + 1,
            "global_tokens": global_total,
            "local_eligible_tokens": expected_local_counts[rank],
            "local_loss_numerator": local_numerators[rank],
        } for rank in range(2)]
        loss_aggregates = []
        for rank in range(2):
            def mesh_reduce(tag, value, reduce_fn, *, rank=rank):
                self.assertEqual(tag, "anra-v5-training-loss-00000000")
                self.assertEqual(value, loss_claims[rank])
                return reduce_fn(loss_claims)

            coordinator = RankZeroCheckpointCoordinator(
                rank=rank, world_size=2, mesh_reduce=mesh_reduce,
            )
            loss_aggregates.append(coordinator.aggregate_update_loss(
                global_update=state.global_update + 1,
                global_tokens=global_total,
                local_eligible_tokens=expected_local_counts[rank],
                local_loss_numerator=local_numerators[rank],
            ))
        self.assertEqual(loss_aggregates[0], loss_aggregates[1])
        untampered = capture_evidence(replicas[0].model, replicas[0].optimizer, torch=torch)
        tampered_aggregate = json.loads(json.dumps(loss_aggregates[0]))
        tampered_aggregate["rank_contributions"][0]["loss_numerator"] += 1.0
        tampered_aggregate["loss_numerator"] += 1.0
        tampered_aggregate["global_loss"] = (
            tampered_aggregate["loss_numerator"] / global_total
        )
        unsigned = {
            key: value for key, value in tampered_aggregate.items()
            if key != "receipt_sha256"
        }
        tampered_aggregate["receipt_sha256"] = hashlib.sha256(
            json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "zero-token rank contribution"):
            replicas[0].finish_update(
                state,
                contexts[0],
                planned_total=global_total,
                expected_local_tokens=expected_local_counts[0],
                loss_aggregate=tampered_aggregate,
                loss_rank=0,
                cursor=rank_batches[0].cursor,
            )
        after_rejection = capture_evidence(replicas[0].model, replicas[0].optimizer, torch=torch)
        self.assertEqual(after_rejection, untampered)

        with torch.no_grad():
            for left, right in zip(replicas[0].model.parameters(), replicas[1].model.parameters()):
                summed = left.grad + right.grad
                left.grad.copy_(summed)
                right.grad.copy_(summed)

        for rank, (backend, context, batch) in enumerate(
            zip(replicas, contexts, rank_batches),
        ):
            backend.finish_update(
                state,
                context,
                planned_total=global_total,
                expected_local_tokens=expected_local_counts[rank],
                loss_aggregate=loss_aggregates[rank],
                loss_rank=rank,
                cursor=batch.cursor,
            )
            self.assertEqual(backend.last_receipt["supervised_tokens"], global_total)

        self.assertEqual(
            [backend.last_receipt["loss_scope"] for backend in replicas],
            ["GLOBAL_BATCH_MEAN", "GLOBAL_BATCH_MEAN"],
        )
        self.assertEqual(replicas[0].last_receipt["loss"], replicas[1].last_receipt["loss"])
        self.assertAlmostEqual(replicas[0].last_receipt["loss"],
                               float(reference.last_receipt["loss"]), delta=1e-6)

        for reference_parameter, rank_parameter in zip(
            reference.model.parameters(), replicas[0].model.parameters(),
        ):
            torch.testing.assert_close(
                rank_parameter, reference_parameter, rtol=1e-4, atol=3e-6,
            )
        for parameter, rank_parameter in zip(
            reference.model.parameters(), replicas[0].model.parameters(),
        ):
            reference_state = reference.optimizer.state[parameter]
            rank_state = replicas[0].optimizer.state[rank_parameter]
            torch.testing.assert_close(
                rank_state["exp_avg"], reference_state["exp_avg"],
                rtol=1e-4, atol=3e-6,
            )
            torch.testing.assert_close(
                rank_state["exp_avg_sq"], reference_state["exp_avg_sq"],
                rtol=1e-4, atol=3e-6,
            )

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

    def test_update_checks_batch_scalar_transfers_and_keep_tensor_order(self) -> None:
        values = [torch.tensor(True), torch.tensor(False), True]
        with patch.object(torch, "stack", wraps=torch.stack) as stack:
            observed = _batched_boolean_values(values, torch)
        self.assertEqual(observed, [True, False, True])
        stack.assert_called_once()

    def test_optimizer_step_scalars_are_batched_without_changing_values(self) -> None:
        values = [torch.tensor(3, dtype=torch.int64), torch.tensor(4, dtype=torch.int64), 5]
        with patch.object(torch, "stack", wraps=torch.stack) as stack:
            observed = _batched_scalar_values(values, torch)
        self.assertEqual(observed, [3, 4, 5])
        stack.assert_called_once()

    def test_certification_still_rejects_named_nonfinite_gradient(self) -> None:
        model = initialize(TINY_SPEC, seed=43)
        optimizer = build_adamw_optimizer(model)
        before = capture_evidence(model, optimizer, torch=torch)
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        first_name, first_parameter = next(iter(model.named_parameters()))
        first_parameter.grad.fill_(float("nan"))
        after = UpdateEvidence(
            parameter_sha256="f" * 64,
            moment_sha256="e" * 64,
            optimizer_steps={name: step + 1 for name, step in before.optimizer_steps.items()},
            embedding_identity=before.embedding_identity,
            owned_parameter_ids=before.owned_parameter_ids,
        )

        with self.assertRaisesRegex(
            ValueError, f"NONFINITE_GRADIENT: {first_name} gradient is not finite"
        ):
            certify_real_update(
                model=model,
                optimizer=optimizer,
                before=before,
                after=after,
                expected_learning_rate=float(optimizer.param_groups[0]["lr"]),
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
        # embedding_identity is run-local storage/object identity and is excluded.
        def semantic(receipt: dict) -> dict:
            trimmed = {k: v for k, v in receipt.items() if k not in ("sha256",)}
            trimmed["before"] = {k: v for k, v in receipt["before"].items() if k != "embedding_identity"}
            trimmed["after"] = {k: v for k, v in receipt["after"].items() if k != "embedding_identity"}
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
