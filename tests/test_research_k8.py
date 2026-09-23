"""K8 vertical slice + I01/I02/I04/I05 focused tests.

The vertical slice connects one tiny episode through preparation, objective
routing, trainer forward/backward, checkpoint serialization, the scorer
adapter, the executive and the tool receipt path — with NO optimizer step
locally (the owner-launched E0 supplies actual updates on GPU).
"""
import json
import os
import tempfile
import unittest

import torch

from bramastra_lab.research.config import BuildConfig, seed_everything
from bramastra_lab.research.errors import CommandError
from bramastra_lab.research.experience.rendering import (
    RendererError,
    render_event_sequence,
    renderer_identity,
)
from bramastra_lab.research.learning.k8_scoring import (
    ScoringError,
    deduplicate_support,
    score_candidates_trainable,
    value_estimate_trainable,
    world_transition_token_loss,
)
from bramastra_lab.research.learning.k8_trainer import (
    AllocationContext,
    K8Trainer,
)
from bramastra_lab.research.experience.supervision import SupervisionWindow


class RenderingTests(unittest.TestCase):
    def test_provenance_leak_rejected_at_boundary(self) -> None:
        with self.assertRaises(RendererError):
            render_event_sequence([("goal", {"task": "x", "episode_id": "ep-9"})])

    def test_undeclared_role_rejected(self) -> None:
        with self.assertRaises(RendererError):
            render_event_sequence([("hidden_answer", {"value": 42})])

    def test_same_state_renders_identically(self) -> None:
        events = [("goal", {"task": "x"}), ("observation", {"value": 42})]
        self.assertEqual(render_event_sequence(events), render_event_sequence(events))
        self.assertTrue(renderer_identity())


class DifferentiableScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything(42)
        self.config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        from bramastra_lab.research.models import IntegratedModel

        self.model = IntegratedModel(self.config)

    def test_action_scores_carry_gradients(self) -> None:
        scored = score_candidates_trainable(
            self.model, self.config, [1, 2, 3], [[70, 71], [80, 81]])
        self.assertTrue(scored["scores"].requires_grad)
        loss = -scored["log_probs"].sum()
        loss.backward()
        self.assertIsNotNone(self.model.action_head.weight.grad)
        self.assertIsNotNone(self.model.decoder.embedding.weight.grad)

    def test_value_estimate_carries_gradients(self) -> None:
        value = value_estimate_trainable(self.model, self.config, [1, 2, 3])
        self.assertTrue(value.requires_grad)
        value.backward()
        self.assertIsNotNone(self.model.value_head.weight.grad)

    def test_world_loss_masks_conditioned_prefix(self) -> None:
        loss = world_transition_token_loss(
            self.model, self.config, [1, 2, 3], action={"kind": "press"},
            target_feedback={"result": "ok"})
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_duplicate_support_deduplication(self) -> None:
        support = [{"feedback": {"kind": "a"}, "terminated": False},
                   {"feedback": {"kind": "a"}, "terminated": False},
                   {"feedback": {"kind": "b"}, "terminated": True}]
        deduped = deduplicate_support(support)
        self.assertEqual(len(deduped), 2)


class AllocationGateTests(unittest.TestCase):
    def test_exhausted_allocation_refuses_updates(self) -> None:
        import time

        seed_everything(1)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        from bramastra_lab.research.models import IntegratedModel

        model = IntegratedModel(config)
        trainer = K8Trainer(config, model, device="cpu")
        trainer.begin_campaign(AllocationContext(
            allocation_id="test", device="cpu", deadline_unix=time.time() + 60,
            remaining_updates=0, job_id="j1", phase="E0"))
        with self.assertRaises(Exception):
            trainer.accumulate(_fake_batch())
        self.assertEqual(trainer.counters.optimizer_updates, 0)

    def test_expired_deadline_refuses(self) -> None:
        seed_everything(1)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        from bramastra_lab.research.models import IntegratedModel

        model = IntegratedModel(config)
        trainer = K8Trainer(config, model, device="cpu")
        trainer.begin_campaign(AllocationContext(
            allocation_id="test", device="cpu", deadline_unix=0.0,
            remaining_updates=10, job_id="j1", phase="E0"))
        # Admission fires at the window start: no new work begins once the
        # allocation's wall budget is gone, and no step is ever taken.
        with self.assertRaises(Exception):
            trainer.accumulate(_fake_batch())
        self.assertEqual(trainer.counters.optimizer_updates, 0)

    def test_admitted_window_completes_after_the_deadline_passes(self) -> None:
        """A window admitted before expiry must still finalize.

        Refusing at the boundary would strand accumulated gradients with no
        legal way to step them, leaving the trainer unpublishable.
        """
        import time

        seed_everything(1)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        from bramastra_lab.research.models import IntegratedModel

        model = IntegratedModel(config)
        trainer = K8Trainer(config, model, device="cpu")
        trainer.begin_campaign(AllocationContext(
            allocation_id="test", device="cpu",
            deadline_unix=time.time() + 60, remaining_updates=10,
            job_id="j1", phase="E0"))
        trainer.accumulate(_fake_batch())
        # The budget expires between accumulation and the optimizer step.
        trainer.allocation = AllocationContext(
            allocation_id="test", device="cpu", deadline_unix=0.0,
            remaining_updates=10, job_id="j1", phase="E0")
        trainer._k8_updates_at_start = trainer.counters.optimizer_updates
        report = trainer.finalize_update()
        self.assertEqual(trainer.counters.optimizer_updates, 1)
        self.assertIsNotNone(report)


def _fake_batch():
    from bramastra_lab.research.experience.sequences import (
        build_answer_row,
        collocate,
    )

    row = build_answer_row(
        [("goal", {"q": "1+1?"})], "2",
        provenance={"kind": "trajectory", "episode_id": "e1",
                    "task_semantic_id": "t", "split": "training",
                    "source": "test", "collection_policy": "fixed",
                    "family": "f"},
        max_tokens=64)
    return collocate([row], max_seq=64)


def _fake_pair_rows():
    from bramastra_lab.research.experience.sequences import build_answer_row

    def _row(answer: str, episode: str):
        return build_answer_row(
            [("goal", {"q": "1+1?"})], answer,
            provenance={"kind": "trajectory", "episode_id": episode,
                        "task_semantic_id": "t", "split": "training",
                        "source": "test", "collection_policy": "fixed",
                        "family": "f"},
            max_tokens=64)

    return [_row("2", "e-pair-own")], [_row("3", "e-pair-swapped")]


class EffectivePairWeightTests(unittest.TestCase):
    """Kaggle E0 regression: campaign default pair weight is 0.0 while arm-B
    windows carry pair 0.1. Pooled pair rows must finalize (scaled by the
    window weight); both-zero must still refuse."""

    def test_arm_window_pair_weight_authorizes_finalize(self) -> None:
        from bramastra_lab.research.models import IntegratedModel

        seed_everything(7)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        model = IntegratedModel(config)
        trainer = K8Trainer(config, model, device="cpu")
        self.assertEqual(trainer.pair_loss_weight, 0.0)
        self.assertEqual(trainer._effective_pair_weight(), 0.0)
        batch = _fake_batch()
        own_rows, swapped_rows = _fake_pair_rows()
        window = SupervisionWindow(
            weights={"token": 1.0, "world": 0.0, "action": 0.0,
                     "value": 0.0, "pair": 0.1, "pg": 0.0},
            enabled_terms=frozenset({"token", "pair"}))
        window.add("token", batch.target_count)
        window.add("pair", 1)
        trainer.accumulate_full_window(
            batch, window_builder=lambda _: window,
            pair_rows=(own_rows, swapped_rows))
        self.assertAlmostEqual(trainer._effective_pair_weight(), 0.1)
        report = trainer.finalize_update()
        self.assertEqual(trainer.counters.optimizer_updates, 1)
        self.assertIsNotNone(report.pair_loss)

    def test_both_zero_pair_weight_still_refuses(self) -> None:
        from bramastra_lab.research.models import IntegratedModel

        seed_everything(9)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        model = IntegratedModel(config)
        trainer = K8Trainer(config, model, device="cpu")
        batch = _fake_batch()
        own_rows, swapped_rows = _fake_pair_rows()
        window = SupervisionWindow(
            weights={"token": 1.0, "world": 0.0, "action": 0.0,
                     "value": 0.0, "pair": 0.0, "pg": 0.0},
            enabled_terms=frozenset({"token"}))
        window.add("token", batch.target_count)
        with self.assertRaises(Exception):
            trainer.accumulate_full_window(
                batch, window_builder=lambda _: window,
                pair_rows=(own_rows, swapped_rows))


class InferenceDeviceTests(unittest.TestCase):
    """Kaggle E2 regression: inference helpers must place input tensors on
    the model's device (CPU-built inputs into a CUDA model died in
    index_select on the T4s). Skipped where no accelerator exists."""

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
    def test_generation_and_scoring_run_on_cuda_model(self) -> None:
        from bramastra_lab.research.models import IntegratedModel
        from bramastra_lab.research.runtime.inference import (
            generate_free_form,
            score_finite_actions,
        )

        seed_everything(11)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        model = IntegratedModel(config).to("cuda:0")
        report = generate_free_form(model, config, [1, 2, 3], max_new_tokens=4)
        self.assertGreaterEqual(report.new_tokens, 0)
        scored = score_finite_actions(model, config, [1, 2, 3, 4],
                                       [2, 3], [True, True])
        self.assertIn(scored.selected_index, (0, 1))
        self.assertEqual(len(scored.scores), 2)


class VerticalSliceTests(unittest.TestCase):
    """One tiny episode through preparation -> objective routing -> trainer
    backward -> checkpoint serialization -> scorer -> executive -> tool path.

    No optimizer step locally (the E0 GPU gate supplies the actual update).
    """

    def test_slice_reaches_all_components(self) -> None:
        seed_everything(77)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        from bramastra_lab.research.models import IntegratedModel

        model = IntegratedModel(config)
        trainer = K8Trainer(config, model, device="cpu")
        # 1. Render a public episode through the canonical renderer.
        tokens = render_event_sequence([
            ("goal", {"task": "2+2"}),
            ("observation", {"display": "2+2=?"}),
        ])
        self.assertGreater(len(tokens), 2)
        # 2. Prepare a batch through the sequence builder.
        batch = _fake_batch()
        # 3. Route through the trainer (backward only, no optimizer step).
        trainer.accumulate(batch)
        self.assertGreater(trainer.counters.microbatches, 0)
        # 4. Checkpoint the state (discard pending to reach a boundary).
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer._pending_targets = 0
        payload = trainer.state_payload()
        self.assertIn("model", payload)
        self.assertIn("scaler_state", payload)
        # 5. Score actions through the differentiable adapter.
        scored = score_candidates_trainable(model, config, [1, 2, 3],
                                            [[70, 71], [80, 81]])
        self.assertTrue(scored["scores"].requires_grad)
        # 6. Executive selects from candidates via the scorer.
        from bramastra_lab.research.cognition.executive import (
            Executive,
            OperationRegistry,
            CognitiveOperation,
        )
        from bramastra_lab.research.cognition.workspace import CognitiveWorkspace

        candidate_ops = [CognitiveOperation(verb="PREDICT", arguments={}),
                         CognitiveOperation(verb="SUBMIT", arguments={})]
        scores = scored["scores"].tolist()
        executive = Executive(registry=OperationRegistry(verbs=("PREDICT", "SUBMIT")),
                              scorer=lambda candidates: scores,
                              decision_origin="model")
        workspace = CognitiveWorkspace(goal={"task": "test"},
                                       success_predicate="test.ok", budget=8)
        decision = executive.decide(workspace)
        self.assertIn(decision.origin, ("model", "fixed_rule", "fallback"))
        # 7. Tool receipt path: classify_failure sees through claimed success.
        from bramastra_lab.research.collection.runner import classify_failure
        from bramastra_lab.research.environments.oracles import EpisodeRecord

        record = EpisodeRecord(environment="tools", episode_id="e",
                               success=False, terminated=True, truncated=False,
                               steps=[{"action": {"kind": "write_result"},
                                        "cost": 1.0,
                                        "feedback": {"kind": "claimed_success"}}])
        self.assertEqual(classify_failure(record), "unknown")


class GatedArchitectureTests(unittest.TestCase):
    def test_zero_gate_migration_functionally_equal(self) -> None:
        from bramastra_lab.research.models.gated import (
            GatedReuseModel,
            migrate_from_parent,
        )

        seed_everything(19)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        from bramastra_lab.research.models import IntegratedModel

        parent = IntegratedModel(config)
        child = migrate_from_parent(parent, config, gates_enabled=True)
        self.assertIsInstance(child, GatedReuseModel)
        self.assertEqual(child.gate_values(), [0.0, 0.0])

    def test_gate_gradients_and_shared_blocks(self) -> None:
        from bramastra_lab.research.models.gated import (
            check_gate_gradients,
            migrate_from_parent,
            nonzero_gate_reaches_shared_blocks,
        )

        seed_everything(23)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        from bramastra_lab.research.models import IntegratedModel

        parent = IntegratedModel(config)
        child = migrate_from_parent(parent, config, gates_enabled=True)
        checks = check_gate_gradients(child)
        self.assertTrue(checks["shared_block_gradients_present"])
        self.assertTrue(checks["embedding_gradients_present"])
        self.assertTrue(nonzero_gate_reaches_shared_blocks(child))

    def test_disabled_gate_slots_for_s0(self) -> None:
        from bramastra_lab.research.models.gated import GatedReuseModel

        seed_everything(29)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        model = GatedReuseModel(config, gates_enabled=False)
        self.assertFalse(model.gates_enabled)
        self.assertEqual(model.gate_values(), [0.0, 0.0])


class K8BundleTests(unittest.TestCase):
    def test_bundle_builds_and_validates(self) -> None:
        from bramastra_lab.research.data.k8_bundle import build_k8_bundle, validate_bundle

        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_k8_bundle(tmp, training_mechanisms=4,
                                       controller_mechanisms=2,
                                       development_mechanisms=2,
                                       confirmation_mechanisms=2,
                                       tool_mechanisms=3, tool_heldout=1,
                                       meta_train=2, meta_validate=1, meta_confirm=1)
            self.assertIn("identity", manifest)
            report = validate_bundle(tmp, min_confirmation=1)
            self.assertTrue(report["valid"], report.get("issues"))

    def test_meta_labels_publish_and_resolve_for_every_reference(self) -> None:
        """Query/protected labels must exist in prepared data.

        Regression: the meta pool is generated separately from every split
        pool, so its answers were never published and its ids collided with
        split mechanism names. E5 resolved labels from split episode rows
        and refused (or silently borrowed a wrong label) for every query.
        """
        from bramastra_lab.research.data.k8_bundle import (
            build_k8_bundle, validate_bundle)

        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_k8_bundle(
                tmp, training_mechanisms=4, controller_mechanisms=2,
                development_mechanisms=2, confirmation_mechanisms=2,
                tool_mechanisms=3, tool_heldout=1, meta_train=2,
                meta_validate=1, meta_confirm=1)
            label_path = os.path.join(tmp, "meta", "meta_labels.jsonl")
            self.assertTrue(os.path.exists(label_path))
            self.assertTrue(
                any(key.replace("\\", "/") == "meta/meta_labels.jsonl"
                    for key in manifest["file_hashes"]),
                sorted(manifest["file_hashes"]))
            labels = {}
            with open(label_path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        labels[(row["family"], row["mechanism_id"])] = row
            self.assertTrue(labels)
            for (_family, mechanism_id), row in labels.items():
                self.assertIn("-meta-", mechanism_id,
                              "meta ids must be namespaced")
                self.assertIsInstance(row["public"], dict)
                self.assertNotIn(row["answer"], (None, ""))
            referenced = 0
            with open(os.path.join(tmp, "meta", "meta_tasks.jsonl"),
                      encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    task = json.loads(line)
                    for example in task["query_examples"]:
                        key = (example["family"], example["mechanism_id"])
                        self.assertIn(key, labels)
                        self.assertNotIn("answer", example)
                        referenced += 1
                    for ref in task["protected_references"]:
                        self.assertIn(
                            (ref["family"], ref["mechanism_id"]), labels)
            self.assertGreater(referenced, 0)
            report = validate_bundle(tmp, min_confirmation=1)
            self.assertTrue(report["valid"], report.get("issues"))

    def test_validate_reports_a_missing_meta_label(self) -> None:
        from bramastra_lab.research.contracts.core import content_identity
        from bramastra_lab.research.data.k8_bundle import (
            _hash_file, build_k8_bundle, validate_bundle)

        with tempfile.TemporaryDirectory() as tmp:
            build_k8_bundle(tmp, training_mechanisms=2,
                            controller_mechanisms=1, development_mechanisms=1,
                            confirmation_mechanisms=1, tool_mechanisms=1,
                            tool_heldout=1, meta_train=1, meta_validate=1,
                            meta_confirm=1)
            label_path = os.path.join(tmp, "meta", "meta_labels.jsonl")
            with open(label_path, encoding="utf-8") as handle:
                rows = [line for line in handle if line.strip()]
            dropped = json.loads(rows[0])
            with open(label_path, "w", encoding="utf-8") as handle:
                handle.writelines(rows[1:])
            # Re-hash so the meta rule, not tampering, is what fails.
            manifest_path = os.path.join(tmp, "manifest.json")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["file_hashes"] = {
                key.replace("\\", "/"): value
                for key, value in manifest["file_hashes"].items()}
            manifest["file_hashes"]["meta/meta_labels.jsonl"] = _hash_file(
                label_path)
            manifest["identity"] = content_identity({
                key: value for key, value in manifest.items()
                if key not in ("identity", "created_unix")})
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
            report = validate_bundle(tmp, min_confirmation=1)
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("meta_label_missing" in issue
                    for issue in report["issues"]),
                report["issues"])
            self.assertIn(dropped["mechanism_id"],
                          " ".join(report["issues"]))

    def test_bundle_rejects_tampered_rows(self) -> None:
        from bramastra_lab.research.data.k8_bundle import build_k8_bundle, validate_bundle

        with tempfile.TemporaryDirectory() as tmp:
            build_k8_bundle(tmp, training_mechanisms=2, controller_mechanisms=1,
                            development_mechanisms=1, confirmation_mechanisms=1,
                            tool_mechanisms=1, tool_heldout=1, meta_train=1,
                            meta_validate=1, meta_confirm=1)
            episode_path = os.path.join(tmp, "episodes")
            family_file = os.path.join(episode_path, os.listdir(episode_path)[0])
            with open(family_file, encoding="utf-8") as handle:
                content = handle.read()
            with open(family_file, "w", encoding="utf-8") as handle:
                handle.write(content + '{"tampered": true}\n')
            report = validate_bundle(tmp, min_confirmation=1)
            self.assertFalse(report["valid"])


class SupervisorTests(unittest.TestCase):
    def _make_ledger(self, tmp):
        from bramastra_lab.research.campaigns.supervisor import CampaignLedger

        return CampaignLedger(tmp)
    def test_reservation_close_and_aggregation(self) -> None:
        import shutil
        from bramastra_lab.research.campaigns.supervisor import (
            CampaignLedger,
            SupervisorError,
        )

        tmp = tempfile.mkdtemp()
        try:
            ledger = CampaignLedger(tmp)
            ledger.record_allocation("alloc-1", "src-hash-1", "data-hash-1", 480.0)
            r1 = ledger.reserve("job-1", worker="w0", device="cuda:0",
                                phase="E0", arm=None, seed=None,
                                reserved_seconds=600.0)
            closed = ledger.close_reservation(
                r1.reservation_id, status="completed", committed_updates=3,
                attempted_updates=3, supervised_exposure=12, device_seconds=120.0)
            self.assertEqual(closed.status, "completed")
            self.assertEqual(ledger.phase_consumption("E0")["committed_updates"], 3)
            again = ledger.close_reservation(
                r1.reservation_id, status="completed", committed_updates=3,
                attempted_updates=3, supervised_exposure=12, device_seconds=120.0)
            self.assertEqual(again.status, "completed")
            r2 = ledger.reserve("job-2", worker="w1", device="cuda:1",
                                phase="E0", arm=None, seed=None,
                                reserved_seconds=600.0)
            ledger.close_reservation(r2.reservation_id, status="failed",
                                     committed_updates=0, attempted_updates=0,
                                     supervised_exposure=0, device_seconds=30.0)
            with self.assertRaises(SupervisorError):
                ledger.close_reservation(r2.reservation_id, status="completed",
                                         committed_updates=1, attempted_updates=1,
                                         supervised_exposure=1, device_seconds=1.0)
            ledger.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_concurrent_reservations_cannot_exceed_capacity(self) -> None:
        import shutil
        from bramastra_lab.research.campaigns.supervisor import (
            CampaignLedger,
            SupervisorError,
        )

        tmp = tempfile.mkdtemp()
        try:
            ledger = CampaignLedger(tmp)
            ledger.record_allocation("alloc-1", "src", "data", 10.0)
            # R02 per-device exclusivity: two different GPUs may each reserve
            # up to the remaining time concurrently (admitted).
            ledger.reserve("j1", worker="w0", device="cuda:0", phase="E0",
                           arm=None, seed=None, reserved_seconds=540.0)
            ledger.reserve("j2", worker="w1", device="cuda:1", phase="E0",
                           arm=None, seed=None, reserved_seconds=540.0)
            # Overlapping second reservation on the SAME GPU is refused.
            with self.assertRaises(SupervisorError):
                ledger.reserve("j3", worker="w0", device="cuda:0", phase="E0",
                               arm=None, seed=None, reserved_seconds=540.0)
            ledger.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_closed_row_rebinding_refuses_loudly(self) -> None:
        import shutil
        from bramastra_lab.research.campaigns.supervisor import (
            CampaignLedger,
            SupervisorError,
        )

        tmp = tempfile.mkdtemp()
        try:
            ledger = CampaignLedger(tmp)
            ledger.record_allocation("alloc-1", "src", "data", 60.0)
            res = ledger.reserve("job-9", worker="w0", device="cuda:0",
                                 phase="E1", arm="B", seed=1702,
                                 reserved_seconds=300.0)
            # Crash-recovery while still open: same fields rebind fine.
            again = ledger.reserve("job-9", worker="w0", device="cuda:0",
                                   phase="E1", arm="B", seed=1702,
                                   reserved_seconds=300.0)
            self.assertEqual(again.reservation_id, res.reservation_id)
            ledger.close_reservation(res.reservation_id, status="failed")
            # Closed rows must never silently rebind (workers bound to them
            # would noop every step, then die on checkpoint confusion).
            with self.assertRaises(SupervisorError):
                ledger.reserve("job-9", worker="w0", device="cuda:0",
                               phase="E1", arm="B", seed=1702,
                               reserved_seconds=300.0)
            ledger.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
