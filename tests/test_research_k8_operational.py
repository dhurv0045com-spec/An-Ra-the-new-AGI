"""Operational learner acceptance tests (O01-O10 local criteria).

No optimizer steps, no GPU, no paid compute anywhere in this file:
- real tiny-model backward checks discard gradients (never finalize);
- deterministic doubles stand in for learning with fixture labels;
- GPU-only paths assert their refused/skipped shapes locally.
"""
import json
import os
import tempfile
import unittest

import torch


def _tiny_trainer(*, require_allocation=True):
    from bramastra_lab.research.config import BuildConfig, seed_everything
    from bramastra_lab.research.learning.k8_trainer import K8Trainer
    from bramastra_lab.research.models import IntegratedModel

    seed_everything(7)
    config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
    model = IntegratedModel(config)
    trainer = K8Trainer(config, model, device="cpu",
                        precision="fp32",
                        require_allocation=require_allocation)
    return config, trainer


def _tiny_batch():
    from bramastra_lab.research.experience.sequences import (
        build_answer_row, collocate)

    row = build_answer_row(
        [("goal", {"q": "1+1?"})], "2",
        provenance={"kind": "trajectory", "episode_id": "e1",
                    "task_semantic_id": "t", "split": "training",
                    "source": "test", "collection_policy": "fixed",
                    "family": "f"},
        max_tokens=64)
    return collocate([row], max_seq=64)


def _fake_job(**overrides):
    from types import SimpleNamespace

    base = {"job_id": "E1-A-1701", "physical_device": "cpu",
            "local_device": "cpu",
            "phase": "E1", "source_hash": "src-test",
            "allocation_id": "alloc-test", "reservation_id": "res-test",
            "reservation_deadline_unix": 9e9,
            "run_dir": tempfile.mkdtemp(),
            "data_dir": tempfile.mkdtemp()}
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_reservation(**overrides):
    base = {"allocation_id": "alloc-test", "reservation_id": "res-test",
            "job_id": "E1-A-1701", "device": "cpu", "phase": "E1",
            "deadline_unix": 9e9, "remaining_updates": 4,
            "source_hash": "src-test"}
    base.update(overrides)
    return base


class O01SessionTests(unittest.TestCase):
    def test_missing_authority_refuses_finalize(self) -> None:
        _, trainer = _tiny_trainer(require_allocation=True)
        trainer.accumulate(_tiny_batch())
        with self.assertRaises(Exception):
            trainer.finalize_update()
        self.assertEqual(trainer.counters.optimizer_updates, 0)

    def test_mismatched_authority_refuses(self) -> None:
        from bramastra_lab.research.campaigns.phases.session import (
            bind_reservation)

        _, trainer = _tiny_trainer(require_allocation=True)
        handle = {"trainer": trainer}
        job = _fake_job()
        with self.assertRaises(Exception):
            bind_reservation(handle, _fake_reservation(device="cuda:0"),
                             job=job)
        with self.assertRaises(Exception):
            bind_reservation(handle, _fake_reservation(phase="E3"), job=job)
        with self.assertRaises(Exception):
            bind_reservation(handle, _fake_reservation(job_id="E1-B-1701"),
                             job=job)
        with self.assertRaises(Exception):
            bind_reservation(handle, _fake_reservation(source_hash="other"),
                             job=job)
        with self.assertRaises(Exception):
            bind_reservation(handle, _fake_reservation(deadline_unix=1.0),
                             job=job)
        self.assertEqual(trainer.counters.optimizer_updates, 0)

    def test_noop_boundary_proves_admission_without_step(self) -> None:
        from bramastra_lab.research.campaigns.phases.session import (
            bind_job_reservation, drive_noop_boundary)
        from bramastra_lab.research.experience.supervision import (
            SupervisionWindow)

        _, trainer = _tiny_trainer(require_allocation=True)
        handle = {"trainer": trainer, "model": trainer.model}
        bind_job_reservation(handle, _fake_job(), remaining_updates=4)
        batch = _tiny_batch()
        window = SupervisionWindow(
            weights={"token": 1.0, "world": 0.0, "action": 0.0,
                     "value": 0.0, "pair": 0.0, "pg": 0.0},
            enabled_terms=frozenset({"token"}))
        window.add("token", batch.target_count)
        proof = drive_noop_boundary(handle, batch=batch, window=window,
                                    extra=None)
        self.assertTrue(proof["admitted"])
        self.assertFalse(proof["step_taken"])
        self.assertEqual(proof["optimizer_updates_before"],
                         proof["optimizer_updates_after"])
        self.assertEqual(trainer.counters.optimizer_updates, 0)
        self.assertTrue(proof["named_grads_finite"])

    def test_doubles_classify_zero_update(self) -> None:
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)
        from bramastra_lab.research.campaigns.phases.session import (
            bind_job_reservation)

        ops = RecordingDoubleOps()
        handle = ops.init_model(seed=1, profile="k8-campaign", device="cpu")
        self.assertEqual(
            bind_job_reservation(handle, _fake_job(), remaining_updates=4),
            "zero-update-double")

    def test_step_or_noop_gates(self) -> None:
        from bramastra_lab.research.campaigns.phases.session import (
            bind_job_reservation, step_or_noop)
        from bramastra_lab.research.experience.supervision import (
            SupervisionWindow)

        # Real trainer, well-formed but non-ledger reservation: no-op
        # boundary with zero steps (fields alone never authorize).
        _, trainer = _tiny_trainer(require_allocation=True)
        handle = {"trainer": trainer, "model": trainer.model}
        job = _fake_job()
        bind_job_reservation(handle, job, remaining_updates=4)
        batch = _tiny_batch()
        window = SupervisionWindow(
            weights={"token": 1.0, "world": 0.0, "action": 0.0,
                     "value": 0.0, "pair": 0.0, "pg": 0.0},
            enabled_terms=frozenset({"token"}))
        window.add("token", batch.target_count)
        from bramastra_lab.research.campaigns.phases.session import (
            job_reservation_record)
        outcome = step_or_noop(
            None, handle, job, batch=batch, window=window, extra={},
            pair_rows=None,
            reservation=job_reservation_record(job, remaining_updates=4))
        self.assertEqual(outcome.get("boundary"), "noop")
        self.assertEqual(outcome.get("committed"), 0)
        self.assertEqual(trainer.counters.optimizer_updates, 0)
        # Doubles take the recording path with fixture counts.
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)

        ops = RecordingDoubleOps()
        double = ops.init_model(seed=1, profile="k8-campaign", device="cpu")
        window2 = SupervisionWindow(
            weights={"token": 1.0, "world": 0.0, "action": 0.0,
                     "value": 0.0, "pair": 0.0, "pg": 0.0},
            enabled_terms=frozenset({"token"}))
        window2.add("token", batch.target_count)
        recorded = step_or_noop(
            ops, double, job, batch=batch, window=window2, extra={},
            pair_rows=None, reservation=None)
        self.assertEqual(recorded.get("committed"), 1)


class O02WindowTests(unittest.TestCase):
    def test_identical_windows_give_identical_gradients(self) -> None:
        from bramastra_lab.research.experience.supervision import (
            SupervisionWindow)

        _, trainer_a = _tiny_trainer(require_allocation=False)
        _, trainer_b = _tiny_trainer(require_allocation=False)
        batch = _tiny_batch()

        def _run(trainer):
            window = SupervisionWindow(
                weights={"token": 1.0, "world": 0.0, "action": 0.0,
                         "value": 0.0, "pair": 0.0, "pg": 0.0},
                enabled_terms=frozenset({"token"}))
            window.add("token", batch.target_count)
            trainer.accumulate_full_window(
                batch, window_builder=lambda _: window)
            grads = {name: param.grad.detach().cpu().clone()
                     for name, param in trainer.model.named_parameters()
                     if param.grad is not None}
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer._pending_targets = 0
            return grads

        grads_a = _run(trainer_a)
        grads_b = _run(trainer_b)
        self.assertTrue(grads_a)
        self.assertEqual(set(grads_a), set(grads_b))
        for name in grads_a:
            self.assertTrue(torch.equal(grads_a[name], grads_b[name]))
        self.assertEqual(trainer_a.counters.optimizer_updates, 0)
        self.assertEqual(trainer_b.counters.optimizer_updates, 0)

    def test_pair_without_rows_refused_before_step(self) -> None:
        from bramastra_lab.research.config import BuildConfig, seed_everything
        from bramastra_lab.research.experience.supervision import (
            SupervisionWindow)
        from bramastra_lab.research.learning.k8_trainer import K8Trainer
        from bramastra_lab.research.models import IntegratedModel

        seed_everything(7)
        config = BuildConfig.from_dict(
            {"model": {"profile": "tiny"},
             "training": {"pair_loss_weight": 0.5}})
        trainer = K8Trainer(config, IntegratedModel(config), device="cpu",
                            precision="fp32", require_allocation=False)
        batch = _tiny_batch()
        window = SupervisionWindow(
            weights={"token": 1.0, "world": 0.0, "action": 0.0,
                     "value": 0.0, "pair": 0.5, "pg": 0.0},
            enabled_terms=frozenset({"token", "pair"}))
        window.add("token", batch.target_count)
        window.add("pair", 1)
        # Accumulation stages the window; the pair boundary refuses at
        # finalization (missing renderings are never silently zero).
        trainer.accumulate_full_window(
            batch, window_builder=lambda _: window)
        with self.assertRaises(Exception):
            trainer.finalize_update()
        self.assertEqual(trainer.counters.optimizer_updates, 0)

    def test_checkpoint_roundtrip_and_refusals(self) -> None:
        import hashlib

        from bramastra_lab.research.runtime import checkpoint as ckpt

        run = tempfile.mkdtemp()
        _, trainer = _tiny_trainer(require_allocation=False)
        payload = trainer.state_payload()
        token = ckpt.acquire_writer_fence(run)
        try:
            manifest = ckpt.save_checkpoint(
                run, payload, run_id="o02-test", update_index=0,
                config_identity="cfg-o02", tokenizer_identity="tok-o02",
                data_identity="data-o02", parent_checkpoint_id=None,
                code_identity="code-o02", writer_token=token)
        finally:
            ckpt.release_writer_fence(run, token)
        loaded, reloaded = ckpt.load_checkpoint(
            run, checkpoint_id=manifest.checkpoint_id,
            expect_config_identity="cfg-o02")
        self.assertEqual(reloaded.checkpoint_id, manifest.checkpoint_id)
        with self.assertRaises(Exception):
            ckpt.load_checkpoint(
                run, checkpoint_id=manifest.checkpoint_id,
                expect_config_identity="different-config")
        with self.assertRaises(Exception):
            ckpt.save_checkpoint(
                run, payload, run_id="o02-stale", update_index=1,
                config_identity="cfg-o02", tokenizer_identity="tok-o02",
                data_identity="data-o02", parent_checkpoint_id=None,
                code_identity="code-o02", writer_token="wrong-token")
        proof = ckpt.restore_verify(
            run_dir=run, checkpoint_id=manifest.checkpoint_id,
            expect_config_identity="cfg-o02",
            expect_tokenizer_identity="tok-o02")
        self.assertTrue(proof["restored_ok"])
        _ = hashlib.sha256(b"o02").hexdigest()


class O03CalibrationTests(unittest.TestCase):
    def test_update_target_selection(self) -> None:
        from bramastra_lab.research.campaigns import calibration as cal

        # 0.75*3600/2.0 = 1350 updates for E1.
        self.assertEqual(
            cal.select_update_target(phase="E1", worst_update_seconds=2.0,
                                     slot_seconds=3600.0), 1350)
        # Below informative minimum -> refusal, not a small silent target.
        with self.assertRaises(Exception):
            cal.select_update_target(phase="E1", worst_update_seconds=100.0,
                                     slot_seconds=3600.0)
        with self.assertRaises(Exception):
            cal.select_update_target(phase="E1", worst_update_seconds=0.0,
                                     slot_seconds=3600.0)
        with self.assertRaises(Exception):
            cal.select_update_target(phase="E9", worst_update_seconds=1.0,
                                     slot_seconds=3600.0)

    def test_confirmation_inventory_selection(self) -> None:
        from bramastra_lab.research.campaigns import calibration as cal

        # Fast eval loop fits the largest inventory.
        self.assertEqual(
            cal.select_confirmation_inventory(eval_cases_per_second=10.0), 128)
        # Slower loops step down honestly; far too slow is insufficient.
        self.assertEqual(
            cal.select_confirmation_inventory(eval_cases_per_second=0.1), 64)
        self.assertEqual(
            cal.select_confirmation_inventory(eval_cases_per_second=0.05), 32)
        with self.assertRaises(Exception):
            cal.select_confirmation_inventory(eval_cases_per_second=0.001)

    def test_freeze_once_and_hardware_match(self) -> None:
        from bramastra_lab.research.campaigns import calibration as cal

        run = tempfile.mkdtemp()
        protocol = cal.freeze_protocol(
            run,
            selection={"update_targets": {"E1": 300, "E3": 100, "E4": 100},
                       "confirmation_clusters_per_family": 64},
            identities={"source_hash": "s", "data_hash": "d",
                        "device_ids": ["cuda:0", "cuda:1"]})
        self.assertEqual(protocol["calibration_source"], "e0-measured")
        loaded = cal.load_frozen_protocol(run)
        assert loaded is not None
        self.assertEqual(loaded["selection"]["update_targets"]["E1"], 300)
        cal.check_hardware_match(loaded, device_ids=["cuda:1", "cuda:0"])
        with self.assertRaises(Exception):
            cal.check_hardware_match(loaded, device_ids=["cuda:0"])
        with self.assertRaises(Exception):
            cal.freeze_protocol(
                run,
                selection={"update_targets": {"E1": 1},
                           "confirmation_clusters_per_family": 32},
                identities={"source_hash": "s", "data_hash": "d",
                            "device_ids": ["cuda:0", "cuda:1"]})
        self.assertIsNone(cal.load_frozen_protocol(tempfile.mkdtemp()))


    def test_freeze_from_e0_samples(self) -> None:
        from bramastra_lab.research.campaigns import calibration as cal
        from bramastra_lab.research.campaigns import runner
        from bramastra_lab.research.campaigns.supervisor import CampaignLedger

        run = tempfile.mkdtemp()
        ledger = CampaignLedger(run)
        ledger.record_allocation("alloc-cal", "src-cal", "data-cal", 480.0)
        results = {"E0-w0": {"calibration_samples": {
            "worst_update_seconds": 2.0, "eval_cases_per_second": 10.0}},
            "E0-w1": {"calibration_samples": {
                "worst_update_seconds": 3.0, "eval_cases_per_second": 8.0}}}
        pending = [{"job_id": "E0-w0"}, {"job_id": "E0-w1"}]
        runner._maybe_freeze_protocol(
            ledger, run, results, pending, source_hash="src-cal",
            data_hash="data-cal", devices=["cuda:0", "cuda:1"])
        ledger.close()
        frozen = cal.load_frozen_protocol(run)
        assert frozen is not None
        # Worst worker governs: floor(0.75*3600/3.0) = 900 for E1.
        self.assertEqual(
            frozen["selection"]["update_targets"]["E1"], 900)
        self.assertEqual(
            frozen["selection"]["confirmation_clusters_per_family"], 128)
        self.assertIsNotNone(runner._load_usable_protocol(
            run, ["cuda:0", "cuda:1"]))
        self.assertIsNone(runner._load_usable_protocol(run, ["cuda:0"]))
        # Second freeze is a no-op (resume consumes the existing artifact;
        # freeze_protocol itself refuses overwrites, tested in O03).
        before = open(os.path.join(run, "frozen_protocol.json")).read()
        ledger2 = CampaignLedger(run)
        runner._maybe_freeze_protocol(
            ledger2, run, results, pending, source_hash="src-cal",
            data_hash="data-cal", devices=["cuda:0", "cuda:1"])
        ledger2.close()
        after = open(os.path.join(run, "frozen_protocol.json")).read()
        self.assertEqual(before, after)

    def test_freeze_refuses_missing_samples(self) -> None:
        from bramastra_lab.research.campaigns import runner
        from bramastra_lab.research.campaigns.supervisor import CampaignLedger

        run = tempfile.mkdtemp()
        ledger = CampaignLedger(run)
        ledger.record_allocation("alloc-cal2", "src", "data", 480.0)
        with self.assertRaises(Exception):
            runner._maybe_freeze_protocol(
                ledger, run, {"E0-w0": {}}, [{"job_id": "E0-w0"}],
                source_hash="src", data_hash="data",
                devices=["cuda:0", "cuda:1"])
        ledger.close()


class O04KernelTests(unittest.TestCase):
    def _mechanism(self, family="inventory", index=0, seed=5):
        from bramastra_lab.research.environments.k8_live import (
            generate_live_mechanism)

        return generate_live_mechanism(family, index, seed=seed)

    def test_live_episode_trace_integrity(self) -> None:
        from bramastra_lab.research.cognition import episode as kernel
        from bramastra_lab.research.environments.k8_live import build_live_env

        mechanism = self._mechanism("inventory")
        env = build_live_env(mechanism, budget=6, seed=5)
        script = [{"kind": "check_dependency", "item": q["item"]}
                  for q in mechanism["queries"][:2]]
        script.append({"kind": "submit",
                       "item": mechanism["dependency_item"]})
        out = kernel.run_episode(
            env, kernel.LearnedPolicyAdapter(),
            model=kernel.ScriptedModel(script), seed=5,
            mechanism=mechanism, session_job_id="o04-test",
            checkpoint_id="ckpt-test")
        summary = out["summary"]
        self.assertTrue(summary["terminated"])
        self.assertTrue(summary["success"])
        events = out["events"]
        for position, event in enumerate(events):
            self.assertEqual(event["event_index"], position)
            self.assertEqual(event["session_job_id"], "o04-test")
            self.assertEqual(event["checkpoint_id"], "ckpt-test")
            if position:
                self.assertEqual(event["predecessor"], position - 1)
        self.assertFalse(out["imagined"])
        # Counters reduce from emitted events (never constants).
        action_events = [event for event in events
                         if event["kind"] == "action"]
        self.assertEqual(summary["actions"], len(action_events))
        model_calls = sum(1 for event in events
                          if event.get("generation_id") is not None)
        self.assertEqual(summary["model_calls"], model_calls)
        self.assertEqual(
            summary["cost"],
            sum(event.get("resource_delta", 0.0) for event in events))

    def test_planner_imagined_stays_separate(self) -> None:
        from bramastra_lab.research.cognition import episode as kernel
        from bramastra_lab.research.environments.k8_live import build_live_env

        mechanism = self._mechanism("inventory")
        env = build_live_env(mechanism, budget=6, seed=5)
        predictions = [{"feedback": {"kind": "guess"}, "success_prob": 0.9,
                        "value": 0.1}]
        planner = kernel.BoundedPlannerAdapter(
            world_model=kernel.CannedWorldModel(predictions))
        out = kernel.run_episode(
            env, planner, model=kernel.ScriptedModel([{"kind": "noop"}]),
            seed=5, mechanism=mechanism, session_job_id="o04-plan",
            checkpoint_id=None)
        self.assertTrue(out["imagined"])
        for node in out["imagined"]:
            self.assertIn(node["provenance"],
                          ("planner-depth1", "planner-depth2"))
        history_text = json.dumps(out["history"])
        self.assertNotIn("guess", history_text)

    def test_invalid_action_becomes_costed_outcome(self) -> None:
        from bramastra_lab.research.cognition import episode as kernel
        from bramastra_lab.research.environments.k8_live import build_live_env

        mechanism = self._mechanism("inventory")
        env = build_live_env(mechanism, budget=6, seed=5)
        script = [{"kind": "fly_to_the_moon"},
                  {"kind": "submit",
                   "item": mechanism["dependency_item"]}]
        out = kernel.run_episode(
            env, kernel.LearnedPolicyAdapter(),
            model=kernel.ScriptedModel(script), seed=5,
            mechanism=mechanism, session_job_id="o04-invalid",
            checkpoint_id=None)
        self.assertEqual(out["summary"]["invalid_actions"], 1)
        kinds = [event.get("observed_result", {}).get("kind")
                 for event in out["events"]]
        self.assertIn("invalid_action", kinds)

    def test_truncation_is_explicit(self) -> None:
        from bramastra_lab.research.cognition import episode as kernel
        from bramastra_lab.research.environments.k8_live import build_live_env

        mechanism = self._mechanism("inventory")
        env = build_live_env(mechanism, budget=6, seed=5)
        out = kernel.run_episode(
            env, kernel.FixedInquiryAdapter(),
            model=kernel.ScriptedModel([{"kind": "noop"}]), seed=5,
            mechanism=mechanism, session_job_id="o04-trunc",
            checkpoint_id=None, action_budget=1)
        reasons = [event.get("observed_result", {}).get("reason")
                   for event in out["events"]
                   if event["kind"] == "truncation"]
        self.assertTrue(reasons)
        self.assertTrue(out["summary"]["truncated"])

    def test_negative_controls_reflect(self) -> None:
        from bramastra_lab.research.cognition import episode as kernel
        from bramastra_lab.research.environments.k8_live import build_live_env

        mechanism = self._mechanism("inventory")
        # Constant value function is recorded on the planner choice path.
        planner = kernel.BoundedPlannerAdapter(
            world_model=kernel.CannedWorldModel(
                [{"feedback": {}, "success_prob": 0.5, "value": 0.0}]),
            value_fn=lambda **kwargs: 1.0, value_fn_name="constant-1.0")
        env = build_live_env(mechanism, budget=6, seed=5)
        out = kernel.run_episode(
            env, planner, model=kernel.ScriptedModel([{"kind": "noop"}]),
            seed=5, mechanism=mechanism, session_job_id="o04-neg",
            checkpoint_id=None)
        metas = [event.get("planner_meta") for event in out["events"]
                 if event.get("planner_meta")]
        self.assertTrue(metas)
        self.assertTrue(all(meta.get("_value_fn") == "constant-1.0"
                            for meta in metas))
        # Reordered execution with seeded adapters gives identical totals.
        def _totals():
            environment = build_live_env(mechanism, budget=6, seed=5)
            result = kernel.run_episode(
                environment, kernel.FixedInquiryAdapter(),
                model=kernel.ScriptedModel([{"kind": "noop"}]), seed=5,
                mechanism=mechanism, session_job_id="o04-re",
                checkpoint_id=None)
            return result["summary"]

        first, second = _totals(), _totals()
        self.assertEqual(first, second)


class O08TrialTests(unittest.TestCase):
    def test_trial_request_validation(self) -> None:
        from bramastra_lab.research.campaigns import trial_service

        def _request(**overrides):
            base = {
                "anchor_checkpoint_id": "deadbeef" * 8,
                "anchor_run_dir": "/none", "method_id": "M1",
                "compiled_recipe": {"identity": "abc"},
                "support_identities": ("s0",), "query_identities": ("q0",),
                "protected_identities": (), "seed": 3,
                "reservation": {"allocation_id": "a", "reservation_id": "r",
                                "job_id": "j", "device": "cpu", "phase": "E5",
                                "deadline_unix": 9e9, "remaining_updates": 4,
                                "source_hash": "s"},
                "deadline_unix": 9e9, "max_updates": 4,
                "task_identity": "t"}
            base.update(overrides)
            return trial_service.TrialRequest(**base)

        _request().validate()
        with self.assertRaises(Exception):
            _request(method_id="M9").validate()
        with self.assertRaises(Exception):
            _request(compiled_recipe={}).validate()
        with self.assertRaises(Exception):
            _request(deadline_unix=1.0).validate()

    def test_production_noop_boundary_without_ledger(self) -> None:
        from bramastra_lab.research.campaigns import trial_service
        from bramastra_lab.research.campaigns.phases.session import (
            ledger_stepping_allowed)

        run = tempfile.mkdtemp()
        self.assertFalse(ledger_stepping_allowed(
            run, {"allocation_id": "nope", "job_id": "nope"}))

    def test_production_trial_reaches_boundary_locally(self) -> None:
        from bramastra_lab.research.campaigns import trial_service

        _, trainer = _tiny_trainer(require_allocation=True)
        handle = {"trainer": trainer, "model": trainer.model,
                  "config": trainer.config, "device": "cpu", "seed": 3}
        request = trial_service.TrialRequest(
            anchor_checkpoint_id="ab" * 32, anchor_run_dir=tempfile.mkdtemp(),
            method_id="M0", compiled_recipe={"identity": "abc"},
            support_identities=("s0",), query_identities=("q0",),
            protected_identities=(), seed=3,
            reservation={"allocation_id": "alloc", "reservation_id": "res",
                         "job_id": "E5-w0", "device": "cpu", "phase": "E5",
                         "deadline_unix": 9e9, "remaining_updates": 4,
                         "source_hash": "src-test"},
            deadline_unix=9e9, max_updates=2, task_identity="t")

        class _Ops:
            def restore_parent(self, *, parent, device, optimizer_policy):
                return dict(handle)

            def fork_child(self, *, parent_handle, optimizer_policy):
                import copy

                return copy.deepcopy(parent_handle)

            def optimizer_updates(self, handle):
                return int(handle["trainer"].counters.optimizer_updates)

        from bramastra_lab.research.campaigns.phases.types import ParentRef

        real_resolve = ParentRef.resolve

        def _fake_resolve(self, run_dir):
            return {"checkpoint_id": request.anchor_checkpoint_id,
                    "manifest": {}, "payload_sha256": "x",
                    "config_identity": "y", "lineage": None,
                    "run_dir": run_dir}

        ParentRef.resolve = _fake_resolve
        try:
            from bramastra_lab.research.metalearning.dispatch import (
                _METHOD_PROGRAMS)
            from bramastra_lab.research.metalearning.method_language import (
                compile_method)
            compiled = compile_method(
                _METHOD_PROGRAMS["M0"],
                runtime_config={"profile": "k8-campaign"})

            def _build(t):
                from bramastra_lab.research.experience.supervision import (
                    SupervisionWindow)

                batch = _tiny_batch()
                window = SupervisionWindow(
                    weights={"token": 1.0, "world": 0.0, "action": 0.0,
                             "value": 0.0, "pair": 0.0, "pg": 0.0},
                    enabled_terms=frozenset({"token"}))
                window.add("token", batch.target_count)
                return batch, window, {}, None

            def _bufferset(method_id, comp, trainer_obj, *, task_identity):
                return "applied"

            import bramastra_lab.research.campaigns.trial_service as ts

            real_dispatch = ts.dispatch_method_to_trainer
            ts.dispatch_method_to_trainer = _bufferset
            try:
                job = _fake_job(job_id="E5-w0", phase="E5",
                                physical_device="cpu",
                                run_dir=tempfile.mkdtemp(),
                                data_dir=tempfile.mkdtemp())
                result = trial_service.run_trial(
                    _Ops(), request, job=job,
                    learning_boundary="production",
                    build_support_batch=_build,
                    evaluate_queries=lambda h: {"measured_success": 0.5})
            finally:
                ts.dispatch_method_to_trainer = real_dispatch
        finally:
            ParentRef.resolve = real_resolve
        # No ledger allocation locally: boundary-only record, zero steps.
        self.assertEqual(result.validation, "boundary-only")
        self.assertEqual(trainer.counters.optimizer_updates, 0)


class O09ProposerTests(unittest.TestCase):
    def test_proposer_batches_from_measured_bests(self) -> None:
        from bramastra_lab.research.campaigns.phases.e5 import (
            _proposer_batches)
        from bramastra_lab.research.metalearning.dispatch import (
            MethodArchive, MethodTrialOutcome)

        outcomes = tuple(
            MethodTrialOutcome(method_id=method, task_identity="mt-0",
                               measured_updates=1, measured_success=success,
                               elapsed_seconds=45.0, validation="measured")
            for method, success in (("M0", 0.5), ("M1", 0.8), ("M2", 0.3)))
        archive = MethodArchive(rows=outcomes, cutoff_event_index=3)
        batches = _proposer_batches([{"meta_task_id": "mt-0",
                                      "family": "rule-inquiry"}], archive)
        self.assertEqual(len(batches), 1)
        self.assertGreater(batches[0].target_count, 0)

    def test_teacher_fallback_labeled_with_doubles(self) -> None:
        from bramastra_lab.research.campaigns.phases.e5 import (
            _capture_proposer_choice)
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)
        from bramastra_lab.research.metalearning.dispatch import (
            MethodArchive, MethodTrialOutcome)

        outcomes = tuple(
            MethodTrialOutcome(method_id=method, task_identity="mt-0",
                               measured_updates=1, measured_success=success,
                               elapsed_seconds=45.0, validation="measured")
            for method, success in (("M0", 0.5), ("M1", 0.8)))
        archive = MethodArchive(rows=outcomes, cutoff_event_index=2)
        ops = RecordingDoubleOps()
        proposer = ops.init_model(seed=1, profile="k8-campaign",
                                  device="cpu")
        choice, capture = _capture_proposer_choice(
            ops, proposer, archive, [{"meta_task_id": "mt-0",
                                      "family": "rule-inquiry"}],
            "anchor-test", _fake_job())
        self.assertEqual(choice, "M1")
        self.assertEqual(capture["capture_origin"],
                         "teacher-majority-control")

    def test_confirmation_attribution_rules(self) -> None:
        from bramastra_lab.research.campaigns.phases.e5 import (
            _summarize_confirmation)

        supported = _summarize_confirmation([
            {"policy": "P1", "measured_success": 0.9},
            {"policy": "P0", "measured_success": 0.5},
            {"policy": "P_fixed", "measured_success": 0.5}])
        self.assertEqual(supported["attribution"],
                         "recursive-benefit-supported")
        extra_only = _summarize_confirmation([
            {"policy": "P1", "measured_success": 0.9},
            {"policy": "P0", "measured_success": 0.5},
            {"policy": "P_fixed", "measured_success": 0.9}])
        self.assertEqual(extra_only["attribution"], "extra-training-only")
        negative = _summarize_confirmation([
            {"policy": "P1", "measured_success": 0.4},
            {"policy": "P0", "measured_success": 0.5},
            {"policy": "P_fixed", "measured_success": 0.5}])
        self.assertEqual(negative["attribution"],
                         "negative-or-inconclusive")


class O05MatchedEvaluationTests(unittest.TestCase):
    def _ledger_with_parents(self, run):
        from bramastra_lab.research.campaigns.supervisor import (
            CampaignLedger)
        from bramastra_lab.research.runtime import checkpoint as ckpt

        ckpts = {}
        for index, arm in enumerate(("A", "B")):
            manifest = ckpt.save_checkpoint(
                run, {"model": {"w": torch.zeros(2)},
                      "counters": {"optimizer_updates": index}},
                run_id=f"E1-{arm}-1701", update_index=index,
                config_identity="cfg", tokenizer_identity="tok",
                data_identity="data", parent_checkpoint_id=None,
                code_identity="code")
            ckpts[arm] = manifest.checkpoint_id
        ledger = CampaignLedger(run)
        ledger.record_allocation("alloc-o05", "src", "data", 480.0)
        for arm in ("A", "B"):
            reservation = ledger.reserve(
                f"E1-{arm}-1701", worker="w", device="cuda:0", phase="E1",
                arm=arm, seed=1701, reserved_seconds=60.0)
            ledger.close_reservation(
                reservation.reservation_id, status="completed",
                committed_updates=1, attempted_updates=1,
                supervised_exposure=1, device_seconds=1.0,
                checkpoint_identity=ckpts[arm])
        ledger.close()
        return ckpts

    def test_matched_live_evaluation_with_doubles(self) -> None:
        import json as _json

        from bramastra_lab.research.campaigns.phases import e2
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)
        from bramastra_lab.research.campaigns.phases.types import JobInput

        data_dir = tempfile.mkdtemp()
        open(os.path.join(data_dir, "manifest.json"), "w").write("{}")
        run = tempfile.mkdtemp()
        ckpts = self._ledger_with_parents(run)
        job = JobInput(phase="E2", slot=0, arm=None, seed=1701,
                       parent="E1-B-1701/E1-A-1701",
                       physical_device="cpu", local_device="cpu",
                       data_dir=data_dir, run_dir=run, precision="fp32",
                       deadline=9e9)
        res = e2.execute(job, ops=RecordingDoubleOps(), eval_cases=1)
        self.assertEqual(res.status, "completed")
        self.assertEqual(res.evidence_kind, "fixture")
        artifact = _json.load(open(os.path.join(
            run, "phase_outputs", "E2", "E2-1701.json")))
        self.assertEqual(artifact["matched_groups"], 1)
        self.assertEqual(artifact["mechanism_source"], "k8-live-eval/v1")
        for mode in ("b-policy", "b-workspace", "b-planner", "a-direct",
                     "a-fixed", "a-random", "symbolic"):
            self.assertIn(mode, artifact["mode_stats"])
        # B modes bind the B checkpoint; controls bind A (or none).
        self.assertEqual(artifact["parent_checkpoint"], ckpts["B"])
        self.assertEqual(len(artifact["paired_deltas"]), 1)
        self.assertEqual(len(artifact["goal_swap"]), 1)
        self.assertEqual(len(artifact["contradiction"]), 1)
        self.assertEqual(len(artifact["complementary"]), 1)

    def test_missing_b_arm_fails(self) -> None:
        from bramastra_lab.research.campaigns.phases import e2
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)
        from bramastra_lab.research.campaigns.phases.types import JobInput
        from bramastra_lab.research.campaigns.supervisor import (
            CampaignLedger)
        from bramastra_lab.research.runtime import checkpoint as ckpt

        data_dir = tempfile.mkdtemp()
        open(os.path.join(data_dir, "manifest.json"), "w").write("{}")
        run = tempfile.mkdtemp()
        manifest = ckpt.save_checkpoint(
            run, {"model": {"w": torch.zeros(2)},
                  "counters": {"optimizer_updates": 0}},
            run_id="E1-A-1701", update_index=0,
            config_identity="cfg", tokenizer_identity="tok",
            data_identity="data", parent_checkpoint_id=None,
            code_identity="code")
        ledger = CampaignLedger(run)
        ledger.record_allocation("alloc-o05b", "src", "data", 480.0)
        reservation = ledger.reserve(
            "E1-A-1701", worker="w", device="cuda:0", phase="E1", arm="A",
            seed=1701, reserved_seconds=60.0)
        ledger.close_reservation(
            reservation.reservation_id, status="completed",
            committed_updates=1, attempted_updates=1, supervised_exposure=1,
            device_seconds=1.0,
            checkpoint_identity=manifest.checkpoint_id)
        ledger.close()
        job = JobInput(phase="E2", slot=0, arm=None, seed=1701,
                       parent="E1-A-1701",
                       physical_device="cpu", local_device="cpu",
                       data_dir=data_dir, run_dir=run, precision="fp32",
                       deadline=9e9)
        res = e2.execute(job, ops=RecordingDoubleOps(), eval_cases=1)
        self.assertEqual(res.status, "failed")
        self.assertIn("B arm", res.error or "")


class O06CanonicalTests(unittest.TestCase):
    def test_protected_canonical_sets(self) -> None:
        import json as _json

        from bramastra_lab.research.campaigns.phases.e3 import (
            _load_protected_canonical)

        data_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(data_dir, "tools"), exist_ok=True)
        with open(os.path.join(data_dir, "splits.json"), "w") as handle:
            _json.dump({"families": {"rule-inquiry": {
                "sealed-confirmation": {
                    "canonical_identities": ["canon-sealed-1"]}}}}, handle)
        with open(os.path.join(data_dir, "tools", "tool_tasks.jsonl"),
                  "w") as handle:
            handle.write(_json.dumps(
                {"split": "tool-heldout",
                 "mechanism_id": "tool-held-1"}) + "\n")
            handle.write(_json.dumps(
                {"split": "tool-training", "composition": "single_filter",
                 "mechanism_id": "tool-train-1", "answer": "3",
                 "expected_sum": 3, "public": {}, "predicate": {},
                 "table": {"columns": [], "rows": []}}) + "\n")
        protected, heldout = _load_protected_canonical(data_dir)
        self.assertIn("canon-sealed-1", protected)
        self.assertIn("tool-held-1", heldout)

    def test_corrupt_tool_row_refused(self) -> None:
        from bramastra_lab.research.campaigns.phases.e3 import _tool_batch

        row = {"mechanism_id": "tool-x", "answer": "10", "expected_sum": 10,
               "composition": "single_filter", "public": {},
               "predicate": {"column": "category", "equals": "b"},
               "table": {"columns": ["id", "category", "value"],
                         "rows": [{"id": 0, "category": "b", "value": 3}]},
               "split": "tool-training"}
        with self.assertRaises(ValueError):
            _tool_batch(row, {"token": 1.0}, frozenset({"token"}))


class O07GateTests(unittest.TestCase):
    def test_fixture_migration_records(self) -> None:
        from bramastra_lab.research.campaigns.phases.e4 import (
            _verify_handle_migration)
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)

        ops = RecordingDoubleOps()
        handle = ops.init_model(seed=1, profile="k8-campaign", device="cpu")
        migrated = ops.migrate_to_gated(handle, gates_enabled=True)
        self.assertIsNone(
            _verify_handle_migration(migrated, gates_enabled=True))
        fresh = ops.init_model(seed=1, profile="k8-campaign", device="cpu")
        self.assertIsNotNone(
            _verify_handle_migration(fresh, gates_enabled=True))

    def test_arch_proof_fixture_path(self) -> None:
        from bramastra_lab.research.campaigns.phases.e4 import (
            _prove_architecture_on_handle)
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)

        ops = RecordingDoubleOps()
        handle = ops.init_model(seed=1, profile="k8-campaign", device="cpu")
        proof = _prove_architecture_on_handle(
            ops, handle, gates_enabled=True)
        self.assertIn("gate_gradients", proof)


class O10NotebookTests(unittest.TestCase):
    def test_campaign_defaults_to_fp32_until_amp_is_calibrated(self) -> None:
        from bramastra_lab.research.campaigns.k8 import build_parser
        from bramastra_lab.research.campaigns.runner import run_campaign
        from bramastra_lab.research.campaigns.phases.ops import (
            K8_CAMPAIGN_PRECISION, ProductionOps)

        parsed = build_parser().parse_args(
            ["run", "--mode", "e0", "--run-dir", "run", "--data", "data"])
        self.assertEqual(parsed.precision, "fp32")
        self.assertEqual(run_campaign.__kwdefaults__["precision"], "fp32")
        self.assertEqual(K8_CAMPAIGN_PRECISION, "fp32")
        self.assertEqual(ProductionOps().precision, "fp32")

    def test_notebook_backed_by_repo(self) -> None:
        notebook = json.load(open("notebooks/bramastra_k8.ipynb"))
        sources = ["".join(cell["source"]) for cell in notebook["cells"]
                   if cell["cell_type"] == "code"]
        joined = "\n".join(sources)
        for command in ("prepare", "validate", "run", "summarize", "export"):
            self.assertIn(f"'{command}'", joined)
        self.assertIn("source_identity", joined)
        self.assertIn("E0", joined)
        # Kaggle mounts operator-supplied source/data datasets below input;
        # the notebook must never rely on an unstated checkout in working.
        self.assertIn("/kaggle/input", joined)
        self.assertIn("_is_source_tree", joined)
        self.assertIn("bramastra-k8-data/v1", joined)
        self.assertIn("BRAMASTRA_BUNDLE_DIR", joined)
        self.assertIn("BRAMASTRA_GIT_URL", joined)
        self.assertIn("git', 'clone'", joined)
        self.assertIn("generated-k8-data", joined)
        self.assertIn("REQUIRED_MODULES", joined)
        self.assertIn("run_k8", joined)
        self.assertIn("'--precision', 'fp32'", joined)
        self.assertNotIn("'--precision', 'fp16_autocast'", joined)
        self.assertIn("package_run_artifacts", joined)
        self.assertIn("shutil.make_archive", joined)
        self.assertIn("FileLink", joined)
        self.assertNotIn("shutil.rmtree", joined)
        # Every mutating subprocess call checks its failure.
        self.assertGreaterEqual(joined.count("returncode"), 2)
        self.assertIn("raise RuntimeError", joined)
        # E0 and full share one RUN_DIR/allocation (no E0 cost reset).
        run_dir_uses = joined.count("RUN_DIR")
        self.assertGreaterEqual(run_dir_uses, 3)

    def test_notebook_code_cells_parse(self) -> None:
        import ast

        notebook = json.load(open("notebooks/bramastra_k8.ipynb"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                with self.subTest(cell=index):
                    ast.parse(source)

    def test_notebook_artifact_packager_writes_verified_zip_and_receipt(self) -> None:
        import ast
        from pathlib import Path
        import zipfile

        notebook = json.load(open("notebooks/bramastra_k8.ipynb"))
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and
            "def package_run_artifacts" in "".join(cell["source"]))
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and
            node.name == "package_run_artifacts")
        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp)
            run_root = working / "bramastra-k8" / "run-test"
            run_root.mkdir(parents=True)
            (run_root / "campaign_ledger.sqlite").write_text("ledger")
            (run_root / "nested").mkdir()
            (run_root / "nested" / "result.json").write_text("result")
            namespace = {"RUN_ROOT": run_root, "WORKING": working,
                         "RUN_ID": "run-test", "INSTANCE_ID": "instance",
                         "Path": Path, "json": json}
            exec(compile(ast.Module(body=[function], type_ignores=[]),
                         "notebook-packager", "exec"), namespace)
            archive = namespace["package_run_artifacts"]("test")
            self.assertTrue(archive.is_file())
            receipt = json.loads(archive.with_suffix(".json").read_text())
            self.assertEqual(receipt["reason"], "test")
            self.assertGreater(receipt["bytes"], 0)
            with zipfile.ZipFile(archive) as bundle:
                self.assertIsNone(bundle.testzip())
                self.assertTrue(any(name.endswith("campaign_ledger.sqlite")
                                    for name in bundle.namelist()))

    def test_export_reload_roundtrip_fresh_process(self) -> None:
        import subprocess
        import sys

        from bramastra_lab.research.campaigns import k8 as k8_module
        from bramastra_lab.research.campaigns.supervisor import CampaignLedger
        from bramastra_lab.research.runtime import checkpoint as ckpt

        run = tempfile.mkdtemp()
        out = tempfile.mkdtemp()
        payload = {"model": {"w": torch.zeros(4)},
                   "counters": {"optimizer_updates": 0}}
        token = ckpt.acquire_writer_fence(run)
        try:
            manifest = ckpt.save_checkpoint(
                run, payload, run_id="o10-test", update_index=0,
                config_identity="cfg-o10", tokenizer_identity="tok-o10",
                data_identity="data-o10", parent_checkpoint_id=None,
                code_identity="code-o10", writer_token=token)
        finally:
            ckpt.release_writer_fence(run, token)
        ledger = CampaignLedger(run)
        ledger.record_allocation("alloc-o10", "src-o10", "data-o10", 480.0)
        reservation = ledger.reserve(
            "E1-B-1701", worker="w", device="cuda:1", phase="E1", arm="B",
            seed=1701, reserved_seconds=60.0)
        ledger.close_reservation(
            reservation.reservation_id, status="completed",
            committed_updates=1, attempted_updates=1, supervised_exposure=1,
            device_seconds=1.0,
            checkpoint_identity=manifest.checkpoint_id)
        ledger.close()
        import argparse

        code = k8_module.cmd_export(
            argparse.Namespace(run_dir=run, out=out))
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(
            os.path.join(out, "artifact_manifest.json")))
        check = (
            "from bramastra_lab.research.runtime.checkpoint import "
            "load_checkpoint;"
            f"payload, manifest = load_checkpoint({run!r}, "
            f"checkpoint_id={manifest.checkpoint_id!r});"
            "assert manifest.checkpoint_id == "
            f"{manifest.checkpoint_id!r}; print('fresh-reload-ok')")
        completed = subprocess.run(
            [sys.executable, "-c", check], capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": "."})
        self.assertIn("fresh-reload-ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
