"""Focused checks for Gandiva's E5 method and cognition execution seams."""
from __future__ import annotations

import json
import os
import tempfile
import unittest


class E5MethodExecutionTests(unittest.TestCase):
    def test_compiled_objectives_reach_support_windows(self) -> None:
        from bramastra_lab.research.campaigns.phases.e5 import (
            _production_support_builder)
        from bramastra_lab.research.metalearning.dispatch import (
            _METHOD_PROGRAMS, compile_method, dispatch_method_to_trainer,
            parse_method_selection)

        task = {"support_examples": [{
            "public": {"question": "Which rule is active?"},
            "answer": "the first rule", "mechanism_id": "train-1",
            "family": "rule-inquiry"}]}
        windows = {}
        compiled = {}
        for method in ("M0", "M1", "M2"):
            compiled[method] = compile_method(
                _METHOD_PROGRAMS[method], runtime_config={"profile": "tiny"})
            _batch, window, _extra, _pairs = _production_support_builder(
                task, compiled[method])(None)
            windows[method] = window
            self.assertEqual(window.enabled_terms, frozenset({"token"}))
            self.assertEqual(window.denominator("token"), _batch.target_count)
            self.assertEqual(window.denominator("world"), 0)
            self.assertEqual(window.denominator("action"), 0)

        self.assertEqual(windows["M0"].weights["token"], 1.0)
        self.assertEqual(windows["M1"].weights["token"], 0.8)
        self.assertEqual(windows["M2"].weights["token"], 1.0)
        self.assertEqual(parse_method_selection("M2")[0], "M2")
        self.assertEqual(compiled["M2"]["gradient_transform"]["bound"], 0.5)

        class Trainer:
            def __init__(self, *, gates_enabled=False):
                from types import SimpleNamespace
                self.optimizer = SimpleNamespace(param_groups=[{"lr": 0.001}])
                self.model = SimpleNamespace(gates_enabled=gates_enabled)
                self.clip_norm = 1.0

            def set_controller_multiplier(self, _multiplier, _reason):
                pass

        m1 = Trainer()
        dispatch_method_to_trainer("M1", compiled["M1"], m1,
                                   task_identity="tiny-m1")
        self.assertEqual(m1.optimizer.param_groups[0]["lr"], 0.0005)
        m2 = Trainer(gates_enabled=True)
        dispatch_method_to_trainer("M2", compiled["M2"], m2,
                                   task_identity="tiny-m2")
        self.assertEqual(m2.clip_norm, 0.5)

    def test_model_origin_capture_requires_exact_restorable_payload(self) -> None:
        from bramastra_lab.research.campaigns.phases.e5 import (
            _capture_proposer_choice, _capture_successor_choice)
        from bramastra_lab.research.campaigns.phases.types import JobInput
        from bramastra_lab.research.metalearning.dispatch import (
            MethodArchive, MethodTrialOutcome, _METHOD_PROGRAMS)
        from bramastra_lab.research.metalearning.generations import (
            ProposalCapture)
        from unittest.mock import patch

        class FakeProposer:
            def __init__(self, _model, _config, *, checkpoint_payload_identity):
                self.checkpoint_id = checkpoint_payload_identity

            def capture_proposal(self, _descriptor, _archive):
                return ProposalCapture(
                    checkpoint_payload_identity=self.checkpoint_id,
                    rendered_input="controlled public archive prompt",
                    sampling={"temperature": 0.0}, raw_output="M0",
                    parsed_program=_METHOD_PROGRAMS["M0"])

        class Ops:
            def restore_verify(self, *, run_dir, checkpoint_id):
                return {"restored_ok": True, "checkpoint_id": checkpoint_id}

        job = JobInput(
            phase="E5", slot=0, arm=None, seed=7, parent="anchor",
            physical_device="cpu", local_device="cpu", data_dir="data",
            run_dir="run", precision="fp32", deadline=9e9)
        archive = MethodArchive(rows=(MethodTrialOutcome(
            method_id="M0", task_identity="mt-0", measured_updates=1,
            measured_success=0.5, elapsed_seconds=1.0),),
            cutoff_event_index=1)
        proposer = {"model": object(), "config": object()}
        with patch("bramastra_lab.research.metalearning.dispatch.MethodProposer",
                   FakeProposer):
            choice, capture = _capture_proposer_choice(
                Ops(), proposer, archive, [{"meta_task_id": "mt-0"}],
                "anchor-id", job, proposer_checkpoint_id="p0-payload")
            self.assertEqual(choice, "M0")
            self.assertEqual(capture["capture_origin"], "model-decoder")
            self.assertEqual(capture["checkpoint_payload_identity"], "p0-payload")

            class ForgedProposer(FakeProposer):
                def capture_proposal(self, _descriptor, _archive):
                    return ProposalCapture(
                        checkpoint_payload_identity="forged-payload",
                        rendered_input="controlled public archive prompt",
                        sampling={"temperature": 0.0}, raw_output="M0",
                        parsed_program=_METHOD_PROGRAMS["M0"])

            with patch(
                    "bramastra_lab.research.metalearning.dispatch.MethodProposer",
                    ForgedProposer):
                _choice, fallback = _capture_proposer_choice(
                    Ops(), proposer, archive, [{"meta_task_id": "mt-0"}],
                    "anchor-id", job, proposer_checkpoint_id="p0-payload")
                self.assertEqual(fallback["capture_origin"],
                                 "teacher-majority-control")
                self.assertIn("differs from published payload",
                              fallback["model_error"])

                with self.assertRaisesRegex(ValueError,
                                            "could not be decoded independently"):
                    _capture_successor_choice(
                        {"model": object(), "config": object()}, archive,
                        [{"meta_task_id": "mt-0"}], "anchor-id", "P1",
                        ops=Ops(), job=job,
                        proposer_checkpoint_id="p1-payload")

    def test_e5_artifact_records_the_actual_proposer_lineage(self) -> None:
        import torch

        from bramastra_lab.research.campaigns.phases import e5
        from bramastra_lab.research.campaigns.phases.ops import RecordingDoubleOps
        from bramastra_lab.research.campaigns.phases.types import JobInput
        from bramastra_lab.research.campaigns.supervisor import CampaignLedger
        from bramastra_lab.research.runtime.checkpoint import save_checkpoint

        with tempfile.TemporaryDirectory() as data_dir, \
                tempfile.TemporaryDirectory() as run_dir:
            os.makedirs(os.path.join(data_dir, "meta"))
            with open(os.path.join(data_dir, "manifest.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({}, handle)
            with open(os.path.join(data_dir, "meta", "meta_tasks.jsonl"),
                      "w", encoding="utf-8") as handle:
                for pool, task_id in (("meta-training", "mt-a"),
                                      ("meta-training", "mt-b"),
                                      ("meta-confirmation", "mc-a")):
                    handle.write(json.dumps({
                        "pool": pool, "meta_task_id": task_id,
                        "family": "rule-inquiry"}) + "\n")
            parent = save_checkpoint(
                run_dir, {"model": {"weight": torch.zeros(1)},
                          "counters": {"optimizer_updates": 0}},
                run_id="E1-B-1701", update_index=0,
                config_identity="cfg", tokenizer_identity="tok",
                data_identity="data", code_identity="code",
                parent_checkpoint_id=None)
            ledger = CampaignLedger(run_dir)
            ledger.record_allocation("alloc-gandiva", "src", "data", 480.0)
            reservation = ledger.reserve(
                "E1-B-1701", worker="fixture", device="cpu", phase="E1",
                arm="B", seed=1701, reserved_seconds=60.0)
            ledger.close_reservation(
                reservation.reservation_id, status="completed",
                committed_updates=0, attempted_updates=0,
                supervised_exposure=0, device_seconds=0.0,
                checkpoint_identity=parent.checkpoint_id)
            ledger.close()
            job = JobInput(
                phase="E5", slot=0, arm=None, seed=1701,
                parent="E1-B-1701", physical_device="cpu",
                local_device="cpu", data_dir=data_dir, run_dir=run_dir,
                precision="fp32", deadline=9e9)
            ops = RecordingDoubleOps()
            result = e5.execute(job, ops=ops, tasks_per_block=1)
            self.assertEqual(result.status, "completed", result.error)

            with open(os.path.join(run_dir, "phase_outputs", "E5",
                                   "E5-1701.json"), encoding="utf-8") as handle:
                artifact = json.load(handle)
            lineage = artifact["checkpoint_lineage"]
            self.assertEqual(lineage["P0"]["parent_checkpoint_id"],
                             parent.checkpoint_id)
            self.assertEqual(lineage["P1"]["parent_checkpoint_id"],
                             lineage["P0"]["checkpoint_id"])
            self.assertEqual(lineage["P_fixed"]["parent_checkpoint_id"],
                             lineage["P0"]["checkpoint_id"])
            self.assertEqual(
                artifact["p0_capture"]["checkpoint_payload_identity"],
                lineage["P0"]["checkpoint_id"])
            for label in ("P1", "P_fixed"):
                self.assertEqual(
                    artifact["successor_choice_captures"][label]
                    ["checkpoint_payload_identity"],
                    lineage[label]["checkpoint_id"])
            chain = artifact["generation_chain"]
            self.assertEqual(chain["registry_namespace"], "fixture")
            self.assertEqual(chain["publication_state"], "fixture-only")
            self.assertEqual(len(chain["receipts"]), 2)
            self.assertEqual(
                chain["receipts"][1]["predecessor_receipt_id"],
                chain["receipts"][0]["receipt_id"])
            self.assertEqual(chain["receipts"][1]["proposer_checkpoint"],
                             lineage["P1"]["checkpoint_id"])
            publications = [details for name, details in ops.calls
                            if name == "publish_checkpoint"]
            parents = {row["arm"]: row["parent_checkpoint_id"]
                       for row in publications}
            self.assertEqual(parents["P0"], parent.checkpoint_id)
            self.assertEqual(parents["P1"], lineage["P0"]["checkpoint_id"])
            self.assertEqual(parents["P_fixed"], lineage["P0"]["checkpoint_id"])


class CognitionMemoryRuntimeTests(unittest.TestCase):
    def test_memory_budget_skips_oversize_ranked_record_and_fills_with_smaller(self) -> None:
        from bramastra_lab.research.experience.codec import encode_event
        from bramastra_lab.research.experience.public_state import (
            compact_memory_content)
        from bramastra_lab.research.memory.store import MemoryIndex, MemoryRecord

        small = MemoryRecord(content="alpha", identity="small",
                             scope="training")
        large = MemoryRecord(content="alpha beta " + "filler " * 80,
                             identity="large", scope="training")
        exact_budget = len(encode_event(
            "observation", compact_memory_content(small.content)))
        context = MemoryIndex((small, large)).retrieve(
            "alpha beta", scope_allowlist={"training"}, top_k=1,
            token_budget=exact_budget)
        self.assertEqual([record.identity for record in context.records], ["small"])
        self.assertEqual(context.token_cost, exact_budget)
        self.assertIn("large", context.omitted_record_ids)

    def test_training_memory_changes_prompt_and_trace_without_sealed_data(self) -> None:
        from types import SimpleNamespace

        from bramastra_lab.research.cognition.episode import (
            Adapter, run_episode)
        from bramastra_lab.research.experience.codec import encode_text
        from bramastra_lab.research.memory.store import MemoryIndex, MemoryRecord

        index = MemoryIndex((
            MemoryRecord(content="training exemplar: emerald answer is silver",
                         identity="training-source-1", scope="training",
                         episode_id="prior-training", kind="training_trajectory"),
            MemoryRecord(content="sealed emerald secret answer is gold",
                         identity="sealed-source-1", scope="sealed",
                         episode_id="heldout", kind="confirmation")))

        class Env:
            def reset(self, *, episode_id):
                return SimpleNamespace(
                    observable_values={"question": "What is the emerald code?"},
                    feedback={"kind": "initial"})

            def legal_actions(self):
                return [{"kind": "submit", "answer": "silver"}]

            def step(self, action):
                return (SimpleNamespace(feedback={"kind": "complete",
                                                  "success": True}),
                        0.25, True, False)

        class CaptureAdapter(Adapter):
            name = "prompt-capture"

            def select(self, *, legal_actions, rendered, **_kwargs):
                self.prompt = list(rendered)
                return {**dict(legal_actions[0]), "_origin": "test-double"}

        adapter = CaptureAdapter()
        trace = run_episode(
            Env(), adapter, model=object(), seed=17,
            mechanism={"mechanism_id": "eval-new"}, memory_index=index,
            memory_token_budget=128)
        prompt = adapter.prompt
        training_tokens = encode_text("training exemplar: emerald answer is silver")
        sealed_tokens = encode_text("sealed emerald secret answer is gold")
        self.assertTrue(any(prompt[i:i + len(training_tokens)] == training_tokens
                            for i in range(len(prompt) - len(training_tokens) + 1)))
        self.assertFalse(any(prompt[i:i + len(sealed_tokens)] == sealed_tokens
                             for i in range(len(prompt) - len(sealed_tokens) + 1)))
        self.assertEqual(trace["summary"]["memory"]["index_identity"],
                         index.identity)
        self.assertEqual(trace["summary"]["memory"]["origin"],
                         "fixed_scope_lexical")
        memory_events = [event for event in trace["events"]
                         if event["kind"] == "memory_retrieval"]
        self.assertEqual(len(memory_events), 1)
        self.assertEqual(memory_events[0]["planner_meta"]["record_count"], 1)
        self.assertNotIn("training-source-1", json.dumps(trace))
        self.assertGreater(trace["summary"]["inference_input_tokens"],
                           trace["summary"]["memory"]["token_cost"])

    def test_e2_memory_arm_uses_only_training_mechanisms(self) -> None:
        from bramastra_lab.research.campaigns.phases import e2
        from bramastra_lab.research.campaigns.phases.ops import RecordingDoubleOps
        from bramastra_lab.research.campaigns.phases.types import JobInput
        from bramastra_lab.research.data.k8_bundle import build_k8_bundle

        with tempfile.TemporaryDirectory() as data_dir, \
                tempfile.TemporaryDirectory() as run_dir:
            build_k8_bundle(
                data_dir, training_mechanisms=1,
                controller_mechanisms=1, development_mechanisms=1,
                confirmation_mechanisms=1, tool_mechanisms=1,
                tool_heldout=1, meta_train=1, meta_validate=1,
                meta_confirm=1)
            job = JobInput(
                phase="E2", slot=0, arm=None, seed=1701,
                parent=None, physical_device="cpu", local_device="cpu",
                data_dir=data_dir, run_dir=run_dir, precision="fp32",
                deadline=9e9)
            memory_index = e2._build_training_memory_index(job)
            self.assertEqual(len(memory_index.records), 3)
            self.assertEqual({record.scope for record in memory_index.records},
                             {"training"})
            ops = RecordingDoubleOps()
            handle = ops.init_model(seed=1701, profile="k8-campaign",
                                    device="cpu")
            bridge = e2._OpsModelBridge(ops, handle)
            groups, _rejected = e2._select_matched_groups(1701, 1)
            episodes = e2._run_matched_group(
                groups[0], b_bridge=bridge, job=job,
                b_checkpoint="fixture-parent", a_checkpoint=None,
                memory_index=memory_index)
            self.assertIn("b-memory", episodes)
            self.assertTrue(episodes["b-memory"]["summary"]
                            ["memory"]["enabled"])
            self.assertEqual(
                episodes["b-memory"]["summary"]["memory"]["index_identity"],
                memory_index.identity)
            self.assertGreater(
                episodes["b-memory"]["summary"]["memory"]["records_read"], 0)
            self.assertFalse(episodes["b-policy"]["summary"]
                             ["memory"]["enabled"])


class GatedMigrationRngTests(unittest.TestCase):
    def test_qualification_keeps_global_rng_mode_and_existing_gradients(self) -> None:
        import torch

        from bramastra_lab.research.config import BuildConfig
        from bramastra_lab.research.models import IntegratedModel
        from bramastra_lab.research.models.gated import (
            check_gate_gradients, migrate_from_parent)

        torch.manual_seed(613)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        parent = IntegratedModel(config)
        parent.train()
        rng_before = torch.random.get_rng_state().clone()
        child = migrate_from_parent(parent, config, gates_enabled=True)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertTrue(parent.training)
        child.decoder.embedding.weight.grad = torch.ones_like(
            child.decoder.embedding.weight)
        old_grad = child.decoder.embedding.weight.grad.clone()
        old_mode = child.training
        rng_before = torch.random.get_rng_state().clone()
        checks = check_gate_gradients(child)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertTrue(child.training == old_mode)
        self.assertTrue(torch.equal(child.decoder.embedding.weight.grad, old_grad))
        self.assertTrue(checks["shared_block_gradients_present"])
        self.assertTrue(checks["embedding_gradients_present"])


if __name__ == "__main__":
    unittest.main()
