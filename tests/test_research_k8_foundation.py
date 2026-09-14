"""Cognition foundation acceptance tests (F1-F6 local criteria).

No optimizer steps, no GPU, no paid compute anywhere in this file:
- real tiny-model backward checks discard gradients (never finalize);
- deterministic doubles stand in for learning with fixture labels;
- GPU-only paths assert their refused/skipped shapes locally.
Each test names the finding it covers; scripted predictions are labeled
mechanics checks, never learned-capability evidence.
"""
import json
import os
import tempfile
import unittest


def _goal(**overrides):
    base = {"question": "is the rule true?"}
    base.update(overrides)
    return base


class F1EncodingTests(unittest.TestCase):
    def test_distinct_variables_render_differently(self) -> None:
        from bramastra_lab.research.cognition.episode import (
            _compact_history_entry, _expand_history_entry)

        left = {"feedback": {"kind": "read", "variable": "x", "value": 7}}
        right = {"feedback": {"kind": "read", "variable": "y", "value": 7}}
        self.assertNotEqual(_compact_history_entry(left),
                            _compact_history_entry(right))
        # Round trip inverts exactly (bijective codec).
        self.assertEqual(_expand_history_entry(_compact_history_entry(left)),
                         {"action": {}, "feedback": dict(left["feedback"])})

    def test_typed_values_stay_distinct(self) -> None:
        from bramastra_lab.research.cognition.episode import (
            _compact_history_entry, _expand_history_entry)

        def _render(value, present=True):
            feedback = {"kind": "read", "variable": "x"}
            if present:
                feedback["value"] = value
            return _compact_history_entry({"feedback": feedback})

        rendered = [_render(7), _render("7"), _render(False),
                    _render(None), _render(0, present=False)]
        self.assertEqual(len({json.dumps(entry, sort_keys=True)
                              for entry in rendered}), 5)
        for entry in rendered:
            self.assertEqual(_expand_history_entry(entry)["feedback"]
                             .get("variable"), "x")

    def test_nested_tool_error_round_trips_full_length(self) -> None:
        from bramastra_lab.research.cognition.episode import (
            _compact_evidence_record)

        payload = {"error": "E" * 500, "request_id": "req-1",
                   "rows": [{"id": i} for i in range(10)]}
        record = {"record_id": "r0", "subject": "tool:write",
                  "predicate": "error", "value": payload, "valid_time": 3,
                  "source_event_id": "e3", "status": "active"}
        compact = _compact_evidence_record(record)
        self.assertEqual(compact["value"], payload)
        self.assertEqual(compact["record_id"], "r0")

    def test_unknown_fields_survive_in_extension_map(self) -> None:
        from bramastra_lab.research.cognition.episode import (
            _compact_history_entry, _expand_history_entry)

        entry = {"action": {"kind": "inspect", "variable": "x",
                            "future_field": "keep-me"},
                 "feedback": {"kind": "read"}}
        compact = _compact_history_entry(entry)
        self.assertEqual(compact["a"]["x"], {"future_field": "keep-me"})
        self.assertEqual(_expand_history_entry(compact)["action"]
                         ["future_field"], "keep-me")

    def test_same_goal_opposite_evidence_prompts_differ(self) -> None:
        from bramastra_lab.research.cognition.episode import (
            render_public_state)

        goal = _goal()
        history_a = [{"action": {"kind": "inspect", "variable": "x"},
                      "feedback": {"kind": "read", "variable": "x",
                                   "value": True}}]
        history_b = [{"action": {"kind": "inspect", "variable": "x"},
                      "feedback": {"kind": "read", "variable": "x",
                                   "value": False}}]
        tokens_a, _ = render_public_state(
            goal=goal, history=history_a, workspace=[], budgets=None)
        tokens_b, _ = render_public_state(
            goal=goal, history=history_b, workspace=[], budgets=None)
        self.assertNotEqual(tokens_a, tokens_b)

    def test_context_exhaustion_reports_omitted_ids(self) -> None:
        from bramastra_lab.research.cognition.episode import (
            admit_observation_evidence, render_public_state)

        workspace: list = []
        for index in range(30):
            admit_observation_evidence(
                workspace,
                observation={"kind": "read", "variable": f"v{index}",
                             "value": True},
                observation_id=f"e{index}")
        tokens, omitted = render_public_state(
            goal=_goal(), history=[], workspace=workspace,
            budgets={"actions_left": 1, "calls_left": 1, "nodes_left": 1},
            max_tokens=512)
        self.assertLessEqual(len(tokens), 512)
        self.assertTrue(omitted)
        # Conflicting records are never omitted: force one and re-render.
        workspace.append({"record_id": "c0", "subject": "variable:v0",
                          "predicate": "value", "value": False,
                          "valid_time": 0, "source_event_id": "probe",
                          "status": "conflicting"})
        _, omitted_again = render_public_state(
            goal=_goal(), history=[], workspace=workspace,
            budgets={"actions_left": 1, "calls_left": 1, "nodes_left": 1},
            max_tokens=512)
        self.assertNotIn("c0", omitted_again)

    def test_train_inference_parity_on_goal_prefix(self) -> None:
        from bramastra_lab.research.cognition.episode import (
            render_public_state)
        from bramastra_lab.research.experience.codec import encode_event
        from bramastra_lab.research.experience.sequences import (
            build_answer_row)

        goal = {"task": "2+2"}
        rendered, _ = render_public_state(
            goal=goal, history=[], workspace=[], budgets=None)
        row = build_answer_row(
            [("goal", dict(goal))], "4",
            provenance={"kind": "trajectory", "episode_id": "parity",
                        "task_semantic_id": "t", "split": "training",
                        "source": "test", "collection_policy": "fixed",
                        "family": "f"},
            max_tokens=64)
        expected_prefix = [259] + encode_event("goal", dict(goal))
        self.assertEqual(list(rendered[:len(expected_prefix)]),
                         expected_prefix)
        self.assertEqual(list(row.tokens[:len(expected_prefix)]),
                         expected_prefix)

    def test_action_codes_fit_envelope_and_full_json_often_does_not(self) -> None:
        from bramastra_lab.research.cognition.episode import (
            RESPONSE_ENVELOPE_TOKENS, decode_action_code,
            encode_action_code)
        from bramastra_lab.research.environments.k8_live import (
            build_live_env, generate_live_mechanism)

        worst_full = 0
        worst_code = 0
        for family in ("rule-inquiry", "inventory", "program", "tools"):
            mechanism = generate_live_mechanism(family, 0, seed=99)
            legal = build_live_env(mechanism, budget=6,
                                   seed=99).legal_actions()
            for action in legal:
                full = len(json.dumps(action, sort_keys=True))
                worst_full = max(worst_full, full)
                code = encode_action_code(action, legal)
                worst_code = max(worst_code, len(code))
                decoded, encoding = decode_action_code(
                    json.loads(code), legal)
                self.assertEqual(encoding, "code")
                for key, value in action.items():
                    self.assertEqual(decoded.get(key), value)
        self.assertLessEqual(worst_code, RESPONSE_ENVELOPE_TOKENS)
        self.assertGreater(worst_full, RESPONSE_ENVELOPE_TOKENS)
        # Stale and out-of-range indexes refuse (never resolve wrongly).
        mechanism = generate_live_mechanism("rule-inquiry", 0, seed=99)
        legal = build_live_env(mechanism, budget=6, seed=99).legal_actions()
        with self.assertRaises(Exception):
            decode_action_code({"a": len(legal)}, legal)
        with self.assertRaises(Exception):
            decode_action_code({"a": 0, "bogus": 1}, legal)
        short_legal = legal[:2]
        with self.assertRaises(Exception):
            decode_action_code({"a": len(short_legal)}, short_legal)


class F2WorkspaceTests(unittest.TestCase):
    def test_different_subjects_coexist(self) -> None:
        from bramastra_lab.research.cognition.episode import (
            admit_observation_evidence, mark_conflicts)

        workspace: list = []
        admit_observation_evidence(
            workspace, observation={"kind": "read", "variable": "x",
                                    "value": 7}, observation_id="a")
        admit_observation_evidence(
            workspace, observation={"kind": "read", "variable": "y",
                                    "value": 8}, observation_id="b")
        self.assertEqual(mark_conflicts(workspace), [])
        self.assertTrue(all(record["status"] == "active"
                            for record in workspace))

    def test_transition_supersedes_without_conflict(self) -> None:
        from bramastra_lab.research.cognition.episode import mark_conflicts

        workspace = [
            {"record_id": "t0", "subject": "door", "predicate": "state",
             "value": "closed", "valid_time": 0, "source_event_id": "t0",
             "status": "active"},
            {"record_id": "t1", "subject": "door", "predicate": "state",
             "value": "open", "valid_time": 1, "source_event_id": "t1",
             "status": "active"}]
        self.assertEqual(mark_conflicts(workspace), [])
        by_id = {record["record_id"]: record for record in workspace}
        self.assertEqual(by_id["t0"]["status"], "superseded")
        self.assertEqual(by_id["t1"]["status"], "active")
        self.assertEqual(by_id["t1"]["supersedes"], "t0")
        self.assertEqual(by_id["t0"]["value"], "closed")

    def test_same_time_contradiction_stays_unresolved(self) -> None:
        from bramastra_lab.research.cognition.episode import mark_conflicts

        workspace = [
            {"record_id": "s1", "subject": "variable:x", "predicate": "value",
             "value": True, "valid_time": 2, "source_event_id": "s1",
             "status": "active"},
            {"record_id": "s2", "subject": "variable:x", "predicate": "value",
             "value": False, "valid_time": 2, "source_event_id": "s2",
             "status": "active"}]
        pairs = mark_conflicts(workspace)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(all(record["status"] == "conflicting"
                            for record in workspace))

    def test_duplicates_do_not_amplify(self) -> None:
        from bramastra_lab.research.cognition.episode import (
            admit_observation_evidence, mark_conflicts)

        workspace: list = []
        for index in range(3):
            admit_observation_evidence(
                workspace, observation={"kind": "read", "variable": "x",
                                        "value": True},
                observation_id=f"dup-{index}")
        self.assertEqual(mark_conflicts(workspace), [])
        sources = {record["source_event_id"] for record in workspace}
        self.assertEqual(len(sources), 3)

    def test_multivalued_members_accumulate(self) -> None:
        from bramastra_lab.research.cognition.episode import mark_conflicts

        workspace = [
            {"record_id": "m1", "subject": "container:red",
             "predicate": "contains", "value": "item_a", "valid_time": 0,
             "source_event_id": "m1", "status": "active"},
            {"record_id": "m2", "subject": "container:red",
             "predicate": "contains", "value": "item_b", "valid_time": 0,
             "source_event_id": "m2", "status": "active"}]
        self.assertEqual(mark_conflicts(workspace), [])
        self.assertTrue(all(record["status"] == "active"
                            for record in workspace))

    def test_false_zero_missing_null_distinct(self) -> None:
        from bramastra_lab.research.cognition.episode import (
            _compact_history_entry)

        def _feedback(value, present=True):
            feedback = {"kind": "read", "variable": "x"}
            if present:
                feedback["value"] = value
            return _compact_history_entry({"feedback": feedback})

        rendered = [_feedback(False), _feedback(0), _feedback(None),
                    _feedback("missing")]
        self.assertEqual(len({json.dumps(entry, sort_keys=True)
                              for entry in rendered}), 4)


class F3PredictorTests(unittest.TestCase):
    def test_prompt_conditioned_on_history(self) -> None:
        from bramastra_lab.research.cognition.episode import ModelWorldModel

        seen: list = []

        class Capture:
            def generate(self, prompt_tokens, *, max_new_tokens):
                seen.append(list(prompt_tokens))
                return {"answer": '{"success_prob": 0.5}',
                        "origin": "capture-double"}

        predictor = ModelWorldModel(Capture())
        base = {"goal": {"target": "x"}, "history": [], "workspace": [],
                "budgets": {}}
        for value in (0, 1):
            predictor(state={**base, "history": [
                {"action": {"kind": "inspect", "variable": "x"},
                 "feedback": {"kind": "read", "variable": "x",
                              "value": value}}]},
                action={"kind": "inspect", "variable": "x"}, depth=1)
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(seen[0], seen[1])

    def test_private_only_change_leaves_prompt_identical(self) -> None:
        from bramastra_lab.research.cognition.episode import ModelWorldModel

        seen: list = []

        class Capture:
            def generate(self, prompt_tokens, *, max_new_tokens):
                seen.append(list(prompt_tokens))
                return {"answer": '{"success_prob": 0.5}',
                        "origin": "capture-double"}

        predictor = ModelWorldModel(Capture())
        state = {"goal": {"target": "x"}, "history": [], "workspace": [],
                 "budgets": {}}
        action = {"kind": "inspect", "variable": "x"}
        predictor(state=state, action=action, depth=1)
        predictor(state=dict(state, _private_hidden_world={"x": True}),
                  action=action, depth=1)
        self.assertEqual(seen[0], seen[1])

    def test_invalid_output_recorded_unknown(self) -> None:
        from bramastra_lab.research.cognition.episode import ModelWorldModel

        class Garbage:
            def generate(self, prompt_tokens, *, max_new_tokens):
                return {"answer": "not json at all",
                        "origin": "garbage-double"}

        predictor = ModelWorldModel(Garbage())
        predicted = predictor(
            state={"goal": {}, "history": [], "workspace": [], "budgets": {}},
            action={"kind": "inspect", "variable": "x"}, depth=1)
        self.assertTrue(predicted["prediction_failed"])
        self.assertIsNone(predicted["success_prob"])

    def test_rejects_out_of_range_probability(self) -> None:
        from bramastra_lab.research.cognition.episode import ModelWorldModel

        class Confident:
            def generate(self, prompt_tokens, *, max_new_tokens):
                return {"answer": '{"success_prob": 7.0}',
                        "origin": "capture-double"}

        predictor = ModelWorldModel(Confident())
        predicted = predictor(
            state={"goal": {}, "history": [], "workspace": [], "budgets": {}},
            action={"kind": "inspect", "variable": "x"}, depth=1)
        self.assertTrue(predicted["prediction_failed"])
        self.assertIsNone(predicted["success_prob"])

    def test_transition_targets_reach_parameters_masked(self) -> None:
        import torch

        from bramastra_lab.research.config import BuildConfig, seed_everything
        from bramastra_lab.research.learning.k8_scoring import (
            world_transition_token_loss)
        from bramastra_lab.research.models import IntegratedModel

        seed_everything(11)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        model = IntegratedModel(config)
        loss = world_transition_token_loss(
            model, config, [1, 2, 3], action={"kind": "press"},
            target_feedback={"result": "ok"})
        self.assertTrue(loss.requires_grad)
        loss.backward()
        grads = [param.grad for param in model.parameters()
                 if param.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(bool(torch.isfinite(grad).all())
                            for grad in grads))


class F4SearchTests(unittest.TestCase):
    def _world(self, table):
        def world(*, state, action, depth):
            prefix = tuple(sorted(
                json.dumps(item, sort_keys=True)
                for item in state.get("applied_prefix", ())))
            key = (prefix, json.dumps(action, sort_keys=True), depth)
            predicted = table.get(key, table.get(
                ((), json.dumps(action, sort_keys=True), depth),
                table.get(json.dumps(action, sort_keys=True),
                          {"feedback": {}, "success_prob": 0.0,
                           "value": 0.0, "origin": "canned-double"})))
            return dict(predicted)
        return world

    def test_delayed_reward_selects_correct_root(self) -> None:
        from bramastra_lab.research.cognition import episode as kernel

        actions = [{"kind": "inspect", "variable": "a"},
                   {"kind": "inspect", "variable": "b"}]
        # Root A looks worse immediately but its depth-2 child wins. Keys
        # are (applied prefix, action, depth): continuations genuinely
        # depend on the branch taken.
        key_a = json.dumps(actions[0], sort_keys=True)
        key_b = json.dumps(actions[1], sort_keys=True)
        table = {
            ((), key_a, 1): {"feedback": {}, "success_prob": 0.4,
                             "value": 0.0, "origin": "canned-double"},
            ((), key_b, 1): {"feedback": {}, "success_prob": 0.6,
                             "value": 0.0, "origin": "canned-double"},
            ((key_a,), key_a, 2): {"feedback": {}, "success_prob": 0.95,
                                   "value": 0.0, "origin": "canned-double"},
            ((key_a,), key_b, 2): {"feedback": {}, "success_prob": 0.1,
                                   "value": 0.0, "origin": "canned-double"},
            ((key_b,), key_a, 2): {"feedback": {}, "success_prob": 0.1,
                                   "value": 0.0, "origin": "canned-double"},
            ((key_b,), key_b, 2): {"feedback": {}, "success_prob": 0.1,
                                   "value": 0.0, "origin": "canned-double"}}
        planner = kernel.BoundedPlannerAdapter(
            world_model=self._world(table))
        chosen = planner.select(
            legal_actions=actions, rendered=[], model=None, workspace=[],
            state_view={"goal": {}, "history": [], "workspace": [],
                        "budgets": {"actions_left": 4, "calls_left": 16,
                                    "nodes_left": 8}})
        self.assertEqual({k: v for k, v in chosen.items()
                          if not k.startswith("_")}, actions[0])
        self.assertTrue(any(len(node.action_prefix) == 2
                            for node in planner.last_imagined))

    def test_permutation_keeps_winner(self) -> None:
        from bramastra_lab.research.cognition import episode as kernel

        actions = [{"kind": "inspect", "variable": name}
                   for name in ("a", "b", "c")]
        table = {json.dumps(action, sort_keys=True):
                 {"feedback": {}, "success_prob": 0.9 if index == 2 else 0.1,
                  "value": 0.0, "origin": "canned-double"}
                 for index, action in enumerate(actions)}
        winners = set()
        for order in (actions, list(reversed(actions))):
            planner = kernel.BoundedPlannerAdapter(
                world_model=self._world(table))
            chosen = planner.select(
                legal_actions=order, rendered=[], model=None, workspace=[],
                state_view={"goal": {}, "history": [], "workspace": [],
                            "budgets": {"actions_left": 4, "calls_left": 16,
                                        "nodes_left": 8}})
            winners.add(json.dumps({k: v for k, v in chosen.items()
                                    if not k.startswith("_")},
                                   sort_keys=True))
        self.assertEqual(len(winners), 1)

    def test_more_roots_than_nodes_records_exclusions(self) -> None:
        from bramastra_lab.research.cognition import episode as kernel

        actions = [{"kind": "inspect", "variable": str(index)}
                   for index in range(10)]
        planner = kernel.BoundedPlannerAdapter(
            world_model=kernel.CannedWorldModel(
                [{"feedback": {}, "success_prob": 0.5, "value": 0.0}]),
            max_nodes=8)
        planner.select(
            legal_actions=actions, rendered=[], model=None, workspace=[],
            state_view={"goal": {}, "history": [], "workspace": [],
                        "budgets": {"actions_left": 4, "calls_left": 16,
                                    "nodes_left": 8}})
        roots = [node for node in planner.last_imagined
                 if len(node.action_prefix) == 1]
        self.assertEqual(len(roots), 8)
        self.assertEqual(len(planner.excluded_root_ids), 2)
        self.assertLessEqual(len(planner.last_imagined), 8)

    def test_all_invalid_predictions_rank_unknown_last(self) -> None:
        from bramastra_lab.research.cognition import episode as kernel

        actions = [{"kind": "inspect", "variable": "a"},
                   {"kind": "submit", "answer": True}]
        planner = kernel.BoundedPlannerAdapter(
            world_model=kernel.CannedWorldModel(
                [{"feedback": {}, "success_prob": "nan", "value": 0.0}]),
            max_nodes=8)
        chosen = planner.select(
            legal_actions=actions, rendered=[], model=None, workspace=[],
            state_view={"goal": {}, "history": [], "workspace": [],
                        "budgets": {"actions_left": 4, "calls_left": 16,
                                    "nodes_left": 8}})
        self.assertIn("kind", chosen)

    def test_real_history_never_mutated(self) -> None:
        from bramastra_lab.research.cognition import episode as kernel
        from bramastra_lab.research.environments.k8_live import (
            build_live_env, generate_live_mechanism)

        mechanism = generate_live_mechanism("inventory", 0, seed=5)
        env = build_live_env(mechanism, budget=6, seed=5)
        planner = kernel.BoundedPlannerAdapter(
            world_model=kernel.CannedWorldModel(
                [{"feedback": {}, "success_prob": 0.5, "value": 0.0}]),
            max_nodes=8)
        out = kernel.run_episode(
            env, planner, model=kernel.ScriptedModel([{"kind": "noop"}]),
            seed=5, mechanism=mechanism, session_job_id="f4-mut",
            checkpoint_id=None)
        for entry in out["history"]:
            self.assertNotIn("predicted_outcome", json.dumps(entry))
            self.assertNotIn("imagined", json.dumps(entry).lower()
                             .replace("imagined_state", ""))


class F5MeasurementTests(unittest.TestCase):
    def test_join_scores_chosen_not_first(self) -> None:
        from bramastra_lab.research.campaigns.phases.e2 import (
            _planner_prediction_gap)

        trace = {
            "history": [
                {"action": {"kind": "inspect", "variable": "b"},
                 "feedback": {"kind": "read", "variable": "b",
                              "value": True}},
                {"action": {"kind": "inspect", "variable": "a"},
                 "feedback": {"kind": "read", "variable": "a",
                              "value": False}}],
            "imagined": [
                {"action_prefix": [{"kind": "inspect", "variable": "a"}],
                 "predicted_outcome": {"success_prob": 0.99,
                                       "feedback": {"kind": "read",
                                                    "variable": "a",
                                                    "value": False}},
                 "node_index": 1},
                {"action_prefix": [{"kind": "inspect", "variable": "b"}],
                 "predicted_outcome": {"success_prob": 0.01,
                                       "feedback": {"kind": "read",
                                                    "variable": "b",
                                                    "value": True}},
                 "node_index": 2}],
            "summary": {"success": True}}
        gap = _planner_prediction_gap(trace)
        self.assertTrue(gap["joined"])
        self.assertEqual(gap["node_index"], 2)
        # Chosen action b failed in prediction (0.01) but succeeded in
        # reality: gap 0.99 measures the CHOSEN node, not the first node.
        self.assertAlmostEqual(gap["success_gap"], 0.99)
        self.assertTrue(gap["feedback_match"])

    def test_unjoinable_trace_excluded_with_reason(self) -> None:
        from bramastra_lab.research.campaigns.phases.e2 import (
            _planner_prediction_gap)

        gap = _planner_prediction_gap(
            {"history": [], "imagined": [], "summary": {}})
        self.assertFalse(gap["joined"])
        self.assertIn("reason", gap)

    def test_forecast_horizon_refused_when_unknown(self) -> None:
        from bramastra_lab.research.cognition.episode import (
            BoundedPlannerAdapter)

        self.assertIsNone(BoundedPlannerAdapter._known_forecast(
            {"success_prob": 0.5, "prediction_failed": True}))
        self.assertIsNone(BoundedPlannerAdapter._known_forecast(
            {"success_prob": "high"}))
        self.assertIsNone(BoundedPlannerAdapter._known_forecast(
            {"success_prob": 7.0}))
        self.assertEqual(BoundedPlannerAdapter._known_forecast(
            {"success_prob": 0.25}), 0.25)


class F6IntegrationTests(unittest.TestCase):
    def test_production_trace_per_family_with_invalid_case(self) -> None:
        import tempfile as _tempfile

        import torch

        from bramastra_lab.research.campaigns.phases import e2
        from bramastra_lab.research.campaigns.phases.ops import (
            RecordingDoubleOps)
        from bramastra_lab.research.campaigns.phases.types import JobInput
        from bramastra_lab.research.campaigns.supervisor import (
            CampaignLedger)
        from bramastra_lab.research.runtime import checkpoint as ckpt

        data_dir = _tempfile.mkdtemp()
        open(os.path.join(data_dir, "manifest.json"), "w").write("{}")
        run = _tempfile.mkdtemp()
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
        ledger.record_allocation("alloc-f6", "src", "data", 480.0)
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
        job = JobInput(phase="E2", slot=0, arm=None, seed=1701,
                       parent="E1-B-1701/E1-A-1701",
                       physical_device="cpu", local_device="cpu",
                       data_dir=data_dir, run_dir=run, precision="fp32",
                       deadline=9e9)
        res = e2.execute(job, ops=RecordingDoubleOps(), eval_cases=1)
        self.assertEqual(res.status, "completed")
        artifact = json.load(open(os.path.join(
            run, "phase_outputs", "E2", "E2-1701.json")))
        # Every supported mode ran, including an all-invalid B arm and the
        # symbolic ceiling; optimizer path stayed frozen.
        for mode in ("b-policy", "b-workspace", "b-planner", "a-direct",
                     "a-fixed", "a-random", "symbolic"):
            self.assertIn(mode, artifact["mode_stats"])
            self.assertGreaterEqual(
                artifact["mode_stats"][mode]["episodes"], 1)
        self.assertIn("calls_reconciled", artifact)
        self.assertTrue(artifact["calls_reconciled"])
        self.assertIn("schema_identities", artifact)
        self.assertIn("planner_calibration", artifact)


if __name__ == "__main__":
    unittest.main()
