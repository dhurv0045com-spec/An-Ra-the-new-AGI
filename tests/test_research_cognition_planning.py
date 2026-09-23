"""Behavioral contracts for the bounded cognitive rollout."""
from __future__ import annotations

import unittest

from bramastra_lab.research.cognition.episode import (
    BoundedPlannerAdapter,
    CannedWorldModel,
    ModelInterface,
    ModelWorldModel,
    run_episode,
)


class _CaptureModel(ModelInterface):
    def __init__(self, response: str = '{"feedback":{},"success_prob":0.5}'):
        self.prompts: list[list[int]] = []
        self.response = response

    def generate(self, prompt_tokens, *, max_new_tokens):
        self.prompts.append(list(prompt_tokens))
        return {"answer": self.response, "origin": "capture-model",
                "new_tokens": 4}


class CognitionPlanningTests(unittest.TestCase):
    def test_world_prediction_prompt_changes_with_received_history(self) -> None:
        model = _CaptureModel()
        predictor = ModelWorldModel(model)
        for observation in ("door closed", "door open"):
            predictor(
                state={"goal": {"target": "enter"},
                       "history": [{
                           "action": {"kind": "inspect", "variable": "door"},
                           "feedback": {"kind": "observation",
                                        "value": observation}}]},
                action={"kind": "submit", "answer": "enter"}, depth=1)
        self.assertEqual(len(model.prompts), 2)
        self.assertNotEqual(model.prompts[0], model.prompts[1])

    def test_world_prediction_prompt_ignores_unreceived_private_state(self) -> None:
        model = _CaptureModel()
        predictor = ModelWorldModel(model)
        public = {"goal": {"target": "enter"},
                  "history": [{"action": {"kind": "inspect",
                                             "variable": "door"},
                               "feedback": {"kind": "observation",
                                            "value": "closed"}}],
                  "workspace": [],
                  "budgets": {"actions_left": 3, "calls_left": 4}}
        for secret in ("north", "south"):
            predictor(state={**public,
                             "private_world": {"correct_door": secret},
                             "oracle_answer": secret,
                             "evaluator_label": secret},
                     action={"kind": "inspect", "variable": "door"},
                     depth=1)

        self.assertEqual(len(model.prompts), 2)
        self.assertEqual(
            model.prompts[0], model.prompts[1],
            "unreceived private/evaluator fields must not enter model input")

    def test_episode_prompt_is_invariant_across_private_world_twins(self) -> None:
        from types import SimpleNamespace

        class HiddenRuleEnv:
            def __init__(self, answer: str) -> None:
                self.answer = answer

            def reset(self, *, episode_id):
                return SimpleNamespace(
                    observable_values={"task": "submit the north door"},
                    feedback={"kind": "start"})

            def legal_actions(self):
                return [{"kind": "submit", "answer": "north"}]

            def step(self, action):
                feedback = {"kind": "verdict",
                            "success": action["answer"] == self.answer}
                return SimpleNamespace(feedback=feedback), 0.0, True, False

        model = _CaptureModel()
        outcomes = []
        for hidden_answer in ("north", "south"):
            planner = BoundedPlannerAdapter(
                world_model=ModelWorldModel(model), max_depth=1, max_nodes=1)
            trace = run_episode(
                HiddenRuleEnv(hidden_answer), planner, model=model, seed=29,
                action_budget=1, call_budget=2, node_budget=1)
            outcomes.append(trace["summary"]["success"])

        self.assertEqual(outcomes, [True, False])
        self.assertEqual(len(model.prompts), 2)
        self.assertEqual(
            model.prompts[0], model.prompts[1],
            "hidden mechanism differences must not alter pre-observation input")

    def test_malformed_world_model_response_is_a_counted_failure(self) -> None:
        class MalformedModel(ModelInterface):
            def generate(self, prompt_tokens, *, max_new_tokens):
                return None

        result = ModelWorldModel(MalformedModel())(
            state={"goal": {"target": "finish"}, "history": []},
            action={"kind": "submit", "answer": "x"}, depth=1)
        self.assertTrue(result["prediction_failed"])
        self.assertEqual(result["model_calls"], 1)
        self.assertGreater(result["input_tokens"], 0)
        self.assertEqual(result["output_tokens"], 0)

    def test_depth_two_prediction_uses_imagined_successor_not_original_state(self) -> None:
        states = []

        def world(*, state, action, depth):
            states.append({"state": state, "action": action, "depth": depth})
            if depth == 1:
                feedback = {"observed": f"after-{action['id']}"}
            else:
                feedback = {"observed": "second-step"}
            return {"feedback": feedback, "success_prob": 0.7,
                    "value": 0.0, "origin": "scripted-model"}

        original_history = [{"action": {"id": "prior"},
                             "feedback": {"observed": "real"}}]
        state_view = {"goal": {"target": "finish"},
                      "history": original_history,
                      "workspace": [],
                      "budgets": {"nodes_left": 3, "calls_left": 3}}
        actions = [{"kind": "inspect", "id": "a"},
                   {"kind": "inspect", "id": "b"}]
        planner = BoundedPlannerAdapter(world_model=world, max_nodes=3)
        planner.select(legal_actions=actions, rendered=[], model=_CaptureModel(),
                       workspace=[], state_view=state_view)

        self.assertEqual([call["depth"] for call in states], [1, 1, 2])
        successor = states[2]["state"]
        self.assertEqual(successor["history"][:1], original_history)
        self.assertEqual(successor["history"][-1]["action"], actions[0])
        self.assertTrue(successor["history"][-1]["imagined"])
        self.assertEqual(
            successor["history"][-1]["feedback"]["predicted_feedback"],
            {"observed": "after-a"})
        # The imagined event is confined to the rollout copy.
        self.assertEqual(state_view["history"], original_history)

    def test_depth_two_budget_is_shared_fairly_across_root_actions(self) -> None:
        calls = []

        def world(*, state, action, depth):
            calls.append((depth, action["id"]))
            return {"feedback": {"after": action["id"]},
                    "success_prob": 0.6, "origin": "scripted-model"}

        actions = [{"kind": "inspect", "id": str(i)} for i in range(3)]
        planner = BoundedPlannerAdapter(world_model=world, max_nodes=6)
        planner.select(
            legal_actions=actions, rendered=[], model=_CaptureModel(), workspace=[],
            state_view={"goal": {}, "history": [], "workspace": [],
                        "budgets": {"nodes_left": 6, "calls_left": 6}})
        depth_two_roots = [node.action_prefix[0]["id"]
                           for node in planner.last_imagined
                           if len(node.action_prefix) == 2]
        self.assertEqual(depth_two_roots, ["0", "1", "2"])
        self.assertEqual(len(calls), 6)

    def test_remaining_budget_caps_search_and_rotates_uncovered_roots(self) -> None:
        world = CannedWorldModel([{"feedback": {}, "success_prob": 0.4}])
        planner = BoundedPlannerAdapter(world_model=world, max_nodes=8)
        actions = [{"kind": "inspect", "id": str(i)} for i in range(4)]
        state = {"goal": {}, "history": [], "workspace": [],
                 "budgets": {"nodes_left": 1, "calls_left": 1}}

        first = planner.select(legal_actions=actions, rendered=[],
                               model=_CaptureModel(), workspace=[],
                               state_view=state)
        second = planner.select(legal_actions=actions, rendered=[],
                                model=_CaptureModel(), workspace=[],
                                state_view=state)
        self.assertEqual(first["_planner_nodes"], 1)
        self.assertEqual(second["_planner_nodes"], 1)
        self.assertEqual(first["_model_calls"], 1)
        self.assertEqual(first["_inference_tokens"], 0)
        self.assertEqual(first["id"], "0")
        self.assertEqual(second["id"], "1")
        self.assertEqual(world.calls, 2)

        exhausted = planner.select(
            legal_actions=[*actions, {"kind": "submit", "answer": "safe"}],
            rendered=[], model=_CaptureModel(), workspace=[],
            state_view={"goal": {}, "history": [], "workspace": [],
                        "budgets": {"nodes_left": 0, "calls_left": 0}})
        self.assertEqual(exhausted["kind"], "submit")
        self.assertEqual(exhausted["_origin"], "fallback")
        self.assertEqual(exhausted["_model_calls"], 0)
        self.assertEqual(world.calls, 2)

    def test_world_model_errors_fall_back_without_claiming_a_prediction(self) -> None:
        def broken_world(**_kwargs):
            raise RuntimeError("simulated predictor failure")

        planner = BoundedPlannerAdapter(world_model=broken_world, max_nodes=4)
        chosen = planner.select(
            legal_actions=[{"kind": "inspect", "id": "x"},
                           {"kind": "submit", "answer": "safe"}],
            rendered=[], model=_CaptureModel(), workspace=[],
            state_view={"goal": {}, "history": [], "workspace": [],
                        "budgets": {"nodes_left": 4, "calls_left": 4}})
        self.assertEqual(chosen["kind"], "submit")
        self.assertEqual(chosen["_origin"], "fallback")
        self.assertEqual(chosen["_planner_fallback_reason"],
                         "all_root_predictions_failed")
        self.assertEqual(len(planner.last_imagined), 2)
        self.assertTrue(all(node.predicted_outcome["prediction_failed"]
                            for node in planner.last_imagined))

    def test_planner_refuses_unimplemented_depth(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_depth must be 1 or 2"):
            BoundedPlannerAdapter(world_model=lambda **_: {}, max_depth=3)

    def test_model_tokens_and_planner_calls_are_reported_in_episode_trace(self) -> None:
        class Env:
            def reset(self, *, episode_id):
                from types import SimpleNamespace
                return SimpleNamespace(observable_values={"goal": "finish"},
                                       feedback={"kind": "start"})

            def legal_actions(self):
                return [{"kind": "submit", "answer": "ok"},
                        {"kind": "inspect", "id": "x"}]

            def step(self, action):
                from types import SimpleNamespace
                return (SimpleNamespace(feedback={"kind": "verdict",
                                                   "success": True}),
                        0.0, True, False)

        model = _CaptureModel(
            '{"feedback":{"kind":"predicted"},"success_prob":0.8}')
        predictor = ModelWorldModel(model)
        planner = BoundedPlannerAdapter(world_model=predictor, max_depth=1,
                                        max_nodes=2)
        trace = run_episode(Env(), planner, model=model, seed=11,
                            call_budget=2, node_budget=2)

        self.assertTrue(trace["summary"]["terminated"])
        self.assertEqual(trace["summary"]["model_calls"], 2)
        self.assertEqual(trace["summary"]["inference_input_tokens"],
                         sum(len(prompt) for prompt in model.prompts))
        event = next(event for event in trace["events"]
                     if event["kind"] == "action")
        self.assertEqual(event["planner_meta"]["model_calls"], 2)
        self.assertEqual(event["planner_meta"]["inference_input_tokens"],
                         trace["summary"]["inference_input_tokens"])
        self.assertEqual(event["model_origin"], "capture-model-imagined")

    def test_calibration_joins_executed_root_not_a_depth_two_child(self) -> None:
        from bramastra_lab.research.campaigns.phases.e2 import (
            _planner_prediction_gap)

        chosen = {"kind": "inspect", "id": "chosen"}
        other = {"kind": "inspect", "id": "other"}
        trace = {
            "history": [{"action": chosen,
                         "feedback": {"kind": "verdict", "success": True}}],
            "summary": {"success": True},
            "imagined": [
                {"node_index": 1, "action_prefix": [other],
                 "predicted_outcome": {"success_prob": 0.95,
                                       "feedback": {"other": True}}},
                {"node_index": 2, "action_prefix": [chosen],
                 "predicted_outcome": {"success_prob": 0.25,
                                       "feedback": {"kind": "verdict",
                                                    "success": False}}},
                {"node_index": 3, "action_prefix": [other, chosen],
                 "predicted_outcome": {"success_prob": 0.99,
                                       "feedback": {"future": True}}},
            ]}
        gap = _planner_prediction_gap(trace)
        self.assertTrue(gap["joined"])
        self.assertEqual(gap["node_index"], 2)
        self.assertEqual(gap["predicted_success_prob"], 0.25)
        self.assertEqual(gap["success_gap"], 0.75)
        self.assertFalse(gap["feedback_match"])


if __name__ == "__main__":
    unittest.main()
