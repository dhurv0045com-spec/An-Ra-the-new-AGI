"""H01 executable acceptance. Run from root with PYTHONPATH=.; no training.

These tests expose known baseline failures. They live outside the default test
suite so an unimplemented work order does not silently become a CI change.
"""
import copy
import json
import unittest

from bramastra_lab.research.campaigns.phases import compiler
from bramastra_lab.research.cognition import episode as kernel
from bramastra_lab.research.environments.k8_live import build_live_env, generate_live_mechanism


def trajectory():
    return {"public": {"goal": "inspect variable x"}, "answer": "label-A",
            "family": "rule-inquiry", "pool": "training", "mechanism_id": "h01",
            "canonical_identity": "h01", "queries": [
                {"kind": "inspect", "variable": "x"},
                {"kind": "inspect", "variable": "y"}],
            "history": [{"action": {"kind": "inspect", "variable": "x"},
                         "feedback": {"kind": "observation", "value": 1}}]}


def action_channel(row):
    batch = compiler.build_batch_for_trajectory(row)
    return compiler.compile_channels_for_row(row, batch,
        arm_weights={"action": 0.5}, arm_enabled=frozenset({"action"}))["action"]


class H01Acceptance(unittest.TestCase):
    def test_distinct_candidates_retain_full_content(self):
        channel = action_channel(trajectory())
        decoded = [json.loads(bytes(tokens).decode()) for tokens in channel["candidates"]]
        self.assertEqual(decoded, trajectory()["queries"])

    def test_public_goal_changes_decision_input(self):
        first = trajectory()
        second = copy.deepcopy(first)
        second["public"]["goal"] = "inspect variable y"
        self.assertNotEqual(action_channel(first)["prefix_tokens"],
                            action_channel(second)["prefix_tokens"])

    def test_future_labels_do_not_enter_decision_input(self):
        first = trajectory()
        second = copy.deepcopy(first)
        second["answer"] = "different label"
        second["history"][0]["feedback"]["value"] = 999
        self.assertEqual(action_channel(first)["prefix_tokens"],
                         action_channel(second)["prefix_tokens"])

    def test_real_submission_roundtrips(self):
        for family in ("rule-inquiry", "inventory", "program"):
            with self.subTest(family=family):
                env = build_live_env(generate_live_mechanism(family, 0, seed=99), budget=6, seed=99)
                env.reset(episode_id="h01")
                action = dict(env.oracle_answer())  # Test ceiling only.
                legal = env.legal_actions()
                code = kernel.encode_action_code(action, legal)
                self.assertLessEqual(len(code.encode("utf-8")), 24)
                decoded, _ = kernel.decode_action_code(json.loads(code), legal)
                self.assertEqual(decoded, action)
                observation, _, _, _ = env.step(decoded)
                self.assertTrue(observation.feedback.get("correct"))

    def test_typed_payload_rejections(self):
        legal = [{"kind": "submit"}]
        for code in ({"a": 0, "b": "true"}, {"a": 0, "n": True},
                     {"a": 0, "i": 2}, {"a": 0, "b": True, "n": 1},
                     {"a": True}, {"a": 9}):
            with self.subTest(code=code), self.assertRaises((ValueError, kernel.EpisodeError)):
                kernel.decode_action_code(code, legal)

    def test_compact_submission_reaches_policy_consumer(self):
        class CodeModel(kernel.ModelInterface):
            def __init__(self, code):
                self.code = code
            def generate(self, prompt_tokens, *, max_new_tokens):
                return {"answer": self.code, "origin": "h01-test-double",
                        "generation_id": "h01-generation"}
        for adapter in (kernel.LearnedPolicyAdapter(), kernel.WorkspacePolicyAdapter()):
            for code, payload in (({"a": 0, "i": "item_561"}, {"item": "item_561"}),
                                  ({"a": 0, "n": 45}, {"value": 45})):
                with self.subTest(adapter=adapter.name, code=code):
                    action = adapter.select(legal_actions=[{"kind": "submit"}], rendered=[259],
                        model=CodeModel(json.dumps(code, separators=(",", ":"))), workspace=[],
                        state_view={"goal": {}, "history": [], "budgets": {}})
                    self.assertEqual(action.get("kind"), "submit")
                    for key, value in payload.items():
                        self.assertEqual(action.get(key), value)
                    self.assertEqual(action.get("_origin"), "h01-test-double")

    def test_policy_input_identifies_action_mapping(self):
        prompts = []
        class Capture(kernel.ModelInterface):
            def __init__(self, variable):
                self.variable = variable
            def generate(self, prompt_tokens, *, max_new_tokens):
                prompts.append(list(prompt_tokens))
                return {"answer": json.dumps({"kind": "inspect", "variable": self.variable}),
                        "origin": "h01-test-double"}
        for variable in ("x", "y"):
            kernel.LearnedPolicyAdapter().select(legal_actions=[{"kind": "inspect", "variable": variable}],
                rendered=[259], model=Capture(variable), workspace=[], state_view={"goal": {}})
        self.assertNotEqual(prompts[0], prompts[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
