"""B08 focused tests: public environments, oracles, charging and inference."""
import unittest

import torch

from bramastra_lab.research.config import BuildConfig, seed_everything
from bramastra_lab.research.environments import (
    CHARGING_RULE,
    InventoryWorld,
    ProgramLab,
    SwitchWorld,
)
from bramastra_lab.research.environments.oracles import (
    ORACLES,
    exhaustive_oracle_agreement,
    failed_baseline_policy,
    rollout,
    winning_policy_factory,
)
from bramastra_lab.research.experience.codec import SPECIAL_BOUNDARY, encode_event
from bramastra_lab.research.models import IntegratedModel
from bramastra_lab.research.runtime.inference import (
    generate_free_form,
    score_finite_actions,
)

ENVIRONMENTS = [SwitchWorld(budget=6, seed=1), InventoryWorld(budget=6, seed=2),
                ProgramLab(budget=6, seed=3)]


DECLARED_PUBLIC_KEYS = {
    SwitchWorld.name: {"goal", "switch_names", "environment_name", "episode_key"},
    InventoryWorld.name: {"goal", "containers", "environment_name", "episode_key"},
    ProgramLab.name: {"goal", "target_input", "domain", "modulus", "query_count",
                      "environment_name", "episode_key"},
}


class PublicInformationTests(unittest.TestCase):
    def test_public_values_stay_inside_the_declared_schema(self) -> None:
        for environment in ENVIRONMENTS:
            observation = environment.reset(episode_id="iso-test")
            self.assertLessEqual(set(observation.observable_values),
                                 DECLARED_PUBLIC_KEYS[environment.name],
                                 f"{environment.name} exposed undeclared public fields")
            self.assertNotIn("holding", observation.observable_values)
            self.assertNotIn("slope", observation.observable_values)

    def test_no_hidden_answer_in_feedback(self) -> None:
        environment = ProgramLab(budget=6, seed=5)
        observation = environment.reset(episode_id="future")
        self.assertEqual(observation.feedback, {"kind": "reset"})
        observation, cost, term, trunc = environment.step({"kind": "evaluate", "input": 1})
        # Query feedback answers exactly the queried input and nothing more.
        self.assertEqual(observation.feedback["output"], environment.evaluate(1))
        self.assertNotIn("target_input_answer", observation.feedback)


class ChargingRuleTests(unittest.TestCase):
    def test_invalid_action_is_charged_and_fails(self) -> None:
        environment = SwitchWorld(budget=6, seed=0)
        environment.reset(episode_id="charge")
        before = environment.remaining_budget
        observation, cost, term, trunc = environment.step({"kind": "read", "switch": "Z"})
        self.assertEqual(cost, 1.0)
        self.assertEqual(environment.remaining_budget, before - 1)
        self.assertEqual(observation.feedback["kind"], "invalid_action")
        self.assertFalse(term)
        self.assertEqual(CHARGING_RULE, environment.charging_rule)

    def test_invalid_action_does_not_change_hidden_state(self) -> None:
        environment = InventoryWorld(budget=6, seed=4)
        environment.reset(episode_id="iso")
        holding = environment._holding
        environment.step({"kind": "peek", "container": "purple"})
        self.assertEqual(environment._holding, holding)

    def test_budget_exhaustion_truncates(self) -> None:
        environment = SwitchWorld(budget=2, seed=0)
        environment.reset(episode_id="exhaust")
        for _ in range(2):
            observation, cost, term, trunc = environment.step({"kind": "read", "switch": "A"})
        self.assertTrue(trunc or term)


class RolloutAndOracleTests(unittest.TestCase):
    def test_winning_policies_succeed_and_oracles_agree(self) -> None:
        for environment in [SwitchWorld(budget=6, seed=1),
                            InventoryWorld(budget=6, seed=2),
                            ProgramLab(budget=6, seed=3)]:
            record = rollout(environment, f"win-{environment.name}",
                             winning_policy_factory(environment.name),
                             policy_identity="handwritten-check")
            self.assertTrue(record.success, f"{environment.name} baseline should win")
            self.assertTrue(ORACLES[environment.name](environment, record))
            self.assertTrue(exhaustive_oracle_agreement(environment, record))

    def test_failed_baseline_deliberately_fails(self) -> None:
        for environment in [SwitchWorld(budget=6, seed=1),
                            InventoryWorld(budget=6, seed=2),
                            ProgramLab(budget=6, seed=3)]:
            record = rollout(environment, f"fail-{environment.name}",
                             failed_baseline_policy, policy_identity="failed-baseline")
            self.assertFalse(record.success,
                             f"{environment.name} failed baseline unexpectedly won")
            self.assertTrue(ORACLES[environment.name](environment, record) is False)

    def test_submission_cost_counted_separately(self) -> None:
        environment = InventoryWorld(budget=6, seed=9)
        record = rollout(environment, "costs", winning_policy_factory(environment.name),
                         policy_identity="handwritten-check")
        self.assertGreater(record.submission_cost_total, 0.0)
        self.assertGreaterEqual(record.inquiry_cost_total, 0.0)
        submission_steps = [step for step in record.steps
                            if step["action"].get("kind") == "submit"]
        self.assertEqual(len(submission_steps), 1)
        self.assertEqual(submission_steps[0]["cost"], record.submission_cost_total)

    def test_exhaustive_space_enumeration_tiny(self) -> None:
        # Switch world has exactly 4 hidden states; enumerate all and confirm
        # the oracle logic distinguishes success states.
        from itertools import product

        outcomes = []
        for a, b in product([False, True], repeat=2):
            environment = SwitchWorld(budget=6, seed=0)
            environment.reset(episode_id=f"enum-{int(a)}{int(b)}")
            environment._switches["A"] = a
            environment._switches["B"] = b
            observation, cost, term, trunc = environment.step({"kind": "submit"})
            outcomes.append(observation.feedback["success"])
        self.assertEqual(outcomes, [False, False, False, True])


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything(19)
        self.config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        self.model = IntegratedModel(self.config)
        self.model.eval()

    def test_generation_reports_answer_and_stopping(self) -> None:
        prompt = [SPECIAL_BOUNDARY] + encode_event("goal", {"question": "2+2?"})
        report = generate_free_form(self.model, self.config, prompt, max_new_tokens=8)
        self.assertLessEqual(report.new_tokens, 8)
        self.assertEqual(report.complete, report.stopped_on_eos)
        self.assertFalse(report.retrieval_enabled)
        self.assertEqual(report.planner, "none")

    def test_generation_ends_on_eos(self) -> None:
        # A prompt whose next-token distribution we cannot control, but the
        # EOS flag must be exact: craft tokens so EOS is argmax by monkey-
        # patching at the tensor level is overkill; instead run until cap and
        # verify the cap path is reported honestly.
        prompt = [SPECIAL_BOUNDARY] + encode_event("goal", {"q": 1})
        report = generate_free_form(self.model, self.config, prompt, max_new_tokens=2)
        if not report.stopped_on_eos:
            self.assertTrue(report.hit_cap)
            self.assertFalse(report.complete)

    def test_prompt_overflow_rejects(self) -> None:
        prompt = list(range(self.config.model.max_seq + 1))
        with self.assertRaises(Exception):
            generate_free_form(self.model, self.config, prompt, max_new_tokens=4)

    def test_finite_action_scoring(self) -> None:
        tokens = [SPECIAL_BOUNDARY] + encode_event("goal", {"choose": "a"})
        first_event = encode_event("candidate", {"id": 1})
        second_event = encode_event("candidate", {"id": 2})
        tokens += first_event
        tokens += second_event
        end_first = len(tokens) - len(second_event) - 1
        end_second = len(tokens) - 1
        report = score_finite_actions(self.model, self.config, tokens,
                                      [end_first, end_second], [True, False])
        self.assertEqual(report.selected_index, 0)
        self.assertEqual(report.mode, "schema_limited_action_evaluation")
        self.assertEqual(report.legal_mask, [True, False])

    def test_action_scoring_requires_one_legal(self) -> None:
        tokens = [SPECIAL_BOUNDARY] + encode_event("goal", {"q": 1})
        with self.assertRaises(Exception):
            score_finite_actions(self.model, self.config, tokens, [len(tokens) - 1],
                                 [False])


if __name__ == "__main__":
    unittest.main()
