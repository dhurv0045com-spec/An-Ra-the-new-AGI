"""B10 focused tests: bounded planning, oracle gating, collection policies."""
import inspect
import unittest

import torch

from bramastra_lab.research.config import BuildConfig, seed_everything
from bramastra_lab.research.experience.codec import SPECIAL_BOUNDARY, encode_event
from bramastra_lab.research.models import IntegratedModel
from bramastra_lab.research.planning.planner import (
    BoundedRolloutPlanner,
    FixedCollectionPolicy,
    LearnedCollectionPolicy,
    OracleAdapter,
    PlanningError,
    PlanStep,
    no_planning_baseline,
    record_real_interaction,
)
from bramastra_lab.research.runtime import inference


def tiny_model():
    seed_everything(29)
    config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
    model = IntegratedModel(config)
    model.eval()
    return model, config


def candidate_view():
    tokens = [SPECIAL_BOUNDARY] + encode_event("goal", {"choose": "route"})
    candidates = []
    spans = []
    for index in range(3):
        event = encode_event("candidate", {"id": index})
        tokens += event
        spans.append(len(tokens) - 1)
        candidates.append({"kind": "move", "target": index})
    return tokens, spans, candidates


class PlannerBoundTests(unittest.TestCase):
    def test_depth_and_node_bounds_stop_search(self) -> None:
        model, config = tiny_model()
        tokens, spans, candidates = candidate_view()
        planner = BoundedRolloutPlanner(model, config, max_depth=2, max_nodes=2)
        result = planner.plan(sequence_tokens=tokens, candidate_spans=spans,
                              candidate_actions=candidates,
                              legal_mask=[True, True, True])
        self.assertLessEqual(len(result.steps), 2)
        self.assertEqual(result.imagined_steps, len(result.steps))
        self.assertTrue(all(step.imagined for step in result.steps))
        self.assertFalse(result.used_oracle)

    def test_tiny_node_budget_can_stop_before_candidates(self) -> None:
        model, config = tiny_model()
        tokens, spans, candidates = candidate_view()
        planner = BoundedRolloutPlanner(model, config, max_depth=1, max_nodes=1)
        result = planner.plan(sequence_tokens=tokens, candidate_spans=spans,
                              candidate_actions=candidates,
                              legal_mask=[True, True, True])
        self.assertLessEqual(len(result.steps), 1)

    def test_illegal_candidates_never_planned(self) -> None:
        model, config = tiny_model()
        tokens, spans, candidates = candidate_view()
        planner = BoundedRolloutPlanner(model, config, max_depth=1, max_nodes=8)
        result = planner.plan(sequence_tokens=tokens, candidate_spans=spans,
                              candidate_actions=candidates,
                              legal_mask=[True, False, False])
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].action, candidates[0])

    def test_predicted_outcomes_cannot_be_written_as_observed(self) -> None:
        model, config = tiny_model()
        tokens, spans, candidates = candidate_view()
        planner = BoundedRolloutPlanner(model, config, max_depth=1, max_nodes=4)
        result = planner.plan(sequence_tokens=tokens, candidate_spans=spans,
                              candidate_actions=candidates,
                              legal_mask=[True, True, False])
        for step in result.steps:
            with self.assertRaises(PlanningError):
                record_real_interaction(step)
        # A step declared real (from an actual environment step) records fine.
        record_real_interaction(PlanStep(action={"kind": "read"}, imagined=False,
                                         predicted_feedback={}, depth=0))

    def test_planner_uses_only_the_public_model_api(self) -> None:
        model, config = tiny_model()
        tokens, spans, candidates = candidate_view()
        planner = BoundedRolloutPlanner(model, config)
        # The planner signature accepts token/candidate inputs only; there is
        # no environment or hidden-state parameter to leak through.
        parameters = inspect.signature(planner.plan).parameters
        self.assertNotIn("environment", parameters)
        self.assertNotIn("hidden_state", parameters)


class OracleGatingTests(unittest.TestCase):
    def test_oracle_adapter_requires_explicit_acknowledgement(self) -> None:
        with self.assertRaises(PlanningError):
            OracleAdapter(lambda view: {"kind": "submit"})
        adapter = OracleAdapter(lambda view: {"kind": "submit"}, allow_oracle=True)
        result = adapter.plan({"legal_actions": [{"kind": "submit"}]})
        self.assertTrue(result.used_oracle)
        self.assertEqual(result.policy_identity, "oracle-diagnostic/v1")
        self.assertIn("used_oracle", result.to_dict())

    def test_primary_inference_has_no_oracle_parameter(self) -> None:
        signature = inspect.signature(inference.generate_free_form)
        self.assertNotIn("oracle", signature.parameters)

    def test_identity_distinguishes_oracle_mode(self) -> None:
        model, config = tiny_model()
        tokens, spans, candidates = candidate_view()
        planner = BoundedRolloutPlanner(model, config, max_depth=1)
        planned = planner.plan(sequence_tokens=tokens, candidate_spans=spans,
                               candidate_actions=candidates,
                               legal_mask=[True, True, True])
        adapter = OracleAdapter(lambda view: candidates[0], allow_oracle=True)
        oracle = adapter.plan({"legal_actions": candidates})
        self.assertNotEqual(planned.identity, oracle.identity)
        self.assertTrue(oracle.used_oracle)
        self.assertFalse(planned.used_oracle)


class CollectionPolicyTests(unittest.TestCase):
    def test_no_planning_baseline_acts_greedily_and_fails_loud(self) -> None:
        view = {"legal_actions": [{"kind": "a"}, {"kind": "b"}]}
        self.assertEqual(no_planning_baseline(view), {"kind": "a"})
        with self.assertRaises(PlanningError):
            no_planning_baseline({"legal_actions": []})

    def test_fixed_policy_counts_real_interactions(self) -> None:
        policy = FixedCollectionPolicy(seed=1)
        view = {"legal_actions": [{"kind": "a"}, {"kind": "b"}]}
        for _ in range(3):
            self.assertIn(policy.select(view), view["legal_actions"])
        self.assertEqual(policy.real_interactions, 3)

    def test_learned_policy_uses_model_and_falls_back(self) -> None:
        model, config = tiny_model()
        tokens, spans, candidates = candidate_view()
        policy = LearnedCollectionPolicy(model, config)
        action = policy.select({
            "candidates": [{"action": candidate, "span_end": span}
                           for candidate, span in zip(candidates, spans)],
            "sequence_tokens": tokens, "legal_mask": [True, True, True],
            "legal_actions": candidates,
        })
        self.assertIn(action, candidates)
        self.assertEqual(policy.real_interactions, 1)
        self.assertEqual(policy.fallback_uses, 0)

        broken = LearnedCollectionPolicy(model, config)
        broken_action = broken.select({"legal_actions": candidates})  # no model inputs
        self.assertIn(broken_action, candidates)
        self.assertEqual(broken.fallback_uses, 1)

    def test_learned_policy_is_not_trained_in_this_build(self) -> None:
        model, config = tiny_model()
        policy = LearnedCollectionPolicy(model, config)
        weights_before = {name: tensor.clone() for name, tensor in
                          policy.model.state_dict().items()}
        policy.select({"legal_actions": [{"kind": "a"}]})  # fallback path
        for name, tensor in policy.model.state_dict().items():
            self.assertTrue(torch.equal(weights_before[name], tensor))


if __name__ == "__main__":
    unittest.main()
