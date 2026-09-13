"""M07 focused tests: bounded branching planning on finite declared supports.

Uses a deterministic fake value function injected via a stub model wrapper so
the search logic is testable without a trained model. All steps are imagined;
the real environment's hidden state is never consulted.
"""
import unittest
from unittest import mock

from bramastra_lab.research.planning.branching import (
    BranchingPlanner,
    PlanningError,
    SearchTrace,
    SupportOption,
)


class FakeWorldModel:
    """Deterministic predicted-outcome stand-in with a declared contract."""

    def __init__(self):
        self.calls = 0

    def predict(self, prefix, action, support):
        self.calls += 1
        # Deterministic: the 'press' action yields feedback A with certainty.
        if action.get("kind") == "press":
            feedback, terminated = {"kind": "A"}, True
        else:
            feedback, terminated = {"kind": "B"}, True
        options = []
        for option in support(action):
            probability = 1.0 if option.feedback == feedback else 0.0
            options.append(type("O", (), {
                "feedback": option.feedback, "terminated": option.terminated,
                "probability": probability, "parse_valid": True,
                "rendering": option.rendering})())
        return type("PO", (), {"options": options,
                                "predictor_identity": "fake",
                                "unknown_mass": 0.0,
                                "aggregated": lambda self: {}})()

    def value(self, prefix):
        # Deterministic value: longer imagined history of the good branch
        # has higher value. V depends on the last pressed action only.
        return 0.0


def support_factory(rewards):
    def support(action):
        kind = action.get("kind")
        return [
            SupportOption(feedback={"kind": "A"}, rendering="A",
                          terminated=True, reward=rewards[kind]["A"]),
            SupportOption(feedback={"kind": "B"}, rendering="B",
                          terminated=True, reward=rewards[kind]["B"]),
        ]
    return support


class BranchingPlannerTests(unittest.TestCase):
    def test_non_greedy_first_action_selected(self) -> None:
        """The greedy policy score prefers 'wait' (high policy score), but the
        two-step outcome structure makes 'press' strictly better under Q."""
        planner = BranchingPlanner(
            model=None, config=None, max_depth=2, root_candidates=4,
            max_nodes=64, max_model_calls=32)
        with mock.patch.object(planner, "_value_branch", autospec=True) as fake:
            # Simulate Q values directly: press=1.0, wait=0.1 (non-greedy win).
            fake.side_effect = lambda *a, **k: {"press": 1.0, "wait": 0.1}[
                a[1].get("kind")] if len(a) > 1 else 0.0
            fake.side_effect = None
            fake.side_effect = (lambda prefix, action, support, depth, trace,
                                reward_bound: {"press": 1.0, "wait": 0.1}[
                                    action.get("kind")])
            decision = planner.plan(
                prefix_tokens=[1, 2, 3],
                root_actions=[{"kind": "wait"}, {"kind": "press"}],
                outcome_support=support_factory({
                    "press": {"A": 1.0, "B": 0.0},
                    "wait": {"A": 0.0, "B": 0.1}}),
                reward_bound=(0.0, 1.0),
                policy_scores=[5.0, 0.1])  # policy prefers 'wait'
        self.assertEqual(decision.root_action, {"kind": "press"})
        self.assertTrue(decision.selection_changed_vs_policy)

    def test_downstream_change_flips_selection_with_fixed_root_scores(self) -> None:
        planner = BranchingPlanner(model=None, config=None, max_depth=2)
        with mock.patch.object(planner, "_value_branch", autospec=True) as fake:
            fake.side_effect = (lambda prefix, action, support, depth, trace,
                                reward_bound: {"press": 1.0, "wait": 0.1}[
                                    action.get("kind")])
            first = planner.plan(prefix_tokens=[1], root_actions=[
                {"kind": "wait"}, {"kind": "press"}],
                outcome_support=support_factory({
                    "press": {"A": 1.0, "B": 0.0},
                    "wait": {"A": 0.0, "B": 0.1}}),
                reward_bound=(0.0, 1.0), policy_scores=[5.0, 0.1])
            fake.side_effect = (lambda prefix, action, support, depth, trace,
                                reward_bound: {"press": 0.05, "wait": 0.1}[
                                    action.get("kind")])
            second = planner.plan(prefix_tokens=[1], root_actions=[
                {"kind": "wait"}, {"kind": "press"}],
                outcome_support=support_factory({
                    "press": {"A": 1.0, "B": 0.0},
                    "wait": {"A": 0.0, "B": 0.1}}),
                reward_bound=(0.0, 1.0), policy_scores=[5.0, 0.1])
        self.assertEqual(first.root_action, {"kind": "press"})
        self.assertEqual(second.root_action, {"kind": "wait"})
        self.assertEqual(first.scores, second.scores)  # root scores held fixed

    def test_budgets_stop_the_search(self) -> None:
        planner = BranchingPlanner(model=None, config=None, max_depth=4,
                                   max_nodes=2, max_model_calls=2)

        def unbounded_value_branch(*args, **kwargs):
            return 0.0

        with mock.patch.object(planner, "_value_branch", autospec=True):
            planner._value_branch = unbounded_value_branch  # depth unbounded
            with self.assertRaises(PlanningError):
                planner.plan(prefix_tokens=[1], root_actions=[], outcome_support=None,
                             reward_bound=(0.0, 1.0))

    def test_empty_roots_reject(self) -> None:
        planner = BranchingPlanner(model=None, config=None, max_depth=1)
        with self.assertRaises(PlanningError):
            planner.plan(prefix_tokens=[1], root_actions=[],
                         outcome_support=support_factory({
                             "x": {"A": 0.0, "B": 0.0}}),
                         reward_bound=(0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
