"""M05/M06/M12 focused tests: objective router, episode compilation,
teacher interface and the candidate transaction state machine.

No optimizer steps; routing tests are bounded backward/gradient-only.
"""
import unittest

import torch

from bramastra_lab.research.experience.prepare_episodes import Teacher, compile_episode
from bramastra_lab.research.experience.supervision import SupervisionWindow
from bramastra_lab.research.experience.trajectory import (
    ExperienceError,
    ObservedEpisode,
    PublicStep,
    TeacherTarget,
)
from bramastra_lab.research.learning.router import (
    FeatureSwitches,
    RouterError,
    huber_value_loss,
    on_policy_loss,
    route_window,
    teacher_action_cross_entropy,
)
from bramastra_lab.research.orchestration.candidates import (
    CandidateRegistry,
    CandidateTransaction,
    TransactionError,
)


def make_episode(*, truncate=False, submit_success=True):
    steps = [
        PublicStep(step_index=0, goal={"g": 1}, observation={"o": 1},
                   action={"kind": "QUERY", "target": "x"}, feedback={"kind": "ok"},
                   remaining_budget=4, cost=1.0, reward=None,
                   terminated=False, truncated=False),
        PublicStep(step_index=1, goal={"g": 1}, observation={"o": 2},
                   action={"kind": "submit", "value": "7" if submit_success else "8"},
                   feedback={"kind": "submit", "success": submit_success},
                   remaining_budget=3, cost=1.0, reward=1.0 if submit_success else 0.0,
                   terminated=not truncate, truncated=truncate),
    ]
    return ObservedEpisode("ep-1", tuple(steps))


class RouterTests(unittest.TestCase):
    def test_teacher_action_cross_entropy_with_distribution(self) -> None:
        logits = torch.tensor([[2.0, 1.0]])
        distribution = torch.tensor([[0.75, 0.25]])
        mask = torch.tensor([[True, True]])
        loss = teacher_action_cross_entropy(logits, distribution, mask)
        self.assertTrue(torch.isfinite(loss))

    def test_teacher_distribution_must_sum_to_one(self) -> None:
        logits = torch.tensor([[2.0, 1.0]])
        bad = torch.tensor([[0.3, 0.2]])  # does not sum to one
        with self.assertRaises(RouterError):
            teacher_action_cross_entropy(logits, bad,
                                         torch.tensor([[True, True]]))

    def test_huber_value_loss(self) -> None:
        loss = huber_value_loss(torch.tensor([0.5]), torch.tensor([0.0]), delta=1.0)
        self.assertAlmostEqual(float(loss), 0.125, places=5)

    def test_on_policy_requires_detached_advantage(self) -> None:
        advantage = torch.tensor([0.5], requires_grad=True)
        log_prob = torch.tensor([-1.0], requires_grad=True)
        with self.assertRaises(RouterError):
            on_policy_loss(log_prob, advantage)

    def test_on_policy_loss_sign(self) -> None:
        advantage = torch.tensor([1.0])
        good = torch.tensor([-0.1], requires_grad=True)   # high prob action
        bad = torch.tensor([-3.0], requires_grad=True)    # low prob action
        loss_good = on_policy_loss(good, advantage)
        loss_bad = on_policy_loss(bad, advantage)
        self.assertLess(float(loss_good), float(loss_bad))

    def test_route_window_uses_window_denominators(self) -> None:
        window = SupervisionWindow()
        window.add("token", 8)
        window.add("value", 2)
        sums = {"token": torch.tensor(4.0), "value": torch.tensor(1.0)}
        total, report = route_window(window, sums)
        # token: 1.0 * 4/8 = 0.5; value: 0.5 * 1/2 = 0.25 -> 0.75
        self.assertAlmostEqual(float(total), 0.75, places=5)
        self.assertEqual(report["applied"]["token"]["denominator"], 8)

    def test_feature_switches_shape_window(self) -> None:
        switches = FeatureSwitches(token=True, world=False)
        window = switches.build_window({"token": 1.0})
        window.add("token", 3)
        with self.assertRaises(ExperienceError):
            window.add("world", 2)


class CompilationTests(unittest.TestCase):
    def test_observed_episode_compiles_to_expected_targets(self) -> None:
        compiled = compile_episode(make_episode(submit_success=True))
        self.assertEqual(compiled.qualified_steps, 2)
        self.assertEqual(len(compiled.answer_targets), 1)  # successful submission
        self.assertEqual(compiled.answer_targets[0]["answer"], "7")
        returns = {target["step_index"]: target["return"]
                   for target in compiled.return_targets}
        self.assertAlmostEqual(returns[0], 1.0)  # gamma=1 MC from receipt
        self.assertAlmostEqual(returns[1], 1.0)

    def test_failed_submission_is_not_gold(self) -> None:
        compiled = compile_episode(make_episode(submit_success=False))
        self.assertEqual(len(compiled.answer_targets), 0)
        reasons = [entry["reason"] for entry in compiled.excluded_steps]
        self.assertIn("failed_submission_not_gold", reasons)

    def test_truncation_excludes_complete_return(self) -> None:
        compiled = compile_episode(make_episode(truncate=True))
        reasons = [entry["reason"] for entry in compiled.excluded_steps]
        self.assertIn("truncated_no_complete_return", reasons)
        self.assertEqual(len(compiled.return_targets), 0)

    def test_teacher_ties_distribute_uniformly(self) -> None:
        teacher = Teacher("uniform-teacher", lambda prefix, actions: [0.0] * len(actions))
        target = teacher.action_distribution({"step": 0}, [{"kind": "a"}, {"kind": "b"}])
        self.assertIsInstance(target, TeacherTarget)
        self.assertAlmostEqual(sum(target.payload["distribution"]), 1.0)
        self.assertEqual(target.payload["distribution"], [0.5, 0.5])

    def test_identity_is_stable(self) -> None:
        compiled = compile_episode(make_episode())
        self.assertEqual(compiled.identity(), compiled.identity())


class CandidateTransactionTests(unittest.TestCase):
    def transaction(self, **overrides):
        values = dict(transaction_id="tx-1", parent_identity="parent-1",
                      child_identity=None, change_kind="weights",
                      evidence_class="learned", execution_mode="candidate_adaptation",
                      learned_updates=10)
        values.update(overrides)
        return CandidateTransaction(**values)

    def test_zero_update_weight_claim_rejected_by_api(self) -> None:
        registry = CandidateRegistry(namespace="learned")
        with self.assertRaises(TransactionError):
            registry.record(self.transaction(learned_updates=0))

    def test_state_machine_happy_path_publishes_parent(self) -> None:
        registry = CandidateRegistry(namespace="learned")
        registry.record(self.transaction())
        registry.construct_child("tx-1", "child-1")
        registry.record_comparison("tx-1", "cmp-1")
        decided = registry.decide("tx-1", "accepted", chief_approval_hash="hash-1")
        self.assertEqual(decided.status, "accepted")
        self.assertEqual(registry.accepted_parent, "child-1")

    def test_learning_approval_required_for_learned_accept(self) -> None:
        registry = CandidateRegistry(namespace="learned")
        registry.record(self.transaction())
        registry.construct_child("tx-1", "child-1")
        registry.record_comparison("tx-1", "cmp-1")
        with self.assertRaises(TransactionError):
            registry.decide("tx-1", "accepted", chief_approval_hash=None)

    def test_comparison_cannot_change_preregistered_identity(self) -> None:
        registry = CandidateRegistry(namespace="fixture")
        registry.record(self.transaction(evidence_class="fixture", learned_updates=0,
                                         change_kind="config"))
        registry.construct_child("tx-1", "child-1")
        registry.record_comparison("tx-1", "cmp-1")
        with self.assertRaises(TransactionError):
            registry.record_comparison("tx-1", "cmp-2")

    def test_rejected_child_preserves_parent(self) -> None:
        registry = CandidateRegistry(namespace="fixture")
        registry.record(self.transaction(evidence_class="fixture", learned_updates=0,
                                         change_kind="config"))
        registry.construct_child("tx-1", "child-1")
        registry.record_comparison("tx-1", "cmp-1")
        decided = registry.decide("tx-1", "rejected")
        self.assertEqual(decided.status, "rejected")
        self.assertIsNone(registry.accepted_parent)

    def test_fixture_cannot_write_learned_pointer(self) -> None:
        registry = CandidateRegistry(namespace="fixture")
        registry.record(self.transaction(evidence_class="fixture", learned_updates=0,
                                         change_kind="config"))
        registry.construct_child("tx-1", "child-1")
        registry.record_comparison("tx-1", "cmp-1")
        registry.decide("tx-1", "accepted", chief_approval_hash="hash")
        self.assertIsNone(registry.accepted_parent)  # fixture namespace disjoint

    def test_idempotent_retry(self) -> None:
        registry = CandidateRegistry(namespace="fixture")
        registry.record(self.transaction(evidence_class="fixture", learned_updates=0,
                                         change_kind="config"))
        registry.construct_child("tx-1", "child-1")
        registry.record_comparison("tx-1", "cmp-1")
        second = registry.record_comparison("tx-1", "cmp-1")
        self.assertEqual(second.retries, 1)


if __name__ == "__main__":
    unittest.main()
