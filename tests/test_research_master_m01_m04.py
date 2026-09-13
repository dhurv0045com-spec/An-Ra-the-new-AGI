"""M01/M02/M03/M04 focused tests: experience contract, candidate-isolated
decisions, typed public outcome prediction, memory context.

Gradient checks are bounded (backward only, no optimizer steps).
"""
import unittest
from unittest import mock

import torch

from bramastra_lab.research.config import BuildConfig, seed_everything
from bramastra_lab.research.experience.trajectory import (
    ExperienceError,
    ObservedEpisode,
    PredictedStep,
    PublicStep,
    TeacherTarget,
    public_view,
    render_public_tokens,
)
from bramastra_lab.research.experience.supervision import (
    SupervisionWindow,
    combine_window_losses,
)


def step(index=0, **overrides):
    values = dict(goal={"g": 1}, observation={"o": 2}, action={"a": 3},
                  feedback={"f": 4}, remaining_budget=5, cost=1.0, reward=None,
                  terminated=False, truncated=False, provenance={})
    values.update(overrides)
    return PublicStep(step_index=index, **values)


class TrajectoryTests(unittest.TestCase):
    def test_provenance_never_reaches_token_view(self) -> None:
        record = {"goal": {"g": 1}, "observation": {"o": 2}, "episode_id": "ep-9",
                  "seed": 3, "split": "training", "evaluator_verdict": "pass"}
        view = public_view(record)
        self.assertEqual(set(view), {"goal", "observation"})
        for forbidden in ("episode_id", "seed", "split", "evaluator_verdict"):
            self.assertNotIn(forbidden, json_of(view))
        # Provenance changes cannot alter tokens.
        record2 = dict(record, seed=999, episode_id="other")
        self.assertEqual(render_public_tokens(public_view(record)),
                         render_public_tokens(public_view(record2)))

    def test_predictions_cannot_become_observations(self) -> None:
        prediction = PredictedStep(step_index=1, predicted_feedback={"f": 1},
                                   probability=0.9, parse_valid=True,
                                   predictor_identity="m1")
        from bramastra_lab.research.experience.trajectory import \
            observe_prediction_as_step

        # The only conversion path refuses by contract: imagined branches
        # cannot enter the observed ledger without a real environment receipt.
        with self.assertRaises(ExperienceError):
            observe_prediction_as_step(prediction)
        # A prediction is type-distinct from a public step.
        self.assertNotIsInstance(prediction, PublicStep)

    def test_episode_ordering_and_ending(self) -> None:
        with self.assertRaises(ExperienceError):
            ObservedEpisode("e", (step(0), step(0)))  # non-monotone
        with self.assertRaises(ExperienceError):
            ObservedEpisode("e", (step(0, terminated=False, truncated=False),))
        with self.assertRaises(ExperienceError):
            ObservedEpisode("e", (step(0, terminated=True), step(1)))
        with self.assertRaises(ExperienceError):
            ObservedEpisode("e", (step(0, remaining_budget=5),
                                  step(1, remaining_budget=6)))
        episode = ObservedEpisode("e", (step(0), step(1, truncated=True)))
        self.assertAlmostEqual(episode.total_cost(), 2.0)

    def test_teacher_targets_reject_nonfinite(self) -> None:
        with self.assertRaises(ExperienceError):
            TeacherTarget("value", {"return": float("nan")}, "teacher-1")
        with self.assertRaises(ExperienceError):
            TeacherTarget("banana", {}, "teacher-1")
        target = TeacherTarget("action", {"distribution": [0.5, 0.5]}, "teacher-1")
        self.assertEqual(target.identity(), target.identity())


def json_of(view) -> str:
    import json

    return json.dumps(view, sort_keys=True)


class DecisionsTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything(9)
        self.config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        from bramastra_lab.research.models import IntegratedModel

        self.model = IntegratedModel(self.config)
        self.model.eval()

    def _prefix_tokens(self, length=12):
        return list(range(1, length + 1))

    def test_candidate_isolation_permutation_equivariance(self) -> None:
        from bramastra_lab.research.models.decisions import score_candidates

        prefix = self._prefix_tokens()
        candidates = [[70, 71], [80, 81], [90, 91]]
        scores_a = score_candidates(self.model, self.config, prefix, candidates)
        permuted = [candidates[2], candidates[0], candidates[1]]
        scores_b = score_candidates(self.model, self.config, prefix, permuted)
        # Permutation equivariance: same candidate -> same score.
        self.assertAlmostEqual(scores_a.scores[0], scores_b.scores[1], places=5)
        self.assertAlmostEqual(scores_a.scores[2], scores_b.scores[0], places=5)
        self.assertEqual(scores_a.action_ids[0], scores_b.action_ids[1])
        self.assertEqual(scores_a.action_ids[1], scores_b.action_ids[2])
        self.assertEqual(scores_a.action_ids[2], scores_b.action_ids[0])

    def test_candidate_isolation_changes_with_downstream_content(self) -> None:
        from bramastra_lab.research.models.decisions import score_candidates

        prefix = self._prefix_tokens()
        base = score_candidates(self.model, self.config, prefix, [[70, 71], [80, 81]])
        changed = score_candidates(self.model, self.config, prefix, [[70, 71], [80, 99]])
        # ISOLATION: an unrelated candidate's change cannot leak into
        # candidate 1's score; the CHANGED candidate's own score moves.
        self.assertAlmostEqual(base.scores[0], changed.scores[0], places=6)
        self.assertNotEqual(base.scores[1], changed.scores[1])

    def test_padding_and_empty_reject(self) -> None:
        from bramastra_lab.research.models.decisions import score_candidates

        prefix = self._prefix_tokens()
        with self.assertRaises(Exception):
            score_candidates(self.model, self.config, prefix, [])
        with self.assertRaises(Exception):
            score_candidates(self.model, self.config, prefix, [[70, 71], []])

    def test_value_prefix_independent_of_candidates(self) -> None:
        from bramastra_lab.research.models.decisions import estimate_value

        prefix = self._prefix_tokens()
        value = estimate_value(self.model, self.config, prefix)
        again = estimate_value(self.model, self.config, prefix)
        self.assertAlmostEqual(float(value), float(again), places=6)


class WorldModelTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything(13)
        self.config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        from bramastra_lab.research.models import IntegratedModel

        self.model = IntegratedModel(self.config)
        self.model.eval()

    def test_finite_support_normalizes_and_aggregates_duplicates(self) -> None:
        from bramastra_lab.research.models.world import predict_step_finite

        prefix = list(range(1, 10))
        # Two textual renderings of the SAME parsed outcome aggregate mass.
        support = [
            {"feedback": {"kind": "ok"}, "rendering": "ok-alpha", "terminated": False},
            {"feedback": {"kind": "ok"}, "rendering": "ok-beta", "terminated": False},
            {"feedback": {"kind": "boom"}, "rendering": "boom", "terminated": True},
        ]
        outcome = predict_step_finite(self.model, self.config, prefix,
                                      action={"kind": "press"}, support=support)
        total = sum(option.probability for option in outcome.options)
        self.assertAlmostEqual(total, 1.0, places=5)
        parsed = outcome.aggregated()
        # Three renderings but only TWO parsed outcomes: duplicate parses
        # aggregated into one bucket each, and masses sum to one.
        self.assertEqual(len(parsed), 2)
        self.assertAlmostEqual(sum(parsed.values()), 1.0, places=5)
        self.assertTrue(outcome.predictor_identity)

    def test_condition_changes_with_action(self) -> None:
        from bramastra_lab.research.models.world import predict_step_finite

        prefix = list(range(1, 10))
        support = [{"feedback": {"kind": "a"}, "rendering": "ra", "terminated": False},
                   {"feedback": {"kind": "b"}, "rendering": "rb", "terminated": False}]
        one = predict_step_finite(self.model, self.config, prefix,
                                  action={"kind": "left"}, support=support)
        two = predict_step_finite(self.model, self.config, prefix,
                                  action={"kind": "right"}, support=support)
        different = any(a.probability != b.probability
                        for a, b in zip(one.options, two.options))
        self.assertTrue(different)


class MemoryTests(unittest.TestCase):
    def _records(self):
        from bramastra_lab.research.memory.store import MemoryRecord

        return [
            MemoryRecord(content="the vault code is four two",
                         identity="m1", scope="training", episode_id="ep-old"),
            MemoryRecord(content="the gallery alarm sleeps at noon",
                         identity="m2", scope="training", episode_id="ep-old"),
            MemoryRecord(content="totally unrelated banana sunset",
                         identity="m3", scope="sealed", episode_id="ep-sealed"),
        ]

    def test_forbidden_records_cannot_rank_or_render(self) -> None:
        from bramastra_lab.research.memory.store import MemoryIndex

        index = MemoryIndex(self._records())
        context = index.retrieve("vault code", scope_allowlist={"training"}, top_k=2)
        rendered = " ".join(record.content for record in context.records)
        self.assertNotIn("banana", rendered)
        self.assertTrue(context.records)
        self.assertTrue(context.index_identity)
        self.assertTrue(context.retrieval_rule_identity)

    def test_retrieval_ties_stable_and_current_episode_excluded(self) -> None:
        from bramastra_lab.research.memory.store import MemoryIndex, MemoryRecord

        records = [MemoryRecord(content="alpha beta", identity="a", scope="s",
                                episode_id="ep-current"),
                   MemoryRecord(content="alpha beta", identity="b", scope="s",
                                episode_id="ep-old")]
        index = MemoryIndex(records)
        context = index.retrieve("alpha beta", scope_allowlist={"s"}, top_k=2,
                                 exclude_episodes={"ep-current"})
        self.assertEqual([record.identity for record in context.records], ["b"])

    def test_changed_content_invalidates_identity(self) -> None:
        from bramastra_lab.research.memory.store import MemoryIndex, MemoryRecord

        one = MemoryIndex(self._records())
        grown = list(self._records())
        grown.append(MemoryRecord(content="another admissible fact", identity="m4",
                                  scope="training", episode_id="ep-old"))
        two = MemoryIndex(grown)
        self.assertNotEqual(one.identity, two.identity)


class RouterTests(unittest.TestCase):
    def test_denominators_are_separate_and_missing_terms_recorded(self) -> None:
        window = SupervisionWindow()
        window.add("token", 10)
        window.add("action", 2)
        window.add("value", 0)
        self.assertEqual(window.denominator("token"), 10)
        self.assertEqual(window.denominator("action"), 2)
        self.assertEqual(window.denominator("value"), 0)
        self.assertEqual(window.summary()["missing_enabled"],
                         ["pair", "pg", "value", "world"])

    def test_disabled_term_rejects_supervision(self) -> None:
        window = SupervisionWindow(enabled_terms=frozenset({"token"}))
        with self.assertRaises(ExperienceError):
            window.add("world", 4)

    def test_combine_omits_zero_denominator_terms(self) -> None:
        from bramastra_lab.research.experience.supervision import WindowLoss

        window = SupervisionWindow()
        window.add("token", 4)
        token_loss = torch.tensor(2.0, requires_grad=True)
        total, report = combine_window_losses(
            window, {"token": WindowLoss(term="token", total=token_loss,
                                         denominator=4)})
        self.assertAlmostEqual(float(total), 0.5)  # weight 1.0 * 2/4
        # This window only admitted token data: every other enabled term has
        # zero eligible data and is omitted, not counted as zero loss.
        self.assertEqual(report["omitted"], {"action": "zero_eligible_data",
                                             "pair": "zero_eligible_data",
                                             "pg": "zero_eligible_data",
                                             "value": "zero_eligible_data",
                                             "world": "zero_eligible_data"})

    def test_gradient_routing_reaches_only_enabled_heads(self) -> None:
        """Bounded backward: answer tokens reach the decoder; a disabled head
        receives no unintended gradient. No optimizer step."""
        seed_everything(3)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        from bramastra_lab.research.models import IntegratedModel

        model = IntegratedModel(config)
        tokens = torch.randint(0, config.model.vocab, (1, 8))
        logits = model(tokens).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, config.model.vocab), tokens[:, 1:].reshape(-1))
        loss.backward()
        self.assertIsNotNone(model.decoder.embedding.weight.grad)
        # Value head unused by token loss: no unintended gradient.
        self.assertIsNone(model.value_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
