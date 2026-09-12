"""B04 focused tests: objectives, treatments, schedules, trainer mechanics.

The arithmetic and state tests below run without any optimizer update.
Tests that perform real optimizer updates are gated behind
``BRAMASTRA_LEARNED_CHECKS=1`` (owner-authorized learned smoke only) and
report themselves as skipped otherwise. These tests validate loss plumbing,
not learning capability.
"""
import os
import unittest

import torch

from bramastra_lab.research.config import BuildConfig, seed_everything
from bramastra_lab.research.experience.sequences import build_answer_row, collocate
from bramastra_lab.research.learning.objectives import (
    answer_eos_loss_sum,
    pair_margin_loss,
    sequence_logprob_scores,
)
from bramastra_lab.research.learning.schedules import make_schedule
from bramastra_lab.research.learning.treatments import (
    TreatmentError,
    TreatmentSchema,
    apply_treatment,
    validate_targets_in_schema,
)

LEARNED_CHECKS = os.environ.get("BRAMASTRA_LEARNED_CHECKS") == "1"
LEARNED_REASON = ("learned checks deferred: owner has not authorized optimizer updates "
                  "(set BRAMASTRA_LEARNED_CHECKS=1 to run)")

VOCAB = 260


def make_batch(answer: str, question: str, *, provenance_suffix: str = ""):
    row = build_answer_row(
        [("goal", {"question": question})], answer,
        provenance={"episode_id": "e" + provenance_suffix, "task_semantic_id": "t",
                    "split": "training", "source": "test", "collection_policy": "fixed"},
        max_tokens=64)
    return collocate([row], max_seq=64)


class AnswerLossTests(unittest.TestCase):
    def test_masked_sum_matches_manual_computation(self) -> None:
        logits = torch.zeros(1, 3, 5)
        logits[0, 0] = torch.tensor([2.0, 1.0, 0.5, 0.1, -1.0])
        labels = torch.tensor([[1, 2, -100]])
        mask = torch.tensor([[True, True, False]])
        report = answer_eos_loss_sum(logits, labels, mask)
        expected = (torch.nn.functional.cross_entropy(
            logits[0].float(), torch.tensor([1, 2, 0]), reduction="none")[:2].sum())
        self.assertAlmostEqual(report.total.item(), expected.item(), places=5)
        self.assertEqual(report.target_count, 2)
        self.assertAlmostEqual(report.mean.item(), expected.item() / 2, places=5)

    def test_one_token_answer_plus_eos_denominator(self) -> None:
        batch = make_batch("7", "1+1?")
        self.assertEqual(batch.target_count, 2)

    def test_mask_label_disagreement_rejects(self) -> None:
        logits = torch.zeros(1, 2, 5)
        labels = torch.tensor([[1, -100]])
        mask = torch.tensor([[True, True]])
        with self.assertRaises(ValueError):
            answer_eos_loss_sum(logits, labels, mask)
        labels_bad = torch.tensor([[1, 2]])
        mask_bad = torch.tensor([[True, False]])
        with self.assertRaises(ValueError):
            answer_eos_loss_sum(logits, labels_bad, mask_bad)

    def test_masking_selects_positions_not_classes(self) -> None:
        # A valid gold class at a supervised position always contributes; the
        # class space is never restricted by the mask.
        logits = torch.zeros(1, 2, VOCAB)
        logits[0, 0, 42] = 3.0
        labels = torch.tensor([[42, -100]])
        mask = torch.tensor([[True, False]])
        report = answer_eos_loss_sum(logits, labels, mask)
        expected = -torch.log_softmax(logits[0, 0].float(), -1)[42]
        self.assertAlmostEqual(report.total.item(), expected.item(), places=5)


class TreatmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = TreatmentSchema(schema_id="arithmetic/v1",
                                      participating=(257, 258, 49, 50, 51)).validated_for_vocab(VOCAB)

    def test_full_treatment_agrees_with_ordinary_ce(self) -> None:
        logits = torch.randn(2, 4, VOCAB)
        labels = torch.randint(0, VOCAB, (2, 4))
        mask = torch.ones(2, 4, dtype=torch.bool)
        treated = apply_treatment(logits, self.schema, "full")
        direct = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB).float(), labels.reshape(-1), reduction="mean")
        through = answer_eos_loss_sum(treated, labels, mask).mean
        self.assertAlmostEqual(direct.item(), through.item(), places=5)

    def test_participating_mask_excludes_competition(self) -> None:
        logits = torch.randn(1, 2, VOCAB)
        treated = apply_treatment(logits, self.schema, "participating_mask")
        outside = [token for token in range(VOCAB) if token not in self.schema.participating]
        self.assertTrue(torch.isinf(treated[..., outside]).all())
        self.assertTrue(torch.isfinite(treated[..., list(self.schema.participating)]).all())
        # Input tensor untouched: treatments are training-only.
        self.assertFalse(torch.isinf(logits[..., outside]).any())

    def test_inactive_offset_shifts_by_declared_amount(self) -> None:
        import math

        participating = len(self.schema.participating)
        effective_vocab = 8
        offset = math.log((VOCAB - participating) / (effective_vocab - participating))
        logits = torch.randn(1, 2, VOCAB)
        treated = apply_treatment(logits, self.schema, "inactive_offset",
                                  effective_vocab=effective_vocab)
        outside = [token for token in range(VOCAB) if token not in self.schema.participating]
        torch.testing.assert_close(treated[..., outside], logits[..., outside] - offset)
        torch.testing.assert_close(treated[..., list(self.schema.participating)],
                                   logits[..., list(self.schema.participating)])

    def test_offset_validates_geometry(self) -> None:
        with self.assertRaises(TreatmentError):
            apply_treatment(torch.zeros(1, 1, VOCAB), self.schema, "inactive_offset",
                            effective_vocab=len(self.schema.participating))
        with self.assertRaises(TreatmentError):
            apply_treatment(torch.zeros(1, 1, VOCAB), self.schema, "inactive_offset",
                            effective_vocab=VOCAB + 1)
        with self.assertRaises(TreatmentError):
            apply_treatment(torch.zeros(1, 1, VOCAB), self.schema, "inactive_offset")

    def test_inconsistent_batch_is_rejected_not_masked(self) -> None:
        labels = torch.tensor([[999]])
        mask = torch.tensor([[True]])
        with self.assertRaises(TreatmentError) as caught:
            validate_targets_in_schema(labels, mask, self.schema)
        self.assertIn("999", str(caught.exception))

    def test_schema_requires_structural_and_positive_tokens(self) -> None:
        with self.assertRaises(TreatmentError):
            TreatmentSchema(schema_id="x", participating=())
        with self.assertRaises(TreatmentError):
            TreatmentSchema(schema_id="x", participating=(1, 1))
        with self.assertRaises(TreatmentError):
            TreatmentSchema(schema_id="x", tokenizer_identity="other/v1", participating=(1,))


class PairObjectiveTests(unittest.TestCase):
    def test_goal_blind_prediction_is_penalized(self) -> None:
        # score(y_other|g_own) = 0.0, score(y_own|g_own) = -1.0, margin 0.5:
        # violation = 0.0 - (-1.0) + 0.5 = 1.5.
        own = torch.tensor([-1.0])
        swapped = torch.tensor([0.0])
        loss, counted = pair_margin_loss(own, swapped, 0.5)
        self.assertEqual(counted, 1)
        self.assertAlmostEqual(loss.item(), 1.5, places=5)

    def test_separated_goals_permit_zero_loss(self) -> None:
        own = torch.tensor([-0.2])
        swapped = torch.tensor([-2.0])
        loss, counted = pair_margin_loss(own, swapped, 0.5)
        self.assertEqual(counted, 1)
        self.assertEqual(loss.item(), 0.0)

    def test_identical_answer_pairs_excluded_with_count(self) -> None:
        own = torch.tensor([-1.0, -1.0])
        swapped = torch.tensor([5.0, 5.0])
        active = torch.tensor([False, True])
        loss, counted = pair_margin_loss(own, swapped, 0.5, active=active)
        self.assertEqual(counted, 1)
        self.assertAlmostEqual(loss.item(), 5.0 - (-1.0) + 0.5, places=5)

    def test_sequence_scores_normalization(self) -> None:
        logits = torch.zeros(2, 3, 4)
        labels = torch.tensor([[1, 2, 3], [1, -100, -100]])
        mask = torch.tensor([[True, True, True], [True, False, False]])
        mean_scores = sequence_logprob_scores(logits, labels, mask,
                                              normalization="eligible_token_mean")
        sum_scores = sequence_logprob_scores(logits, labels, mask,
                                             normalization="eligible_token_sum")
        self.assertAlmostEqual(mean_scores[0].item(), sum_scores[0].item() / 3, places=5)
        self.assertAlmostEqual(mean_scores[1].item(), sum_scores[1].item(), places=5)
        with self.assertRaises(ValueError):
            sequence_logprob_scores(logits, labels, mask, normalization="nonsense")


class ScheduleTests(unittest.TestCase):
    def test_fixed_schedule(self) -> None:
        schedule = make_schedule("fixed", base_lr=1e-3)
        self.assertEqual(schedule(0), 1e-3)
        self.assertEqual(schedule(100), 1e-3)

    def test_warmup_linear(self) -> None:
        schedule = make_schedule("warmup_linear", base_lr=1.0, warmup_updates=4)
        self.assertAlmostEqual(schedule(0), 0.25)
        self.assertAlmostEqual(schedule(3), 1.0)
        self.assertAlmostEqual(schedule(10), 1.0)

    def test_unknown_schedule_rejects(self) -> None:
        with self.assertRaises(ValueError):
            make_schedule("cosine", base_lr=1e-3)


class TrainerConstructionTests(unittest.TestCase):
    def test_trainer_builds_with_full_loss_defaults(self) -> None:
        from bramastra_lab.research.learning.trainer import Trainer
        from bramastra_lab.research.models import IntegratedModel

        seed_everything(3)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        model = IntegratedModel(config)
        trainer = Trainer(config, model)
        self.assertEqual(trainer.treatment, "full")
        self.assertEqual(trainer.pair_loss_weight, 0.0)
        self.assertEqual(trainer.counters.optimizer_updates, 0)
        with self.assertRaises(Exception):
            trainer.finalize_update()  # nothing accumulated


class TrainerUpdateTests(unittest.TestCase):
    """Real optimizer updates; owner-authorized learned smoke only."""

    @unittest.skipUnless(LEARNED_CHECKS, LEARNED_REASON)
    def test_accumulated_update_matches_reference_global_batch(self) -> None:
        from bramastra_lab.research.learning.trainer import Trainer
        from bramastra_lab.research.models import IntegratedModel

        batch_a = make_batch("4", "2+2?", provenance_suffix="a")
        batch_b = make_batch("6", "3+3?", provenance_suffix="b")
        merged_labels = torch.cat([batch_a.labels, batch_b.labels], dim=0)
        merged_mask = torch.cat([batch_a.loss_mask, batch_b.loss_mask], dim=0)
        merged_inputs = torch.cat([batch_a.input_ids, batch_b.input_ids], dim=0)
        merged_padding = torch.cat([batch_a.padding_mask, batch_b.padding_mask], dim=0)
        merged_segments = torch.cat([batch_a.segment_ids, batch_b.segment_ids], dim=0)
        from bramastra_lab.research.experience.sequences import CollocatedBatch

        global_batch = CollocatedBatch(
            input_ids=merged_inputs, padding_mask=merged_padding, labels=merged_labels,
            loss_mask=merged_mask, segment_ids=merged_segments,
            target_count=batch_a.target_count + batch_b.target_count,
            pair_group_ids=(None, None), provenance=batch_a.provenance + batch_b.provenance,
            sidecar_identity=batch_a.sidecar_identity)

        seed_everything(17)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"},
                                        "training": {"learning_rate": 0.01}})
        reference_trainer = Trainer(config, IntegratedModel(config))
        reference_trainer.training_step(global_batch)

        seed_everything(17)
        accumulated_trainer = Trainer(config, IntegratedModel(config))
        accumulated_trainer.accumulate(batch_a)
        accumulated_trainer.accumulate(batch_b)
        accumulated_trainer.finalize_update()

        for (name_a, tensor_a), (name_b, tensor_b) in zip(
                reference_trainer.model.state_dict().items(),
                accumulated_trainer.model.state_dict().items()):
            self.assertEqual(name_a, name_b)
            # Declared fp32 reduction-order tolerance: the reference reduces
            # one global mean; accumulation sums micro sums and divides at
            # the boundary, a different summation order over identical math.
            torch.testing.assert_close(tensor_a, tensor_b, rtol=1e-4, atol=1e-5)
        self.assertEqual(reference_trainer.counters.optimizer_updates, 1)
        self.assertEqual(accumulated_trainer.counters.supervised_targets_seen,
                         global_batch.target_count)
        self.assertEqual(accumulated_trainer.counters.presentations, 2)

    @unittest.skipUnless(LEARNED_CHECKS, LEARNED_REASON)
    def test_treatment_training_changes_only_training_loss_path(self) -> None:
        from bramastra_lab.research.learning.trainer import (
            Trainer,
            TrainerDiagnostics,
        )
        from bramastra_lab.research.models import IntegratedModel
        batch = make_batch("4", "2+2?")
        schema = TreatmentSchema(schema_id="plus-minus/v1", participating=(257, 52, 53))
        config = BuildConfig.from_dict({
            "model": {"profile": "tiny"},
            "training": {"logit_treatment": "participating_mask"}})
        seed_everything(23)
        trainer = Trainer(config, IntegratedModel(config))
        trainer.set_schema(schema)
        trainer.set_diagnostics(TrainerDiagnostics(trainer.model, make_batch("6", "3+3?",
                                                                             provenance_suffix="d")))
        report = trainer.training_step(batch)
        self.assertEqual(report.optimizer_update, 1)
        self.assertIn("post_clip_grad_norm", report.telemetry)
        # Evaluation path still sees full logits (fresh forward).
        with torch.no_grad():
            output = trainer.model(batch.input_ids, batch.padding_mask,
                                   segment_ids=batch.segment_ids)
        self.assertEqual(output.logits.shape[-1], config.model.vocab)

    @unittest.skipUnless(LEARNED_CHECKS, LEARNED_REASON)
    def test_instrumentation_does_not_change_weights_or_optimizer_state(self) -> None:
        from bramastra_lab.research.learning.trainer import Trainer, TrainerDiagnostics
        from bramastra_lab.research.models import IntegratedModel

        batch = make_batch("4", "2+2?")
        diagnostic_batch = make_batch("6", "3+3?", provenance_suffix="d")
        seed_everything(31)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        trainer = Trainer(config, IntegratedModel(config))
        trainer.set_diagnostics(TrainerDiagnostics(trainer.model, diagnostic_batch))
        trainer.accumulate(batch)  # gradients exist, no update performed
        grads_before = {name: parameter.grad.clone() for name, parameter in
                        trainer.model.named_parameters() if parameter.grad is not None}
        optimizer_before = repr(sorted(trainer.optimizer.state_dict()["state"].keys()))
        rng_before = torch.get_rng_state()
        weights_before = {name: tensor.clone() for name, tensor in
                          trainer.model.state_dict().items()}

        telemetry = trainer.diagnostics.collect()

        for name, tensor in trainer.model.state_dict().items():
            torch.testing.assert_close(weights_before[name], tensor)
        for name, tensor in grads_before.items():
            tensor_grad = dict(trainer.model.named_parameters())[name].grad
            self.assertIsNotNone(tensor_grad)
            torch.testing.assert_close(tensor, tensor_grad)
        self.assertEqual(repr(sorted(trainer.optimizer.state_dict()["state"].keys())),
                         optimizer_before)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertIn("diagnostic_target_prob_mean", telemetry)
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer._pending_targets = 0  # test cleanup; no update was performed

    @unittest.skipUnless(LEARNED_CHECKS, LEARNED_REASON)
    def test_pair_input_with_zero_weight_rejects(self) -> None:
        from bramastra_lab.research.learning.trainer import Trainer, PairUpdateInput
        from bramastra_lab.research.models import IntegratedModel

        batch = make_batch("4", "2+2?")
        seed_everything(41)
        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        trainer = Trainer(config, IntegratedModel(config))
        trainer.accumulate(batch)
        with self.assertRaises(Exception):
            trainer.finalize_update(pair_input=PairUpdateInput(own=batch, swapped=batch))


if __name__ == "__main__":
    unittest.main()
