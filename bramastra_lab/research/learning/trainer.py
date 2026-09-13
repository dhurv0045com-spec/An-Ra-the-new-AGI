"""The real integrated trainer (B04).

One canonical update path: accumulate answer/EOS sum-losses over micro
batches, normalize by the actual global supervised-target denominator at the
accumulation boundary, add the optional pair-margin term, check finiteness,
clip with a documented dtype tolerance certificate, step once, advance the
schedule and counters once, and only then run instrumentation. The plasticity
controller (B06) multiplies the learning rate at optimizer boundaries and
never touches tensors itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from bramastra_lab.research.config import BuildConfig
from bramastra_lab.research.learning.objectives import (
    answer_eos_loss_sum,
    pair_margin_loss,
    sequence_logprob_scores,
)
from bramastra_lab.research.learning.schedules import make_schedule
from bramastra_lab.research.learning.treatments import (
    TreatmentSchema,
    apply_treatment,
    validate_targets_in_schema,
)
from bramastra_lab.research.models import IntegratedModel

# Post-clip certificate tolerance, following the R1C lesson: a numerical
# identity check may reject a valid update when its tolerance is tighter than
# the dtype/operation error floor. The certificate allows a 1e-4 relative
# excess and rejects genuine breaches and nonfinite values.
CLIP_CERTIFICATE_TOLERANCE = 1e-4
# Parameter-displacement telemetry snapshots weights only for small models.
DISPLACEMENT_SNAPSHOT_MAX_PARAMETERS = 2_000_000


class TrainerStateError(RuntimeError):
    """The trainer was driven against its declared update-boundary rules."""


def _total_grad_norm(parameters) -> float:
    """Total L2 norm over gradients, reduced in float32."""
    norms = [parameter.grad.norm(dtype=torch.float32) for parameter in parameters
             if parameter.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms)).item())


@dataclass
class TrainerCounters:
    optimizer_updates: int = 0
    presentations: int = 0
    supervised_targets_seen: int = 0
    encoded_tokens_seen: int = 0
    real_interactions: int = 0
    attempted_batches: int = 0
    microbatches: int = 0
    replay_entries_consumed: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, raw: Mapping[str, int]) -> "TrainerCounters":
        counters = cls()
        for key in counters.__dict__:
            value = raw.get(key, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TrainerStateError(f"counter {key} must be a nonnegative integer")
            setattr(counters, key, value)
        return counters


@dataclass
class PairUpdateInput:
    """Two collocated batches of equal shape: pairs scored under their own
    prompt and under the swapped answer. Identical-answer pairs were already
    excluded at rendering and appear in neither batch."""

    own: Any            # CollocatedBatch (prompt_i + answer_i)
    swapped: Any        # CollocatedBatch (prompt_i + answer_other_i)


@dataclass
class StepReport:
    optimizer_update: int
    answer_loss_mean: float
    pair_loss: float | None
    lr: float
    grad_norm: float
    clipped: bool
    supervised_targets: int
    presentations: int
    telemetry: dict[str, Any] = field(default_factory=dict)


class Trainer:
    """Drives the IntegratedModel through real optimizer updates."""

    def __init__(self, config: BuildConfig, model: IntegratedModel, *,
                 device: str | None = None) -> None:
        self.config = config
        self.model = model
        self.device = device or "cpu"
        self.model.to(self.device)
        training = config.training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=training.learning_rate,
            weight_decay=training.weight_decay)
        self._schedule = make_schedule(
            "warmup_linear" if training.warmup_updates else "fixed",
            base_lr=training.learning_rate, warmup_updates=training.warmup_updates)
        self.treatment = training.logit_treatment
        self.effective_vocab = training.effective_vocab
        self.pair_loss_weight = training.pair_loss_weight
        self.pair_margin = training.pair_margin
        self.pair_score_normalization = training.pair_score_normalization
        self.clip_norm = training.clip_norm
        self.grad_accum_steps = training.grad_accum_steps
        self.controller_multiplier = 1.0
        self.controller_reason: str | None = None
        self.counters = TrainerCounters()
        self._pending_targets = 0
        self._pending_answer_sum = 0.0
        self._pending_presentations = 0
        self._pending_pair_own: list = []
        self._pending_pair_swapped: list = []
        self.schema: TreatmentSchema | None = None
        self.diagnostics: TrainerDiagnostics | None = None

    # -- configuration --------------------------------------------------------

    def set_schema(self, schema: TreatmentSchema | None) -> None:
        if schema is not None and self.treatment != "full":
            schema.validated_for_vocab(self.config.model.vocab)
        self.schema = schema

    def set_controller_multiplier(self, multiplier: float, reason: str | None) -> None:
        """Called by the plasticity controller at optimizer boundaries only."""
        if not isinstance(multiplier, (int, float)) or not 0.0 < float(multiplier) <= 4.0:
            raise TrainerStateError("controller multiplier must be in (0, 4]")
        self.controller_multiplier = float(multiplier)
        self.controller_reason = reason

    def set_diagnostics(self, diagnostics: "TrainerDiagnostics") -> None:
        self.diagnostics = diagnostics

    def record_real_interactions(self, count: int) -> None:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise TrainerStateError("interaction count must be a nonnegative integer")
        self.counters.real_interactions += count

    # -- accumulation ---------------------------------------------------------

    def _treated_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.treatment == "full":
            return logits
        if self.schema is None:
            raise TrainerStateError(
                f"treatment {self.treatment!r} requires a declared TreatmentSchema")
        return apply_treatment(logits, self.schema, self.treatment,
                               effective_vocab=self.effective_vocab)

    def accumulate(self, batch, *, pair_rows: tuple[list, list] | None = None) -> dict[str, float]:
        """Backward one micro batch's unnormalized sum loss.

        Gradients accumulate raw; the global supervised-target denominator is
        applied once at the accumulation boundary. Pair renderings supplied
        with a micro batch are pooled across the complete accumulation group
        and scored together at ``finalize_update`` (B2.2 R5), so the pair
        objective is defined over all eligible pairs of the group, not per
        invocation.
        """
        if self._pending_targets and batch.target_count == 0:
            raise TrainerStateError("micro batch declares zero supervised targets")
        if self.treatment != "full" and self.schema is not None:
            validate_targets_in_schema(batch.labels.to(self.device),
                                       batch.loss_mask.to(self.device), self.schema)
        self.model.train()
        output = self.model(batch.input_ids.to(self.device),
                            batch.padding_mask.to(self.device),
                            segment_ids=batch.segment_ids.to(self.device))
        treated = self._treated_logits(output.logits)
        report = answer_eos_loss_sum(treated, batch.labels.to(self.device),
                                     batch.loss_mask.to(self.device))
        if report.target_count != batch.target_count:
            raise TrainerStateError(
                f"batch declares {batch.target_count} targets but loss sees "
                f"{report.target_count}")
        report.total.backward()
        if pair_rows is not None:
            own_rows, swapped_rows = pair_rows
            if len(own_rows) != len(swapped_rows):
                raise TrainerStateError("pair renderings must align within a micro batch")
            self._pending_pair_own.extend(own_rows)
            self._pending_pair_swapped.extend(swapped_rows)
        self._pending_targets += report.target_count
        self._pending_answer_sum += float(report.total.detach().item())
        self._pending_presentations += int(batch.batch_size)
        self.counters.microbatches += 1
        self.counters.presentations += int(batch.batch_size)
        self.counters.supervised_targets_seen += report.target_count
        self.counters.encoded_tokens_seen += int(batch.padding_mask.sum().item())
        return {"answer_loss_sum": self._pending_answer_sum,
                "pending_targets": float(self._pending_targets)}

    def _apply_pair_term(self, pair_input: PairUpdateInput) -> float:
        own_output = self.model(pair_input.own.input_ids.to(self.device),
                                pair_input.own.padding_mask.to(self.device),
                                segment_ids=pair_input.own.segment_ids.to(self.device))
        swapped_output = self.model(pair_input.swapped.input_ids.to(self.device),
                                    pair_input.swapped.padding_mask.to(self.device),
                                    segment_ids=pair_input.swapped.segment_ids.to(self.device))
        own_scores = sequence_logprob_scores(
            own_output.logits, pair_input.own.labels.to(self.device),
            pair_input.own.loss_mask.to(self.device),
            normalization=self.pair_score_normalization)
        swapped_scores = sequence_logprob_scores(
            swapped_output.logits, pair_input.swapped.labels.to(self.device),
            pair_input.swapped.loss_mask.to(self.device),
            normalization=self.pair_score_normalization)
        loss_pair, counted = pair_margin_loss(swapped_scores, own_scores, self.pair_margin)
        if counted == 0:
            return 0.0
        (self.pair_loss_weight * loss_pair).backward()
        return float(loss_pair.detach().item())

    # -- update boundary ------------------------------------------------------

    def finalize_update(self) -> StepReport:
        """Complete one optimizer update from accumulated micro batches.

        Reduction happens exactly once, here: accumulated raw gradients are
        divided by the true global target count; the optional pair term is
        pooled across the complete accumulation group and added at this
        boundary; then finiteness, clipping, one step, one schedule advance
        and one counter increment.
        """
        if self._pending_targets <= 0:
            raise TrainerStateError("finalize_update called with no accumulated batches")
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(self._pending_targets)
        pair_loss_value: float | None = None
        if self._pending_pair_own or self._pending_pair_swapped:
            if self.pair_loss_weight <= 0:
                raise TrainerStateError(
                    "pair renderings accumulated while training.pair_loss_weight is zero")
            if len(self._pending_pair_own) != len(self._pending_pair_swapped):
                raise TrainerStateError(
                    "pair renderings must align across the accumulation group")
            from bramastra_lab.research.experience.sequences import collocate as _collocate

            max_seq = self.config.model.max_seq
            own_batch = _collocate(list(self._pending_pair_own), max_seq=max_seq)
            swapped_batch = _collocate(list(self._pending_pair_swapped), max_seq=max_seq)
            pair_loss_value = self._apply_pair_term(
                PairUpdateInput(own=own_batch, swapped=swapped_batch))
            self._pending_pair_own = []
            self._pending_pair_swapped = []

        nonfinite = [name for name, parameter in self.model.named_parameters()
                     if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
        if nonfinite:
            raise TrainerStateError(f"nonfinite gradients in {nonfinite[:4]}; update refused")

        pre_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm))
        clipped = pre_norm > self.clip_norm
        post_norm = _total_grad_norm(self.model.parameters())
        if not post_norm <= self.clip_norm * (1.0 + CLIP_CERTIFICATE_TOLERANCE):
            raise TrainerStateError(
                f"clip certificate violated: post-clip norm {post_norm} exceeds clip_norm "
                f"{self.clip_norm} beyond tolerance {CLIP_CERTIFICATE_TOLERANCE}")

        lr = self._schedule(self.counters.optimizer_updates) * self.controller_multiplier
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        snapshot = None
        if (self.diagnostics is not None and self.diagnostics.track_displacement
                and sum(parameter.numel() for parameter in self.model.parameters())
                <= DISPLACEMENT_SNAPSHOT_MAX_PARAMETERS):
            snapshot = {name: parameter.detach().clone()
                        for name, parameter in self.model.named_parameters()}

        self.optimizer.step()

        self.counters.optimizer_updates += 1
        report = StepReport(
            optimizer_update=self.counters.optimizer_updates,
            answer_loss_mean=self._pending_answer_sum / self._pending_targets,
            pair_loss=pair_loss_value,
            lr=lr,
            grad_norm=pre_norm,
            clipped=clipped,
            supervised_targets=self._pending_targets,
            presentations=self._pending_presentations,
        )
        if self.diagnostics is not None:
            report.telemetry = self.diagnostics.collect(snapshot=snapshot)
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_targets = 0
        self._pending_answer_sum = 0.0
        self._pending_presentations = 0
        return report

    def training_step(self, batch, *, pair_rows: tuple[list, list] | None = None) -> StepReport:
        """One optimizer update from a single micro batch.

        Only valid while ``grad_accum_steps == 1``: the configured window is
        the public contract, and the CLI loop implements it. A configured
        multi-microbatch window rejects this convenience method instead of
        silently disagreeing with the CLI (B2.2 chief F5).
        """
        if self.grad_accum_steps != 1:
            raise TrainerStateError(
                f"training_step is a single-microbatch convenience and this config "
                f"declares grad_accum_steps={self.grad_accum_steps}; use the "
                "accumulate/finalize_update window boundary")
        self.accumulate(batch, pair_rows=pair_rows)
        return self.finalize_update()

    # -- checkpoint payload ---------------------------------------------------

    def state_payload(self) -> dict[str, Any]:
        if self._pending_targets:
            raise TrainerStateError(
                "checkpoints exist only at update boundaries; "
                f"{self._pending_targets} targets are still accumulated")
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "counters": self.counters.to_dict(),
            "schedule": {"warmup_updates": self.config.training.warmup_updates,
                         "base_lr": self.config.training.learning_rate},
            "controller_multiplier": self.controller_multiplier,
            "controller_reason": self.controller_reason,
        }

    def load_state_payload(self, payload: Mapping[str, Any]) -> None:
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.counters = TrainerCounters.from_dict(payload["counters"])
        self.controller_multiplier = float(payload["controller_multiplier"])
        self.controller_reason = payload["controller_reason"]
        self._pending_targets = 0
        self._pending_answer_sum = 0.0
        self._pending_presentations = 0
        self._pending_pair_own = []
        self._pending_pair_swapped = []


class TrainerDiagnostics:
    """Training-only telemetry on a fixed diagnostic batch.

    Runs after the optimizer step, reads gradients before they are cleared,
    and consumes no RNG. It never alters weights, gradients or optimizer
    state. Counterfactual gradients, when requested, use ``torch.autograd.grad``
    so training gradients are untouched.
    """

    def __init__(self, model: IntegratedModel, diagnostic_batch, *,
                 schema: TreatmentSchema | None = None,
                 diagnostic_pair_input: PairUpdateInput | None = None,
                 track_displacement: bool = True) -> None:
        self.model = model
        self.diagnostic_batch = diagnostic_batch
        self.schema = schema
        self.diagnostic_pair_input = diagnostic_pair_input
        self.track_displacement = track_displacement
        self.last_telemetry: dict[str, Any] = {}

    @torch.no_grad()
    def _forward_metrics(self) -> dict[str, Any]:
        batch = self.diagnostic_batch
        output = self.model(batch.input_ids, batch.padding_mask,
                            segment_ids=batch.segment_ids, return_hidden=True)
        logits = output.logits.float()
        probabilities = torch.softmax(logits, dim=-1)
        supervised = batch.loss_mask
        target_positions = probabilities[supervised]
        target_labels = batch.labels[batch.loss_mask]
        target_prob = target_positions.gather(
            1, target_labels.clamp_min(0).unsqueeze(1)).squeeze(1)
        masked_probs = target_positions.clone()
        masked_probs.scatter_(1, target_labels.clamp_min(0).unsqueeze(1), 0.0)
        wrong_max = masked_probs.max(dim=-1).values
        entropy = -(target_positions.clamp_min(1e-12).log()
                    * target_positions).sum(dim=-1)
        finite = torch.isfinite(target_prob)
        telemetry = {
            "diagnostic_target_prob_mean": float(target_prob[finite].mean().item()) if bool(finite.any()) else 0.0,
            "diagnostic_wrong_max_mean": float(wrong_max[finite].mean().item()) if bool(finite.any()) else 0.0,
            "diagnostic_margin_mean": float((target_prob - wrong_max)[finite].mean().item()) if bool(finite.any()) else 0.0,
            "diagnostic_entropy_mean": float(entropy[finite].mean().item()) if bool(finite.any()) else 0.0,
        }
        # Active/inactive competition split over supervised positions: with a
        # declared schema the participating classes are "active"; without one,
        # the full vocabulary is active and inactive mass is zero by definition
        # (R1C dense diagnostics, dense-mechanism list).
        if self.schema is not None and self.schema.participating:
            active_ids = torch.tensor(sorted(self.schema.participating), dtype=torch.long,
                                      device=probabilities.device)
            active_mass = target_positions[:, active_ids].sum(dim=-1)
            inactive_mass = 1.0 - active_mass
            inactive_probs = target_positions.clone()
            inactive_probs[:, active_ids] = 0.0
            max_inactive_prob = inactive_probs.max(dim=-1).values
            telemetry.update({
                "diagnostic_active_mass_mean": float(active_mass[finite].mean().item()) if bool(finite.any()) else 0.0,
                "diagnostic_inactive_mass_mean": float(inactive_mass[finite].mean().item()) if bool(finite.any()) else 0.0,
                "diagnostic_max_inactive_prob_mean": float(max_inactive_prob[finite].mean().item()) if bool(finite.any()) else 0.0,
            })
        hidden_norms = output.hidden.float().pow(2).sum(dim=-1).sqrt() \
            if output.hidden is not None else None
        if hidden_norms is not None:
            telemetry["diagnostic_hidden_l2_mean"] = float(hidden_norms.mean().item())
        return telemetry

    def collect(self, *, snapshot: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
        telemetry = self._forward_metrics()
        grads = [parameter.grad for parameter in self.model.parameters()
                 if parameter.grad is not None]
        if grads:
            stacked_norm = torch.norm(torch.stack(
                [grad.norm(dtype=torch.float32) for grad in grads]))
            telemetry["post_clip_grad_norm"] = float(stacked_norm.item())
        embedding_norms = [parameter.grad.norm(dtype=torch.float32).item()
                           for name, parameter in self.model.named_parameters()
                           if parameter.grad is not None and "embedding" in name]
        if embedding_norms:
            telemetry["embedding_grad_norm"] = float(torch.tensor(embedding_norms).max().item())
        if snapshot is not None:
            squared = sum(
                float((parameter.detach() - snapshot[name]).float().pow(2).sum().item())
                for name, parameter in self.model.named_parameters() if name in snapshot)
            displacement = squared ** 0.5
            telemetry["parameter_displacement_l2"] = displacement
            reference_norm = sum(
                float(snapshot[name].float().pow(2).sum().item())
                for name in snapshot)
            if reference_norm > 0.0:
                # Relative displacement (ARK-007R: ~0.008 LOW vs ~0.379 HIGH);
                # near-freezing must be visible as near-zero relative movement.
                telemetry["parameter_displacement_relative"] = \
                    displacement / (reference_norm ** 0.5)
        if self.diagnostic_pair_input is not None and self.schema is not None:
            cosine = self._counterfactual_gradient_cosine()
            if cosine is not None:
                telemetry["counterfactual_gradient_cosine"] = cosine
        self.last_telemetry = telemetry
        return telemetry

    def _counterfactual_gradient_cosine(self) -> float | None:
        """Cosine between the pair-loss gradient and the full-loss gradient.

        Both are computed with ``torch.autograd.grad`` on the same frozen
        model state and diagnostic batch, so neither enters an optimizer step
        (R1C: prove what the treatment does to update direction).
        """
        pair_input = self.diagnostic_pair_input
        was_mode = self.model.training
        try:
            self.model.train()
            own_output = self.model(pair_input.own.input_ids, pair_input.own.padding_mask,
                                    segment_ids=pair_input.own.segment_ids)
            swapped_output = self.model(pair_input.swapped.input_ids,
                                        pair_input.swapped.padding_mask,
                                        segment_ids=pair_input.swapped.segment_ids)
            own_scores = sequence_logprob_scores(
                own_output.logits, pair_input.own.labels, pair_input.own.loss_mask)
            swapped_scores = sequence_logprob_scores(
                swapped_output.logits, pair_input.swapped.labels,
                pair_input.swapped.loss_mask)
            loss_pair, _ = pair_margin_loss(swapped_scores, own_scores, 1.0)
            parameters = [parameter for parameter in self.model.parameters()
                          if parameter.requires_grad]
            grads = torch.autograd.grad(
                loss_pair, parameters, allow_unused=True, materialize_grads=False)

            batch = self.diagnostic_batch
            output = self.model(batch.input_ids, batch.padding_mask,
                                segment_ids=batch.segment_ids)
            report = answer_eos_loss_sum(output.logits, batch.labels, batch.loss_mask)
            full_grads = torch.autograd.grad(
                report.total, parameters, allow_unused=True, materialize_grads=False)

            def flatten(tensors):
                return torch.cat([tensor.detach().reshape(-1).float()
                                  for tensor in tensors if tensor is not None])
            pair_flat, full_flat = flatten(grads), flatten(full_grads)
            if pair_flat.numel() == 0 or full_flat.numel() == 0:
                return None
            denominator = pair_flat.norm() * full_flat.norm()
            if float(denominator.item()) <= 0.0:
                return None
            return float((pair_flat @ full_flat / denominator).item())
        finally:
            self.model.train(was_mode)
