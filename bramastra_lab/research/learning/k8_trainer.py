"""K8 campaign trainer (I01): extends the canonical trainer with CUDA
device/precision routing, AMP unscale/clip/step, skipped-update accounting
and allocation-aware update admission.

This is the existing trainer extended — never a notebook-only optimizer.
All CPU paths keep working (device="cpu" disables AMP entirely).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from bramastra_lab.research.learning.trainer import (
    CLIP_CERTIFICATE_TOLERANCE,
    DISPLACEMENT_SNAPSHOT_MAX_PARAMETERS,
    StepReport,
    Trainer,
    TrainerStateError,
    _total_grad_norm,
    answer_eos_loss_sum,
    validate_targets_in_schema,
)
from bramastra_lab.research.experience.supervision import SupervisionWindow
from bramastra_lab.research.learning.router import route_window


@dataclass(frozen=True)
class AllocationContext:
    """The active campaign allocation; updates are admitted only when the
    context validates (B2.2/K8 I01: allocation-aware runtime)."""

    allocation_id: str
    device: str
    deadline_unix: float
    remaining_updates: int
    job_id: str = "local"
    phase: str = "local"

    def __post_init__(self) -> None:
        if not isinstance(self.allocation_id, str) or not self.allocation_id:
            raise TrainerStateError("allocation_id must be a nonempty string")
        if not isinstance(self.remaining_updates, int) or isinstance(
                self.remaining_updates, bool) or self.remaining_updates < 0:
            raise TrainerStateError("remaining_updates must be a nonnegative integer")


class K8Trainer(Trainer):
    """The canonical trainer with K8 campaign semantics:

    - device/precision propagated to model, batches, optimizer restore,
      diagnostics, masks and targets (AMP fp16 autocast + GradScaler on CUDA);
    - AMP unscale happens before clipping; the scaler state is checkpointed;
    - a skipped nonfinite step does not increment successful optimizer
      updates or consume a schedule step, but its time/exposure stays charged
      through the attempted/microbatch counters;
    - updates are admitted only against a live AllocationContext.
    """

    def __init__(self, config, model, *, device: str | None = None,
                 allocation: AllocationContext | None = None,
                 precision: str = "fp32",
                 require_allocation: bool = False) -> None:
        super().__init__(config, model, device=device)
        if precision not in ("fp32", "fp16_autocast"):
            raise TrainerStateError(f"unsupported precision {precision!r}")
        self.precision = precision
        self.allocation = allocation
        self.require_allocation = bool(require_allocation)
        self.use_amp = precision == "fp16_autocast" and str(self.device).startswith("cuda")
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp) if hasattr(torch, "amp") else \
            torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.skipped_updates = 0
        self.attempted_updates = 0
        self._pending_normalized = False
        self._last_route_report = {}

    # -- allocation admission -------------------------------------------------

    def _admit_update(self) -> None:
        if self.allocation is None:
            if self.require_allocation:
                raise TrainerStateError(
                    "campaign update refused: no live AllocationContext bound "
                    "(missing allocation is never admitted in campaign mode)")
            return  # local/non-campaign mode: no allocation gate
        import time

        if self.counters.optimizer_updates >= self.allocation.remaining_updates \
                + self._updates_at_start:
            raise TrainerStateError(
                f"allocation {self.allocation.allocation_id!r} exhausted its "
                f"{self.allocation.remaining_updates}-update reservation for job "
                f"{self.allocation.job_id!r}")
        if time.time() > self.allocation.deadline_unix:
            raise TrainerStateError(
                f"allocation {self.allocation.allocation_id!r} deadline passed")

    @property
    def _updates_at_start(self) -> int:
        return getattr(self, "_k8_updates_at_start", 0)

    def begin_campaign(self, allocation: AllocationContext) -> None:
        self.allocation = allocation
        self._k8_updates_at_start = self.counters.optimizer_updates

    # -- AMP-aware accumulation and finalize ----------------------------------

    def accumulate(self, batch, *, pair_rows=None) -> dict[str, float]:
        """Backward one micro batch under the configured precision.

        The batch tensors are moved to the trainer device first; the loss is
        scaled by the GradScaler on CUDA so AMP unscale happens before the
        boundary clip. The token term is routed through route_window with its
        own denominator (R04: no answer-only bypass); pair_rows are pooled
        across the accumulation group for the boundary pair term.
        """
        if self._pending_targets and batch.target_count == 0:
            raise TrainerStateError("micro batch declares zero supervised targets")
        if self.treatment != "full" and self.schema is not None:
            validate_targets_in_schema(batch.labels.to(self.device),
                                       batch.loss_mask.to(self.device), self.schema)
        self.model.train()
        move = lambda tensor: tensor.to(self.device)
        with torch.autocast(device_type=self._amp_device_type(), enabled=self.use_amp):
            output = self.model(move(batch.input_ids), move(batch.padding_mask),
                                segment_ids=move(batch.segment_ids))
            treated = self._treated_logits(output.logits)
            report = answer_eos_loss_sum(treated, move(batch.labels),
                                         move(batch.loss_mask))
        if report.target_count != batch.target_count:
            raise TrainerStateError(
                f"batch declares {batch.target_count} targets but loss sees "
                f"{report.target_count}")
        # Route the token term through the canonical window boundary so the
        # router is consumed (not merely imported): single-term window with
        # its own denominator, weight 1.0.
        from bramastra_lab.research.experience.supervision import SupervisionWindow
        from bramastra_lab.research.learning.router import route_window

        window = SupervisionWindow(
            weights={"token": 1.0, "world": 0.0, "action": 0.0,
                     "value": 0.0, "pair": 0.0, "pg": 0.0},
            enabled_terms=frozenset({"token"}))
        window.add("token", report.target_count)
        combined, _route_report = route_window(window, {"token": report.total})
        # Single scaled backward for this microbatch (AMP-consistent).
        scaled = self.scaler.scale(combined) if self.use_amp else combined
        scaled.backward()
        if pair_rows is not None:
            own_rows, swapped_rows = pair_rows
            if len(own_rows) != len(swapped_rows):
                raise TrainerStateError("pair renderings must align within a micro batch")
            self._pending_pair_own.extend(own_rows)
            self._pending_pair_swapped.extend(swapped_rows)
        self._pending_targets += report.target_count
        self._pending_answer_sum += float(report.total.detach().item())
        self._pending_presentations += int(batch.batch_size)
        self._pending_normalized = False
        self.counters.microbatches += 1
        self.counters.presentations += int(batch.batch_size)
        self.counters.supervised_targets_seen += report.target_count
        self.counters.encoded_tokens_seen += int(batch.padding_mask.sum().item())
        return {"answer_loss_sum": self._pending_answer_sum,
                "pending_targets": float(self._pending_targets)}

    def accumulate_full_window(self, batch, *, window_builder=None,
                               extra_terms_fn=None,
                               pair_rows=None) -> dict[str, float]:
        """Canonical multi-objective accumulation boundary (R04/R05).

        Computes answer + differentiable world/action/value sums as LIVE
        tensors, builds ONE SupervisionWindow with per-term eligibility counts
        computed BEFORE backward, routes through route_window (per-term OWN
        denominators, never a single answer count), then performs ONE scaled
        backward. Both E0 branches call this exact function with the same
        treatment (identical objectives, no mixed scaled/unscaled gradients).

        window_builder(target_count) -> SupervisionWindow
        extra_terms_fn() -> {term: live Tensor sum} for non-token terms.
        """
        from bramastra_lab.research.experience.supervision import SupervisionWindow
        from bramastra_lab.research.learning.router import route_window

        if self._pending_targets and batch.target_count == 0:
            raise TrainerStateError("micro batch declares zero supervised targets")
        if self.treatment != "full" and self.schema is not None:
            validate_targets_in_schema(batch.labels.to(self.device),
                                       batch.loss_mask.to(self.device), self.schema)
        self.model.train()
        move = lambda tensor: tensor.to(self.device)
        with torch.autocast(device_type=self._amp_device_type(), enabled=self.use_amp):
            output = self.model(move(batch.input_ids), move(batch.padding_mask),
                                segment_ids=move(batch.segment_ids))
            treated = self._treated_logits(output.logits)
            report = answer_eos_loss_sum(treated, move(batch.labels),
                                         move(batch.loss_mask))
        if report.target_count != batch.target_count:
            raise TrainerStateError(
                f"batch declares {batch.target_count} targets but loss sees "
                f"{report.target_count}")
        # Window with per-term denominators BEFORE backward.
        if window_builder is not None:
            window = window_builder(report.target_count)
            if not isinstance(window, SupervisionWindow):
                raise TrainerStateError("window_builder must return a SupervisionWindow")
        else:
            window = SupervisionWindow(
                weights={"token": 1.0, "world": 0.0, "action": 0.0,
                         "value": 0.0, "pair": 0.0, "pg": 0.0},
                enabled_terms=frozenset({"token"}))
            window.add("token", report.target_count)
        sums: dict[str, Any] = {"token": report.total}
        if extra_terms_fn is not None:
            # Extra forwards must run under autocast for AMP consistency; they
            # reuse the already-moved model on its device. Re-enter autocast
            # so action/value/world paths share the precision policy.
            with torch.autocast(device_type=self._amp_device_type(),
                                enabled=self.use_amp):
                extra = extra_terms_fn()
            if not isinstance(extra, dict):
                raise TrainerStateError("extra_terms_fn must return a dict")
            for term, total in extra.items():
                if term == "token":
                    raise TrainerStateError("extra terms must not redefine 'token'")
                sums[term] = total
        combined, route_report = route_window(window, sums)
        # ONE scaled backward for the whole window (no mixed scaled/unscaled).
        scaled = self.scaler.scale(combined) if self.use_amp else combined
        scaled.backward()
        if pair_rows is not None:
            own_rows, swapped_rows = pair_rows
            if len(own_rows) != len(swapped_rows):
                raise TrainerStateError("pair renderings must align within a micro batch")
            self._pending_pair_own.extend(own_rows)
            self._pending_pair_swapped.extend(swapped_rows)
        self._pending_targets += report.target_count
        self._pending_answer_sum += float(report.total.detach().item())
        self._pending_presentations += int(batch.batch_size)
        # Mark normalized: finalize must NOT re-divide by answer count because
        # each term already carries its own denominator (R04).
        self._pending_normalized = True
        self._last_route_report = route_report
        self.counters.microbatches += 1
        self.counters.presentations += int(batch.batch_size)
        self.counters.supervised_targets_seen += report.target_count
        self.counters.encoded_tokens_seen += int(batch.padding_mask.sum().item())
        return {"answer_loss_sum": self._pending_answer_sum,
                "pending_targets": float(self._pending_targets),
                "route_applied": route_report.get("applied", {})}

    def _amp_device_type(self) -> str:
        device = str(self.device)
        if device.startswith("cuda"):
            return "cuda"
        if device.startswith("xla"):
            return "xla"
        return "cpu"

    def finalize_update(self) -> StepReport:
        """One optimizer boundary: unscale (AMP), check finiteness, clip,
        step once. A nonfinite skip charges time/exposure but does not
        increment successful optimizer updates or the schedule.

        Per-term denominators were already applied at accumulation via
        route_window (R04); this boundary does NOT re-divide already-
        normalized gradients by the answer count. Legacy answer-only
        accumulation (pending_normalized False) retains the single-denominator
        division for backwards compatibility. A failed report publication
        after a committed step never erases the step (counters increment
        before report construction).
        """
        self._admit_update()
        self.attempted_updates += 1
        if self._pending_targets <= 0:
            raise TrainerStateError("finalize_update called with no accumulated batches")
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        normalized = bool(getattr(self, "_pending_normalized", False))
        if not normalized:
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        # Skip the update: charge exposure, keep counters honest.
                        self.skipped_updates += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        self._pending_targets = 0
                        self._pending_answer_sum = 0.0
                        self._pending_presentations = 0
                        self._pending_pair_own = []
                        self._pending_pair_swapped = []
                        raise TrainerStateError(
                            "nonfinite gradients after unscale; update skipped and "
                            "charged (attempted but not committed)")
                    parameter.grad.div_(self._pending_targets)
        else:
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        self.skipped_updates += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        self._pending_targets = 0
                        self._pending_answer_sum = 0.0
                        self._pending_presentations = 0
                        self._pending_pair_own = []
                        self._pending_pair_swapped = []
                        self._pending_normalized = False
                        raise TrainerStateError(
                            "nonfinite gradients after unscale; update skipped and "
                            "charged (attempted but not committed)")
                    # No re-division: per-term denominators already applied.
        pre_norm = float(torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.clip_norm))
        clipped = pre_norm > self.clip_norm
        post_norm = _total_grad_norm(self.model.parameters())
        if not post_norm <= self.clip_norm * (1.0 + CLIP_CERTIFICATE_TOLERANCE):
            raise TrainerStateError(
                f"clip certificate violated: post-clip norm {post_norm}")
        snapshot = None
        if self.diagnostics is not None and self.diagnostics.track_displacement \
                and sum(parameter.numel() for parameter in self.model.parameters()) \
                <= DISPLACEMENT_SNAPSHOT_MAX_PARAMETERS:
            snapshot = {name: parameter.detach().clone()
                        for name, parameter in self.model.named_parameters()}
        # Apply the scheduled LR before stepping (R04: actual LR application).
        scheduled_lr = self._schedule(self.counters.optimizer_updates)             * self.controller_multiplier
        for group in self.optimizer.param_groups:
            group["lr"] = scheduled_lr
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.counters.optimizer_updates += 1
        report = StepReport(
            optimizer_update=self.counters.optimizer_updates,
            answer_loss_mean=self._pending_answer_sum / self._pending_targets,
            pair_loss=None, lr=scheduled_lr,
            grad_norm=pre_norm, clipped=clipped,
            supervised_targets=self._pending_targets,
            presentations=self._pending_presentations)
        if self.diagnostics is not None:
            report.telemetry = self.diagnostics.collect(snapshot=snapshot)
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_targets = 0
        self._pending_answer_sum = 0.0
        self._pending_presentations = 0
        self._pending_normalized = False
        self._last_route_report = {}
        return report

    # -- scaler state in checkpoints ------------------------------------------

    def state_payload(self) -> dict[str, Any]:
        payload = super().state_payload()
        payload["precision"] = self.precision
        payload["use_amp"] = self.use_amp
        payload["scaler_state"] = self.scaler.state_dict() if self.use_amp else None
        payload["skipped_updates"] = self.skipped_updates
        payload["attempted_updates"] = int(getattr(self, "attempted_updates", 0))
        payload["allocation"] = ({"allocation_id": self.allocation.allocation_id,
                                  "job_id": self.allocation.job_id,
                                  "phase": self.allocation.phase}
                                 if self.allocation else None)
        return payload

    def load_state_payload(self, payload: Mapping[str, Any]) -> None:
        super().load_state_payload(payload)
        self._k8_updates_at_start = self.counters.optimizer_updates
        if payload.get("use_amp") and not self.use_amp:
            raise TrainerStateError(
                "checkpoint was trained with AMP on CUDA but this trainer is not "
                "AMP-enabled; restore on the same precision path or declare a "
                "migration")
        if payload.get("scaler_state") and self.use_amp:
            self.scaler.load_state_dict(payload["scaler_state"])
        self.skipped_updates = int(payload.get("skipped_updates", 0))
        self.attempted_updates = int(payload.get("attempted_updates",
                                                 self.counters.optimizer_updates
                                                 + self.skipped_updates))
        self._pending_normalized = False
        self._last_route_report = {}
