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
    PairUpdateInput,
    StepReport,
    Trainer,
    TrainerStateError,
    _total_grad_norm,
    answer_eos_loss_sum,
    validate_targets_in_schema,
)
from bramastra_lab.research.experience.supervision import SupervisionWindow
from bramastra_lab.research.learning.router import route_window
from bramastra_lab.research.learning.objectives import (
    pair_margin_loss,
    sequence_logprob_scores,
)


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
      diagnostics, masks and targets (CUDA fp16 or TPU bfloat16 autocast);
    - CUDA AMP unscale happens before clipping; scaler state is checkpointed;
      XLA replicas use global objective denominators and an all-reduced step;
    - a skipped nonfinite step does not increment successful optimizer
      updates or consume a schedule step, but its time/exposure stays charged
      through the attempted/microbatch counters;
    - updates are admitted only against a live AllocationContext.
    """

    def __init__(self, config, model, *, device: str | None = None,
                 allocation: AllocationContext | None = None,
                 precision: str = "fp32",
                 require_allocation: bool = False,
                 replica_backend: Any | None = None) -> None:
        super().__init__(config, model, device=device)
        if precision not in ("fp32", "fp16_autocast", "bf16_autocast"):
            raise TrainerStateError(f"unsupported precision {precision!r}")
        device_name = str(self.device)
        is_cuda = device_name.startswith("cuda")
        is_xla = device_name.startswith("xla")
        if precision == "bf16_autocast" and not is_xla:
            raise TrainerStateError("bf16_autocast is reserved for the TPU/XLA path")
        if is_xla and precision != "bf16_autocast":
            raise TrainerStateError("TPU/XLA training requires explicit bf16_autocast")
        self.precision = precision
        self.allocation = allocation
        self.require_allocation = bool(require_allocation)
        self.use_amp = ((precision == "fp16_autocast" and is_cuda)
                        or (precision == "bf16_autocast" and is_xla))
        self.scaler_enabled = precision == "fp16_autocast" and is_cuda
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.scaler_enabled) if hasattr(torch, "amp") else \
            torch.cuda.amp.GradScaler(enabled=self.scaler_enabled)
        if replica_backend is None and is_xla:
            from bramastra_lab.research.runtime.tpu import XLAReplicaBackend

            replica_backend = XLAReplicaBackend.current(self.device)
        self.replica_backend = replica_backend
        if self.replica_backend is not None:
            replicas = getattr(self.replica_backend, "world_size", None)
            if not isinstance(replicas, int) or isinstance(replicas, bool) or replicas < 1:
                raise TrainerStateError("replica backend must declare a positive world_size")
            if not all(callable(getattr(self.replica_backend, name, None))
                       for name in ("reduce_counts", "materialize_counts",
                                    "reduce_metric_sum", "reduce_gradients",
                                    "mark_step", "optimizer_step")):
                raise TrainerStateError(
                    "replica backend must implement count reduction/materialization, "
                    "gradient reduction, and optimizer step")
        self.skipped_updates = 0
        self.attempted_updates = 0
        self._pending_normalized = False
        self._last_route_report = {}
        self._active_window_weights = {}
        self._active_window_enabled = frozenset()
        self._active_global_counts: dict[str, int] = {}
        self._pending_pair_loss_value: float | None = None

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

    # -- Single optimizer-window contract (D2) --------------------------------
    #
    # One update window = one accumulate call + one explicit finalize_update
    # that steps exactly once. Accumulation never steps; finalization steps
    # exactly once. Each accumulate routes its terms through route_window with
    # per-term OWN denominators, so there is no double normalization and no
    # mean-of-microbatch-means. The XLA pair path uses a second backward after
    # flushing base activations; both losses still share one reduced/clipped
    # optimizer boundary. A second
    # accumulate before finalize is refused (declare multi-microbatch windows
    # explicitly instead of silently averaging means). Pair rows are consumed
    # at the boundary when the pair weight is positive; disabled terms must
    # neither execute nor contribute gradients; missing eligible terms for
    # positively weighted objectives are refused.

    def _require_clean_boundary(self) -> None:
        if getattr(self, "_pending_targets", 0):
            raise TrainerStateError(
                "explicit finalize_update required before next accumulation "
                "(one optimizer window = one accumulate + one finalize; "
                "multi-microbatch means are refused)")

    def _clear_pending(self) -> None:
        self._pending_targets = 0
        self._pending_answer_sum = 0.0
        self._pending_presentations = 0
        self._pending_pair_own = []
        self._pending_pair_swapped = []
        self._pending_normalized = False
        self._last_route_report = {}
        self._active_window_weights = {}
        self._active_window_enabled = frozenset()
        self._active_global_counts = {}
        self._pending_pair_loss_value = None

    def accumulate(self, batch, *, pair_rows=None) -> dict[str, float]:
        """Single-window token accumulation (no step).

        Routes the token term through route_window with its own denominator;
        pair_rows are pooled for boundary consumption. Refuses a second
        accumulate before finalize.
        """
        self._require_clean_boundary()
        if self.replica_backend is not None and pair_rows is not None:
            raise TrainerStateError(
                "replica pair objectives require accumulate_full_window so "
                "their global eligibility count is reduced before backward")
        if batch.target_count == 0:
            raise TrainerStateError("micro batch declares zero supervised targets")
        if self.treatment != "full" and self.schema is not None:
            validate_targets_in_schema(batch.labels.to(self.device),
                                       batch.loss_mask.to(self.device), self.schema)
        self.model.train()
        move = lambda tensor: tensor.to(self.device)
        with self._autocast_context():
            output = self.model(move(batch.input_ids), move(batch.padding_mask),
                                segment_ids=move(batch.segment_ids))
            treated = self._treated_logits(output.logits)
            report = answer_eos_loss_sum(treated, move(batch.labels),
                                         move(batch.loss_mask))
        if report.target_count != batch.target_count:
            raise TrainerStateError(
                f"batch declares {batch.target_count} targets but loss sees "
                f"{report.target_count}")
        from bramastra_lab.research.experience.supervision import SupervisionWindow
        from bramastra_lab.research.learning.router import route_window

        window = SupervisionWindow(
            weights={"token": 1.0, "world": 0.0, "action": 0.0,
                     "value": 0.0, "pair": 0.0, "pg": 0.0},
            enabled_terms=frozenset({"token"}))
        window.add("token", report.target_count)
        global_denominators = None
        if self.replica_backend is not None:
            global_denominators = self.replica_backend.reduce_counts(
                {"token": report.target_count})
            global_counts = self.replica_backend.materialize_counts(
                global_denominators)
            if global_counts.get("token", 0) <= 0:
                raise TrainerStateError(
                    "token objective has zero eligibility across TPU replicas")
        combined, _route_report = route_window(
            window, {"token": report.total},
            global_denominators=global_denominators,
            replica_gradient_scale=(self.replica_backend.world_size
                                    if self.replica_backend is not None else 1.0))
        scaled = self.scaler.scale(combined) if self.scaler_enabled else combined
        scaled.backward()
        if pair_rows is not None:
            own_rows, swapped_rows = pair_rows
            if len(own_rows) != len(swapped_rows):
                raise TrainerStateError("pair renderings must align within a micro batch")
            if float(self.pair_loss_weight) <= 0:
                raise TrainerStateError(
                    "pair renderings supplied while training.pair_loss_weight "
                    "is zero (disabled terms must not contribute)")
            self._pending_pair_own.extend(own_rows)
            self._pending_pair_swapped.extend(swapped_rows)
        self._pending_targets += report.target_count
        self._pending_answer_sum += float(report.total.detach().item())
        self._pending_presentations += int(batch.batch_size)
        self._pending_normalized = True
        self._active_window_weights = {"token": 1.0}
        self._active_window_enabled = frozenset({"token"})
        self._active_global_counts = (global_counts
                                      if self.replica_backend is not None else {})
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
        denominators, never a single answer count), then performs backward.
        The XLA pair path flushes base activations before its pair forward to
        bound memory; all objective gradients still reduce and clip once before
        the single optimizer step. Both E0 branches use identical objectives.

        window_builder(target_count) -> SupervisionWindow
        extra_terms_fn() -> {term: live Tensor sum} for non-token terms.
        """
        from bramastra_lab.research.experience.supervision import SupervisionWindow
        from bramastra_lab.research.learning.router import route_window

        self._require_clean_boundary()
        if batch.target_count == 0:
            raise TrainerStateError("micro batch declares zero supervised targets")
        if self.treatment != "full" and self.schema is not None:
            validate_targets_in_schema(batch.labels.to(self.device),
                                       batch.loss_mask.to(self.device), self.schema)
        self.model.train()
        move = lambda tensor: tensor.to(self.device)
        with self._autocast_context():
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
            with self._autocast_context():
                extra = extra_terms_fn()
            if not isinstance(extra, dict):
                raise TrainerStateError("extra_terms_fn must return a dict")
            for term, total in extra.items():
                if term == "token":
                    raise TrainerStateError("extra terms must not redefine 'token'")
                if term not in window.enabled_terms:
                    raise TrainerStateError(
                        f"objective term {term!r} executed while disabled by "
                        "feature switch (disabled terms must not contribute)")
                if float(window.weights.get(term, 0.0)) <= 0:
                    raise TrainerStateError(
                        f"objective term {term!r} executed with non-positive "
                        "weight (disabled terms must not contribute gradients)")
                sums[term] = total
        pair_rows_validated = None
        pair_metric_value = None
        if pair_rows is not None:
            if not isinstance(pair_rows, (tuple, list)) or len(pair_rows) != 2:
                raise TrainerStateError(
                    "pair_rows must be a pair of own/swapped row sequences")
            own_rows, swapped_rows = pair_rows
            if len(own_rows) != len(swapped_rows):
                raise TrainerStateError("pair renderings must align within a micro batch")
            pair_weight = float(window.weights.get("pair", self.pair_loss_weight))
            if "pair" not in window.enabled_terms or pair_weight <= 0:
                raise TrainerStateError(
                    "pair renderings supplied while the pair objective is disabled")
            pair_rows_validated = (own_rows, swapped_rows)
            if self.replica_backend is not None:
                if len(own_rows) != window.denominator("pair"):
                    raise TrainerStateError(
                        "pair eligibility count disagrees with supplied own/swapped "
                        f"rows: declared={window.denominator('pair')}, "
                        f"observed={len(own_rows)}")
                if "pair" in sums:
                    raise TrainerStateError(
                        "pair loss cannot be supplied both as an extra term and "
                        "as pair_rows")
        elif self.replica_backend is not None and window.denominator("pair") > 0 \
                and "pair" not in sums:
            raise TrainerStateError(
                "pair objective declares local eligible groups but supplies no rows")
        pair_from_rows = (
            self.replica_backend is not None
            and "pair" in window.enabled_terms
            and float(window.weights.get("pair", self.pair_loss_weight)) > 0
            and "pair" not in sums)
        global_denominators = None
        if self.replica_backend is not None:
            local_counts = {term: window.denominator(term)
                            for term in sorted(window.enabled_terms)}
            global_denominators = self.replica_backend.reduce_counts(local_counts)
            # Check positive-weight terms against the all-reduced eligibility,
            # not this replica's local shard. This is one small count-vector
            # synchronization; objective losses stay on device.
            global_count_values = self.replica_backend.materialize_counts(
                global_denominators)
            for term in sorted(window.enabled_terms):
                if float(window.weights.get(term, 0.0)) > 0 \
                        and global_count_values.get(term, 0) <= 0:
                    raise TrainerStateError(
                        f"objective term {term!r} has zero eligibility across "
                        "all TPU replicas; refusing a silent zero-loss update")
                if window.denominator(term) == 0 and term not in sums \
                        and not (term == "pair" and pair_from_rows):
                    # A different replica may carry this term. Supply a local
                    # differentiable zero so every replica executes the same
                    # objective boundary before the gradient all-reduce.
                    sums[term] = report.total * 0.0

        # Missing eligible terms for positively weighted objectives refuse.
        for term in sorted(window.enabled_terms):
            if self.replica_backend is None \
                    and float(window.weights.get(term, 0.0)) > 0 \
                    and window.denominator(term) <= 0:
                raise TrainerStateError(
                    f"objective term {term!r} weighted "
                    f"{window.weights.get(term)} but has zero eligible data "
                    "in this window; refusing silent zero loss")
        route_window_obj = window
        if pair_from_rows:
            base_enabled = frozenset(window.enabled_terms - {"pair"})
            base_counts = dict(window.counts)
            base_counts["pair"] = 0
            route_window_obj = SupervisionWindow(
                weights=dict(window.weights), counts=base_counts,
                payloads=dict(window.payloads), enabled_terms=base_enabled)
        combined, route_report = route_window(
            route_window_obj, sums, global_denominators=global_denominators,
            replica_gradient_scale=(self.replica_backend.world_size
                                    if self.replica_backend is not None else 1.0))
        # One normalized base-objective backward for this logical window.
        scaled = self.scaler.scale(combined) if self.scaler_enabled else combined
        scaled.backward()
        if pair_from_rows:
            # Flush the base objective before running the two additional pair
            # forwards. This bounds activation residency for the 100M profile
            # while keeping one logical optimizer window and one final step.
            self.replica_backend.mark_step()
            del output, treated
            if pair_rows_validated is not None:
                local_pair_sum, pair_count = self._pair_objective_sum(
                    pair_rows_validated)
                if pair_count != window.denominator("pair"):
                    raise TrainerStateError(
                        "computed pair count changed after pre-backward validation")
            else:
                local_pair_sum = torch.zeros((), dtype=torch.float32,
                                             device=self.device)
                pair_count = 0
            pair_window = SupervisionWindow(
                weights=dict(window.weights),
                enabled_terms=frozenset({"pair"}))
            pair_window.add("pair", pair_count)
            pair_total, pair_report = route_window(
                pair_window, {"pair": local_pair_sum},
                global_denominators={"pair": global_denominators["pair"]},
                replica_gradient_scale=self.replica_backend.world_size)
            if pair_count > 0:
                pair_total.backward()
            route_report.setdefault("applied", {}).update(
                pair_report.get("applied", {}))
            if global_count_values.get("pair", 0) > 0:
                global_pair_sum = self.replica_backend.reduce_metric_sum(
                    local_pair_sum.detach())
                pair_metric_value = float(
                    (global_pair_sum.detach().float()
                     / global_count_values["pair"]).cpu().item())
        if pair_rows_validated is not None:
            if self.replica_backend is None:
                self._pending_pair_own.extend(pair_rows_validated[0])
                self._pending_pair_swapped.extend(pair_rows_validated[1])
        if pair_rows_validated is not None or pair_from_rows:
            self._pending_pair_loss_value = pair_metric_value
        self._pending_targets += report.target_count
        self._pending_answer_sum += float(report.total.detach().item())
        self._pending_presentations += int(batch.batch_size)
        self._pending_normalized = True
        self._active_window_weights = dict(window.weights)
        self._active_window_enabled = frozenset(window.enabled_terms)
        self._active_global_counts = (global_count_values
                                      if self.replica_backend is not None else {})
        self._last_route_report = route_report
        self.counters.microbatches += 1
        self.counters.presentations += int(batch.batch_size)
        self.counters.supervised_targets_seen += report.target_count
        self.counters.encoded_tokens_seen += int(batch.padding_mask.sum().item())
        return {"answer_loss_sum": self._pending_answer_sum,
                "pending_targets": float(self._pending_targets),
                "route_applied": route_report.get("applied", {})}

    def _pair_objective_sum(self, pair_rows) -> tuple[torch.Tensor, int]:
        """Build a differentiable, unnormalized pair-loss sum before routing."""
        from bramastra_lab.research.experience.sequences import collocate

        own_rows, swapped_rows = pair_rows
        own_batch = collocate(list(own_rows), max_seq=self.config.model.max_seq)
        swapped_batch = collocate(list(swapped_rows), max_seq=self.config.model.max_seq)
        move = lambda tensor: tensor.to(self.device)
        with self._autocast_context():
            own_output = self.model(
                move(own_batch.input_ids), move(own_batch.padding_mask),
                segment_ids=move(own_batch.segment_ids))
            swapped_output = self.model(
                move(swapped_batch.input_ids), move(swapped_batch.padding_mask),
                segment_ids=move(swapped_batch.segment_ids))
        own_scores = sequence_logprob_scores(
            own_output.logits.float(), move(own_batch.labels),
            move(own_batch.loss_mask), normalization=self.pair_score_normalization)
        swapped_scores = sequence_logprob_scores(
            swapped_output.logits.float(), move(swapped_batch.labels),
            move(swapped_batch.loss_mask), normalization=self.pair_score_normalization)
        mean_loss, counted = pair_margin_loss(
            own_scores, swapped_scores, self.pair_margin)
        if counted == 0:
            return mean_loss, 0
        return mean_loss * counted, counted

    def _amp_device_type(self) -> str:
        device = str(self.device)
        if device.startswith("cuda"):
            return "cuda"
        if device.startswith("xla"):
            return "xla"
        return "cpu"

    def _autocast_context(self):
        dtype = (torch.bfloat16 if self.precision == "bf16_autocast"
                 else torch.float16)
        return torch.autocast(device_type=self._amp_device_type(), dtype=dtype,
                              enabled=self.use_amp)

    def finalize_update(self) -> StepReport:
        """One explicit optimizer boundary: steps exactly once (D2).

        Gradients are already per-term normalized at accumulation (no
        re-division here). Consumes pooled pair rows through the pair-margin
        term when the pair weight is positive; refuses pair rows when the
        pair weight is zero and refuses missing pair data when it is positive.
        A failed report publication after a committed step never erases the
        step.
        """
        self._admit_update()
        self.attempted_updates += 1
        if self._pending_targets <= 0:
            raise TrainerStateError("finalize_update called with no accumulated batches")
        if not bool(getattr(self, "_pending_normalized", False)):
            self._clear_pending()
            raise TrainerStateError(
                "pending gradients are not window-normalized; accumulate via "
                "the canonical window boundary before finalize")
        if self.scaler_enabled:
            self.scaler.unscale_(self.optimizer)
        if self.replica_backend is not None:
            missing_required = []
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    continue
                head_term = ("action" if name.startswith("action_head.") else
                             "value" if name.startswith("value_head.") else None)
                if head_term is not None:
                    globally_active = (
                        float(self._active_window_weights.get(head_term, 0.0)) > 0
                        and self._active_global_counts.get(head_term, 0) > 0)
                    if globally_active:
                        # This objective exists on another shard; zero-fill
                        # its absent local head gradient so the all-reduce sees
                        # the same parameter set on every replica.
                        parameter.grad = torch.zeros_like(parameter)
                    # A globally inactive head must stay grad=None. In
                    # particular, do not zero-fill it: optimizers such as
                    # AdamW would apply weight decay to an untrained head.
                    continue
                missing_required.append(name)
            if missing_required:
                self.optimizer.zero_grad(set_to_none=True)
                self._clear_pending()
                raise TrainerStateError(
                    "TPU replica has unused trainable parameters outside the "
                    f"declared action/value heads: {missing_required[:5]}")
            # Reduce before clipping: clipping a per-replica gradient and then
            # averaging is not equivalent to clipping the global gradient.
            self.replica_backend.reduce_gradients(self.optimizer)
        if self.replica_backend is not None:
            gradients = [parameter.grad for parameter in self.model.parameters()
                         if parameter.grad is not None]
            finite = (torch.stack([torch.isfinite(gradient).all()
                                   for gradient in gradients]).all()
                      if gradients else torch.tensor(True, device=self.device))
            has_nonfinite = not bool(finite.detach().cpu().item())
        else:
            has_nonfinite = any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in self.model.parameters())
        if has_nonfinite:
            self.skipped_updates += 1
            self.optimizer.zero_grad(set_to_none=True)
            self._clear_pending()
            raise TrainerStateError(
                "nonfinite gradients after unscale; update skipped and "
                "charged (attempted but not committed)")
        # No re-division: per-term denominators already applied at
        # accumulation (single-window contract, no mean-of-means).
        pair_loss_value: float | None = self._pending_pair_loss_value
        if self.replica_backend is None and (
                self._pending_pair_own or self._pending_pair_swapped):
            effective_pair_weight = self._effective_pair_weight()
            if effective_pair_weight <= 0:
                self._clear_pending()
                raise TrainerStateError(
                    "pair renderings pending while the effective pair weight "
                    "(trainer default and active window) is zero "
                    "(disabled terms must not contribute)")
            if len(self._pending_pair_own) != len(self._pending_pair_swapped):
                self._clear_pending()
                raise TrainerStateError(
                    "pair renderings must align across the accumulation window")
            from bramastra_lab.research.experience.sequences import collocate as _collocate

            max_seq = self.config.model.max_seq
            own_batch = _collocate(list(self._pending_pair_own), max_seq=max_seq)
            swapped_batch = _collocate(list(self._pending_pair_swapped), max_seq=max_seq)
            pair_loss_value = self._apply_pair_term(
                PairUpdateInput(own=own_batch, swapped=swapped_batch),
                weight=effective_pair_weight)
        elif self.replica_backend is None and self._effective_pair_weight() > 0 and \
                "pair" in getattr(self, "_active_window_enabled", frozenset()):
            self._clear_pending()
            raise TrainerStateError(
                "pair term enabled with positive weight but no pair renderings "
                "in this window; refusing silent zero pair loss")
        pre_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.clip_norm)
        if self.replica_backend is not None:
            post_norm_parts = [parameter.grad.norm(dtype=torch.float32)
                               for parameter in self.model.parameters()
                               if parameter.grad is not None]
            post_norm_tensor = torch.norm(torch.stack(post_norm_parts)) \
                if post_norm_parts else torch.zeros((), device=self.device)
            certificate = (post_norm_tensor <= self.clip_norm
                           * (1.0 + CLIP_CERTIFICATE_TOLERANCE))
            measured = torch.stack((pre_norm_tensor.float(), post_norm_tensor.float(),
                                    certificate.to(dtype=torch.float32)))
            pre_norm, post_norm, certificate_value = measured.detach().cpu().tolist()
            clipped = pre_norm > self.clip_norm
            clip_ok = bool(certificate_value)
        else:
            pre_norm = float(pre_norm_tensor)
            clipped = pre_norm > self.clip_norm
            post_norm = _total_grad_norm(self.model.parameters())
            clip_ok = post_norm <= self.clip_norm * (
                1.0 + CLIP_CERTIFICATE_TOLERANCE)
        if not clip_ok:
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
        if self.replica_backend is not None:
            self.replica_backend.optimizer_step(self.optimizer)
        elif self.scaler_enabled:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.counters.optimizer_updates += 1
        report = StepReport(
            optimizer_update=self.counters.optimizer_updates,
            answer_loss_mean=self._pending_answer_sum / self._pending_targets,
            pair_loss=pair_loss_value, lr=scheduled_lr,
            grad_norm=pre_norm, clipped=clipped,
            supervised_targets=self._pending_targets,
            presentations=self._pending_presentations)
        if self.diagnostics is not None:
            report.telemetry = self.diagnostics.collect(snapshot=snapshot)
        self.optimizer.zero_grad(set_to_none=True)
        self._clear_pending()
        return report

    # -- scaler state in checkpoints ------------------------------------------

    def state_payload(self) -> dict[str, Any]:
        """Full checkpoint contract (D2): config, architecture, model,
        optimizer, scaler, RNG, stream cursor, controller + allocation/job."""
        import random as _random

        payload = super().state_payload()
        payload["precision"] = self.precision
        payload["use_amp"] = self.use_amp
        payload["scaler_state"] = (self.scaler.state_dict()
                                    if self.scaler_enabled else None)
        payload["replica_world_size"] = int(
            self.replica_backend.world_size
            if self.replica_backend is not None else 1)
        payload["skipped_updates"] = self.skipped_updates
        payload["attempted_updates"] = int(getattr(self, "attempted_updates", 0))
        payload["allocation"] = ({"allocation_id": self.allocation.allocation_id,
                                  "job_id": self.allocation.job_id,
                                  "phase": self.allocation.phase}
                                 if self.allocation else None)
        try:
            payload["config_identity"] = self.config.identity()
        except Exception:
            payload["config_identity"] = None
        try:
            payload["architecture_id"] = getattr(
                self.model, "architecture_id", "bramastra-base-decoder/v1")
        except Exception:
            payload["architecture_id"] = None
        try:
            import torch as _torch

            payload["rng_state"] = {
                "python": _random.getstate()[1][:8],
                "torch": _torch.get_rng_state().tolist()[:16],
                "torch_cuda": None,
            }
        except Exception:
            payload["rng_state"] = None
        payload["profile"] = getattr(getattr(self.config, "model", None),
                                     "profile", None)
        return payload

    def load_state_payload(self, payload: Mapping[str, Any], *,
                           expected_allocation: AllocationContext | None = None,
                           expected_config_identity: str | None = None) -> None:
        """Restore with contract validation (D2).

        Validates precision/profile path, config identity when supplied, and a
        live reservation (expected_allocation) instead of granting a new
        allowance: device must match, deadline must be live, and remaining
        updates must be non-negative. Sets the update baseline from the
        restored counters (no new allowance).
        """
        super().load_state_payload(payload)
        if payload.get("precision") and payload["precision"] != self.precision:
            raise TrainerStateError(
                "checkpoint precision does not match this trainer; resume on "
                "the same precision path or declare a migration")
        if payload.get("use_amp") and not self.use_amp:
            raise TrainerStateError(
                "checkpoint was trained with AMP on CUDA but this trainer is not "
                "AMP-enabled; restore on the same precision path or declare a "
                "migration")
        saved_replicas = int(payload.get("replica_world_size", 1))
        current_replicas = int(self.replica_backend.world_size
                               if self.replica_backend is not None else 1)
        if saved_replicas != current_replicas:
            raise TrainerStateError(
                f"checkpoint replica count {saved_replicas} does not match "
                f"trainer replica count {current_replicas}")
        if payload.get("scaler_state") and self.scaler_enabled:
            self.scaler.load_state_dict(payload["scaler_state"])
        if expected_config_identity is not None and payload.get("config_identity") \
                and payload["config_identity"] != expected_config_identity:
            raise TrainerStateError(
                "checkpoint config identity does not match the declared "
                "configuration; refusing cross-config resume")
        if payload.get("profile") and getattr(getattr(self.config, "model", None),
                                              "profile", None) \
                and payload["profile"] != self.config.model.profile:
            raise TrainerStateError(
                f"checkpoint profile {payload['profile']!r} does not match "
                f"trainer profile {self.config.model.profile!r}")
        if expected_allocation is not None:
            import time as _time

            if self.allocation is None:
                self.allocation = expected_allocation
            if self.allocation.device != expected_allocation.device:
                raise TrainerStateError(
                    "checkpoint device does not match the live reservation")
            if _time.time() > expected_allocation.deadline_unix:
                raise TrainerStateError(
                    "live reservation deadline already passed; refusing resume")
            if expected_allocation.remaining_updates < 0:
                raise TrainerStateError("live reservation has no remaining updates")
        self._k8_updates_at_start = self.counters.optimizer_updates
        self.skipped_updates = int(payload.get("skipped_updates", 0))
        self.attempted_updates = int(payload.get("attempted_updates",
                                                 self.counters.optimizer_updates
                                                 + self.skipped_updates))
        self._pending_normalized = False
        self._last_route_report = {}
        self._active_window_weights = {}
        self._active_window_enabled = frozenset()
        self._active_global_counts = {}
