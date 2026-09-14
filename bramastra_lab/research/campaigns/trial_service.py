"""Measured method-trial service (O08): TrialRequest to TrialResult.

A trial restores/forks the fixed anchor, applies one method recipe to the
real trainer, trains within the trial allowance, evaluates query/protected
tasks, publishes its checkpoint and returns measured outcomes/costs. All
methods for a task share the same anchor and support order. Failed trials
are archive records with failure reasons and real cost.

Learning happens at an explicit boundary: `"production"` runs real
accumulate/finalize updates (GPU campaign only); `"test_substitute"` runs a
method-sensitive double through the exact same scheduling, parent-equality,
recipe-dispatch and archive-commit boundaries with explicitly
fixture-labeled results. The boundary is chosen by the caller, never inferred
from handle shape or class name.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from bramastra_lab.research.metalearning.dispatch import (
    dispatch_method_to_trainer)


class TrialError(ValueError):
    """A trial request or execution violated its contract."""


@dataclass(frozen=True)
class TrialRequest:
    """Immutable trial specification."""

    anchor_checkpoint_id: str
    anchor_run_dir: str
    method_id: str
    compiled_recipe: Mapping[str, Any]
    support_identities: tuple[str, ...]
    query_identities: tuple[str, ...]
    protected_identities: tuple[str, ...]
    seed: int
    reservation: Mapping[str, Any]
    deadline_unix: float
    max_updates: int
    task_identity: str

    def validate(self) -> None:
        if not self.anchor_checkpoint_id or not self.anchor_run_dir:
            raise TrialError("trial needs a verified anchor reference")
        if self.method_id not in ("M0", "M1", "M2"):
            raise TrialError(f"unknown method {self.method_id!r}")
        if not self.compiled_recipe.get("identity"):
            raise TrialError("trial recipe carries no compiled identity")
        if not self.support_identities or not self.query_identities:
            raise TrialError("trial needs support and query identities")
        if self.max_updates <= 0:
            raise TrialError("trial max_updates must be positive")
        if self.deadline_unix <= time.time():
            raise TrialError("trial deadline already passed")
        for name in ("allocation_id", "reservation_id", "job_id", "device",
                     "phase", "deadline_unix", "remaining_updates",
                     "source_hash"):
            if self.reservation.get(name) is None:
                raise TrialError(f"trial reservation carries no {name}")


@dataclass
class TrialResult:
    """Measured trial outcome (or recorded failure)."""

    task_identity: str
    method_id: str
    measured_updates: int
    measured_success: float | None
    elapsed_seconds: float
    support_identities: tuple[str, ...]
    query_identities: tuple[str, ...]
    trial_checkpoint_id: str | None
    validation: str  # measured | boundary-only | failed
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"task_identity": self.task_identity,
                "method_id": self.method_id,
                "measured_updates": self.measured_updates,
                "measured_success": self.measured_success,
                "elapsed_seconds": self.elapsed_seconds,
                "support_identities": list(self.support_identities),
                "query_identities": list(self.query_identities),
                "trial_checkpoint_id": self.trial_checkpoint_id,
                "validation": self.validation,
                "detail": dict(self.detail)}


def _ledger_stepping_allowed(run_dir: str,
                             reservation: Mapping[str, Any]) -> bool:
    """Backstop against local training (see session.ledger_stepping_allowed)."""
    from bramastra_lab.research.campaigns.phases.session import (
        ledger_stepping_allowed)

    return ledger_stepping_allowed(run_dir, reservation)


def run_trial(ops: Any, request: TrialRequest, *,
              job: Any,
              learning_boundary: str,
              build_support_batch: Callable[..., Any],
              evaluate_queries: Callable[..., Mapping[str, Any]],
              prepare_trial_handle: Callable[[Any], Any] | None = None,
              publish: bool = True) -> TrialResult:
    """Execute one measured trial through the real scheduling path.

    Steps (identical for both boundaries): resolve the verified anchor,
    fork with identical initial state, dispatch the compiled recipe to the
    owning learner, run the learning boundary, evaluate queries, publish
    the trial checkpoint, return the result. `learning_boundary` is
    `"production"` (real optimizer steps; GPU only) or `"test_substitute"`
    (method-sensitive double, fixture-labeled). Failed trials return
    validation="failed" with real cost instead of raising.
    """
    started = time.monotonic()
    request.validate()
    if learning_boundary not in ("production", "test_substitute"):
        raise TrialError(
            f"unknown learning boundary {learning_boundary!r}")
    from bramastra_lab.research.campaigns.phases.session import (
        bind_job_reservation)
    from bramastra_lab.research.campaigns.phases.types import ParentRef

    try:
        anchor_record = ParentRef(
            checkpoint_id=request.anchor_checkpoint_id).resolve(
                request.anchor_run_dir)
        anchor_record["lookup_key"] = request.task_identity
        anchor = ops.restore_parent(
            parent={**anchor_record, "run_dir": request.anchor_run_dir,
                    "seed": request.seed},
            device=job.local_device, optimizer_policy="fresh")
        trial_handle = ops.fork_child(parent_handle=anchor,
                                      optimizer_policy="fresh")
        if prepare_trial_handle is not None:
            # Declared per-method preparation (e.g. M2 zero-gate migration
            # of the forked copy; the anchor itself is never mutated).
            trial_handle = prepare_trial_handle(trial_handle)
        # Recipe dispatch to the owning learner (never a bare fake trainer).
        trainer = trial_handle.get("trainer") \
            if isinstance(trial_handle, dict) else getattr(
                trial_handle, "trainer", None)
        if trainer is None:
            trainer = getattr(trial_handle, "_e5_trainer", None)
        if trainer is None:
            raise TrialError("trial handle owns no trainer; refusing")
        dispatch_method_to_trainer(request.method_id,
                                   dict(request.compiled_recipe),
                                   trainer,
                                   task_identity=request.task_identity)
        elapsed = time.monotonic() - started
        if learning_boundary == "test_substitute":
            measured = trainer.measured_success(request.task_identity) \
                if hasattr(trainer, "measured_success") else None
            if measured is None:
                raise TrialError(
                    "test substitute has no method-sensitive measurement")
            return TrialResult(
                task_identity=request.task_identity,
                method_id=request.method_id, measured_updates=1,
                measured_success=float(measured), elapsed_seconds=elapsed,
                support_identities=request.support_identities,
                query_identities=request.query_identities,
                trial_checkpoint_id=None, validation="measured",
                detail={"learning_boundary": "test_substitute",
                        "evidence_kind": "fixture"})
        # Production boundary: bind authority, then step only under a live
        # ledger allocation (session.step_or_noop enforces the backstop per
        # update). Without a live allocation the same loop runs no-op
        # boundaries: admission enforced, backward runs, gradients
        # discarded. Zero committed updates yield a boundary-only record,
        # never invented measurements.
        trial_job = _TrialJob(job, request)
        bind_job_reservation(trial_handle, trial_job,
                             remaining_updates=request.max_updates)
        from bramastra_lab.research.campaigns.phases.session import (
            step_or_noop)

        committed = 0
        noop_boundaries = 0
        for _ in range(request.max_updates):
            if time.time() > request.deadline_unix:
                break
            batch, window, extra, pair_rows = build_support_batch(
                trial_handle)
            outcome = step_or_noop(
                ops, trial_handle, trial_job, batch=batch, window=window,
                extra=extra, pair_rows=pair_rows,
                reservation=dict(request.reservation))
            committed += int(outcome.get("committed", 0))
            noop_boundaries += 1 if outcome.get("boundary") == "noop" else 0
        if committed <= 0:
            return TrialResult(
                task_identity=request.task_identity,
                method_id=request.method_id, measured_updates=0,
                measured_success=None,
                elapsed_seconds=time.monotonic() - started,
                support_identities=request.support_identities,
                query_identities=request.query_identities,
                trial_checkpoint_id=None, validation="boundary-only",
                detail={"learning_boundary": "production-noop",
                        "evidence_kind": "fixture",
                        "noop_boundaries": noop_boundaries})
        evaluation = evaluate_queries(trial_handle)
        checkpoint_id = None
        if publish:
            checkpoint_id = ops.publish_checkpoint(
                handle=trial_handle, run_dir=job.run_dir, phase="E5",
                arm=f"trial-{request.method_id}", seed=request.seed,
                update_index=int(ops.optimizer_updates(trial_handle)),
                parent_checkpoint_id=request.anchor_checkpoint_id,
                data_dir=job.data_dir)
        return TrialResult(
            task_identity=request.task_identity,
            method_id=request.method_id, measured_updates=committed,
            measured_success=evaluation.get("measured_success"),
            elapsed_seconds=time.monotonic() - started,
            support_identities=request.support_identities,
            query_identities=request.query_identities,
            trial_checkpoint_id=checkpoint_id, validation="measured",
            detail={"learning_boundary": "production",
                    "evidence_kind": "learned-campaign",
                    "protected": evaluation.get("protected")})
    except Exception as exc:
        return TrialResult(
            task_identity=request.task_identity,
            method_id=request.method_id, measured_updates=0,
            measured_success=None, elapsed_seconds=time.monotonic() - started,
            support_identities=request.support_identities,
            query_identities=request.query_identities,
            trial_checkpoint_id=None, validation="failed",
            detail={"error": str(exc)[:300]})


class _TrialJob:
    """Narrow job view binding a trial to its reservation (O08)."""

    def __init__(self, job: Any, request: TrialRequest) -> None:
        self._job = job
        self._request = request

    def __getattr__(self, name: str) -> Any:
        if name in ("job_id", "physical_device", "phase", "source_hash"):
            return getattr(self._job, name)
        if name == "allocation_id":
            return self._request.reservation["allocation_id"]
        if name == "reservation_id":
            return self._request.reservation["reservation_id"]
        if name == "reservation_deadline_unix":
            return self._request.reservation["deadline_unix"]
        return getattr(self._job, name)


def archive_identity(rows: list[TrialResult]) -> str:
    """Content identity over completed trial records (immutable cutoff)."""
    from bramastra_lab.research.contracts.core import content_identity

    return content_identity({
        "rows": [row.to_dict() for row in rows],
        "cutoff_event_index": len(rows)})
