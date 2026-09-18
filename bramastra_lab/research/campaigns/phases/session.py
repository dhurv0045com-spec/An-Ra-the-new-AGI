"""Authorized learner sessions (O01): bind supervisor authority to the learner.

A session is the unit of execution. Its durable identity (source/data/
tokenizer/config/protocol hashes, architecture ID, parent checkpoint, seed,
allocation/reservation/job/phase/slot IDs, physical device) travels with the
job; its runtime state (model, optimizer, scaler, RNG, stream cursor,
objective window, controller, event log) lives on the handle. Permission
comes only from the supervisor reservation: a session cannot finalize
without an active, compatible reservation, and restoring never grants a new
allowance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import time


class SessionError(ValueError):
    """A session authority or lifecycle rule was violated."""


SESSION_STATES = frozenset({
    "created", "restored", "active", "checkpointing", "completed", "failed",
})
SESSION_TRANSITIONS: Mapping[str, frozenset] = {
    "created": frozenset({"active", "failed"}),
    "restored": frozenset({"active", "failed"}),
    "active": frozenset({"checkpointing", "completed", "failed"}),
    "checkpointing": frozenset({"active", "completed", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
}


@dataclass(frozen=True)
class SessionIdentity:
    """Durable identity bound at session creation/restoration."""

    source_hash: str
    data_hash: str
    tokenizer_identity: str
    config_identity: str
    protocol_hash: str
    architecture_id: str
    parent_checkpoint_id: str | None
    seed: int
    allocation_id: str
    reservation_id: str
    job_id: str
    phase: str
    slot: int | None
    arm: str | None
    physical_device: str

    def validate(self) -> None:
        for name in ("source_hash", "data_hash", "tokenizer_identity",
                     "config_identity", "protocol_hash", "architecture_id",
                     "allocation_id", "reservation_id", "job_id", "phase",
                     "physical_device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SessionError(f"session identity {name} must be nonempty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) \
                or self.seed < 0:
            raise SessionError("session seed must be a nonnegative integer")


@dataclass
class LearnerSession:
    """Runtime session with an explicit lifecycle state."""

    identity: SessionIdentity
    state: str = "created"
    events: list[dict[str, Any]] = field(default_factory=list)
    updates_at_bind: int = 0

    def transition(self, to_state: str) -> None:
        allowed = SESSION_TRANSITIONS.get(self.state, frozenset())
        if to_state not in allowed:
            raise SessionError(
                f"session cannot move {self.state!r} -> {to_state!r}")
        self.state = to_state

    def record(self, kind: str, payload: Mapping[str, Any] | None = None) -> None:
        self.events.append({"at_unix": time.time(), "kind": kind,
                            "session": self.identity.job_id,
                            "state": self.state,
                            **dict(payload or {})})


def reservation_from_spec(*, spec: Mapping[str, Any],
                          allocation_id: str,
                          source_hash: str) -> dict[str, Any]:
    """Build the reservation record the worker binds (O01).

    The runner reserves first via the ledger, then spawns; the spec carries
    the resulting reservation fields so the worker binds the actual ledger
    reservation instead of inventing authority.
    """
    for name in ("reservation_id", "job_id", "device", "phase",
                 "deadline_unix", "remaining_updates"):
        if spec.get(name) is None:
            raise SessionError(
                f"spec carries no {name}; refusing invented authority")
    return {"allocation_id": allocation_id,
            "reservation_id": str(spec["reservation_id"]),
            "job_id": str(spec["job_id"]),
            "device": str(spec["device"]),
            "phase": str(spec["phase"]),
            "deadline_unix": float(spec["deadline_unix"]),
            "remaining_updates": int(spec["remaining_updates"]),
            "source_hash": source_hash}


def bind_reservation(handle: Any, reservation: Mapping[str, Any], *,
                     job: Any) -> Any:
    """Attach a supervisor reservation to the learner (O01).

    Validates job/device/phase/source/deadline compatibility, then admits
    through the trainer's supported `begin_campaign` API. Restoring never
    resets already-charged work: the update baseline is the live counter.
    Missing or mismatched authority is refused (never silently admitted).
    """
    from bramastra_lab.research.learning.k8_trainer import AllocationContext

    if not isinstance(reservation, Mapping):
        raise SessionError("reservation must be a mapping")
    for name in ("allocation_id", "reservation_id", "job_id", "device",
                 "phase", "deadline_unix", "remaining_updates", "source_hash"):
        if reservation.get(name) is None:
            raise SessionError(f"reservation carries no {name}; refusing")
    if str(reservation["job_id"]) != str(job.job_id):
        raise SessionError(
            f"reservation job {reservation['job_id']!r} != session job "
            f"{job.job_id!r}; refusing")
    if str(reservation["device"]) != str(job.physical_device):
        raise SessionError(
            f"reservation device {reservation['device']!r} != session device "
            f"{job.physical_device!r}; refusing")
    if str(reservation["phase"]) != str(job.phase):
        raise SessionError(
            f"reservation phase {reservation['phase']!r} != session phase "
            f"{job.phase!r}; refusing")
    if job.source_hash and str(reservation["source_hash"]) != str(job.source_hash):
        raise SessionError("reservation source does not match session source")
    if float(reservation["deadline_unix"]) <= time.time():
        raise SessionError("reservation deadline already passed; refusing")
    if int(reservation["remaining_updates"]) < 0:
        raise SessionError("reservation has negative update allowance")
    trainer = handle.get("trainer") if isinstance(handle, dict) else getattr(
        handle, "trainer", None)
    if trainer is None or not hasattr(trainer, "begin_campaign"):
        raise SessionError("learner handle owns no bindable trainer")
    trainer.begin_campaign(AllocationContext(
        allocation_id=str(reservation["allocation_id"]),
        device=str(reservation["device"]),
        deadline_unix=float(reservation["deadline_unix"]),
        remaining_updates=int(reservation["remaining_updates"]),
        job_id=str(reservation["job_id"]),
        phase=str(reservation["phase"])))
    if isinstance(handle, dict):
        handle["reservation"] = dict(reservation)
        handle["reservation_bound"] = True
    return handle


def bind_job_reservation(handle: Any, job: Any, *,
                       remaining_updates: int) -> str:
    """Bind the job's supervisor reservation or classify the handle.

    Returns `"bound"` when a real trainer was admitted through
    `begin_campaign`, or `"zero-update-double"` when the handle owns no
    bindable trainer (deterministic doubles perform zero optimizer updates
    by construction, so there is no allowance to admit). A real trainer
    without reservation fields fails (missing authority is never admitted).
    """
    trainer = handle.get("trainer") if isinstance(handle, dict) else getattr(
        handle, "trainer", None)
    if trainer is None or not hasattr(trainer, "begin_campaign"):
        return "zero-update-double"
    if not getattr(job, "reservation_id", None) \
            or not getattr(job, "allocation_id", None) \
            or getattr(job, "reservation_deadline_unix", None) is None:
        raise SessionError(
            "training job carries no supervisor reservation "
            "(allocation/reservation/deadline required); refusing")
    bind_reservation(handle, {
        "allocation_id": job.allocation_id,
        "reservation_id": job.reservation_id,
        "job_id": job.job_id,
        "device": job.physical_device,
        "phase": job.phase,
        "deadline_unix": float(job.reservation_deadline_unix),
        "remaining_updates": int(remaining_updates),
        "source_hash": job.source_hash or "unbound-source",
    }, job=job)
    return "bound"


def ledger_stepping_allowed(run_dir: str,
                            reservation: Mapping[str, Any]) -> bool:
    """Whether a live ledger allocation authorizes real optimizer steps.

    Real steps require the allocation row plus an OPEN reservation for the
    job in this run's ledger with a live deadline. Anything else (local
    runs, simulated reservations, closed/expired jobs) forces the no-op
    boundary: admission is still enforced and backward still runs, but no
    step is taken and records stay fixture-labeled. This backstop makes
    local training impossible by construction, even with well-formed
    reservation fields.
    """
    return bool(stepping_readiness(run_dir, reservation).get("allowed", False))


def stepping_readiness(run_dir: str,
                       reservation: Mapping[str, Any] | None) -> dict[str, Any]:
    """Explain the stepping decision (fail-fast diagnostics).

    Same contract as `ledger_stepping_allowed` but returns the reason:
    executors call this once after binding so a dead authority fails the
    job in seconds with the concrete cause (missing allocation row,
    reservation not open, deadline passed, ledger unreadable) instead of
    noop-ing hundreds of steps into a confusing downstream failure.
    """
    import os as _os
    import sqlite3 as _sqlite

    if not isinstance(reservation, Mapping):
        return {"allowed": False, "reason": "no reservation record"}
    ledger_path = _os.path.join(run_dir, "campaign_ledger.sqlite")
    if not _os.path.exists(ledger_path):
        return {"allowed": False, "reason": f"no ledger at {ledger_path}"}
    try:
        ledger_stat = _os.stat(ledger_path)
        ledger_sig = f"size={ledger_stat.st_size} mtime={ledger_stat.st_mtime}"
    except OSError:
        ledger_sig = "stat-unavailable"
    try:
        conn = _sqlite.connect(ledger_path, timeout=60.0)
        try:
            allocation = conn.execute(
                "SELECT deadline_unix FROM allocation WHERE allocation_id=?",
                (str(reservation.get("allocation_id")),)).fetchone()
            if allocation is None:
                return {"allowed": False,
                        "reason": "allocation row missing for "
                                  f"{reservation.get('allocation_id')}",
                        "ledger_path": ledger_path, "ledger": ledger_sig}
            rows = conn.execute(
                "SELECT status, reserved_at_unix, reserved_seconds "
                "FROM reservations WHERE job_id=? ORDER BY rowid DESC",
                (str(reservation.get("job_id")),)).fetchall()
            if not rows:
                return {"allowed": False,
                        "reason": f"no ledger row for job {reservation.get('job_id')} "
                                  f"(ledger {ledger_path} [{ledger_sig}])",
                        "ledger_path": ledger_path, "ledger": ledger_sig}
            row = rows[0]
            if row[0] != "open":
                return {"allowed": False,
                        "reason": f"reservation for job {reservation.get('job_id')} "
                                  f"is {row[0]!r}, not 'open' (stale rebinding "
                                  "never authorizes steps)",
                        "ledger_path": ledger_path, "ledger": ledger_sig}
            live_until = min(float(allocation[0]), float(row[1]) + float(row[2]))
            if not time.time() < live_until:
                return {"allowed": False,
                        "reason": "reservation deadline passed",
                        "ledger_path": ledger_path, "ledger": ledger_sig}
            return {"allowed": True, "reservation_status": row[0],
                    "live_until_unix": live_until,
                    "ledger_path": ledger_path, "ledger": ledger_sig}
        finally:
            conn.close()
    except Exception as exc:
        return {"allowed": False, "reason": f"ledger unreadable: {exc}"}


def await_stepping_allowed(run_dir: str,
                           reservation: Mapping[str, Any] | None, *,
                           attempts: int = 3,
                           pause_seconds: float = 5.0) -> dict[str, Any]:
    """Stepping readiness with transient-lock tolerance.

    Retries the read a few times (a writer may briefly hold the ledger
    lock at spawn time); returns the LAST readiness dict either way so
    persistent refusal still fails with full diagnostics, never silently.
    """
    last: dict[str, Any] = {"allowed": False, "reason": "no attempts made"}
    for _ in range(max(1, int(attempts))):
        last = stepping_readiness(run_dir, reservation)
        if last.get("allowed", False):
            return last
        try:
            time.sleep(max(0.0, float(pause_seconds)))
        except Exception:
            break
    return last


def step_or_noop(ops: Any, handle: Any, job: Any, *,
                 batch: Any, window: Any,
                 extra: dict[str, Any] | None = None,
                 pair_rows: Any | None = None,
                 reservation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One training step through the real interface or the no-op boundary.

    Deterministic doubles (no window-capable trainer) take the recording
    interface with fixture evidence. Real trainers step ONLY under a live
    ledger allocation; otherwise they run the admission-checked no-op
    boundary (backward runs, gradients discarded, committed 0). The caller
    selects nothing by handle shape: the ledger decides stepping.
    """
    trainer = handle.get("trainer") if isinstance(handle, dict) else getattr(
        handle, "trainer", None)
    real_trainer = trainer is not None and hasattr(
        trainer, "accumulate_full_window")
    if not real_trainer:
        return ops.apply_update(handle, batch=batch, window=window,
                                extra=extra, pair_rows=pair_rows)
    if reservation is not None and ledger_stepping_allowed(
            job.run_dir, reservation):
        return ops.apply_update(handle, batch=batch, window=window,
                                extra=extra, pair_rows=pair_rows)
    proof = drive_noop_boundary(handle, batch=batch, window=window,
                                extra=extra, pair_rows=pair_rows)
    return {"committed": 0, "attempted": 1,
            "exposure": int(getattr(batch, "target_count", 0)),
            "optimizer_update": proof.get("optimizer_updates_after", 0),
            "boundary": "noop",
            "admission": {key: proof[key] for key in
                          ("admitted", "optimizer_updates_before",
                           "optimizer_updates_after") if key in proof}}


def job_reservation_record(job: Any, *, remaining_updates: int,
                           deadline_unix: float | None = None
                           ) -> dict[str, Any]:
    """Build the reservation record for stepping decisions (O01).

    Requires the job to carry supervisor-issued reservation fields;
    otherwise raises (missing authority is never synthesized).
    """
    if not getattr(job, "reservation_id", None) \
            or not getattr(job, "allocation_id", None) \
            or getattr(job, "reservation_deadline_unix", None) is None:
        raise SessionError(
            "job carries no supervisor reservation fields "
            "(allocation/reservation/deadline required); refusing")
    return {"allocation_id": job.allocation_id,
            "reservation_id": job.reservation_id,
            "job_id": job.job_id,
            "device": job.physical_device,
            "phase": job.phase,
            "deadline_unix": float(deadline_unix)
            if deadline_unix is not None
            else float(job.reservation_deadline_unix),
            "remaining_updates": int(remaining_updates),
            "source_hash": job.source_hash or "unbound-source"}


def check_update_admission(handle: Any) -> dict[str, Any]:
    """Prove the optimizer boundary would admit (no step taken).

    Runs the exact admission checks `finalize_update` enforces
    (live reservation, remaining allowance, live deadline) without consuming
    an update. Local acceptance uses this plus a real backward, then discards
    gradients: authority reaches the boundary with zero optimizer steps.
    """
    trainer = handle.get("trainer") if isinstance(handle, dict) else getattr(
        handle, "trainer", None)
    if trainer is None or not hasattr(trainer, "_admit_update"):
        raise SessionError("handle owns no admission-checked trainer")
    trainer._admit_update()
    try:
        consumed = int(trainer.counters.optimizer_updates)
    except Exception:
        consumed = 0
    return {"admitted": True,
            "optimizer_updates": consumed,
            "step_taken": False}


def drive_noop_boundary(handle: Any, *, batch: Any, window: Any,
                        extra: dict[str, Any] | None = None,
                        pair_rows: Any | None = None) -> dict[str, Any]:
    """Accumulate + backward, prove admission, discard (no optimizer step).

    The local no-op substitute for the optimizer boundary: real target
    construction happened upstream, real backward runs here, admission is
    enforced, then gradients are discarded with no schedule advance and no
    counter increment. Returns the boundary proof.
    """
    trainer = handle.get("trainer") if isinstance(handle, dict) else getattr(
        handle, "trainer", None)
    if trainer is None:
        raise SessionError("handle owns no trainer for the noop boundary")
    before = int(trainer.counters.optimizer_updates)
    trainer.accumulate_full_window(
        batch, window_builder=lambda _: window,
        extra_terms_fn=(lambda: extra) if extra else None,
        pair_rows=pair_rows)
    proof = check_update_admission(handle)
    named_grads: dict[str, bool] = {}
    try:
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                import torch as _torch

                named_grads[name] = bool(
                    _torch.isfinite(param.grad).all().item())
    except Exception:
        pass
    trainer.optimizer.zero_grad(set_to_none=True)
    try:
        trainer._clear_pending()
    except Exception:
        trainer._pending_targets = 0
    after = int(trainer.counters.optimizer_updates)
    if after != before:
        raise SessionError("noop boundary consumed an update; refusing")
    return {**proof, "had_pending_targets": True,
            "named_grads_finite": named_grads,
            "optimizer_updates_before": before,
            "optimizer_updates_after": after}
