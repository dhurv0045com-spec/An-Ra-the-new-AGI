"""Campaign supervisor (I05/R02): immutable allocation, unique idempotent
jobs, per-device occupancy, transactional capacity checks and partial-work
accounting. The supervisor is the only accounting writer.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from bramastra_lab.research.contracts.core import content_identity

LEDGER_SCHEMA = "bramastra-k8-ledger/v2"


class SupervisorError(RuntimeError):
    """The supervisor refused an operation."""


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    job_id: str
    parent_job: str | None
    worker: str
    device: str
    phase: str
    arm: str | None
    seed: int | None
    reserved_seconds: float
    reserved_at_unix: float
    status: str
    committed_updates: int
    attempted_updates: int
    supervised_exposure: int
    device_seconds: float
    checkpoint_identity: str | None

    def to_row(self) -> tuple:
        return (self.reservation_id, self.job_id, self.parent_job, self.worker,
                self.device, self.phase, self.arm, self.seed,
                self.reserved_seconds, self.reserved_at_unix, self.status,
                self.committed_updates, self.attempted_updates,
                self.supervised_exposure, self.device_seconds,
                self.checkpoint_identity)


class CampaignLedger:
    """SQLite journal with immutable allocation, unique jobs, per-device
    occupancy and transactional capacity checks."""

    def __init__(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "campaign_ledger.sqlite")
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS allocation (
                allocation_id TEXT PRIMARY KEY,
                source_hash TEXT NOT NULL,
                data_hash TEXT NOT NULL DEFAULT '',
                started_utc TEXT NOT NULL,
                deadline_unix REAL NOT NULL,
                max_wall_minutes REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reservations (
                reservation_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL UNIQUE,
                parent_job TEXT,
                worker TEXT, device TEXT, phase TEXT, arm TEXT, seed INTEGER,
                reserved_seconds REAL, reserved_at_unix REAL, status TEXT,
                committed_updates INTEGER, attempted_updates INTEGER,
                supervised_exposure INTEGER, device_seconds REAL,
                checkpoint_identity TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                event_index INTEGER PRIMARY KEY AUTOINCREMENT,
                at_unix REAL NOT NULL,
                event TEXT NOT NULL,
                payload TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def record_allocation(self, allocation_id: str, source_hash: str,
                          data_hash: str, max_wall_minutes: float) -> float:
        """Create or validate. The deadline never moves once recorded.

        A run directory binds to exactly one immutable allocation: a second
        distinct allocation_id in the same directory is rejected (R02). An
        independently authorized new campaign must use a new directory.
        e0->full transitions share the same derived allocation_id and retain
        the exact original deadline.
        """
        existing = self.conn.execute(
            "SELECT source_hash, data_hash, max_wall_minutes, deadline_unix "
            "FROM allocation WHERE allocation_id=?",
            (allocation_id,)).fetchone()
        if existing is not None:
            if existing[0] != source_hash or existing[1] != data_hash:
                raise SupervisorError(
                    f"allocation {allocation_id!r} created with different "
                    "source/data hashes; conflicting identity rejected")
            if existing[2] != max_wall_minutes:
                raise SupervisorError(
                    f"allocation {allocation_id!r} max_wall_minutes="
                    f"{existing[2]} differs from {max_wall_minutes}")
            return existing[3]
        # Immutable binding: one allocation per run directory.
        other = self.conn.execute(
            "SELECT allocation_id FROM allocation LIMIT 1").fetchone()
        if other is not None:
            raise SupervisorError(
                f"run directory already bound to allocation {other[0]!r}; "
                f"refusing second allocation {allocation_id!r} in the same "
                "directory (use a new run directory for a new campaign)")
        started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        deadline = time.time() + max_wall_minutes * 60.0
        self.conn.execute(
            "INSERT INTO allocation VALUES (?,?,?,?,?,?)",
            (allocation_id, source_hash, data_hash, started_utc, deadline,
             max_wall_minutes))
        self.conn.commit()
        self.append_event("allocation_created", {
            "allocation_id": allocation_id, "started_utc": started_utc,
            "deadline_unix": deadline})
        return deadline

    def deadline(self) -> float | None:
        row = self.conn.execute(
            "SELECT deadline_unix FROM allocation ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def _campaign_start(self) -> float:
        row = self.conn.execute(
            "SELECT started_utc FROM allocation ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if not row:
            return 0.0
        import calendar
        return calendar.timegm(time.strptime(row[0], "%Y-%m-%dT%H:%M:%SZ"))

    def reserve(self, job_id: str, *, worker: str, device: str, phase: str,
                arm: str | None = None, seed: int | None = None,
                reserved_seconds: float = 0.0,
                parent_job: str | None = None) -> Reservation:
        """Reserve a bounded time slice. Unique job_id (idempotent retry
        returns the existing reservation only for compatible retries).

        Compatible retry: same job_id with different worker/device/phase/arm/
        seed/parent_job raises (R02). Per-device occupancy: overlapping open
        reservations on the SAME device must not exceed the remaining campaign
        time; two different devices may each reserve up to the remaining time
        concurrently. Transactionally checked via the UNIQUE(job_id)
        constraint plus explicit capacity validation.
        """
        existing = self.conn.execute(
            "SELECT * FROM reservations WHERE job_id=?", (job_id,)).fetchone()
        if existing is not None:
            prior = self._row_to_reservation(existing)
            mismatches = []
            if prior.worker != worker:
                mismatches.append(f"worker {prior.worker!r}!={worker!r}")
            if prior.device != device:
                mismatches.append(f"device {prior.device!r}!={device!r}")
            if prior.phase != phase:
                mismatches.append(f"phase {prior.phase!r}!={phase!r}")
            if prior.arm != arm:
                mismatches.append(f"arm {prior.arm!r}!={arm!r}")
            if prior.seed != seed:
                mismatches.append(f"seed {prior.seed!r}!={seed!r}")
            if prior.parent_job != parent_job:
                mismatches.append(
                    f"parent_job {prior.parent_job!r}!={parent_job!r}")
            if mismatches:
                raise SupervisorError(
                    f"incompatible retry for job {job_id!r}: "
                    + "; ".join(mismatches))
            return prior
        deadline = self.deadline()
        now = time.time()
        if deadline is not None and now + reserved_seconds > deadline:
            raise SupervisorError(
                f"reservation of {reserved_seconds}s exceeds the campaign "
                f"deadline ({deadline - now:.0f}s remain)")
        # Per-device occupancy: sum open reservations on THIS device only.
        device_open = self.conn.execute(
            "SELECT COALESCE(SUM(reserved_seconds),0) FROM reservations "
            "WHERE status='open' AND device=?", (device,)).fetchone()[0]
        remaining = (deadline - now) if deadline else float("inf")
        if remaining > 0 and device_open + reserved_seconds > remaining:
            raise SupervisorError(
                f"device {device!r} over-subscribed: open {device_open:.0f}s + "
                f"requested {reserved_seconds:.0f}s exceeds {remaining:.0f}s "
                "remaining (per-device exclusivity)")
        reservation = Reservation(
            reservation_id=uuid.uuid4().hex, job_id=job_id,
            parent_job=parent_job, worker=worker, device=device, phase=phase,
            arm=arm, seed=seed, reserved_seconds=reserved_seconds,
            reserved_at_unix=now, status="open", committed_updates=0,
            attempted_updates=0, supervised_exposure=0, device_seconds=0.0,
            checkpoint_identity=None)
        self.conn.execute(
            "INSERT INTO reservations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            reservation.to_row())
        self.conn.commit()
        return reservation

    def close_reservation(self, reservation_id: str, *, status: str,
                          committed_updates: int = 0, attempted_updates: int = 0,
                          supervised_exposure: int = 0, device_seconds: float = 0.0,
                          checkpoint_identity: str | None = None) -> Reservation:
        """Close with actual consumption. Failed jobs retain partial work."""
        if status not in ("completed", "failed", "timed_out"):
            raise SupervisorError(f"invalid closing status {status!r}")
        row = self.conn.execute(
            "SELECT * FROM reservations WHERE reservation_id=?",
            (reservation_id,)).fetchone()
        if row is None:
            raise SupervisorError(f"unknown reservation {reservation_id!r}")
        if row[10] != "open":
            if row[10] == status:
                return self._row_to_reservation(row)
            raise SupervisorError(
                f"reservation {reservation_id!r} already closed as {row[10]!r}")
        self.conn.execute(
            "UPDATE reservations SET status=?, committed_updates=?, "
            "attempted_updates=?, supervised_exposure=?, device_seconds=?, "
            "checkpoint_identity=? WHERE reservation_id=?",
            (status, committed_updates, attempted_updates, supervised_exposure,
             device_seconds, checkpoint_identity, reservation_id))
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM reservations WHERE reservation_id=?",
            (reservation_id,)).fetchone()
        return self._row_to_reservation(row)

    @staticmethod
    def _row_to_reservation(row: tuple) -> Reservation:
        return Reservation(*row)

    def phase_receipts(self, phase: str) -> list[Reservation]:
        rows = self.conn.execute(
            "SELECT * FROM reservations WHERE phase=? AND status='completed'",
            (phase,)).fetchall()
        return [self._row_to_reservation(row) for row in rows]

    @staticmethod
    def _receipt_qualified(reservation: Reservation) -> bool:
        """A qualified receipt carries actual E0 evidence, not a bare status.

        Requires: completed status (caller filters), non-null checkpoint
        identity, positive committed AND attempted updates, positive exposure,
        positive device time, and a concrete device string. Two zero-work
        receipts on the same GPU must NOT pass (R01/R02).
        """
        if reservation.status != "completed":
            return False
        if not reservation.checkpoint_identity:
            return False
        if reservation.committed_updates <= 0:
            return False
        if reservation.attempted_updates <= 0:
            return False
        if reservation.supervised_exposure <= 0:
            return False
        if reservation.device_seconds <= 0:
            return False
        if not reservation.device:
            return False
        return True

    def qualified_phase_receipts(self, phase: str) -> list[Reservation]:
        """Completed receipts with actual outcomes for this phase."""
        return [row for row in self.phase_receipts(phase)
                if self._receipt_qualified(row)]

    def phase_success(self, phase: str, *, required_workers: int = 2) -> bool:
        """Admission gate: distinct qualified workers/devices, not row count.

        Requires at least `required_workers` qualified receipts on DISTINCT
        devices. Two zero-work receipts for w0/cuda:0 fail (same device and
        unqualified). Source/data/configuration identity is bound by the
        immutable run-directory allocation (R02).
        """
        qualified = self.qualified_phase_receipts(phase)
        distinct_devices = {row.device for row in qualified}
        return len(qualified) >= required_workers \
            and len(distinct_devices) >= required_workers

    def phase_consumption(self, phase: str) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT committed_updates, attempted_updates, device_seconds, "
            "supervised_exposure FROM reservations WHERE phase=?",
            (phase,)).fetchall()
        return {"committed_updates": sum(r[0] for r in rows),
                "attempted_updates": sum(r[1] for r in rows),
                "device_seconds": sum(r[2] for r in rows),
                "supervised_exposure": sum(r[3] for r in rows)}

    def append_event(self, event: str, payload: Mapping[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO events (at_unix, event, payload) VALUES (?,?,?)",
            (time.time(), event, json.dumps(payload, sort_keys=True)))
        self.conn.commit()

    def export(self) -> dict[str, Any]:
        allocations = self.conn.execute("SELECT * FROM allocation").fetchall()
        reservations = self.conn.execute("SELECT * FROM reservations").fetchall()
        events = self.conn.execute(
            "SELECT event_index, at_unix, event, payload FROM events "
            "ORDER BY event_index").fetchall()
        return {
            "schema": LEDGER_SCHEMA,
            "allocations": allocations,
            "reservations": [self._row_to_reservation(row).to_row()
                             for row in reservations],
            "events": [{"index": row[0], "at_unix": row[1], "event": row[2],
                        "payload": json.loads(row[3])} for row in events],
        }


class SupervisorLease:
    """Exclusive supervisor lease via an exclusive lock file with verified
    stale-lease recovery (R02).

    acquire() refuses when another live supervisor holds the lease. A lock
    whose writer process is gone (PID check) or whose mtime exceeds
    `stale_after_seconds` is treated as stale: it is removed and acquisition
    retries once, recording the recovery. Close database resources on every
    path (runner closes its ledger in finally).
    """

    def __init__(self, run_dir: str) -> None:
        self.path = os.path.join(run_dir, "supervisor.lock")

    def _lock_probe(self) -> tuple[str | None, float | None]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                content = handle.read().strip()
        except OSError:
            return None, None
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            mtime = None
        return content or None, mtime

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            # Signal 0 probes liveness without delivering a signal.
            # On Windows, os.kill exists for termination; fall back to mtime.
            os.kill(pid, 0)  # type: ignore[attr-defined]
        except AttributeError:
            return True  # Cannot probe; treat as alive, rely on mtime.
        except OSError:
            return False
        except Exception:
            return True
        return True

    def acquire(self, *, stale_after_seconds: float = 1800.0) -> str:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            content, mtime = self._lock_probe()
            pid: int | None = None
            if content and ":" in content:
                try:
                    pid = int(content.split(":")[0])
                except ValueError:
                    pid = None
            stale = False
            reason = ""
            if mtime is not None and (time.time() - mtime) > stale_after_seconds:
                stale = True
                reason = f"mtime age exceeds {stale_after_seconds:.0f}s"
            elif pid is not None and not self._pid_alive(pid):
                stale = True
                reason = f"writer pid {pid} is gone"
            if not stale:
                raise SupervisorError(
                    "another live supervisor holds the lease; recovery must prove "
                    "the prior writer is gone (remove the lock explicitly)")
            # Verified stale: recover once.
            try:
                os.remove(self.path)
            except OSError as exc:
                raise SupervisorError(
                    f"stale lease detected ({reason}) but removal failed: {exc}"
                ) from exc
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:
                raise SupervisorError(
                    "stale lease recovered but another writer won the race; "
                    "refusing to run concurrently") from exc
        with os.fdopen(fd, "w") as handle:
            handle.write(f"{os.getpid()}:{time.time():.6f}")
        return f"{os.getpid()}:{time.time():.6f}"

    def release(self) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass
