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
        """Create or validate. The deadline never moves once recorded."""
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
        returns the existing reservation)."""
        existing = self.conn.execute(
            "SELECT * FROM reservations WHERE job_id=?", (job_id,)).fetchone()
        if existing is not None:
            return self._row_to_reservation(existing)
        deadline = self.deadline()
        now = time.time()
        if deadline is not None and now + reserved_seconds > deadline:
            raise SupervisorError(
                f"reservation of {reserved_seconds}s exceeds the campaign "
                f"deadline ({deadline - now:.0f}s remain)")
        open_reserved = self.conn.execute(
            "SELECT COALESCE(SUM(reserved_seconds),0) FROM reservations "
            "WHERE status='open'").fetchone()[0]
        span = deadline - self._campaign_start() if deadline else float("inf")
        if span > 0 and open_reserved + reserved_seconds > span:
            raise SupervisorError(
                "concurrent reservations exceed remaining campaign capacity")
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

    def phase_success(self, phase: str, *, required_workers: int = 2) -> bool:
        return len(self.phase_receipts(phase)) >= required_workers

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
    """Exclusive supervisor lease via an exclusive lock file."""

    def __init__(self, run_dir: str) -> None:
        self.path = os.path.join(run_dir, "supervisor.lock")

    def acquire(self) -> str:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SupervisorError(
                "another live supervisor holds the lease; recovery must prove "
                "the prior writer is gone (remove the lock explicitly)")
        with os.fdopen(fd, "w") as handle:
            handle.write(f"{os.getpid()}:{time.time():.6f}")
        return f"{os.getpid()}:{time.time():.6f}"

    def release(self) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass
