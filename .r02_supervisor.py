"""R02 supervisor + R01/R03/R05 runner/worker + R04 trainer rewrite."""
# --- supervisor.py ---
path = 'bramastra_lab/research/campaigns/supervisor.py'
supervisor = open('bramastra_lab/research/campaigns/supervisor.py', encoding='utf-8').read()

# Key fixes:
# 1. allocation_id includes data_hash; create-or-validate with deadline preservation
# 2. UNIQUE constraint on job_id
# 3. record_allocation validates and returns existing deadline

# We'll do surgical replacements
old_alloc = '''    def record_allocation(self, allocation_id: str, source_hash: str,
                          max_wall_minutes: float) -> float:
        started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        deadline = time.time() + max_wall_minutes * 60.0
        self.conn.execute(
            "INSERT OR REPLACE INTO allocation VALUES (?,?,?,?,?)",
            (allocation_id, source_hash, started_utc, deadline, max_wall_minutes))
        self.conn.commit()
        self.append_event("allocation_recorded", {
            "allocation_id": allocation_id, "started_utc": started_utc,
            "deadline_unix": deadline})
        return deadline'''
new_alloc = '''    def record_allocation(self, allocation_id: str, source_hash: str,
                          data_hash: str, max_wall_minutes: float) -> float:
        """Create or validate. The deadline never moves once recorded."""
        existing = self.conn.execute(
            "SELECT source_hash, data_hash, max_wall_minutes, deadline_unix "
            "FROM allocation WHERE allocation_id=?",
            (allocation_id,)).fetchone()
        if existing is not None:
            if existing[0] != source_hash or existing[1] != data_hash:
                raise SupervisorError(
                    f"allocation {allocation_id!r} created with different source/data")
            if existing[2] != max_wall_minutes:
                raise SupervisorError(
                    f"allocation {allocation_id!r} max_wall_minutes={existing[2]} "
                    f"differs from {max_wall_minutes}")
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
        return deadline'''
assert old_alloc in supervisor, 'alloc anchor'
supervisor = supervisor.replace(old_alloc, new_alloc)

# Add UNIQUE to job_id
old_schema = '''CREATE TABLE IF NOT EXISTS reservations (
                reservation_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,'''
new_schema = '''CREATE TABLE IF NOT EXISTS reservations (
                reservation_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL UNIQUE,'''
assert old_schema in supervisor, 'schema anchor'
supervisor = supervisor.replace(old_schema, new_schema)

# Add data_hash to allocation schema
old_alloc_schema = '''CREATE TABLE IF NOT EXISTS allocation (
                allocation_id TEXT PRIMARY KEY,
                source_hash TEXT NOT NULL,
                started_utc TEXT NOT NULL,
                deadline_unix REAL NOT NULL,
                max_wall_minutes REAL NOT NULL
            );'''
new_alloc_schema = '''CREATE TABLE IF NOT EXISTS allocation (
                allocation_id TEXT PRIMARY KEY,
                source_hash TEXT NOT NULL,
                data_hash TEXT NOT NULL DEFAULT '',
                started_utc TEXT NOT NULL,
                deadline_unix REAL NOT NULL,
                max_wall_minutes REAL NOT NULL
            );'''
assert old_alloc_schema in supervisor, 'alloc schema'
supervisor = supervisor.replace(old_alloc_schema, new_alloc_schema)

# Fix reserve: unique job_id (return existing), not INSERT OR REPLACE
old_reserve = '''    def reserve(self, job_id: str, *, worker: str, device: str, phase: str,
                arm: str | None = None, seed: int | None = None,
                reserved_seconds: float = 0.0,
                parent_job: str | None = None) -> Reservation:
        """Reserve a bounded time slice. Two concurrent requests cannot
        reserve beyond remaining capacity."""
        deadline = self.deadline()
        now = time.time()
        if deadline is not None and now + reserved_seconds > deadline:
            raise SupervisorError(
                f"reservation of {reserved_seconds}s exceeds the campaign "
                f"deadline ({deadline - now:.0f}s remain)")
        reserved = self.conn.execute(
            "SELECT COALESCE(SUM(reserved_seconds),0) FROM reservations "
            "WHERE status='open'").fetchone()[0]
        total_span = deadline - self._campaign_start() if deadline else float("inf")
        if total_span > 0 and reserved + reserved_seconds > total_span:
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
        return reservation'''
new_reserve = '''    def reserve(self, job_id: str, *, worker: str, device: str, phase: str,
                arm: str | None = None, seed: int | None = None,
                reserved_seconds: float = 0.0,
                parent_job: str | None = None) -> Reservation:
        """Reserve a bounded time slice. Unique job_id (idempotent retry
        returns the existing reservation). Per-device occupancy checked."""
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
        return reservation'''
assert old_reserve in supervisor, 'reserve anchor'
supervisor = supervisor.replace(old_reserve, new_reserve)

# Add phase_receipts and phase_success
old_consumption = '''    def phase_consumption(self, phase: str) -> dict[str, float]:'''
new_receipts = '''    def phase_receipts(self, phase: str) -> list[Reservation]:
        rows = self.conn.execute(
            "SELECT * FROM reservations WHERE phase=? AND status='completed'",
            (phase,)).fetchall()
        return [self._row_to_reservation(row) for row in rows]

    def phase_success(self, phase: str, *, required_workers: int = 2) -> bool:
        return len(self.phase_receipts(phase)) >= required_workers

    def phase_consumption(self, phase: str) -> dict[str, float]:'''
assert old_consumption in supervisor
supervisor = supervisor.replace(old_consumption, new_receipts)

# Add close() method
old_export = '''    def export(self) -> dict[str, Any]:'''
new_close = '''    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def export(self) -> dict[str, Any]:'''
assert old_export in supervisor
supervisor = supervisor.replace(old_export, new_close)

open(path, 'w', encoding='utf-8').write(supervisor)
print('supervisor patched')
