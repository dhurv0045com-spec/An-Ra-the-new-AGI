"""
autonomy/decisions.py — Autonomous Decision Framework

Four-tier autonomy system. Every autonomous decision is logged and auditable.
Owner configures tier boundaries. System defaults conservative on ambiguity.
"""

import uuid, json, sqlite3, threading
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from enum import IntEnum


STATE_DIR = Path("state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH   = STATE_DIR / "decisions.db"


class Tier(IntEnum):
    """
    TIER 1 — Fully autonomous: routine, low-stakes, reversible.
    TIER 2 — Notify + proceed unless stopped within timeout.
    TIER 3 — Wait for explicit approval.
    TIER 4 — NEVER do without direct instruction.
    """
    ONE   = 1
    TWO   = 2
    THREE = 3
    FOUR  = 4


@dataclass
class DecisionRequest:
    request_id:   str
    timestamp:    str
    title:        str
    description:  str
    proposed_action: str
    tier:         int
    category:     str
    reversible:   bool
    stakes:       str   # "low" / "medium" / "high" / "critical"
    status:       str = "pending"   # pending / approved / rejected / timeout / executed / cancelled
    decided_at:   Optional[str]  = None
    decided_by:   str            = "system"
    outcome:      Optional[str]  = None
    metadata:     Dict[str, Any] = field(default_factory=dict)


@dataclass
class TierPolicy:
    category:        str
    assigned_tier:   int
    description:     str
    notify_on_exec:  bool = True
    timeout_seconds: int  = 300   # for TIER 2: how long to wait before proceeding


class DecisionDB:
    def __init__(self, path=DB_PATH):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init()

    def _init(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS decisions (
                    request_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE TABLE IF NOT EXISTS tier_policies (
                    category TEXT PRIMARY KEY, data TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_dec_ts ON decisions(request_id);
            """)
            self._conn.commit()

    def save_decision(self, d: DecisionRequest):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO decisions VALUES (?,?)",
                               (d.request_id, json.dumps(asdict(d))))
            self._conn.commit()

    def get_decision(self, request_id: str) -> Optional[DecisionRequest]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM decisions WHERE request_id=?", (request_id,)).fetchone()
        return DecisionRequest(**json.loads(row[0])) if row else None

    def get_pending(self) -> List[DecisionRequest]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM decisions").fetchall()
        all_d = [DecisionRequest(**json.loads(r[0])) for r in rows]
        return [d for d in all_d if d.status == "pending"]

    def save_policy(self, p: TierPolicy):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO tier_policies VALUES (?,?)",
                               (p.category, json.dumps(asdict(p))))
            self._conn.commit()

    def get_policies(self) -> Dict[str, TierPolicy]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM tier_policies").fetchall()
        return {p.category: p for p in (TierPolicy(**json.loads(r[0])) for r in rows)}

    def audit_trail(self, limit=100) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM decisions ORDER BY request_id DESC LIMIT ?",
                (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]


class DecisionFramework:
    """
    Manages autonomous decisions with configurable tier policies.
    Every decision is logged with full context for later inspection.
    """

    DEFAULT_POLICIES = [
        TierPolicy("file_read",         Tier.ONE,   "Read local files — always autonomous"),
        TierPolicy("memory_write",      Tier.ONE,   "Write to memory — always autonomous"),
        TierPolicy("web_search",        Tier.ONE,   "Search the web — always autonomous"),
        TierPolicy("tool_use",          Tier.ONE,   "Use existing tools — autonomous"),
        TierPolicy("file_write",        Tier.TWO,   "Write/modify files — notify owner"),
        TierPolicy("goal_update",       Tier.TWO,   "Update goal plan — notify owner"),
        TierPolicy("training_run",      Tier.TWO,   "Run model training — notify owner"),
        TierPolicy("external_api",      Tier.TWO,   "Call external API — notify owner"),
        TierPolicy("install_package",   Tier.THREE, "Install software — need approval"),
        TierPolicy("delete_file",       Tier.THREE, "Delete files — need approval"),
        TierPolicy("model_deploy",      Tier.THREE, "Deploy new model — need approval"),
        TierPolicy("send_message",      Tier.THREE, "Send message/email — need approval"),
        TierPolicy("financial",         Tier.FOUR,  "Any financial action — NEVER autonomous"),
        TierPolicy("credential_access", Tier.FOUR,  "Access credentials — NEVER autonomous"),
        TierPolicy("irreversible_delete",Tier.FOUR, "Permanent delete — NEVER autonomous"),
    ]

    def __init__(self, notify_cb: Optional[Callable] = None):
        self.db         = DecisionDB()
        self._notify_cb = notify_cb   # called when owner needs notification
        self._pending_timers: Dict[str, threading.Timer] = {}
        self._init_policies()

    def _init_policies(self):
        existing = self.db.get_policies()
        for p in self.DEFAULT_POLICIES:
            if p.category not in existing:
                self.db.save_policy(p)

    def _classify(self, category: str, reversible: bool, stakes: str) -> int:
        """Determine tier for a decision based on policy + context."""
        policies = self.db.get_policies()

        # Explicit policy match
        if category in policies:
            base_tier = policies[category].assigned_tier
        else:
            # Default by stakes
            base_tier = {"low": Tier.ONE, "medium": Tier.TWO,
                         "high": Tier.THREE, "critical": Tier.FOUR}.get(stakes, Tier.THREE)

        # Escalate if not reversible
        if not reversible and base_tier < Tier.THREE:
            base_tier = min(base_tier + 1, Tier.THREE)

        return int(base_tier)

    def request(
        self,
        title:           str,
        description:     str,
        proposed_action: str,
        category:        str,
        reversible:      bool  = True,
        stakes:          str   = "low",
        execute_fn:      Optional[Callable] = None,
        metadata:        dict  = None,
    ) -> DecisionRequest:
        """
        Submit a decision for processing.
        Returns the DecisionRequest (check .status to see what happened).
        """
        tier = self._classify(category, reversible, stakes)
        req  = DecisionRequest(
            request_id      = str(uuid.uuid4()),
            timestamp       = datetime.utcnow().isoformat(),
            title           = title,
            description     = description,
            proposed_action = proposed_action,
            tier            = tier,
            category        = category,
            reversible      = reversible,
            stakes          = stakes,
            metadata        = metadata or {},
        )

        if tier == Tier.FOUR:
            req.status   = "cancelled"
            req.decided_by = "policy"
            req.outcome  = "TIER 4: Never autonomous. Direct instruction required."
            self.db.save_decision(req)
            return req

        if tier == Tier.ONE:
            req.status   = "approved"
            req.decided_by = "autonomous"
            req.decided_at = datetime.utcnow().isoformat()
            self.db.save_decision(req)
            if execute_fn:
                self._execute(req, execute_fn)
            return req

        if tier == Tier.TWO:
            req.status = "pending"
            self.db.save_decision(req)
            self._notify_tier2(req, execute_fn)
            return req

        # Tier 3: require explicit approval
        req.status = "pending"
        self.db.save_decision(req)
        self._notify_tier3(req)
        return req

    def _execute(self, req: DecisionRequest, fn: Callable):
        try:
            result = fn()
            req.outcome = str(result)[:500] if result else "executed"
            req.status  = "executed"
        except Exception as e:
            req.outcome = f"ERROR: {e}"
            req.status  = "error"
        req.decided_at = datetime.utcnow().isoformat()
        self.db.save_decision(req)

    def _notify_tier2(self, req: DecisionRequest, execute_fn: Optional[Callable]):
        policies = self.db.get_policies()
        timeout  = policies.get(req.category, TierPolicy(req.category, 2, "", timeout_seconds=300)).timeout_seconds

        if self._notify_cb:
            self._notify_cb({
                "type":    "tier2_action",
                "request": asdict(req),
                "message": f"[TIER 2] Will execute in {timeout}s unless stopped: {req.title}",
            })

        # Auto-execute after timeout unless owner rejects
        def _auto_execute():
            current = self.db.get_decision(req.request_id)
            if current and current.status == "pending":
                current.status    = "approved"
                current.decided_by = "timeout"
                self.db.save_decision(current)
                if execute_fn:
                    self._execute(current, execute_fn)

        timer = threading.Timer(timeout, _auto_execute)
        timer.daemon = True
        timer.start()
        self._pending_timers[req.request_id] = timer

    def _notify_tier3(self, req: DecisionRequest):
        if self._notify_cb:
            self._notify_cb({
                "type":    "tier3_approval_needed",
                "request": asdict(req),
                "message": f"[TIER 3] Awaiting your approval: {req.title}",
            })

    def approve(self, request_id: str, execute_fn: Optional[Callable] = None) -> bool:
        req = self.db.get_decision(request_id)
        if not req or req.status != "pending":
            return False
        req.status    = "approved"
        req.decided_by = "owner"
        req.decided_at = datetime.utcnow().isoformat()
        self.db.save_decision(req)
        if request_id in self._pending_timers:
            self._pending_timers[request_id].cancel()
        if execute_fn:
            self._execute(req, execute_fn)
        return True

    def reject(self, request_id: str, reason: str = "") -> bool:
        req = self.db.get_decision(request_id)
        if not req or req.status != "pending":
            return False
        if request_id in self._pending_timers:
            self._pending_timers[request_id].cancel()
        req.status    = "rejected"
        req.decided_by = "owner"
        req.decided_at = datetime.utcnow().isoformat()
        req.outcome   = reason
        self.db.save_decision(req)
        return True

    def set_tier_policy(self, category: str, tier: int, description: str = ""):
        policy = TierPolicy(category, tier, description)
        self.db.save_policy(policy)

    def audit_trail(self, limit: int = 100) -> List[dict]:
        return self.db.audit_trail(limit)

    def pending_approvals(self) -> List[DecisionRequest]:
        return [d for d in self.db.get_pending() if d.tier >= Tier.THREE]
