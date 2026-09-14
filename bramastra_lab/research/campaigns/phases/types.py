"""Typed job input and phase result for K8 executors (D4 + real execution contracts).

Contracts (K8_SEMANTIC_REVIEW_20260914/EXECUTOR_CONTRACTS.md S1):
job carries immutable job/slot/phase/arm/seed IDs, physical+local devices,
source/data/tokenizer/config/protocol hashes, reservation, deadline and an
explicit update target or time-budget rule. Parent is a verified checkpoint
reference, not a bare lookup string. Receipts derive counts from durable
events and declare evidence kind; fixture receipts never enter campaign
aggregates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EVIDENCE_FIXTURE = "fixture"
EVIDENCE_LOCAL_INTEGRATION = "local-integration"
EVIDENCE_LEARNED_CAMPAIGN = "learned-campaign"
EVIDENCE_KINDS = frozenset({EVIDENCE_FIXTURE, EVIDENCE_LOCAL_INTEGRATION,
                            EVIDENCE_LEARNED_CAMPAIGN})


@dataclass(frozen=True)
class ParentRef:
    """Verified parent checkpoint reference (contracts S1).

    `lookup_key` (e.g. "E1-B-1701") is only a lookup key. `resolve()` must
    return a verified checkpoint record (manifest + payload hash + identities)
    or fail. Callers must never treat the key as model state.
    """

    lookup_key: str | None = None
    checkpoint_id: str | None = None
    manifest_path: str | None = None
    payload_sha256: str | None = None
    config_identity: str | None = None
    architecture_id: str | None = None
    lineage: str | None = None

    def validate(self) -> None:
        if not self.lookup_key and not self.checkpoint_id:
            raise ValueError("ParentRef requires lookup_key or checkpoint_id")

    def resolve(self, run_dir: str) -> dict[str, Any]:
        """Resolve to a verified checkpoint record or raise.

        Searches the campaign ledger for a completed receipt matching the
        lookup key, then loads and hash-validates the payload via the real
        checkpoint API. Reinitializing with the same seed is forbidden.
        """
        self.validate()
        from bramastra_lab.research.runtime.checkpoint import (
            CheckpointError, load_checkpoint)
        import os
        import sqlite3

        wanted = self.checkpoint_id or self.lookup_key or ""
        # Direct checkpoint-ID load: still requires a completed ledger receipt
        # referencing that exact checkpoint (never an arbitrary payload).
        if self.checkpoint_id and len(self.checkpoint_id) == 64:
            import sqlite3 as _sqlite

            _ledger = os.path.join(run_dir, "campaign_ledger.sqlite")
            if not os.path.exists(_ledger):
                raise ValueError(
                    f"parent checkpoint {self.checkpoint_id[:12]}... has no ledger; "
                    "missing evidence must fail")
            _conn = _sqlite.connect(_ledger)
            try:
                _hits = _conn.execute(
                    "SELECT job_id FROM reservations WHERE status='completed' "
                    "AND checkpoint_identity=? ORDER BY rowid",
                    (self.checkpoint_id,)).fetchall()
            finally:
                _conn.close()
            if not _hits:
                raise ValueError(
                    f"parent checkpoint {self.checkpoint_id[:12]}... has no "
                    "completed ledger receipt; refusing arbitrary payload")
            try:
                _payload, manifest = load_checkpoint(
                    run_dir, checkpoint_id=self.checkpoint_id,
                    expect_config_identity=self.config_identity,
                    expect_data_identity=None)
            except Exception as exc:
                raise ValueError(
                    f"parent checkpoint {self.checkpoint_id[:12]}... failed "
                    f"verification: {exc}") from exc
            if self.payload_sha256 and manifest.payload_sha256 != self.payload_sha256:
                raise ValueError("parent payload hash mismatch; refusing")
            return {"checkpoint_id": manifest.checkpoint_id,
                    "manifest": manifest.to_dict(),
                    "payload_sha256": manifest.payload_sha256,
                    "config_identity": manifest.config_identity,
                    "lineage": self.lineage,
                    "run_dir": run_dir}
        # Lookup-key path: exact job_id or checkpoint_id match only (never
        # substring: E1-B-1701 must not match E1-B-17010 or combined A/B keys).
        ledger_path = os.path.join(run_dir, "campaign_ledger.sqlite")
        if not os.path.exists(ledger_path):
            raise ValueError(
                f"parent {wanted!r} has no ledger; missing parent must fail "
                "(reinitializing with the same seed is forbidden)")
        conn = sqlite3.connect(ledger_path)
        try:
            rows = conn.execute(
                "SELECT job_id, checkpoint_identity FROM reservations "
                "WHERE status='completed' ORDER BY rowid").fetchall()
        finally:
            conn.close()
        candidates = [(jid, cid) for jid, cid in rows
                      if cid and (str(jid) == wanted or str(cid) == wanted)]
        if not candidates:
            raise ValueError(
                f"parent {wanted!r} has no qualified completed receipt with exact "
                "lineage; missing parent must fail")
        # Exact matches only; multiple exact hits (retries) use the latest.
        _job_id, checkpoint_id = candidates[-1]
        try:
            _payload, manifest = load_checkpoint(
                run_dir, checkpoint_id=checkpoint_id,
                expect_config_identity=self.config_identity)
        except Exception as exc:
            raise ValueError(
                f"parent {wanted!r} checkpoint {str(checkpoint_id)[:12]}... "
                f"failed verification: {exc}") from exc
        if self.payload_sha256 and manifest.payload_sha256 != self.payload_sha256:
            raise ValueError("parent payload hash mismatch; refusing")
        return {"checkpoint_id": manifest.checkpoint_id,
                "manifest": manifest.to_dict(),
                "payload_sha256": manifest.payload_sha256,
                "config_identity": manifest.config_identity,
                "lineage": self.lineage,
                "run_dir": run_dir}


@dataclass(frozen=True)
class JobInput:
    """One typed phase job (D4 + contracts S1)."""

    phase: str
    slot: int | None
    arm: str | None
    seed: int | None
    parent: str | None
    physical_device: str
    local_device: str
    data_dir: str
    run_dir: str
    precision: str
    deadline: float
    # Contracts S1 identities (optional for backward compat; required on
    # production paths that bind parents/checkpoints).
    job_id: str | None = None
    update_target: int | None = None
    time_budget_seconds: float | None = None
    source_hash: str | None = None
    data_hash: str | None = None
    tokenizer_identity: str | None = None
    config_identity: str | None = None
    protocol_hash: str | None = None
    reservation_id: str | None = None
    allocation_id: str | None = None
    reservation_deadline_unix: float | None = None
    parent_ref: ParentRef | None = None

    def validate(self) -> None:
        if self.phase not in ("E1", "E2", "E3", "E4", "E5", "E6", "E0"):
            raise ValueError(f"unknown phase {self.phase!r}")
        if not self.data_dir or not self.run_dir:
            raise ValueError("data_dir and run_dir are required")
        if not self.physical_device or not self.local_device:
            raise ValueError("physical and local devices are required")
        if self.parent_ref is not None and not isinstance(self.parent_ref, ParentRef):
            raise ValueError("parent_ref must be a ParentRef")
        if self.update_target is not None:
            if not isinstance(self.update_target, int) or isinstance(
                    self.update_target, bool) or self.update_target <= 0:
                raise ValueError("update_target must be a positive integer")

    def require_update_target(self, explicit: int | None) -> int:
        """Return the explicit frozen update target; never silently default.

        Training phases cannot silently choose 4/3 updates from function
        defaults. The caller must supply either an explicit argument or
        job.update_target (bound to the frozen protocol/E0 calibration).
        """
        if explicit is not None:
            if not isinstance(explicit, int) or isinstance(explicit, bool) \
                    or explicit <= 0:
                raise ValueError("update_target must be a positive integer")
            return explicit
        if self.update_target is not None:
            return self.update_target
        raise ValueError(
            f"{self.phase} requires an explicit update target bound to the "
            "frozen protocol/E0 calibration; refusing silent function-default "
            "updates")

    def resolved_parent_key(self) -> str | None:
        if self.parent_ref is not None and self.parent_ref.lookup_key:
            return self.parent_ref.lookup_key
        return self.parent


@dataclass
class PhaseResult:
    """Typed executor result with actual counts (D4 + contracts S1).

    Counts must derive from durable update events / verifier outcomes, never
    from a caller-supplied task count. `evidence_kind` is fixture,
    local-integration or learned-campaign; fixture receipts cannot enter
    accepted campaign aggregates.
    """

    status: str  # completed | failed
    committed_updates: int = 0
    attempted_updates: int = 0
    supervised_exposure: int = 0
    device_seconds: float = 0.0
    checkpoint_identity: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    evidence_kind: str = EVIDENCE_FIXTURE

    def validate(self) -> None:
        if self.status not in ("completed", "failed"):
            raise ValueError(f"unknown status {self.status!r}")
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence_kind {self.evidence_kind!r}")
        reserved = {"status", "committed_updates", "attempted_updates",
                    "supervised_exposure", "device_seconds",
                    "checkpoint_identity", "evidence_kind", "error"}
        collisions = reserved & set(self.extra)
        if collisions:
            raise ValueError(
                f"extra carries reserved receipt keys {sorted(collisions)}; "
                "refusing spoofable receipt")
        for name in ("committed_updates", "attempted_updates",
                     "supervised_exposure"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.status == "completed" and self.evidence_kind == EVIDENCE_LEARNED_CAMPAIGN:
            # Learned training receipts must carry real work; zero-work
            # learned success is a fabricated counter. Phase is explicit in
            # extra["phase"]; E2 frozen eval and E6 export legitimately carry
            # zero committed updates. Missing phase with zero work is refused.
            phase = str(self.extra.get("phase", ""))
            if phase in ("E2", "E6"):
                pass
            elif phase in ("E0", "E1", "E3", "E4", "E5"):
                if self.committed_updates <= 0:
                    raise ValueError(
                        "learned-campaign training receipt with zero committed "
                        "updates; refusing fabricated counter")
            elif self.committed_updates <= 0:
                raise ValueError(
                    "learned-campaign receipt without phase carries zero "
                    "committed updates; refusing")

    def phase_kind_is_training(self) -> bool:
        # Explicit known training phases only. Unknown/missing phase is NOT
        # assumed training (fail-closed the other way: callers must set phase
        # for learned receipts; see validate).
        return str(self.extra.get("phase", "")) in (
            "E0", "E1", "E3", "E4", "E5")

    def qualifies_for_campaign(self) -> bool:
        """Fixture receipts never qualify; training phases need real work."""
        if self.status != "completed":
            return False
        if self.evidence_kind == EVIDENCE_FIXTURE:
            return False
        if self.phase_kind_is_training() and self.committed_updates <= 0:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        # Extra first so reserved receipt keys cannot be spoofed by extras.
        out: dict[str, Any] = dict(self.extra)
        out.update({"status": self.status,
                    "committed_updates": self.committed_updates,
                    "attempted_updates": self.attempted_updates,
                    "supervised_exposure": self.supervised_exposure,
                    "device_seconds": self.device_seconds,
                    "checkpoint_identity": self.checkpoint_identity,
                    "evidence_kind": self.evidence_kind})
        if self.error:
            out["error"] = self.error
        return out
