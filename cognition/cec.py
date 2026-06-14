"""Idempotent, consented continuous experience consolidation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Callable


@dataclass(frozen=True)
class ConsolidationReport:
    session_id: str
    content_hash: str
    status: str
    processing_latency_ms: float
    owner_lessons: int
    domain_corrections: int
    verified_claims: int
    successes: int
    failures: int
    training_candidates: int
    quarantined_candidates: int
    lhm_updates: int
    epistemic_updates: int
    created_at: float = field(default_factory=time.time)


class ContinuousExperienceConsolidator:
    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self._reports = self._load()

    @staticmethod
    def content_hash(session_id: str, turns: list[dict[str, object]]) -> str:
        payload = json.dumps({"session_id": session_id, "turns": turns}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def consolidate(
        self,
        session_id: str,
        turns: list[dict[str, object]],
        *,
        opted_in: bool,
        verify: Callable[[dict[str, object]], bool] | None = None,
    ) -> ConsolidationReport:
        started = time.perf_counter()
        digest = self.content_hash(session_id, turns)
        existing = self._reports.get(digest)
        if existing:
            return ConsolidationReport(**existing)
        if not opted_in:
            report = ConsolidationReport(session_id, digest, "skipped_no_consent", 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            self._persist(report)
            return report
        owner_lessons = domain_corrections = verified_claims = successes = failures = 0
        training_candidates = quarantined = 0
        for turn in turns:
            tags = set(turn.get("tags", ()))
            owner_lessons += int("owner_lesson" in tags)
            domain_corrections += int("domain_correction" in tags)
            successes += int(turn.get("success") is True)
            failures += int(turn.get("success") is False)
            candidate = bool(tags & {"owner_lesson", "domain_correction", "training_candidate"})
            verified = bool(verify(turn)) if verify else bool(turn.get("verified", False))
            verified_claims += int(verified)
            training_candidates += int(candidate and verified)
            quarantined += int(candidate and not verified)
        latency = (time.perf_counter() - started) * 1000
        report = ConsolidationReport(
            session_id,
            digest,
            "completed",
            latency,
            owner_lessons,
            domain_corrections,
            verified_claims,
            successes,
            failures,
            training_candidates,
            quarantined,
            0,
            verified_claims,
        )
        self._persist(report)
        return report

    def rollback(self, session_id: str) -> int:
        keys = [key for key, value in self._reports.items() if value["session_id"] == session_id]
        for key in keys:
            del self._reports[key]
        self._write()
        return len(keys)

    delete_session = rollback

    def backlog(self) -> int:
        return sum(int(row["quarantined_candidates"]) for row in self._reports.values())

    def _persist(self, report: ConsolidationReport) -> None:
        self._reports[report.content_hash] = asdict(report)
        self._write()

    def _write(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._reports, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)

    def _load(self) -> dict[str, dict[str, object]]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
