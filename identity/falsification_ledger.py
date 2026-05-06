from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from pathlib import Path
from typing import Any


VALID_STATUSES = {"VERIFIED", "INFERRED", "ASSUMED", "UNKNOWN", "FALSIFIED"}


def _claim_id(claim: str) -> str:
    return hashlib.sha1(claim.strip().encode("utf-8")).hexdigest()[:16]


@dataclass
class ClaimRecord:
    claim: str
    status: str = "UNKNOWN"
    confidence: float = 0.0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    would_be_false_if: str = ""
    next_verifier: str = ""
    claim_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.status = self.status.upper()
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid claim status: {self.status}")
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if not self.claim_id:
            self.claim_id = _claim_id(self.claim)


class FalsificationLedger:
    """Persistent claim ledger with explicit falsifiers and verifier paths."""

    def __init__(self, path: str | Path, memory_router=None) -> None:
        self.path = Path(path)
        self.memory_router = memory_router
        self.records: dict[str, ClaimRecord] = {}
        if self.path.exists():
            self.load()

    def append(
        self,
        claim: str,
        *,
        status: str = "UNKNOWN",
        confidence: float = 0.0,
        evidence: list[dict[str, Any]] | None = None,
        would_be_false_if: str = "",
        next_verifier: str = "",
    ) -> ClaimRecord:
        record = ClaimRecord(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=evidence or [],
            would_be_false_if=would_be_false_if,
            next_verifier=next_verifier,
        )
        existing = self.records.get(record.claim_id)
        if existing is not None:
            record.created_at = existing.created_at
            record.evidence = [*existing.evidence, *record.evidence]
        record.updated_at = time.time()
        self.records[record.claim_id] = record
        self.save()
        if self.memory_router is not None:
            try:
                self.memory_router.write(
                    json.dumps(asdict(record), sort_keys=True),
                    metadata={"kind": "falsification_ledger", "claim_id": record.claim_id, "salience": record.confidence},
                    tier="ghost",
                )
            except Exception:
                pass
        return record

    def query(self, text: str = "", *, status: str | None = None) -> list[ClaimRecord]:
        needle = text.lower().strip()
        wanted = status.upper() if status else None
        rows = []
        for record in self.records.values():
            if wanted and record.status != wanted:
                continue
            if needle and needle not in record.claim.lower():
                continue
            rows.append(record)
        rows.sort(key=lambda r: (r.updated_at, r.confidence), reverse=True)
        return rows

    def export_training_data(self) -> list[dict[str, Any]]:
        items = []
        for record in self.records.values():
            items.append(
                {
                    "template": "HYPOTHESIS_CHAIN" if record.status != "FALSIFIED" else "FAILURE_REPLAY",
                    "claim": record.claim,
                    "verify": record.status,
                    "confidence": record.confidence,
                    "evidence": record.evidence,
                    "falsifier": record.would_be_false_if,
                    "next_verifier": record.next_verifier,
                    "text": (
                        f"<hyp>{record.claim}</hyp>\n"
                        f"<verify>{record.status} confidence={record.confidence:.2f}</verify>\n"
                        f"<err>{record.would_be_false_if}</err>\n"
                        f"<act>{{\"tool\":\"{record.next_verifier}\"}}</act>"
                    ),
                }
            )
        return items

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [asdict(record) for record in self.records.values()]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = {
            str(row.get("claim_id") or _claim_id(str(row.get("claim", "")))): ClaimRecord(**row)
            for row in data.get("records", [])
        }

