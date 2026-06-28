"""Provenance-backed epistemic state and calibration."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from anra.anra_paths import OUTPUT_V2_DIR

SourceType = Literal[
    "training_verified",
    "training_unverified",
    "conversation_stated",
    "derived",
    "world_model",
    "confabulated",
]
VerificationStatus = Literal["verified", "unverified", "falsified", "unknown"]


@dataclass
class EpistemicState:
    claim: str
    source_type: SourceType
    confidence: float
    calibrated_conf: float
    falsification: str
    verification_status: VerificationStatus
    domain: str
    time_sensitive: bool
    source_count: int
    evidence_ids: tuple[str, ...] = ()
    source_timestamps: tuple[float, ...] = ()
    source_hashes: tuple[str, ...] = ()
    independent_source_count: int = 0
    derivation_depth: int = 0
    verifier_results: tuple[dict[str, object], ...] = ()
    expires_at: float | None = None
    calibration_model_version: int = 1
    created_at: float = field(default_factory=time.time)

    THRESHOLDS = {
        "medical": 0.92,
        "legal": 0.90,
        "technical": 0.85,
        "personal": 0.75,
        "general": 0.70,
    }

    def should_present_as_fact(self, *, now: float | None = None) -> bool:
        threshold = self.THRESHOLDS.get(self.domain, 0.75)
        stale = self.expires_at is not None and (now or time.time()) >= self.expires_at
        return (
            self.calibrated_conf >= threshold
            and self.verification_status != "falsified"
            and self.source_type != "confabulated"
            and not stale
        )

    def caveat(self) -> str:
        if self.source_type == "confabulated":
            return "I cannot trace this claim to reliable evidence."
        if self.verification_status == "falsified":
            return "Available verification contradicts this claim."
        if self.expires_at is not None and time.time() >= self.expires_at:
            return "This information is stale and requires a current source."
        if not self.should_present_as_fact():
            return f"This is uncertain (calibrated confidence {self.calibrated_conf:.0%})."
        return ""


class EpistemicTracker:
    def __init__(
        self,
        history_path: str | Path = OUTPUT_V2_DIR / "epistemic_history.jsonl",
        calibration_path: str | Path = OUTPUT_V2_DIR / "epistemic_calibration.json",
    ) -> None:
        self.history_path = Path(history_path)
        self.calibration_path = Path(calibration_path)
        self._session_claims: dict[str, EpistemicState] = {}
        self._calibration = self._load_calibration()

    @staticmethod
    def _key(claim: str) -> str:
        return hashlib.sha256(claim.strip().lower().encode("utf-8")).hexdigest()

    def assess(
        self,
        claim: str,
        *,
        domain: str = "general",
        source_type: SourceType = "training_unverified",
        evidence: list[dict[str, object]] | None = None,
        derivation_chain: list[str] | None = None,
    ) -> EpistemicState:
        evidence = evidence or []
        verified = [item for item in evidence if bool(item.get("verified", False))]
        independent = len(
            {str(item.get("source_id", "")) for item in verified if item.get("source_id")}
        )
        provenance = sum(float(item.get("provenance", 0.0)) for item in evidence) / max(
            1, len(evidence)
        )
        recency = sum(float(item.get("recency", 1.0)) for item in evidence) / max(1, len(evidence))
        verifier = sum(float(item.get("score", 0.0)) for item in verified) / max(1, len(verified))
        depth = len(derivation_chain or [])
        base = (
            0.15
            + 0.30 * provenance
            + 0.30 * verifier
            + 0.15 * min(1.0, independent / 3)
            + 0.10 * recency
        )
        if not evidence:
            base = 0.25 if source_type == "confabulated" else 0.50
        base *= 0.95**depth
        calibrated = max(0.0, min(0.99, base * float(self._calibration.get(domain, 1.0))))
        now = time.time()
        sensitive = domain in {"medical", "legal", "technical", "current_events"}
        state = EpistemicState(
            claim=claim,
            source_type=source_type,
            confidence=base,
            calibrated_conf=calibrated,
            falsification=f"Reliable evidence directly contradicts: {claim[:120]}",
            verification_status="verified" if verified else "unknown",
            domain=domain,
            time_sensitive=sensitive,
            source_count=len(evidence),
            evidence_ids=tuple(str(item.get("evidence_id", "")) for item in evidence),
            source_timestamps=tuple(float(item.get("timestamp", now)) for item in evidence),
            source_hashes=tuple(str(item.get("hash", "")) for item in evidence),
            independent_source_count=independent,
            derivation_depth=depth,
            verifier_results=tuple(dict(item) for item in evidence),
            expires_at=now + 86400 if sensitive else None,
        )
        self._session_claims[self._key(claim)] = state
        return state

    def record_outcome(
        self,
        claim: str,
        *,
        was_correct: bool,
        domain: str,
        verifier: str,
    ) -> None:
        state = self._session_claims.get(self._key(claim))
        predicted = state.calibrated_conf if state else 0.5
        record = {
            "claim_hash": self._key(claim),
            "domain": domain,
            "predicted_conf": predicted,
            "was_correct": bool(was_correct),
            "verifier": verifier,
            "timestamp": time.time(),
        }
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if state:
            state.verification_status = "verified" if was_correct else "falsified"

    def recalibrate(self) -> dict[str, float]:
        history = self._history()
        if len(history) < 100:
            return dict(self._calibration)
        domains = {str(item["domain"]) for item in history}
        for domain in domains:
            rows = [item for item in history if item["domain"] == domain]
            if len(rows) < 10:
                continue
            predicted = sum(float(item["predicted_conf"]) for item in rows) / len(rows)
            actual = sum(bool(item["was_correct"]) for item in rows) / len(rows)
            self._calibration[domain] = max(0.25, min(1.5, actual / max(0.05, predicted)))
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.calibration_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._calibration, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(self.calibration_path)
        return dict(self._calibration)

    def calibration_report(self) -> dict[str, object]:
        history = self._history()
        if not history:
            return {"status": "insufficient_data", "n_outcomes": 0, "target": 0.15}
        brier = sum(
            (float(item["predicted_conf"]) - float(bool(item["was_correct"]))) ** 2
            for item in history
        ) / len(history)
        bins: list[float] = []
        for low in (0.0, 0.2, 0.4, 0.6, 0.8):
            rows = [item for item in history if low <= float(item["predicted_conf"]) < low + 0.2]
            if rows:
                confidence = sum(float(item["predicted_conf"]) for item in rows) / len(rows)
                accuracy = sum(bool(item["was_correct"]) for item in rows) / len(rows)
                bins.append(abs(confidence - accuracy) * len(rows) / len(history))
        domain_reliability = {}
        for domain in sorted({str(item["domain"]) for item in history}):
            rows = [item for item in history if item["domain"] == domain]
            domain_reliability[domain] = {
                "sample_count": len(rows),
                "mean_confidence": sum(float(item["predicted_conf"]) for item in rows) / len(rows),
                "accuracy": sum(bool(item["was_correct"]) for item in rows) / len(rows),
                "brier_score": sum(
                    (float(item["predicted_conf"]) - float(bool(item["was_correct"]))) ** 2
                    for item in rows
                )
                / len(rows),
                "calibration_multiplier": float(self._calibration.get(domain, 1.0)),
            }
        return {
            "status": "measured",
            "brier_score": brier,
            "expected_calibration_error": sum(bins),
            "n_outcomes": len(history),
            "target": 0.15,
            "passing": len(history) >= 100 and brier < 0.15,
            "by_domain": domain_reliability,
        }

    def _history(self) -> list[dict[str, object]]:
        if not self.history_path.exists():
            return []
        rows = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
            except json.JSONDecodeError:
                continue
        return rows

    def _load_calibration(self) -> dict[str, float]:
        try:
            payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            return {str(key): float(value) for key, value in payload.items()}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}
