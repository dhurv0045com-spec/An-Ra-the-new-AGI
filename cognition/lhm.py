"""Consent-bound longitudinal owner model with encrypted persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Literal

from cognition.storage import EncryptionUnavailable, SensitiveStateStore


ConfirmationState = Literal["inferred", "confirmed", "rejected"]


@dataclass
class ProfileField:
    name: str
    value: Any
    category: str
    source_session: str
    evidence_span: str
    confidence: float
    confirmation_state: ConfirmationState = "inferred"
    expires_at: float | None = None
    last_used: float | None = None
    updated_at: float = field(default_factory=time.time)


@dataclass
class ConsentPolicy:
    sensitive_inference: bool = False
    persistence: bool = False
    proactive_checks: bool = False
    training_use: bool = False
    session_consolidation: bool = False
    updated_at: float = field(default_factory=time.time)


class LongitudinalHumanModel:
    CATEGORIES = {
        "cognitive_profile",
        "emotional_profile",
        "life_context",
        "relationship_history",
        "wellbeing",
        "style",
    }

    def __init__(
        self,
        store: SensitiveStateStore,
        consent: ConsentPolicy,
        *,
        checkin_cooldown_seconds: float = 7 * 86400,
    ) -> None:
        self.store = store
        self.consent = consent
        self.checkin_cooldown_seconds = float(checkin_cooldown_seconds)
        self.fields: dict[str, ProfileField] = {}
        self.last_checkin: float | None = None
        self._load()

    @property
    def persistence_available(self) -> bool:
        return self.consent.persistence and self.store.available

    def _load(self) -> None:
        if not self.persistence_available:
            return
        payload = self.store.read("owner_model") or {}
        for item in payload.get("fields", []):
            field_value = ProfileField(**item)
            self.fields[field_value.name] = field_value
        self.last_checkin = payload.get("last_checkin")

    def _save(self) -> None:
        if not self.consent.persistence:
            return
        if not self.store.available:
            raise EncryptionUnavailable("Owner-model persistence is enabled but encryption is unavailable.")
        self.store.write(
            "owner_model",
            {
                "schema_version": 1,
                "fields": [asdict(item) for item in self.fields.values()],
                "last_checkin": self.last_checkin,
            },
        )

    def update(
        self,
        *,
        name: str,
        value: Any,
        category: str,
        source_session: str,
        evidence_span: str,
        confidence: float,
        confirmed: bool = False,
        expires_at: float | None = None,
    ) -> ProfileField:
        if category not in self.CATEGORIES:
            raise ValueError(f"Unknown owner-model category: {category}")
        if not confirmed and not self.consent.sensitive_inference:
            raise PermissionError("Sensitive inference requires explicit owner consent.")
        item = ProfileField(
            name=name,
            value=value,
            category=category,
            source_session=source_session,
            evidence_span=evidence_span,
            confidence=max(0.0, min(1.0, float(confidence))),
            confirmation_state="confirmed" if confirmed else "inferred",
            expires_at=expires_at,
        )
        previous = self.fields.get(name)
        self.fields[name] = item
        try:
            self._save()
        except Exception:
            if previous is None:
                self.fields.pop(name, None)
            else:
                self.fields[name] = previous
            raise
        return item

    def inspect(self) -> dict[str, object]:
        now = time.time()
        return {
            name: asdict(item)
            for name, item in self.fields.items()
            if item.confirmation_state != "rejected"
            and (item.expires_at is None or item.expires_at > now)
        }

    def confirm(self, name: str) -> ProfileField:
        item = self.fields[name]
        item.confirmation_state = "confirmed"
        item.updated_at = time.time()
        self._save()
        return item

    def correct(self, name: str, value: Any, *, evidence_span: str = "owner correction") -> ProfileField:
        item = self.fields[name]
        item.value = value
        item.evidence_span = evidence_span
        item.confirmation_state = "confirmed"
        item.confidence = 1.0
        item.updated_at = time.time()
        self._save()
        return item

    def reject(self, name: str) -> None:
        self.fields[name].confirmation_state = "rejected"
        self.fields[name].updated_at = time.time()
        self._save()

    def delete_session(self, session_id: str) -> int:
        names = [name for name, item in self.fields.items() if item.source_session == session_id]
        for name in names:
            del self.fields[name]
        self._save()
        return len(names)

    def delete(self, name: str) -> bool:
        removed = self.fields.pop(name, None) is not None
        self._save()
        return removed

    def wipe(self) -> int:
        count = len(self.fields)
        self.fields.clear()
        self.store.delete("owner_model")
        return count

    def export(self) -> dict[str, object]:
        return {"schema_version": 1, "owner_model": self.inspect(), "consent": asdict(self.consent)}

    def disable_proactive_checks(self) -> None:
        self.consent.proactive_checks = False

    def proactive_check_due(self, *, now: float | None = None) -> bool:
        current = now or time.time()
        if not self.consent.proactive_checks:
            return False
        confirmed_signals = [
            item
            for item in self.fields.values()
            if item.category == "wellbeing"
            and item.confirmation_state == "confirmed"
            and item.confidence >= 0.75
        ]
        if len(confirmed_signals) < 2:
            return False
        return self.last_checkin is None or current - self.last_checkin >= self.checkin_cooldown_seconds

    def mark_checkin(self, *, now: float | None = None) -> None:
        self.last_checkin = now or time.time()
        self._save()

    @staticmethod
    def wellbeing_language(observation: str) -> str:
        return f"Non-diagnostic wellbeing observation: {observation}"
