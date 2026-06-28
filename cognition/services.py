"""Canonical facade for versioned cognitive services."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from anra.anra_paths import OUTPUT_V2_DIR, STATE_DIR
from engine.feature_flags import is_enabled

from cognition.cdse import CrossDomainSynthesisEngine
from cognition.cec import ContinuousExperienceConsolidator
from cognition.cre import CausalReasoningEngine
from cognition.epistemic_tracker import EpistemicTracker
from cognition.lhm import ConsentPolicy, LongitudinalHumanModel
from cognition.self_debate import MultiAgentSelfDebate
from cognition.ssie import ScientificSelfImprovementEngine
from cognition.storage import SensitiveStateStore


class CognitionServices:
    RELEASE = "cognition-v1"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        state_dir: str | Path = STATE_DIR / "cognition",
        output_dir: str | Path = OUTPUT_V2_DIR / "cognition",
        encryption_key: str | bytes | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.output_dir = Path(output_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.consent_path = self.state_dir / "consent.json"
        self.consent = self._load_consent()
        self.store = SensitiveStateStore(self.state_dir / "private", key=encryption_key)
        self.cre = CausalReasoningEngine()
        self.et = EpistemicTracker(
            self.output_dir / "epistemic_history.jsonl",
            self.output_dir / "epistemic_calibration.json",
        )
        self.lhm = LongitudinalHumanModel(self.store, self.consent)
        self.ssie = ScientificSelfImprovementEngine(state_path=self.state_dir / "ssie.json")
        self.cdse = CrossDomainSynthesisEngine()
        self.cec = ContinuousExperienceConsolidator(self.state_dir / "consolidation.json")
        self.debate = MultiAgentSelfDebate()

    def _load_consent(self) -> ConsentPolicy:
        try:
            payload = json.loads(self.consent_path.read_text(encoding="utf-8"))
            allowed = {
                key: payload[key] for key in ConsentPolicy.__dataclass_fields__ if key in payload
            }
            return ConsentPolicy(**allowed)
        except (OSError, json.JSONDecodeError, TypeError):
            return ConsentPolicy()

    def update_consent(self, **changes: bool) -> ConsentPolicy:
        for name, value in changes.items():
            if name not in ConsentPolicy.__dataclass_fields__ or name == "updated_at":
                raise ValueError(f"Unknown consent field: {name}")
            setattr(self.consent, name, bool(value))
        self.consent.updated_at = time.time()
        self.consent_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.consent_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(self.consent), indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(self.consent_path)
        return self.consent

    def classify_goal(self, goal: str) -> dict[str, object]:
        judgment = self.cre.classify_query(goal)
        return {
            "release": self.RELEASE,
            "causal": judgment.to_dict(),
            "debate_reasons": self.debate.classifier.classify(goal),
        }

    def status(self) -> dict[str, object]:
        calibration = self.et.calibration_report()
        enabled = {
            name: is_enabled(name)
            for name in (
                "cognition",
                "causal_reasoning",
                "epistemic_tracker",
                "human_model",
                "ssie",
                "cdse",
                "cec",
                "self_debate",
            )
        }
        return {
            "release": self.RELEASE,
            "schema_version": self.SCHEMA_VERSION,
            "enabled": enabled,
            "consent": asdict(self.consent),
            "encrypted_owner_storage": self.store.available,
            "calibration": calibration,
            "pending_experiments": sum(
                item.status == "proposed" for item in self.ssie.proposals.values()
            ),
            "cec_backlog": self.cec.backlog(),
        }

    def health(self) -> dict[str, object]:
        blockers = []
        if self.consent.persistence and not self.store.available:
            blockers.append("sensitive persistence consented but encrypted storage unavailable")
        return {"status": "degraded" if blockers else "ok", "blockers": blockers, **self.status()}
