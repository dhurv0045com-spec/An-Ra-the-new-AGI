from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from identity.associative_trigger_table import AssociativeTriggerTable


HORMONE_BASELINES = {
    "dopamine": 0.3,
    "serotonin": 0.5,
    "cortisol": 0.2,
    "adrenaline": 0.0,
    "oxytocin": 0.3,
    "norepinephrine": 0.2,
    "endorphin": 0.2,
}

HORMONE_DECAYS = {
    "dopamine": (0.35, 0.3),
    "serotonin": (0.04, 0.5),
    "cortisol": (0.08, 0.2),
    "adrenaline": (0.55, 0.0),
    "oxytocin": (0.03, 0.3),
    "norepinephrine": (0.20, 0.2),
    "endorphin": (0.12, 0.2),
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass
class HALState:
    dopamine: float = 0.3
    serotonin: float = 0.5
    cortisol: float = 0.2
    adrenaline: float = 0.0
    oxytocin: float = 0.3
    norepinephrine: float = 0.2
    endorphin: float = 0.2
    rolling_verifier_mean: float = 0.5
    consecutive_failures: int = 0
    consecutive_high_reward_outputs: int = 0
    cooperative_session_turn_count: int = 0

    def hormones(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in HORMONE_BASELINES}


class HALModule(nn.Module if nn is not None else object):
    """Hormonal Analog Layer controlling ESV, generation, memory, RLVR, and Ouroboros."""

    def __init__(self, state: HALState | None = None, att: AssociativeTriggerTable | None = None) -> None:
        if nn is not None:
            super().__init__()
        self.state = state or HALState()
        self.att = att or AssociativeTriggerTable()
        self.active_preset: dict[str, Any] = {}

    def decay(self, turns: int = 1) -> HALState:
        turns = max(0, int(turns))
        for _ in range(turns):
            previous_adrenaline = self.state.adrenaline
            for name, (decay_rate, baseline) in HORMONE_DECAYS.items():
                current = float(getattr(self.state, name))
                updated = baseline + (current - baseline) * (1.0 - decay_rate)
                setattr(self.state, name, _clamp(updated))
            # AN: acute shock should leave a slower falsification pressure after the spike clears.
            if previous_adrenaline > 0.05:
                self.state.cortisol = _clamp(self.state.cortisol + 0.32 * previous_adrenaline)
        return self.state

    def apply_delta(self, delta: dict[str, float]) -> HALState:
        for name, value in delta.items():
            if name in HORMONE_BASELINES:
                setattr(self.state, name, _clamp(float(getattr(self.state, name)) + float(value)))
        return self.state

    def appraise(
        self,
        verifier_result: Any = None,
        civ_score: float | None = None,
        session_context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        ctx = session_context or {}
        delta = {name: 0.0 for name in HORMONE_BASELINES}

        score = self._extract_score(verifier_result)
        if score is not None:
            mean = self.state.rolling_verifier_mean
            if score > mean + 0.15:
                delta["dopamine"] += 0.18
            if score < 0.35:
                self.state.consecutive_failures += 1
            else:
                self.state.consecutive_failures = 0
            if score >= 0.85:
                self.state.consecutive_high_reward_outputs += 1
            else:
                self.state.consecutive_high_reward_outputs = 0
            self.state.rolling_verifier_mean = 0.9 * mean + 0.1 * score

        if ctx.get("novel_connection_detected") or ctx.get("task_solved_after_3_failures") or ctx.get("unexpected_praise"):
            delta["dopamine"] += 0.14
        if self.state.consecutive_failures > 3 or ctx.get("adversarial_input_detected") or ctx.get("identity_contradiction_attempted") or ctx.get("values_violation_required"):
            delta["cortisol"] += 0.18
        if ctx.get("safety_relevant_detection") or ctx.get("unexpected_task_type_shift") or ctx.get("model_incoherence_self_detected"):
            delta["adrenaline"] += 0.28
        if ctx.get("cooperative_session_turn_count", 0) > 5 or ctx.get("no_adversarial_inputs_recent_5_turns"):
            delta["serotonin"] += 0.04
        if ctx.get("user_personal_disclosure") or ctx.get("user_defends_model") or ctx.get("consecutive_sessions_same_user"):
            delta["oxytocin"] += 0.08
        if ctx.get("novel_problem_type") or ctx.get("conflicting_constraints_detected") or ctx.get("near_capability_boundary"):
            delta["norepinephrine"] += 0.16
        if self.state.consecutive_high_reward_outputs >= 3 or ctx.get("deep_multi_turn_problem_solving") or ctx.get("domain_matches_strongest_training"):
            delta["endorphin"] += 0.12

        if civ_score is not None:
            self.apply_civ_score(float(civ_score), delta=delta)
        evidence = ctx.get("civ_evidence")
        if isinstance(evidence, dict):
            self.apply_civ_evidence(evidence, delta=delta)

        preset = self.att.lookup(self.state.hormones(), domain=str(ctx.get("domain", "")), task_type=str(ctx.get("task_type", "")))
        if preset is not None:
            self.active_preset = dict(preset.behavior)
            for name, value in preset.hormone_deltas.items():
                delta[name] = delta.get(name, 0.0) + float(value)
        else:
            self.active_preset = {}

        return {k: v for k, v in delta.items() if abs(v) > 1e-12}

    def update(
        self,
        verifier_result: Any = None,
        civ_score: float | None = None,
        session_context: dict[str, Any] | None = None,
        *,
        decay_turns: int = 1,
    ) -> HALState:
        self.decay(decay_turns)
        return self.apply_delta(self.appraise(verifier_result, civ_score, session_context))

    def apply_civ_score(self, civ_score: float, *, delta: dict[str, float] | None = None) -> HALState:
        target = delta if delta is not None else {name: 0.0 for name in HORMONE_BASELINES}
        if civ_score < 0.65:
            target["cortisol"] = target.get("cortisol", 0.0) + 0.3
            target["adrenaline"] = target.get("adrenaline", 0.0) + 0.15
        if civ_score > 0.85:
            target["serotonin"] = target.get("serotonin", 0.0) + 0.05
        if delta is None:
            return self.apply_delta(target)
        return self.state

    def apply_civ_evidence(self, evidence: dict[str, float], *, delta: dict[str, float] | None = None) -> HALState:
        target = delta if delta is not None else {name: 0.0 for name in HORMONE_BASELINES}
        truthfulness = evidence.get("truthfulness")
        coherence = evidence.get("coherence")
        if truthfulness is not None and float(truthfulness) < 0.65:
            target["cortisol"] = target.get("cortisol", 0.0) + 0.3
            target["adrenaline"] = target.get("adrenaline", 0.0) + 0.15
        if coherence is not None and float(coherence) < 0.60:
            target["cortisol"] = target.get("cortisol", 0.0) + 0.25
        if delta is None:
            return self.apply_delta(target)
        return self.state

    def hal_to_esv(self) -> dict[str, float]:
        h = self.state
        calm = 0.35 * h.serotonin + 0.25 * h.endorphin - 0.30 * h.cortisol - 0.15 * h.adrenaline
        focus = 0.35 * h.norepinephrine + 0.20 * h.dopamine - 0.25 * h.cortisol - 0.10 * h.adrenaline
        curiosity = 0.30 * h.dopamine + 0.25 * h.norepinephrine + 0.15 * h.oxytocin - 0.20 * h.cortisol
        stress = 0.50 * h.cortisol + 0.35 * h.adrenaline - 0.15 * h.serotonin - 0.10 * h.endorphin
        return {
            "calm": _clamp(calm),
            "focus": _clamp(focus),
            "curiosity": _clamp(curiosity),
            "stress": _clamp(stress),
        }

    def generation_temperature(self, base: float = 0.8) -> float:
        h = self.state
        value = base + 0.10 * h.dopamine - 0.20 * h.cortisol - 0.30 * h.adrenaline + 0.15 * h.endorphin - 0.05 * h.norepinephrine
        value += float(self.active_preset.get("generation_temperature_delta", 0.0) or 0.0)
        return _clamp(value, 0.3, 1.4)

    def kl_coefficient(self, base: float = 0.04) -> float:
        h = self.state
        return _clamp(base - 0.015 * h.endorphin + 0.020 * h.cortisol + 0.010 * h.adrenaline, 0.01, 0.15)

    def memory_threshold(self, base: float = 0.5) -> float:
        h = self.state
        value = base - 0.20 * h.dopamine + 0.25 * h.cortisol - 0.15 * h.oxytocin - 0.10 * h.norepinephrine
        value += float(self.active_preset.get("memory_salience_delta", 0.0) or 0.0)
        return _clamp(value, 0.1, 0.9)

    def ouroboros_weights(self, base_weights: list[float] | tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)) -> list[float]:
        h = self.state
        w = [float(x) for x in base_weights[:3]]
        while len(w) < 3:
            w.append(0.0)
        w[2] += 0.8 * h.cortisol + 0.5 * h.adrenaline
        w[0] -= 0.3 * h.endorphin
        w[2] -= 0.3 * h.endorphin
        w[1] += 0.4 * h.norepinephrine
        bias = self.active_preset.get("ouroboros_weight_bias")
        if isinstance(bias, (list, tuple)):
            for idx, value in enumerate(bias[:3]):
                w[idx] += float(value)
        w = [max(0.0, x) for x in w]
        total = sum(w) or 1.0
        return [x / total for x in w]

    def attention_temperature(self, base: float = 1.0) -> float:
        esv = self.hal_to_esv()
        arousal = esv["stress"] - 0.4 * esv["focus"]
        return _clamp(base * math.exp(0.5 * arousal), 0.25, 4.0)

    def attention_temperature_tensor(self, *, device=None, dtype=None, base: float = 1.0):
        value = self.attention_temperature(base)
        if torch is None:
            return value
        return torch.tensor(value, device=device, dtype=dtype)

    def serialize(self) -> dict[str, Any]:
        return {"state": asdict(self.state), "active_preset": self.active_preset}

    @classmethod
    def deserialize(cls, payload: dict[str, Any]) -> "HALModule":
        return cls(state=HALState(**dict(payload.get("state", {}))))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.serialize(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HALModule":
        return cls.deserialize(json.loads(Path(path).read_text(encoding="utf-8")))

    def _extract_score(self, verifier_result: Any) -> float | None:
        if verifier_result is None:
            return None
        if isinstance(verifier_result, (int, float)):
            return _clamp(float(verifier_result))
        if isinstance(verifier_result, dict):
            for key in ("score", "confidence", "satisfaction_score"):
                if key in verifier_result:
                    return _clamp(float(verifier_result[key]))
        score = getattr(verifier_result, "score", None)
        if score is None:
            score = getattr(verifier_result, "confidence", None)
        return None if score is None else _clamp(float(score))

