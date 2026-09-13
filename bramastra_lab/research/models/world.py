"""Public world-model prediction (M03): typed outcomes from the shared
decoder's ordinary token prediction — no latent model, no privileged state.

Finite declared supports normalize exact sequence scores and aggregate
duplicate parsed outcomes; predicted outcomes are typed values that can
never enter the observed ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from bramastra_lab.research.config import BuildConfig
from bramastra_lab.research.experience.codec import SPECIAL_EOS, encode_event
from bramastra_lab.research.models import IntegratedModel


class WorldModelError(ValueError):
    """A world-prediction call violated its contract."""


@dataclass(frozen=True)
class OutcomeOption:
    feedback: Mapping[str, Any]
    terminated: bool
    probability: float
    log_probability: float
    parse_valid: bool
    rendering: str

    def identity(self) -> str:
        from bramastra_lab.research.contracts.core import content_identity

        return content_identity({"feedback": dict(self.feedback),
                                 "terminated": self.terminated})


@dataclass(frozen=True)
class PredictedOutcome:
    options: tuple[OutcomeOption, ...]
    predictor_identity: str
    unknown_mass: float
    condition_identity: str

    def aggregated(self) -> dict[str, float]:
        """Duplicate parses of the same outcome aggregate their mass once.

        Keys are canonical-JSON renderings of (feedback, terminated), so two
        renderings of one parsed outcome contribute to a single bucket.
        """
        aggregated: dict[str, float] = {}
        for option in self.options:
            key = json_key({"feedback": option.feedback,
                            "terminated": option.terminated})
            aggregated[key] = aggregated.get(key, 0.0) + option.probability
        return aggregated


def json_key(value: Any) -> tuple:
    import json

    return json.dumps(value, sort_keys=True)


def _render_prefix(prefix: Sequence[int], goal: Mapping[str, Any] | None,
                   action: Mapping[str, Any]) -> list[int]:
    tokens = [259]  # SPECIAL_BOUNDARY
    if goal is not None:
        tokens += encode_event("goal", goal)
    tokens += encode_event("action", action)
    return tokens + list(prefix)


def predict_step_finite(model: IntegratedModel, config: BuildConfig,
                        prefix_tokens: Sequence[int], *, action: Mapping[str, Any],
                        support: Sequence[Mapping[str, Any]],
                        goal: Mapping[str, Any] | None = None) -> PredictedOutcome:
    """Score each declared public support rendering as its own continuation.

    The support must be declared from the PUBLIC schema (never from hidden
    state). Each rendering is scored as an independent branch; duplicate
    parsed outcomes aggregate their probability at the end.
    """
    if not support:
        raise WorldModelError("an empty outcome support cannot be predicted")
    condition_prefix = _render_prefix(prefix_tokens, goal, action)
    max_new = max(len(encode_event("feedback", option["feedback"])) + 1
                  for option in support)
    if len(condition_prefix) + max_new > config.model.max_seq:
        raise WorldModelError("prefix + longest rendering exceeds the context limit")
    scored: list[tuple[float, OutcomeOption]] = []
    for option in support:
        if "feedback" not in option or "rendering" not in option:
            raise WorldModelError("support options need feedback and rendering")
        rendering_tokens = encode_event("feedback", option["feedback"]) + [SPECIAL_EOS]
        sequence = condition_prefix + rendering_tokens
        inputs = torch.tensor([sequence[:-1]], dtype=torch.long)
        with torch.no_grad():
            logits = model(inputs).logits[0]
        targets = torch.tensor(sequence[1:], dtype=torch.long)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        total = float(log_probs.gather(1, targets.unsqueeze(1)).sum().item())
        scored.append((total, OutcomeOption(
            feedback=dict(option["feedback"]),
            terminated=bool(option.get("terminated", False)),
            probability=0.0, log_probability=total, parse_valid=True,
            rendering=option["rendering"])))
    ceiling = max(total for total, _ in scored)
    weights = [torch.exp(torch.tensor(total - ceiling)).item() for total, _ in scored]
    mass = sum(weights)
    options = []
    for (total, option), weight in zip(scored, weights):
        options.append(OutcomeOption(
            feedback=option.feedback, terminated=option.terminated,
            probability=weight / mass, log_probability=total,
            parse_valid=option.parse_valid, rendering=option.rendering))
    from bramastra_lab.research.contracts.core import content_identity

    return PredictedOutcome(
        options=tuple(sorted(options, key=lambda item: -item.probability)),
        predictor_identity=f"shared-decoder:{sum(p.numel() for p in model.parameters())}",
        unknown_mass=0.0,
        condition_identity=content_identity(
            {"prefix": list(prefix_tokens), "goal": goal, "action": dict(action),
             "support": [dict(option) for option in support]}))
