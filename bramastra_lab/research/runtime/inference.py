"""Inference adapters (B08): free generation and finite-action scoring.

All inference runs through the canonical IntegratedModel with the unmodified
full vocabulary for primary generation. Generation ends on the declared EOS
token or an explicit cap, and the report states both the answer and whether
stopping was valid — an answer without valid stopping is never silently
treated as complete.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from bramastra_lab.research.config import BuildConfig
from bramastra_lab.research.experience.codec import (
    SPECIAL_EOS,
    decode_to_bytes,
)
from bramastra_lab.research.models import IntegratedModel


class InferenceError(RuntimeError):
    """Inference was invoked against its declared contract."""


@dataclass(frozen=True)
class GenerationReport:
    answer: str
    stopped_on_eos: bool
    hit_cap: bool
    new_tokens: int
    retrieval_enabled: bool = False
    planner: str = "none"

    @property
    def complete(self) -> bool:
        """A complete answer requires the answer text AND valid stopping."""
        return self.stopped_on_eos

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer, "stopped_on_eos": self.stopped_on_eos,
            "hit_cap": self.hit_cap, "new_tokens": self.new_tokens,
            "complete": self.complete, "retrieval_enabled": self.retrieval_enabled,
            "planner": self.planner,
        }


@torch.no_grad()
def generate_free_form(
    model: IntegratedModel,
    config: BuildConfig,
    prompt_tokens: list[int],
    *,
    max_new_tokens: int | None = None,
    greedy: bool = True,
    generator: torch.Generator | None = None,
) -> GenerationReport:
    """Primary free generation over the full physical vocabulary.

    ``prompt_tokens`` must already be encoded through the public codec. The
    loop never exceeds the model's context limit; EOS terminates generation.
    No schema restriction and no oracle assistance participate here.
    """
    context_limit = config.model.max_seq
    if len(prompt_tokens) >= context_limit:
        raise InferenceError(
            f"prompt length {len(prompt_tokens)} leaves no room inside the "
            f"context limit {context_limit}; refusing to truncate the prompt")
    cap = max_new_tokens if max_new_tokens is not None else context_limit - len(prompt_tokens)
    tokens = list(prompt_tokens)
    stopped_on_eos = False
    hit_cap = False
    model.eval()
    while len(tokens) < context_limit and len(tokens) - len(prompt_tokens) < cap:
        window = tokens[-context_limit + 1:]
        inputs = torch.tensor([window], dtype=torch.long)
        logits = model(inputs).logits[0, -1]
        if greedy:
            next_token = int(torch.argmax(logits).item())
        else:
            probabilities = torch.softmax(logits.float(), dim=-1)
            next_token = int(torch.multinomial(
                probabilities, 1, generator=generator).item())
        tokens.append(next_token)
        if next_token == SPECIAL_EOS:
            stopped_on_eos = True
            break
    else:
        hit_cap = len(tokens) - len(prompt_tokens) >= cap
    answer_bytes = decode_to_bytes(tokens[len(prompt_tokens):])
    return GenerationReport(
        answer=answer_bytes.decode("utf-8", errors="replace"),
        stopped_on_eos=stopped_on_eos, hit_cap=hit_cap,
        new_tokens=len(tokens) - len(prompt_tokens))


@dataclass(frozen=True)
class ActionScoreReport:
    scores: list[float]
    legal_mask: list[bool]
    selected_index: int
    mode: str = "schema_limited_action_evaluation"

    def to_dict(self) -> dict[str, Any]:
        return {"scores": self.scores, "legal_mask": self.legal_mask,
                "selected_index": self.selected_index,
                "mode": "schema_limited_action_evaluation"}


@torch.no_grad()
def score_finite_actions(
    model: IntegratedModel,
    config: BuildConfig,
    sequence_tokens: list[int],
    candidate_span_ends: list[int],
    legal_mask: list[bool],
) -> ActionScoreReport:
    """Schema-limited finite-action evaluation, typed separately from
    primary generation. Illegal candidates receive -inf scores."""
    if len(candidate_span_ends) != len(legal_mask):
        raise InferenceError("candidate spans and legal mask must align")
    if not any(legal_mask):
        raise InferenceError("at least one legal candidate is required")
    if len(sequence_tokens) > config.model.max_seq:
        raise InferenceError("sequence exceeds the configured context limit")
    tokens = torch.tensor([sequence_tokens], dtype=torch.long)
    spans = torch.tensor([candidate_span_ends], dtype=torch.long)
    mask = torch.tensor([legal_mask], dtype=torch.bool)
    output = model(tokens, action_span_ends=spans, action_mask=mask)
    scores = output.action_scores[0]
    legal_indices = [index for index, legal in enumerate(legal_mask) if legal]
    best = max(legal_indices, key=lambda index: scores[index].item())
    return ActionScoreReport(
        scores=[float(score) for score in scores.tolist()],
        legal_mask=list(legal_mask), selected_index=best)
