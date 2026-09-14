"""Differentiable training scoring APIs (I02/K8).

The inference helpers (`score_candidates`, `estimate_value`,
`predict_step_finite`) return detached values at an inference boundary.
These training twins return live differentiable tensors so the router can
build real loss terms; using an inference float as a learned objective
silently severs the gradient and is exactly the defect this module closes.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from bramastra_lab.research.config import BuildConfig
from bramastra_lab.research.models import IntegratedModel
from bramastra_lab.research.models.world import json_key


class ScoringError(ValueError):
    """A training-scoring call violated its contract."""


def _model_device(model: IntegratedModel) -> torch.device:
    """Return the device hosting the model parameters.

    Scoring inputs are part of the model call contract.  Constructing them
    with torch's default (CPU) device breaks as soon as the owner moves the
    model to CUDA, even though the scorer itself contains no explicit device
    argument.
    """
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ScoringError("model has no parameters from which to infer device") from exc


def _model_hidden(model, input_ids, padding) -> Any:
    """Canonical hidden path that respects gated reuse (R07).

    All models expose forward_hidden (base IntegratedModel routes to the
    decoder; GatedReuseModel routes through _decoder_with_reuse BEFORE final
    norm). Scorers must use this helper — direct decoder access bypasses
    gates and is forbidden here.
    """
    forward_hidden = getattr(model, "forward_hidden", None)
    if not callable(forward_hidden):
        raise ScoringError(
            "model carries no forward_hidden; gated reuse cannot be honored")
    return forward_hidden(input_ids, padding)


def score_candidates_trainable(model: IntegratedModel, config: BuildConfig,
                               prefix_tokens: Sequence[int],
                               candidates: Sequence[Sequence[int]],
                               *, teacher_distribution: torch.Tensor | None = None,
                               legal_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Batched candidate scoring with a live gradient path.

    Each candidate is an isolated branch of the same prefix (flattened to
    independent batch rows). Returns raw scores [K] (or [B,K] when a batch
    of prefixes is supplied) plus log-softmax over the legal mask.
    """
    if not candidates:
        raise ScoringError("no candidates supplied")
    if any(len(candidate) == 0 for candidate in candidates):
        raise ScoringError("candidate rows must be nonempty")
    if legal_mask is not None:
        if tuple(legal_mask.shape) != (len(candidates),):
            raise ScoringError("legal mask shape mismatch")
        if not bool(legal_mask.to(dtype=torch.bool).any().item()):
            raise ScoringError("legal mask must contain at least one legal candidate")
    prefix_list = list(prefix_tokens)
    max_len = max(len(prefix_list) + len(candidate) for candidate in candidates)
    if max_len > config.model.max_seq:
        raise ScoringError("prefix + candidate exceeds the configured context limit")
    rows = [prefix_list + list(candidate) for candidate in candidates]
    device = _model_device(model)
    input_ids = torch.full((len(rows), max_len), 0, dtype=torch.long,
                           device=device)
    padding = torch.zeros((len(rows), max_len), dtype=torch.bool,
                          device=device)
    for row_index, row in enumerate(rows):
        input_ids[row_index, :len(row)] = torch.tensor(
            row, dtype=torch.long, device=device)
        padding[row_index, :len(row)] = True
    hidden = _model_hidden(model, input_ids, padding)
    spans = torch.tensor([[len(row) - 1] for row in rows], dtype=torch.long,
                         device=device)
    gathered = hidden.gather(
        1, spans.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])).squeeze(1)
    scores = model.action_head(gathered).squeeze(-1)  # [K] live gradient
    if legal_mask is None:
        legal_mask = torch.ones_like(scores, dtype=torch.bool)
    else:
        legal_mask = legal_mask.to(device=device, dtype=torch.bool)
    log_probs = torch.log_softmax(scores.double().masked_fill(~legal_mask, -1e30), dim=-1)
    return {"scores": scores, "log_probs": log_probs.to(scores.dtype),
            "legal_mask": legal_mask}


def value_estimate_trainable(model: IntegratedModel, config: BuildConfig,
                             prefix_tokens: Sequence[int]) -> torch.Tensor:
    """Value head at the action-free prefix with a live gradient path."""
    prefix = list(prefix_tokens)
    if not prefix:
        raise ScoringError("value prefix must be nonempty")
    if len(prefix) > config.model.max_seq:
        raise ScoringError("value prefix exceeds the configured context limit")
    tokens = torch.tensor([prefix], dtype=torch.long, device=_model_device(model))
    return model(tokens, return_value=True).value[0]


def world_transition_token_loss(model: IntegratedModel, config: BuildConfig,
                                prefix_tokens: Sequence[int],
                                action: Mapping[str, Any],
                                target_feedback: Mapping[str, Any]) -> torch.Tensor:
    """Next-public-outcome token NLL over the DECLARED target span (A4).

    The conditioned prefix is exactly the caller's complete context tokens —
    the compiler builds it once upstream (boundary + goal event) so training
    and inference share one representation; this function never re-encodes
    or truncates it. The span is exactly the action event's outcome: the
    feedback event plus required EOS; everything before the span is masked
    out of the loss.
    """
    from bramastra_lab.research.experience.codec import SPECIAL_EOS, \
        encode_event

    condition = list(prefix_tokens)
    if not condition:
        raise ScoringError(
            "world condition prefix must be nonempty; refusing a context-free "
            "transition target")
    condition += encode_event("action", action)
    target_span = encode_event("feedback", target_feedback) + [SPECIAL_EOS]
    sequence = condition + target_span
    if len(sequence) - 1 > config.model.max_seq:
        raise ScoringError("sequence exceeds the context limit")
    device = _model_device(model)
    inputs = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor(sequence[1:], dtype=torch.long, device=device)
    logits = model(inputs).logits[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    # Mask the conditioned prefix: only target-span positions contribute.
    token_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    span_lp = token_lp[len(condition) - 1:]
    return -span_lp.sum()


def deduplicate_support(support: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Deduplicate identical parsed public outcomes before normalization."""
    seen: dict[str, Mapping[str, Any]] = {}
    for option in support:
        key = json_key({"feedback": option.get("feedback"),
                        "terminated": option.get("terminated", False)})
        if key not in seen:
            seen[key] = option
    return list(seen.values())
