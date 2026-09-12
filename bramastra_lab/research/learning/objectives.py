"""Loss components (B04): complete-answer cross-entropy and counterfactual
pair grounding.

The supervised denominator is the number of eligible answer tokens plus the
required EOS — loss masks define it explicitly, never padding. The pair
margin loss scores full answer sequences (including EOS) under each goal and
penalizes a goal-blind prediction. All functions here are pure tensor
arithmetic so their correctness is testable without any optimizer update.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

PAIR_SCORE_NORMALIZATIONS = frozenset({"eligible_token_mean", "eligible_token_sum"})


@dataclass(frozen=True)
class AnswerLossReport:
    total: torch.Tensor          # sum over supervised targets (unnormalized)
    target_count: int
    mean: torch.Tensor           # total / target_count (0 when no targets)


def supervised_target_count(loss_mask: torch.Tensor) -> int:
    return int(loss_mask.sum().item())


def answer_eos_loss_sum(logits: torch.Tensor, labels: torch.Tensor,
                        loss_mask: torch.Tensor) -> AnswerLossReport:
    """Masked causal cross-entropy, returned as an unnormalized SUM.

    ``labels`` are already shifted by the batch builder: position t in
    ``labels`` is the target when consuming position t of ``logits``. Loss
    applies exactly where ``loss_mask`` is True; the class space is the full
    physical vocabulary — masking selects positions, it never restricts which
    classes can be predicted.
    """
    if logits.shape[:2] != labels.shape or labels.shape != loss_mask.shape:
        raise ValueError("logits, labels and loss_mask must share [batch, length] shape")
    if logits.shape[-1] < 2:
        raise ValueError("logits must cover at least two classes")
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_labels = labels.reshape(-1)
    flat_mask = loss_mask.reshape(-1)
    if bool((flat_mask & (flat_labels < 0)).any()):
        raise ValueError("supervised positions must carry a valid gold class; "
                         "mask and labels disagree")
    if bool((~flat_mask & (flat_labels >= 0)).any()):
        raise ValueError("unsupervised positions must not carry labels; "
                         "mask and labels disagree")
    per_token = torch.nn.functional.cross_entropy(
        flat_logits, flat_labels.clamp_min(0), reduction="none")
    target_count = supervised_target_count(loss_mask)
    total = (per_token * flat_mask).sum()
    mean = total / target_count if target_count > 0 else torch.zeros((), dtype=total.dtype)
    return AnswerLossReport(total=total, target_count=target_count, mean=mean)


def sequence_logprob_scores(logits: torch.Tensor, labels: torch.Tensor,
                            loss_mask: torch.Tensor, *,
                            normalization: str = "eligible_token_mean") -> torch.Tensor:
    """Per-sequence score of the full answer (all eligible tokens incl. EOS).

    Higher is better. ``normalization`` is a declared rule stored in config:
    ``eligible_token_mean`` (default) divides by the eligible-token count,
    ``eligible_token_sum`` does not normalize.
    """
    if normalization not in PAIR_SCORE_NORMALIZATIONS:
        raise ValueError(f"unknown pair score normalization {normalization!r}")
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_labels = labels.reshape(-1).clamp_min(0)
    log_probs = torch.log_softmax(flat_logits, dim=-1)
    token_lp = log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
    token_lp = token_lp.reshape(labels.shape)
    masked = token_lp * loss_mask
    counts = loss_mask.sum(dim=-1)
    if normalization == "eligible_token_sum":
        return masked.sum(dim=-1)
    safe_counts = counts.clamp_min(1)
    return masked.sum(dim=-1) / safe_counts


def pair_margin_loss(own_scores: torch.Tensor, swapped_scores: torch.Tensor, margin: float,
                     *, active: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """Counterfactual grounding margin over precomputed sequence scores.

    For pair i, ``own_scores[i] = score(y_own | g_own)`` and
    ``swapped_scores[i] = score(y_other | g_own)``. The loss penalizes
    ``score(y_other|g_own) >= score(y_own|g_own) - margin`` with hinge
    ``relu(swapped - own + margin)``. Pairs flagged inactive (legitimately
    identical answers, excluded at rendering) are counted, not relabeled
    negative, and contribute nothing.
    """
    if own_scores.shape != swapped_scores.shape or own_scores.ndim != 1:
        raise ValueError("own_scores and swapped_scores must be 1-D with equal shape")
    if not isinstance(margin, (int, float)) or not torch.isfinite(
            torch.tensor(float(margin))) or margin <= 0:
        raise ValueError("margin must be a positive finite number")
    if active is None:
        active = torch.ones_like(own_scores, dtype=torch.bool)
    if active.shape != own_scores.shape or active.dtype != torch.bool:
        raise ValueError("active must be a bool tensor matching the score shape")
    violation = torch.relu(swapped_scores - own_scores + float(margin))
    counted = int(active.sum().item())
    if counted == 0:
        return torch.zeros((), dtype=own_scores.dtype), 0
    return (violation * active).sum() / counted, counted
