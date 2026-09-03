"""Query-swap contrastive challenger for matched-compute E3 comparison.

Frozen challenger: lambda in {0, 0.05, 0.15} set by the caller outside this
pure loss, margin exactly 0, one gold plus three plausible negatives.
Refuses to compute unless explicitly enabled: an unproven auxiliary must not
enter the expensive run silently.
"""

from __future__ import annotations

from typing import Any


def query_swap_loss(
    gold_scores: Any,
    negative_scores: Any,
    *,
    enabled: bool,
    margin: float = 0.0,
    torch_module: Any = None,
) -> Any:
    """Hinge over best-negative minus gold; compiled only when enabled."""

    if not enabled:
        raise ValueError("query-swap objective is not promoted; launch uses CE only")
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if margin != 0.0:
        raise ValueError("the frozen challenger fixes margin at exactly zero")
    if gold_scores.shape != negative_scores.shape[:1]:
        raise ValueError("one gold score per query row is required")
    if negative_scores.ndim != 2 or negative_scores.shape[1] != 3:
        raise ValueError("the frozen challenger uses one gold plus three negatives")
    best_negative, _ = negative_scores.max(dim=1)
    return torch.clamp(margin + best_negative - gold_scores, min=0.0).mean()


__all__ = ["query_swap_loss"]
