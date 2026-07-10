"""Minimal reference-policy DPO objective; campaign use remains preflight-gated."""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: N812 - canonical PyTorch alias


def direct_preference_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """Return mean DPO loss for aligned chosen/rejected sequence log-probabilities."""
    tensors = (
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    if beta <= 0 or any(tensor.shape != policy_chosen_logps.shape for tensor in tensors):
        raise ValueError("beta must be positive and all DPO log-prob tensors must share a shape")
    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    return -F.logsigmoid(beta * (policy_margin - reference_margin)).mean()
