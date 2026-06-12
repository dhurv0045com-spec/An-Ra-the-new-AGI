"""Lightweight uncertain action-conditioned world model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PredictiveWorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 128,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.next_state = nn.Linear(hidden_dim, state_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        self.termination = nn.Linear(hidden_dim, 1)
        self.log_variance = nn.Linear(hidden_dim, state_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state_features = F.silu(self.state_encoder(state))
        action_features = F.silu(self.action_encoder(action))
        if hidden is None:
            hidden = torch.zeros_like(state_features)
        hidden = self.gru(torch.cat([state_features, action_features], dim=-1), hidden)
        return {
            "next_state": self.next_state(hidden),
            "reward": self.reward(hidden).squeeze(-1),
            "termination_probability": torch.sigmoid(self.termination(hidden)).squeeze(-1),
            "epistemic_uncertainty": F.softplus(self.log_variance(hidden)),
            "hidden": hidden,
        }

    @staticmethod
    def activation_allowed(
        *,
        held_out_accuracy: float,
        planning_improvement: float,
    ) -> bool:
        return held_out_accuracy >= 0.70 and planning_improvement >= 0.10
