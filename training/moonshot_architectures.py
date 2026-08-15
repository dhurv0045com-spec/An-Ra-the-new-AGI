"""Isolated M1/M3 pilot architectures; neither mutates the production backbone."""

from __future__ import annotations

import torch
from torch import nn


class StateSpaceMixer(nn.Module):
    """Causal gated state-space mixer used only by an M1 pilot branch."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.decay = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gate, update = self.in_proj(values).chunk(2, dim=-1)
        state = torch.zeros_like(update[:, 0])
        outputs = []
        decay = torch.sigmoid(self.decay)
        for token in update.unbind(dim=1):
            state = decay * state + (1.0 - decay) * token
            outputs.append(state)
        return self.out_proj(torch.stack(outputs, dim=1) * torch.sigmoid(gate))


class LatentReasoningChannel(nn.Module):
    """M3 recurrent latent steps, explicitly separate from visible token reasoning."""

    def __init__(self, d_model: int, latent_steps: int = 4) -> None:
        super().__init__()
        self.latent_steps = int(latent_steps)
        self.cell = nn.GRUCell(d_model, d_model)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        state = context.mean(dim=1)
        for _ in range(self.latent_steps):
            state = self.cell(state, state)
        return state
