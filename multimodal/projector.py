"""Frozen encoder plus trainable residual-stream projector."""

from __future__ import annotations

import torch
from torch import nn


class FrozenEncoderProjector(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_dim: int, d_model: int) -> None:
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.projector = nn.Linear(encoder_dim, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.encoder(inputs)
        if isinstance(encoded, (tuple, list)):
            encoded = encoded[0]
        return self.projector(encoded)
