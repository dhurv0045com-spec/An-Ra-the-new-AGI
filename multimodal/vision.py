"""In-house M2 patch encoder and soft-token projector; no external weights."""

from __future__ import annotations

import torch
from torch import nn


class InHouseVisionEncoder(nn.Module):
    def __init__(self, *, width: int = 128, patch_size: int = 16) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(images).flatten(2).transpose(1, 2)
        return self.norm(tokens)


class VisionSoftTokenProjector(nn.Module):
    def __init__(self, vision_dim: int, d_model: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(vision_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        return self.layers(vision_tokens)
