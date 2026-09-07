"""Recurrent prediction and query selection with no privileged rule input.

The policy imitates a training-only information-gain teacher. At evaluation
it receives observations and the requested target, never a hypothesis table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LearnerConfig:
    bits: int = 6
    width: int = 96

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 10 or self.width < 8:
            raise ValueError("bits must be 2..10 and width at least 8")


class Investigator(nn.Module):
    """GRU working memory, binary outcome head, and shared query scorer."""

    def __init__(self, config: LearnerConfig):
        super().__init__()
        self.config = config
        w, d = config.width, config.bits
        self.observation = nn.Sequential(nn.Linear(d + 1, w), nn.Tanh())
        self.memory = nn.GRUCell(w, w)
        self.predictor = nn.Sequential(nn.Linear(w + d, w), nn.SiLU(), nn.Linear(w, 1))
        self.selector = nn.Sequential(nn.Linear(w + 2 * d, w), nn.SiLU(), nn.Linear(w, 1))

    def encode(self, observations: Tensor, lengths: Tensor) -> Tensor:
        if observations.ndim != 3 or observations.shape[-1] != self.config.bits + 1:
            raise ValueError("observations must be [batch,time,bits+1]")
        if lengths.shape != (observations.shape[0],):
            raise ValueError("one length required per episode")
        if bool(((lengths < 0) | (lengths > observations.shape[1])).any()):
            raise ValueError("history length out of bounds")
        state = observations.new_zeros((len(observations), self.config.width))
        for t in range(observations.shape[1]):
            updated = self.memory(self.observation(observations[:, t]), state)
            state = torch.where((lengths > t).unsqueeze(-1), updated, state)
        return state

    def predict(self, state: Tensor, target: Tensor) -> Tensor:
        return self.predictor(torch.cat((state, target), dim=-1)).squeeze(-1)

    def select(self, state: Tensor, target: Tensor, candidates: Tensor, legal: Tensor) -> Tensor:
        if legal.ndim != 2 or legal.shape != (len(state), len(candidates)):
            raise ValueError("legal mask shape mismatch")
        if not bool(legal.any(dim=1).all()):
            raise ValueError("every episode needs a legal query")
        n = len(candidates)
        features = torch.cat((state[:, None].expand(-1, n, -1),
                              target[:, None].expand(-1, n, -1),
                              candidates[None].expand(len(state), -1, -1)), dim=-1)
        return self.selector(features).squeeze(-1).masked_fill(~legal, -1e9)

    def specification(self) -> dict:
        return {**asdict(self.config), "parameters": sum(p.numel() for p in self.parameters()),
                "initialization": "random", "policy_training": "training-only information-gain imitation"}
