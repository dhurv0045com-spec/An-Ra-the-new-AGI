"""Experimental Layerwise Projected Gradient Accumulation prototype."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class _ProjectedGradient:
    left: torch.Tensor
    singular: torch.Tensor
    right: torch.Tensor

    def reconstruct(self) -> torch.Tensor:
        return (self.left * self.singular.unsqueeze(0)) @ self.right


class LPGAAccumulator:
    """Stores low-rank gradient coordinates between microbatches.

    This is deliberately an experiment, not a claim that full gradient
    materialization is eliminated. Hooks compress gradients as they arrive and
    restore them only at the optimizer boundary.
    """

    def __init__(self, model: torch.nn.Module, rank: int = 32) -> None:
        self.rank = int(rank)
        self._projected: dict[torch.nn.Parameter, _ProjectedGradient] = {}
        self._handles = []
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.ndim == 2:
                self._handles.append(parameter.register_hook(self._hook(parameter)))

    def _hook(self, parameter: torch.nn.Parameter):
        def compress(gradient: torch.Tensor) -> torch.Tensor:
            rank = min(self.rank, min(gradient.shape))
            left, singular, right = torch.linalg.svd(
                gradient.detach().float(), full_matrices=False
            )
            projected = _ProjectedGradient(
                left[:, :rank].cpu(),
                singular[:rank].cpu(),
                right[:rank, :].cpu(),
            )
            previous = self._projected.get(parameter)
            if previous is None:
                self._projected[parameter] = projected
            else:
                combined = previous.reconstruct() + projected.reconstruct()
                l2, s2, r2 = torch.linalg.svd(combined, full_matrices=False)
                self._projected[parameter] = _ProjectedGradient(
                    l2[:, :rank], s2[:rank], r2[:rank, :]
                )
            return torch.zeros_like(gradient)

        return compress

    def materialize(self) -> None:
        for parameter, projected in self._projected.items():
            parameter.grad = projected.reconstruct().to(
                device=parameter.device, dtype=parameter.dtype
            )

    def clear(self) -> None:
        self._projected.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
