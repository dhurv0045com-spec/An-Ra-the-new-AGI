"""True projected-conflicting-gradient support for protected parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class PCGradTelemetry:
    dot_product: float
    cosine: float
    conflict: bool
    projection_norm: float


def project_conflicting_gradient(
    primary: torch.Tensor,
    secondary: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, PCGradTelemetry]:
    p = primary.reshape(-1)
    s = secondary.reshape(-1)
    dot = torch.dot(p, s)
    denom = torch.dot(s, s).clamp_min(eps)
    projected = primary
    correction = torch.zeros_like(primary)
    conflict = bool(dot.detach().item() < 0.0)
    if conflict:
        correction = (dot / denom) * secondary
        projected = primary - correction
    cosine = dot / (p.norm() * s.norm()).clamp_min(eps)
    return projected, PCGradTelemetry(
        dot_product=float(dot.detach().item()),
        cosine=float(cosine.detach().item()),
        conflict=conflict,
        projection_norm=float(correction.detach().norm().item()),
    )


def apply_pcgrad(
    owner_loss: torch.Tensor,
    other_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
) -> list[PCGradTelemetry]:
    params = [parameter for parameter in parameters if parameter.requires_grad]
    owner_grads = torch.autograd.grad(owner_loss, params, retain_graph=True, allow_unused=True)
    other_grads = torch.autograd.grad(other_loss, params, retain_graph=True, allow_unused=True)
    telemetry: list[PCGradTelemetry] = []
    for parameter, owner_grad, other_grad in zip(params, owner_grads, other_grads):
        if owner_grad is None and other_grad is None:
            continue
        if owner_grad is None:
            parameter.grad = other_grad
            continue
        if other_grad is None:
            parameter.grad = owner_grad
            continue
        projected, report = project_conflicting_gradient(owner_grad, other_grad)
        parameter.grad = projected + other_grad
        telemetry.append(report)
    return telemetry
