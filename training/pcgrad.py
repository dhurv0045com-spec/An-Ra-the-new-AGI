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


class PCGradAccumulator:
    """Accumulate separate objectives and replace only protected gradients."""

    def __init__(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.owner = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.other = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.owner_steps = 0
        self.other_steps = 0

    def accumulate(
        self,
        *,
        owner_loss: torch.Tensor | None,
        other_loss: torch.Tensor | None,
        grad_scale: float = 1.0,
    ) -> None:
        for loss, destination, counter in (
            (owner_loss, self.owner, "owner_steps"),
            (other_loss, self.other, "other_steps"),
        ):
            if loss is None:
                continue
            gradients = torch.autograd.grad(
                loss * float(grad_scale),
                self.parameters,
                retain_graph=True,
                allow_unused=True,
            )
            for target, gradient in zip(destination, gradients):
                if gradient is not None:
                    target.add_(gradient.detach())
            setattr(self, counter, getattr(self, counter) + 1)

    def accumulate_existing_gradients(self, *, owner: bool) -> None:
        """Capture gradients from the normal backward pass for one-source batches.

        A batch containing only owner or only non-owner data does not need a
        second ``autograd.grad`` traversal. The regular backward pass already
        produced exactly that source's gradients. We retain protected gradients
        here, clear them from the normal accumulator, and apply PCGrad at the
        optimizer boundary as usual.
        """
        destination = self.owner if owner else self.other
        for parameter, target in zip(self.parameters, destination):
            if parameter.grad is not None:
                target.add_(parameter.grad.detach())
                parameter.grad.zero_()
        counter = "owner_steps" if owner else "other_steps"
        setattr(self, counter, getattr(self, counter) + 1)

    def materialize(self) -> list[PCGradTelemetry]:
        telemetry: list[PCGradTelemetry] = []
        for parameter, owner_gradient, other_gradient in zip(
            self.parameters, self.owner, self.other
        ):
            if self.owner_steps == 0:
                parameter.grad = other_gradient.clone()
            elif self.other_steps == 0:
                parameter.grad = owner_gradient.clone()
            else:
                projected, report = project_conflicting_gradient(
                    owner_gradient, other_gradient
                )
                parameter.grad = projected + other_gradient
                telemetry.append(report)
        return telemetry

    def clear(self) -> None:
        for gradient in (*self.owner, *self.other):
            gradient.zero_()
        self.owner_steps = 0
        self.other_steps = 0
