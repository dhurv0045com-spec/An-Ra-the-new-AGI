"""Reversible continual learning through candidate LoRA/DoRA adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Callable, Iterable

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dora: bool = False) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.scale = float(alpha) / max(1, self.rank)
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.magnitude = (
            nn.Parameter(base.weight.detach().norm(dim=1)) if dora else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale
        if self.magnitude is not None:
            direction = self.base.weight.detach() + self.scale * (self.lora_b @ self.lora_a)
            norm = direction.norm(dim=1).clamp_min(1e-6)
            delta = delta * (self.magnitude / norm).view(*([1] * (delta.ndim - 1)), -1)
        return base_output + delta


def attach_candidate_adapters(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dora: bool = False,
    predicate: Callable[[str, nn.Linear], bool] | None = None,
) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    attached: list[str] = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if predicate is not None and not predicate(module_name, module):
            continue
        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dora=dora))
        attached.append(module_name)
    return attached


def compute_fisher_diagonal(
    model: nn.Module,
    losses: Iterable[torch.Tensor],
) -> dict[str, torch.Tensor]:
    fisher = {
        name: torch.zeros_like(parameter, device="cpu")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    count = 0
    for loss in losses:
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        for name, parameter in model.named_parameters():
            if name in fisher and parameter.grad is not None:
                fisher[name] += parameter.grad.detach().float().cpu().pow(2)
        count += 1
    if count:
        for name in fisher:
            fisher[name] /= count
    return fisher


def ewc_penalty(
    model: nn.Module,
    reference: dict[str, torch.Tensor],
    fisher: dict[str, torch.Tensor],
    coefficient: float,
) -> torch.Tensor:
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, parameter in model.named_parameters():
        if name not in reference or name not in fisher:
            continue
        penalty = penalty + (
            fisher[name].to(parameter.device)
            * (parameter - reference[name].to(parameter.device)).pow(2)
        ).sum()
    return float(coefficient) * penalty


@dataclass(frozen=True)
class ContinualCandidate:
    candidate_id: str
    adapter_path: str
    base_checkpoint: str
    replay_fraction: float
    ewc_coefficient: float
    eval_report: dict[str, object]


def promote_candidate_atomically(
    candidate: ContinualCandidate,
    promoted_path: str | Path,
    *,
    promotion_allowed: bool,
) -> Path:
    if not promotion_allowed:
        raise RuntimeError("Candidate failed capability promotion.")
    source = Path(candidate.adapter_path)
    target = Path(promoted_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    manifest = target.with_suffix(target.suffix + ".json")
    manifest.write_text(json.dumps(candidate.__dict__, indent=2, sort_keys=True), encoding="utf-8")
    return target
