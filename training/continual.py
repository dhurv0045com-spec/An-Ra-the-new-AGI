"""Reversible continual learning through candidate LoRA/DoRA adapters."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time
from typing import Callable, Iterable, Sequence
import uuid

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


@dataclass(frozen=True)
class ContinualRunResult:
    status: str
    usable_examples: int
    candidate: ContinualCandidate | None
    promotion_manifest: dict[str, object] | None
    blockers: tuple[str, ...]


class ContinualLearningOrchestrator:
    """Own the isolated adapter candidate, evaluation, quarantine, and promotion lifecycle."""

    MIN_EXAMPLES = 100
    SEEDS = (1301, 1302, 1303)

    def __init__(
        self,
        candidate_dir: str | Path,
        promoted_adapter: str | Path,
        *,
        replay_fraction: float = 0.20,
        ewc_coefficient: float = 0.10,
        rank: int = 8,
        dora: bool = True,
    ) -> None:
        self.candidate_dir = Path(candidate_dir)
        self.promoted_adapter = Path(promoted_adapter)
        self.replay_fraction = float(replay_fraction)
        self.ewc_coefficient = float(ewc_coefficient)
        self.rank = int(rank)
        self.dora = bool(dora)

    @staticmethod
    def _adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
            if ".lora_a" in name or ".lora_b" in name or ".magnitude" in name
        }

    def run(
        self,
        *,
        model: nn.Module,
        base_checkpoint: str | Path,
        examples: Sequence[object],
        replay_examples: Sequence[object],
        train_candidate: Callable[
            [nn.Module, Sequence[object], dict[str, torch.Tensor], float], None
        ],
        evaluate: Callable[[nn.Module | None, int], dict[str, object]],
        smoke_test: Callable[[Path], bool] | None = None,
        deployment_checks: dict[str, bool] | None = None,
    ) -> ContinualRunResult:
        usable = len(examples)
        if usable < self.MIN_EXAMPLES:
            return ContinualRunResult(
                status="skipped",
                usable_examples=usable,
                candidate=None,
                promotion_manifest=None,
                blockers=(f"usable examples {usable} < {self.MIN_EXAMPLES}",),
            )

        attached = attach_candidate_adapters(
            model,
            rank=self.rank,
            dora=self.dora,
        )
        if not attached:
            return ContinualRunResult(
                status="blocked",
                usable_examples=usable,
                candidate=None,
                promotion_manifest=None,
                blockers=("no eligible linear modules for adapters",),
            )
        replay_count = min(
            len(replay_examples),
            max(1, math.ceil(usable * self.replay_fraction)),
        )
        training_rows = list(examples) + list(replay_examples[:replay_count])
        reference = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        train_candidate(model, training_rows, reference, self.ewc_coefficient)

        candidate_id = f"adapter-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = self.candidate_dir / f"{candidate_id}.pt"
        torch.save(
            {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "base_checkpoint": str(base_checkpoint),
                "attached_modules": attached,
                "state_dict": self._adapter_state(model),
            },
            adapter_path,
        )

        baseline_reports = [evaluate(None, seed) for seed in self.SEEDS]
        candidate_reports = [evaluate(model, seed) for seed in self.SEEDS]
        from evaluation.promotion import (
            CapabilityPromotionGate,
            DeploymentPromotionGate,
            combine_promotion_decisions,
            promote_checkpoint_atomically,
        )

        owner_baseline = sum(
            float(report.get("dimensions", {}).get("owner_task", 0.0))
            for report in baseline_reports
        ) / len(baseline_reports)
        owner_candidate = sum(
            float(report.get("dimensions", {}).get("owner_task", 0.0))
            for report in candidate_reports
        ) / len(candidate_reports)
        capability_decision = CapabilityPromotionGate().compare(
            baseline_reports,
            candidate_reports,
            owner_baseline=owner_baseline,
            owner_candidate=owner_candidate,
        )
        deployment_decision = DeploymentPromotionGate().evaluate(
            deployment_checks or {}
        )
        decision = combine_promotion_decisions(
            capability_decision,
            deployment_decision,
        )
        candidate = ContinualCandidate(
            candidate_id=candidate_id,
            adapter_path=str(adapter_path),
            base_checkpoint=str(base_checkpoint),
            replay_fraction=replay_count / max(1, len(training_rows)),
            ewc_coefficient=self.ewc_coefficient,
            eval_report={
                "seed_count": len(self.SEEDS),
                "baseline": baseline_reports,
                "candidate": candidate_reports,
                "decision": asdict(decision),
            },
        )
        report_path = adapter_path.with_suffix(".json")
        report_path.write_text(
            json.dumps(asdict(candidate), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not decision.allowed:
            from anra.anra_paths import QUARANTINE_DIR

            QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
            quarantined = QUARANTINE_DIR / adapter_path.name
            shutil.move(str(adapter_path), quarantined)
            return ContinualRunResult(
                status="quarantined",
                usable_examples=usable,
                candidate=candidate,
                promotion_manifest=None,
                blockers=decision.reasons,
            )

        manifest = promote_checkpoint_atomically(
            candidate_path=adapter_path,
            promoted_path=self.promoted_adapter,
            decision=decision,
            metadata={
                "artifact_type": "continual_adapter",
                "candidate": asdict(candidate),
                "base_checkpoint_immutable": True,
            },
            smoke_test=smoke_test,
        )
        return ContinualRunResult(
            status="promoted",
            usable_examples=usable,
            candidate=candidate,
            promotion_manifest=manifest,
            blockers=(),
        )


def proposal_auto_application_allowed(
    proposal_results: Iterable[bool],
    *,
    minimum_improvement_rate: float = 0.20,
) -> bool:
    outcomes = list(proposal_results)
    return bool(outcomes) and sum(bool(value) for value in outcomes) / len(outcomes) >= float(
        minimum_improvement_rate
    )


def assess_continual_readiness(usable_examples: int) -> dict[str, object]:
    count = max(0, int(usable_examples))
    return {
        "usable_examples": count,
        "minimum_examples": ContinualLearningOrchestrator.MIN_EXAMPLES,
        "ready": count >= ContinualLearningOrchestrator.MIN_EXAMPLES,
        "action": "train_isolated_adapter" if count >= ContinualLearningOrchestrator.MIN_EXAMPLES else "skip",
    }


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
