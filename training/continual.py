"""Reversible continual learning through candidate LoRA/DoRA adapters."""

from __future__ import annotations

import json
import math
import shutil
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from anra.extensions import (
    adapter_state_dict,
    attach_candidate_adapters,
    save_capability_adapter,
    sha256_file,
)
from torch import nn


def compute_fisher_diagonal(
    model: nn.Module,
    losses: Iterable[torch.Tensor],
) -> dict[str, torch.Tensor]:
    fisher = {
        name: torch.zeros_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    count = 0
    for loss in losses:
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        for name, parameter in model.named_parameters():
            if name in fisher and parameter.grad is not None:
                fisher[name] += parameter.grad.detach().float().pow(2)
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
        reference_tensor = reference[name].to(device=parameter.device, dtype=parameter.dtype)
        fisher_tensor = fisher[name].to(device=parameter.device, dtype=parameter.dtype)
        if reference_tensor.shape != parameter.shape or fisher_tensor.shape != parameter.shape:
            continue
        penalty = penalty + (fisher_tensor * (parameter - reference_tensor).pow(2)).sum()
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
    EVALUATION_SEEDS = (1301,)

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
        return adapter_state_dict(model)

    def run(
        self,
        *,
        model: nn.Module,
        base_checkpoint: str | Path,
        tokenizer_hash: str,
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
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        train_candidate(model, training_rows, reference, self.ewc_coefficient)

        candidate_id = f"adapter-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = self.candidate_dir / f"{candidate_id}.pt"
        base_path = Path(base_checkpoint)
        if not base_path.is_file():
            raise FileNotFoundError("continual adapter requires the immutable base checkpoint")
        save_capability_adapter(
            model,
            adapter_path,
            capability_id=candidate_id,
            base_model_profile="anra-v4-180m",
            base_checkpoint_sha256=sha256_file(base_path),
            tokenizer_sha256=tokenizer_hash,
            source_commit="continual-learning-orchestrator",
        )

        baseline_reports = [evaluate(None, seed) for seed in self.EVALUATION_SEEDS]
        candidate_reports = [evaluate(model, seed) for seed in self.EVALUATION_SEEDS]
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
        deployment_decision = DeploymentPromotionGate().evaluate(deployment_checks or {})
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
                "evaluation_run_count": len(self.EVALUATION_SEEDS),
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
            adapter_manifest = adapter_path.with_suffix(adapter_path.suffix + ".manifest.json")
            if adapter_manifest.is_file():
                shutil.move(
                    str(adapter_manifest),
                    QUARANTINE_DIR / adapter_manifest.name,
                )
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
        adapter_manifest = adapter_path.with_suffix(adapter_path.suffix + ".manifest.json")
        promoted_manifest = self.promoted_adapter.with_suffix(
            self.promoted_adapter.suffix + ".manifest.json"
        )
        promoted_manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = promoted_manifest.with_suffix(promoted_manifest.suffix + ".tmp")
        shutil.copy2(adapter_manifest, temporary_manifest)
        temporary_manifest.replace(promoted_manifest)
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
        "action": "train_isolated_adapter"
        if count >= ContinualLearningOrchestrator.MIN_EXAMPLES
        else "skip",
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
