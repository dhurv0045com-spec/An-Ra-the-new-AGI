"""Causal-extension trainer used by the canonical brain trainer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from training.anra_optimizer import build_optimizer_with_report
from training.cdr import CorrectedFailureCurriculum
from training.mixed_precision import MixedPrecisionTrainer
from training.pcgrad import PCGradAccumulator
from training.wsd_scheduler import get_wsd_schedule


@dataclass(frozen=True)
class CausalLosses:
    causal_type: torch.Tensor
    variable_extraction: torch.Tensor
    intervention_extraction: torch.Tensor
    confounder: torch.Tensor
    requires_experiment: torch.Tensor
    counterfactual_consistency: torch.Tensor
    verified_answer: torch.Tensor
    calibration: torch.Tensor
    sparsity: torch.Tensor
    zero_gate: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return (
            self.causal_type
            + self.variable_extraction
            + self.intervention_extraction
            + self.confounder
            + self.requires_experiment
            + self.counterfactual_consistency
            + self.verified_answer
            + self.calibration
            + 0.001 * self.sparsity
            + 0.01 * self.zero_gate
        )


class CausalExtensionTrainer:
    def __init__(
        self,
        model: nn.Module,
        extension: nn.Module,
        *,
        total_steps: int,
        warmup_steps: int,
        cdr_path: str,
        optimizer_name: str = "auto",
        lr: float = 3e-4,
    ) -> None:
        self.model = model
        self.extension = extension
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in extension.parameters():
            parameter.requires_grad = True
        self.optimizer, self.optimizer_report = build_optimizer_with_report(
            extension, optimizer_name=optimizer_name, lr=lr, weight_decay=0.0
        )
        self.scheduler = get_wsd_schedule(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        self.pcgrad = PCGradAccumulator(extension.parameters())
        self.cdr = CorrectedFailureCurriculum(cdr_path)
        self.mp = MixedPrecisionTrainer(device=next(model.parameters()).device)

    @staticmethod
    def losses(
        language_loss: torch.Tensor,
        evidence: tuple[dict[str, torch.Tensor], ...],
        labels: dict[str, torch.Tensor],
    ) -> CausalLosses:
        if not evidence:
            raise RuntimeError("Causal extension produced no routing evidence.")
        latest = evidence[-1]
        type_loss = F.cross_entropy(latest["routing_logits"], labels["causal_type"])
        variable_extraction = F.binary_cross_entropy_with_logits(
            latest["variable_logits"],
            labels["variable_mask"].float(),
        )
        intervention_extraction = F.binary_cross_entropy_with_logits(
            latest["intervention_logits"],
            labels["intervention_mask"].float(),
        )
        confounder = F.binary_cross_entropy(
            latest["confounder_risk"], labels["has_confounder"].float()
        )
        requires_experiment = F.binary_cross_entropy(
            latest["requires_experiment"].clamp(1e-6, 1 - 1e-6),
            labels["requires_experiment"].float(),
        )
        counterfactual_target = labels["counterfactual_embedding"].float()
        consistency_per_item = 1.0 - F.cosine_similarity(
            latest["counterfactual_embedding"],
            F.normalize(counterfactual_target, dim=-1),
            dim=-1,
        )
        counterfactual_mask = labels["is_counterfactual"].float()
        counterfactual_consistency = (
            (consistency_per_item * counterfactual_mask).sum()
            / counterfactual_mask.sum().clamp_min(1.0)
        )
        calibration = F.mse_loss(latest["confidence"], labels["confidence"].float())
        return CausalLosses(
            causal_type=type_loss,
            variable_extraction=variable_extraction,
            intervention_extraction=intervention_extraction,
            confounder=confounder,
            requires_experiment=requires_experiment,
            counterfactual_consistency=counterfactual_consistency,
            verified_answer=language_loss,
            calibration=calibration,
            sparsity=latest["routing_sparsity"],
            zero_gate=latest["gate"].abs(),
        )

    def step(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        labels: dict[str, torch.Tensor],
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        with self.mp.autocast():
            _, language_loss, evidence = self.model.forward_cognitive(
                input_ids,
                target_ids,
                attention_mask=attention_mask,
            )
            assert language_loss is not None
            losses = self.losses(language_loss, evidence, labels)
        self.pcgrad.accumulate(
            owner_loss=losses.total,
            other_loss=language_loss,
            grad_scale=self.mp.scale,
        )
        telemetry = self.pcgrad.materialize()
        if self.mp._needs_scaler:
            self.mp.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.extension.parameters(), 1.0)
        self.mp.step(self.optimizer)
        self.mp.update()
        self.scheduler.step()
        self.pcgrad.clear()
        return {
            "total": float(losses.total.detach()),
            "causal_type": float(losses.causal_type.detach()),
            "variable_extraction": float(losses.variable_extraction.detach()),
            "intervention_extraction": float(losses.intervention_extraction.detach()),
            "confounder": float(losses.confounder.detach()),
            "requires_experiment": float(losses.requires_experiment.detach()),
            "counterfactual_consistency": float(
                losses.counterfactual_consistency.detach()
            ),
            "verified_answer": float(losses.verified_answer.detach()),
            "calibration": float(losses.calibration.detach()),
            "sparsity": float(losses.sparsity.detach()),
            "gate": float(losses.zero_gate.detach()),
            "pcgrad_conflict_rate": (
                sum(item.conflict for item in telemetry) / len(telemetry) if telemetry else 0.0
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.mp.state_dict(),
            "optimizer_report": self.optimizer_report,
        }


class CausalCorpusDataset(Dataset):
    TYPE_IDS = {
        "observational": 0,
        "interventional": 1,
        "counterfactual": 2,
        "confounded": 3,
    }

    def __init__(self, path: str | Path, tokenizer, block_size: int, rank: int) -> None:
        self.rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.rank = int(rank)
        self.pad_id = int(tokenizer.pad_token_id)
        self.bos_id = int(tokenizer.bos_token_id)
        self.eos_id = int(tokenizer.eos_token_id)

    def __len__(self) -> int:
        return len(self.rows)

    def _span_mask(self, ids: list[int], phrases: list[str]) -> torch.Tensor:
        mask = torch.zeros(self.block_size, dtype=torch.float32)
        for phrase in phrases:
            phrase_ids = self.tokenizer.encode(str(phrase), add_special_tokens=False)
            if not phrase_ids:
                continue
            for start in range(max(0, len(ids) - len(phrase_ids) + 1)):
                if ids[start : start + len(phrase_ids)] == phrase_ids:
                    mask[start : start + len(phrase_ids)] = 1.0
        return mask

    def __getitem__(self, index: int):
        row = self.rows[index]
        prompt_ids = self.tokenizer.encode(
            f"H: {row['prompt']}\nANRA:",
            add_special_tokens=False,
        )
        answer_ids = self.tokenizer.encode(
            f" {row['answer']}",
            add_special_tokens=False,
        )
        full = [self.bos_id, *prompt_ids, *answer_ids, self.eos_id]
        full = full[: self.block_size + 1]
        full += [self.pad_id] * (self.block_size + 1 - len(full))
        x_ids = full[: self.block_size]
        y_ids = full[1 : self.block_size + 1]
        attention = [int(token != self.pad_id) for token in x_ids]
        digest = hashlib.sha256(row["content_hash"].encode("utf-8")).digest()
        target = torch.tensor(
            [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(self.rank)],
            dtype=torch.float32,
        )
        intervention = [row["intervention"]] if row.get("intervention") else []
        return {
            "input_ids": torch.tensor(x_ids, dtype=torch.long),
            "target_ids": torch.tensor(y_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "causal_type": torch.tensor(self.TYPE_IDS[row["causal_type"]], dtype=torch.long),
            "variable_mask": self._span_mask(x_ids, list(row.get("variables", []))),
            "intervention_mask": self._span_mask(x_ids, intervention),
            "has_confounder": torch.tensor(bool(row.get("confounders")), dtype=torch.float32),
            "requires_experiment": torch.tensor(
                bool(row["requires_experiment"]),
                dtype=torch.float32,
            ),
            "confidence": torch.tensor(float(row["confidence_label"]), dtype=torch.float32),
            "counterfactual_embedding": target,
            "is_counterfactual": torch.tensor(
                row["causal_type"] == "counterfactual",
                dtype=torch.float32,
            ),
        }
