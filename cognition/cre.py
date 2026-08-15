"""Causal reasoning extension with zero-gated low-rank experts."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - canonical PyTorch alias

CausalType = Literal["observational", "interventional", "counterfactual", "unknown"]


@dataclass(frozen=True)
class CausalJudgment:
    query_text: str
    causal_type: CausalType
    confidence: float
    key_variables: tuple[str, ...] = ()
    confounders: tuple[str, ...] = ()
    intervention: str | None = None
    requires_experiment: bool = False
    evidence_required: tuple[str, ...] = ()
    falsification_path: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CausalRouter(nn.Module):
    """Shared low-rank experts; the zero gate makes attachment function-preserving."""

    def __init__(self, d_model: int, rank: int = 32) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.rank = min(int(rank), self.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.rank),
            nn.SiLU(),
            nn.Linear(self.rank, 4),
        )
        self.down = nn.Linear(self.d_model, self.rank, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(self.rank, self.d_model, bias=False) for _ in range(3)]
        )
        self.confidence_head = nn.Linear(self.d_model, 1)
        self.confounder_head = nn.Linear(self.d_model, 1)
        self.requires_experiment_head = nn.Linear(self.d_model, 1)
        self.variable_head = nn.Linear(self.d_model, 1)
        self.intervention_head = nn.Linear(self.d_model, 1)
        self.counterfactual_head = nn.Linear(self.d_model, self.rank)
        self.raw_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            weights = attention_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(pooled)
        routing = F.softmax(logits, dim=-1)
        hidden = self.down(x)
        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=2)
        mixed = (expert_outputs * routing[:, None, :3, None].to(dtype=expert_outputs.dtype)).sum(
            dim=2
        )
        gate = 0.25 * torch.tanh(self.raw_gate)
        output = x + gate * mixed
        return output, {
            "routing_logits": logits,
            "routing_weights": routing,
            "confidence": torch.sigmoid(self.confidence_head(pooled)).squeeze(-1),
            "confounder_risk": torch.sigmoid(self.confounder_head(pooled)).squeeze(-1),
            "requires_experiment": torch.sigmoid(self.requires_experiment_head(pooled)).squeeze(-1),
            "variable_logits": self.variable_head(x).squeeze(-1),
            "intervention_logits": self.intervention_head(x).squeeze(-1),
            "counterfactual_embedding": F.normalize(
                self.counterfactual_head(pooled),
                dim=-1,
            ),
            "routing_sparsity": routing[:, :3].abs().mean(),
            "gate": gate,
        }


class CognitiveCausalExtension(nn.Module):
    """Separately checkpointed causal extension shared across selected layers."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        d_model: int,
        *,
        rank: int = 32,
        integration_layers: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.integration_layers = tuple(sorted({int(i) for i in integration_layers}))
        self.router = CausalRouter(self.d_model, self.rank)

    def applies_to(self, layer_index: int) -> bool:
        return not self.integration_layers or layer_index in self.integration_layers

    def apply_layer(
        self,
        x: torch.Tensor,
        layer_index: int,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.applies_to(layer_index):
            return x, {}
        output, evidence = self.router(x, attention_mask=attention_mask)
        evidence["layer_index"] = torch.tensor(layer_index, device=x.device)
        return output, evidence

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "kind": "causal_low_rank_extension",
            "d_model": self.d_model,
            "rank": self.rank,
            "integration_layers": list(self.integration_layers),
            "gate": float((0.25 * torch.tanh(self.router.raw_gate)).detach().cpu()),
        }


class CausalReasoningEngine:
    _INTERVENTIONAL = re.compile(
        r"\b(if|intervene|intervention|cause|effect|impact|will|should|treat|change)\b",
        re.I,
    )
    _COUNTERFACTUAL = re.compile(
        r"\b(what if|would have|had not|counterfactual|instead of)\b",
        re.I,
    )
    _OBSERVATIONAL = re.compile(
        r"\b(correlat|associated|observed|relationship|linked|rate|survey)\w*\b",
        re.I,
    )
    _VARIABLE = re.compile(r"\b[A-Z][A-Za-z0-9_-]{1,30}\b")

    def __init__(self, model: object | None = None, tokenizer: object | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)

    def classify_query(self, query: str) -> CausalJudgment:
        lower = query.lower()
        counterfactual = bool(self._COUNTERFACTUAL.search(query))
        interventional = bool(self._INTERVENTIONAL.search(query))
        observational = bool(self._OBSERVATIONAL.search(query))
        if counterfactual:
            causal_type: CausalType = "counterfactual"
            confidence = 0.86
        elif interventional:
            causal_type = "interventional"
            confidence = 0.80
        elif observational:
            causal_type = "observational"
            confidence = 0.78
        else:
            causal_type = "unknown"
            confidence = 0.45
        confounders: tuple[str, ...] = ()
        if observational or any(word in lower for word in ("population", "survey", "income")):
            confounders = ("selection bias", "common cause", "measurement bias")
        requires_experiment = causal_type in {"interventional", "counterfactual"} and (
            confidence < 0.85 or "medical" in lower or "treatment" in lower
        )
        variables = tuple(dict.fromkeys(self._VARIABLE.findall(query)))[:8]
        intervention = query[:160] if causal_type == "interventional" else None
        return CausalJudgment(
            query_text=query,
            causal_type=causal_type,
            confidence=confidence,
            key_variables=variables,
            confounders=confounders,
            intervention=intervention,
            requires_experiment=requires_experiment,
            evidence_required=(
                ("controlled experiment", "causal model")
                if requires_experiment
                else ("observational evidence",)
            ),
            falsification_path=(
                "A controlled intervention or valid causal model contradicts the predicted effect."
            ),
        )

    def detect_confounder_risk(self, claim: str) -> float:
        judgment = self.classify_query(claim)
        return min(1.0, 0.2 * len(judgment.confounders))

    @staticmethod
    def caveat(judgment: CausalJudgment) -> str:
        if judgment.requires_experiment:
            return (
                "This causal question cannot be settled by observation alone; "
                "a controlled experiment or validated causal model is required."
            )
        if judgment.confounders:
            return "Possible confounders: " + ", ".join(judgment.confounders)
        return ""
