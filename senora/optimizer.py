"""Real production optimizer implementation and parameter grouping manifest.

Implements AdamW parameter grouping with explicit weight decay isolation:
- 2D weight matrices (linear projections) -> weight decay = 0.1
- 1D normalization weights & affine QK scales -> weight decay = 0.0
- Embedding / unembedding weights -> weight decay = 0.0 (preserves vocabulary manifold stability)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParameterEntry:
    name: str
    shape: list[int]
    numel: int
    decay_group: str  # "decay_0.1" or "no_decay_0.0"
    lr_group: str  # "base"
    trainable: bool
    sha256: str


@dataclass(frozen=True, slots=True)
class OptimizerManifest:
    schema: str
    family: str
    beta1: float
    beta2: float
    epsilon: float
    base_lr: float
    weight_decay: float
    total_trainable_parameters: int
    decayed_parameters_count: int
    non_decayed_parameters_count: int
    parameters: list[ParameterEntry]

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def classify_parameter_decay(name: str, tensor: Any) -> str:
    """Classify whether a parameter should receive weight decay.
    
    1D tensors (biases, layer norms, RMSNorm weights, affine QK-norm scales)
    and token embeddings do NOT receive weight decay.
    2D weight matrices (linear projections) DO receive weight decay.
    """
    if "norm" in name.lower() or "scale" in name.lower():
        return "no_decay_0.0"
    if "embedding" in name.lower():
        return "no_decay_0.0"
    if hasattr(tensor, "ndim") and tensor.ndim <= 1:
        return "no_decay_0.0"
    return "decay_0.1"


try:
    import torch

    def build_p35_optimizer(
        model: torch.nn.Module,
        *,
        learning_rate: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.95,
        epsilon: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> tuple[torch.optim.AdamW, OptimizerManifest]:
        """Construct verified AdamW optimizer with explicit decay isolation and parameter manifest."""
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        param_entries: list[ParameterEntry] = []

        # Ensure parameters are uniquely tracked (especially with tied weights)
        seen_params: set[int] = set()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)

            group = classify_parameter_decay(name, param)
            if group == "decay_0.1":
                decay_params.append(param)
            else:
                no_decay_params.append(param)

            # Compute initial parameter hash
            param_data = param.detach().cpu().float().numpy().tobytes()
            param_sha = hashlib.sha256(param_data).hexdigest()

            param_entries.append(
                ParameterEntry(
                    name=name,
                    shape=list(param.shape),
                    numel=param.numel(),
                    decay_group=group,
                    lr_group="base",
                    trainable=param.requires_grad,
                    sha256=param_sha,
                )
            )

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
        )

        # Invariant check: optimizer must own the exact same live Parameter objects as model
        model_trainable_params = {p for p in model.parameters() if p.requires_grad}
        optimizer_owned_params = {
            p for group in optimizer.param_groups for p in group["params"]
        }
        if model_trainable_params != optimizer_owned_params:
            missing = model_trainable_params - optimizer_owned_params
            extra = optimizer_owned_params - model_trainable_params
            raise AssertionError(
                f"Optimizer parameter ownership mismatch: missing={len(missing)}, extra={len(extra)}"
            )

        manifest = OptimizerManifest(
            schema="anra-v5-optimizer-manifest/v1",
            family="AdamW-decay-grouped",
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            base_lr=learning_rate,
            weight_decay=weight_decay,
            total_trainable_parameters=sum(e.numel for e in param_entries),
            decayed_parameters_count=sum(p.numel() for p in decay_params),
            non_decayed_parameters_count=sum(p.numel() for p in no_decay_params),
            parameters=param_entries,
        )

        return optimizer, manifest

except ImportError:  # pragma: no cover
    build_p35_optimizer = None  # type: ignore