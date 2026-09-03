"""Canonical V5 AdamW construction and parameter-group provenance.

The training specification intentionally leaves no optimizer-group choice to a
framework entry point.  This module is the single constructor used by the
canaries and, eventually, the trainer: every trainable model parameter is
owned exactly once.  Grouping is semantic, not dimensional: RMSNorm scales
and affine QK scales (``query_scale``/``key_scale``, shape [heads, head_dim])
never receive weight decay even though the QK scales are rank two; embedding,
attention projections, and FFN matrices do.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


OPTIMIZER_SCHEMA = "anra-v5-optimizer-receipt/v1"
OPTIMIZER_NAME = "AdamW"
BETA1 = 0.9
BETA2 = 0.95
EPSILON = 1e-8
WEIGHT_DECAY = 0.1
PEAK_LEARNING_RATE = 3e-4


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _named_parameters(model: Any) -> list[tuple[str, Any]]:
    """Return one name for each parameter, retaining tied parameters once."""

    # PyTorch's default ``remove_duplicate=True`` is important: tied input
    # and output embeddings are one optimizer parameter, not two updates.
    return list(model.named_parameters())


def validate_parameter_ownership(model: Any, optimizer: Any) -> None:
    """Require that optimizer groups own every model parameter exactly once."""

    model_parameters = [parameter for _, parameter in _named_parameters(model)]
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    model_ids = [id(parameter) for parameter in model_parameters]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("model exposes duplicate parameter identities")
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise ValueError("optimizer owns a parameter more than once")
    if set(model_ids) != set(optimizer_ids):
        missing = len(set(model_ids) - set(optimizer_ids))
        extra = len(set(optimizer_ids) - set(model_ids))
        raise ValueError(f"optimizer parameter ownership mismatch; missing={missing}, extra={extra}")


def optimizer_group_receipt(model: Any, optimizer: Any) -> dict[str, object]:
    """Return a canonical, name-ordered receipt of optimizer group ownership."""

    validate_parameter_ownership(model, optimizer)
    names_by_id = {id(parameter): name for name, parameter in _named_parameters(model)}
    groups: list[dict[str, object]] = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = list(group["params"])
        names = sorted(names_by_id[id(parameter)] for parameter in parameters)
        groups.append(
            {
                "index": index,
                "name": "decay" if float(group["weight_decay"]) == WEIGHT_DECAY else "no_decay",
                "weight_decay": float(group["weight_decay"]),
                "parameter_names": names,
                "parameter_count": len(names),
                "parameter_numel": sum(
                    int(parameter.numel()) for parameter in parameters
                ),
            }
        )
    groups.sort(key=lambda group: (str(group["name"]), int(group["index"])))
    receipt: dict[str, object] = {
        "schema": OPTIMIZER_SCHEMA,
        "optimizer": OPTIMIZER_NAME,
        "beta1": BETA1,
        "beta2": BETA2,
        "epsilon": EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "groups": groups,
        "parameter_count": sum(int(group["parameter_count"]) for group in groups),
        "parameter_numel": sum(int(group["parameter_numel"]) for group in groups),
    }
    receipt["sha256"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    return receipt


def is_normalization_parameter(name: str, parameter: Any) -> bool:
    """Classify by parameter semantics, not just dimensionality.

    RMSNorm scales and affine QK scales never receive weight decay.  QK scales
    have shape [heads, head_dim] (rank two), so a purely dimensional rule
    would wrongly decay them; the name carries the semantics.
    """

    if parameter.ndim < 2:
        return True
    if name.endswith((".query_scale", ".key_scale")):
        return True
    return ".weight" in name and ".norm" in name


def build_adamw_optimizer(
    model: Any,
    *,
    lr: float = PEAK_LEARNING_RATE,
    torch_module: Any | None = None,
) -> Any:
    """Build the specification-mandated AdamW optimizer for ``model``.

    ``torch_module`` is injectable for lightweight framework tests; normal
    callers leave it unset and use the installed PyTorch package.
    """

    torch = torch_module
    if torch is None:
        try:
            import torch as torch_package
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("build_adamw_optimizer requires PyTorch") from exc
        torch = torch_package
    named = _named_parameters(model)
    if not named:
        raise ValueError("model has no parameters")
    if any(not parameter.requires_grad for _, parameter in named):
        raise ValueError("all model parameters must be trainable")
    decay = [
        (name, parameter)
        for name, parameter in named
        if not is_normalization_parameter(name, parameter)
    ]
    no_decay = [
        (name, parameter)
        for name, parameter in named
        if is_normalization_parameter(name, parameter)
    ]
    groups: list[dict[str, object]] = []
    if decay:
        groups.append({"params": [parameter for _, parameter in sorted(decay)], "weight_decay": WEIGHT_DECAY})
    if no_decay:
        groups.append({"params": [parameter for _, parameter in sorted(no_decay)], "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(
        groups,
        lr=float(lr),
        betas=(BETA1, BETA2),
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    validate_parameter_ownership(model, optimizer)
    return optimizer


# Concise aliases for trainer integrations.
build_optimizer = build_adamw_optimizer
group_receipt = optimizer_group_receipt


__all__ = [
    "BETA1",
    "is_normalization_parameter",
    "BETA2",
    "EPSILON",
    "OPTIMIZER_NAME",
    "OPTIMIZER_SCHEMA",
    "PEAK_LEARNING_RATE",
    "WEIGHT_DECAY",
    "build_adamw_optimizer",
    "build_optimizer",
    "group_receipt",
    "optimizer_group_receipt",
    "validate_parameter_ownership",
]
