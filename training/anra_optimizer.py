from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch.optim import AdamW


OPTIMIZER_REPORT_SCHEMA_VERSION = 3
OPTIMIZER_CANDIDATES = (
    "adamw",
    "adam8bit",
    "adafactor",
    "muon",
    "scale",
    "galore",
    "qgalore",
)
IDENTITY_PARAMETER_PATTERNS = (
    "esv_module",
    "hal_module",
    "rim_modules",
    "civ",
    "dstp_temperature",
    "residual_depth",
    "layer_temperature_bias",
    "token_embedding_table",
    "token_embedding",
    "lm_head",
    "norm_f",
)


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "auto"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    identity_lr_multiplier: float = 2.0
    galore_rank: int = 64
    galore_projection_gap: int = 200


def _trainable_named_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def _param_count(params: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(parameter.numel() for parameter in params))


def _adamw_state_bytes(param_count: int, *, bytes_per_state_value: int = 4) -> int:
    return int(param_count * 2 * bytes_per_state_value)


def is_identity_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    return parameter.ndim < 2 or any(pattern in name for pattern in IDENTITY_PARAMETER_PATTERNS)


def partition_parameters(
    model: torch.nn.Module,
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    identity: list[tuple[str, torch.nn.Parameter]] = []
    matrix: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in _trainable_named_parameters(model):
        (identity if is_identity_parameter(name, parameter) else matrix).append((name, parameter))
    return identity, matrix


def _is_muon_param(name: str, parameter: torch.nn.Parameter) -> bool:
    return parameter.ndim == 2 and not is_identity_parameter(name, parameter)


def candidate_report(model: torch.nn.Module, *, config: OptimizerConfig | None = None) -> dict[str, object]:
    cfg = config or OptimizerConfig()
    named = _trainable_named_parameters(model)
    identity_named, matrix_named = partition_parameters(model)
    trainable = _param_count(parameter for _, parameter in named)
    identity = _param_count(parameter for _, parameter in identity_named)
    matrix = _param_count(parameter for _, parameter in matrix_named)
    muon = _param_count(parameter for name, parameter in named if _is_muon_param(name, parameter))
    candidates = [
        {
            "name": "adamw",
            "status": "available",
            "implementation": "torch.optim.AdamW",
            "optimizer_state_bytes_estimate": _adamw_state_bytes(trainable),
        },
        {
            "name": "adam8bit",
            "status": "optional_dependency",
            "implementation": "bitsandbytes.optim.AdamW8bit",
            "optimizer_state_bytes_estimate": trainable * 2,
        },
        {
            "name": "adafactor",
            "status": "optional_dependency",
            "implementation": "transformers.Adafactor",
            "optimizer_state_bytes_estimate": 0,
        },
        {
            "name": "muon",
            "status": "optional_dependency",
            "implementation": "muon.Muon",
            "muon_params": muon,
            "optimizer_state_bytes_estimate": _adamw_state_bytes(trainable - muon),
        },
        {
            "name": "galore",
            "status": "optional_dependency",
            "implementation": "galore_torch.GaLoreAdamW8bit or GaLoreAdamW",
            "optimizer_state_bytes_estimate": 0,
        },
        {
            "name": "scale",
            "status": "watchlist_unavailable",
            "implementation": "not installed",
            "optimizer_state_bytes_estimate": _adamw_state_bytes(trainable),
        },
        {
            "name": "qgalore",
            "status": "watchlist_unavailable",
            "implementation": "not installed",
            "optimizer_state_bytes_estimate": _adamw_state_bytes(trainable),
        },
    ]
    return {
        "schema_version": OPTIMIZER_REPORT_SCHEMA_VERSION,
        "generated_at": time.time(),
        "config": {**asdict(cfg), "betas": list(cfg.betas)},
        "trainable_params": trainable,
        "identity_params": identity,
        "matrix_params": matrix,
        "identity_patterns": list(IDENTITY_PARAMETER_PATTERNS),
        "candidates": candidates,
    }


def _regular_groups(
    matrix_params: list[torch.nn.Parameter],
    identity_params: list[torch.nn.Parameter],
    cfg: OptimizerConfig,
) -> list[dict[str, object]]:
    identity_lr = min(cfg.lr * cfg.identity_lr_multiplier, cfg.lr + 3e-4)
    return [
        {"params": matrix_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": identity_params, "lr": identity_lr, "weight_decay": 0.0},
    ]


def repair_optimizer_param_group_defaults(optimizer: torch.optim.Optimizer) -> tuple[str, ...]:
    """Restore defaults omitted by checkpoints made with older optimizer versions."""
    defaults = dict(getattr(optimizer, "defaults", {}))
    repaired: set[str] = set()
    for group in optimizer.param_groups:
        for key, value in defaults.items():
            if key not in group:
                group[key] = value
                repaired.add(key)
    return tuple(sorted(repaired))


def repair_optimizer_resume_state(optimizer: torch.optim.Optimizer) -> tuple[str, ...]:
    """Repair optimizer metadata and discard only malformed Adafactor moments.

    Older Transformers releases saved Adafactor state with different fields.
    Loading those tensors into a newer release can succeed but fail at the first
    optimizer step. Model weights and the global training step remain valid, so
    reset only the incompatible optimizer moments in that case.
    """
    repaired = list(repair_optimizer_param_group_defaults(optimizer))
    if optimizer.__class__.__name__ != "Adafactor":
        return tuple(repaired)

    invalid_state = False
    for group in optimizer.param_groups:
        for parameter in group.get("params", []):
            state = optimizer.state.get(parameter, {})
            if not state:
                continue
            try:
                factored, use_first_moment = optimizer._get_options(group, parameter.shape)
            except (AttributeError, KeyError, TypeError):
                invalid_state = True
                break
            required = {"step"}
            if factored:
                required.update({"exp_avg_sq_row", "exp_avg_sq_col"})
            else:
                required.add("exp_avg_sq")
            if use_first_moment:
                required.add("exp_avg")
            if not required.issubset(state):
                invalid_state = True
                break
        if invalid_state:
            break

    if invalid_state:
        optimizer.state.clear()
        repaired.append("adafactor_moments_reset")
    return tuple(repaired)


def build_optimizer_with_report(
    model: torch.nn.Module,
    *,
    optimizer_name: str = "auto",
    lr: float = 3e-4,
    weight_decay: float = 0.01,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    requested = optimizer_name.strip().lower()
    if requested not in {"auto", *OPTIMIZER_CANDIDATES}:
        raise ValueError(
            f"Unknown optimizer {optimizer_name!r}. Expected auto or {', '.join(OPTIMIZER_CANDIDATES)}"
        )
    cfg = OptimizerConfig(name=requested, lr=lr, weight_decay=weight_decay)
    report = candidate_report(model, config=cfg)
    named = _trainable_named_parameters(model)
    identity_named, matrix_named = partition_parameters(model)
    identity_params = [parameter for _, parameter in identity_named]
    matrix_params = [parameter for _, parameter in matrix_named]
    groups = _regular_groups(matrix_params, identity_params, cfg)

    selected = requested
    actual = requested
    status = "active"
    reason = ""

    def adamw() -> torch.optim.Optimizer:
        return AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)

    def adafactor() -> torch.optim.Optimizer:
        from transformers import Adafactor  # type: ignore

        return Adafactor(
            groups,
            lr=cfg.lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )

    if requested == "adamw":
        optimizer = adamw()
    elif requested == "adam8bit":
        try:
            from bitsandbytes.optim import AdamW8bit  # type: ignore

            optimizer = AdamW8bit(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
            reason = "bitsandbytes AdamW8bit active"
        except Exception as exc:
            optimizer = adamw()
            actual, status = "adamw", "fallback"
            reason = f"AdamW8bit unavailable ({exc}); using AdamW"
    elif requested == "adafactor":
        try:
            optimizer = adafactor()
            reason = "transformers Adafactor active"
        except Exception as exc:
            optimizer = adamw()
            actual, status = "adamw", "fallback"
            reason = f"Adafactor unavailable ({exc}); using AdamW"
    elif requested in {"auto", "muon"}:
        muon_params = [parameter for name, parameter in named if _is_muon_param(name, parameter)]
        side_ids = {id(parameter) for parameter in muon_params}
        side_params = [parameter for _, parameter in named if id(parameter) not in side_ids]
        try:
            from muon import Muon  # type: ignore

            muon_groups = [
                {"params": muon_params, "use_muon": True},
                {"params": side_params, "use_muon": False, "weight_decay": 0.0},
            ]
            optimizer = Muon(muon_groups, lr=cfg.lr, adamw_betas=cfg.betas, adamw_eps=cfg.eps)
            actual = "muon"
            selected = "muon" if requested == "auto" else requested
            reason = f"Muon active; projected params={_param_count(muon_params):,}"
        except Exception as exc:
            muon_error = exc
            try:
                optimizer = adafactor()
                actual, status = "adafactor", "fallback"
                reason = f"Muon unavailable ({muon_error}); using memory-light Adafactor"
            except Exception as adafactor_exc:
                optimizer = adamw()
                actual, status = "adamw", "fallback"
                reason = (
                    f"Muon unavailable ({muon_error}); Adafactor unavailable "
                    f"({adafactor_exc}); using identity-aware AdamW"
                )
    elif requested == "galore":
        try:
            try:
                from galore_torch import GaLoreAdamW8bit as GaLoreOptimizer  # type: ignore

                implementation = "GaLoreAdamW8bit"
            except ImportError:
                from galore_torch import GaLoreAdamW as GaLoreOptimizer  # type: ignore

                implementation = "GaLoreAdamW"
            galore_groups = [
                {
                    "params": matrix_params,
                    "rank": cfg.galore_rank,
                    "update_proj_gap": cfg.galore_projection_gap,
                    "scale": 0.25,
                    "proj_type": "std",
                    "weight_decay": cfg.weight_decay,
                },
                {
                    "params": identity_params,
                    "rank": None,
                    "lr": min(cfg.lr * cfg.identity_lr_multiplier, cfg.lr + 3e-4),
                    "weight_decay": 0.0,
                },
            ]
            optimizer = GaLoreOptimizer(galore_groups, lr=cfg.lr, betas=cfg.betas)
            reason = f"{implementation} active with full-rank identity parameters"
        except Exception as exc:
            optimizer = adamw()
            actual, status = "adamw", "fallback"
            reason = f"GaLore unavailable ({exc}); using identity-aware AdamW"
    else:
        optimizer = adamw()
        actual, status = "adamw", "fallback"
        reason = f"{requested.upper()} is unavailable; using identity-aware AdamW"

    report["selected"] = {
        "requested": requested,
        "selected": selected,
        "actual": actual,
        "status": status,
        "reason": reason or f"{actual} selected",
        "lr": cfg.lr,
        "identity_lr": min(cfg.lr * cfg.identity_lr_multiplier, cfg.lr + 3e-4),
        "weight_decay": cfg.weight_decay,
        "param_groups": len(optimizer.param_groups),
        "identity_params": _param_count(identity_params),
        "matrix_params": _param_count(matrix_params),
        "galore_rank": cfg.galore_rank if actual == "galore" else None,
    }
    setattr(optimizer, "_anra_optimizer_report", report)
    selected_report = report["selected"]
    print(
        f"[anra_optimizer] requested={requested} actual={actual} "
        f"status={status} {selected_report['reason']}"
    )
    return optimizer, report


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    optimizer_name: str = "auto",
):
    optimizer, _ = build_optimizer_with_report(
        model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer
