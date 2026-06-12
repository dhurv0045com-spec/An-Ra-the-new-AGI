from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch.optim import AdamW


OPTIMIZER_REPORT_SCHEMA_VERSION = 2
OPTIMIZER_CANDIDATES = (
    "adamw",
    "adam8bit",
    "adafactor",
    "muon",
    "scale",
    "galore",
    "qgalore",
)


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "auto"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8


def _is_muon_param(name: str, p: torch.nn.Parameter) -> bool:
    if p.dim() != 2:
        return False
    lname = name.lower()
    if "embedding" in lname or "lm_head" in lname:
        return False
    return True


def _trainable_named_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def _param_count(params: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(p.numel() for p in params))


def _adamw_state_bytes(param_count: int, *, bytes_per_state_value: int = 4) -> int:
    return int(param_count * 2 * bytes_per_state_value)


def candidate_report(model: torch.nn.Module, *, config: OptimizerConfig | None = None) -> dict[str, object]:
    cfg = config or OptimizerConfig()
    named_params = _trainable_named_parameters(model)
    trainable_params = _param_count(p for _, p in named_params)
    muon_params = _param_count(p for name, p in named_params if _is_muon_param(name, p))
    adamw_side_params = trainable_params - muon_params

    candidates = [
        {
            "name": "adamw",
            "status": "available",
            "implementation": "torch.optim.AdamW",
            "trainable_params": trainable_params,
            "optimizer_state_bytes_estimate": _adamw_state_bytes(trainable_params),
            "notes": "Baseline optimizer; stable default for quality comparisons.",
        },
        {
            "name": "adam8bit",
            "status": "optional_dependency",
            "implementation": "bitsandbytes.optim.AdamW8bit if installed",
            "trainable_params": trainable_params,
            "optimizer_state_bytes_estimate": trainable_params * 2,
            "notes": "8-bit optimizer-state baseline for constrained adaptation.",
        },
        {
            "name": "adafactor",
            "status": "optional_dependency",
            "implementation": "transformers.Adafactor if installed",
            "trainable_params": trainable_params,
            "optimizer_state_bytes_estimate": 0,
            "notes": "Factored-state fallback; exact state size depends on matrix shapes.",
        },
        {
            "name": "muon",
            "status": "optional_dependency",
            "implementation": "muon.Muon if installed; AdamW fallback otherwise",
            "muon_params": muon_params,
            "adamw_side_params": adamw_side_params,
            "optimizer_state_bytes_estimate": _adamw_state_bytes(adamw_side_params),
            "notes": "Applies Muon only to 2D non-embedding parameters.",
        },
        {
            "name": "scale",
            "status": "watchlist_unavailable",
            "implementation": "SCALE optimizer package not vendored",
            "trainable_params": trainable_params,
            "optimizer_state_bytes_estimate": _adamw_state_bytes(trainable_params),
            "notes": "Report-only candidate until a verified implementation is added.",
        },
        {
            "name": "galore",
            "status": "optional_dependency",
            "implementation": "galore_torch.GaLoreAdamW if installed",
            "trainable_params": trainable_params,
            "optimizer_state_bytes_estimate": 0,
            "notes": "Low-rank optimizer-state experiment; measure actual resident gradients.",
        },
        {
            "name": "qgalore",
            "status": "optional_dependency",
            "implementation": "Q-GaLore package adapter when installed",
            "trainable_params": trainable_params,
            "optimizer_state_bytes_estimate": 0,
            "notes": "Quantized low-rank optimizer experiment; never reported active on fallback.",
        },
    ]
    return {
        "schema_version": OPTIMIZER_REPORT_SCHEMA_VERSION,
        "generated_at": time.time(),
        "config": {
            **asdict(cfg),
            "betas": list(cfg.betas),
        },
        "trainable_params": trainable_params,
        "candidates": candidates,
    }


def _build_adamw(params: Iterable[torch.nn.Parameter], cfg: OptimizerConfig) -> AdamW:
    return AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, eps=cfg.eps)


def build_optimizer_with_report(
    model: torch.nn.Module,
    *,
    optimizer_name: str = "auto",
    lr: float = 3e-4,
    weight_decay: float = 0.01,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    requested = optimizer_name.strip().lower()
    if requested not in {"auto", *OPTIMIZER_CANDIDATES}:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Expected one of auto, {', '.join(OPTIMIZER_CANDIDATES)}")

    cfg = OptimizerConfig(name=requested, lr=lr, weight_decay=weight_decay)
    report = candidate_report(model, config=cfg)
    named_params = _trainable_named_parameters(model)
    all_params = [p for _, p in named_params]

    selected = requested
    actual = requested
    status = "active"
    reason = ""

    if requested == "adamw":
        optimizer = _build_adamw(all_params, cfg)
    elif requested == "adam8bit":
        try:
            from bitsandbytes.optim import AdamW8bit  # type: ignore

            optimizer = AdamW8bit(
                all_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=cfg.betas,
                eps=cfg.eps,
            )
            actual = "adam8bit"
            reason = "bitsandbytes AdamW8bit active"
        except Exception as exc:
            optimizer = _build_adamw(all_params, cfg)
            actual = "adamw"
            status = "fallback"
            reason = f"AdamW8bit unavailable ({exc}); using AdamW fallback"
    elif requested == "adafactor":
        try:
            from transformers import Adafactor  # type: ignore

            optimizer = Adafactor(
                all_params,
                lr=lr,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                weight_decay=weight_decay,
            )
            actual = "adafactor"
            reason = "transformers Adafactor active"
        except Exception as exc:
            optimizer = _build_adamw(all_params, cfg)
            actual = "adamw"
            status = "fallback"
            reason = f"Adafactor unavailable ({exc}); using AdamW fallback"
    elif requested in {"auto", "muon"}:
        muon_params = [p for name, p in named_params if _is_muon_param(name, p)]
        adamw_params = [p for name, p in named_params if not _is_muon_param(name, p)]
        try:
            from muon import Muon  # type: ignore

            groups = []
            if muon_params:
                groups.append({"params": muon_params, "use_muon": True})
            if adamw_params:
                groups.append({"params": adamw_params, "use_muon": False, "weight_decay": weight_decay})
            optimizer = Muon(groups, lr=lr, adamw_betas=cfg.betas, adamw_eps=cfg.eps)
            actual = "muon"
            selected = "muon" if requested == "auto" else requested
            reason = (
                f"Muon active; muon_params={_param_count(muon_params):,}, "
                f"adamw_side_params={_param_count(adamw_params):,}"
            )
        except Exception as exc:
            optimizer = _build_adamw(all_params, cfg)
            actual = "adamw"
            status = "fallback"
            reason = f"Muon unavailable ({exc}); using AdamW fallback"
    elif requested == "galore":
        matrix_params = [
            p for name, p in named_params if p.ndim == 2 and "esv_module" not in name
        ]
        matrix_param_ids = {id(p) for p in matrix_params}
        regular_params = [p for _, p in named_params if id(p) not in matrix_param_ids]
        try:
            from galore_torch import GaLoreAdamW  # type: ignore

            groups = [
                {
                    "params": matrix_params,
                    "rank": 64,
                    "update_proj_gap": 200,
                    "scale": 0.25,
                    "proj_type": "std",
                },
                {"params": regular_params},
            ]
            optimizer = GaLoreAdamW(
                groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=cfg.betas,
            )
            actual = "galore"
            reason = "GaLoreAdamW active with identity modules exempted from projection"
        except Exception as exc:
            optimizer = _build_adamw(all_params, cfg)
            actual = "adamw"
            status = "fallback"
            reason = f"GaLore unavailable ({exc}); using AdamW fallback"
    elif requested in {"scale", "qgalore"}:
        optimizer = _build_adamw(all_params, cfg)
        actual = "adamw"
        status = "fallback"
        reason = f"{requested.upper()} implementation is not installed; using AdamW fallback"
    else:  # pragma: no cover - guarded above.
        optimizer = _build_adamw(all_params, cfg)

    selected_report = {
        "requested": requested,
        "selected": selected,
        "actual": actual,
        "status": status,
        "reason": reason or f"{actual} selected",
        "lr": lr,
        "weight_decay": weight_decay,
        "param_groups": len(optimizer.param_groups),
    }
    report["selected"] = selected_report
    setattr(optimizer, "_anra_optimizer_report", report)
    print(f"[anra_optimizer] requested={requested} actual={actual} status={status} {selected_report['reason']}")
    return optimizer, report


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    optimizer_name: str = "auto",
):
    optimizer, _report = build_optimizer_with_report(
        model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer
