from __future__ import annotations

import torch
from torch.optim import AdamW


def _is_muon_param(name: str, p: torch.nn.Parameter) -> bool:
    if p.dim() != 2:
        return False
    lname = name.lower()
    if "embedding" in lname or "lm_head" in lname:
        return False
    return True


def build_optimizer(model: torch.nn.Module, lr: float = 3e-4, weight_decay: float = 0.01):
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_muon_param(name, p):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    try:
        from muon import Muon  # type: ignore

        groups = []
        if muon_params:
            groups.append({"params": muon_params, "use_muon": True})
        if adamw_params:
            groups.append({"params": adamw_params, "use_muon": False, "weight_decay": weight_decay})
        optimizer = Muon(groups, lr=lr, adamw_betas=(0.9, 0.95), adamw_eps=1e-8)
        print(f"[anra_optimizer] Muon active | muon_params={sum(p.numel() for p in muon_params):,} "
              f"adamw_params={sum(p.numel() for p in adamw_params):,}")
        return optimizer
    except Exception as e:
        print(f"[anra_optimizer] Muon unavailable ({e}); using AdamW fallback")
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
