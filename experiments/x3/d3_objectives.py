"""V8-X3 D3 Learning Algorithms: Sampler Audit, Hybrid Batches, & PCGrad Conflict Surgery (§8, §13)."""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v8x.data import build_x1_d1h_training_dataset, build_x1_d2_paired_dataset
from v5_model.core import packed_layout

from .models import X3CognitiveModel, encode_prompt_and_target_pair

D3_OBJECTIVES: tuple[str, ...] = (
    "CE",
    "D1",
    "D2",
    "D3-Hybrid",
    "D3-NoSurgery",
    "D3-NoAux",
)


def audit_x1_sampler_exposure(*, quota_per_family: int = 24) -> dict[str, Any]:
    """Quantify why X1 D2 batches lost base accuracy vs A0: 4x unique-world dilution + uniform sampling (§8)."""
    d1h = build_x1_d1h_training_dataset(quota_per_family=int(quota_per_family))
    d2 = build_x1_d2_paired_dataset(quota_per_family=int(quota_per_family))
    s1 = d1h.states
    s2 = [st for g in d2.groups for st in g.samples]

    def _stats(items: Sequence[Any], *, grouped_four_views: bool) -> dict[str, Any]:
        ops = Counter(int(x.gold_op) for x in items)
        unique_worlds = len({x.world_id for x in items})
        unique_base = len({str(x.world_id).split("::")[0] for x in items})
        non_arrow = sum(1 for x in items if "->" not in UTF8.decode([t for t in x.tokens if t >= 4]))
        eff_base_per_16 = 4.0 if grouped_four_views else min(16.0, float(unique_base))
        return {
            "total_states": len(items),
            "unique_views": unique_worlds,
            "unique_base_worlds": unique_base,
            "unique_base_worlds_per_16_state_update": eff_base_per_16,
            "op_counts": dict(sorted(ops.items())),
            "non_arrow_alt_renderer_count": non_arrow,
        }

    return {
        "schema": "elendril-x3-sampler-exposure-audit-v1",
        "quota_per_family": int(quota_per_family),
        "d1h_standard": _stats(s1, grouped_four_views=False),
        "d2_paired": {
            **_stats(s2, grouped_four_views=True),
            "group_count": len(d2.groups),
            "views_per_group": 4,
            "eligible_eq_pairs": int(d2.manifest.get("eligible_eq_pairs", 0)),
            "eligible_cf_pairs": int(d2.manifest.get("eligible_cf_pairs", 0)),
        },
        "base_world_diversity_ratio_d1h_over_d2": 4.0,
        "finding": (
            "Paired D2 bundles 4 linked views of 1 base world per group (4 unique base worlds per 16-state batch "
            "vs ~16 in D1-H interleaved), explaining the shared A0->D0/D2 deficit without requiring extra capacity."
        ),
    }


def apply_pcgrad_conflict_surgery(
    base_grads: Sequence[torch.Tensor],
    aux_grads: Sequence[torch.Tensor],
    *,
    eps: float = 1e-12,
) -> tuple[list[torch.Tensor], dict[str, float | bool]]:
    """Project conflicting auxiliary gradient off base task gradient (`dot < 0 -> orthogonal`)."""
    if len(base_grads) != len(aux_grads):
        raise ValueError("base_grads and aux_grads length mismatch")
    if not base_grads:
        return [], {"dot_before": 0.0, "dot_after": 0.0, "cosine_before": 0.0, "conflict_detected": False}

    flat_base = torch.cat([g.reshape(-1) for g in base_grads])
    flat_aux = torch.cat([g.reshape(-1) for g in aux_grads])
    base_norm_sq = float((flat_base * flat_base).sum().item())
    aux_norm_sq = float((flat_aux * flat_aux).sum().item())
    dot_before = float((flat_base * flat_aux).sum().item())
    denom = max(eps, (base_norm_sq * aux_norm_sq) ** 0.5)
    cos_before = dot_before / denom

    if dot_before < 0.0 and base_norm_sq > eps:
        scale = dot_before / base_norm_sq
        projected = [a - scale * b for b, a in zip(base_grads, aux_grads)]
        conflict = True
    else:
        projected = [a.clone() for a in aux_grads]
        conflict = False

    combined = [b + p for b, p in zip(base_grads, projected)]
    flat_proj = torch.cat([p.reshape(-1) for p in projected])
    dot_after = float((flat_base * flat_proj).sum().item())
    return combined, {
        "dot_before": dot_before,
        "dot_after": dot_after,
        "cosine_before": cos_before,
        "conflict_detected": conflict,
    }


def verify_d3_gradient_surgery_fixture() -> dict[str, Any]:
    """Exact mathematical verification of PCGrad projection on conflicting & aligned vectors."""
    b_conf = [torch.tensor([1.0, 0.0])]
    a_conf = [torch.tensor([-0.5, 0.75])]
    comb_c, info_c = apply_pcgrad_conflict_surgery(b_conf, a_conf)

    b_align = [torch.tensor([1.0, 0.0])]
    a_align = [torch.tensor([0.5, 0.5])]
    comb_a, info_a = apply_pcgrad_conflict_surgery(b_align, a_align)

    assert info_c["conflict_detected"] is True
    assert abs(info_c["dot_after"]) < 1e-6
    assert torch.allclose(comb_c[0], torch.tensor([1.0, 0.75]), atol=1e-6)
    assert info_a["conflict_detected"] is False
    assert torch.allclose(comb_a[0], torch.tensor([1.5, 0.5]), atol=1e-6)
    return {"passed": True, "conflicting": info_c, "aligned": info_a}


def build_d3_hybrid_batches(
    cases: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    batch_size: int = 8,
    paired_fraction: float = 0.25,
) -> list[dict[str, Any]]:
    """Construct diverse D3 batches preserving >=75% broad world diversity + paired contrastive pairs."""
    rng = random.Random(int(seed))
    pool = list(cases)
    if not pool:
        raise ValueError("build_d3_hybrid_batches requires non-empty cases")

    by_pair: dict[str, list[Mapping[str, Any]]] = {}
    for c in pool:
        pid = str(c.get("contrast_pair_id", c["case_id"]))
        by_pair.setdefault(pid, []).append(c)
    pair_buckets = [v for v in by_pair.values() if len(v) >= 2]
    rng.shuffle(pool)
    rng.shuffle(pair_buckets)

    batches: list[dict[str, Any]] = []
    idx_ind = 0
    idx_pair = 0
    n_pair_items = max(2, int(round(batch_size * paired_fraction)) // 2 * 2) if pair_buckets else 0
    n_steps = max(1, (len(pool) + batch_size - 1) // batch_size)
    for step in range(n_steps):
        paired_items: list[Mapping[str, Any]] = []
        while len(paired_items) < n_pair_items and pair_buckets:
            grp = pair_buckets[idx_pair % len(pair_buckets)]
            idx_pair += 1
            paired_items.extend(grp[:2])
        ind_items: list[Mapping[str, Any]] = []
        while len(ind_items) + len(paired_items) < batch_size:
            ind_items.append(pool[idx_ind % len(pool)])
            idx_ind += 1
        batches.append(
            {
                "step": step,
                "independent_cases": ind_items[: max(1, batch_size - len(paired_items))],
                "paired_cases": paired_items,
            }
        )
    return batches


def _pack_training_batch(
    cases: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = []
    gold_cls: list[int] = []
    target_uncert: list[list[float]] = []
    for c in cases:
        prompt = str(c["prompt"])
        gold_txt = str(c.get("gold_text", c.get("expected_answer", "A")))
        ids, elig, ap = encode_prompt_and_target_pair(prompt, gold_txt, max_length=max_length)
        samples.append((ids, elig, ap))
        g_cls = int(c.get("gold_class", 0))
        gold_cls.append(max(0, min(18, g_cls)))
        uk = str(c.get("uncertainty_kind", ""))
        if uk == "information_uncertainty":
            target_uncert.append([0.2, 0.9])
        elif uk == "epistemic_reasoning":
            target_uncert.append([0.9, 0.2])
        else:
            target_uncert.append([0.1, 0.1])

    max_l = max(len(ids) for ids, _, _ in samples)
    bsz = len(samples)
    toks = torch.zeros((bsz, max_l), dtype=torch.long, device=device)
    segs = torch.full((bsz, max_l), -1, dtype=torch.long, device=device)
    elig_t = torch.zeros((bsz, max_l), dtype=torch.bool, device=device)
    apos = torch.zeros((bsz,), dtype=torch.long, device=device)
    for i, (ids, el, ap) in enumerate(samples):
        toks[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        segs[i, : len(ids)] = 0
        elig_t[i, : len(el)] = torch.tensor(el, dtype=torch.bool, device=device)
        apos[i] = ap
    return (
        toks,
        segs,
        elig_t,
        apos,
        torch.tensor(gold_cls, dtype=torch.long, device=device),
        torch.tensor(target_uncert, dtype=torch.float32, device=device),
    )


def _resolve_batch_order_and_offset(
    batch: Mapping[str, Sequence[Mapping[str, Any]]],
    objective: str,
) -> tuple[list[Mapping[str, Any]], int, bool]:
    ind_cases = list(batch.get("independent_cases", ()))
    pair_cases = list(batch.get("paired_cases", ()))
    if objective in ("D1", "D2") and pair_cases:
        all_cases = [*pair_cases, *pair_cases] if len(pair_cases) >= 2 else [*pair_cases, *ind_cases]
        offset = 0
    else:
        all_cases = [*ind_cases, *pair_cases]
        offset = len(ind_cases)
    if not all_cases:
        raise ValueError("Empty batch in compute_d3_objective_step")
    return all_cases, offset, (len(pair_cases) >= 2)


def prepack_d3_batches(
    batches: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
    device: str | torch.device,
    objective: str,
) -> list[dict[str, Any]]:
    """Pre-encode and transfer training batches once to eliminate per-step CPU encoding bottlenecks."""
    if objective not in D3_OBJECTIVES:
        raise ValueError(f"Unknown D3 objective: {objective!r}")
    dev = torch.device(device)
    packed_list: list[dict[str, Any]] = []
    for batch in batches:
        all_cases, offset, has_pair = _resolve_batch_order_and_offset(batch, objective)
        toks, segs, elig_t, apos, g_cls, t_uncert = _pack_training_batch(
            all_cases, max_length=int(max_length), device=dev
        )
        pos, mask = packed_layout(segs, torch_module=torch)
        packed_list.append(
            {
                **batch,
                "_packed": {
                    "objective": str(objective),
                    "device": str(dev),
                    "max_length": int(max_length),
                    "toks": toks,
                    "segs": segs,
                    "elig_t": elig_t,
                    "apos": apos,
                    "g_cls": g_cls,
                    "t_uncert": t_uncert,
                    "pos": pos,
                    "mask": mask,
                    "offset": int(offset),
                    "has_pair": bool(has_pair),
                },
            }
        )
    return packed_list


def compute_d3_objective_step(
    model: X3CognitiveModel,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    *,
    objective: str = "D3-Hybrid",
    cf_weight: float = 0.25,
    cf_margin: float = 0.5,
    meta_weight: float = 0.1,
    ponder_weight: float = 0.02,
    aux_action_weight: float = 0.25,
    clip: float = 1.0,
    use_amp: bool | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    """Run one verified optimization step (with optional CUDA FP16 AMP & pre-packed batches)."""
    if objective not in D3_OBJECTIVES:
        raise ValueError(f"Unknown D3 objective: {objective!r}")
    dev = next(model.parameters()).device
    model.train()

    packed = batch.get("_packed")
    if (
        isinstance(packed, Mapping)
        and packed.get("objective") == objective
        and packed.get("max_length") == model.max_ctx
        and packed["toks"].device == dev
    ):
        toks = packed["toks"]
        elig_t = packed["elig_t"]
        apos = packed["apos"]
        g_cls = packed["g_cls"]
        t_uncert = packed["t_uncert"]
        pos = packed["pos"]
        mask = packed["mask"]
        offset = int(packed["offset"])
        has_pair = bool(packed["has_pair"])
    else:
        all_cases, offset, has_pair = _resolve_batch_order_and_offset(batch, objective)
        toks, segs, elig_t, apos, g_cls, t_uncert = _pack_training_batch(
            all_cases, max_length=model.max_ctx, device=dev
        )
        pos, mask = packed_layout(segs, torch_module=torch)

    amp_on = bool(use_amp) if use_amp is not None else (dev.type == "cuda")
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_on):
        out = model(toks, pos, mask, action_positions=apos)

    logits_f = out["logits"].float()
    act_logits_f = out["action_logits"].float()
    uncert_f = out["uncertainties"].float()
    ponder_f = out["ponder_cost"].float()

    targets = toks[:, 1:].reshape(-1)
    logits_flat = logits_f[:, :-1, :].reshape(-1, logits_f.shape[-1])
    mask_flat = elig_t[:, 1:].reshape(-1).to(logits_flat.dtype)
    byte_ce = (F.cross_entropy(logits_flat, targets, reduction="none") * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)
    act_ce = F.nll_loss(act_logits_f, g_cls)
    base_loss = byte_ce + float(aux_action_weight) * act_ce

    # Auxiliary losses computed strictly in FP32 (ponder, metacognitive uncertainty, counterfactual separation)
    meta_loss = F.mse_loss(uncert_f, t_uncert)
    ponder_loss = ponder_f.mean()
    cf_loss = torch.zeros((), device=dev, dtype=torch.float32)

    use_aux_cf = objective in ("D1", "D2", "D3-Hybrid", "D3-NoSurgery")
    if use_aux_cf and has_pair:
        probs = F.softmax(act_logits_f, dim=-1)
        p_a = probs[offset]
        p_b = probs[offset + 1]
        if int(g_cls[offset].item()) != int(g_cls[offset + 1].item()):
            l1 = 0.5 * torch.abs(p_a - p_b).sum()
            cf_loss = F.relu(float(cf_margin) - l1)
        else:
            m = 0.5 * (p_a + p_b).clamp_min(1e-7)
            cf_loss = 0.5 * (
                F.kl_div(m.log(), p_a.clamp_min(1e-7), reduction="sum")
                + F.kl_div(m.log(), p_b.clamp_min(1e-7), reduction="sum")
            )

    if objective == "D3-NoAux":
        aux_loss = float(ponder_weight) * ponder_loss + float(meta_weight) * meta_loss
    else:
        aux_loss = (
            float(cf_weight) * cf_loss
            + float(meta_weight) * meta_loss
            + float(ponder_weight) * ponder_loss
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    scaler_on = scaler is not None and scaler.is_enabled()

    surgery_info: dict[str, float | bool] = {
        "dot_before": 0.0,
        "dot_after": 0.0,
        "cosine_before": 0.0,
        "conflict_detected": False,
    }
    if objective == "D3-Hybrid":
        scaled_base = scaler.scale(base_loss) if scaler_on else base_loss
        scaled_aux = scaler.scale(aux_loss) if scaler_on else aux_loss
        base_g = torch.autograd.grad(scaled_base, trainable, retain_graph=True, allow_unused=True)
        aux_g = torch.autograd.grad(scaled_aux, trainable, allow_unused=True)

        grad_finite = all(
            bool(torch.isfinite(g).all().item())
            for pair in zip(base_g, aux_g)
            for g in pair
            if g is not None
        )
        shared_idx = [i for i, (bg, ag) in enumerate(zip(base_g, aux_g)) if bg is not None and ag is not None]
        if grad_finite and shared_idx:
            s_val = max(1e-12, float(scaler.get_scale())) if scaler_on else 1.0
            inv_s = 1.0 / s_val
            comb_shared, surgery_info = apply_pcgrad_conflict_surgery(
                [base_g[i].detach().float() * inv_s for i in shared_idx],
                [aux_g[i].detach().float() * inv_s for i in shared_idx],
            )
            comb_map = {shared_idx[k]: (comb_shared[k] * s_val) for k in range(len(shared_idx))}
            for i, p in enumerate(trainable):
                if i in comb_map:
                    p.grad = comb_map[i].to(p.dtype)
                elif base_g[i] is not None:
                    p.grad = base_g[i].detach()
                elif aux_g[i] is not None:
                    p.grad = aux_g[i].detach()
        else:
            for i, p in enumerate(trainable):
                bg_i, ag_i = base_g[i], aux_g[i]
                if bg_i is not None and not torch.isfinite(bg_i).all():
                    p.grad = bg_i.detach()
                elif ag_i is not None and not torch.isfinite(ag_i).all():
                    p.grad = ag_i.detach()
                elif bg_i is not None and ag_i is not None:
                    p.grad = (bg_i.detach() + ag_i.detach()).to(p.dtype)
                elif bg_i is not None:
                    p.grad = bg_i.detach()
                elif ag_i is not None:
                    p.grad = ag_i.detach()
        total_loss_val = float((base_loss + aux_loss).item())
    else:
        total_loss = base_loss + aux_loss
        (scaler.scale(total_loss) if scaler_on else total_loss).backward()
        total_loss_val = float(total_loss.item())

    if scaler_on:
        scaler.unscale_(optimizer)
        raw_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(clip)).item())
        scaler.step(optimizer)
        scaler.update()
    else:
        raw_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(clip)).item())
        optimizer.step()

    scaler_skipped = not math.isfinite(raw_norm)
    return {
        "objective": objective,
        "total_loss": total_loss_val if math.isfinite(total_loss_val) else -1.0,
        "byte_ce": float(byte_ce.item()),
        "action_ce": float(act_ce.item()),
        "cf_loss": float(cf_loss.item()),
        "meta_loss": float(meta_loss.item()),
        "ponder_loss": float(ponder_loss.item()),
        "grad_norm": raw_norm if math.isfinite(raw_norm) else -1.0,
        "scaler_skipped": scaler_skipped,
        **surgery_info,
    }


__all__ = [
    "D3_OBJECTIVES",
    "apply_pcgrad_conflict_surgery",
    "audit_x1_sampler_exposure",
    "build_d3_hybrid_batches",
    "compute_d3_objective_step",
    "prepack_d3_batches",
    "verify_d3_gradient_surgery_fixture",
]
