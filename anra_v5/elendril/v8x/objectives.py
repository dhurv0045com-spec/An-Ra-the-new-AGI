"""Grouped D2 (CE / PAIR-CE / PAIR-EQ / PAIR-EQ-CF) and standard X1 objective math."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from anra_v5.elendril.training import batch_tensors
from anra_v5.planthir.adapters.x1_rd2_adapter import compute_x1_d2_grouped_loss

from .data import X1D2PairedGroup, X1SupervisedState


def compute_x1_objective_batch(
    model: Any,
    *,
    objective: str,
    states: Sequence[X1SupervisedState] | None = None,
    groups: Sequence[X1D2PairedGroup] | None = None,
    device: torch.device,
    eq_weight: float = 0.1,
    cf_weight: float = 0.1,
    cf_margin: float = 0.5,
    ce_denom_override: float | None = None,
    eq_denom_override: float | None = None,
    cf_denom_override: float | None = None,
    backward: bool = False,
) -> dict[str, Any]:
    """Evaluate (and optionally backprop) exact X1 standard or grouped D2 loss with global denominators."""

    if objective == "standard":
        if not states:
            raise ValueError("standard objective requires non-empty states")
        tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in states], device)
        apos = torch.tensor([s.action_position for s in states], device=device, dtype=torch.int64)
        gold_ops = torch.tensor([s.gold_op for s in states], device=device, dtype=torch.int64)
        gold_args = torch.tensor([s.gold_arg for s in states], device=device, dtype=torch.int64)
        out = model.forward_action(tokens, segments, apos)
        op_nll = -out.log_p_op.float().gather(1, gold_ops.unsqueeze(1)).squeeze(1)
        is_ans = (gold_ops == 0).to(op_nll.dtype)
        val_nll = -out.log_p_val.float().gather(1, gold_args.clamp(0, 15).unsqueeze(1)).squeeze(1) * is_ans
        nll_sum = (op_nll + val_nll).sum()
        denom = float(ce_denom_override if ce_denom_override is not None else len(states))
        loss = nll_sum / denom
        if backward:
            loss.backward()
        return {
            "loss_value": float(nll_sum.detach().item() / denom),
            "ce_sum": float(nll_sum.detach().item()),
            "eq_sum": 0.0,
            "cf_sum": 0.0,
            "active_states": len(states),
            "eligible_eq_pairs": 0,
            "eligible_cf_pairs": 0,
            "has_answer_targets": any(s.gold_op == 0 for s in states),
        }

    if not groups:
        raise ValueError(f"{objective} requires non-empty D2 groups")

    eff_eq_w = float(eq_weight) if objective in ("PAIR-EQ", "PAIR-EQ-CF") else 0.0
    eff_cf_w = float(cf_weight) if objective == "PAIR-EQ-CF" else 0.0

    # Flatten supplied micro-chunk of D2 groups (e.g. 1 group = 4 views, or all groups)
    flat_samples = []
    eq_a, eq_b, eq_perm, eq_mask = [], [], [], []
    cf_a, cf_b, cf_ta, cf_tb, cf_mask = [], [], [], [], []
    for grp in groups:
        base = len(flat_samples)
        flat_samples.extend(grp.samples)
        for ep in grp.eq_pairs:
            eq_a.append(base + ep.idx_a)
            eq_b.append(base + ep.idx_b)
            eq_perm.append(list(ep.perm_16))
            eq_mask.append(bool(ep.eligible))
        for cp in grp.cf_pairs:
            cf_a.append(base + cp.idx_a)
            cf_b.append(base + cp.idx_b)
            cf_ta.append(int(cp.target_a))
            cf_tb.append(int(cp.target_b))
            cf_mask.append(bool(cp.eligible))

    tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in flat_samples], device)
    apos = torch.tensor([s.action_position for s in flat_samples], device=device, dtype=torch.int64)
    gold_ops = torch.tensor([s.gold_op for s in flat_samples], device=device, dtype=torch.int64)
    gold_args = torch.tensor([s.gold_arg for s in flat_samples], device=device, dtype=torch.int64)
    state_mask = torch.tensor([bool(s.state_mask) for s in flat_samples], device=device, dtype=torch.bool)

    eq_a_t = torch.tensor(eq_a, device=device, dtype=torch.int64)
    eq_b_t = torch.tensor(eq_b, device=device, dtype=torch.int64)
    eq_perm_t = torch.tensor(eq_perm, device=device, dtype=torch.int64) if eq_perm else torch.zeros((0, 16), device=device, dtype=torch.int64)
    eq_mask_t = torch.tensor(eq_mask, device=device, dtype=torch.bool)
    cf_a_t = torch.tensor(cf_a, device=device, dtype=torch.int64)
    cf_b_t = torch.tensor(cf_b, device=device, dtype=torch.int64)
    cf_ta_t = torch.tensor(cf_ta, device=device, dtype=torch.int64)
    cf_tb_t = torch.tensor(cf_tb, device=device, dtype=torch.int64)
    cf_mask_t = torch.tensor(cf_mask, device=device, dtype=torch.bool)

    out = model.forward_action(tokens, segments, apos)
    local_loss, metrics = compute_x1_d2_grouped_loss(
        out.log_probs_19,
        out.log_p_op,
        out.log_p_val,
        gold_ops=gold_ops,
        gold_args=gold_args,
        state_mask=state_mask,
        eq_a=eq_a_t,
        eq_b=eq_b_t,
        eq_perm_16=eq_perm_t,
        eq_mask=eq_mask_t,
        cf_a=cf_a_t,
        cf_b=cf_b_t,
        cf_target_a=cf_ta_t,
        cf_target_b=cf_tb_t,
        cf_mask=cf_mask_t,
        eq_weight=eff_eq_w,
        cf_weight=eff_cf_w,
        cf_margin=cf_margin,
    )

    # Scale each term cleanly by (local_active / global_macro_active) so microbatching by 1 group
    # matches exact full-batch gradients over all 4 groups (16 states)!
    anchor_zero = out.log_probs_19.sum() * 0.0
    local_sm = max(int(metrics["active_states"]), 0)
    local_eq = max(int(metrics["eligible_eq_pairs"]), 0)
    local_cf = max(int(metrics["eligible_cf_pairs"]), 0)

    ce_g = float(ce_denom_override if ce_denom_override is not None else local_sm)
    eq_g = float(eq_denom_override if eq_denom_override is not None else local_eq)
    cf_g = float(cf_denom_override if cf_denom_override is not None else local_cf)

    # Recompute exact scaled loss components directly when global denominators differ from local chunk
    if ce_denom_override is None and eq_denom_override is None and cf_denom_override is None:
        scaled_loss = local_loss
    else:
        sm = state_mask.to(dtype=torch.float32)
        op_nll = -out.log_p_op.float().gather(1, gold_ops.unsqueeze(1)).squeeze(1)
        is_ans = (gold_ops == 0).to(dtype=torch.float32)
        val_nll = -out.log_p_val.float().gather(1, gold_args.clamp(0, 15).unsqueeze(1)).squeeze(1) * is_ans
        l_ce = ((op_nll + val_nll) * sm).sum() / max(ce_g, 1.0) if local_sm > 0 else anchor_zero

        if local_eq > 0 and eff_eq_w > 0.0 and eq_g > 0.0:
            p19 = out.log_probs_19.float().exp().clamp_min(1e-12)
            pa, pb = p19[eq_a_t], p19[eq_b_t]
            pa_ans = torch.zeros_like(pa[:, :16])
            pa_ans.scatter_add_(1, eq_perm_t, pa[:, :16])
            pa_perm = torch.cat([pa_ans, pa[:, 16:]], dim=-1).clamp_min(1e-12)
            pa_perm = pa_perm / pa_perm.sum(dim=-1, keepdim=True)
            pb_norm = pb / pb.sum(dim=-1, keepdim=True)
            mid = 0.5 * (pa_perm + pb_norm)
            js = 0.5 * (
                (pa_perm * (pa_perm.log() - mid.log())).sum(dim=-1)
                + (pb_norm * (pb_norm.log() - mid.log())).sum(dim=-1)
            )
            l_eq = (js * eq_mask_t.to(dtype=torch.float32)).sum() / eq_g
        else:
            l_eq = anchor_zero

        if local_cf > 0 and eff_cf_w > 0.0 and cf_g > 0.0:
            lp_val = out.log_p_val.float()
            ma = lp_val[cf_a_t, cf_ta_t] - lp_val[cf_a_t, cf_tb_t]
            mb = lp_val[cf_b_t, cf_tb_t] - lp_val[cf_b_t, cf_ta_t]
            margin = 0.5 * (torch.relu(cf_margin - ma) + torch.relu(cf_margin - mb))
            l_cf = (margin * cf_mask_t.to(dtype=torch.float32)).sum() / cf_g
        else:
            l_cf = anchor_zero

        scaled_loss = l_ce + eff_eq_w * l_eq + eff_cf_w * l_cf

    if backward:
        scaled_loss.backward()

    return {
        "loss_value": float(scaled_loss.detach().item()),
        "ce_weighted": float(metrics["ce"] * (local_sm / max(ce_g, 1.0)) if local_sm else 0.0),
        "eq_weighted": float(metrics["eq"] * (local_eq / max(eq_g, 1.0)) if local_eq else 0.0),
        "cf_weighted": float(metrics["cf"] * (local_cf / max(cf_g, 1.0)) if local_cf else 0.0),
        "active_states": local_sm,
        "eligible_eq_pairs": local_eq,
        "eligible_cf_pairs": local_cf,
        "has_answer_targets": any(s.gold_op == 0 for s in flat_samples if s.state_mask),
    }


__all__ = ["compute_x1_objective_batch"]
