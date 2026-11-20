"""X3 checkpoint suite, Geiger pointer readout interchange interventions, and stratified scoring (§4)."""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.v8x.models import X1TypedModel
from experiments.x2.models import AP_MAX_TOKEN_ID, AP_MIN_TOKEN_ID, X2TypedModel
from experiments.x2.tasks import ACTION_19_LABELS, encode_x2_text


def _encode_texts(texts: Sequence[str], device: torch.device, max_ctx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [encode_x2_text(t, max_length=max_ctx, allow_truncate=False) for t in texts]
    max_len = max(len(s) for s, _ in encoded)
    toks = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
    segs = torch.full((len(encoded), max_len), -1, dtype=torch.long, device=device)
    apos = torch.zeros((len(encoded),), dtype=torch.long, device=device)
    for i, (seq, ap) in enumerate(encoded):
        toks[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        segs[i, : len(seq)] = 0
        apos[i] = ap
    return toks, segs, apos


def _find_sham_positions(text: str, seq_len: int) -> tuple[int, ...]:
    """Locate byte offsets of the 2 inactive ep=0 RHS symbol tokens (+1 for BOS) without touching active chains."""
    raw = text.encode("utf-8")
    positions: list[int] = []
    offset = 0
    for line in raw.split(b"\n"):
        if b"ep=0" in line or b"ep 0:" in line:
            # Find the arrow/to destination uppercase symbol A..P inside this line
            for rel_i in range(len(line) - 1, -1, -1):
                b = line[rel_i]
                if 65 <= b <= 80:  # 'A'..'P'
                    tok_pos = offset + rel_i + 1  # +1 for BOS token
                    if 0 <= tok_pos < seq_len:
                        positions.append(tok_pos)
                    break
        offset += len(line) + 1
        if len(positions) >= 2:
            break
    return tuple(positions)


def _model_max_ctx(model: nn.Module) -> int:
    base_m = getattr(model, "base", model)
    if hasattr(base_m, "x2_spec"):
        return int(base_m.x2_spec.base_spec.context_length)
    if hasattr(base_m, "x1_spec"):
        return int(base_m.x1_spec.max_sequence_length)
    return 640


def _model_arch(model: nn.Module) -> str:
    base_m = getattr(model, "base", model)
    if hasattr(base_m, "x2_spec"):
        return str(base_m.x2_spec.architecture)
    if hasattr(base_m, "x1_spec"):
        return str(base_m.x1_spec.architecture)
    return "unknown"


@torch.no_grad()
def _compute_x1_pointer_state(
    m: X1TypedModel,
    toks: torch.Tensor,
    segs: torch.Tensor,
    apos: torch.Tensor,
    *,
    H_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    B, L = toks.shape
    b_idx = torch.arange(B, device=toks.device, dtype=torch.int64)
    pos_idx = apos.to(dtype=torch.int64)
    H = H_override if H_override is not None else m.forward_hidden_only(toks, segs).float()
    h = H[b_idx, pos_idx]
    arch = m.x1_spec.architecture

    pos_grid = torch.arange(L, device=toks.device).unsqueeze(0)
    same_seg = (segs == segs[b_idx, pos_idx].unsqueeze(1)) & (segs >= 0)
    valid_prefix = (pos_grid <= pos_idx.unsqueeze(1)) & same_seg

    if arch == "F-cap":
        h_op, h_val = h, h + m.cap_out(F.silu(m.cap_in(h)))
    else:
        h_op, h_val = h, h

    op_logits = m.op_head(h_op)
    log_p_op = F.log_softmax(op_logits, dim=-1)
    z_val = m.val_head(h_val)

    supports_ptr = arch in ("P", "P-capR", "R", "R-noedge") and bool(m.x1_spec.copy_enabled)
    if not supports_ptr:
        return {
            "supports_pointer": False,
            "H": H,
            "h_val": h_val,
            "op_logits": op_logits,
            "log_p_op": log_p_op,
            "z_val": z_val,
            "q_ptr": None,
            "k_all": None,
            "gate": torch.zeros((B, 1), dtype=torch.float32, device=H.device),
            "cand_mask": torch.zeros((B, L), dtype=torch.bool, device=H.device),
            "toks": toks,
            "pointer_dim": 1,
        }

    q_ptr = m.q_proj(h)
    if arch == "P-capR":
        q_ptr = q_ptr + m.cap_r_out(F.silu(m.cap_r_in(h)))
    elif arch in ("R", "R-noedge"):
        s = int(m.x1_spec.slot_dim)
        k_slots = int(m.x1_spec.num_slots)
        x_tok = m.tok_proj(m.tok_ln(H))
        slots = m.slots_init.unsqueeze(0).expand(B, k_slots, s).contiguous()
        k_tok = m.slot_k(x_tok)
        v_tok = m.slot_v(x_tok)
        for _ in range(int(m.x1_spec.agg_iterations)):
            q_s = m.slot_q(slots)
            logits = torch.bmm(q_s, k_tok.transpose(1, 2)) / math.sqrt(s)
            a_ki = F.softmax(logits, dim=1) * valid_prefix.unsqueeze(1).to(logits.dtype)
            w_ki = a_ki / a_ki.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            u_k = torch.bmm(w_ki, v_tok)
            slots = m.slot_gru(u_k.reshape(B * k_slots, s), slots.reshape(B * k_slots, s)).view(B, k_slots, s)
            slots = slots + m.agg_mlp_2(F.silu(m.agg_mlp_1(m.agg_ln(slots))))
        q_goal = m.goal_q_proj(h)
        for _ in range(int(m.x1_spec.msg_iterations)):
            if arch == "R":
                s_ln = m.rel_ln(slots)
                s_l = s_ln.unsqueeze(1).expand(B, k_slots, k_slots, s)
                s_k = s_ln.unsqueeze(2).expand(B, k_slots, k_slots, s)
                q_exp = q_goal.unsqueeze(1).unsqueeze(2).expand(B, k_slots, k_slots, s)
                e_kl = m.edge_mlp_2(F.silu(m.edge_mlp_1(torch.cat([s_k, s_l, q_exp], dim=-1)))).squeeze(-1)
                if k_slots > 1:
                    eye = torch.eye(k_slots, dtype=torch.bool, device=slots.device).unsqueeze(0)
                    e_kl = e_kl.masked_fill(eye, float("-inf"))
                alpha = F.softmax(e_kl, dim=-1)
                m_k = torch.bmm(alpha, m.msg_proj(slots))
                slots = m.rel_gru(m_k.reshape(B * k_slots, s), slots.reshape(B * k_slots, s)).view(B, k_slots, s)
            else:
                up = m.noedge_mlp_2(F.silu(m.noedge_mlp_1(m.noedge_ln(slots))))
                slots = m.noedge_gru(up.reshape(B * k_slots, s), slots.reshape(B * k_slots, s)).view(B, k_slots, s)
        r_scores = torch.bmm(q_goal.unsqueeze(1), m.read_ln(slots).transpose(1, 2)).squeeze(1) / math.sqrt(s)
        r_alpha = F.softmax(r_scores, dim=-1)
        r_vec = torch.bmm(r_alpha.unsqueeze(1), slots).squeeze(1)
        q_ptr = q_ptr + m.r_to_ptr(r_vec)

    cand_mask = (pos_grid < pos_idx.unsqueeze(1)) & same_seg & (toks >= AP_MIN_TOKEN_ID) & (toks <= AP_MAX_TOKEN_ID)
    return {
        "supports_pointer": True,
        "H": H,
        "h_val": h_val,
        "op_logits": op_logits,
        "log_p_op": log_p_op,
        "z_val": z_val,
        "q_ptr": q_ptr,
        "k_all": m.k_proj(H),
        "gate": torch.sigmoid(m.gate(h)),
        "cand_mask": cand_mask,
        "toks": toks,
        "pointer_dim": int(m.x1_spec.pointer_dim),
    }


@torch.no_grad()
def _compute_x2_pointer_state(
    m: X2TypedModel,
    toks: torch.Tensor,
    segs: torch.Tensor,
    apos: torch.Tensor,
    *,
    H_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    B, L = toks.shape
    b_idx = torch.arange(B, device=toks.device, dtype=torch.int64)
    pos_idx = apos.to(dtype=torch.int64)
    H = H_override if H_override is not None else m.forward_hidden_only(toks, segs).float()
    h = H[b_idx, pos_idx]
    pos_grid = torch.arange(L, device=toks.device).unsqueeze(0)
    q_seg = segs[b_idx, pos_idx].unsqueeze(1)
    valid_prefix = (pos_grid <= pos_idx.unsqueeze(1)) & (segs == q_seg) & (segs >= 0)
    cand_mask = valid_prefix & (toks >= AP_MIN_TOKEN_ID) & (toks <= AP_MAX_TOKEN_ID)

    arch = m.x2_spec.architecture
    d = int(m.x2_spec.pointer_dim)
    if arch in ("P", "R", "R2", "R2-pool", "DM", "DM-add"):
        extra_op = torch.zeros((B, 4), dtype=torch.float32, device=H.device)
        extra_q = torch.zeros((B, d), dtype=torch.float32, device=H.device)
        extra_val = torch.zeros((B, 16), dtype=torch.float32, device=H.device)
        h_op, h_val = h, h
        if arch in ("R", "R2"):
            r_read, _ = m._compute_slots(H, valid_prefix, h)
            if arch == "R":
                extra_q = m.r_to_ptr(r_read)
            else:
                extra_op = torch.sigmoid(m.r_op_gate(h)) * m.r_to_op(r_read)
                extra_q = torch.sigmoid(m.r_val_gate(h)) * m.r_to_ptr(r_read)
        elif arch == "R2-pool":
            mask_f = valid_prefix.to(torch.float32).unsqueeze(-1)
            pooled = (m.pool_ln(H) * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            p_read = m.read_ln(m.pool_mlp_2(F.silu(m.pool_mlp_1(pooled))))
            extra_op = torch.sigmoid(m.r_op_gate(h)) * m.r_to_op(p_read)
            extra_q = torch.sigmoid(m.r_val_gate(h)) * m.r_to_ptr(p_read)
        elif arch in ("DM", "DM-add"):
            mv, _ = m._run_delta_memory(H, valid_prefix, h, additive=(arch == "DM-add"))
            mg = torch.sigmoid(m.mem_gate(h))
            extra_op = mg * m.mem_to_op(mv)
            extra_q = mg * m.mem_to_ptr(mv)
            extra_val = mg * m.mem_to_val(mv)
        op_logits = m.op_head(h_op) + extra_op
        log_p_op = F.log_softmax(op_logits, dim=-1)
        z_val = m.val_head(h_val) + extra_val
        q_ptr = m.q_proj(h_val) + extra_q
        k_all = m.k_proj(H)
        return {
            "supports_pointer": True,
            "H": H,
            "h_val": h_val,
            "op_logits": op_logits,
            "log_p_op": log_p_op,
            "z_val": z_val,
            "q_ptr": q_ptr,
            "k_all": k_all,
            "gate": torch.sigmoid(m.gate(h_val)),
            "cand_mask": cand_mask,
            "toks": toks,
            "pointer_dim": d,
        }
    # P2 / P2-dense use custom role/dense query vectors
    if arch == "P2-dense":
        q_ptr = m.dense_goal_2(F.silu(m.dense_goal_1(h)))
        k_all = m.dense_pos_2(F.silu(m.dense_pos_1(H)))
        return {
            "supports_pointer": True,
            "H": H,
            "h_val": h,
            "op_logits": m.op_head(h),
            "log_p_op": F.log_softmax(m.op_head(h), dim=-1),
            "z_val": m.val_head(h),
            "q_ptr": q_ptr,
            "k_all": k_all,
            "gate": torch.sigmoid(m.gate(h)),
            "cand_mask": cand_mask,
            "toks": toks,
            "pointer_dim": d,
        }
    out = m.forward_action(toks, segs, apos)
    return {
        "supports_pointer": False,
        "H": H,
        "h_val": h,
        "op_logits": torch.zeros((B, 4), device=H.device),
        "log_p_op": out.log_p_op,
        "z_val": torch.zeros((B, 16), device=H.device),
        "log_probs_19_direct": out.log_probs_19,
        "gate": out.gate.unsqueeze(-1) if out.gate is not None else torch.zeros((B, 1), device=H.device),
        "cand_mask": cand_mask,
        "toks": toks,
        "pointer_dim": d,
    }


def _decode_from_pointer_state(
    state: Mapping[str, Any],
    *,
    q_ptr_override: torch.Tensor | None = None,
    disable_pointer: bool = False,
    equalize_norm: bool = False,
) -> dict[str, torch.Tensor]:
    if "log_probs_19_direct" in state and q_ptr_override is None:
        lp19 = state["log_probs_19_direct"]
        return {"log_probs_19": lp19, "gate": state["gate"].squeeze(-1)}

    log_p_op = state["log_p_op"]
    z_val = state["z_val"]
    p_vocab = F.softmax(z_val, dim=-1)
    if disable_pointer or not state["supports_pointer"]:
        log_p_val = F.log_softmax(z_val, dim=-1)
        lp19 = torch.cat([log_p_op[:, :1] + log_p_val, log_p_op[:, 1:]], dim=-1)
        return {"log_probs_19": lp19, "gate": torch.zeros_like(state["gate"].squeeze(-1))}

    q_ptr = state["q_ptr"]
    if q_ptr_override is not None:
        q_new = q_ptr_override.to(device=q_ptr.device, dtype=q_ptr.dtype)
        if equalize_norm:
            orig_norm = q_ptr.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            new_norm = q_new.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            q_new = q_new * (orig_norm / new_norm)
        q_ptr = q_new

    cand_mask = state["cand_mask"]
    has_cand = cand_mask.any(dim=-1, keepdim=True)
    k_all = state["k_all"]
    d = int(state["pointer_dim"])
    scores = torch.bmm(q_ptr.unsqueeze(1), k_all.transpose(1, 2)).squeeze(1) / math.sqrt(d)
    safe_mask = cand_mask.clone()
    if (~has_cand.squeeze(-1)).any():
        safe_mask[~has_cand.squeeze(-1), 0] = True
    attn = F.softmax(scores.masked_fill(~safe_mask, -1e9), dim=-1) * cand_mask.to(torch.float32)
    bins = (state["toks"] - AP_MIN_TOKEN_ID).clamp(0, 15)
    p_copy = torch.zeros_like(p_vocab)
    p_copy.scatter_add_(1, bins, attn)
    g = state["gate"]
    p_arg = torch.where(has_cand, (1.0 - g) * p_vocab + g * p_copy, p_vocab).clamp_min(1e-12)
    p_arg = p_arg / p_arg.sum(dim=-1, keepdim=True)
    log_p_val = p_arg.log()
    lp19 = torch.cat([log_p_op[:, :1] + log_p_val, log_p_op[:, 1:]], dim=-1)
    return {"log_probs_19": lp19, "gate": g.squeeze(-1)}


def _extract_state(
    model: nn.Module,
    toks: torch.Tensor,
    segs: torch.Tensor,
    apos: torch.Tensor,
    *,
    H_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    base_m = getattr(model, "base", model)
    if isinstance(base_m, X1TypedModel):
        return _compute_x1_pointer_state(base_m, toks, segs, apos, H_override=H_override)
    if isinstance(base_m, X2TypedModel):
        return _compute_x2_pointer_state(base_m, toks, segs, apos, H_override=H_override)
    raise TypeError(f"Unsupported model type for X3 evaluation: {type(model).__name__}")


def _action_and_margin(log_probs_row: torch.Tensor, expected_symbol: str) -> dict[str, Any]:
    probs = log_probs_row.exp()
    pred_idx = int(torch.argmax(log_probs_row).item())
    pred_label = ACTION_19_LABELS[pred_idx]
    target_idx = ACTION_19_LABELS.index(expected_symbol)
    target_lp = float(log_probs_row[target_idx].item())
    other_lp = float(torch.cat([log_probs_row[:target_idx], log_probs_row[target_idx + 1 :]]).max().item())
    entropy = float(-(probs * log_probs_row.clamp_min(-30.0)).sum().item())
    return {
        "pred": pred_label,
        "correct": pred_label == expected_symbol,
        "target_margin": target_lp - other_lp,
        "top1_prob": float(probs.max().item()),
        "abstained": pred_label == "ABSTAIN",
        "entropy": entropy,
    }


@torch.no_grad()
def evaluate_x3_checkpoint_suite(
    model: nn.Module,
    model_spec: Any,
    panel: Mapping[str, Any],
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate a frozen checkpoint on X3 decisions under `intact`, `disabled_pointer`, and `sham_perturbation`."""
    _ = model_spec
    model.eval()
    dev = next(model.parameters()).device
    max_ctx = _model_max_ctx(model)
    arch = _model_arch(model)
    decisions = list(panel.get("decisions", ()))
    if not decisions:
        raise ValueError("X3 panel has zero decisions")

    cond_outputs: dict[str, list[dict[str, Any]]] = {"intact": [], "disabled_pointer": [], "sham_perturbation": []}
    pair_buckets: dict[str, dict[str, dict[str, Any]]] = {
        "intact": {},
        "disabled_pointer": {},
        "sham_perturbation": {},
    }

    for start in range(0, len(decisions), batch_size):
        chunk = decisions[start : start + batch_size]
        texts = [str(d["rendered_text"]) for d in chunk]
        toks, segs, apos = _encode_texts(texts, dev, max_ctx)
        st_intact = _extract_state(model, toks, segs, apos)
        H_sham = st_intact["H"].clone()
        for bi, txt in enumerate(texts):
            for s_pos in _find_sham_positions(txt, toks.shape[1]):
                H_sham[bi, s_pos] = 0.0
        st_sham = _extract_state(model, toks, segs, apos, H_override=H_sham)

        decoded_map = {
            "intact": _decode_from_pointer_state(st_intact, disable_pointer=False),
            "disabled_pointer": _decode_from_pointer_state(st_intact, disable_pointer=True),
            "sham_perturbation": _decode_from_pointer_state(st_sham, disable_pointer=False),
        }

        for cond_name, dec in decoded_map.items():
            lp19 = dec["log_probs_19"].cpu()
            for bi, d in enumerate(chunk):
                am = _action_and_margin(lp19[bi], str(d["expected_answer"]))
                rec = {
                    "world_id": d["world_id"],
                    "pair_key": d["pair_key"],
                    "graph_id": d["graph_id"],
                    "depth": int(d["depth"]),
                    "renderer": d["renderer"],
                    "world_variant": d["world_variant"],
                    "query_role": d["query_role"],
                    "expected": d["expected_answer"],
                    "distractor": d["distractor_value"],
                    "followed_distractor": am["pred"] == str(d["distractor_value"]),
                    **am,
                }
                cond_outputs[cond_name].append(rec)
                pk = str(d["pair_key"])
                cell_k = f"{d['world_variant']}_{d['query_role']}"
                pb = pair_buckets[cond_name].setdefault(
                    pk,
                    {
                        "pair_key": pk,
                        "graph_id": d["graph_id"],
                        "depth": int(d["depth"]),
                        "renderer": d["renderer"],
                        "cells": {},
                    },
                )
                pb["cells"][cell_k] = rec

    summary_by_cond: dict[str, dict[str, Any]] = {}
    pair_records_by_cond: dict[str, list[dict[str, Any]]] = {}
    for cond_name, buckets in pair_buckets.items():
        plist: list[dict[str, Any]] = []
        for pk in sorted(buckets):
            b = buckets[pk]
            cells = b["cells"]
            both_orig = cells["original_affected"]["correct"] and cells["edited_affected"]["correct"]
            both_sib = cells["original_sibling"]["correct"] and cells["edited_sibling"]["correct"]
            four_way = bool(both_orig and both_sib)
            err_sib_chg = cells["original_sibling"]["pred"] != cells["edited_sibling"]["pred"]
            plist.append(
                {
                    "pair_key": pk,
                    "graph_id": b["graph_id"],
                    "depth": int(b["depth"]),
                    "renderer": b["renderer"],
                    "primary_4way": four_way,
                    "affected_both_correct": bool(both_orig),
                    "sibling_both_correct": bool(both_sib),
                    "erroneous_sibling_change": bool(err_sib_chg),
                }
            )
        pair_records_by_cond[cond_name] = plist
        all_recs = cond_outputs[cond_name]
        can_p = [x for x in plist if x["renderer"] == "canonical"]
        nar_p = [x for x in plist if x["renderer"] == "narrative"]
        can_rate = sum(1 for x in can_p if x["primary_4way"]) / max(1, len(can_p))
        nar_rate = sum(1 for x in nar_p if x["primary_4way"]) / max(1, len(nar_p))
        n_pairs = len(plist)
        n_ok = sum(1 for x in plist if x["primary_4way"])
        summary_by_cond[cond_name] = {
            "primary_4way_accuracy": round(n_ok / max(1, n_pairs), 6),
            "primary_numerator": n_ok,
            "primary_denominator": n_pairs,
            "canonical_primary": round(can_rate, 6),
            "narrative_primary": round(nar_rate, 6),
            "narrative_loss": round(can_rate - nar_rate, 6),
            "erroneous_sibling_change_rate": round(sum(1 for x in plist if x["erroneous_sibling_change"]) / max(1, n_pairs), 6),
            "distractor_following_rate": round(sum(1 for r in all_recs if r["followed_distractor"]) / max(1, len(all_recs)), 6),
            "mean_target_margin": round(sum(float(r["target_margin"]) for r in all_recs) / max(1, len(all_recs)), 6),
        }

    p_int = summary_by_cond["intact"]["primary_4way_accuracy"]
    p_dis = summary_by_cond["disabled_pointer"]["primary_4way_accuracy"]
    p_shm = summary_by_cond["sham_perturbation"]["primary_4way_accuracy"]
    return {
        "schema": "elendril-x3-checkpoint-suite-v1",
        "architecture": arch,
        "conditions": summary_by_cond,
        "targeted_pointer_drop": round(p_int - p_dis, 6),
        "sham_drop": round(p_int - p_shm, 6),
        "targeted_minus_sham_contribution": round((p_int - p_dis) - (p_int - p_shm), 6),
        "pair_records": pair_records_by_cond,
        "decision_records": cond_outputs["intact"],
    }


@torch.no_grad()
def run_x3_pointer_interchange_intervention(
    model: nn.Module,
    model_spec: Any,
    aligned_pairs: Sequence[Mapping[str, Any]],
    *,
    device: str = "cpu",
    equalize_norm: bool = False,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Geiger et al. (NeurIPS 2021) pointer readout transplant across byte-aligned before/after pairs."""
    _ = model_spec
    model.eval()
    dev = next(model.parameters()).device
    max_ctx = _model_max_ctx(model)
    arch = _model_arch(model)
    if not aligned_pairs:
        raise ValueError("aligned_pairs must be non-empty")

    aff_orig_ok = 0
    aff_switch_to_after = 0
    aff_same_world_ok = 0
    aff_same_world_max_diff = 0.0
    aff_irrel_preserve = 0
    aff_margin_shifts: list[float] = []

    sib_orig_ok = 0
    sib_edit_donor_preserve = 0
    sib_irrel_donor_preserve = 0

    all_top1_orig: list[float] = []
    all_top1_edit: list[float] = []
    abstain_orig = 0
    abstain_edit = 0
    supports_ptr_flag = True

    aff_total = 0
    sib_total = 0

    for start in range(0, len(aligned_pairs), batch_size):
        chunk = aligned_pairs[start : start + batch_size]
        t_orig, s_orig, a_orig = _encode_texts([str(p["original_text"]) for p in chunk], dev, max_ctx)
        t_edit, s_edit, a_edit = _encode_texts([str(p["edited_text"]) for p in chunk], dev, max_ctx)
        t_irrel, s_irrel, a_irrel = _encode_texts([str(p["irrelevant_text"]) for p in chunk], dev, max_ctx)

        st_o = _extract_state(model, t_orig, s_orig, a_orig)
        if not st_o["supports_pointer"]:
            supports_ptr_flag = False
            break
        st_e = _extract_state(model, t_edit, s_edit, a_edit)
        st_i = _extract_state(model, t_irrel, s_irrel, a_irrel)

        lp_o = _decode_from_pointer_state(st_o)["log_probs_19"].cpu()
        lp_sw = _decode_from_pointer_state(st_o, q_ptr_override=st_o["q_ptr"], equalize_norm=False)["log_probs_19"].cpu()
        lp_ed = _decode_from_pointer_state(st_o, q_ptr_override=st_e["q_ptr"], equalize_norm=equalize_norm)["log_probs_19"].cpu()
        lp_ir = _decode_from_pointer_state(st_o, q_ptr_override=st_i["q_ptr"], equalize_norm=equalize_norm)["log_probs_19"].cpu()

        diff_sw = float((lp_o.exp() - lp_sw.exp()).abs().max().item())
        aff_same_world_max_diff = max(aff_same_world_max_diff, diff_sw)

        for bi, pair in enumerate(chunk):
            exp_before = str(pair["expected_before"])
            exp_after = str(pair["expected_after"])
            m_o = _action_and_margin(lp_o[bi], exp_before)
            m_sw = _action_and_margin(lp_sw[bi], exp_before)
            m_ed_before = _action_and_margin(lp_ed[bi], exp_before)
            m_ed_after = _action_and_margin(lp_ed[bi], exp_after)
            m_ir = _action_and_margin(lp_ir[bi], exp_before)

            all_top1_orig.append(m_o["top1_prob"])
            all_top1_edit.append(m_ed_after["top1_prob"])
            if m_o["abstained"]:
                abstain_orig += 1
            if m_ed_after["abstained"]:
                abstain_edit += 1

            # Margin shift toward expected_after relative to intact original
            idx_after = ACTION_19_LABELS.index(exp_after)
            idx_before = ACTION_19_LABELS.index(exp_before)
            orig_diff_lp = float(lp_o[bi, idx_after].item() - lp_o[bi, idx_before].item())
            edit_diff_lp = float(lp_ed[bi, idx_after].item() - lp_ed[bi, idx_before].item())

            if pair["query_role"] == "affected":
                aff_total += 1
                if m_o["correct"]:
                    aff_orig_ok += 1
                if m_ed_after["pred"] == exp_after:
                    aff_switch_to_after += 1
                if m_sw["correct"]:
                    aff_same_world_ok += 1
                if m_ir["correct"]:
                    aff_irrel_preserve += 1
                aff_margin_shifts.append(edit_diff_lp - orig_diff_lp)
            else:
                sib_total += 1
                if m_o["correct"]:
                    sib_orig_ok += 1
                if m_ed_before["pred"] == exp_before:
                    sib_edit_donor_preserve += 1
                if m_ir["correct"]:
                    sib_irrel_donor_preserve += 1

    if not supports_ptr_flag:
        return {
            "schema": "elendril-x3-interchange-v1",
            "architecture": arch,
            "supports_pointer_interchange": False,
            "equalize_norm": bool(equalize_norm),
            "status": "NOT_APPLICABLE_NO_POINTER_HEAD",
            "global_confidence_collapse": False,
        }

    n_all = max(1, len(all_top1_orig))
    orig_top1_mean = sum(all_top1_orig) / n_all
    edit_top1_mean = sum(all_top1_edit) / n_all
    top1_drop = orig_top1_mean - edit_top1_mean
    abstain_rate_orig = abstain_orig / n_all
    abstain_rate_edit = abstain_edit / n_all
    aff_target_rate = aff_switch_to_after / max(1, aff_total)
    sib_pres_rate = sib_edit_donor_preserve / max(1, sib_total)
    irrel_pres_rate = aff_irrel_preserve / max(1, aff_total)

    collapse = bool(
        abstain_rate_edit > 0.25
        or top1_drop > 0.25
        or (aff_target_rate < 0.20 and sib_pres_rate < 0.50 and top1_drop > 0.10)
    )
    return {
        "schema": "elendril-x3-interchange-v1",
        "architecture": arch,
        "supports_pointer_interchange": True,
        "equalize_norm": bool(equalize_norm),
        "status": "INCONCLUSIVE_CONFIDENCE_COLLAPSE" if collapse else "EVALUATED",
        "global_confidence_collapse": collapse,
        "affected_count": aff_total,
        "sibling_count": sib_total,
        "affected_intact_accuracy": round(aff_orig_ok / max(1, aff_total), 6),
        "affected_edited_donor_target_accuracy": round(aff_target_rate, 6),
        "affected_same_world_donor_max_prob_diff": round(aff_same_world_max_diff, 8),
        "affected_irrelevant_donor_preservation": round(irrel_pres_rate, 6),
        "affected_mean_margin_shift_toward_after": round(sum(aff_margin_shifts) / max(1, len(aff_margin_shifts)), 6),
        "sibling_intact_accuracy": round(sib_orig_ok / max(1, sib_total), 6),
        "sibling_edited_donor_preservation": round(sib_pres_rate, 6),
        "sibling_irrelevant_donor_preservation": round(sib_irrel_donor_preserve / max(1, sib_total), 6),
        "mean_top1_prob_original": round(orig_top1_mean, 6),
        "mean_top1_prob_edited_donor": round(edit_top1_mean, 6),
        "top1_prob_drop": round(top1_drop, 6),
        "abstain_rate_original": round(abstain_rate_orig, 6),
        "abstain_rate_edited_donor": round(abstain_rate_edit, 6),
    }


def _whole_graph_within_depth_bootstrap(
    trt_pairs: Sequence[Mapping[str, Any]],
    ctrl_pairs: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 1337,
) -> dict[str, Any]:
    ctrl_by_pk = {str(r["pair_key"]): r for r in ctrl_pairs}
    # Average paired differences over renderers within each (depth, graph_id) whole graph
    by_depth_graph: dict[int, dict[str, list[float]]] = {}
    for tr in trt_pairs:
        pk = str(tr["pair_key"])
        if pk not in ctrl_by_pk:
            continue
        cr = ctrl_by_pk[pk]
        delta = float(bool(tr["primary_4way"])) - float(bool(cr["primary_4way"]))
        dp = int(tr["depth"])
        gid = str(tr["graph_id"])
        by_depth_graph.setdefault(dp, {}).setdefault(gid, []).append(delta)

    strata: list[list[float]] = []
    for dp in sorted(by_depth_graph):
        g_vals = [sum(vs) / len(vs) for _, vs in sorted(by_depth_graph[dp].items())]
        if g_vals:
            strata.append(g_vals)

    flat = [x for st in strata for x in st]
    if not flat:
        return {"mean_delta": 0.0, "ci95": [0.0, 0.0], "depths": 0, "graphs": 0}
    point = sum(flat) / len(flat)
    rng = np.random.default_rng(int(seed))
    boot_vals = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        acc = 0.0
        cnt = 0
        for st in strata:
            arr = np.asarray(st, dtype=np.float64)
            idx = rng.integers(0, len(arr), size=len(arr))
            acc += float(arr[idx].sum())
            cnt += len(arr)
        boot_vals[b] = acc / max(1, cnt)
    lo, hi = np.quantile(boot_vals, [0.025, 0.975]).tolist()
    return {
        "schema": "anra-x3-depth-graph-bootstrap-v1",
        "mean_delta": round(float(point), 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "depths": len(strata),
        "graphs": len(flat),
        "replicates": int(n_boot),
    }


def score_x3_campaign(
    model_results_by_arch: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    stage: str = "dev",
    n_boot: int = 2000,
    seed: int = 1337,
) -> dict[str, Any]:
    """Score X3 frozen model suites across seeds with whole-graph-within-depth bootstrap & advancement gate."""
    per_arch: dict[str, Any] = {}
    for arch, seed_map in sorted(model_results_by_arch.items()):
        seed_summaries: dict[str, Any] = {}
        for s_id, res in sorted(seed_map.items()):
            c_int = res["conditions"]["intact"]
            c_dis = res["conditions"]["disabled_pointer"]
            c_shm = res["conditions"]["sham_perturbation"]
            seed_summaries[str(s_id)] = {
                "intact_primary": c_int["primary_4way_accuracy"],
                "intact_numerator": c_int["primary_numerator"],
                "intact_denominator": c_int["primary_denominator"],
                "disabled_pointer_primary": c_dis["primary_4way_accuracy"],
                "sham_perturbation_primary": c_shm["primary_4way_accuracy"],
                "targeted_minus_sham": res["targeted_minus_sham_contribution"],
                "canonical_primary": c_int["canonical_primary"],
                "narrative_primary": c_int["narrative_primary"],
                "narrative_loss": c_int["narrative_loss"],
                "erroneous_sibling_change_rate": c_int["erroneous_sibling_change_rate"],
                "distractor_following_rate": c_int["distractor_following_rate"],
            }
        int_vals = [v["intact_primary"] for v in seed_summaries.values()]
        per_arch[arch] = {
            "mean_intact_primary": round(sum(int_vals) / max(1, len(int_vals)), 6),
            "seeds": seed_summaries,
        }

    p_map = model_results_by_arch.get("P", {})
    f_map = model_results_by_arch.get("F-cap", {})
    shared_pf = sorted(set(p_map) & set(f_map))
    pf_deltas = [
        p_map[s]["conditions"]["intact"]["primary_4way_accuracy"] - f_map[s]["conditions"]["intact"]["primary_4way_accuracy"]
        for s in shared_pf
    ]
    p_tms = [p_map[s]["targeted_minus_sham_contribution"] for s in sorted(p_map)]

    r_map = model_results_by_arch.get("R", {})
    rn_map = model_results_by_arch.get("R-noedge", {})
    shared_rn = sorted(set(r_map) & set(rn_map))
    rn_deltas = [
        r_map[s]["conditions"]["intact"]["primary_4way_accuracy"] - rn_map[s]["conditions"]["intact"]["primary_4way_accuracy"]
        for s in shared_rn
    ]

    pooled_p_pairs = [rec for s in shared_pf for rec in p_map[s]["pair_records"]["intact"]]
    pooled_f_pairs = [rec for s in shared_pf for rec in f_map[s]["pair_records"]["intact"]]
    pf_boot = _whole_graph_within_depth_bootstrap(pooled_p_pairs, pooled_f_pairs, n_boot=n_boot, seed=seed)

    pooled_r_pairs = [rec for s in shared_rn for rec in r_map[s]["pair_records"]["intact"]]
    pooled_rn_pairs = [rec for s in shared_rn for rec in rn_map[s]["pair_records"]["intact"]]
    rn_boot = _whole_graph_within_depth_bootstrap(pooled_r_pairs, pooled_rn_pairs, n_boot=n_boot, seed=seed + 1)

    pf_mean = sum(pf_deltas) / max(1, len(pf_deltas)) if pf_deltas else 0.0
    pf_pos = sum(1 for d in pf_deltas if d > 0.0)
    tms_mean = sum(p_tms) / max(1, len(p_tms)) if p_tms else 0.0

    gate_criteria = {
        "p_minus_fcap_ge_0_05": pf_mean >= 0.05,
        "positive_in_at_least_2_of_3_seeds": pf_pos >= min(2, len(shared_pf)) and len(shared_pf) > 0,
        "positive_targeted_minus_sham_pointer": tms_mean > 0.0,
    }
    if stage == "confirmation":
        gate_criteria["confirmation_replicated"] = bool(
            gate_criteria["p_minus_fcap_ge_0_05"] and gate_criteria["positive_in_at_least_2_of_3_seeds"]
        )
    gate_passed = bool(shared_pf and all(gate_criteria.values()))

    return {
        "schema": "elendril-x3-campaign-score-v1",
        "stage": stage,
        "gate_passed": gate_passed,
        "gate_criteria": gate_criteria,
        "p_vs_fcap": {
            "shared_seeds": shared_pf,
            "per_seed_deltas": {str(s): round(d, 6) for s, d in zip(shared_pf, pf_deltas)},
            "mean_delta": round(pf_mean, 6),
            "positive_seeds": pf_pos,
            "bootstrap": pf_boot,
        },
        "r_vs_r_noedge": {
            "shared_seeds": shared_rn,
            "per_seed_deltas": {str(s): round(d, 6) for s, d in zip(shared_rn, rn_deltas)},
            "mean_delta": round(sum(rn_deltas) / max(1, len(rn_deltas)), 6) if rn_deltas else 0.0,
            "bootstrap": rn_boot,
        },
        "p_targeted_minus_sham_by_seed": {str(s): round(p_map[s]["targeted_minus_sham_contribution"], 6) for s in sorted(p_map)},
        "p_mean_targeted_minus_sham": round(tms_mean, 6),
        "architectures": per_arch,
    }


__all__ = [
    "evaluate_x3_checkpoint_suite",
    "run_x3_pointer_interchange_intervention",
    "score_x3_campaign",
]
