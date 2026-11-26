"""V8-X4 Reconstructed Learning Algorithms: Strictly Exposure-Matched D0, D1, D2, D3 & D4-ECCO."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from anra_v5.elendril.codec import digest
from anra_v5.elendril.representation import UTF8
from experiments.x3.d3_objectives import apply_pcgrad_conflict_surgery, d3_warmup_ramp_weights
from v5_model.core import packed_layout

from .battery import build_x4_curriculum_schedule, format_x4_typed_target
from .models import X4CognitiveModel, encode_prompt_and_target_pair

X4_OBJECTIVES: tuple[str, ...] = (
    "D0-ByteCE",
    "D0-Typed",
    "D1-CorrInvar",
    "D2-ExactCF",
    "D3-SimpleWeight",
    "D3-Adaptive",
    "D4-ECCO",
)


def build_x4_matched_batches(
    cases_or_schedule: Sequence[Mapping[str, Any]],
    *,
    seed: int = 7,
    batch_size: int = 16,
    is_schedule: bool = False,
) -> dict[str, Any]:
    """Build identical ordered batches with explicit semantic pair maps and exact exposure manifest (§6, §7)."""
    batches: list[dict[str, Any]] = []
    if is_schedule:
        raw_steps = list(cases_or_schedule)
    else:
        pool = list(cases_or_schedule)
        if not pool:
            raise ValueError("build_x4_matched_batches requires non-empty cases")
        raw_steps = []
        step = 0
        # Keep adjacent pair partners together when partitioning into fixed batches
        for s_i in range(0, len(pool), max(1, int(batch_size))):
            chunk = [dict(c) for c in pool[s_i : s_i + int(batch_size)]]
            if len(chunk) < int(batch_size):
                pad_n = int(batch_size) - len(chunk)
                chunk.extend(dict(pool[j % len(pool)]) for j in range(pad_n))
            raw_steps.append({"update_idx": step, "phase": 3, "cases": chunk})
            step += 1

    unique_ids: set[str] = set()
    total_pres = 0
    total_cf_pairs = 0
    total_eq_pairs = 0
    total_irr_pairs = 0
    for st in raw_steps:
        items = list(st["cases"])
        by_gid: dict[str, list[int]] = {}
        for idx_i, c in enumerate(items):
            unique_ids.add(str(c["case_id"]))
            gid = str(c.get("pair_group_id", f"solo:{idx_i}"))
            by_gid.setdefault(gid, []).append(idx_i)
        total_pres += len(items)

        cf_pairs: list[tuple[int, int]] = []
        eq_pairs: list[tuple[int, int]] = []
        irr_pairs: list[tuple[int, int]] = []
        for idxs in by_gid.values():
            if len(idxs) < 2:
                continue
            for p_a in range(len(idxs)):
                for p_b in range(p_a + 1, len(idxs)):
                    ia, ib = idxs[p_a], idxs[p_b]
                    ra = str(items[ia].get("pair_relation", "base"))
                    rb = str(items[ib].get("pair_relation", "base"))
                    ta = format_x4_typed_target(items[ia])
                    tb = format_x4_typed_target(items[ib])
                    if ra == "cf_irrelevant" or rb == "cf_irrelevant":
                        irr_pairs.append((ia, ib))
                    elif ta != tb or ra == "cf_relevant" or rb == "cf_relevant":
                        if ta != tb:
                            cf_pairs.append((ia, ib))
                    elif ra in ("equiv_surface", "equiv_perm") or rb in ("equiv_surface", "equiv_perm"):
                        eq_pairs.append((ia, ib))

        total_cf_pairs += len(cf_pairs)
        total_eq_pairs += len(eq_pairs)
        total_irr_pairs += len(irr_pairs)
        batches.append(
            {
                "update_idx": int(st.get("update_idx", len(batches))),
                "phase": int(st.get("phase", 3)),
                "cases": items,
                "cf_pairs": cf_pairs,
                "eq_pairs": eq_pairs,
                "irr_pairs": irr_pairs,
            }
        )

    manifest = {
        "schema": "elendril-x4-exposure-manifest-v1",
        "seed": int(seed),
        "batches_count": len(batches),
        "total_presentations": int(total_pres),
        "unique_case_ids_count": len(unique_ids),
        "total_cf_relevant_pairs": int(total_cf_pairs),
        "total_equiv_pairs": int(total_eq_pairs),
        "total_irr_pairs": int(total_irr_pairs),
        "batch_plan_sha256": digest([[str(c["case_id"]) for c in b["cases"]] for b in batches]),
    }
    return {"batches": batches, "exposure_manifest": manifest}


def _pack_cases_tensors(
    cases: Sequence[Mapping[str, Any]],
    *,
    targets_override: Sequence[str] | None = None,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = []
    op_prefix_masks = []
    colon_byte = UTF8.encode(":")[0]
    for idx_i, c in enumerate(cases):
        prompt = str(c["prompt"])
        tgt = str(targets_override[idx_i]) if targets_override is not None else format_x4_typed_target(c)
        ids, elig, end_p = encode_prompt_and_target_pair(prompt, tgt, max_length=max_length)
        encoded.append((ids, elig, end_p))
        # Mark prefix operation token span (up to ':' or first 7 target bytes)
        pmask = [False] * len(ids)
        for pos_k in range(end_p + 1, min(len(ids), end_p + 9)):
            pmask[pos_k] = True
            if ids[pos_k] == colon_byte or ids[pos_k] == UTF8.eos_id:
                break
        op_prefix_masks.append(pmask)

    bsz = len(encoded)
    max_l = max(len(ids) for ids, _, _ in encoded)
    toks = torch.zeros((bsz, max_l), dtype=torch.long, device=device)
    segs = torch.full((bsz, max_l), -1, dtype=torch.long, device=device)
    elig_t = torch.zeros((bsz, max_l), dtype=torch.bool, device=device)
    prefix_t = torch.zeros((bsz, max_l), dtype=torch.bool, device=device)
    end_t = torch.zeros((bsz,), dtype=torch.long, device=device)
    for i, ((ids, el, ep), pm) in enumerate(zip(encoded, op_prefix_masks)):
        toks[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        segs[i, : len(ids)] = 0
        elig_t[i, : len(el)] = torch.tensor(el, dtype=torch.bool, device=device)
        prefix_t[i, : len(pm)] = torch.tensor(pm, dtype=torch.bool, device=device)
        end_t[i] = ep
    return toks, segs, elig_t, prefix_t, end_t


def prepack_x4_batches(
    matched_bundle: Mapping[str, Any],
    *,
    max_length: int,
    device: str | torch.device,
) -> list[dict[str, Any]]:
    """Pre-pack identical tensor batches once so every objective executes on exact same tokens & pairs."""
    dev = torch.device(device)
    out: list[dict[str, Any]] = []
    for b in matched_bundle["batches"]:
        cases = b["cases"]
        toks, segs, elig_t, prefix_t, end_t = _pack_cases_tensors(cases, max_length=int(max_length), device=dev)
        pos, mask = packed_layout(segs, torch_module=torch)

        # Prepack counterfactual swapped-target sequences for all cf_pairs (x_a -> y_b, x_b -> y_a)
        cf_pairs: list[tuple[int, int]] = list(b["cf_pairs"])
        swap_pack: dict[str, torch.Tensor] | None = None
        if cf_pairs:
            swap_cases: list[Mapping[str, Any]] = []
            swap_targets: list[str] = []
            for ia, ib in cf_pairs:
                swap_cases.append(cases[ia])
                swap_targets.append(format_x4_typed_target(cases[ib]))
                swap_cases.append(cases[ib])
                swap_targets.append(format_x4_typed_target(cases[ia]))
            sw_toks, sw_segs, sw_elig, _, sw_end = _pack_cases_tensors(
                swap_cases, targets_override=swap_targets, max_length=int(max_length), device=dev
            )
            sw_pos, sw_mask = packed_layout(sw_segs, torch_module=torch)
            swap_pack = {
                "toks": sw_toks,
                "pos": sw_pos,
                "mask": sw_mask,
                "elig_t": sw_elig,
                "end_t": sw_end,
            }

        out.append(
            {
                **b,
                "_packed": {
                    "device": str(dev),
                    "max_length": int(max_length),
                    "toks": toks,
                    "pos": pos,
                    "mask": mask,
                    "elig_t": elig_t,
                    "prefix_t": prefix_t,
                    "end_t": end_t,
                    "swap_pack": swap_pack,
                    "eligible_target_bytes": int(elig_t[:, 1:].sum().item()),
                },
            }
        )
    return out


def _per_sample_mean_target_logprob(log_p_unified: torch.Tensor, toks: torch.Tensor, elig_t: torch.Tensor) -> torch.Tensor:
    """Compute exact per-sample mean target log-probability across eligible bytes `[B]`."""
    targets = toks[:, 1:]  # [B, L-1]
    gathered = log_p_unified[:, :-1, :].gather(2, targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    m = elig_t[:, 1:].float()
    return (gathered * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def compute_x4_objective_losses(
    model: X4CognitiveModel,
    batch: Mapping[str, Any],
    *,
    objective: str,
    update_idx: int = 0,
    total_updates: int = 1024,
    eq_weight: float = 0.15,
    cf_weight: float = 0.25,
    cf_margin: float = 0.5,
    ecco_beta: float = 1.5,
    use_amp: bool | None = None,
) -> dict[str, Any]:
    """Compute mathematically distinct, exposure-matched losses without grad step (supports fixture testing & training)."""
    if objective not in X4_OBJECTIVES:
        raise ValueError(f"Unknown X4 objective: {objective!r}")
    dev = next(model.parameters()).device
    packed = batch.get("_packed")
    if isinstance(packed, Mapping) and packed.get("device") == str(dev) and packed.get("max_length") == model.max_ctx:
        toks = packed["toks"]
        pos = packed["pos"]
        mask = packed["mask"]
        elig_t = packed["elig_t"]
        prefix_t = packed["prefix_t"]
        end_t = packed["end_t"]
        swap_pack = packed["swap_pack"]
    else:
        cases = batch["cases"]
        toks, segs, elig_t, prefix_t, end_t = _pack_cases_tensors(cases, max_length=model.max_ctx, device=dev)
        pos, mask = packed_layout(segs, torch_module=torch)
        cf_pairs = list(batch.get("cf_pairs", ()))
        swap_pack = None
        if cf_pairs:
            swap_cases = []
            swap_targets = []
            for ia, ib in cf_pairs:
                swap_cases.extend((cases[ia], cases[ib]))
                swap_targets.extend((format_x4_typed_target(cases[ib]), format_x4_typed_target(cases[ia])))
            sw_toks, sw_segs, sw_elig, _, sw_end = _pack_cases_tensors(
                swap_cases, targets_override=swap_targets, max_length=model.max_ctx, device=dev
            )
            sw_pos, sw_mask = packed_layout(sw_segs, torch_module=torch)
            swap_pack = {"toks": sw_toks, "pos": sw_pos, "mask": sw_mask, "elig_t": sw_elig, "end_t": sw_end}

    amp_on = bool(use_amp) if use_amp is not None else (dev.type == "cuda")
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_on):
        out = model(toks, pos, mask, prompt_end_positions=end_t)

    log_p = out["log_p_unified"].float()  # already log-normalized across vocab
    targets = toks[:, 1:]
    per_tok_nll = -log_p[:, :-1, :].gather(2, targets.unsqueeze(-1)).squeeze(-1)
    m_all = elig_t[:, 1:].float()
    byte_ce = (per_tok_nll * m_all).sum() / m_all.sum().clamp_min(1.0)

    zero_scalar = torch.zeros((), device=dev, dtype=torch.float32)
    op_prefix_loss = zero_scalar
    if objective == "D0-Typed":
        m_pre = prefix_t[:, 1:].float()
        op_prefix_loss = 0.25 * ((per_tok_nll * m_pre).sum() / m_pre.sum().clamp_min(1.0))

    # Mean target sequence log-likelihood [B] under unified copy+generate distribution
    seq_ll = _per_sample_mean_target_logprob(log_p, toks, elig_t)

    eq_loss = zero_scalar
    eq_pairs: Sequence[tuple[int, int]] = batch.get("eq_pairs", ())
    irr_pairs: Sequence[tuple[int, int]] = batch.get("irr_pairs", ())
    need_eq = objective in ("D1-CorrInvar", "D3-SimpleWeight", "D3-Adaptive", "D4-ECCO")
    if need_eq and (eq_pairs or irr_pairs):
        eq_terms = []
        for ia, ib in eq_pairs:
            # Correctness-gated invariance: suppress synchronization when BOTH predictions are low-confidence on gold
            anchor_ll = torch.maximum(seq_ll[ia], seq_ll[ib]).detach()
            gate = torch.sigmoid(4.0 * (anchor_ll + 1.5))
            eq_terms.append(gate * (seq_ll[ia] - seq_ll[ib]).pow(2))
        for ia, ib in irr_pairs:
            anchor_ll = torch.maximum(seq_ll[ia], seq_ll[ib]).detach()
            gate = torch.sigmoid(4.0 * (anchor_ll + 1.5))
            eq_terms.append(gate * (seq_ll[ia] - seq_ll[ib]).pow(2))
        if eq_terms:
            eq_loss = torch.stack(eq_terms).mean()

    cf_loss = zero_scalar
    cf_pairs: Sequence[tuple[int, int]] = batch.get("cf_pairs", ())
    need_cf = objective in ("D2-ExactCF", "D3-SimpleWeight", "D3-Adaptive", "D4-ECCO")
    mean_cf_asymmetry = 0.0
    if need_cf and cf_pairs and swap_pack is not None:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_on):
            sw_out = model(
                swap_pack["toks"],
                swap_pack["pos"],
                swap_pack["mask"],
                prompt_end_positions=swap_pack["end_t"],
            )
        sw_log_p = sw_out["log_p_unified"].float()
        sw_ll = _per_sample_mean_target_logprob(sw_log_p, swap_pack["toks"], swap_pack["elig_t"])

        cf_terms = []
        asym_vals = []
        for k_i, (ia, ib) in enumerate(cf_pairs):
            ll_aa = seq_ll[ia]
            ll_bb = seq_ll[ib]
            ll_ab = sw_ll[2 * k_i]      # x_a scored on wrong twin target y_b
            ll_ba = sw_ll[2 * k_i + 1]  # x_b scored on wrong twin target y_a
            margin_a = F.relu(float(cf_margin) - (ll_aa - ll_ab))
            margin_b = F.relu(float(cf_margin) - (ll_bb - ll_ba))
            pair_cf = 0.5 * (margin_a + margin_b)
            if objective == "D4-ECCO":
                # Error-conditioned credit assignment weighting asymmetric right/wrong twins cleanly
                pa_soft = torch.exp(ll_aa.clamp(-20.0, 0.0)).detach()
                pb_soft = torch.exp(ll_bb.clamp(-20.0, 0.0)).detach()
                asym = torch.abs(pa_soft - pb_soft)
                asym_vals.append(float(asym.item()))
                w_pair = 1.0 + float(ecco_beta) * asym
                cf_terms.append(w_pair * pair_cf)
            else:
                cf_terms.append(pair_cf)
        if cf_terms:
            cf_loss = torch.stack(cf_terms).mean()
        if asym_vals:
            mean_cf_asymmetry = float(sum(asym_vals) / len(asym_vals))

    # Combine with explicit schedule weights (D0-ByteCE has strictly ZERO aux terms!)
    if objective == "D0-ByteCE":
        eff_eq, eff_cf = 0.0, 0.0
        base_loss = byte_ce
        aux_loss = zero_scalar
    elif objective == "D0-Typed":
        eff_eq, eff_cf = 0.0, 0.0
        base_loss = byte_ce + op_prefix_loss
        aux_loss = zero_scalar
    elif objective == "D1-CorrInvar":
        eff_eq, eff_cf = float(eq_weight), 0.0
        base_loss = byte_ce
        aux_loss = eff_eq * eq_loss
    elif objective == "D2-ExactCF":
        eff_eq, eff_cf = 0.0, float(cf_weight)
        base_loss = byte_ce
        aux_loss = eff_cf * cf_loss
    elif objective == "D3-SimpleWeight":
        eff_eq, eff_cf = float(eq_weight), float(cf_weight)
        base_loss = byte_ce
        aux_loss = eff_eq * eq_loss + eff_cf * cf_loss
    elif objective == "D3-Adaptive":
        wu = max(1, int(0.25 * total_updates))
        ru = max(1, int(0.35 * total_updates))
        eff_eq, eff_cf = d3_warmup_ramp_weights(update_idx, warmup_updates=wu, ramp_updates=ru, target_eq=eq_weight, target_cf=cf_weight)
        base_loss = byte_ce
        aux_loss = eff_eq * eq_loss + eff_cf * cf_loss
    elif objective == "D4-ECCO":
        wu = max(1, int(0.15 * total_updates))
        ru = max(1, int(0.30 * total_updates))
        eff_eq, eff_cf = d3_warmup_ramp_weights(update_idx, warmup_updates=wu, ramp_updates=ru, target_eq=eq_weight, target_cf=cf_weight)
        base_loss = byte_ce
        aux_loss = eff_eq * eq_loss + eff_cf * cf_loss
    else:
        raise AssertionError(f"Unreachable objective: {objective}")

    total_loss = base_loss + aux_loss
    return {
        "total_loss_t": total_loss,
        "base_loss_t": base_loss,
        "aux_loss_t": aux_loss,
        "byte_ce": float(byte_ce.detach().item()),
        "op_prefix_loss": float(op_prefix_loss.detach().item()),
        "eq_loss": float(eq_loss.detach().item()),
        "cf_loss": float(cf_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "eff_eq_weight": float(eff_eq),
        "eff_cf_weight": float(eff_cf),
        "mean_cf_asymmetry": float(mean_cf_asymmetry),
        "eligible_target_bytes": int(m_all.sum().item()),
        "cf_pairs_evaluated": len(cf_pairs) if need_cf else 0,
        "eq_pairs_evaluated": (len(eq_pairs) + len(irr_pairs)) if need_eq else 0,
    }


def compute_x4_objective_step(
    model: X4CognitiveModel,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    *,
    objective: str = "D0-ByteCE",
    update_idx: int = 0,
    total_updates: int = 1024,
    eq_weight: float = 0.15,
    cf_weight: float = 0.25,
    cf_margin: float = 0.5,
    ecco_beta: float = 1.5,
    clip: float = 1.0,
    use_amp: bool | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    """Run one verified optimizer step across ANY X4 objective with PCGrad for `D3-Adaptive` & AMP support."""
    dev = next(model.parameters()).device
    model.train()
    amp_on = bool(use_amp) if use_amp is not None else (dev.type == "cuda")
    losses = compute_x4_objective_losses(
        model,
        batch,
        objective=objective,
        update_idx=update_idx,
        total_updates=total_updates,
        eq_weight=eq_weight,
        cf_weight=cf_weight,
        cf_margin=cf_margin,
        ecco_beta=ecco_beta,
        use_amp=amp_on,
    )

    optimizer.zero_grad(set_to_none=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    scaler_on = scaler is not None and scaler.is_enabled()
    surgery_info = {"dot_before": 0.0, "dot_after": 0.0, "cosine_before": 0.0, "conflict_detected": False}

    use_pcgrad = objective == "D3-Adaptive" and float(losses["eff_eq_weight"] + losses["eff_cf_weight"]) > 0.0 and float(losses["aux_loss_t"].detach().item()) > 0.0
    if use_pcgrad:
        sb = scaler.scale(losses["base_loss_t"]) if scaler_on else losses["base_loss_t"]
        sa = scaler.scale(losses["aux_loss_t"]) if scaler_on else losses["aux_loss_t"]
        base_g = torch.autograd.grad(sb, trainable, retain_graph=True, allow_unused=True)
        aux_g = torch.autograd.grad(sa, trainable, allow_unused=True)
        shared_idx = [i for i, (bg, ag) in enumerate(zip(base_g, aux_g)) if bg is not None and ag is not None]
        if shared_idx:
            comb_shared, surgery_info = apply_pcgrad_conflict_surgery([base_g[i].float() for i in shared_idx], [aux_g[i].float() for i in shared_idx])
            shared_map = {idx: cg for idx, cg in zip(shared_idx, comb_shared)}
        else:
            shared_map = {}
        for i, p in enumerate(trainable):
            if i in shared_map:
                p.grad = shared_map[i].to(dtype=p.dtype)
            elif base_g[i] is not None and aux_g[i] is not None:
                p.grad = (base_g[i] + aux_g[i]).to(dtype=p.dtype)
            elif base_g[i] is not None:
                p.grad = base_g[i].to(dtype=p.dtype)
            elif aux_g[i] is not None:
                p.grad = aux_g[i].to(dtype=p.dtype)
    else:
        tot = losses["total_loss_t"]
        if scaler_on:
            scaler.scale(tot).backward()
        else:
            tot.backward()

    if scaler_on:
        scaler.unscale_(optimizer)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(clip)))
    amp_skipped = not math.isfinite(grad_norm)
    if scaler_on:
        scaler.step(optimizer)
        scaler.update()
    elif not amp_skipped:
        optimizer.step()

    return {
        "objective": str(objective),
        "byte_ce": float(losses["byte_ce"]),
        "op_prefix_loss": float(losses["op_prefix_loss"]),
        "eq_loss": float(losses["eq_loss"]),
        "cf_loss": float(losses["cf_loss"]),
        "total_loss": float(losses["total_loss"]),
        "eff_eq_weight": float(losses["eff_eq_weight"]),
        "eff_cf_weight": float(losses["eff_cf_weight"]),
        "mean_cf_asymmetry": float(losses["mean_cf_asymmetry"]),
        "eligible_target_bytes": int(losses["eligible_target_bytes"]),
        "cf_pairs_evaluated": int(losses["cf_pairs_evaluated"]),
        "eq_pairs_evaluated": int(losses["eq_pairs_evaluated"]),
        "grad_norm": grad_norm,
        "amp_skipped": bool(amp_skipped),
        "conflict_detected": bool(surgery_info["conflict_detected"]),
        "cosine_before": float(surgery_info["cosine_before"]),
    }


__all__ = [
    "X4_OBJECTIVES",
    "build_x4_curriculum_schedule",
    "build_x4_matched_batches",
    "compute_x4_objective_losses",
    "compute_x4_objective_step",
    "prepack_x4_batches",
]
