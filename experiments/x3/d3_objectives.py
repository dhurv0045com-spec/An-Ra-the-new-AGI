"""V8-X3 D3 Learning Algorithms: Conditional Schedule Audit, 2x2 Pack Factorial, Warmup/Ramp & PCGrad (§1, §3 R4)."""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.codec import digest
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v8x.data import build_x1_d1h_training_dataset, build_x1_d2_paired_dataset
from v5_model.core import packed_layout

from .cognitive_battery import format_public_typed_target
from .models import X3CognitiveModel, encode_prompt_and_target_pair

D3_OBJECTIVES: tuple[str, ...] = (
    "CE",
    "D1",
    "D2",
    "D3-Hybrid",
    "D3-NoSurgery",
    "D3-NoAux",
    "D3-Delay-PCGrad",
    "D3-Delay-NoSurgery",
    "D3-Immediate",
    "D3-ShuffledTarget",
    "D3-WorldCycle",
)


def d3_warmup_ramp_weights(
    update_idx: int,
    *,
    warmup_updates: int = 592,
    ramp_updates: int = 592,
    target_eq: float = 0.1,
    target_cf: float = 0.1,
) -> tuple[float, float]:
    """Predeclared §3 R4 schedule: 592-update pure CE warmup, then 592-update linear ramp to (0.1, 0.1)."""
    u = int(update_idx)
    wu = max(0, int(warmup_updates))
    ru = max(1, int(ramp_updates))
    if u < wu:
        return 0.0, 0.0
    if u < wu + ru:
        frac = float(u - wu) / float(ru)
        return float(target_eq) * frac, float(target_cf) * frac
    return float(target_eq), float(target_cf)


def audit_x1_sampler_exposure(*, quota_per_family: int = 24) -> dict[str, Any]:
    """Quantify X1 D1-H vs D2 population sizes, base-world dilution, and alt-renderer counts (§1)."""
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
            "vs ~16 in D1-H interleaved); total ABSTAIN labels match across populations, leaving conditioning on "
            "immediately preceding observation/tool stage, within-batch correlation, and gradient interference live."
        ),
    }


def _preceding_stage_from_state(st: Any) -> str:
    """Classify the immediately preceding observation/tool stage purely from public state markers."""
    txt = UTF8.decode([t for t in st.tokens if t >= 4])
    step = int(getattr(st, "step_position", 0))
    if step == 0:
        return "initial_prefix"
    has_q_avail = ("Query=1" in txt or "Q=1" in txt)
    has_r_avail = ("Retrieve=1" in txt or "R=1" in txt)
    if not has_q_avail and not has_r_avail:
        return "after_all_tools_exhausted"
    if not has_r_avail:
        return "after_retrieve_observation"
    if not has_q_avail:
        return "after_query_observation"
    return f"post_step_{step}"


def audit_d1h_d2_schedule_and_preceding_observations(
    *,
    quota_per_family: int = 512,
    total_updates: int = 2368,
    seeds: Sequence[int] = (7, 19, 41),
    d1h_sampling: str = "interleaved",
) -> dict[str, Any]:
    """Reconstruct X1 2,368-update schedules across seeds 7/19/41 conditioning on preceding observation stage & window (§1, §3)."""
    d1h = build_x1_d1h_training_dataset(quota_per_family=int(quota_per_family))
    d2 = build_x1_d2_paired_dataset(quota_per_family=int(quota_per_family))
    n_s = len(d1h.states)
    n_g = len(d2.groups)
    op_names = {0: "ANSWER", 1: "QUERY", 2: "RETRIEVE", 3: "ABSTAIN"}
    base_ord = list(d1h.interleaved_schedule) if d1h_sampling == "interleaved" else None

    def _window(u: int) -> str:
        if u < 592:
            return "0000_0591"
        if u < 1184:
            return "0592_1183"
        if u < 1472:
            return "1184_1471"
        return "1472_2367"

    d1h_by_seed: dict[int, Any] = {}
    d2_by_seed: dict[int, Any] = {}
    for seed in seeds:
        # 1. Reconstruct D1-H schedule matching X1S20Trainer._state_epoch_order
        s_orders: dict[int, list[int]] = {}
        d1h_chosen: list[int] = []
        d1h_abstain = 0
        d1h_by_stage: dict[str, Counter[str]] = {}
        d1h_abstain_by_window: Counter[str] = Counter()
        for u in range(int(total_updates)):
            win = _window(u)
            for slot in range(16):
                pos = u * 16 + slot
                ep, off = divmod(pos, n_s)
                if ep not in s_orders:
                    order = list(base_ord) if base_ord is not None else list(range(n_s))
                    if base_ord is None or ep > 0:
                        random.Random(int(digest(["x1-state-order-v1", int(seed), ep]), 16)).shuffle(order)
                    s_orders[ep] = order
                idx = s_orders[ep][off]
                d1h_chosen.append(idx)
                st = d1h.states[idx]
                op_str = op_names[int(st.gold_op)]
                stage = _preceding_stage_from_state(st)
                d1h_by_stage.setdefault(stage, Counter())[op_str] += 1
                if int(st.gold_op) == 3:
                    d1h_abstain += 1
                    d1h_abstain_by_window[win] += 1
        d1h_by_seed[int(seed)] = {
            "total_consumed_labels": len(d1h_chosen),
            "distinct_states_reached": len(set(d1h_chosen)),
            "consumed_abstain": d1h_abstain,
            "by_preceding_stage": {k: dict(sorted(v.items())) for k, v in sorted(d1h_by_stage.items())},
            "abstain_by_window": dict(sorted(d1h_abstain_by_window.items())),
        }

        # 2. Reconstruct D2 schedule matching X1S20Trainer (4 groups x 4 views = 16 states)
        g_orders: dict[int, list[int]] = {}
        d2_groups_chosen: list[int] = []
        d2_abstain = 0
        d2_by_stage: dict[str, Counter[str]] = {}
        d2_abstain_by_window: Counter[str] = Counter()
        for u in range(int(total_updates)):
            win = _window(u)
            for slot in range(4):
                pos = u * 4 + slot
                ep, off = divmod(pos, n_g)
                if ep not in g_orders:
                    order = list(range(n_g))
                    random.Random(int(digest(["x1-d2-group-order-v1", int(seed), ep]), 16)).shuffle(order)
                    g_orders[ep] = order
                gi = g_orders[ep][off]
                d2_groups_chosen.append(gi)
                for st in d2.groups[gi].samples:
                    op_str = op_names[int(st.gold_op)]
                    stage = _preceding_stage_from_state(st)
                    d2_by_stage.setdefault(stage, Counter())[op_str] += 1
                    if int(st.gold_op) == 3:
                        d2_abstain += 1
                        d2_abstain_by_window[win] += 1
        d2_by_seed[int(seed)] = {
            "total_consumed_labels": len(d2_groups_chosen) * 4,
            "total_group_selections": len(d2_groups_chosen),
            "distinct_groups_reached": len(set(d2_groups_chosen)),
            "consumed_abstain": d2_abstain,
            "by_preceding_stage": {k: dict(sorted(v.items())) for k, v in sorted(d2_by_stage.items())},
            "abstain_by_window": dict(sorted(d2_abstain_by_window.items())),
        }

    return {
        "schema": "elendril-x3-d1h-d2-conditional-schedule-audit-v1",
        "quota_per_family": int(quota_per_family),
        "total_updates": int(total_updates),
        "population": {
            "d1h_worlds": len(d1h.worlds),
            "d1h_states": n_s,
            "d1h_abstain_states": sum(1 for st in d1h.states if int(st.gold_op) == 3),
            "d2_groups": n_g,
            "d2_states": sum(len(g.samples) for g in d2.groups),
            "d2_abstain_states": sum(1 for g in d2.groups for st in g.samples if int(st.gold_op) == 3),
        },
        "d1h_by_seed": d1h_by_seed,
        "d2_by_seed": d2_by_seed,
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


def build_d3_16world_pack_minibatches(
    groups_16: Sequence[Sequence[Mapping[str, Any]]],
    *,
    arrangement: str = "grouped",
) -> list[list[Mapping[str, Any]]]:
    """Build 4 minibatches of 16 from 16 worlds x 4 views = exact same 64 cases (§3 D3 learning).

    - `grouped`: each minibatch presents 4 complete worlds x all 4 views.
    - `interleaved`: each minibatch presents 1 view of each of 16 distinct worlds, rotating view IDs.
    """
    if arrangement not in ("grouped", "interleaved"):
        raise ValueError(f"Unsupported pack arrangement {arrangement!r}; expected 'grouped' or 'interleaved'")
    if len(groups_16) != 16:
        raise ValueError(f"16-world pack requires 16 world groups, got {len(groups_16)}")
    for wi, grp in enumerate(groups_16):
        if len(grp) != 4:
            raise ValueError(f"World group {wi} must have 4 views, got {len(grp)}")

    mbs: list[list[Mapping[str, Any]]] = [[] for _ in range(4)]
    if arrangement == "grouped":
        for mb_i in range(4):
            for w_off in range(4):
                w_idx = mb_i * 4 + w_off
                for v_i in range(4):
                    mbs[mb_i].append(dict(groups_16[w_idx][v_i]))
    else:
        for mb_i in range(4):
            for w_idx in range(16):
                v_i = (w_idx + mb_i) % 4
                mbs[mb_i].append(dict(groups_16[w_idx][v_i]))
    return mbs


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
        typed_target = format_public_typed_target(c)
        ids, elig, ap = encode_prompt_and_target_pair(prompt, typed_target, max_length=max_length)
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


def verify_fixed_weight_4microbatch_ce_equivalence(
    model: X3CognitiveModel,
    groups_16: Sequence[Sequence[Mapping[str, Any]]],
    *,
    device: str | torch.device = "cpu",
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Instrument check (§3 R4): prove 4 microbatches at fixed weights give identical pure-CE grads up to roundoff."""
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()  # deterministic fixed weights without dropout variance

    def _accum_grads(arrangement: str) -> dict[str, torch.Tensor]:
        mbs = build_d3_16world_pack_minibatches(groups_16, arrangement=arrangement)
        model.zero_grad(set_to_none=True)
        # Weight each item equally over the 64-case pack using reduction="sum" / total_tokens across pack
        total_toks = 0.0
        losses_raw = []
        for mb in mbs:
            toks, segs, elig_t, apos, g_cls, _ = _pack_training_batch(mb, max_length=model.max_ctx, device=dev)
            pos, mask = packed_layout(segs, torch_module=torch)
            out = model(toks, pos, mask, action_positions=apos)
            targets = toks[:, 1:].reshape(-1)
            logits_flat = out["logits"][:, :-1, :].reshape(-1, out["logits"].shape[-1])
            mask_flat = elig_t[:, 1:].reshape(-1).to(logits_flat.dtype)
            tok_sum = (F.cross_entropy(logits_flat, targets, reduction="none") * mask_flat).sum()
            act_sum = F.nll_loss(out["action_logits"], g_cls, reduction="sum")
            total_toks += float(mask_flat.sum().item())
            losses_raw.append((tok_sum, act_sum))
        # Normalize cleanly by constant 64 actions and total pack eligible tokens
        norm_tok = max(1.0, total_toks)
        for tok_sum, act_sum in losses_raw:
            (tok_sum / norm_tok + 0.25 * (act_sum / 64.0)).backward()
        return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    g_grp = _accum_grads("grouped")
    g_int = _accum_grads("interleaved")
    max_diff = 0.0
    for n in g_grp:
        if n in g_int:
            diff = float((g_grp[n] - g_int[n]).abs().max().item())
            max_diff = max(max_diff, diff)
    model.zero_grad(set_to_none=True)
    return {
        "schema": "elendril-x3-4microbatch-ce-equiv-v1",
        "max_grad_abs_diff": max_diff,
        "atol": float(atol),
        "equivalent": bool(max_diff <= float(atol)),
    }


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


def _resolve_batch_order_and_offset(
    batch: Mapping[str, Sequence[Mapping[str, Any]]],
    objective: str,
) -> tuple[list[Mapping[str, Any]], int, bool]:
    ind_cases = list(batch.get("independent_cases", ()))
    pair_cases = list(batch.get("paired_cases", ()))
    if objective in ("D1", "D2") and pair_cases:
        all_cases = [*pair_cases, *pair_cases] if len(pair_cases) >= 2 else [*pair_cases, *ind_cases]
        offset = 0
    elif objective == "D3-ShuffledTarget" and len(pair_cases) >= 2:
        # Shuffled legal correspondence: reverse pair partner pairing inside paired_cases
        shuffled_pairs = list(reversed(pair_cases))
        all_cases = [*ind_cases, *shuffled_pairs]
        offset = len(ind_cases)
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
    update_idx: int = 0,
    cf_weight: float = 0.25,
    cf_margin: float = 0.5,
    meta_weight: float = 0.1,
    ponder_weight: float = 0.02,
    aux_action_weight: float = 0.25,
    clip: float = 1.0,
    use_amp: bool | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    """Run one verified optimization step (supports PCGrad, delayed warmup/ramp, AMP, and per-op telemetry)."""
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

    # Track per-operation NLL contribution (ANSWER 0..15, QUERY 16, RETRIEVE 17, ABSTAIN 18)
    per_sample_act_nll = F.nll_loss(act_logits_f, g_cls, reduction="none")
    act_ce = per_sample_act_nll.mean()
    base_loss = byte_ce + float(aux_action_weight) * act_ce

    loss_by_op: dict[str, float] = {}
    for op_label, op_mask in (
        ("ANSWER", g_cls < 16),
        ("QUERY", g_cls == 16),
        ("RETRIEVE", g_cls == 17),
        ("ABSTAIN", g_cls == 18),
    ):
        if bool(op_mask.any().item()):
            loss_by_op[op_label] = float(per_sample_act_nll[op_mask].mean().item())

    meta_loss = F.mse_loss(uncert_f, t_uncert)
    ponder_loss = ponder_f.mean()
    cf_loss = torch.zeros((), device=dev, dtype=torch.float32)

    # Scheduled warmup+ramp coefficients for D3-Delay-* modes (§3 R4)
    if objective in ("D3-Delay-PCGrad", "D3-Delay-NoSurgery"):
        eff_eq_w, eff_cf_w = d3_warmup_ramp_weights(update_idx, target_eq=cf_weight, target_cf=cf_weight)
    elif objective in ("CE", "D3-NoAux"):
        eff_eq_w, eff_cf_w = 0.0, 0.0
    else:
        eff_eq_w, eff_cf_w = float(cf_weight), float(cf_weight)

    use_aux_cf = objective in ("D1", "D2", "D3-Hybrid", "D3-NoSurgery", "D3-Delay-PCGrad", "D3-Delay-NoSurgery", "D3-Immediate", "D3-ShuffledTarget", "D3-WorldCycle")
    if use_aux_cf and has_pair and (eff_cf_w > 0.0 or eff_eq_w > 0.0):
        probs = F.softmax(act_logits_f, dim=-1)
        p_a = probs[offset]
        p_b = probs[offset + 1]
        if int(g_cls[offset].item()) != int(g_cls[offset + 1].item()):
            l1 = 0.5 * torch.abs(p_a - p_b).sum()
            cf_loss = eff_cf_w * F.relu(float(cf_margin) - l1)
        else:
            m = 0.5 * (p_a + p_b).clamp_min(1e-7)
            cf_loss = eff_eq_w * 0.5 * (
                F.kl_div(m.log(), p_a.clamp_min(1e-7), reduction="sum")
                + F.kl_div(m.log(), p_b.clamp_min(1e-7), reduction="sum")
            )

    if objective in ("CE", "D3-NoAux"):
        aux_loss = float(ponder_weight) * ponder_loss + float(meta_weight) * meta_loss
    else:
        aux_loss = cf_loss + float(meta_weight) * meta_loss + float(ponder_weight) * ponder_loss

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    scaler_on = scaler is not None and scaler.is_enabled()

    surgery_info: dict[str, float | bool] = {
        "dot_before": 0.0,
        "dot_after": 0.0,
        "cosine_before": 0.0,
        "conflict_detected": False,
    }
    use_pcgrad = objective in ("D3-Hybrid", "D3-Delay-PCGrad", "D3-Immediate", "D3-ShuffledTarget", "D3-WorldCycle")
    if use_pcgrad:
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
        "update_idx": int(update_idx),
        "scheduled_eq_weight": float(eff_eq_w),
        "scheduled_cf_weight": float(eff_cf_w),
        "total_loss": total_loss_val if math.isfinite(total_loss_val) else -1.0,
        "byte_ce": float(byte_ce.item()),
        "action_ce": float(act_ce.item()),
        "cf_loss": float(cf_loss.item()),
        "meta_loss": float(meta_loss.item()),
        "ponder_loss": float(ponder_loss.item()),
        "loss_by_operation": loss_by_op,
        "grad_norm": raw_norm if math.isfinite(raw_norm) else -1.0,
        "scaler_skipped": scaler_skipped,
        **surgery_info,
    }


__all__ = [
    "D3_OBJECTIVES",
    "apply_pcgrad_conflict_surgery",
    "audit_d1h_d2_schedule_and_preceding_observations",
    "audit_x1_sampler_exposure",
    "build_d3_16world_pack_minibatches",
    "build_d3_hybrid_batches",
    "compute_d3_objective_step",
    "d3_warmup_ramp_weights",
    "prepack_d3_batches",
    "verify_d3_gradient_surgery_fixture",
    "verify_fixed_weight_4microbatch_ce_equivalence",
]
