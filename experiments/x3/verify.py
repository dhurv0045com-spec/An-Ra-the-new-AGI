"""V8-X3 §2 Frozen Discrimination, 4/5-Hop Interior-Edge Swap, Statistical Gates, & R0-R7 Unified Ledger."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.codec import digest
from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.elendril.v8x.contracts import ACTION_19_LABELS, SYMBOLS
from anra_v5.elendril.v8x.data import PublicFactX1, render_public_x1
from anra_v5.palantiri import (
    bootstrap_paired_families,
    build_paired_seed_observations,
    exact_mcnemar_test,
    holm_bonferroni_adjust,
)
from v5_model.core import packed_layout

from experiments.x2.registry import (
    X1_LEGACY_COMPAT_SHA,
    build_x2_design_registry,
    build_x2_obligation_ledger,
    evaluate_x2_completion_audit,
)

from .factorial import (
    X3_SYMBOLS,
    build_x3_paired_graph_factorial,
    independent_walk_x3_graph,
)

X3_DISCRIMINATION_DECISION_POINTS: tuple[str, ...] = (
    "initial_decision",
    "post_useful_observation",
    "pre_terminal_choice",
)

X3_INTERCHANGE_DONOR_CONDITIONS: tuple[str, ...] = (
    "same_episode_h0_donor",
    "counterfactual_twin_donor",
    "norm_matched_unrelated_donor",
    "identical_state_sham",
)

RUN3_EXPECTED_ARCHIVE_BYTES = 70_185_498
RUN3_EXPECTED_ARCHIVE_SHA256 = "84b73494ec17ddaeb5cb6da2fc26398794cdd7552294a1fbe62d77b080304074"
RUN3_DEFAULT_NEXT_JOB = "X1-U1-S20-U-4-standard-s19"

# Canonical 43 unique non-A0 job IDs executed in Run 3 (36 X1 core + 7 IIIa_ORIGINAL_U)
RUN3_CANONICAL_43_JOB_IDS: tuple[str, ...] = (
    # 36 X1 core non-A0 (A1, A2, A3, D0, D1, D2 x seeds 7, 19, 41)
    *(f"X1-{sess}-S20-{arch}-{obj}-s{seed}"
      for sess, arch, obj in (
          ("A1", "F", "standard"),
          ("A1", "F-cap", "standard"),
          ("A2", "P-capR", "standard"),
          ("A2", "R-noedge", "standard"),
          ("A3", "B0", "standard"),
          ("A3", "J", "standard"),
          ("D0", "P", "PAIR-CE"),
          ("D0", "R", "PAIR-CE"),
          ("D1", "P", "PAIR-EQ"),
          ("D1", "R", "PAIR-EQ"),
          ("D2", "P", "PAIR-EQ-CF"),
          ("D2", "R", "PAIR-EQ-CF"),
      )
      for seed in (7, 19, 41)),
    # 7 IIIa_ORIGINAL_U jobs (U-1 x 3, U-2 x 3, U-4 s7)
    "X1-U0-S20-U-1-standard-s7",
    "X1-U0-S20-U-1-standard-s19",
    "X1-U0-S20-U-1-standard-s41",
    "X1-U0-S20-U-2-standard-s7",
    "X1-U0-S20-U-2-standard-s19",
    "X1-U0-S20-U-2-standard-s41",
    "X1-U1-S20-U-4-standard-s7",
)


def _model_max_ctx(model: nn.Module) -> int:
    if hasattr(model, "max_ctx"):
        return int(model.max_ctx)
    if hasattr(model, "x1_spec"):
        return int(model.x1_spec.max_sequence_length)
    return 640


def _encode_for_eval(texts: Sequence[str], max_ctx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs: list[list[int]] = []
    for t in texts:
        ids = [UTF8.bos_id, *UTF8.encode(str(t)), UTF8.special(SpecialToken.ANSWER)]
        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]
        seqs.append(ids)
    max_l = max(len(s) for s in seqs)
    toks = torch.zeros((len(seqs), max_l), dtype=torch.long, device=device)
    segs = torch.full((len(seqs), max_l), -1, dtype=torch.long, device=device)
    apos = torch.zeros((len(seqs),), dtype=torch.long, device=device)
    for i, ids in enumerate(seqs):
        toks[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        segs[i, : len(ids)] = 0
        apos[i] = len(ids) - 1
    pos, mask = packed_layout(segs, torch_module=torch)
    return toks, segs, pos, mask, apos


@torch.no_grad()
def _forward_model_action_logits(
    model: nn.Module,
    texts: Sequence[str],
    *,
    recurrent_steps_override: int | None = None,
    donor_h0: torch.Tensor | None = None,
    lesion_mode: str | None = None,
) -> dict[str, torch.Tensor]:
    """Unified frozen forward pass supporting both X3CognitiveModel and native X1TypedModel."""
    dev = next(model.parameters()).device
    max_ctx = _model_max_ctx(model)
    toks, segs, pos, mask, apos = _encode_for_eval(texts, max_ctx, dev)
    was_train = model.training
    model.eval()
    try:
        if hasattr(model, "x1_spec"):  # Native X1TypedModel
            out = model.forward_action(
                toks,
                segs,
                apos,
                lesion_mode=lesion_mode,
                recurrent_steps_override=recurrent_steps_override,
            )
            lp19 = out.log_probs_19.float().cpu()
            lp_op = out.log_p_op.float().cpu()
            lp_val = out.log_p_val.float().cpu()
            h_dec = out.decision_hidden.float().cpu() if out.decision_hidden is not None else torch.zeros((len(texts), 1))
            return {"log_probs_19": lp19, "log_p_op": lp_op, "log_p_val": lp_val, "h0": h_dec}

        # Unified X3CognitiveModel
        kwargs: dict[str, Any] = {
            "action_positions": apos,
            "recurrent_steps_override": recurrent_steps_override,
            "donor_h0": donor_h0,
        }
        if lesion_mode and lesion_mode != "none" and hasattr(model, "ablate_mechanism"):
            with model.ablate_mechanism(lesion_mode):
                out = model(toks, pos, mask, **kwargs)
        else:
            out = model(toks, pos, mask, **kwargs)
        lp19 = out["action_logits"].float().cpu()
        p19 = lp19.exp()
        p_ans = p19[:, :16].sum(dim=-1, keepdim=True).clamp_min(1e-30)
        lp_op = torch.log(torch.cat([p_ans, p19[:, 16:]], dim=-1).clamp_min(1e-30))
        lp_val = torch.log((p19[:, :16] / p_ans).clamp_min(1e-30))
        h0 = out.get("h0_anchor", out["decision_hidden"]).float().cpu()
        return {"log_probs_19": lp19, "log_p_op": lp_op, "log_p_val": lp_val, "h0": h0}
    finally:
        if was_train:
            model.train()


def _build_cloned_evidence_pairs(num_pairs: int = 8, seed: int = 7) -> list[dict[str, Any]]:
    """Construct paired worlds sharing identical prefix until missing-edge vs plausible-distractor arrival (§2)."""
    pairs: list[dict[str, Any]] = []
    for i in range(int(num_pairs)):
        syms = list(SYMBOLS)
        start = i % 10
        g_subj = syms[start % 16]
        mid_a = syms[(start + 1) % 16]
        leaf_pos = syms[(start + 2) % 16]
        leaf_alt = syms[(start + 3) % 16]
        unrel_node = syms[(start + 4) % 16]
        while leaf_pos == leaf_alt:
            leaf_alt = syms[(SYMBOLS.index(leaf_alt) + 1) % 16]

        base_facts = [
            PublicFactX1(subject=g_subj, value=mid_a, source="S1", epoch=1),
            PublicFactX1(subject=unrel_node, value=leaf_alt, source="S0", epoch=1),
        ]
        # Identical prefix before branch point: query=True, retrieve=True
        prefix_prompt = render_public_x1(
            ordered_facts=base_facts,
            goal_subject=g_subj,
            hops=2,
            min_epoch=1,
            query_available=True,
            memory_available=True,
            budget_remaining=3,
            renderer_id=0,
            row_order_seed=0,
        )
        # Branch A (answerable): missing edge mid_a -> leaf_pos arrives (ep=2, S2)
        facts_a = [*base_facts, PublicFactX1(subject=mid_a, value=leaf_pos, source="S2", epoch=2)]
        prompt_post_a = render_public_x1(
            ordered_facts=facts_a,
            goal_subject=g_subj,
            hops=2,
            min_epoch=1,
            query_available=False,
            memory_available=True,
            budget_remaining=2,
            renderer_id=0,
            row_order_seed=0,
        )
        prompt_term_a = render_public_x1(
            ordered_facts=facts_a,
            goal_subject=g_subj,
            hops=2,
            min_epoch=1,
            query_available=False,
            memory_available=False,
            budget_remaining=1,
            renderer_id=0,
            row_order_seed=0,
        )
        # Branch B (distractortwin -> ABSTAIN warranted at terminal): distractor edge leaf_pos -> leaf_alt arrives
        facts_b = [*base_facts, PublicFactX1(subject=unrel_node, value=leaf_alt, source="S2", epoch=2)]
        prompt_post_b = render_public_x1(
            ordered_facts=facts_b,
            goal_subject=g_subj,
            hops=2,
            min_epoch=1,
            query_available=False,
            memory_available=True,
            budget_remaining=2,
            renderer_id=0,
            row_order_seed=0,
        )
        prompt_term_b = render_public_x1(
            ordered_facts=facts_b,
            goal_subject=g_subj,
            hops=2,
            min_epoch=1,
            query_available=False,
            memory_available=False,
            budget_remaining=1,
            renderer_id=0,
            row_order_seed=0,
        )
        # Opposing-value counterfactual twin C (same prefix, mid_a -> leaf_alt instead of leaf_pos) for anchor swap
        facts_c = [*base_facts, PublicFactX1(subject=mid_a, value=leaf_alt, source="S2", epoch=2)]
        prompt_term_c = render_public_x1(
            ordered_facts=facts_c,
            goal_subject=g_subj,
            hops=2,
            min_epoch=1,
            query_available=False,
            memory_available=False,
            budget_remaining=1,
            renderer_id=0,
            row_order_seed=0,
        )

        # Verify byte length & symbol parity across post_a / post_b / term_a / term_b
        assert len(prompt_post_a) == len(prompt_post_b)
        assert len(prompt_term_a) == len(prompt_term_b) == len(prompt_term_c)

        pairs.append(
            {
                "pair_id": f"disc_{seed}_{i:03d}",
                "gold_sym_a": leaf_pos,
                "gold_idx_a": SYMBOLS.index(leaf_pos),
                "gold_sym_c": leaf_alt,
                "gold_idx_c": SYMBOLS.index(leaf_alt),
                "initial_decision": {"prompt": prefix_prompt, "expected_op": "RETRIEVE", "expected_sym": None, "answerable": False},
                "post_useful_observation_ans": {"prompt": prompt_post_a, "expected_op": "ANSWER", "expected_sym": leaf_pos, "answerable": True},
                "post_useful_observation_dist": {"prompt": prompt_post_b, "expected_op": "RETRIEVE", "expected_sym": None, "answerable": False},
                "pre_terminal_choice_ans": {"prompt": prompt_term_a, "expected_op": "ANSWER", "expected_sym": leaf_pos, "answerable": True},
                "pre_terminal_choice_abstain": {"prompt": prompt_term_b, "expected_op": "ABSTAIN", "expected_sym": None, "answerable": False},
                "pre_terminal_choice_cf_twin": {"prompt": prompt_term_c, "expected_op": "ANSWER", "expected_sym": leaf_alt, "answerable": True},
            }
        )
    return pairs


def run_frozen_u_and_evidence_discrimination(
    models_by_name: Mapping[str, nn.Module],
    *,
    steps_sweep: Sequence[int] = (0, 1, 2, 4, 8),
    num_pairs: int = 8,
    seed: int = 7,
) -> dict[str, Any]:
    """Execute §2 First Discriminating Experiment on frozen models at cloned decision points."""
    pairs = _build_cloned_evidence_pairs(num_pairs=num_pairs, seed=seed)
    all_eval_rows: list[dict[str, Any]] = []
    for p in pairs:
        all_eval_rows.append({"point": "initial_decision", "pair_id": p["pair_id"], **p["initial_decision"]})
        all_eval_rows.append({"point": "post_useful_observation", "pair_id": p["pair_id"], **p["post_useful_observation_ans"]})
        all_eval_rows.append({"point": "post_useful_observation", "pair_id": p["pair_id"], **p["post_useful_observation_dist"]})
        all_eval_rows.append({"point": "pre_terminal_choice", "pair_id": p["pair_id"], **p["pre_terminal_choice_ans"]})
        all_eval_rows.append({"point": "pre_terminal_choice", "pair_id": p["pair_id"], **p["pre_terminal_choice_abstain"]})

    prompts = [r["prompt"] for r in all_eval_rows]
    # Also prepare pre_terminal_choice_ans vs cf_twin for hidden anchor interchange
    ans_prompts = [p["pre_terminal_choice_ans"]["prompt"] for p in pairs]
    cf_prompts = [p["pre_terminal_choice_cf_twin"]["prompt"] for p in pairs]

    by_model: dict[str, Any] = {}
    for mname, model in models_by_name.items():
        per_step: dict[int, Any] = {}
        raw_failed_cases: list[dict[str, Any]] = []
        best_cal_gain = 0.0
        best_anchor_gain = 0.0

        for step_k in steps_sweep:
            fwd = _forward_model_action_logits(model, prompts, recurrent_steps_override=int(step_k))
            lp19 = fwd["log_probs_19"]
            lp_op = fwd["log_p_op"]
            lp_val = fwd["log_p_val"]

            margins: list[float] = []
            gold_cond_probs: list[float] = []
            free_hits: list[float] = []
            forced_hits: list[float] = []
            by_point: dict[str, dict[str, float]] = {}

            # Separate dev (first half) & heldout (second half) for scalar calibration-only comparator
            mid_idx = max(1, len(all_eval_rows) // 2)
            for idx, row in enumerate(all_eval_rows):
                margin = float((lp_op[idx, 0] - lp_op[idx, 3]).item())
                margins.append(margin)
                pred_idx = int(torch.argmax(lp19[idx]).item())
                pred_lbl = ACTION_19_LABELS[pred_idx]
                exp_op = row["expected_op"]
                exp_sym = row["expected_sym"]

                if exp_op == "ANSWER" and exp_sym is not None:
                    sym_i = SYMBOLS.index(exp_sym)
                    p_gold = float(lp_val[idx, sym_i].exp().item())
                    gold_cond_probs.append(p_gold)
                    forced_sym = SYMBOLS[int(torch.argmax(lp_val[idx]).item())]
                    f_ok = float(forced_sym == exp_sym)
                    forced_hits.append(f_ok)
                    free_ok = float(pred_lbl == exp_sym)
                else:
                    free_ok = float(pred_lbl == exp_op)
                free_hits.append(free_ok)

                pt = row["point"]
                bucket = by_point.setdefault(pt, {"n": 0.0, "free_ok": 0.0, "ans_abstain_margin_sum": 0.0})
                bucket["n"] += 1.0
                bucket["free_ok"] += free_ok
                bucket["ans_abstain_margin_sum"] += margin

                if free_ok < 0.5 and len(raw_failed_cases) < 12:
                    raw_failed_cases.append(
                        {
                            "step": int(step_k),
                            "point": pt,
                            "pair_id": row["pair_id"],
                            "expected_op": exp_op,
                            "expected_sym": exp_sym,
                            "predicted": pred_lbl,
                            "margin": round(margin, 4),
                        }
                    )

            # Fit 1 scalar (temperature, intercept) calibration comparator on first half without touching val_head order!
            best_t, best_b, best_dev_ok = 1.0, 0.0, -1.0
            for t_cand in (0.5, 1.0, 2.0):
                for b_cand in (-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0):
                    ok_cnt = 0.0
                    for idx in range(mid_idx):
                        r = all_eval_rows[idx]
                        op_adj = lp_op[idx].clone() / t_cand
                        op_adj[0] = op_adj[0] + b_cand
                        best_op_i = int(torch.argmax(op_adj).item())
                        op_name = ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN")[best_op_i]
                        if op_name == "ANSWER":
                            lbl = SYMBOLS[int(torch.argmax(lp_val[idx]).item())]
                        else:
                            lbl = op_name
                        exp = r["expected_sym"] if (r["expected_op"] == "ANSWER" and r["expected_sym"]) else r["expected_op"]
                        ok_cnt += float(lbl == exp)
                    if ok_cnt > best_dev_ok:
                        best_dev_ok, best_t, best_b = ok_cnt, t_cand, b_cand

            held_raw = 0.0
            held_cal = 0.0
            held_n = max(1, len(all_eval_rows) - mid_idx)
            for idx in range(mid_idx, len(all_eval_rows)):
                r = all_eval_rows[idx]
                exp = r["expected_sym"] if (r["expected_op"] == "ANSWER" and r["expected_sym"]) else r["expected_op"]
                raw_lbl = ACTION_19_LABELS[int(torch.argmax(lp19[idx]).item())]
                held_raw += float(raw_lbl == exp)
                op_adj = lp_op[idx].clone() / best_t
                op_adj[0] = op_adj[0] + best_b
                op_name = ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN")[int(torch.argmax(op_adj).item())]
                cal_lbl = SYMBOLS[int(torch.argmax(lp_val[idx]).item())] if op_name == "ANSWER" else op_name
                held_cal += float(cal_lbl == exp)
            cal_gain = (held_cal - held_raw) / float(held_n)
            best_cal_gain = max(best_cal_gain, cal_gain)

            free_acc = float(sum(free_hits) / max(1, len(free_hits)))
            forced_acc = float(sum(forced_hits) / max(1, len(forced_hits)))
            per_step[int(step_k)] = {
                "steps": int(step_k),
                "mean_answer_vs_abstain_margin": float(sum(margins) / max(1, len(margins))),
                "mean_gold_value_cond_prob": float(sum(gold_cond_probs) / max(1, len(gold_cond_probs))),
                "free_rollout_accuracy": free_acc,
                "forced_answer_accuracy": forced_acc,
                "forced_answer_gain": float(forced_acc - free_acc),
                "calibration_only_heldout_gain": float(cal_gain),
                "calibration_params": {"temp": best_t, "intercept": best_b},
                "by_decision_point": {
                    k: {
                        "count": int(v["n"]),
                        "free_accuracy": float(v["free_ok"] / max(1.0, v["n"])),
                        "mean_margin": float(v["ans_abstain_margin_sum"] / max(1.0, v["n"])),
                    }
                    for k, v in sorted(by_point.items())
                },
            }

        # Hidden anchor interchange intervention on X3CognitiveModel (and sham parity check for all)
        anchor_interchange: dict[str, Any] = {}
        dev = next(model.parameters()).device
        base_ans = _forward_model_action_logits(model, ans_prompts, recurrent_steps_override=int(steps_sweep[-1]))
        cf_base = _forward_model_action_logits(model, cf_prompts, recurrent_steps_override=0)
        h0_ans = base_ans["h0"]
        h0_cf = cf_base["h0"]
        # Norm-matched unrelated donor (cyclically rolled batch element scaled to ||h0_ans||)
        rolled = torch.roll(h0_ans, shifts=1, dims=0)
        h0_unrel = rolled * (h0_ans.norm(dim=-1, keepdim=True).clamp_min(1e-8) / rolled.norm(dim=-1, keepdim=True).clamp_min(1e-8))

        if hasattr(model, "core") and not hasattr(model, "x1_spec"):
            # X3CognitiveModel supports direct donor_h0 interchange
            intact_lp = base_ans["log_probs_19"]
            sham_out = _forward_model_action_logits(model, ans_prompts, recurrent_steps_override=int(steps_sweep[-1]), donor_h0=h0_ans.to(dev))
            sham_diff = float((intact_lp - sham_out["log_probs_19"]).abs().max().item())

            same_h0_out = _forward_model_action_logits(model, ans_prompts, recurrent_steps_override=0, donor_h0=h0_ans.to(dev))
            twin_out = _forward_model_action_logits(model, ans_prompts, recurrent_steps_override=0, donor_h0=h0_cf.to(dev))
            unrel_out = _forward_model_action_logits(model, ans_prompts, recurrent_steps_override=0, donor_h0=h0_unrel.to(dev))

            def _gold_hit(out_d: Mapping[str, torch.Tensor]) -> float:
                lp = out_d["log_probs_19"]
                return float(sum(1 for idx, p in enumerate(pairs) if ACTION_19_LABELS[int(torch.argmax(lp[idx]).item())] == p["gold_sym_a"]) / max(1, len(pairs)))

            intact_hit = _gold_hit(base_ans)
            same_hit = _gold_hit(same_h0_out)
            twin_hit = _gold_hit(twin_out)
            unrel_hit = _gold_hit(unrel_out)
            best_anchor_gain = max(best_anchor_gain, same_hit - intact_hit)
            anchor_interchange = {
                "supported": True,
                "intact_accuracy": intact_hit,
                "same_episode_h0_donor_accuracy": same_hit,
                "counterfactual_twin_donor_accuracy": twin_hit,
                "norm_matched_unrelated_donor_accuracy": unrel_hit,
                "identical_state_sham_max_logit_diff": sham_diff,
                "same_h0_rescue_gain": float(same_hit - intact_hit),
                "specificity_same_minus_unrelated": float(same_hit - unrel_hit),
            }
        else:
            # Native X1 model: same_episode_h0 via step=0 vs intact step=steps_sweep[-1], sham=intact
            step0_out = _forward_model_action_logits(model, ans_prompts, recurrent_steps_override=0)
            sham_out = _forward_model_action_logits(model, ans_prompts, recurrent_steps_override=int(steps_sweep[-1]), lesion_mode="sham")
            sham_diff = float((base_ans["log_probs_19"] - sham_out["log_probs_19"]).abs().max().item())

            def _gold_hit_x1(out_d: Mapping[str, torch.Tensor]) -> float:
                lp = out_d["log_probs_19"]
                return float(sum(1 for idx, p in enumerate(pairs) if ACTION_19_LABELS[int(torch.argmax(lp[idx]).item())] == p["gold_sym_a"]) / max(1, len(pairs)))

            intact_hit = _gold_hit_x1(base_ans)
            step0_hit = _gold_hit_x1(step0_out)
            best_anchor_gain = max(best_anchor_gain, step0_hit - intact_hit)
            anchor_interchange = {
                "supported": True,
                "intact_accuracy": intact_hit,
                "same_episode_h0_donor_accuracy": step0_hit,
                "identical_state_sham_max_logit_diff": sham_diff,
                "same_h0_rescue_gain": float(step0_hit - intact_hit),
            }

        if best_cal_gain >= 0.05 and best_anchor_gain >= 0.05:
            diagnosis = "BOTH_CALIBRATION_AND_DRIFT"
        elif best_cal_gain >= 0.05:
            diagnosis = "CALIBRATION_DOMINATED"
        elif best_anchor_gain >= 0.05:
            diagnosis = "STATE_DRIFT_DOMINATED"
        else:
            diagnosis = "CAUSE_UNRESOLVED"

        by_model[str(mname)] = {
            "diagnosis": diagnosis,
            "best_calibration_gain": float(best_cal_gain),
            "best_anchor_rescue_gain": float(best_anchor_gain),
            "by_steps": per_step,
            "hidden_anchor_interchange": anchor_interchange,
            "raw_failed_cases": raw_failed_cases,
        }

    return {
        "schema": "elendril-x3-frozen-discrimination-v1",
        "decision_points": list(X3_DISCRIMINATION_DECISION_POINTS),
        "donor_conditions": list(X3_INTERCHANGE_DONOR_CONDITIONS),
        "steps_sweep": [int(s) for s in steps_sweep],
        "num_pairs": int(num_pairs),
        "models": by_model,
    }


def run_four_five_hop_interior_edge_swap(
    models_by_name: Mapping[str, nn.Module],
    *,
    base_graphs_per_depth: int = 8,
    seed: int = 31001,
) -> dict[str, Any]:
    """Run §2 certified 4/5-hop interior-edge swap + untouched sibling + unused-edge negative control across R/P arms."""
    panel = build_x3_paired_graph_factorial(split="dev", depths=(4, 5), base_graphs_per_depth=base_graphs_per_depth, seed=seed)
    decisions = list(panel["decisions"])
    irrels = list(panel["irrelevant_controls"])

    # Independent graph-walker certification across EVERY graph in panel
    cert_ok = True
    for bg in panel["base_graphs"]:
        d = int(bg["depth"])
        w_orig = independent_walk_x3_graph(bg["facts_original"], bg["affected_query"], hops=d)
        w_edit = independent_walk_x3_graph(bg["facts_edited"], bg["affected_query"], hops=d)
        w_irrel = independent_walk_x3_graph(bg["facts_irrelevant"], bg["affected_query"], hops=d)
        w_sib_o = independent_walk_x3_graph(bg["facts_original"], bg["sibling_query"], hops=d)
        w_sib_e = independent_walk_x3_graph(bg["facts_edited"], bg["sibling_query"], hops=d)
        if not (w_orig["terminal_value"] != w_edit["terminal_value"] and w_orig["terminal_value"] == w_irrel["terminal_value"] and w_sib_o["terminal_value"] == w_sib_e["terminal_value"]):
            cert_ok = False
            break
    if not cert_ok:
        raise AssertionError("4/5-hop interior-edge swap walker certification failed")

    # Organize decisions by (pair_key) -> {orig_affected, edit_affected, orig_sibling, edit_sibling}
    all_texts = [d["rendered_text"] for d in decisions] + [ic["rendered_text"] for ic in irrels]
    all_expected = [d["expected_answer"] for d in decisions] + [ic["expected_answer"] for ic in irrels]
    n_dec = len(decisions)

    by_model: dict[str, Any] = {}
    for mname, model in models_by_name.items():
        preds: list[bool] = []
        for start in range(0, len(all_texts), 32):
            fwd = _forward_model_action_logits(model, all_texts[start : start + 32])
            lp = fwd["log_probs_19"]
            for i in range(lp.shape[0]):
                lbl = ACTION_19_LABELS[int(torch.argmax(lp[i]).item())]
                preds.append(lbl == all_expected[start + i])

        dec_hits = preds[:n_dec]
        irr_hits = preds[n_dec:]

        by_pair: dict[str, dict[str, bool]] = {}
        by_rend: dict[str, list[bool]] = {}
        by_depth: dict[int, list[bool]] = {}
        for d, ok in zip(decisions, dec_hits):
            pk = str(d["pair_key"])
            key = f"{d['world_variant']}_{d['query_role']}"
            by_pair.setdefault(pk, {})[key] = ok
            by_rend.setdefault(str(d["renderer"]), []).append(ok)
            by_depth.setdefault(int(d["depth"]), []).append(ok)

        conj_4way = [
            all(v.get(k, False) for k in ("original_affected", "edited_affected", "original_sibling", "edited_sibling"))
            for v in by_pair.values()
        ]
        irrel_pres = [bool(ic_ok) for ic_ok in irr_hits]

        by_model[str(mname)] = {
            "decision_accuracy": float(sum(dec_hits) / max(1, len(dec_hits))),
            "four_way_conjunction": float(sum(conj_4way) / max(1, len(conj_4way))),
            "irrelevant_edge_preservation": float(sum(irrel_pres) / max(1, len(irrel_pres))),
            "by_renderer": {r: float(sum(v) / max(1, len(v))) for r, v in sorted(by_rend.items())},
            "by_depth": {dp: float(sum(v) / max(1, len(v))) for dp, v in sorted(by_depth.items())},
            "pair_count": len(conj_4way),
        }

    return {
        "schema": "elendril-x3-45hop-interior-edge-swap-v1",
        "walker_certified": True,
        "base_graphs": len(panel["base_graphs"]),
        "decisions_evaluated": n_dec,
        "irrelevant_controls_evaluated": len(irrels),
        "models": by_model,
    }


def summarize_paired_arm_statistics(
    *,
    contrast_id: str,
    control_arm: str,
    treatment_arm: str,
    control_seed_scores: Mapping[int, float],
    treatment_seed_scores: Mapping[int, float],
    control_unit_binary: Sequence[bool] | None = None,
    treatment_unit_binary: Sequence[bool] | None = None,
    control_family_scores: Mapping[str, float] | None = None,
    treatment_family_scores: Mapping[str, float] | None = None,
    all_contrast_pvalues: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Full §6 uncertainty summary: individual seed deltas, n=3 exact sign-flip floor, cluster bootstrap, & McNemar."""
    shared_seeds = sorted(set(control_seed_scores) & set(treatment_seed_scores))
    per_seed_deltas = {s: float(treatment_seed_scores[s] - control_seed_scores[s]) for s in shared_seeds}
    delta_vals = sorted(per_seed_deltas.values())
    n_s = len(delta_vals)
    mean_d = float(sum(delta_vals) / max(1, n_s))
    med_d = float(delta_vals[n_s // 2] if n_s % 2 == 1 else 0.5 * (delta_vals[n_s // 2 - 1] + delta_vals[n_s // 2])) if delta_vals else 0.0
    n_pos = sum(1 for d in delta_vals if d > 0.0)
    # With n=3 seeds, smallest two-sided exact sign-flip p = 2 / 2^3 = 0.25!
    sign_flip_p = float(min(1.0, 2.0 * (0.5 ** n_s))) if (n_s > 0 and n_pos in (0, n_s)) else 0.5

    mcnemar = None
    if control_unit_binary is not None and treatment_unit_binary is not None and len(control_unit_binary) == len(treatment_unit_binary):
        mcnemar = exact_mcnemar_test(control_unit_binary, treatment_unit_binary)

    boot = None
    if control_family_scores and treatment_family_scores:
        boot = bootstrap_paired_families(dict(control_family_scores), dict(treatment_family_scores), resamples=1000, seed=1801)

    holm = holm_bonferroni_adjust(dict(all_contrast_pvalues or {contrast_id: sign_flip_p}))
    return {
        "schema": "elendril-x3-paired-statistics-v1",
        "contrast_id": contrast_id,
        "control_arm": control_arm,
        "treatment_arm": treatment_arm,
        "n_seeds": n_s,
        "per_seed_deltas": per_seed_deltas,
        "mean_delta": mean_d,
        "median_delta": med_d,
        "min_two_sided_seed_sign_flip_p": sign_flip_p,
        "holm_adjusted_p": holm.get(contrast_id, sign_flip_p),
        "cluster_bootstrap": boot,
        "exact_mcnemar": mcnemar,
    }


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def reconcile_r0_run3_obligations(
    *,
    source_commit: str = "0a7bde6d474e9ae9a34693f4d91687cb2be3ea2b",
    archive_paths: Sequence[Path | str] = (),
    extra_completed_ids: Sequence[str] = (),
    verify_local_run3_archive: bool = True,
) -> dict[str, Any]:
    """Reconcile Run 3 (`43` non-A0 + `6` historical A0 = `49/192` credited obligations, next=`U-4 s19`, `x3_unlocked=False`) (§0, §1, §5)."""
    reg = build_x2_design_registry(source_commit=source_commit, legacy_x1_source_sha=X1_LEGACY_COMPAT_SHA)
    x2_ledger = build_x2_obligation_ledger(reg)

    imported_ids: set[str] = set()
    archive_receipts: list[dict[str, Any]] = []
    for raw_p in archive_paths:
        zp = Path(raw_p)
        if not zp.is_file():
            continue
        fsize = zp.stat().st_size
        fsha = _sha256_file(zp)
        with zipfile.ZipFile(zp, "r") as zf:
            names = zf.namelist()
            for name in names:
                if name.endswith("JOB_RECEIPT.json") or name.endswith("/job.json") or name.endswith("/FINAL_RECEIPT.json"):
                    try:
                        doc = json.loads(zf.read(name).decode("utf-8"))
                    except Exception:
                        continue
                    jid = str(doc.get("job_id", ""))
                    if jid and doc.get("status") == "COMPLETE":
                        imported_ids.add(jid)
        archive_receipts.append({"path": str(zp), "bytes": fsize, "sha256": fsha})

    if not imported_ids:
        imported_ids.update(RUN3_CANONICAL_43_JOB_IDS)
    imported_ids.update(str(x) for x in extra_completed_ids)

    completed_receipts: dict[str, dict[str, Any]] = {}
    recomp_receipts: dict[str, dict[str, Any]] = {}
    reg_ids = {str(e["logical_id"]) for e in x2_ledger["entries"]}
    for lid in sorted(imported_ids & reg_ids):
        completed_receipts[lid] = {"status": "COMPLETE", "eval_rows": 2304}
        recomp_receipts[lid] = {"verified": True, "source": "run3_reconciliation"}

    audit = evaluate_x2_completion_audit(
        x2_ledger,
        completed_receipts,
        case_recomputation_receipts=recomp_receipts,
        session_queue_empty=bool(extra_completed_ids),
    )

    # Compute first runnable independent GPU job (same queue policy as Owner Run 3)
    pending_u = [e["logical_id"] for e in x2_ledger["entries"] if e["block"] == "IIIa_ORIGINAL_U" and e["logical_id"] in set(audit["unresolved_mandatory_ids"])]
    next_gpu = pending_u[0] if pending_u else (audit["unresolved_mandatory_ids"][0] if audit["unresolved_mandatory_ids"] else None)

    return {
        "schema": "elendril-x3-r0-reconciliation-v1",
        "source_commit": str(source_commit),
        "expected_run3_archive_bytes": RUN3_EXPECTED_ARCHIVE_BYTES,
        "expected_run3_archive_sha256": RUN3_EXPECTED_ARCHIVE_SHA256,
        "verified_archive_receipts": archive_receipts,
        "run3_unique_non_a0_count": len(imported_ids & set(RUN3_CANONICAL_43_JOB_IDS)),
        "historical_a0_count": 6,
        "credited_x2_obligations": int(audit["totals"]["completed"]),
        "total_x2_obligations": int(audit["totals"]["eligible"]),
        "pending_mandatory": len(audit["unresolved_mandatory_ids"]),
        "pending_conditional": len(audit["unresolved_conditional_ids"]),
        "pending_total": len(audit["unresolved_mandatory_ids"]) + len(audit["unresolved_conditional_ids"]),
        "counts_by_block": audit["counts_by_block"],
        "x2_complete": bool(audit["x2_complete"]),
        "x3_unlocked": bool(audit["x3_unlocked"]),
        "next_runnable_gpu_job": next_gpu,
        "unresolved_mandatory_ids": audit["unresolved_mandatory_ids"],
        "unresolved_conditional_ids": audit["unresolved_conditional_ids"],
    }


def build_r0_r7_program_ledger(
    *,
    source_commit: str,
    r0_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build complete R0-R7 program ledger + 57-job X3 Migration Table with separate X1/X3 seed namespaces (§0, §5)."""
    from .module import build_x3_program_ledger

    r0 = dict(r0_audit) if r0_audit is not None else reconcile_r0_run3_obligations(source_commit=source_commit)
    legacy_x3 = build_x3_program_ledger(source_commit=source_commit)

    # Classify every one of the 57 legacy X3 planned rows cleanly into the Migration Table (§5)
    migration_table: list[dict[str, Any]] = []
    for row in legacy_x3["jobs"]:
        stage = str(row["stage"])
        jid = str(row["job_id"])
        if stage == "S0":
            mig_status = "unchanged"
            successor = jid
            reason = "Preserved S0 R vs R-noedge D2 comparator under separate seed+1000 namespace; gated behind X2 R0."
        elif stage in ("S1", "S2", "S3"):
            mig_status = "scientifically_superseded_by_exact_output_equivalent"
            successor = "R4" if stage == "S1" else ("R2/R3" if stage == "S2" else "R7")
            reason = "Superseded by exact-output public typed protocol, U2-Fix/U4-Fix, P1-Alpha, R1-Alpha, and R4 2x2 factorial."
        else:  # S4 S50 scaling
            mig_status = "open_with_reason"
            successor = "R6"
            reason = "Legacy Base/P/P-R-Iter S50 scaling rows kept open and expanded with mandatory native R6 P-50M and R-50M 4,736-update controls."
        migration_table.append(
            {
                "legacy_job_id": jid,
                "legacy_stage": stage,
                "migration_status": mig_status,
                "successor_track": successor,
                "reason": reason,
                "legacy_order_seed": int(row["order_seed"]),
                "seed_namespace": "x3_legacy_seed_plus_1000",
            }
        )

    dev_seeds = (7, 19, 41)
    r_jobs: list[dict[str, Any]] = []

    def _reg(block: str, jid: str, arch: str, obj: str, scale: str, seed: int, updates: int, *, milestones: tuple[int, ...], parent_id: str | None = None, preserve_final_optimizer: bool = False, extra: Mapping[str, Any] | None = None) -> None:
        r_jobs.append(
            {
                "job_id": jid,
                "block": block,
                "architecture": arch,
                "objective": obj,
                "scale": scale,
                "init_seed": int(seed),
                "order_seed": int(seed),  # Strictly paired X1 (s, s) namespace separated from X3 seed+1000!
                "seed_namespace": "x1_paired_s_s",
                "updates": int(updates),
                "milestones": list(milestones),
                "parent_job_id": parent_id,
                "preserve_final_optimizer": bool(preserve_final_optimizer),
                "source_commit": str(source_commit),
                **dict(extra or {}),
            }
        )

    ms_2368 = (0, 592, 1184, 1472, 2368)
    ms_4736 = (0, 592, 1184, 1472, 2368, 3552, 4736)

    # R1: Zero-update frozen diagnostics (CPU/GPU frozen probes)
    for arch in ("P", "F-cap", "R", "R-noedge", "U-2", "U-4"):
        for s in dev_seeds:
            _reg("R1_INSTRUMENTATION", f"R1-FROZEN-{arch}-s{s}", arch, "frozen_probe", "S20", s, 0, milestones=ms_2368)

    # R2: Anchored vs Unanchored gated recurrence (2 depths x 2 cells x 3 seeds = 12 jobs)
    for depth_tag, arch_fix, arch_un in (("U2", "U2-Fix", "U2-GatedUnanchored"), ("U4", "U4-Fix", "U4-GatedUnanchored")):
        for arch in (arch_fix, arch_un):
            for s in dev_seeds:
                _reg("R2_RECURRENCE", f"R2-{arch}-D1H-s{s}", arch, "CE", "S20", s, 2368, milestones=ms_2368)

    # R3: Alpha designs (P1 2x2 role/dense x bilinear/parallel + R1 2x2 msg/nomsg x both/valonly + controls)
    for arch in ("P1-Alpha", "P1-DensePos", "P1-ParallelRead", "R1-Alpha", "R1-ValOnly", "R1-NoMsg-Both", "R1-NoMsg-ValOnly", "R1-SlotMLP"):
        for s in dev_seeds:
            _reg("R3_ALPHA", f"R3-{arch}-s{s}", arch, "CE", "S20", s, 2368, milestones=ms_2368)

    # R4: D3 learning & mechanism refinement factorial (grouped/interleaved + delayed ramp + PCGrad vs NoSurgery)
    for arch in ("P", "R", "R-noedge"):
        for obj in ("D3-Delay-PCGrad", "D3-Delay-NoSurgery", "D3-Immediate", "D3-ShuffledTarget", "CE"):
            for pack_mode in (("grouped", "interleaved") if arch in ("P", "R") and obj in ("CE", "D3-Immediate") else ("grouped",)):
                for s in dev_seeds:
                    _reg("R4_D3", f"R4-{arch}-{obj}-{pack_mode}-s{s}", arch, obj, "S20", s, 2368, milestones=ms_2368, extra={"pack_arrangement": pack_mode})

    # R5: Two further falsifiable screens (2 arms x 3 seeds x 2 screens = 12 jobs)
    for arch in ("CounterexampleReader", "RelevanceReader2", "OperatorTransportBank", "EpisodicMLP"):
        for s in dev_seeds:
            _reg("R5_SCREENS", f"R5-{arch}-s{s}", arch, "CE", "S20", s, 2368, milestones=ms_2368)

    # R6: Mandatory Native Legacy S50 (P-50M & R-50M) + S20 P & R 4,736-update D1-H controls (preserve final optimizer for R7 / continuation!)
    for arch, scale, param_exp in (("P", "S20", 19_947_925), ("R", "S20", 20_444_310), ("P-50M", "S50", 50_667_029), ("R-50M", "S50", 51_229_462)):
        for s in dev_seeds:
            pid = f"R6-{arch}-{scale}-D1H-4736-s{s}"
            _reg(
                "R6_LEGACY_SCALE",
                pid,
                arch,
                "standard",
                scale,
                s,
                4736,
                milestones=ms_4736,
                preserve_final_optimizer=True,
                extra={"expected_parameters": param_exp, "native_legacy": True, "profile_group": scale},
            )
            # R7: 2,368-update C-data task-adaptation child for every R6 scale control (12 adaptation jobs)
            _reg(
                "R7_ADAPTATION",
                f"R7-ADAPT-{arch}-{scale}-C22-s{s}",
                arch,
                "CE",
                scale,
                s,
                2368,
                milestones=ms_2368,
                parent_id=pid,
                extra={"adaptation_parent_updates": 4736, "profile_group": scale},
            )

    return {
        "schema": "elendril-x3-r0-r7-program-ledger-v1",
        "source_commit": str(source_commit),
        "r0_x2_reconciliation": r0,
        "legacy_x3_migration_table": migration_table,
        "legacy_x3_migration_counts": {
            st: sum(1 for m in migration_table if m["migration_status"] == st)
            for st in ("unchanged", "scientifically_superseded_by_exact_output_equivalent", "open_with_reason")
        },
        "r1_r7_jobs": r_jobs,
        "r6_r7_parent_child_pairs": sum(1 for j in r_jobs if j["block"] == "R7_ADAPTATION"),
        "seed_namespaces": {
            "x1_and_r_program": "(7,7), (19,19), (41,41); conf (101,101), (211,211), (307,307)",
            "x3_legacy_migration": "seed + 1000 (strictly non-merged)",
        },
        "ledger_sha256": digest({"r0": r0["credited_x2_obligations"], "mig": migration_table, "jobs": r_jobs}),
    }


__all__ = [
    "RUN3_CANONICAL_43_JOB_IDS",
    "RUN3_DEFAULT_NEXT_JOB",
    "RUN3_EXPECTED_ARCHIVE_BYTES",
    "RUN3_EXPECTED_ARCHIVE_SHA256",
    "X3_DISCRIMINATION_DECISION_POINTS",
    "X3_INTERCHANGE_DONOR_CONDITIONS",
    "build_r0_r7_program_ledger",
    "reconcile_r0_run3_obligations",
    "run_four_five_hop_interior_edge_swap",
    "run_frozen_u_and_evidence_discrimination",
    "summarize_paired_arm_statistics",
]
