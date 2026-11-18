"""Avendi V8-Y2 computable odds, 4-cell family metrics, paired bootstrap, and stage routing decisions."""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np

from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.planthir.design import SeedPairSpec, analyze_paired_seed_runs
from .protocol import (
    INIT_SEEDS,
    ORDER_SEEDS,
    QUALIFICATION_GATES,
    STAGE_STATUS_VALUES,
    SYMBOLS,
    derive_seed_int,
)
from .worlds import AvendiCase, AvendiFamily


AP_TOKEN_IDS: dict[str, int] = {sym: int(UTF8.encode(sym)[0]) for sym in SYMBOLS}
TOKEN_ID_TO_AP: dict[int, str] = {v: k for k, v in AP_TOKEN_IDS.items()}
Q_TOKEN_ID = int(UTF8.encode("Q")[0])
R_TOKEN_ID = int(UTF8.encode("R")[0])
ABSTAIN_TOKEN_ID = int(UTF8.encode("!")[0])
EOS_TOKEN_ID = int(UTF8.eos_id)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def classify_winning_token(raw_tokens: Sequence[int]) -> str:
    """Classify the first emitted raw token into ANSWER_AP, Q, R, !, EOS_PREMATURE, or MALFORMED."""
    if not raw_tokens:
        return "MALFORMED"
    first = int(raw_tokens[0])
    if first in TOKEN_ID_TO_AP:
        return "ANSWER_AP"
    if first == Q_TOKEN_ID:
        return "Q"
    if first == R_TOKEN_ID:
        return "R"
    if first == ABSTAIN_TOKEN_ID:
        return "!"
    if first == EOS_TOKEN_ID:
        return "EOS_PREMATURE"
    return "MALFORMED"


def score_single_case_prediction(
    case: AvendiCase,
    *,
    raw_tokens: Sequence[int],
    first_logits: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Compute exact raw answer+EOS, first divergence, p(ANSWER), within-A-P rank, and signed margin."""
    logits = np.asarray(first_logits, dtype=np.float64)
    if logits.shape != (260,):
        raise ValueError(f"expected 260-way first_logits for {case.case_id}, got shape {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError(f"non-finite logits on {case.case_id}")

    probs = _softmax_np(logits)
    p_answer_mass = float(sum(probs[AP_TOKEN_IDS[s]] for s in SYMBOLS))

    ap_sorted = sorted(SYMBOLS, key=lambda s: (-float(logits[AP_TOKEN_IDS[s]]), s))
    ap_top1 = ap_sorted[0]
    within_ap_rank = ap_sorted.index(case.expected_answer) + 1 if case.expected_answer in ap_sorted else 16

    exp_tok_id = int(case.target_tokens[0])
    full_sorted = sorted(range(260), key=lambda tid: (-float(logits[tid]), tid))
    full_vocab_rank = full_sorted.index(exp_tok_id) + 1

    toks = tuple(int(t) for t in raw_tokens)
    target_toks = tuple(int(t) for t in case.target_tokens)
    raw_exact_correct = toks == target_toks

    first_divergence: int | None = None
    divergence_reason = "NONE"
    if not raw_exact_correct:
        if not toks or toks[0] != target_toks[0]:
            first_divergence = 0
            w_class = classify_winning_token(toks)
            if w_class in ("Q", "R", "!"):
                divergence_reason = f"PREMATURE_{w_class}"
            elif w_class == "ANSWER_AP":
                divergence_reason = "WRONG_ANSWER_SYMBOL"
            else:
                divergence_reason = "MALFORMED_FIRST_BYTE"
        else:
            first_divergence = 1
            divergence_reason = "WRONG_OR_MISSING_EOS"

    win_class = classify_winning_token(toks)
    false_abstention = (case.expected_operation == "ANSWER") and (win_class == "!")
    malformed_action = (
        win_class in ("EOS_PREMATURE", "MALFORMED")
        or len(toks) != 2
        or toks[-1] != EOS_TOKEN_ID
    )

    id_g = AP_TOKEN_IDS[case.endpoint_g]
    id_h = AP_TOKEN_IDS[case.endpoint_h]
    m_gh = float(logits[id_g] - logits[id_h])

    id_gold = AP_TOKEN_IDS[case.expected_answer]
    id_rival = AP_TOKEN_IDS[case.rival_answer]
    signed_margin = float(logits[id_gold] - logits[id_rival])

    # Answer-only constrained EOS rescue check (identical public state under answer-only decoding)
    ap_constrained_correct = ap_top1 == case.expected_answer

    return {
        "case_id": case.case_id,
        "family_id": case.family_id,
        "cell_key": case.cell_key,
        "expected_answer": case.expected_answer,
        "rival_answer": case.rival_answer,
        "raw_tokens": list(toks),
        "raw_exact_correct": raw_exact_correct,
        "ap_top1": ap_top1,
        "ap_constrained_correct": ap_constrained_correct,
        "within_ap_rank": within_ap_rank,
        "full_vocab_rank": full_vocab_rank,
        "p_answer_mass": p_answer_mass,
        "m_logit_g_minus_h": m_gh,
        "signed_margin": signed_margin,
        "first_divergence": first_divergence,
        "divergence_reason": divergence_reason,
        "winning_token_class": win_class,
        "false_abstention": false_abstention,
        "malformed_action": malformed_action,
    }


def score_crossover_family(
    family: AvendiFamily,
    case_scores_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute 4-cell family metrics: all-4 raw, latent flip, min signed margin, interaction, and pair sensitivity."""
    cmap = family.cell_map()
    s00 = case_scores_by_id[cmap["00"].case_id]
    s01 = case_scores_by_id[cmap["01"].case_id]
    s10 = case_scores_by_id[cmap["10"].case_id]
    s11 = case_scores_by_id[cmap["11"].case_id]
    cells = (s00, s01, s10, s11)

    m00 = float(s00["m_logit_g_minus_h"])
    m01 = float(s01["m_logit_g_minus_h"])
    m10 = float(s10["m_logit_g_minus_h"])
    m11 = float(s11["m_logit_g_minus_h"])

    signed_margins = {
        "00": float(s00["signed_margin"]),
        "01": float(s01["signed_margin"]),
        "10": float(s10["signed_margin"]),
        "11": float(s11["signed_margin"]),
    }
    min_signed_margin = min(signed_margins.values())
    interaction = (m00 - m01 - m10 + m11) / 4.0
    latent_flip = all(v > 0.0 for v in signed_margins.values())

    all_four_raw_correct = all(bool(c["raw_exact_correct"]) for c in cells)
    all_four_ap_constrained_correct = all(bool(c["ap_constrained_correct"]) for c in cells)
    raw_correct_count = sum(int(bool(c["raw_exact_correct"])) for c in cells)

    t00 = tuple(s00["raw_tokens"])
    t01 = tuple(s01["raw_tokens"])
    t10 = tuple(s10["raw_tokens"])
    t11 = tuple(s11["raw_tokens"])

    query_orig_same = t00 == t10
    query_swap_same = t01 == t11
    bridge_qa_same = t00 == t01
    bridge_qb_same = t10 == t11
    all_four_same_tokens = t00 == t01 == t10 == t11

    query_orig_both_correct = bool(s00["raw_exact_correct"] and s10["raw_exact_correct"])
    query_swap_both_correct = bool(s01["raw_exact_correct"] and s11["raw_exact_correct"])
    bridge_qa_both_correct = bool(s00["raw_exact_correct"] and s01["raw_exact_correct"])
    bridge_qb_both_correct = bool(s10["raw_exact_correct"] and s11["raw_exact_correct"])

    return {
        "family_id": family.family_id,
        "hops": family.hops,
        "endpoint_g": family.endpoint_g,
        "endpoint_h": family.endpoint_h,
        "all_four_raw_correct": all_four_raw_correct,
        "all_four_ap_constrained_correct": all_four_ap_constrained_correct,
        "raw_correct_count": raw_correct_count,
        "latent_flip": latent_flip,
        "min_signed_margin": min_signed_margin,
        "interaction": interaction,
        "signed_margins": signed_margins,
        "m_logit_g_minus_h": {"00": m00, "01": m01, "10": m10, "11": m11},
        "query_orig_same_tokens": query_orig_same,
        "query_swap_same_tokens": query_swap_same,
        "bridge_qa_same_tokens": bridge_qa_same,
        "bridge_qb_same_tokens": bridge_qb_same,
        "all_four_same_tokens": all_four_same_tokens,
        "greedy_changed_query_orig": not query_orig_same,
        "greedy_changed_query_swap": not query_swap_same,
        "greedy_changed_bridge_qa": not bridge_qa_same,
        "greedy_changed_bridge_qb": not bridge_qb_same,
        "query_orig_both_correct": query_orig_both_correct,
        "query_swap_both_correct": query_swap_both_correct,
        "bridge_qa_both_correct": bridge_qa_both_correct,
        "bridge_qb_both_correct": bridge_qb_both_correct,
        "repeated_greedy_with_latent_flip": bool(all_four_same_tokens and latent_flip),
        "positive_interaction_but_not_latent_flip": bool(interaction > 0.0 and not latent_flip),
    }


def summarize_panel_predictions(
    *,
    panel_name: str,
    cases: Sequence[AvendiCase],
    families: Sequence[AvendiFamily],
    case_scores_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate case-level and 4-cell family-level metrics for one panel."""
    n_cases = max(1, len(cases))
    c_rows = [case_scores_by_id[c.case_id] for c in cases]

    win_counts: Counter[str] = Counter(str(r["winning_token_class"]) for r in c_rows)
    div_counts: Counter[str] = Counter(str(r["divergence_reason"]) for r in c_rows)

    case_summary = {
        "panel": panel_name,
        "cases_total": len(cases),
        "raw_exact_count": sum(int(bool(r["raw_exact_correct"])) for r in c_rows),
        "raw_exact_accuracy": sum(int(bool(r["raw_exact_correct"])) for r in c_rows) / n_cases,
        "ap_constrained_count": sum(int(bool(r["ap_constrained_correct"])) for r in c_rows),
        "ap_constrained_accuracy": sum(int(bool(r["ap_constrained_correct"])) for r in c_rows) / n_cases,
        "mean_p_answer_mass": sum(float(r["p_answer_mass"]) for r in c_rows) / n_cases,
        "mean_within_ap_rank": sum(float(r["within_ap_rank"]) for r in c_rows) / n_cases,
        "mean_signed_margin": sum(float(r["signed_margin"]) for r in c_rows) / n_cases,
        "false_abstention_count": sum(int(bool(r["false_abstention"])) for r in c_rows),
        "false_abstention_rate": sum(int(bool(r["false_abstention"])) for r in c_rows) / n_cases,
        "malformed_action_count": sum(int(bool(r["malformed_action"])) for r in c_rows),
        "malformed_action_rate": sum(int(bool(r["malformed_action"])) for r in c_rows) / n_cases,
        "winning_token_counts": dict(sorted(win_counts.items())),
        "divergence_reason_counts": dict(sorted(div_counts.items())),
    }

    if not families:
        return {**case_summary, "families_total": 0}

    n_fams = len(families)
    f_rows = [score_crossover_family(f, case_scores_by_id) for f in families]
    repeated_fams = [f for f in f_rows if f["all_four_same_tokens"]]
    rep_win_counts: Counter[str] = Counter()
    for f in families:
        f_sc = score_crossover_family(f, case_scores_by_id)
        if f_sc["all_four_same_tokens"]:
            c00 = case_scores_by_id[f.cell_map()["00"].case_id]
            rep_win_counts[str(c00["winning_token_class"])] += 1

    family_summary = {
        "families_total": n_fams,
        "all_four_raw_count": sum(int(bool(r["all_four_raw_correct"])) for r in f_rows),
        "all_four_raw_rate": sum(int(bool(r["all_four_raw_correct"])) for r in f_rows) / n_fams,
        "all_four_ap_constrained_count": sum(int(bool(r["all_four_ap_constrained_correct"])) for r in f_rows),
        "all_four_ap_constrained_rate": sum(int(bool(r["all_four_ap_constrained_correct"])) for r in f_rows) / n_fams,
        "latent_flip_count": sum(int(bool(r["latent_flip"])) for r in f_rows),
        "latent_flip_rate": sum(int(bool(r["latent_flip"])) for r in f_rows) / n_fams,
        "positive_interaction_but_not_latent_flip_count": sum(
            int(bool(r["positive_interaction_but_not_latent_flip"])) for r in f_rows
        ),
        "mean_min_signed_margin": sum(float(r["min_signed_margin"]) for r in f_rows) / n_fams,
        "mean_interaction": sum(float(r["interaction"]) for r in f_rows) / n_fams,
        "same_token_query_pairs_count": sum(
            int(bool(r["query_orig_same_tokens"])) + int(bool(r["query_swap_same_tokens"])) for r in f_rows
        ),
        "same_token_query_pairs_total": 2 * n_fams,
        "same_token_query_pairs_rate": sum(
            int(bool(r["query_orig_same_tokens"])) + int(bool(r["query_swap_same_tokens"])) for r in f_rows
        ) / (2 * n_fams),
        "same_token_bridge_pairs_count": sum(
            int(bool(r["bridge_qa_same_tokens"])) + int(bool(r["bridge_qb_same_tokens"])) for r in f_rows
        ),
        "same_token_bridge_pairs_total": 2 * n_fams,
        "same_token_bridge_pairs_rate": sum(
            int(bool(r["bridge_qa_same_tokens"])) + int(bool(r["bridge_qb_same_tokens"])) for r in f_rows
        ) / (2 * n_fams),
        "query_both_correct_pairs_count": sum(
            int(bool(r["query_orig_both_correct"])) + int(bool(r["query_swap_both_correct"])) for r in f_rows
        ),
        "bridge_both_correct_pairs_count": sum(
            int(bool(r["bridge_qa_both_correct"])) + int(bool(r["bridge_qb_both_correct"])) for r in f_rows
        ),
        "all_four_same_tokens_families": len(repeated_fams),
        "repeated_greedy_with_latent_flip_count": sum(
            int(bool(r["repeated_greedy_with_latent_flip"])) for r in f_rows
        ),
        "repeated_greedy_winning_token_counts": dict(sorted(rep_win_counts.items())),
        "decoding_gap_ap_minus_raw": (
            sum(int(bool(r["all_four_ap_constrained_correct"])) for r in f_rows)
            - sum(int(bool(r["all_four_raw_correct"])) for r in f_rows)
        ) / n_fams,
        "family_scores": {r["family_id"]: r for r in f_rows},
    }
    return {**case_summary, **family_summary}


def bootstrap_paired_family_contrasts(
    treatment_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]],
    control_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    metric_key: str = "all_four_raw_correct",
    n_resamples: int = 2000,
    seed: int = 2217,
) -> dict[str, Any]:
    """Stratified 2,000-resample paired family bootstrap across seeds + Planthir degenerate-variance check."""
    seeds = sorted(set(treatment_by_seed) & set(control_by_seed))
    if not seeds:
        raise ValueError("bootstrap_paired_family_contrasts requires at least one shared seed")

    first_s = seeds[0]
    strata: dict[int, list[str]] = {}
    for fid, row in sorted(treatment_by_seed[first_s].items()):
        hops = int(row.get("hops", 1))
        strata.setdefault(hops, []).append(fid)

    rng = random.Random(derive_seed_int("avendi-family-bootstrap", seed, n_resamples, metric_key, seeds))
    total_fams = sum(len(v) for v in strata.values())
    per_seed_samples: dict[int, list[float]] = {s: [] for s in seeds}
    mean_samples: list[float] = []

    for _ in range(n_resamples):
        sampled: list[str] = []
        for h in sorted(strata):
            pool = strata[h]
            sampled.extend(rng.choice(pool) for _ in range(len(pool)))
        s_diffs: list[float] = []
        for s in seeds:
            t_map = treatment_by_seed[s]
            c_map = control_by_seed[s]
            d = sum(float(t_map[f][metric_key]) - float(c_map[f][metric_key]) for f in sampled) / max(1, total_fams)
            per_seed_samples[s].append(d)
            s_diffs.append(d)
        mean_samples.append(sum(s_diffs) / len(s_diffs))

    def _ci(vals: Sequence[float]) -> tuple[float, float]:
        sv = sorted(vals)
        lo = sv[max(0, min(len(sv) - 1, int(0.025 * len(sv))))]
        hi = sv[max(0, min(len(sv) - 1, int(0.975 * len(sv)) - 1))]
        return float(lo), float(hi)

    obs_by_seed: dict[int, float] = {}
    paired_rows: list[dict[str, Any]] = []

    for s in seeds:
        t_map = treatment_by_seed[s]
        c_map = control_by_seed[s]
        t_mean = sum(float(r[metric_key]) for r in t_map.values()) / max(1, len(t_map))
        c_mean = sum(float(r[metric_key]) for r in c_map.values()) / max(1, len(c_map))
        obs_by_seed[s] = t_mean - c_mean
        oseed = ORDER_SEEDS.get(s, 1000 + s)
        paired_rows.append({
            "contrast_id": "treatment_vs_control",
            "pair_id": f"seed_{s}",
            "init_seed": int(s),
            "order_seed": int(oseed),
            "control_value": c_mean,
            "treatment_value": t_mean,
        })

    paired_seed_report = analyze_paired_seed_runs(
        paired_rows,
        n_bootstrap=n_resamples,
        bootstrap_seed=seed,
    )["contrasts"].get("treatment_vs_control", {})

    m_lo, m_hi = _ci(mean_samples)
    return {
        "schema": "anra-avendi-v8-y2-bootstrap-v1",
        "metric_key": metric_key,
        "resamples": n_resamples,
        "seed": seed,
        "inferential_unit": "family_cluster_stratified_by_depth_conditional_on_seeds",
        "note": "Interval is conditional on the chosen training seeds; no small-seed population significance claim.",
        "mean_difference": sum(obs_by_seed.values()) / len(obs_by_seed),
        "mean_ci95_low": m_lo,
        "mean_ci95_high": m_hi,
        "per_seed": {
            str(s): {
                "observed_difference": obs_by_seed[s],
                "ci95_low": _ci(per_seed_samples[s])[0],
                "ci95_high": _ci(per_seed_samples[s])[1],
            }
            for s in seeds
        },
        "paired_seed_summary": paired_seed_report,
    }


def evaluate_frozen_routing_decision(
    arms_by_seed: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Evaluate P2 frozen gates across F00 and F11 and determine the first failing dependency."""
    gates = QUALIFICATION_GATES["p2_frozen"]
    lit_thresh = float(gates["literal_min_accuracy_per_seed"])
    one_thresh = float(gates["one_hop_all_four_min_per_seed"])
    three_thresh = float(gates["three_hop_all_four_min_per_seed"])

    arm_reports: dict[str, Any] = {}
    for arm in ("F00", "F11"):
        seed_map = arms_by_seed.get(arm, {})
        if not seed_map:
            arm_reports[arm] = {"status": "BLOCKED", "reason": "missing_arm_evaluations"}
            continue
        seeds = sorted(seed_map)
        lit_by_s = {s: float(seed_map[s]["literal_copy"]["raw_exact_accuracy"]) for s in seeds}
        one_by_s = {s: float(seed_map[s]["one_hop"]["all_four_raw_rate"]) for s in seeds}
        three_by_s = {s: float(seed_map[s]["three_hop"]["all_four_raw_rate"]) for s in seeds}
        one_flip_by_s = {s: float(seed_map[s]["one_hop"]["latent_flip_rate"]) for s in seeds}
        three_flip_by_s = {s: float(seed_map[s]["three_hop"]["latent_flip_rate"]) for s in seeds}

        lit_pass = all(v >= lit_thresh for v in lit_by_s.values())
        one_pass = all(v >= one_thresh for v in one_by_s.values())
        three_pass = all(v >= three_thresh for v in three_by_s.values())
        literal_format_sensitive = (not lit_pass) and one_pass

        one_sign_passes_but_raw_fails = (not one_pass) and all(v >= one_thresh for v in one_flip_by_s.values())
        three_sign_passes_but_raw_fails = (not three_pass) and all(v >= three_thresh for v in three_flip_by_s.values())

        arm_reports[arm] = {
            "seeds_evaluated": seeds,
            "literal_accuracy_by_seed": {str(k): v for k, v in lit_by_s.items()},
            "one_hop_all_four_by_seed": {str(k): v for k, v in one_by_s.items()},
            "three_hop_all_four_by_seed": {str(k): v for k, v in three_by_s.items()},
            "one_hop_latent_flip_by_seed": {str(k): v for k, v in one_flip_by_s.items()},
            "three_hop_latent_flip_by_seed": {str(k): v for k, v in three_flip_by_s.items()},
            "mean_literal_accuracy": sum(lit_by_s.values()) / len(lit_by_s),
            "mean_one_hop_all_four": sum(one_by_s.values()) / len(one_by_s),
            "mean_three_hop_all_four": sum(three_by_s.values()) / len(three_by_s),
            "literal_qualified": lit_pass,
            "literal_format_sensitive": literal_format_sensitive,
            "one_hop_qualified": one_pass,
            "three_hop_qualified": three_pass,
            "one_hop_decoding_gap_diagnostic": one_sign_passes_but_raw_fails,
            "three_hop_decoding_gap_diagnostic": three_sign_passes_but_raw_fails,
        }

    # Select anchor arm (select F00 on diagnostic tie)
    f00_rep = arm_reports.get("F00", {})
    f11_rep = arm_reports.get("F11", {})
    if f11_rep.get("one_hop_qualified") and not f00_rep.get("one_hop_qualified"):
        anchor_arm = "F11"
    else:
        anchor_arm = "F00"

    anchor = arm_reports.get(anchor_arm, {})
    if anchor.get("status") == "BLOCKED":
        return {
            "schema": "anra-avendi-v8-y2-routing-receipt-v1",
            "status": "BLOCKED",
            "selected_anchor_arm": anchor_arm,
            "first_failing_dependency": "P1_CHECKPOINT_RESTORATION",
            "next_admitted_stage": "P1_RESTORE_OR_P3_FRESH_ACQUISITION",
            "arm_reports": arm_reports,
        }

    if not anchor["literal_qualified"] and not anchor["one_hop_qualified"]:
        first_fail = "LITERAL_READOUT_AND_ONE_HOP_SELECTION"
        next_stage = "P3_CONTROLLED_ACQUISITION"
        diag_note = "Literal copy and 1-hop selection both below gate; prioritize renderer/EOS audit and P3 1-hop acquisition."
    elif not anchor["one_hop_qualified"]:
        first_fail = "ONE_HOP_TWO_KEY_SELECTION"
        next_stage = "P3_CONTROLLED_ACQUISITION"
        diag_note = (
            "Within-A-P sign patterns pass while raw fails (decoding/action-choice gap)."
            if anchor["one_hop_decoding_gap_diagnostic"]
            else "1-hop two-key selection failed (< 80% all-four); admit P3 paired counterfactual margin acquisition."
        )
    elif not anchor["three_hop_qualified"]:
        first_fail = "THREE_HOP_COMPOSITION"
        next_stage = "P4_TRANSITION_INTERCHANGE"
        diag_note = "1-hop lookup qualified (>= 80%), 3-hop crossover failed (< 70%); admit P4 intermediate-state transition study."
    else:
        first_fail = "NONE_IN_CORE_CROSSOVER"
        next_stage = "P4_DONOR_DEPENDENCE_AND_P5_CONDITIONAL_ACTION"
        diag_note = "Both 1-hop and 3-hop crossover qualified; test P4 donor state intervention and P5 epistemic action relevance."

    return {
        "schema": "anra-avendi-v8-y2-routing-receipt-v1",
        "status": "DIAGNOSTIC" if (anchor.get("one_hop_decoding_gap_diagnostic") or anchor.get("three_hop_decoding_gap_diagnostic")) else (
            "QUALIFIED" if anchor.get("one_hop_qualified") and anchor.get("three_hop_qualified") else "FAILED"
        ),
        "selected_anchor_arm": anchor_arm,
        "first_failing_dependency": first_fail,
        "next_admitted_stage": next_stage,
        "diagnostic_note": diag_note,
        "arm_reports": arm_reports,
    }
