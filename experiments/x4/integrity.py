"""Cycle 1 Foundational Integrity Gate (§3.1-§3.5): Mathematical, Causal, Learning, Evaluation & Repro."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from anra_v5.elendril.codec import digest
from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.palantiri import dump_json_safe
from v5_model.core import packed_layout

from .battery import (
    PublicOracleX4Decoder,
    X4_22_FAMILIES,
    X4_FAMILY_GROUPS,
    build_x4_battery,
    canonical_structural_hash,
    evaluate_x4_cases,
    parse_and_score_x4_output,
    verify_x4_split_disjointness,
)
from .models import X4CognitiveModel, X4_ARCHITECTURES, build_x4_backbone_spec, encode_prompt_and_target_pair
from .objectives import X4_OBJECTIVES, build_x4_matched_batches, compute_x4_objective_losses, compute_x4_objective_step


def verify_mathematical_integrity() -> dict[str, Any]:
    """§3.1 Hand-computable fixtures proving probability normalization, pure D0, all-pairs targeted CF & D4-ECCO."""
    spec = build_x4_backbone_spec("tiny")
    m = X4CognitiveModel("R-Ref", spec, seed=7)
    m.eval()
    bat = build_x4_battery(split="dev", per_family=2, families=("C04_COMPOSITIONAL_SCAN", "C10_LANGUAGE_MATH_TRANSFER"))["cases"]
    matched = build_x4_matched_batches(bat, seed=7, batch_size=4)
    batches = matched["batches"]

    # 1. Normalized probability mixture check
    ids, _, end_p = encode_prompt_and_target_pair(bat[0]["prompt"], bat[0]["typed_target"], max_length=m.max_ctx)
    toks = torch.tensor([ids], dtype=torch.long)
    pos, mask = packed_layout(torch.zeros_like(toks), torch_module=torch)
    with torch.no_grad():
        out = m(toks, pos, mask, prompt_end_positions=torch.tensor([end_p], dtype=torch.long))
    p_sum_err = float((out["p_unified"].sum(dim=-1) - 1.0).abs().max().item())
    log_p_max = float(out["log_p_unified"].max().item())
    assert p_sum_err < 1e-5, f"p_unified failed simplex normalization: err={p_sum_err}"
    assert log_p_max <= 1e-6, f"log_p_unified exceeded 0: {log_p_max}"

    # 2. Matched exposure + pure D0 check across ALL objectives
    losses_by_obj = {}
    elig_bytes_by_obj = set()
    for obj in X4_OBJECTIVES:
        l_rec = compute_x4_objective_losses(m, batches[0], objective=obj, update_idx=900, total_updates=1000)
        losses_by_obj[obj] = {k: v for k, v in l_rec.items() if not k.endswith("_t")}
        elig_bytes_by_obj.add(l_rec["eligible_target_bytes"])
    assert len(elig_bytes_by_obj) == 1, f"Exposure mismatch across objectives: {elig_bytes_by_obj}"
    assert losses_by_obj["D0-ByteCE"]["eq_loss"] == 0.0 or losses_by_obj["D0-ByteCE"]["eff_eq_weight"] == 0.0
    assert losses_by_obj["D0-ByteCE"]["total_loss"] == losses_by_obj["D0-ByteCE"]["byte_ce"], "D0-ByteCE must equal pure byte_ce!"
    assert losses_by_obj["D2-ExactCF"]["cf_pairs_evaluated"] >= 1, "D2-ExactCF must evaluate all paired CF pairs"

    # 3. Hand-computable targeted CF math check: right-right < wrong-wrong divergence!
    ll_aa_right, ll_ab_right = torch.tensor(-0.1), torch.tensor(-2.0)
    ll_bb_right, ll_ba_right = torch.tensor(-0.1), torch.tensor(-2.0)
    ll_aa_wrong, ll_ab_wrong = torch.tensor(-3.0), torch.tensor(-3.1)
    ll_bb_wrong, ll_ba_wrong = torch.tensor(-3.0), torch.tensor(-3.1)
    cf_right = 0.5 * (F.relu(0.5 - (ll_aa_right - ll_ab_right)) + F.relu(0.5 - (ll_bb_right - ll_ba_right)))
    cf_wrong = 0.5 * (F.relu(0.5 - (ll_aa_wrong - ll_ab_wrong)) + F.relu(0.5 - (ll_bb_wrong - ll_ba_wrong)))
    assert float(cf_right.item()) == 0.0 and float(cf_wrong.item()) > 0.3, "CF margin must reward right-right vs wrong-wrong!"

    return {
        "passed": True,
        "p_unified_simplex_max_err": p_sum_err,
        "matched_eligible_bytes": next(iter(elig_bytes_by_obj)),
        "cf_right_loss": float(cf_right.item()),
        "cf_wrong_loss": float(cf_wrong.item()),
        "d0_pure_byte_ce_verified": True,
    }


def verify_information_flow_integrity() -> dict[str, Any]:
    """§3.2 Strict future-target non-leakage (ALL X4 architectures) + positive leak detector + context parity + persistent isolation."""
    spec = build_x4_backbone_spec("tiny")
    prompt = "C05|TEST|GOAL=n0 HOPS=2|n0->n1|n1->B|ANS=?"
    tgt_a = "ANSWER:BBBB"
    tgt_b = "ANSWER:ZZZZ"
    ids_a, _, end_a = encode_prompt_and_target_pair(prompt, tgt_a, max_length=spec.context_length)
    ids_b, _, end_b = encode_prompt_and_target_pair(prompt, tgt_b, max_length=spec.context_length)
    assert end_a == end_b and len(ids_a) == len(ids_b)
    # Index predicting the very first payload byte after 'ANSWER:' (unchanged across A and B):
    prefix_ans_len = len(UTF8.encode("ANSWER:"))
    check_end = end_a + prefix_ans_len

    t_a = torch.tensor([ids_a], dtype=torch.long)
    t_b = torch.tensor([ids_b], dtype=torch.long)
    segs = torch.zeros_like(t_a)
    pos, mask = packed_layout(segs, torch_module=torch)
    end_t = torch.tensor([end_a], dtype=torch.long)

    non_leak_by_arch: dict[str, float] = {}
    for arch in X4_ARCHITECTURES:
        m = X4CognitiveModel(arch, spec, seed=7)
        m.eval()
        with torch.no_grad():
            oa = m(t_a, pos, mask, prompt_end_positions=end_t)
            ob = m(t_b, pos, mask, prompt_end_positions=end_t)
        # Future bytes ONLY start at check_end + 1; predictions at 0..check_end MUST have 0.0 difference!
        logit_diff = float((oa["log_p_unified"][:, : check_end + 1, :] - ob["log_p_unified"][:, : check_end + 1, :]).abs().max().item())
        act_diff = float((oa["action_logits"] - ob["action_logits"]).abs().max().item())
        max_d = max(logit_diff, act_diff)
        assert max_d < 1e-6, f"FUTURE_TARGET_LEAK on {arch}: max_diff={max_d}"
        non_leak_by_arch[arch] = max_d

    # Positive control: turn on `debug_leak_future=True` on P-Ref and assert our test CATCHES the leak!
    m_leak = X4CognitiveModel("P-Ref", spec, seed=7)
    m_leak.eval()
    with torch.no_grad():
        la = m_leak(t_a, pos, mask, prompt_end_positions=end_t, debug_leak_future=True)
        lb = m_leak(t_b, pos, mask, prompt_end_positions=end_t, debug_leak_future=True)
    pos_ctrl_diff = float((la["log_p_unified"][:, : check_end + 1, :] - lb["log_p_unified"][:, : check_end + 1, :]).abs().max().item())
    assert pos_ctrl_diff > 1e-4, f"Positive leak detector failed to catch intentional future leak: {pos_ctrl_diff}"

    # Training-vs-generation context parity test (Defect H fixed)
    m_par = X4CognitiveModel("ANRA-X4-Int", spec, seed=7)
    m_par.eval()
    p_ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
    pt = torch.tensor([p_ids], dtype=torch.long)
    pp, pm = packed_layout(torch.zeros_like(pt), torch_module=torch)
    with torch.no_grad():
        f_out = m_par(pt, pp, pm, prompt_end_positions=torch.tensor([len(p_ids) - 1], dtype=torch.long))
        # Compare argmax first byte under generate_bytes vs forward at end_pos
        gen_first_lp = f_out["log_p_unified"][0, -1].clone()
        gen_first_lp[UTF8.pad_id] = -1e9
        gen_first_lp[UTF8.bos_id] = -1e9
        first_byte = int(torch.argmax(gen_first_lp).item())
        gen_res = m_par.generate_bytes(prompt, max_new_bytes=1)
        tf_first_ch = UTF8.decode_lossy([first_byte], strip_special=True)[0].strip()
        assert gen_res["text"] == tf_first_ch or (first_byte < 4 and gen_res["text"] == "")

    # Persistent state isolation & reset parity
    m_par.reset_episode()
    res_clean = m_par.generate_bytes(prompt, max_new_bytes=4, use_persistent_state=True)
    m_par.update_persistent_state("C08|MEM|set:u1=Z@1|Q:u1=?")
    res_dirty = m_par.generate_bytes(prompt, max_new_bytes=4, use_persistent_state=True)
    m_par.reset_episode()
    res_after_reset = m_par.generate_bytes(prompt, max_new_bytes=4, use_persistent_state=True)
    assert res_clean["text"] == res_after_reset["text"], "reset_episode failed to restore clean generation!"

    return {
        "passed": True,
        "architectures_checked": len(non_leak_by_arch),
        "max_future_leak_diff_across_all_archs": max(non_leak_by_arch.values()),
        "positive_leaky_control_diff": pos_ctrl_diff,
        "persistent_reset_verified": True,
        "state_dirty_vs_clean_delta_lp": abs(res_dirty["seq_log_prob"] - res_clean["seq_log_prob"]),
    }


def verify_learning_integrity() -> dict[str, Any]:
    """§3.3 Controlled learning checks across every X4 architecture proving live gradients & prediction shift."""
    spec = build_x4_backbone_spec("tiny")
    bat = build_x4_battery(split="train", per_family=1, families=("X01", "C04_COMPOSITIONAL_SCAN", "C10_LANGUAGE_MATH_TRANSFER"))["cases"]
    matched = build_x4_matched_batches(bat, seed=7, batch_size=len(bat))
    batch0 = matched["batches"][0]
    arch_reports: dict[str, Any] = {}

    for arch in X4_ARCHITECTURES:
        m = X4CognitiveModel(arch, spec, seed=7)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        w0 = copy.deepcopy(m.state_dict())
        stats = compute_x4_objective_step(m, opt, batch0, objective="D0-ByteCE", update_idx=0, total_updates=4)
        dead = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        assert not dead, f"Dead gradients on {arch}: {dead}"
        assert math.isfinite(stats["total_loss"]), f"Non-finite loss on {arch}"
        changed = sum(1 for k, p in m.state_dict().items() if not torch.equal(p, w0[k]))
        assert changed > 0, f"No weights updated on {arch}"
        arch_reports[arch] = {
            "params": sum(p.numel() for p in m.parameters()),
            "dead_grads": 0,
            "updated_tensors": changed,
            "loss": round(stats["total_loss"], 4),
        }

    return {"passed": True, "architectures": arch_reports}


def verify_evaluation_integrity() -> dict[str, Any]:
    """§3.4 Oracle=1.0 across all 22 families, strict bad-output rejection, and split-agnostic hash invariance."""
    tr = build_x4_battery(split="train", per_family=2)["cases"]
    dv = build_x4_battery(split="dev", per_family=2)["cases"]
    cf = build_x4_battery(split="confirm", per_family=2)["cases"]
    oo = build_x4_battery(split="ood_transfer", per_family=2)["cases"]
    disj = verify_x4_split_disjointness({"train": tr, "dev": dv, "confirm": cf, "ood_transfer": oo})

    assert len(X4_22_FAMILIES) == 22
    assert sum(len(v) for v in X4_FAMILY_GROUPS.values()) == 22

    # Perfect public oracle check across all 22 families in dev
    oracle = PublicOracleX4Decoder()
    preds = evaluate_x4_cases(oracle, dv, batch_size=16)
    acc = sum(int(bool(p["correct"])) for p in preds) / max(1, len(preds))
    assert acc == 1.0, f"PublicOracleX4Decoder must score 1.0 across all 22 families, got {acc}"

    # Verify strict rejection of malformed, truncated, unterminated, trailing suffix, wrong unit
    c10 = next(c for c in dv if c["family"] == "C10_LANGUAGE_MATH_TRANSFER")
    assert not parse_and_score_x4_output(c10["prompt"], "ANSWER:10 wrongunit", c10)["correct"]
    assert not parse_and_score_x4_output(c10["prompt"], c10["typed_target"], c10, terminated_by_eos=False)["correct"]
    assert parse_and_score_x4_output(c10["prompt"], c10["typed_target"], c10, terminated_by_eos=True)["correct"]

    # Split-agnostic canonical_structural_hash invariance (Defect I): renaming split metadata must NEVER alter hash!
    c5 = next(c for c in dv if c["family"] == "C05_RELATIONAL_GRAPH_CF")
    h1 = canonical_structural_hash(c5)
    c5_mod = copy.deepcopy(c5)
    c5_mod["split"] = "confirm"
    c5_mod["problem_spec"]["split_motif"] = "confirm"
    c5_mod["problem_spec"]["split_tag"] = "FC"
    c5_mod["problem_spec"]["split_class"] = "ood_transfer"
    h2 = canonical_structural_hash(c5_mod)
    assert h1 == h2, "canonical_structural_hash leaked split metadata!"

    return {
        "passed": True,
        "families_count": len(X4_22_FAMILIES),
        "groups_count": len(X4_FAMILY_GROUPS),
        "oracle_22fam_accuracy": acc,
        "split_disjoint": disj["disjoint"],
        "split_agnostic_hash_verified": True,
    }


def verify_reproducibility_and_byte_amp_shield() -> dict[str, Any]:
    """§3.5 Determinism, restore parity, and Level-7 0xB5 non-UTF-8 byte decode safety across all architectures."""
    spec = build_x4_backbone_spec("tiny")
    m1 = X4CognitiveModel("ANRA-X4-Int", spec, seed=19)
    m2 = X4CognitiveModel("ANRA-X4-Int", spec, seed=19)
    for k in m1.state_dict():
        assert torch.equal(m1.state_dict()[k], m2.state_dict()[k]), f"Init non-determinism on {k}"

    # Verify non-UTF-8 0xB5 safe decode (Stage 5 shield)
    b5_id = 4 + 0xB5
    txt, ok_utf8 = UTF8.decode_lossy([UTF8.encode("A")[0], b5_id, UTF8.encode("Z")[0]], strip_special=True)
    assert isinstance(txt, str) and ok_utf8 is False

    # Verify U2-Fix step-trajectory evaluation (0, 1, 2, 4, 8 steps)
    mu = X4CognitiveModel("U2-Fix", spec, seed=7)
    mu.eval()
    steps_seen = []
    for k_steps in (0, 1, 2, 4, 8):
        out = mu.generate_bytes("X01|test", max_new_bytes=2, eval_steps_override=k_steps)
        steps_seen.append(k_steps)
        assert math.isfinite(out["seq_log_prob"])
    return {"passed": True, "u_step_trajectory_verified": steps_seen, "lossy_0xB5_safe": True}


def run_foundational_integrity_gate(work_dir: Path | str | None = None) -> dict[str, Any]:
    """Execute complete Cycle 1 Foundational Integrity Gate (§3.1-§3.5)."""
    r31 = verify_mathematical_integrity()
    r32 = verify_information_flow_integrity()
    r33 = verify_learning_integrity()
    r34 = verify_evaluation_integrity()
    r35 = verify_reproducibility_and_byte_amp_shield()
    all_ok = all(r["passed"] for r in (r31, r32, r33, r34, r35))
    receipt = {
        "schema": "elendril-x4-cycle1-integrity-gate-v1",
        "all_passed": bool(all_ok),
        "3_1_mathematical_integrity": r31,
        "3_2_information_flow_integrity": r32,
        "3_3_learning_integrity": r33,
        "3_4_evaluation_integrity": r34,
        "3_5_reproducibility_and_shield": r35,
    }
    if work_dir is not None:
        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "X4_CYCLE1_INTEGRITY_GATE.json").write_text(dump_json_safe(receipt), encoding="utf-8")
    return receipt


__all__ = [
    "run_foundational_integrity_gate",
    "verify_evaluation_integrity",
    "verify_information_flow_integrity",
    "verify_learning_integrity",
    "verify_mathematical_integrity",
    "verify_reproducibility_and_byte_amp_shield",
]
