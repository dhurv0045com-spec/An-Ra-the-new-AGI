"""Gate 1-4 release verification tests for AN-RA V8-X3 (2026-11-24 Execution Spec)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import math
import shutil

import pytest
import torch
import torch.nn.functional as F

from v5_model.core import packed_layout
from experiments.x3 import (
    MANDATORY_22_FAMILIES,
    RUN3_EXPECTED_ARCHIVE_BYTES,
    RUN3_EXPECTED_ARCHIVE_SHA256,
    SPLIT_SEEDS,
    X3CognitiveModel,
    X3_ARCHITECTURES,
    X3_DISCRIMINATION_DECISION_POINTS,
    X3_INTERCHANGE_DONOR_CONDITIONS,
    X3_VERSIONED_CONTROL_PAIRS,
    audit_d1h_d2_schedule_and_preceding_observations,
    audit_legacy_class_collisions,
    build_legacy_x1_scaled_model,
    build_r0_r7_program_ledger,
    build_x3_backbone_spec,
    build_x3_cognitive_battery,
    canonical_structural_hash,
    compute_public_version_space_posterior,
    d3_warmup_ramp_weights,
    encode_prompt_and_target_pair,
    evaluate_x3_batched,
    parse_and_score_public_typed_output,
    reconcile_r0_run3_obligations,
    run_all_architectures_u0000_smoke,
    run_four_five_hop_interior_edge_swap,
    run_frozen_u_and_evidence_discrimination,
    run_x3_campaign,
    run_x3_single_job,
    summarize_paired_arm_statistics,
    verify_c09_budget_reversal_oracle,
    verify_fixed_weight_4microbatch_ce_equivalence,
    verify_perfect_public_decoder_through_evaluator,
    verify_x3_split_disjointness,
)
from experiments.x3.cognitive_battery import _generate_c01_case


def test_gate1_all_29_architectures_legacy_b0_and_s20_s50_param_parity(tmp_path: Path) -> None:
    assert len(X3_ARCHITECTURES) == 29
    rep = run_all_architectures_u0000_smoke(
        tmp_path / "u0000",
        source_commit="a" * 40,
        include_s50_construction=True,
    )
    assert rep["all_passed"] is True
    assert rep["legacy_b0_tested"] is True
    assert set(rep["architectures_tested"]) == set(X3_ARCHITECTURES)
    for arch in X3_ARCHITECTURES:
        det = rep["details"][arch]
        assert det["passed"] is True
        assert det["restore_max_diff"] < 1e-6
    assert rep["details"]["legacy-B0"]["passed"] is True

    # Native S20 & S50 constructor parameter verification (§2, §5 R6)
    s20_p = build_legacy_x1_scaled_model("P", scale="S20", init_seed=7)
    s20_r = build_legacy_x1_scaled_model("R", scale="S20", init_seed=7)
    assert s20_p["total_parameters"] == 19_947_925
    assert s20_r["total_parameters"] == 20_444_310
    del s20_p, s20_r
    assert rep["s50_constructed_params"]["legacy-P-50M"] == 50_667_029
    assert rep["s50_constructed_params"]["legacy-R-50M"] == 51_229_462

    # All 10 paired versioned treatment/control arms parameter-matched within <= 2% at S20
    s20_spec = build_x3_backbone_spec("S20")
    param_cache: dict[str, int] = {}
    for treatment, control in X3_VERSIONED_CONTROL_PAIRS:
        for name in (treatment, control):
            if name not in param_cache:
                m = X3CognitiveModel(name, s20_spec, seed=7)
                param_cache[name] = sum(int(p.numel()) for p in m.parameters())
                del m
        p_t = param_cache[treatment]
        p_c = param_cache[control]
        rel_diff = abs(p_t - p_c) / max(1, p_c)
        assert rel_diff <= 0.02, f"{treatment} ({p_t}) vs {control} ({p_c}) rel_diff={rel_diff:.4f}"


def test_gate2_public_typed_protocol_collisions_oracles_and_isomorphic_leaks() -> None:
    assert len(MANDATORY_22_FAMILIES) == 22

    # 1. Mod-16 legacy collision audit + 0 exact-scorer false acceptances (§1, §4, §6)
    dev_cases_64 = build_x3_cognitive_battery(split="dev", per_family=64, shot_counts=(0, 1, 4, 16, 64))["cases"]
    col = audit_legacy_class_collisions(dev_cases_64)
    for fam in (
        "C03_GRID_TRANSFORM_ARC",
        "C04_COMPOSITIONAL_SCAN",
        "C09_METACOGNITION_SELF_CHECK",
        "C10_LANGUAGE_MATH_TRANSFER",
    ):
        assert col["colliding_pairs_by_family"][fam] > 0, f"Expected mod-16 collisions in {fam}"
    assert col["zero_false_acceptances"] is True
    assert col["exact_scorer_false_acceptances"] == 0

    # 2. C01 k=0 public ambiguity posterior + nested 0/1/4/16/64 version-space shrinkage
    c01_nested = {
        k: _generate_c01_case("dev", SPLIT_SEEDS["dev"], 0, k)
        for k in (0, 1, 4, 16, 64)
    }
    c01_k0 = c01_nested[0]
    pub_k0 = compute_public_version_space_posterior(str(c01_k0["prompt"]))
    assert pub_k0["is_ambiguous"] is True
    assert pub_k0["consistent_rule_count"] > 1
    assert abs(sum(pub_k0["target_posterior"].values()) - 1.0) < 1e-6

    s_ambig = parse_and_score_public_typed_output(str(c01_k0["prompt"]), "ABSTAIN:AMBIGUOUS", c01_k0)
    assert s_ambig["correct"] is True and s_ambig["exact_score"] == 1.0
    s_guess = parse_and_score_public_typed_output(str(c01_k0["prompt"]), f"ANSWER:{c01_k0['gold_text']}", c01_k0)
    assert s_guess["exact_score"] == 0.0
    assert 0.0 <= s_guess["ambiguity_posterior_credit"] < 1.0

    # Verify nested prefixes across 0 -> 1 -> 4 -> 16 -> 64 and monotonic V-space shrinkage
    demo_lists = {}
    v_sizes = {}
    for k, case_k in c01_nested.items():
        seg = str(case_k["prompt"]).split("|")[3]
        demo_lists[k] = [x for x in seg.split(";") if x]
        assert len(demo_lists[k]) == k
        v_sizes[k] = int(compute_public_version_space_posterior(str(case_k["prompt"]))["consistent_rule_count"])
    assert demo_lists[1] == demo_lists[4][:1]
    assert demo_lists[4] == demo_lists[16][:4]
    assert demo_lists[16] == demo_lists[64][:16]
    assert v_sizes[64] <= v_sizes[16] <= v_sizes[4] <= v_sizes[1] <= v_sizes[0]
    assert v_sizes[64] < v_sizes[0]

    # 3. C04 corrupted suffix rejection after matching first token + >20-byte tail gradient flow
    long_c04 = next(
        c for c in dev_cases_64
        if c["family"] == "C04_COMPOSITIONAL_SCAN" and len(str(c["gold_text"]).encode("utf-8")) > 20
    )
    first_tok = str(long_c04["gold_text"]).split()[0]
    bad_suffix = "I_JUMP" if str(long_c04["gold_text"]).split()[-1] != "I_JUMP" else "I_LOOK"
    corrupt_c04 = parse_and_score_public_typed_output(
        str(long_c04["prompt"]), f"ANSWER:{first_tok} {bad_suffix}", long_c04
    )
    assert corrupt_c04["correct"] is False
    assert corrupt_c04["corrupted_suffix"] is True
    good_c04 = parse_and_score_public_typed_output(
        str(long_c04["prompt"]), f"ANSWER:{long_c04['gold_text']}", long_c04
    )
    assert good_c04["correct"] is True

    # Full-sequence gradient from target bytes at index >= 20
    tiny_spec = build_x3_backbone_spec("tiny")
    m_grad = X3CognitiveModel("Base", tiny_spec, seed=7)
    target_str = f"ANSWER:{long_c04['gold_text']}"
    ids, elig, ap = encode_prompt_and_target_pair(str(long_c04["prompt"]), target_str, max_length=m_grad.max_ctx)
    ans_start = ap + 1
    assert len(ids) - ans_start > 20
    toks = torch.tensor([ids], dtype=torch.long)
    segs = torch.zeros_like(toks)
    pos, mask = packed_layout(segs, torch_module=torch)
    out = m_grad(toks, pos, mask, action_positions=torch.tensor([ap]))
    tail_mask = torch.zeros(len(ids) - 1, dtype=torch.float32)
    tail_mask[(ans_start - 1 + 20) :] = 1.0
    assert float(tail_mask.sum().item()) > 0.0
    nll = F.cross_entropy(out["logits"][0, :-1, :], toks[0, 1:], reduction="none")
    tail_loss = (nll * tail_mask).sum() / tail_mask.sum()
    tail_loss.backward()
    emb_grad = float(m_grad.core.embedding.weight.grad.abs().sum().item())
    assert math.isfinite(emb_grad) and emb_grad > 0.0

    # 4. NaN rejection, C03 dimension mismatch, C09 Bellman 0.02 vs 0.35 (1e-8), C10 rational/unit eq
    c10_case = next(c for c in dev_cases_64 if c["family"] == "C10_LANGUAGE_MATH_TRANSFER")
    nan_res = parse_and_score_public_typed_output(str(c10_case["prompt"]), "ANSWER:NaN", c10_case)
    assert nan_res["valid"] is False and nan_res["correct"] is False

    c03_case = next(c for c in dev_cases_64 if c["family"] == "C03_GRID_TRANSFORM_ARC")
    bad_grid = parse_and_score_public_typed_output(str(c03_case["prompt"]), "ANSWER:01/23/01/23", c03_case)
    assert bad_grid["valid"] is False and bad_grid["syntax_reason"] == "GRID_EXTENT_MISMATCH"

    c09_bellman = verify_c09_budget_reversal_oracle(p_cert=0.2, low_cost=0.02, high_cost=0.35, tol=1e-8)
    assert c09_bellman["reversal_verified"] is True

    # Build a matched C10 rational equivalence test from gold integer/fraction + exact unit
    gold_c10_parts = str(c10_case["gold_text"]).split()
    gold_unit = gold_c10_parts[1] if len(gold_c10_parts) > 1 else ""
    from fractions import Fraction
    gf = Fraction(gold_c10_parts[0])
    scaled = f"{gf.numerator * 2}/{gf.denominator * 2}"
    eq_out = f"ANSWER:{scaled} {gold_unit}".strip()
    assert parse_and_score_public_typed_output(str(c10_case["prompt"]), eq_out, c10_case)["correct"] is True
    bad_unit = "kelvin" if gold_unit != "kelvin" else "radians"
    assert parse_and_score_public_typed_output(str(c10_case["prompt"]), f"ANSWER:{scaled} {bad_unit}", c10_case)["correct"] is False

    # 5. Split disjointness & isomorphic renamed leak rejection + PublicOracleTypedDecoder 1.0
    splits = {
        sp: build_x3_cognitive_battery(split=sp, per_family=20, shot_counts=(0, 1, 4, 16, 64))["cases"]
        for sp in ("train", "dev", "confirm", "ood_transfer")
    }
    disj = verify_x3_split_disjointness(splits)
    assert disj["disjoint"] is True and disj["overlap_count"] == 0

    # Exact prompt leak
    with pytest.raises(ValueError, match="SPLIT_LEAK_DETECTED"):
        verify_x3_split_disjointness({"train": splits["train"], "confirm": [*splits["confirm"], dict(splits["train"][0])]})

    # Isomorphic renamed C06 Horn-clause leak (different predicate names, canon_sig, prompt; same AST)
    tr_c06 = next(c for c in splits["train"] if c["family"] == "C06_OPEN_WORLD_DEDUCTION")
    iso_clone = dict(tr_c06)
    spec = dict(tr_c06["problem_spec"])
    rename_map = {k: f"z{i}" for i, k in enumerate(sorted(set(spec["facts"]) | {spec["query"]} | {h for _, h, _ in spec["rules"]} | {p for cs, _, _ in spec["rules"] for p in cs}))}
    iso_clone["split"] = "confirm"
    iso_clone["case_id"] = "C06:confirm:planted_iso"
    iso_clone["canon_sig"] = "c06:confirm:planted_iso_unique_sig"
    iso_clone["prompt"] = "C06|DED|F:z0=T|R:z0->z1:T|Q:z1?"
    iso_clone["structural_hash"] = ""
    iso_clone["problem_spec"] = {
        "facts": {rename_map[k]: v for k, v in spec["facts"].items()},
        "rules": [(tuple(rename_map[c] for c in cs), rename_map[h], pol) for cs, h, pol in spec["rules"]],
        "query": rename_map[spec["query"]],
        "depth": spec["depth"],
    }
    assert canonical_structural_hash(iso_clone) == canonical_structural_hash(tr_c06)
    with pytest.raises(ValueError, match="SPLIT_LEAK_DETECTED"):
        verify_x3_split_disjointness({"train": splits["train"], "confirm": [*splits["confirm"], iso_clone]})

    # PublicOracleTypedDecoder 1.0 through evaluate_x3_batched
    dec_res = verify_perfect_public_decoder_through_evaluator(splits["dev"], batch_size=32)
    assert dec_res["passed"] is True and dec_res["micro_mean"] == 1.0 and dec_res["worst_dimension_floor"] == 1.0


def test_gate3_evaluator_ce_equivalence_warmup_and_causal_mechanisms() -> None:
    tiny_spec = build_x3_backbone_spec("tiny")

    # 1. Batched vs serial exact evaluator parity across all 10 C families (20 cases)
    bat20 = build_x3_cognitive_battery(split="dev", per_family=2, shot_counts=(0, 1, 4, 16, 64))["cases"]
    assert len(bat20) == 20
    m_eval = X3CognitiveModel("P1-Alpha", tiny_spec, seed=7)
    ev_b8 = evaluate_x3_batched(m_eval, bat20, batch_size=8, device="cpu", use_generative_bytes=True)
    ev_b1 = evaluate_x3_batched(m_eval, bat20, batch_size=1, device="cpu", use_generative_bytes=True)
    for r8, r1 in zip(ev_b8, ev_b1):
        assert float((r8["action_logits"] - r1["action_logits"]).abs().max().item()) <= 1e-4
        assert r8["predicted_text"] == r1["predicted_text"]
        assert r8["correct"] == r1["correct"]

    # 2. 4-microbatch fixed-weight pure-CE gradient equivalence on 16-world pack
    pool64 = build_x3_cognitive_battery(split="train", per_family=7, shot_counts=(0, 1, 4, 16, 64))["cases"][:64]
    groups_16 = [pool64[i * 4 : (i + 1) * 4] for i in range(16)]
    eq_res = verify_fixed_weight_4microbatch_ce_equivalence(X3CognitiveModel("P", tiny_spec, seed=7), groups_16, atol=1e-4)
    assert eq_res["equivalent"] is True and eq_res["max_grad_abs_diff"] <= 1e-4

    # 3. D3 warmup + linear ramp schedule + interleaved D1-H / D2 consumed ABSTAIN parity
    assert d3_warmup_ramp_weights(0) == (0.0, 0.0)
    assert d3_warmup_ramp_weights(591) == (0.0, 0.0)
    mid_eq, mid_cf = d3_warmup_ramp_weights(592 + 296)
    assert abs(mid_eq - 0.05) < 1e-9 and abs(mid_cf - 0.05) < 1e-9
    assert d3_warmup_ramp_weights(1184) == (0.1, 0.1)

    d1h_d2_audit = audit_d1h_d2_schedule_and_preceding_observations(
        quota_per_family=512, total_updates=2368, seeds=(7, 19, 41), d1h_sampling="interleaved"
    )
    assert d1h_d2_audit["population"]["d1h_states"] == 10410
    assert d1h_d2_audit["population"]["d2_groups"] == 10410
    assert {s: d1h_d2_audit["d1h_by_seed"][s]["consumed_abstain"] for s in (7, 19, 41)} == {7: 3730, 19: 3743, 41: 3728}
    assert {s: d1h_d2_audit["d2_by_seed"][s]["consumed_abstain"] for s in (7, 19, 41)} == {7: 3752, 19: 3728, 41: 3752}

    # 4. Frozen U & evidence discrimination (0/1/2/4/8 steps, 3 cloned decision points, 4 donor conditions, sham <= 1e-6)
    u_models = {
        "U2-Fix": X3CognitiveModel("U2-Fix", tiny_spec, seed=7),
        "U4-Fix": X3CognitiveModel("U4-Fix", tiny_spec, seed=7),
        "U-2-native": build_legacy_x1_scaled_model("U-2", scale="tiny", init_seed=7)["model"],
    }
    u_res = run_frozen_u_and_evidence_discrimination(u_models, steps_sweep=(0, 1, 2, 4, 8), num_pairs=4, seed=7)
    assert tuple(u_res["decision_points"]) == X3_DISCRIMINATION_DECISION_POINTS
    assert tuple(u_res["donor_conditions"]) == X3_INTERCHANGE_DONOR_CONDITIONS
    for name in ("U2-Fix", "U4-Fix", "U-2-native"):
        assert set(u_res["models"][name]["by_steps"].keys()) == {0, 1, 2, 4, 8}
        assert u_res["models"][name]["hidden_anchor_interchange"]["identical_state_sham_max_logit_diff"] <= 1e-6

    # 5. Certified 4/5-hop interior-edge swap + unused-edge negative control + R1-Alpha lesions
    r1_m = X3CognitiveModel("R1-Alpha", tiny_spec, seed=7)
    p1_m = X3CognitiveModel("P1-Alpha", tiny_spec, seed=7)
    swap_res = run_four_five_hop_interior_edge_swap({"R1-Alpha": r1_m, "P1-Alpha": p1_m}, base_graphs_per_depth=4, seed=31001)
    assert swap_res["walker_certified"] is True
    assert swap_res["irrelevant_controls_evaluated"] > 0

    # Sham vs intact parity & active lesion contexts on R1-Alpha / CounterexampleReader / OperatorTransportBank
    sample_toks = torch.tensor([[1, 65, 66, 61, 67, 3]], dtype=torch.long)
    sample_segs = torch.zeros_like(sample_toks)
    sample_pos, sample_mask = packed_layout(sample_segs, torch_module=torch)
    with torch.no_grad():
        intact = r1_m(sample_toks, sample_pos, sample_mask)["action_logits"]
        with r1_m.ablate_mechanism("sham"):
            sham = r1_m(sample_toks, sample_pos, sample_mask)["action_logits"]
        assert float((intact - sham).abs().max().item()) <= 1e-6
        with r1_m.ablate_mechanism("op_branch_lesion"):
            op_lesion = r1_m(sample_toks, sample_pos, sample_mask)["action_logits"]
        assert torch.isfinite(op_lesion).all()
    for scr in ("CounterexampleReader", "OperatorTransportBank"):
        sm = X3CognitiveModel(scr, tiny_spec, seed=7)
        mode = "shuffle_counterexample" if scr == "CounterexampleReader" else "swap_alignment"
        with sm.ablate_mechanism(mode):
            out_s = sm(sample_toks, sample_pos, sample_mask)["action_logits"]
            assert torch.isfinite(out_s).all()


def test_gate4_fault_injection_resume_disk_missing_parent_and_r0_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "b" * 40
    tiny_spec_sha = build_x3_backbone_spec("tiny").sha256()
    base_job = {
        "job_id": "G4-RESUME-BASE",
        "stage": "R3_ALPHA",
        "architecture": "Base",
        "objective": "CE",
        "scale": "tiny",
        "init_seed": 7,
        "order_seed": 7,
        "updates": 4,
        "source_commit": commit,
        "spec_sha256": tiny_spec_sha,
    }

    # 1. Mid-update crash/resume exact parity (0->2 resumed to 4 == uninterrupted 0->4)
    unint = run_x3_single_job(
        job=base_job, work_root=tmp_path / "unint", device="cpu", canary=False, resume=False, ckpt_every=2, updates_override=4
    )
    part = run_x3_single_job(
        job=base_job, work_root=tmp_path / "resumed", device="cpu", canary=False, resume=False, ckpt_every=2, updates_override=2
    )
    resumed = run_x3_single_job(
        job=base_job, work_root=tmp_path / "resumed", device="cpu", canary=False, resume=True, ckpt_every=2, updates_override=4
    )
    assert part["completed_updates"] == 2 and resumed["completed_updates"] == 4
    assert unint["checkpoint_sha256"] == resumed["checkpoint_sha256"]

    # 2. Dependency-aware final optimizer retention (preserve_final_optimizer True vs False)
    keep_job = {**base_job, "job_id": "G4-KEEP-OPT", "updates": 2, "preserve_final_optimizer": True}
    drop_job = {**base_job, "job_id": "G4-DROP-OPT", "updates": 2, "preserve_final_optimizer": False}
    keep_rec = run_x3_single_job(
        job=keep_job, work_root=tmp_path / "keep_opt", device="cpu", canary=False, resume=False, ckpt_every=2, updates_override=2
    )
    drop_rec = run_x3_single_job(
        job=drop_job, work_root=tmp_path / "drop_opt", device="cpu", canary=False, resume=False, ckpt_every=2, updates_override=2
    )
    keep_opt_bin = tmp_path / "keep_opt" / "c" / "m" / "objects" / keep_rec["checkpoint_sha256"] / "optimizer.bin"
    drop_opt_bin = tmp_path / "drop_opt" / "c" / "m" / "objects" / drop_rec["checkpoint_sha256"] / "optimizer.bin"
    assert keep_opt_bin.stat().st_size > 4096
    assert drop_opt_bin.stat().st_size < 2048
    assert keep_opt_bin.stat().st_size > 10 * drop_opt_bin.stat().st_size

    # 3. Missing-parent isolation alongside healthy independent job in run_x3_campaign
    healthy_job = {
        "job_id": "G4-HEALTHY",
        "stage": "R3_ALPHA",
        "architecture": "Base",
        "objective": "CE",
        "scale": "tiny",
        "init_seed": 7,
        "order_seed": 7,
        "updates": 1,
        "confirmation_eligible": False,
        "source_commit": commit,
        "spec_sha256": tiny_spec_sha,
    }
    orphan_child = {
        **healthy_job,
        "job_id": "G4-ORPHAN-CHILD",
        "stage": "R7_ADAPTATION",
        "parent_job_id": "MISSING-PARENT-XYZ",
    }

    def _in_proc(_wspec, _env, cmd) -> int:
        spec_p = Path(cmd[cmd.index("--job-json") + 1])
        wroot = Path(cmd[cmd.index("--work-root") + 1])
        spec_doc = json.loads(spec_p.read_text(encoding="utf-8"))
        run_x3_single_job(job=spec_doc, work_root=wroot, device="cpu", canary=True, resume=True, ckpt_every=1, updates_override=1)
        return 0

    iso_rep = run_x3_campaign(
        work_dir=tmp_path / "iso_work",
        export_dir=tmp_path / "iso_export",
        source_commit=commit,
        devices=("cpu",),
        canary=True,
        in_process_runner=_in_proc,
        jobs_override=[healthy_job, orphan_child],
    )
    assert "G4-ORPHAN-CHILD" in iso_rep["blocked_missing_parent_job_ids"]
    assert iso_rep["completed_count"] == 1
    assert (tmp_path / "iso_work" / "jobs" / "G4-HEALTHY" / "JOB_RECEIPT.json").is_file()

    # 4. Full-disk export guard cleanly surfaces DISK_FULL_INSUFFICIENT_EXPORT_SPACE
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: SimpleNamespace(total=100_000_000, used=99_999_990, free=10))
    disk_rep = run_x3_campaign(
        work_dir=tmp_path / "disk_work",
        export_dir=tmp_path / "disk_export",
        source_commit=commit,
        devices=("cpu",),
        canary=True,
        in_process_runner=_in_proc,
        jobs_override=[healthy_job],
    )
    assert disk_rep.get("export_error") == "DISK_FULL_INSUFFICIENT_EXPORT_SPACE"
    assert disk_rep["analysis_archive_verified"] is False

    # 5. R0 Run 3 reconciliation (49/192) & R0-R7 program ledger & n=3 statistical summary
    local_run3_zip = Path(r"C:\Users\ankit\Downloads\ELENDRIL_V8_X2_ANALYSIS (2).zip")
    archives = [local_run3_zip] if local_run3_zip.is_file() else ()
    r0 = reconcile_r0_run3_obligations(source_commit=commit, archive_paths=archives)
    if local_run3_zip.is_file():
        rec0 = r0["verified_archive_receipts"][0]
        assert rec0["bytes"] == RUN3_EXPECTED_ARCHIVE_BYTES
        assert rec0["sha256"] == RUN3_EXPECTED_ARCHIVE_SHA256
    assert r0["run3_unique_non_a0_count"] == 43
    assert r0["historical_a0_count"] == 6
    assert r0["credited_x2_obligations"] == 49
    assert r0["total_x2_obligations"] == 192
    assert r0["pending_mandatory"] == 125
    assert r0["pending_conditional"] == 18
    assert r0["pending_total"] == 143
    assert r0["next_runnable_gpu_job"] == "X1-U1-S20-U-4-standard-s19"
    assert r0["x3_unlocked"] is False

    full_ledger = build_r0_r7_program_ledger(source_commit=commit, r0_audit=r0)
    assert len(full_ledger["legacy_x3_migration_table"]) == 57
    assert full_ledger["legacy_x3_migration_counts"] == {
        "unchanged": 6,
        "scientifically_superseded_by_exact_output_equivalent": 39,
        "open_with_reason": 12,
    }
    assert full_ledger["r6_r7_parent_child_pairs"] == 12

    stat_n3 = summarize_paired_arm_statistics(
        contrast_id="U2_Fix_vs_Unanchored",
        control_arm="U2-GatedUnanchored",
        treatment_arm="U2-Fix",
        control_seed_scores={7: 0.40, 19: 0.42, 41: 0.41},
        treatment_seed_scores={7: 0.50, 19: 0.52, 41: 0.51},
    )
    assert stat_n3["min_two_sided_seed_sign_flip_p"] == 0.25
    assert stat_n3["n_seeds"] == 3


def test_gate5_probe_cpu_handoff_and_auto_notebook_defaults() -> None:
    import ast
    from anra_v5.planthir.cartridge import _collect_capsule_arrays
    from anra_v5.planthir.causal import _eval_ridge_classifier, _fit_ridge_classifier, run_representation_linear_probe
    from experiments.x3.module import X3CognitiveKernel, X3CognitivePolicy, _wrap_planthir_worlds
    from tools.build_elendril_x3_notebook import build_x3_notebook_dict

    tiny_spec = build_x3_backbone_spec("tiny")
    model = X3CognitiveModel("Base", tiny_spec, seed=7)
    policy = X3CognitivePolicy(model)
    kern = X3CognitiveKernel()
    tr_cases = build_x3_cognitive_battery(split="train", per_family=1, shot_counts=(0,))["cases"][:6]
    ev_cases = build_x3_cognitive_battery(split="dev", per_family=1, shot_counts=(0,))["cases"][:4]
    tr_worlds = _wrap_planthir_worlds(tr_cases)
    ev_worlds = _wrap_planthir_worlds(ev_cases)

    obs = [kern.observe(kern.initial_state(w)) for w in ev_worlds]
    reps = policy.extract_representations(obs)
    assert reps["decision_hidden"].device.type == "cpu"
    assert reps["decision_hidden"].requires_grad is False

    cap = _collect_capsule_arrays(policy, kern, ev_worlds)
    assert cap["decision_hidden"].device.type == "cpu"
    probe_res = run_representation_linear_probe(policy, kern, tr_worlds, ev_worlds)
    assert 0.0 <= probe_res["probe_operation_accuracy"] <= 1.0

    x_tr = torch.randn(12, 8)
    y_tr = torch.tensor([i % 4 for i in range(12)], dtype=torch.int64)
    w = _fit_ridge_classifier(x_tr, y_tr, 4, l2_reg=1.0)
    assert w.device.type == "cpu"
    acc = _eval_ridge_classifier(w, x_tr, y_tr)
    assert 0.0 <= acc <= 1.0

    # Zero-edit automatic Kaggle defaults: OWNER_AUTHORIZED = True & RESERVE_MINUTES = 45
    nb = build_x3_notebook_dict(source_commit="c" * 40)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    cfg_src = "".join(nb["cells"][1]["source"])
    assert "OWNER_AUTHORIZED = True" in cfg_src
    assert "RESERVE_MINUTES = 45" in cfg_src
    nb_dis = build_x3_notebook_dict(source_commit="c" * 40, owner_authorized_default=False)
    assert "OWNER_AUTHORIZED = False" in "".join(nb_dis["cells"][1]["source"])


def test_gate6_all_29_architectures_fp16_autocast_padded_finite() -> None:
    """Verify all 29 X3 architectures run finite forward + backward under float16 autocast with padding."""
    tiny_spec = build_x3_backbone_spec("tiny")
    # Padded 2-sample batch with A..P symbols (65..80), non-A..P delimiter (61), ANSWER (3), and trailing pad=0
    toks = torch.tensor(
        [
            [1, 65, 66, 61, 67, 3, 0, 0],
            [1, 68, 61, 69, 3, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    segs = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, -1, -1],
            [0, 0, 0, 0, 0, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    pos, mask = packed_layout(segs, torch_module=torch)
    apos = torch.tensor([5, 4], dtype=torch.long)

    for arch in X3_ARCHITECTURES:
        m = X3CognitiveModel(arch, tiny_spec, seed=7)
        m.train()
        m.zero_grad(set_to_none=True)
        with torch.amp.autocast("cpu", dtype=torch.float16):
            out = m(toks, pos, mask, action_positions=apos)
            loss = (
                out["logits"].float().mean()
                + out["action_logits"].float().mean()
                + 0.1 * out["uncertainties"].float().mean()
                + 0.02 * out["ponder_cost"].float().mean()
            )
        assert torch.isfinite(out["logits"]).all(), f"Non-finite logits under FP16 on {arch}"
        assert torch.isfinite(out["action_logits"]).all(), f"Non-finite action_logits under FP16 on {arch}"
        assert torch.isfinite(out["uncertainties"]).all(), f"Non-finite uncertainties under FP16 on {arch}"
        assert torch.isfinite(out["ponder_cost"]).all(), f"Non-finite ponder_cost under FP16 on {arch}"
        assert math.isfinite(float(loss.item())), f"Non-finite FP16 loss on {arch}"
        loss.backward()
        dead = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        assert not dead, f"Dead grads under FP16 on {arch}: {dead}"
        non_finite_grads = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not non_finite_grads, f"Non-finite grads under FP16 on {arch}: {non_finite_grads}"

    # Regression for Kaggle Stage 5 UnicodeDecodeError when byte LM emits invalid UTF-8 bytes (e.g. 0xB5)
    m_bad_utf8 = X3CognitiveModel("Base", tiny_spec, seed=7)
    with torch.no_grad():
        m_bad_utf8.lm_bias.fill_(-1e4)
        m_bad_utf8.lm_bias[4 + 0xB5] = 1e4
    gen_bad = m_bad_utf8.generate_bytes("C01|RULE|k=1|1*1=3|Q:2*2=?", max_new_bytes=4)
    assert gen_bad["valid_utf8"] is False
    assert isinstance(gen_bad["text"], str)
    bad_case = build_x3_cognitive_battery(split="dev", per_family=1, shot_counts=(0,))["cases"][0]
    ev_bad = evaluate_x3_batched(m_bad_utf8, [bad_case], batch_size=1, device="cpu", use_generative_bytes=True)
    assert len(ev_bad) == 1 and ev_bad[0]["correct"] is False
