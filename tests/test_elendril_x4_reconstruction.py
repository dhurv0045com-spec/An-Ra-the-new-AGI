"""Comprehensive CPU verification for AN-RA V8-X4 Fundamental Reconstruction & Defects A-P."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v8x.contracts import X1_FAMILIES
from experiments.x4.battery import (
    PublicOracleX4Decoder,
    X4_22_FAMILIES,
    X4_FAMILY_GROUPS,
    build_l0_l5_diagnostic_suite,
    build_x01_x12_interventions,
    build_x4_battery,
    build_x4_curriculum_schedule,
    canonical_structural_hash,
    diagnose_c03_c04_c10_failure_modes,
    evaluate_x4_cases,
    format_x4_typed_target,
    score_x4_multidimensional_profile,
    verify_x4_split_disjointness,
)
from experiments.x4.integrity import run_foundational_integrity_gate
from experiments.x4.models import X4CognitiveModel, X4_ARCHITECTURES, build_x4_backbone_spec
from experiments.x4.module import (
    X4_PAIRED_CONTRASTS,
    build_x4_cycle_ledger,
    evaluate_capability_preservation,
    evaluate_x4_learning_modes,
    evaluate_x4_qualification_thresholds,
    run_x4_campaign,
)
from experiments.x4.objectives import X4_OBJECTIVES, build_x4_matched_batches, compute_x4_objective_losses


def test_cycle1_foundational_integrity_gate(tmp_path: Path) -> None:
    rep = run_foundational_integrity_gate(tmp_path)
    assert rep["all_passed"] is True
    assert rep["3_2_information_flow_integrity"]["architectures_checked"] == len(X4_ARCHITECTURES)
    assert rep["3_2_information_flow_integrity"]["max_future_leak_diff_across_all_archs"] < 1e-6
    assert rep["3_2_information_flow_integrity"]["positive_leaky_control_diff"] > 1e-4
    assert rep["3_4_evaluation_integrity"]["oracle_22fam_accuracy"] == 1.0
    assert (tmp_path / "X4_CYCLE1_INTEGRITY_GATE.json").is_file()


def test_defects_a_to_p_explicit_regressions() -> None:
    tiny = build_x4_backbone_spec("tiny")
    # Defect A: R-Ref and R-NoEdge explicitly preserve pointer access
    r_ref = X4CognitiveModel("R-Ref", tiny, seed=7)
    r_no = X4CognitiveModel("R-NoEdge", tiny, seed=7)
    assert r_ref.uses_pointer is True and hasattr(r_ref, "edge_mlp")
    assert r_no.uses_pointer is True and not hasattr(r_no, "edge_mlp")

    # S50 rejection & S20 parameter bound (~20M regime ONLY)
    with pytest.raises(ValueError):
        build_x4_backbone_spec("S50")
    s20_spec = build_x4_backbone_spec("S20")
    assert 19_000_000 <= s20_spec.parameter_receipt().total <= 22_000_000

    # Defect O: Repaired C03/C04/C10 train coverage includes atomic verbs/dirs/mults/rationals/colors
    tr_bat = build_x4_battery(split="train", per_family=16, phase=2)["cases"]
    c04_prompts = " ".join(c["prompt"] for c in tr_bat if c["family"] == "C04_COMPOSITIONAL_SCAN")
    for tok in ("walk", "run", "jump", "look", "left", "right", "opposite", "around", "twice", "thrice", "and", "after"):
        assert tok in c04_prompts, f"Missing atomic C04 train token: {tok}"
    c10_golds = [c["gold_text"] for c in tr_bat if c["family"] == "C10_LANGUAGE_MATH_TRANSFER"]
    assert any("/" in g for g in c10_golds), "C10 train must include reduced rationals"
    # Defect N: No C09 prompt leaks split string
    c09_prompts = [c["prompt"] for c in tr_bat if c["family"] == "C09_METACOGNITION_SELF_CHECK"]
    assert all("split=" not in p for p in c09_prompts)

    # Defect I: split-agnostic canonical_structural_hash invariance + X01..X12 interventions
    x_cases = [c for c in tr_bat if c["family"] in X1_FAMILIES][:4]
    intervs = build_x01_x12_interventions(x_cases)
    assert len(intervs) >= 4
    h_orig = canonical_structural_hash(x_cases[0])
    c_mod = dict(x_cases[0], split="confirm")
    assert canonical_structural_hash(c_mod) == h_orig

    # Defect L: zero self-contrasts in X4_PAIRED_CONTRASTS & zero S50 jobs in 4-cycle ledger
    ledger = build_x4_cycle_ledger(source_commit="0" * 40)
    assert ledger["s50_jobs"] == 0 and ledger["s20_jobs"] == ledger["total_gpu_jobs"] == 81
    for cid, c_k, t_k in X4_PAIRED_CONTRASTS:
        assert c_k != t_k, f"Self-contrast found: {cid}"


def test_curriculum_fresh_stream_vs_fixed_and_diagnostics() -> None:
    sched_fresh = build_x4_curriculum_schedule(seed=7, total_updates=4, batch_size=8, data_regime="fresh_stream", curriculum_mode="curriculum_p2c")
    sched_fixed = build_x4_curriculum_schedule(seed=7, total_updates=4, batch_size=8, data_regime="fixed_400", curriculum_mode="random_interleaved")
    matched_a = build_x4_matched_batches(sched_fresh, is_schedule=True)
    matched_b = build_x4_matched_batches(sched_fixed, is_schedule=True)
    assert matched_a["exposure_manifest"]["unique_case_ids_count"] == 32
    assert matched_b["exposure_manifest"]["batches_count"] == 4

    tiny = build_x4_backbone_spec("tiny")
    m_op = X4CognitiveModel("P1-OpBank", tiny, seed=7)
    dv = build_x4_battery(split="dev", per_family=2)["cases"]
    tr = build_x4_battery(split="train", per_family=2)["cases"]
    oo = build_x4_battery(split="ood_transfer", per_family=2)["cases"]
    l_suite = build_l0_l5_diagnostic_suite(train_cases=tr, dev_cases=dv, ood_cases=oo, per_level_limit=1)
    preds = evaluate_x4_cases(m_op, l_suite[:12], batch_size=6)
    assert preds[0]["op_routing_weights"] is not None
    prof = score_x4_multidimensional_profile(preds, l_suite[:12])
    assert "unified_brier_score" in prof and "capability_axes" in prof
    c_diag = diagnose_c03_c04_c10_failure_modes(m_op, train_cases=tr, unseen_same_fmt=dv, unseen_reformatted=oo)
    assert "unseen_same_format" in c_diag["splits"]

    # Capability preservation & honest threshold checks
    pres = evaluate_capability_preservation(prof, prof)
    assert pres["preserved"] is True
    qual = evaluate_x4_qualification_thresholds(prof, prof)
    assert qual["tier"] == "NOT_QUALIFIED"


def test_repaired_5_learning_modes_and_end_to_end_canary_campaign(tmp_path: Path) -> None:
    tiny = build_x4_backbone_spec("tiny")
    m = X4CognitiveModel("ANRA-X4-Int", tiny, seed=7)
    dev = build_x4_battery(split="dev", per_family=2)["cases"]
    ood = build_x4_battery(split="ood_transfer", per_family=2)["cases"]
    icl = {0: [c for c in dev if c["family"] == "C01_NOVEL_OP_INDUCTION"][:1], 4: [c for c in dev if c["family"] == "C01_NOVEL_OP_INDUCTION"][:1]}
    modes = evaluate_x4_learning_modes(
        m,
        icl_cases_by_shot=icl,
        state_adapt_sequence=dev[:2],
        grad_adapt_demos=dev[:2],
        grad_adapt_queries=ood[:2],
        retention_queries=[c for c in dev if c["family"] == "X01"][:2],
        tool_free_cases=[c for c in ood if c["family"] == "C03_GRID_TRANSFORM_ARC"][:2],
        cross_domain_cases=[c for c in ood if c["family"] == "C10_LANGUAGE_MATH_TRANSFER"][:2],
    )
    assert modes["mode4_distinct_transfer_sets_verified"] is True
    assert "mode2_unrelated_history_acc" in modes and "mode3_post_retention_acc" in modes

    # End-to-end 4-cycle canary campaign (small 4-job slice across C1..C4)
    src = "a" * 40
    subset_jobs = [
        next(j for j in build_x4_cycle_ledger(source_commit=src, updates=3, cycles=("C1", "C2", "C3", "C4"))["jobs"] if j["cycle"] == cyc)
        for cyc in ("C1", "C2", "C3", "C4")
    ]
    wdir = tmp_path / "camp_work"
    edir = tmp_path / "camp_export"
    summary = run_x4_campaign(
        work_dir=wdir, export_dir=edir, source_commit=src, devices=("cpu",), canary=True, jobs_override=subset_jobs
    )
    assert summary["status"] == "COMPLETE"
    assert summary["integrity_gate_passed"] is True
    assert summary["s50_jobs"] == 0
    assert summary["analysis_archive_verified"] is True
    assert (wdir / "CONFIRMATION_LOCK.json").is_file()
    assert (wdir / "X4_CAPABILITY_REGISTRY.json").is_file()
    assert (wdir / "X4_FAILURE_EVIDENCE_REGISTRY.json").is_file()


def test_palantiri_level7_cli_chain_for_elendril_x4(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PLANTHIR_BACKEND_MODULES"] = "experiments.x4.module"
    spec_p = tmp_path / "x4_spec.json"
    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "init", "--backend", "elendril_x4", "--out", str(spec_p)], check=True, env=env)
    doc = json.loads(spec_p.read_text(encoding="utf-8"))
    src = doc["source_commit"]
    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "validate", "--spec", str(spec_p), "--canary"], check=True, env=env)
    run_root = tmp_path / "cli_run"
    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "run", "--spec", str(spec_p), "--work-root", str(run_root), "--canary", "--device", "cpu"], check=True, env=env)
    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "verify-bundle", "--bundle-dir", str(run_root / "result_bundle"), "--expected-source-commit", src], check=True, env=env)
    subprocess.run([sys.executable, "-m", "anra_v5.palantiri", "replay-bundle", "--bundle-dir", str(run_root / "result_bundle"), "--expected-source-commit", src, "--device", "cpu", "--output", str(tmp_path / "REPLAY.json")], check=True, env=env)
    rep = json.loads((tmp_path / "REPLAY.json").read_text(encoding="utf-8"))
    assert rep["status"] == "REPLAY_VERIFIED"


def test_all_x4_architectures_padded_amp_forward_backward_and_0xb5_shield() -> None:
    tiny = build_x4_backbone_spec("tiny")
    bat = build_x4_battery(split="train", per_family=1, families=("X01", "C04_COMPOSITIONAL_SCAN", "C10_LANGUAGE_MATH_TRANSFER"))["cases"]
    matched = build_x4_matched_batches(bat, seed=7, batch_size=len(bat))
    b0 = matched["batches"][0]
    for arch in X4_ARCHITECTURES:
        m = X4CognitiveModel(arch, tiny, seed=7)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        # Exercises compute_x4_objective_losses + backward + 0xB5 lossy decode parity
        res = compute_x4_objective_losses(m, b0, objective="D4-ECCO", update_idx=2, total_updates=4, use_amp=False)
        opt.zero_grad(set_to_none=True)
        res["total_loss_t"].backward()
        dead = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        assert not dead, f"Dead grads in D4-ECCO on {arch}: {dead}"
        opt.step()
    txt, ok = UTF8.decode_lossy([UTF8.encode("X")[0], 4 + 0xB5, UTF8.encode("1")[0]], strip_special=True)
    assert isinstance(txt, str) and ok is False
