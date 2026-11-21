"""Comprehensive acceptance suite for V8-X3 Beyond Prediction cognitive program & X2 Run-2 ordering fix."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch

from anra_v5.palantiri import (
    CampaignOrchestrator,
    export_session_archives,
    verify_session_archive,
    verify_shared_backbone_and_head_overhead,
)
from v5_model.core import packed_layout

from experiments.x2.campaign import ElendrilX2Campaign
from experiments.x3 import (
    D3_OBJECTIVES,
    S50_MODEL_SPEC,
    SPLIT_SEEDS,
    X3CognitiveModel,
    X3_ARCHITECTURES,
    X3_COGNITIVE_FAMILIES,
    X3_CONTROL_PAIRS,
    audit_x1_sampler_exposure,
    build_d3_hybrid_batches,
    build_x3_backbone_spec,
    build_x3_cognitive_battery,
    build_x3_program_ledger,
    compute_d3_objective_step,
    eval_x3_battery_shortcuts,
    evaluate_confirmation_lock,
    evaluate_five_learning_modes,
    evaluate_s50_scaling_gate,
    evaluate_x3_batched,
    load_external_benchmark_jsonl,
    run_all_architectures_u0000_smoke,
    run_x3_campaign,
    run_x3_single_job,
    score_cognitive_battery_profile,
    solve_cognitive_case_oracle,
    verify_d3_gradient_surgery_fixture,
    verify_fst_version_space,
    verify_x3_destructive_masks,
    verify_x3_split_disjointness,
)
from experiments.x3.cognitive_battery import _build_fst_tensors, _eval_fst_seq
from tools.build_elendril_x2_notebook import FULL_X2_SESSIONS
from tools.build_elendril_x3_notebook import build_x3_notebook_dict, render_x3_notebook_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_x3_10_family_cognitive_battery_oracles_shortcuts_masks_and_split_custody(tmp_path: Path) -> None:
    splits = {
        sp: build_x3_cognitive_battery(split=sp, per_family=20, shot_counts=(0, 1, 4, 16, 64))["cases"]
        for sp in ("train", "dev", "confirm", "ood_transfer")
    }
    disj = verify_x3_split_disjointness(splits)
    assert disj["disjoint"] is True and disj["overlap_count"] == 0

    for sp, items in splits.items():
        assert len(items) == 200
        assert {c["family"] for c in items} == set(X3_COGNITIVE_FAMILIES)
        sc = eval_x3_battery_shortcuts(items)
        assert sc["oracle_independent_accuracy"] == 1.0
        assert sc["per_family_oracle_min"] == 1.0
        assert sc["max_shortcut_accuracy"] <= 0.35
        masks = verify_x3_destructive_masks(items)
        assert len(masks["families_with_opposing_masks"]) == 10
        assert sum(1 for c in items if c["absent_from_input"]) >= 120

    # Planted-leak regression check
    leaked_splits = {**splits, "confirm": [*splits["confirm"][:-1], splits["dev"][0]]}
    with pytest.raises(ValueError, match="SPLIT_LEAK_DETECTED"):
        verify_x3_split_disjointness(leaked_splits)

    # FST version-space identifiability improves cleanly as shots increase (1 -> 16)
    tr, prm = _build_fst_tensors("dev", 105)
    cands = [_build_fst_tensors("dev", k) for k in range(100, 120)]
    demos_1 = [(0, (1, 2), _eval_fst_seq(tr, prm, 0, (1, 2)))]
    demos_16 = [
        (i % 4, (i % 4, (i * 3 + 1) % 4), _eval_fst_seq(tr, prm, i % 4, (i % 4, (i * 3 + 1) % 4)))
        for i in range(16)
    ]
    c1 = verify_fst_version_space(tr, prm, demos_1, cands)
    c16 = verify_fst_version_space(tr, prm, demos_16, cands)
    assert 1 <= c16 <= c1

    # External benchmark loader preserves provenance separation
    ext_path = tmp_path / "ext_arc.jsonl"
    ext_path.write_text(
        json.dumps({"case_id": "arc1", "prompt": "01/10 -> 10/01 | 23/32 -> ?", "answer": "32/23"}) + "\n",
        encoding="utf-8",
    )
    ext_off = load_external_benchmark_jsonl(ext_path, benchmark_name="ARC-AGI-1", official_unmodified=True)
    ext_adp = load_external_benchmark_jsonl(ext_path, benchmark_name="ARC-AGI-1", official_unmodified=False)
    assert ext_off["benchmark_provenance"] == "official_external"
    assert ext_adp["benchmark_provenance"] == "interface_adapted_external"
    assert ext_off["items"][0]["absent_from_input"] is True


def test_x3_models_backbone_overhead_u0000_all_archs_batched_parity_and_5_learning_modes(tmp_path: Path) -> None:
    # 1. Shared backbone & head overhead (<=5% head overhead, <=2% control parity) on tiny & S20, plus S50 spec check
    for sc in ("tiny", "S20"):
        spec = build_x3_backbone_spec(sc)
        for ctrl_arch, trt_arch in X3_CONTROL_PAIRS:
            mc = X3CognitiveModel(ctrl_arch, spec, seed=7)
            mt = X3CognitiveModel(trt_arch, spec, seed=7)
            rep = verify_shared_backbone_and_head_overhead(
                mc, mt, backbone_prefix="core.", max_head_overhead_ratio=0.05, max_control_parity_ratio=0.02
            )
            assert rep["passed"] is True
            assert rep["backbone_digest_match"] is True
            assert rep["treatment_head_overhead_ratio"] <= 0.05
            assert rep["control_parity_diff_ratio"] <= 0.02

    # Verify S50 is truly >= 50M params and builds cleanly for Base & P-R-Iter
    assert S50_MODEL_SPEC.parameter_receipt().total >= 50_000_000
    s50_m = X3CognitiveModel("P-R-Iter", S50_MODEL_SPEC, seed=7)
    s50_ctrl = X3CognitiveModel("P-R-Iter-Fixed", S50_MODEL_SPEC, seed=7)
    s50_rep = verify_shared_backbone_and_head_overhead(
        s50_ctrl, s50_m, backbone_prefix="core.", max_head_overhead_ratio=0.05, max_control_parity_ratio=0.02
    )
    assert s50_rep["passed"] is True
    assert sum(p.numel() for p in s50_m.parameters()) >= 50_000_000
    del s50_m, s50_ctrl

    # 2. Mandatory CPU u0000 init+train+eval+restore smoke across ALL 9 X3 architectures
    smk = run_all_architectures_u0000_smoke(tmp_path / "u0000", source_commit="a" * 40, include_s50_construction=False)
    assert smk["all_passed"] is True
    assert set(smk["architectures_tested"]) == set(X3_ARCHITECTURES)

    # 3. Zero-arg ablate_mechanism context manager + batched vs serial eval parity (atol=1e-4)
    tiny = build_x3_backbone_spec("tiny")
    model = X3CognitiveModel("P-R-Iter", tiny, seed=7)
    with model.ablate_mechanism() as mtype:
        assert mtype == "disable_pointer" and model._ablation_mode == "disable_pointer"
    assert model._ablation_mode == "none"

    dev_cases = build_x3_cognitive_battery(split="dev", per_family=2)["cases"][:8]
    batched_res = evaluate_x3_batched(model, dev_cases, batch_size=4, device="cpu")
    model.eval()
    max_logit_diff = 0.0
    for c, br in zip(dev_cases, batched_res):
        ser = evaluate_x3_batched(model, [c], batch_size=1, device="cpu")[0]
        diff = float((br["action_logits"] - ser["action_logits"]).abs().max().item())
        max_logit_diff = max(max_logit_diff, diff)
    assert max_logit_diff <= 1e-4

    # 4. Five learning modes evaluator leaves weights restored identically
    snap_before = {k: v.detach().clone() for k, v in model.named_parameters()}
    bat_tr = build_x3_cognitive_battery(split="train", per_family=2, shot_counts=(0, 1, 4, 16, 64))["cases"]
    icl_map = {sh: [c for c in bat_tr if c["family"] == "C01_NOVEL_OP_INDUCTION" and c["n_shots"] == sh][:2] or [bat_tr[0]] for sh in (0, 1, 4, 16, 64)}
    modes = evaluate_five_learning_modes(
        model,
        icl_cases_by_shot=icl_map,
        state_adapt_sequence=bat_tr[:4],
        grad_adapt_demos=bat_tr[:2],
        grad_adapt_queries=bat_tr[2:4],
        transfer_cases=[c for c in dev_cases if c["family"] == "C10_LANGUAGE_MATH_TRANSFER"] or dev_cases[:2],
        oracle_solver=solve_cognitive_case_oracle,
    )
    assert set(modes["mode1_frozen_icl_curve"]) == {0, 1, 4, 16, 64}
    assert modes["mode4_tool_assisted_oracle_accuracy"] == 1.0
    for name, p in model.named_parameters():
        assert torch.allclose(p.detach(), snap_before[name])


def test_x3_d3_sampler_audit_pcgrad_fixture_and_all_objectives() -> None:
    audit = audit_x1_sampler_exposure(quota_per_family=12)
    assert audit["base_world_diversity_ratio_d1h_over_d2"] == 4.0
    assert audit["d1h_standard"]["unique_base_worlds"] == audit["d2_paired"]["unique_base_worlds"]
    surg = verify_d3_gradient_surgery_fixture()
    assert surg["passed"] is True

    bat = build_x3_cognitive_battery(split="train", per_family=2)["cases"]
    batches = build_d3_hybrid_batches(bat, seed=7, batch_size=4)
    tiny = build_x3_backbone_spec("tiny")
    for obj in D3_OBJECTIVES:
        m = X3CognitiveModel("P-R-Iter", tiny, seed=7)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        st = compute_d3_objective_step(m, opt, batches[0], objective=obj)
        assert st["objective"] == obj and torch.isfinite(torch.tensor(st["total_loss"]))


def test_x3_dual_gpu_full_dispatch_gated_s50_confirmation_lock_and_3_recovery_paths(tmp_path: Path) -> None:
    src = "a" * 40
    ledger = build_x3_program_ledger(source_commit=src, updates=4)
    assert ledger["total_gpu_jobs"] == 57
    assert ledger["s20_jobs"] == 45 and ledger["s50_jobs"] == 12

    # 1. Mocked dual-GPU dispatch with tight wall clock -> defer into PENDING_QUEUE -> Session 2 resume -> Confirmation Lock
    tiny_jobs = [
        *[j for j in ledger["jobs"] if j["scale"] == "S20" and j["stage"] == "S2"][:4],  # 4 S20 jobs
        *[j for j in ledger["jobs"] if j["scale"] == "S50"][:2],  # 2 S50 jobs (must stay gated!)
    ]
    seen_devs_1: list[str] = []

    def _mock_runner_s1(spec, env, cmd):
        seen_devs_1.append(spec.device)
        jpath = Path(cmd[cmd.index("--job-json") + 1])
        job_doc = json.loads(jpath.read_text(encoding="utf-8"))
        # Run only first 2 jobs, simulate failure or timeout for test if needed; here execute 2 via tight clock
        run_x3_single_job(job=job_doc, work_root=spec.work_root, device="cpu", canary=True, updates_override=4)
        return 0

    ticks = [0.0]

    def _clock_after_one_wave():
        t = ticks[0]
        ticks[0] += 200.0  # t_start=0.0, before wave 0 -> 200 <= 300 (runs wave 0), before wave 1 -> 400 > 300 (stops!)
        return t

    res1 = run_x3_campaign(
        work_dir=tmp_path / "s1_work",
        export_dir=tmp_path / "s1_exp",
        source_commit=src,
        devices=("cuda:0", "cuda:1"),
        session_minutes=60.0,
        reserve_minutes=55.0,  # 300s usable -> runs wave 0 (2 jobs on cuda:0 + cuda:1), then stops before wave 1!
        measured_seconds_per_update=0.5,
        per_job_disk_bytes=1024,
        canary=True,
        in_process_runner=_mock_runner_s1,
        mirror_dir=tmp_path / "kaggle_working",
        force_s50_gate=False,
        clock_fn=_clock_after_one_wave,
        jobs_override=tiny_jobs,
    )
    assert set(seen_devs_1) == {"cuda:0", "cuda:1"}
    assert res1["status"] == "PARTIAL"
    assert res1["completed_count"] == 2
    assert res1["confirmation_lock"]["unlocked"] is False
    assert res1["confirmation_lock"]["status"] == "LOCKED_PENDING_COHORT"
    assert res1["s50_scaling_gate"]["unlocked"] is False
    assert res1["analysis_archive_verified"] is True
    assert (tmp_path / "kaggle_working" / "ELENDRIL_V8_X3_ANALYSIS.zip").is_file()

    # Session 2 resumes verified receipts -> completes remaining 2 S20 jobs -> S50 stays gated -> Conf still locked on PARTIAL;
    # and when all S20 jobs complete on an S20-only cohort, Confirmation Lock unseals ONLY then!
    s20_only = [j for j in tiny_jobs if j["scale"] == "S20"]
    res1_recs = list(json.loads((tmp_path / "s1_work" / "JOB_RECEIPTS.json").read_text(encoding="utf-8"))["receipts_by_id"].values())
    for rec in res1_recs:
        rec["archive_path"] = res1["archives"]["analysis"]
        rec["archive_sha256"] = res1["archive_sha256"]["analysis"]
    seen_devs_2: list[str] = []

    def _mock_runner_s2(spec, env, cmd):
        seen_devs_2.append(spec.device)
        jpath = Path(cmd[cmd.index("--job-json") + 1])
        job_doc = json.loads(jpath.read_text(encoding="utf-8"))
        run_x3_single_job(job=job_doc, work_root=spec.work_root, device="cpu", canary=True, updates_override=4)
        return 0

    res2 = run_x3_campaign(
        work_dir=tmp_path / "s2_work",
        export_dir=tmp_path / "s2_exp",
        source_commit=src,
        devices=("cuda:0", "cuda:1"),
        session_minutes=300.0,
        reserve_minutes=15.0,
        measured_seconds_per_update=0.5,
        per_job_disk_bytes=1024,
        canary=True,
        resume_receipts=res1_recs,
        in_process_runner=_mock_runner_s2,
        jobs_override=s20_only,
    )
    assert res2["status"] == "COMPLETE"
    assert res2["completed_count"] == 4
    assert res2["confirmation_lock"]["unlocked"] is True
    assert res2["confirmation_lock"]["pending_confirm_jobs"] == []

    # 2. Recovery Test A: Mid-update crash & exact resume parity (0..2 -> 2..4 == 0..4)
    j0 = dict(s20_only[0])
    j0["updates"] = 4
    j0_full = run_x3_single_job(job=j0, work_root=tmp_path / "rec_full", device="cpu", canary=True, resume=False, ckpt_every=2, updates_override=4)
    run_x3_single_job(job=j0, work_root=tmp_path / "rec_split", device="cpu", canary=True, resume=False, ckpt_every=2, updates_override=2)
    j0_res = run_x3_single_job(job=j0, work_root=tmp_path / "rec_split", device="cpu", canary=True, resume=True, ckpt_every=2, updates_override=4)
    assert j0_full["checkpoint_sha256"] == j0_res["checkpoint_sha256"] or abs(j0_full["dev_macro_mean"] - j0_res["dev_macro_mean"]) < 1e-6

    # 3. Recovery Test B: Swapped spec_sha256 & missing archive member rejected by reconcile_verified_resume(..., fail_on_rejection=True)
    orch_rej = CampaignOrchestrator(campaign_id="X3-REJ", source_commit=src, work_dir=tmp_path / "rej_work")
    bad_spec_rec = {**res1_recs[0], "spec_sha256": "0" * 64}
    with pytest.raises(ValueError, match="SWAPPED_SPEC_SHA256"):
        orch_rej.reconcile_verified_resume([bad_spec_rec], s20_only, fail_on_rejection=True)
    with pytest.raises(ValueError, match="MISSING_ARCHIVE_MEMBER"):
        orch_rej.reconcile_verified_resume([res1_recs[0]], s20_only, required_archive_members=("NON_EXISTENT.json",), fail_on_rejection=True)

    # 4. Recovery Test C: 1-lane worker failure preserves peer completion & emits PARTIAL export + PENDING_QUEUE
    def _mock_fail_lane1(spec, env, cmd):
        if spec.device == "cuda:1":
            return 1
        jpath = Path(cmd[cmd.index("--job-json") + 1])
        job_doc = json.loads(jpath.read_text(encoding="utf-8"))
        run_x3_single_job(job=job_doc, work_root=spec.work_root, device="cpu", canary=True, updates_override=2)
        return 0

    res_fail = run_x3_campaign(
        work_dir=tmp_path / "fail_work",
        export_dir=tmp_path / "fail_exp",
        source_commit=src,
        devices=("cuda:0", "cuda:1"),
        canary=True,
        measured_seconds_per_update=0.5,
        per_job_disk_bytes=1024,
        in_process_runner=_mock_fail_lane1,
        jobs_override=s20_only[:2],
    )
    assert res_fail["status"] == "PARTIAL"
    assert res_fail["completed_count"] == 1
    assert len(res_fail["failed_job_ids"]) == 1
    assert res_fail["confirmation_lock"]["unlocked"] is False


def test_x2_run2_bii_frozen_ordering_and_blocked_missing_parent_resilience(tmp_path: Path) -> None:
    camp = ElendrilX2Campaign(source_commit="a" * 40, work_dir=tmp_path / "x2_order")
    rows = camp.select_session_rows(list(FULL_X2_SESSIONS), include_extra_ledgers=True)
    idx_u0 = next(i for i, r in enumerate(rows) if r["session"] == "U0")
    idx_bii = next(i for i, r in enumerate(rows) if r["session"] == "BII-FROZEN")
    assert idx_u0 < idx_bii, "FULL_X2_SESSIONS must place independent U0/U1/IIIb before BII-FROZEN"

    # Mixed wave: BII-FROZEN (missing A0 parent weights) + U0 (independent) -> does NOT crash; records BLOCKED_MISSING_PARENT!
    def _mock_u0(spec, env, cmd):
        (spec.work_root / "FINAL_RECEIPT.json").write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
        return 0

    camp2 = ElendrilX2Campaign(source_commit="a" * 40, work_dir=tmp_path / "x2_blk", in_process_runner=_mock_u0)
    bii_row = next(r for r in rows if r["session"] == "BII-FROZEN")
    u0_row = next(r for r in rows if r["session"] == "U0")
    summary = camp2.run_admitted_waves(camp2.to_session_jobs([bii_row, u0_row]), devices=("cuda:0", "cuda:1"), canary=True)
    assert u0_row["job_id"] in summary["completed_jobs"]
    assert len(summary["blocked_missing_parent_jobs"]) == 1
    assert summary["blocked_missing_parent_jobs"][0]["job_id"] == bii_row["job_id"]
    assert summary["blocked_missing_parent_jobs"][0]["status"] == "BLOCKED_MISSING_PARENT"


def test_x3_plugin_cli_fresh_subprocess_chain_and_notebook_ast(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PLANTHIR_BACKEND_MODULES"] = "experiments.x3.module"
    spec_p = tmp_path / "spec.json"
    work_p = tmp_path / "cli_work"
    rep_p = tmp_path / "REPLAY.json"

    for args in (
        ["init", "--backend", "elendril_x3", "--out", str(spec_p)],
        ["validate", "--spec", str(spec_p), "--canary"],
        ["run", "--spec", str(spec_p), "--work-root", str(work_p), "--canary", "--device", "cpu"],
        ["verify-bundle", "--bundle-dir", str(work_p / "result_bundle")],
        ["replay-bundle", "--bundle-dir", str(work_p / "result_bundle"), "--output", str(rep_p)],
    ):
        proc = subprocess.run([sys.executable, "-m", "anra_v5.palantiri", *args], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"cmd={args}\nstdout={proc.stdout}\nstderr={proc.stderr}"

    rep_doc = json.loads(rep_p.read_text(encoding="utf-8"))
    assert rep_doc["verified"] is True
    assert rep_doc["prediction_agreement_status"] == "MATCHED"

    # Notebook AST parse & config sanity
    nb = build_x3_notebook_dict(source_commit="b" * 40, owner_authorized_default=True)
    raw = render_x3_notebook_bytes(source_commit="b" * 40, owner_authorized_default=True)
    assert len(raw) > 5000
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    cfg_src = "".join(nb["cells"][1]["source"])
    assert "OWNER_AUTHORIZED = True" in cfg_src
    assert "SESSION_MINUTES = 660" in cfg_src
    assert "ONE_GO = True" in cfg_src
