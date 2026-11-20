"""Comprehensive verification for 21 Nov 2026 X2 handoff repairs (Priorities 1-5) and gated X3 (§4)."""

from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest
import torch

from anra_v5.elendril.v8x.contracts import X1ModelSpec
from anra_v5.elendril.v8x.models import build_x1_s20_model
from anra_v5.elendril.v8x.training import create_x1_s20_trainer
from anra_v5.elendril.v8x.data import build_x1_d1h_training_dataset
from anra_v5.palantiri import GenericExperimentTrainer, GenericModelSpec, GenericTrainRecipe
from experiments.elendril_x2 import REGISTERED_X2_DATASETS, x2_dataset_factory, x2_loss_fn, x2_model_factory
from experiments.x2.campaign import (
    ElendrilX2Campaign,
    build_verified_resume_set,
    compute_protected_parent_ids,
    discover_kaggle_input_receipts,
    materialize_x2_job_spec,
    prune_completed_job_heavy_artifacts,
)
from experiments.x2.lineage_runner import (
    load_parent_checkpoint,
    run_frozen_probe,
    run_horizon_continuation,
    run_scratch_single,
    run_task_adaptation,
)
from experiments.x2.registry import (
    X1_LEGACY_COMPAT_SHA,
    X1_LEGACY_ORIGINAL_SHA,
    build_x2_design_registry,
    build_x2_obligation_ledger,
    evaluate_confirmation_trigger,
    evaluate_x2_completion_audit,
    evaluate_x2_qualification,
)
from experiments.x2.tasks import (
    T13VersionSpaceSolver,
    X2EnvironmentKernel,
    build_generator_b_multihop_panel,
    build_t13_decisive_fixture,
    build_t14_price_crossover_fixture,
    encode_x2_text,
    select_balanced_matched_cohort,
    solve_t14_public_dp,
    verify_x2_dataset_dispatch_preflight,
)
from experiments.x3 import (
    build_x3_paired_graph_factorial,
    build_x3_x2_crosswalk,
    eval_x3_shortcut_baselines,
    evaluate_x3_checkpoint_suite,
    independent_walk_x3_graph,
    run_gated_x3_campaign,
    run_x3_pointer_interchange_intervention,
    score_x3_campaign,
)
from tools.build_elendril_x2_notebook import DEFAULT_X2_SESSIONS, FULL_X2_SESSIONS, build_notebook_dict
from tools.elendril_v8x_worker import _resolve_x1_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUN1_ZIP = Path(r"C:\Users\ankit\Downloads\ELENDRIL_V8_X2_ANALYSIS.zip")


def _head_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()


def _make_x2_canary_ckpt(work_dir: Path, *, arch: str = "P", seed: int = 7, updates: int = 2) -> Path:
    commit = _head_commit()
    ds = x2_dataset_factory({"dataset_name": "x2_canary_fixture", "seed": seed}, cpu_canary=True)
    mcfg = {"width": 16, "layers": 1, "context_length": 640, "mem_dim": 16}
    m = x2_model_factory(arch, mcfg, seed=seed)
    mspec = GenericModelSpec(
        architecture=arch,
        model_config=mcfg,
        parameter_count=sum(int(p.numel()) for p in m.parameters()),
        backend_id="elendril_x2",
    )
    trainer = GenericExperimentTrainer(
        backend_id="elendril_x2",
        model=m,
        model_spec=mspec,
        recipe=GenericTrainRecipe(seed=seed, total_updates=updates, batch_size=4, micro_batch_size=2, lr=1e-3, objective="standard"),
        loss_fn=x2_loss_fn,
        items=ds["train_data"].items,
        data_provenance=ds["train_data"].manifest,
        order_seed=seed,
        source_commit=commit,
        root=work_dir / "checkpoints",
        lineage=f"ckpt-{arch.lower()}-s{seed}",
        device="cpu",
        cpu_canary=True,
        resume=False,
    )
    for _ in range(updates):
        trainer.step()
    trainer.save()
    return work_dir


def _make_x1_canary_ckpt(work_dir: Path, *, arch: str = "R", seed: int = 7) -> tuple[Path, X1ModelSpec]:
    commit = _head_commit()
    x1_spec = _resolve_x1_spec(arch, cpu_fixtures=True)
    d1h = build_x1_d1h_training_dataset(quota_per_family=1)
    job = {
        "job_id": f"X1-CANARY-{arch}-s{seed}",
        "architecture": arch,
        "session": "A0",
        "dataset": "x1-d1h-v1",
        "objective": "standard",
        "init_seed": seed,
        "order_seed": seed,
        "updates": 2,
    }
    tr = create_x1_s20_trainer(
        d1h,
        job,
        source_commit=commit,
        root=work_dir / "checkpoints",
        device="cpu",
        cpu_canary=True,
        x1_spec_override=x1_spec,
        updates_override=2,
        batch_size_override=4,
    )
    tr.step()
    tr.step()
    tr.save()
    return work_dir, x1_spec


def test_real_run1_zip_migration_and_tamper_rejection(tmp_path: Path) -> None:
    """Verify Run 1 archive imports all 12 A1/A2 jobs cleanly, starts at B0 seed 7, and rejects tampers."""
    commit = _head_commit()
    camp = ElendrilX2Campaign(source_commit=commit, work_dir=tmp_path / "run1_mig")
    reg = camp.registry
    assert reg["legacy_x1_source_sha"] == X1_LEGACY_COMPAT_SHA
    assert reg["legacy_x1_original_sha"] == X1_LEGACY_ORIGINAL_SHA

    if LOCAL_RUN1_ZIP.is_file():
        k_in = tmp_path / "k_in_real"
        k_in.mkdir()
        real_copy = k_in / LOCAL_RUN1_ZIP.name
        shutil.copy2(LOCAL_RUN1_ZIP, real_copy)
        discovered = discover_kaggle_input_receipts(k_in, tmp_path / "ext_real", reg)
        assert len(discovered) == 12
        res_rec = build_verified_resume_set(registry=reg, work_dir=tmp_path / "res_real", imported_receipts=discovered)
        assert res_rec["verified_count"] == 12
        assert res_rec["rejected"] == []
        assert len(res_rec["compatibility_mapped_imports"]) == 12
        assert all(
            c["original_source_commit"] == X1_LEGACY_ORIGINAL_SHA and c["repaired_source_commit"] == X1_LEGACY_COMPAT_SHA
            for c in res_rec["compatibility_mapped_imports"]
        )
        plan = camp.admit_slice(
            list(FULL_X2_SESSIONS),
            measured_seconds_per_update=0.65,
            measured_per_job_disk_bytes=32 * 1024 * 1024,
            parallel_workers=2,
            available_disk_bytes=15 * 1024**3,
            completed_ids=res_rec["verified_completed_ids"],
            include_extra_ledgers=True,
        )
        assert plan["admitted_job_ids"][0] == "X1-A3-S20-B0-standard-s7"

        # Tampered archive with .sha256 sidecar mismatch -> rejected
        k_bad = tmp_path / "k_in_bad_sha"
        k_bad.mkdir()
        tampered_zip = k_bad / LOCAL_RUN1_ZIP.name
        shutil.copy2(LOCAL_RUN1_ZIP, tampered_zip)
        tampered_zip.with_name(tampered_zip.name + ".sha256").write_text("0" * 64 + "  ELENDRIL_V8_X2_ANALYSIS.zip\n", encoding="utf-8")
        disc_bad = discover_kaggle_input_receipts(k_bad, tmp_path / "ext_bad_sha", reg)
        res_bad = build_verified_resume_set(registry=reg, work_dir=tmp_path / "res_bad_sha", imported_receipts=disc_bad)
        assert res_bad["verified_count"] == 0
        assert all(r["reason"] == "ARCHIVE_HASH_MISMATCH" for r in res_bad["rejected"])

    # Modified job.json scientific field without ZIP sidecar -> caught by legacy_scientific_identity_sha256
    a1_row = next(r for r in reg["jobs"] if r["job_id"] == "X1-A1-S20-F-standard-s7")
    job_dir = tmp_path / "fake_unpacked" / "jobs" / a1_row["job_id"]
    (job_dir / "milestones" / "u2368").mkdir(parents=True, exist_ok=True)
    (job_dir / "milestones" / "u2368" / "episodes.jsonl").write_text('{"exact": true}\n', encoding="utf-8")
    tampered_job = {
        "job_id": a1_row["job_id"],
        "architecture": a1_row["architecture"],
        "dataset": a1_row["dataset"],
        "objective": a1_row["objective"],
        "init_seed": 99,  # TAMPERED seed
        "order_seed": a1_row["order_seed"],
        "target_updates": a1_row["updates"],
        "completed_updates": a1_row["updates"],
        "milestones_recorded": ["u0000", "u0592", "u1184", "u1472", "u2368"],
        "model_spec_sha256": a1_row["extra"]["legacy_model_spec_sha256"],
        "source_commit": X1_LEGACY_ORIGINAL_SHA,
        "status": "COMPLETE",
    }
    (job_dir / "job.json").write_text(json.dumps(tampered_job) + "\n", encoding="utf-8")
    disc_t = discover_kaggle_input_receipts(tmp_path / "fake_unpacked", tmp_path / "ext_t", reg)
    res_t = build_verified_resume_set(registry=reg, work_dir=tmp_path / "res_t", imported_receipts=disc_t)
    assert res_t["verified_count"] == 0
    assert res_t["rejected"][0]["reason"] == "LEGACY_SCIENTIFIC_IDENTITY_MISMATCH"


def test_priority2_dataset_dispatch_coverage_and_causal_t13_t14_checks() -> None:
    """Verify distinct dispatch across all 13 prod datasets, fail-closed errors, and T13/T14 causal sensitivity."""
    prod_ds = [
        k for k in sorted(REGISTERED_X2_DATASETS)
        if k not in ("grad_check", "resume_parity", "trunc_ckpt", "x2_ds", "x2_canary_fixture")
    ]
    pre = verify_x2_dataset_dispatch_preflight(x2_dataset_factory, prod_ds, seed=7)
    assert pre["passed"] is True
    assert pre["datasets_checked"] == len(prod_ds) == 13

    with pytest.raises(ValueError, match="UNKNOWN_X2_DATASET"):
        x2_dataset_factory({"dataset_name": "nonexistent_ds", "seed": 7}, cpu_canary=False)
    with pytest.raises(ValueError, match="UNKNOWN_X2_SPLIT"):
        x2_dataset_factory({"dataset_name": "x2-t13-fst-train32-v1", "split": "bad_split", "seed": 7}, cpu_canary=False)

    # Reject zero T13 rows or equal cohorts under different dataset labels
    def _bad_equal_factory(cfg, cpu_canary=False):
        return x2_dataset_factory({"dataset_name": "x2-t14-costly-6strata-v1", "seed": cfg["seed"], "split": "dev"}, cpu_canary=False)

    with pytest.raises(ValueError, match="INCOMPATIBLE_DATASETS_EQUAL_COHORT"):
        verify_x2_dataset_dispatch_preflight(_bad_equal_factory, ["ds_a", "ds_b"], seed=7)

    # T13 witness label-flip changes posterior derived answer
    fix = build_t13_decisive_fixture()
    w_id = fix["worlds"]["id_disc"]
    solver = T13VersionSpaceSolver(max_laws=512)
    p_orig = solver.posterior(w_id["support"], w_id["query"])
    flipped_sup = [dict(x) for x in w_id["support"]]
    flipped_sup[-1] = {**flipped_sup[-1], "output": (int(flipped_sup[-1]["output"]) + 1) % 4}
    p_flip = solver.posterior(flipped_sup, w_id["query"])
    assert (
        p_orig["bayes_optimal_label"] != p_flip["bayes_optimal_label"]
        or p_orig["output_probs"] != p_flip["output_probs"]
    )

    # T14 tool result changes next optimal DP action
    t14_fix = build_t14_price_crossover_fixture()
    w_cheap = t14_fix["cheap_cert"]
    kernel = X2EnvironmentKernel()
    st0 = kernel.initial_state(w_cheap)
    dp0 = solve_t14_public_dp(w_cheap)
    assert dp0["optimal_operation"] in ("QUERY", "RETRIEVE")
    st1 = kernel.step(st0, dp0["optimal_operation"], None)
    assert st1.certified is True and st1.certified_answer is not None
    assert kernel.ideal_action(st1)[0] == "ANSWER"
    st_nc = kernel.step(kernel.initial_state(t14_fix["cheap_nocert"]), dp0["optimal_operation"], None)
    assert kernel.ideal_action(st_nc)[0] == "ABSTAIN"


def test_priority3_obligation_ledger_global_audit_and_no_auto_dm_winner() -> None:
    """Verify 192-row ledger, scope-specific PARTIAL on single missing row, and no auto-selected DM."""
    reg = build_x2_design_registry(source_commit="a" * 40)
    ledger = build_x2_obligation_ledger(reg)
    assert ledger["total_logical_rows"] == 192
    assert ledger["mandatory_rows"] == 174
    assert ledger["conditional_confirmation_rows"] == 18

    # Evaluate confirmation trigger when all candidates fail -> NOT_TRIGGERED_SIGNED_OFF, NEVER auto-DM
    trig_fail = evaluate_confirmation_trigger(
        {
            "DM": {"control": "DM-add", "old_macro_retention_delta": 0.0, "paired_seed_scores": [{"treatment_score": 0.50, "control_score": 0.55}, {"treatment_score": 0.51, "control_score": 0.52}, {"treatment_score": 0.52, "control_score": 0.52}]},
            "R2": {"control": "R2-pool", "old_macro_retention_delta": 0.0, "paired_seed_scores": [{"treatment_score": 0.50, "control_score": 0.50}, {"treatment_score": 0.51, "control_score": 0.50}, {"treatment_score": 0.52, "control_score": 0.51}]},
        }
    )
    assert trig_fail["triggered"] is False
    assert trig_fail["decision"] == "NOT_TRIGGERED_SIGNED_OFF"
    assert trig_fail["selected_candidate"] is None

    # Complete all mandatory rows EXCEPT drop 1 frozen row -> PARTIAL even if session_queue_empty=True
    full_receipts = {
        e["logical_id"]: {"status": "COMPLETE", "eval_rows": 16}
        for e in ledger["entries"]
        if e["mandatory"] and e["execution_route"] != "evidenced_historical"
    }
    full_recomp = {lid: {"verified": True} for lid in full_receipts}
    a_ok = evaluate_x2_completion_audit(
        ledger,
        full_receipts,
        confirmation_decision=trig_fail,
        case_recomputation_receipts=full_recomp,
        session_queue_empty=True,
    )
    assert a_ok["global_x2_status"] == "X2_COMPLETE" and a_ok["x3_unlocked"] is True

    for drop_id in ("X2-BII-FROZEN-A0-P-s7", "X2-IIIb-DM-S20-DM-add-s7"):
        partial_r = dict(full_receipts)
        partial_r.pop(drop_id)
        a_bad = evaluate_x2_completion_audit(
            ledger,
            partial_r,
            confirmation_decision=trig_fail,
            case_recomputation_receipts=full_recomp,
            session_queue_empty=True,
        )
        assert a_bad["global_x2_status"] == "PARTIAL"
        assert a_bad["x2_complete"] is False
        assert a_bad["x3_unlocked"] is False
        assert drop_id in a_bad["unresolved_mandatory_ids"]

    # Triggered confirmation with 1 missing confirmation row -> PARTIAL
    trig_pass = evaluate_confirmation_trigger(
        {
            "R2": {
                "control": "R2-pool",
                "old_macro_retention_delta": 0.0,
                "paired_seed_scores": [
                    {"treatment_score": 0.80, "control_score": 0.60},
                    {"treatment_score": 0.82, "control_score": 0.61},
                    {"treatment_score": 0.81, "control_score": 0.60},
                ],
            }
        }
    )
    assert trig_pass["triggered"] is True and trig_pass["selected_candidate"] == "R2"
    a_conf_missing = evaluate_x2_completion_audit(
        ledger,
        full_receipts,
        confirmation_decision=trig_pass,
        case_recomputation_receipts=full_recomp,
        session_queue_empty=True,
    )
    assert a_conf_missing["global_x2_status"] == "PARTIAL"
    assert len(a_conf_missing["unresolved_conditional_ids"]) == 18


def test_priority4_lineage_routes_spec_refusals_and_protected_pruning(tmp_path: Path) -> None:
    """Verify spec refusals, all explicit lineage_runner routes, parity checks, and protected parent pruning."""
    commit = _head_commit()
    reg = build_x2_design_registry(source_commit=commit)

    frozen_row = next(r for r in reg["jobs"] if r["job_id"] == "X2-BII-FROZEN-A0-P-s7")
    t13_row = next(r for r in reg["jobs"] if r["job_id"] == "X2-BIV-T13-S20-DM-s7")
    unpaired_ctrl = next(r for r in reg["jobs"] if r["job_id"] == "X2-IIIb-DM-S20-DM-add-s7")

    with pytest.raises(ValueError, match="FROZEN_PROBE_MUST_USE_FROZEN_ROUTE"):
        materialize_x2_job_spec(frozen_row, tmp_path / "f.json", source_commit=commit, canary=True)
    with pytest.raises(ValueError, match="NON_SCRATCH_ROUTE_MUST_USE_LINEAGE_RUNNER"):
        materialize_x2_job_spec(t13_row, tmp_path / "t.json", source_commit=commit, canary=True)
    with pytest.raises(ValueError, match="PAIRED_CARTRIDGE_REQUIRES_REGISTERED_CONTROL_ARM"):
        materialize_x2_job_spec(unpaired_ctrl, tmp_path / "u.json", source_commit=commit, canary=True)

    with pytest.raises(FileNotFoundError, match="MISSING_PARENT_CHECKPOINT"):
        load_parent_checkpoint(tmp_path / "does_not_exist", architecture="P")

    # Build bounded X1 R and X2 P canary checkpoints
    x1_r_dir, _ = _make_x1_canary_ckpt(tmp_path / "p_x1_r", arch="R", seed=7)
    x2_p_dir = _make_x2_canary_ckpt(tmp_path / "p_x2_p", arch="P", seed=7, updates=2)

    # 1. Frozen probe: 0 optimizer steps + missing_original_receipt reproduced mode
    miss_path = tmp_path / "missing_a0.json"
    miss_path.write_text('{"reason": "A0_BIN_PRUNED", "job_id": "X1-A0-S20-R-standard-s7"}\n', encoding="utf-8")
    f_res = run_frozen_probe(
        job_id="X2-BII-FROZEN-A0-R-s7",
        architecture="R",
        work_root=tmp_path / "out_frozen",
        source_commit=commit,
        seed=7,
        parent_job_id="X1-A0-S20-R-standard-s7",
        missing_original_receipt=miss_path,
        reproduced_checkpoint=x1_r_dir,
        device="cpu",
        canary=True,
    )
    assert f_res["optimizer_steps_executed"] == 0
    assert f_res["weights_immutable_verified"] is True
    assert f_res["evidence_source"] == "reproduced_with_missing_original_receipt"
    assert f_res["reproduced_a0_parity"]["reference_available"] is True

    # 2. Task adaptation on X1 R parent (T13) + X2 P parent (T14) -> pre-update parity <= 1e-6 & AdamW reset
    ad_t13 = run_task_adaptation(
        job_id="X2-BIV-T13-S20-R-s7",
        architecture="R",
        parent_job_id="X1-A0-S20-R-standard-s7",
        parent_checkpoint=x1_r_dir,
        dataset_name="x2-t13-fst-train32-v1",
        objective="T13_ADAPT",
        work_root=tmp_path / "out_t13",
        source_commit=commit,
        seed=7,
        updates=2,
        milestones=(1, 2),
        device="cpu",
        canary=True,
    )
    assert ad_t13["parent_receipt"]["optimizer_reset"] is True
    assert ad_t13["parent_receipt"]["pre_update_parity"]["max_abs_logit_diff"] <= 1e-6

    ad_t14 = run_task_adaptation(
        job_id="X2-BIV-T14-S20-P-s7",
        architecture="P",
        parent_job_id="X1-A0-S20-P-standard-s7",
        parent_checkpoint=x2_p_dir,
        dataset_name="x2-t14-costly-6strata-v1",
        objective="T14_BELIEF",
        work_root=tmp_path / "out_t14",
        source_commit=commit,
        seed=7,
        updates=2,
        milestones=(1, 2),
        device="cpu",
        canary=True,
    )
    assert "cheap_regime" in ad_t14["summary"]["task_evaluation"]
    assert "expensive_regime" in ad_t14["summary"]["task_evaluation"]

    # 3. Horizon continuation: steps 2 -> 3 -> 4 under registered 0.3x -> 0.1x second cosine schedule
    h_res = run_horizon_continuation(
        job_id="X2-BV-H4736-S20-P-s7",
        architecture="P",
        parent_job_id="X1-A0-S20-P-standard-s7",
        parent_checkpoint=x2_p_dir,
        work_root=tmp_path / "out_h",
        source_commit=commit,
        seed=7,
        parent_updates=2,
        extension_updates=2,
        mid_milestone=3,
        end_milestone=4,
        device="cpu",
        canary=True,
    )
    assert h_res["global_steps_executed"] == [3, 4]
    assert h_res["lr_start"] == pytest.approx(0.3 * 1e-3, rel=1e-5)
    assert h_res["lr_end"] == pytest.approx(0.1 * 1e-3, rel=1e-5)
    reloaded = load_parent_checkpoint(tmp_path / "out_h", architecture="P", tiny_canary=True)
    assert reloaded["parent_global_update"] == 4

    # 4. Scratch single (no fake extra arm) + protected parent pruning & export
    s_res = run_scratch_single(
        job_id="X2-EXTRA-T13-PARENT-HIST_A-S20-P-s7",
        architecture="P",
        dataset_name="x2-t13-history-hist_a-v1",
        objective="T13_ADAPT",
        work_root=tmp_path / "out_single",
        source_commit=commit,
        seed=7,
        updates=2,
        milestones=(1, 2),
        device="cpu",
        canary=True,
    )
    assert s_res["summary"]["arm_count"] == 1
    assert s_res["summary"]["synthesized_extra_arms"] is False

    # Protected parent pruning refuses to delete X2-IIIb-DM-S20-DM-s7 checkpoints while BIV/BV incomplete
    camp = ElendrilX2Campaign(source_commit=commit, work_dir=tmp_path / "prune_camp")
    camp.prepare()
    prot = compute_protected_parent_ids(camp.registry, completed_ids=())
    assert "X2-IIIb-DM-S20-DM-s7" in prot["protected_job_ids"]
    dm_job = camp.work_dir / "jobs" / "X2-IIIb-DM-S20-DM-s7"
    _make_x2_canary_ckpt(dm_job, arch="DM", seed=7, updates=2)
    (dm_job / "FINAL_RECEIPT.json").write_text('{"status": "COMPLETE"}\n', encoding="utf-8")
    pr = prune_completed_job_heavy_artifacts(dm_job, protected_job_ids=prot["protected_job_ids"])
    assert pr["pruned"] is False and pr["protected"] is True
    exp_p = camp.export_parent_checkpoints(tmp_path / "out_parents", protected_job_ids=prot["protected_job_ids"])
    assert exp_p["verified_readable"] is True and "X2-IIIb-DM-S20-DM-s7" in exp_p["exported_parent_ids"]


def test_priority5_overflow_balanced_cohort_bootstrap_and_explicit_unknown_receipts() -> None:
    """Verify overflow fails closed, balanced cohort stratification, and explicit UNKNOWN_MISSING_RECEIPT."""
    with pytest.raises(ValueError, match="X2_CONTEXT_OVERFLOW_DROPPED_EVIDENCE"):
        encode_x2_text("A" * 260, max_length=256, allow_truncate=False)

    gen_b = build_generator_b_multihop_panel(split="dev", num_skeletons=4, seed=77, depths=(3, 4, 5), include_exhaustive_perm=False)
    cohort = select_balanced_matched_cohort(gen_b["episodes"], max_total=12)
    assert {int(w["hops"]) for w in cohort} == {3, 4, 5}
    assert {str(w["fact_style"]) for w in cohort} == {"canonical", "narrative"}

    qual_unk = evaluate_x2_qualification(
        campaign_id="X2-Q-UNK",
        candidate_name="R2",
        control_name="R2-pool",
        paired_seed_scores=[
            {"seed_id": "s7", "treatment_score": 0.72, "control_score": 0.60},
            {"seed_id": "s19", "treatment_score": 0.74, "control_score": 0.61},
            {"seed_id": "s41", "treatment_score": 0.71, "control_score": 0.60},
        ],
        paired_world_or_family_scores={f"fam_{i}": (0.60, 0.72) for i in range(6)},
    )
    vm_unk = {row["check"]: row["status"] for row in qual_unk["verification_matrix"]}
    assert vm_unk["recorded_action_replay"] == "UNKNOWN_MISSING_RECEIPT"
    assert vm_unk["checkpoint_restore"] == "UNKNOWN_MISSING_RECEIPT"
    assert vm_unk["regenerated_per_case_agreement"] == "UNKNOWN_MISSING_RECEIPT"
    assert qual_unk["family_bootstrap"]["n_units"] == 6

    qual_pass = evaluate_x2_qualification(
        campaign_id="X2-Q-PASS",
        candidate_name="R2",
        control_name="R2-pool",
        paired_seed_scores=[
            {"seed_id": "s7", "treatment_score": 0.72, "control_score": 0.60},
            {"seed_id": "s19", "treatment_score": 0.74, "control_score": 0.61},
            {"seed_id": "s41", "treatment_score": 0.71, "control_score": 0.60},
        ],
        paired_world_or_family_scores={f"fam_{i}": (0.60, 0.72) for i in range(6)},
        verification_receipts={
            "byte_integrity_verified": True,
            "oracle_fixture_verified": True,
            "recorded_action_replay_verified": True,
            "checkpoint_restore_verified": True,
            "regenerated_per_case_verified": True,
            "neural_replay_verified": True,
        },
    )
    vm_pass = {row["check"]: row["status"] for row in qual_pass["verification_matrix"]}
    assert vm_pass["recorded_action_replay"] == "PASS"
    assert vm_pass["checkpoint_restore"] == "PASS"
    assert vm_pass["regenerated_per_case_agreement"] == "PASS"


def test_x3_factorial_interventions_scoring_and_strict_x2_gate(tmp_path: Path) -> None:
    """Verify full X3 dev/conf factorial sizes, walker certification, interventions, collapse detector, and X2 gate."""
    # Full factorial scale counts (1,024 dev / 2,048 conf decisions) + walker certification
    dev_panel = build_x3_paired_graph_factorial(split="dev", depths=(4, 5), base_graphs_per_depth=64)
    conf_panel = build_x3_paired_graph_factorial(split="confirmation", depths=(4, 5), base_graphs_per_depth=128)
    assert dev_panel["decision_count"] == 1024
    assert conf_panel["decision_count"] == 2048

    b_dev = eval_x3_shortcut_baselines(dev_panel)
    assert b_dev["oracle_independent_walker_accuracy"] == 1.0
    assert b_dev["last_row_accuracy"] < 0.25

    cw = build_x3_x2_crosswalk(conf_panel, dev_panel["decisions"])
    assert cw["strictly_novel_for_confirmation"] is True

    # Spot-check byte alignment & independent walker
    sample_pair = dev_panel["aligned_pairs"][0]
    assert sample_pair["byte_len_original"] == sample_pair["byte_len_edited"] == sample_pair["byte_len_irrelevant"]
    w0 = independent_walk_x3_graph(dev_panel["base_graphs"][0]["facts_original"], dev_panel["base_graphs"][0]["affected_query"], hops=4)
    assert w0["reachable"] is True and w0["terminal_value"] == dev_panel["base_graphs"][0]["expected_affected_before"]

    # Gate blocks X3 with 0.0 GPU seconds whenever X2 is incomplete
    gated_res = run_gated_x3_campaign(
        x2_completion_audit={"x2_complete": False, "global_x2_status": "PARTIAL", "unresolved_mandatory_ids": ["X1-A3-S20-B0-standard-s7"]},
        output_dir=tmp_path / "x3_blocked",
        source_commit=_head_commit(),
    )
    assert gated_res["status"] == "GATED_PENDING_X2_COMPLETE"
    assert gated_res["x3_executed"] is False
    assert gated_res["gpu_seconds_spent"] == 0.0
    assert not (tmp_path / "x3_blocked" / "ELENDRIL_V8_X3_ANALYSIS.zip").exists()

    # Unlocked X3 runs cleanly with tiny checkpoints into separate ELENDRIL_V8_X3_ANALYSIS.zip
    p_dir, _ = _make_x1_canary_ckpt(tmp_path / "ckpt_p7", arch="P", seed=7)
    f_dir, _ = _make_x1_canary_ckpt(tmp_path / "ckpt_f7", arch="F-cap", seed=7)
    r_dir, _ = _make_x1_canary_ckpt(tmp_path / "ckpt_r7", arch="R", seed=7)
    rn_dir, _ = _make_x1_canary_ckpt(tmp_path / "ckpt_rn7", arch="R-noedge", seed=7)

    unlocked = run_gated_x3_campaign(
        x2_completion_audit={"x2_complete": True, "global_x2_status": "X2_COMPLETE"},
        output_dir=tmp_path / "x3_unlocked",
        source_commit=_head_commit(),
        checkpoint_map={"P": {7: p_dir}, "F-cap": {7: f_dir}, "R": {7: r_dir}, "R-noedge": {7: rn_dir}},
        device="cpu",
        run_confirmation=True,
        tiny_canary=True,
        dev_base_graphs_per_depth=2,
        conf_base_graphs_per_depth=2,
    )
    assert unlocked["status"] == "COMPLETE"
    assert (tmp_path / "x3_unlocked" / "ELENDRIL_V8_X3_ANALYSIS.zip").is_file()
    assert unlocked["interchange_interventions"]["F-cap"]["7"]["raw"]["supports_pointer_interchange"] is False
    assert unlocked["interchange_interventions"]["P"]["7"]["raw"]["affected_same_world_donor_max_prob_diff"] <= 1e-6

    # Confidence collapse detector catches a model biased heavily toward ABSTAIN on interchange
    x1_p = build_x1_s20_model(_resolve_x1_spec("P", cpu_fixtures=True), 7)
    with torch.no_grad():
        x1_p.op_head.bias[3] = 20.0  # Force ABSTAIN collapse
    ic_col = run_x3_pointer_interchange_intervention(x1_p, x1_p.x1_spec, dev_panel["aligned_pairs"][:8], device="cpu")
    assert ic_col["global_confidence_collapse"] is True
    assert ic_col["status"] == "INCONCLUSIVE_CONFIDENCE_COLLAPSE"

    # Notebook sanity check for FULL_X2_SESSIONS & X1_LEGACY_COMPAT_SHA pin
    nb = build_notebook_dict(source_commit=_head_commit())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    cfg_src = "".join(nb["cells"][1]["source"])
    assert f'LEGACY_X1_SHA = "{X1_LEGACY_COMPAT_SHA}"' in cfg_src
    assert all(s in cfg_src for s in FULL_X2_SESSIONS)
    assert len(DEFAULT_X2_SESSIONS) < len(FULL_X2_SESSIONS)
