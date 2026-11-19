"""Campaign orchestration, dual-source routing, 6 failure-injection recovery tests, and notebook checks for X2."""

from __future__ import annotations

import ast
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pytest
import torch

from anra_v5.palantiri import (
    GenericExperimentTrainer,
    GenericModelSpec,
    GenericTrainRecipe,
    build_paired_seed_observations,
)
from experiments.elendril_x2 import x2_dataset_factory, x2_loss_fn, x2_model_factory
from experiments.x2.campaign import (
    ElendrilX2Campaign,
    build_legacy_x1_worker_command,
    build_verified_resume_set,
    build_x2_palantiri_command,
    compute_effective_budget_minutes,
    discover_kaggle_input_receipts,
    materialize_x2_job_spec,
    prune_completed_job_heavy_artifacts,
)
from experiments.x2.registry import X1_LEGACY_SOURCE_SHA, build_x2_design_registry, evaluate_x2_qualification
from tools.build_elendril_x2_notebook import DEFAULT_X2_SESSIONS, build_notebook_dict, render_notebook_bytes
from v5_training.checkpoint import CheckpointStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def _head_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()


def test_failure_injection_1_lost_worker_after_update_and_resume_parity(tmp_path: Path) -> None:
    """§0.4.5 #1: stop after update 1 and resume -> identical sample order, losses, and final weights vs uninterrupted."""
    commit = _head_commit()
    ds = x2_dataset_factory({"dataset_name": "resume_parity", "seed": 7}, cpu_canary=True)
    items = ds["train_data"].items
    prov = ds["train_data"].manifest
    mcfg = {"width": 16, "layers": 1, "context_length": 256, "mem_dim": 16}
    recipe = GenericTrainRecipe(seed=7, total_updates=2, batch_size=4, micro_batch_size=2, lr=1e-3, objective="standard")

    def _make_trainer(root: Path, *, resume: bool) -> GenericExperimentTrainer:
        m = x2_model_factory("DM", mcfg, seed=7)
        mspec = GenericModelSpec(
            architecture="DM",
            model_config=mcfg,
            parameter_count=sum(int(p.numel()) for p in m.parameters()),
            backend_id="elendril_x2",
        )
        return GenericExperimentTrainer(
            backend_id="elendril_x2",
            model=m,
            model_spec=mspec,
            recipe=recipe,
            loss_fn=x2_loss_fn,
            items=items,
            data_provenance=prov,
            order_seed=7,
            source_commit=commit,
            root=root,
            lineage="dm-s7",
            device="cpu",
            cpu_canary=True,
            resume=resume,
        )

    # Uninterrupted reference: 2 steps
    t_full = _make_trainer(tmp_path / "full", resume=False)
    r1_full = t_full.step()
    r2_full = t_full.step()
    w_full = {k: v.detach().clone() for k, v in t_full.model.state_dict().items()}

    # Interrupted worker: step 1 -> save -> crash/discard -> resume -> step 2
    t_part = _make_trainer(tmp_path / "resumed", resume=False)
    r1_part = t_part.step()
    t_part.save()
    del t_part

    t_res = _make_trainer(tmp_path / "resumed", resume=True)
    r2_res = t_res.step()
    w_res = {k: v.detach().clone() for k, v in t_res.model.state_dict().items()}

    assert abs(float(r1_full["loss"]) - float(r1_part["loss"])) < 1e-6
    assert abs(float(r2_full["loss"]) - float(r2_res["loss"])) < 1e-6
    for k in w_full:
        assert torch.allclose(w_full[k], w_res[k], atol=1e-6), f"resumed weight diverged at {k}"


def test_failure_injection_2_truncated_checkpoint_rejected(tmp_path: Path) -> None:
    """§0.4.5 #2: truncated checkpoint payload fails closed on restore."""
    commit = _head_commit()
    ds = x2_dataset_factory({"dataset_name": "trunc_ckpt", "seed": 7}, cpu_canary=True)
    mcfg = {"width": 16, "layers": 1, "context_length": 256, "mem_dim": 16}
    m = x2_model_factory("DM", mcfg, seed=7)
    mspec = GenericModelSpec(
        architecture="DM",
        model_config=mcfg,
        parameter_count=sum(int(p.numel()) for p in m.parameters()),
        backend_id="elendril_x2",
    )
    t = GenericExperimentTrainer(
        backend_id="elendril_x2",
        model=m,
        model_spec=mspec,
        recipe=GenericTrainRecipe(seed=7, total_updates=2, batch_size=4, micro_batch_size=2, lr=1e-3, objective="standard"),
        loss_fn=x2_loss_fn,
        items=ds["train_data"].items,
        data_provenance=ds["train_data"].manifest,
        order_seed=7,
        source_commit=commit,
        root=tmp_path / "ckpts",
        lineage="dm-trunc",
        device="cpu",
        cpu_canary=True,
    )
    t.step()
    sha = t.save()

    # Deliberately truncate model.bin on disk
    model_bin = tmp_path / "ckpts" / "dm-trunc" / "objects" / sha / "model.bin"
    assert model_bin.is_file()
    model_bin.write_bytes(model_bin.read_bytes()[:32])

    store = CheckpointStore(tmp_path / "ckpts", "dm-trunc")
    with pytest.raises(Exception):
        store.restore(sha)


def test_failure_injections_3_and_4_swapped_spec_and_missing_archive_member(tmp_path: Path) -> None:
    """§0.4.5 #3 & #4: verified resume rejects swapped spec & missing archive member (never blindly trusts COMPLETED_JOBS)."""
    commit = _head_commit()
    reg = build_x2_design_registry(source_commit=commit)
    target_row = next(r for r in reg["jobs"] if r["block"] == "IIIb_NEW_MECHANISMS")

    # Good archive
    good_zip = tmp_path / "good.zip"
    with zipfile.ZipFile(good_zip, "w") as zf:
        zf.writestr("SCIENTIFIC_SUMMARY.json", "{}\n")
        zf.writestr("REPLAY_REFERENCE.json", "{}\n")
    good_sha = sha256(good_zip.read_bytes()).hexdigest()

    # Archive missing required REPLAY_REFERENCE.json
    bad_zip = tmp_path / "missing_member.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("SCIENTIFIC_SUMMARY.json", "{}\n")
    bad_sha = sha256(bad_zip.read_bytes()).hexdigest()

    # Pre-seed an unverified COMPLETED_JOBS.json claiming completion — must be overwritten by verified audit
    (tmp_path / "COMPLETED_JOBS.json").write_text(json.dumps({"completed_jobs": [target_row["job_id"]]}) + "\n", encoding="utf-8")

    # Inject swapped spec + missing member
    rec = build_verified_resume_set(
        registry=reg,
        work_dir=tmp_path,
        imported_receipts=[
            {
                "job_id": target_row["job_id"],
                "source_commit": target_row["source_commit"],
                "spec_sha256": "f" * 64,  # SWAPPED SPEC
                "completed_updates": target_row["updates"],
                "eval_rows": 128,
                "archive_path": str(good_zip),
                "archive_sha256": good_sha,
                "required_members": ["SCIENTIFIC_SUMMARY.json", "REPLAY_REFERENCE.json"],
            },
            {
                "job_id": reg["jobs"][10]["job_id"],
                "source_commit": reg["jobs"][10]["source_commit"],
                "spec_sha256": reg["jobs"][10]["spec_sha256"],
                "completed_updates": reg["jobs"][10]["updates"],
                "eval_rows": 128,
                "archive_path": str(bad_zip),
                "archive_sha256": bad_sha,
                "required_members": ["SCIENTIFIC_SUMMARY.json", "REPLAY_REFERENCE.json"],
            },
        ],
    )
    assert rec["verified_count"] == 0
    reasons = {r["reason"] for r in rec["rejected"]}
    assert "SWAPPED_SPEC_MISMATCH" in reasons
    assert "MISSING_ARCHIVE_MEMBER" in reasons
    # Stale unverified COMPLETED_JOBS.json was wiped cleanly
    assert json.loads((tmp_path / "COMPLETED_JOBS.json").read_text(encoding="utf-8"))["completed_jobs"] == []

    # Valid receipt passes cleanly
    rec_ok = build_verified_resume_set(
        registry=reg,
        work_dir=tmp_path,
        imported_receipts=[
            {
                "job_id": target_row["job_id"],
                "source_commit": target_row["source_commit"],
                "spec_sha256": target_row["spec_sha256"],
                "completed_updates": target_row["updates"],
                "eval_rows": 128,
                "archive_path": str(good_zip),
                "archive_sha256": good_sha,
                "required_members": ["SCIENTIFIC_SUMMARY.json", "REPLAY_REFERENCE.json"],
            }
        ],
    )
    assert rec_ok["verified_completed_ids"] == [target_row["job_id"]]


def test_failure_injection_5_missing_paired_id_disclosed_and_preserved() -> None:
    """§0.4.5 #5: missing paired ID is explicitly detected while preserving completed records."""
    arm_records = [
        {"architecture": "DM-add", "init_seed": 7, "order_seed": 7, "value": 0.60},
        {"architecture": "DM", "init_seed": 7, "order_seed": 7, "value": 0.68},
        {"architecture": "DM", "init_seed": 19, "order_seed": 19, "value": 0.69},  # missing DM-add seed 19!
    ]
    bundle = build_paired_seed_observations(
        arm_records,
        contrast_id="DM_vs_DM_add",
        control_arm="DM-add",
        treatment_arm="DM",
    )
    assert len(bundle["observations"]) == 1
    assert len(bundle["missing_pairs"]) == 1
    assert len(bundle["preserved_arm_records"]) == 3


def test_failure_injection_6_one_failed_lane_preserves_peer_and_exports_partial(tmp_path: Path) -> None:
    """§0.4.5 #6: failing lane 1 preserves peer lane 0 completion, records unfinished IDs, and exports verified PARTIAL ZIP."""
    commit = _head_commit()

    def _runner(spec: Any) -> int:
        spec.work_root.mkdir(parents=True, exist_ok=True)
        if spec.worker_id.endswith("-s19"):
            (spec.work_root / "FAILURE_RECEIPT.json").write_text('{"reason": "INJECTED_LANE_CRASH"}\n', encoding="utf-8")
            return 1
        (spec.work_root / "METRICS.json").write_text('{"accuracy": 0.75}\n', encoding="utf-8")
        return 0

    camp = ElendrilX2Campaign(
        source_commit=commit,
        work_dir=tmp_path / "work",
        outer_max_minutes=480.0,
        session_max_minutes=330.0,
        export_reserve_minutes=45.0,
        in_process_runner=_runner,
    )
    camp.prepare(["IIIb-DM"])
    rows = camp.select_session_rows(["IIIb-DM"])[:2]  # s7 (succeeds) & s19 (fails)
    summary = camp.run_admitted_waves(camp.to_session_jobs(rows), devices=("cuda:0", "cuda:1"), canary=True)

    assert len(summary["completed_jobs"]) == 1
    assert any(j.endswith("-s7") for j in summary["completed_jobs"])
    assert any(j.endswith("-s19") for j in summary["unfinished_job_ids"])
    assert any(j.endswith("-s19") for j in summary["failed_job_ids"])

    delivery = camp.export_archives(tmp_path / "out", prefix="ELENDRIL_V8_X2_TEST", status="PARTIAL")
    assert delivery["verified"] is True
    assert delivery["final_status"]["status"] == "PARTIAL"
    assert Path(delivery["archives"]["analysis"]).is_file()
    assert Path(delivery["archives"]["full"]).is_file()


def test_dual_source_routing_and_dual_clock_admission(tmp_path: Path) -> None:
    commit = _head_commit()
    bud = compute_effective_budget_minutes(outer_remaining_minutes=480.0, session_max_minutes=330.0, export_reserve_minutes=45.0)
    assert bud["governing_cap_minutes"] == 330.0
    assert bud["usable_train_minutes"] == 285.0

    bud2 = compute_effective_budget_minutes(outer_remaining_minutes=200.0, session_max_minutes=330.0, export_reserve_minutes=45.0)
    assert bud2["governing_cap_minutes"] == 200.0
    assert bud2["usable_train_minutes"] == 155.0

    camp = ElendrilX2Campaign(source_commit=commit, work_dir=tmp_path / "camp", legacy_x1_repo_dir=tmp_path / "legacy")
    camp.prepare(["A1", "IIIb-DM"])
    # Tight budget admits partial slice while preserving deferred pending IDs without deletion
    plan = camp.admit_slice(
        ["A1", "IIIb-DM"],
        measured_seconds_per_update=2.0,  # ~79 min per job / 2 lanes => ~7 jobs fit in 285m
        parallel_workers=2,
        available_disk_bytes=50 * 1024**3,
    )
    assert len(plan["admitted_job_ids"]) >= 1
    assert len(plan["deferred_pending_job_ids"]) >= 1
    assert plan["next_resumable_job_id"] == plan["deferred_pending_job_ids"][0]

    # Verify legacy X1 vs X2 command routing
    legacy_row = next(r for r in camp.registry["jobs"] if r["session"] == "A1")
    leg_cmd = build_legacy_x1_worker_command(legacy_row, legacy_repo_dir=tmp_path / "legacy", work_root=tmp_path / "w1")
    assert leg_cmd["lane_kind"] == "legacy_x1_worker"
    assert leg_cmd["source_commit"] == X1_LEGACY_SOURCE_SHA
    assert "tools.elendril_v8x_worker" in leg_cmd["command"]
    assert "run-job" in leg_cmd["command"]
    assert "--source" in leg_cmd["command"]
    assert "--session" not in leg_cmd["command"]

    x2_row = next(r for r in camp.registry["jobs"] if r["session"] == "IIIb-DM")
    spec_info = materialize_x2_job_spec(x2_row, tmp_path / "x2.json", source_commit=commit, canary=True)
    x2_cmd = build_x2_palantiri_command(x2_row, spec_path=spec_info["spec_path"], work_root=tmp_path / "w2", canary=True)
    assert x2_cmd["lane_kind"] == "elendril_x2_palantiri"
    assert x2_cmd["env_overrides"]["PLANTHIR_BACKEND_MODULES"] == "experiments.elendril_x2"
    assert "anra_v5.palantiri" in x2_cmd["command"]


def test_x2_kaggle_notebook_ast_zero_edit_default_and_disabled_dry_run() -> None:
    commit = _head_commit()
    nb_on = build_notebook_dict(source_commit=commit, owner_authorized_default=True)
    nb_off = build_notebook_dict(source_commit=commit, owner_authorized_default=False)

    assert nb_on["metadata"]["elendril_x2_source_commit"] == commit
    assert nb_on["metadata"]["elendril_x1_legacy_commit"] == X1_LEGACY_SOURCE_SHA
    assert render_notebook_bytes(source_commit=commit) == render_notebook_bytes(source_commit=commit)

    # Every code cell must parse cleanly as valid Python AST
    for nb in (nb_on, nb_off):
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    # Zero-edit published notebook must arm OWNER_AUTHORIZED = True & 10h results-only defaults
    cfg_src = "".join(nb_on["cells"][1]["source"])
    assert "OWNER_AUTHORIZED = True" in cfg_src
    assert "SESSION_MAX_MINUTES = 600" in cfg_src
    assert "OUTER_MAX_MINUTES = 660" in cfg_src
    assert "RESULTS_ONLY_EXPORT = True" in cfg_src
    assert "PRUNE_COMPLETED_WEIGHTS = True" in cfg_src
    assert commit in cfg_src
    assert X1_LEGACY_SOURCE_SHA in cfg_src

    # Disabled notebook executes inertly emitting NOT LAUNCHED without touching git/GPU
    ns: dict[str, object] = {}
    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        for cell in nb_off["cells"]:
            if cell["cell_type"] == "code" and cell["id"] != "x2-links":
                exec("".join(cell["source"]), ns)
    finally:
        sys.stdout = old_stdout
    out = buf.getvalue()
    assert "NOT LAUNCHED:" in out


def test_paired_x2_deduplication_and_double_update_billing(tmp_path: Path) -> None:
    """Paired elendril_x2 sessions schedule 1 unit per (session, seed) billed at 2x updates for both arms."""
    commit = _head_commit()
    camp = ElendrilX2Campaign(source_commit=commit, work_dir=tmp_path / "work")

    dedup_rows = camp.select_session_rows(["IIIb-DM"], deduplicate_paired_x2=True)
    raw_rows = camp.select_session_rows(["IIIb-DM"], deduplicate_paired_x2=False)
    assert len(raw_rows) == 6
    assert len(dedup_rows) == 3
    assert all(r["architecture"] == "DM" and r["control_architecture"] == "DM-add" for r in dedup_rows)

    sjobs = camp.to_session_jobs(dedup_rows)
    assert all(j.updates == 2 * 2368 for j in sjobs)
    assert all(j.extra["single_arm_updates"] == 2368 for j in sjobs)


def test_discover_kaggle_input_receipts_and_fail_fast_diagnostics(tmp_path: Path) -> None:
    """Verify mounted Kaggle input archive discovery and all-failing-wave fail-fast diagnostics."""
    commit = _head_commit()
    camp_src = ElendrilX2Campaign(source_commit=commit, work_dir=tmp_path / "src_camp")
    legacy_row = next(r for r in camp_src.registry["jobs"] if r["session"] == "A1")
    x2_row = next(r for r in camp_src.registry["jobs"] if r["session"] == "IIIb-DM")

    leg_jid = legacy_row["job_id"]
    leg_dir = tmp_path / "src_camp" / "jobs" / leg_jid / "jobs" / leg_jid
    (leg_dir / "u2368").mkdir(parents=True, exist_ok=True)
    (leg_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": leg_jid,
                "status": "COMPLETE",
                "source_commit": X1_LEGACY_SOURCE_SHA,
                "completed_updates": 2368,
                "model_spec_sha256": legacy_row["extra"]["legacy_model_spec_sha256"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (leg_dir / "u2368" / "episodes.jsonl").write_text('{"ok": true}\n' * 8, encoding="utf-8")

    x2_jid = x2_row["job_id"]
    x2_dir = tmp_path / "src_camp" / "jobs" / x2_jid
    (x2_dir / "evidence_bundles" / "TRT" / "tier1_summary").mkdir(parents=True, exist_ok=True)
    (x2_dir / "RUN_STATUS.json").write_text(
        json.dumps({"status": "COMPLETE", "campaign_id": x2_jid, "source_commit": commit, "completed_updates": 2368}) + "\n",
        encoding="utf-8",
    )
    (x2_dir / "evidence_bundles" / "TRT" / "tier1_summary" / "SCIENTIFIC_SUMMARY.json").write_text(
        json.dumps({"process_funnel": {"total_worlds": 12}}) + "\n",
        encoding="utf-8",
    )

    k_in = tmp_path / "kaggle_input" / "prev_session"
    camp_src.export_archives(k_in, prefix="ELENDRIL_V8_X2", status="PARTIAL")

    camp_dst = ElendrilX2Campaign(source_commit=commit, work_dir=tmp_path / "dst_camp")
    discovered = discover_kaggle_input_receipts(k_in, tmp_path / "ext", camp_dst.registry)
    res_rec = build_verified_resume_set(registry=camp_dst.registry, work_dir=tmp_path / "dst_camp", imported_receipts=discovered)
    assert set(res_rec["verified_completed_ids"]) == {leg_jid, x2_jid}

    # All-failing wave must abort remaining waves immediately and surface diagnostics
    def _fail_runner(spec: Any) -> dict[str, Any]:
        spec.work_root.mkdir(parents=True, exist_ok=True)
        return {"exit_code": 2, "stderr": "argparse: unrecognized arguments: --bogus"}

    fail_camp = ElendrilX2Campaign(source_commit=commit, work_dir=tmp_path / "fail_camp", in_process_runner=_fail_runner)
    rows_4 = fail_camp.select_session_rows(["A1"])[:4]
    summary = fail_camp.run_admitted_waves(
        fail_camp.to_session_jobs(rows_4),
        devices=("cuda:0", "cuda:1"),
        canary=True,
        fail_fast_on_empty_wave=True,
    )
    assert summary["completed_count"] == 0
    assert summary["stopped_reason"] == "FAIL_FAST_ALL_WORKERS_IN_WAVE_FAILED"
    assert len(summary["waves"]) == 1
    assert len(summary["failed_worker_diagnostics"]) == 2
    assert "unrecognized arguments" in summary["failed_worker_diagnostics"][0]["stderr_tail"]


def test_real_subprocess_legacy_x1_worker_8e62ca0f_runs_cleanly() -> None:
    """Execute the REAL pinned 8e62ca0f legacy X1 worker subprocess (no mock!) with isolated cwd/PYTHONPATH."""
    with tempfile.TemporaryDirectory(prefix="x2leg_") as td:
        root_tmp = Path(td)
        legacy_dir = root_tmp / "legacy_x1"
        subprocess.run(["git", "clone", "--no-checkout", str(REPO_ROOT), str(legacy_dir)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(legacy_dir), "checkout", "--detach", X1_LEGACY_SOURCE_SHA], check=True, capture_output=True)

        commit = _head_commit()
        work_dir = root_tmp / "work"
        camp = ElendrilX2Campaign(
            source_commit=commit,
            work_dir=work_dir,
            legacy_x1_repo_dir=legacy_dir,
        )
        row = next(r for r in camp.registry["jobs"] if r["job_id"] == "X1-A1-S20-F-standard-s7")
        summary = camp.run_admitted_waves(
            camp.to_session_jobs([row]),
            devices=("cpu",),
            canary=True,
            legacy_updates_override=2,
        )
        assert summary["failed_job_ids"] == [], summary.get("failed_worker_diagnostics")
        assert summary["completed_jobs"] == ["X1-A1-S20-F-standard-s7"]
        job_json = work_dir / "jobs" / "X1-A1-S20-F-standard-s7" / "jobs" / "X1-A1-S20-F-standard-s7" / "job.json"
        assert job_json.is_file()
        doc = json.loads(job_json.read_text(encoding="utf-8"))
        assert doc["completed_updates"] == 2
        assert doc["source_commit"] == X1_LEGACY_SOURCE_SHA

        exp_dir = root_tmp / "out"
        camp.export_archives(exp_dir, prefix="ELENDRIL_V8_X2", status="PARTIAL")
        discovered = discover_kaggle_input_receipts(exp_dir, root_tmp / "ext", camp.registry)
        by_id = {r["job_id"]: r for r in discovered}
        assert "X1-A1-S20-F-standard-s7" in by_id
        assert by_id["X1-A1-S20-F-standard-s7"]["completed_updates"] == 2
        assert by_id["X1-A1-S20-F-standard-s7"]["eval_rows"] > 0

        res_rec = build_verified_resume_set(
            registry=camp.registry,
            work_dir=root_tmp / "check_resume",
            imported_receipts=discovered,
        )
        assert res_rec["verified_count"] == 0
        assert "LEGACY_MODEL_SPEC_MISMATCH" in [r["reason"] for r in res_rec["rejected"]]


def test_real_subprocess_elendril_x2_worker_and_resume_discovery() -> None:
    """Execute a REAL elendril_x2 canary subprocess, verify anti-bloat ZIP export and FINAL_RECEIPT resume discovery."""
    with tempfile.TemporaryDirectory(prefix="x2run_") as td:
        root_tmp = Path(td)
        commit = _head_commit()
        work_dir = root_tmp / "work"
        camp = ElendrilX2Campaign(source_commit=commit, work_dir=work_dir)
        row = next(
            r for r in camp.select_session_rows(["IIIb-DM"]) if r["job_id"] == "X2-IIIb-DM-S20-DM-s7"
        )
        summary = camp.run_admitted_waves(
            camp.to_session_jobs([row]),
            devices=("cpu",),
            canary=True,
        )
        assert summary["failed_job_ids"] == [], summary.get("failed_worker_diagnostics")
        assert summary["completed_jobs"] == ["X2-IIIb-DM-S20-DM-s7"]
        jdir = work_dir / "jobs" / "X2-IIIb-DM-S20-DM-s7"
        assert (jdir / "FINAL_RECEIPT.json").is_file()
        assert (jdir / "ARM_RESULTS.json").is_file()

        exp_dir = root_tmp / "out"
        delivery = camp.export_archives(exp_dir, prefix="ELENDRIL_V8_X2", status="PARTIAL")
        full_zip = Path(delivery["archives"]["full"])
        with zipfile.ZipFile(full_zip, "r") as zf:
            names = set(zf.namelist())
        assert "jobs/X2-IIIb-DM-S20-DM-s7/result_bundle/checkpoint_full.zip" in names
        assert not any("/result_bundle/generations/" in n for n in names)
        assert not any("/checkpoints/" in n for n in names)

        discovered = discover_kaggle_input_receipts(exp_dir, root_tmp / "ext", camp.registry)
        by_id = {r["job_id"]: r for r in discovered}
        assert "X2-IIIb-DM-S20-DM-s7" in by_id
        assert by_id["X2-IIIb-DM-S20-DM-s7"]["completed_updates"] == 2
        assert by_id["X2-IIIb-DM-S20-DM-s7"]["eval_rows"] > 0


def test_dual_gpu_cuda_visible_devices_and_backend_env_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure dual-GPU workers receive disjoint CUDA_VISIBLE_DEVICES and x2 workers get PLANTHIR_BACKEND_MODULES."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    seen_env: dict[str, dict[str, str]] = {}

    def _rec_runner(spec: Any, env: dict[str, str], cmd: list[str]) -> int:
        seen_env[spec.worker_id] = {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", ""),
            "PLANTHIR_BACKEND_MODULES": env.get("PLANTHIR_BACKEND_MODULES", ""),
        }
        return 0

    commit = _head_commit()
    camp = ElendrilX2Campaign(source_commit=commit, work_dir=tmp_path / "gpu_env", in_process_runner=_rec_runner)
    rows = camp.select_session_rows(["IIIb-DM"])[:2]
    summary = camp.run_admitted_waves(
        camp.to_session_jobs(rows),
        devices=("cuda:0", "cuda:1"),
        canary=True,
    )
    assert summary["failed_job_ids"] == []
    assert seen_env[rows[0]["job_id"]]["CUDA_VISIBLE_DEVICES"] == "0"
    assert seen_env[rows[1]["job_id"]]["CUDA_VISIBLE_DEVICES"] == "1"
    assert seen_env[rows[0]["job_id"]]["PLANTHIR_BACKEND_MODULES"] == "experiments.elendril_x2"
    assert seen_env[rows[1]["job_id"]]["PLANTHIR_BACKEND_MODULES"] == "experiments.elendril_x2"


def test_combined_10h_session_pruning_and_results_only_export(tmp_path: Path) -> None:
    """Verify completed-job weight pruning, PARTIAL protection, results_only=True complete ZIP, and 43-job admission."""
    commit = _head_commit()
    camp = ElendrilX2Campaign(
        source_commit=commit,
        work_dir=tmp_path / "camp10h",
        outer_max_minutes=660.0,
        session_max_minutes=600.0,
        export_reserve_minutes=45.0,
    )
    camp.prepare(DEFAULT_X2_SESSIONS)
    (tmp_path / "camp10h" / "specs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "camp10h" / "specs" / "sample_spec.json").write_text('{"ok": true}\n', encoding="utf-8")

    rows_leg = [r for r in camp.select_session_rows(["A1"]) if r["backend_id"] == "x1_s20_legacy"]
    leg_done_row = rows_leg[0]
    leg_partial_row = rows_leg[1]
    x2_row = next(r for r in camp.select_session_rows(["IIIb-DM"]))

    # 1. Completed legacy X1 job with heavy weights alongside complete results/episodes
    ld_jid = leg_done_row["job_id"]
    ld_root = tmp_path / "camp10h" / "jobs" / ld_jid
    ld_sub = ld_root / "jobs" / ld_jid
    (ld_sub / "milestones" / "u2368").mkdir(parents=True, exist_ok=True)
    (ld_root / "checkpoints" / ld_jid / "objects" / "abc").mkdir(parents=True, exist_ok=True)
    (ld_root / "shared_backbones").mkdir(parents=True, exist_ok=True)
    (ld_sub / "job.json").write_text(
        json.dumps(
            {
                "job_id": ld_jid,
                "status": "COMPLETE",
                "source_commit": X1_LEGACY_SOURCE_SHA,
                "completed_updates": 2368,
                "model_spec_sha256": leg_done_row["extra"]["legacy_model_spec_sha256"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (ld_sub / "updates.jsonl").write_text('{"update": 1, "loss": 0.4}\n', encoding="utf-8")
    (ld_sub / "milestones" / "u2368" / "episodes.jsonl").write_text('{"correct": false}\n' * 4, encoding="utf-8")
    (ld_sub / "milestones" / "u2368" / "summary.json").write_text('{"accuracy": 0.75}\n', encoding="utf-8")
    (ld_sub / "milestones" / "u2368" / "model_inference.bin").write_bytes(b"WEIGHTS" * 100)
    (ld_root / "checkpoints" / ld_jid / "objects" / "abc" / "optimizer.bin").write_bytes(b"OPT" * 100)
    (ld_root / "shared_backbones" / "init.bin").write_bytes(b"BB" * 100)

    # 2. Completed X2 job with probes.npz + result.zip + heavy checkpoint_full.zip/generations/checkpoints
    x2_jid = x2_row["job_id"]
    x2_root = tmp_path / "camp10h" / "jobs" / x2_jid
    (x2_root / "evidence_bundles" / "TRT" / "tier2_capsule").mkdir(parents=True, exist_ok=True)
    (x2_root / "result_bundle" / "generations" / "gen-0001").mkdir(parents=True, exist_ok=True)
    (x2_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (x2_root / "FINAL_RECEIPT.json").write_text(
        json.dumps({"status": "COMPLETE", "source_commit": commit}) + "\n", encoding="utf-8"
    )
    (x2_root / "ARM_RESULTS.json").write_text(
        json.dumps(
            {
                "CTRL": {"milestones": [{"global_update": 2368}], "process_funnel": {"episodes": 6}},
                "TRT": {"milestones": [{"global_update": 2368}], "process_funnel": {"episodes": 6}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez(x2_root / "evidence_bundles" / "TRT" / "tier2_capsule" / "probes.npz", acts=np.zeros((2, 3)))
    (x2_root / "result_bundle" / "CURRENT.json").write_text('{"current_generation": "gen-0001"}\n', encoding="utf-8")
    (x2_root / "result_bundle" / "result.zip").write_bytes(b"RESULT_ZIP_BYTES")
    (x2_root / "result_bundle" / "checkpoint_full.zip").write_bytes(b"FULL_CKPT" * 100)
    (x2_root / "result_bundle" / "generations" / "gen-0001" / "checkpoint_full.zip").write_bytes(b"GEN_CKPT" * 100)
    (x2_root / "checkpoints" / "model.bin").write_bytes(b"CKPT_BIN" * 100)

    # 3. PARTIAL legacy X1 job — must NEVER be pruned
    lp_jid = leg_partial_row["job_id"]
    lp_root = tmp_path / "camp10h" / "jobs" / lp_jid
    (lp_root / "jobs" / lp_jid).mkdir(parents=True, exist_ok=True)
    (lp_root / "checkpoints" / "state").mkdir(parents=True, exist_ok=True)
    (lp_root / "jobs" / lp_jid / "job.json").write_text(
        json.dumps({"job_id": lp_jid, "status": "PARTIAL", "completed_updates": 64}) + "\n",
        encoding="utf-8",
    )
    partial_opt = lp_root / "checkpoints" / "state" / "optimizer.bin"
    partial_opt.write_bytes(b"KEEP_PARTIAL_OPT")

    p_ld = prune_completed_job_heavy_artifacts(ld_root)
    p_x2 = prune_completed_job_heavy_artifacts(x2_root)
    p_lp = prune_completed_job_heavy_artifacts(lp_root)

    assert p_ld["pruned"] is True and p_ld["bytes_freed"] > 0
    assert not (ld_sub / "milestones" / "u2368" / "model_inference.bin").exists()
    assert not (ld_root / "checkpoints").exists()
    assert not (ld_root / "shared_backbones").exists()
    assert (ld_sub / "job.json").is_file()
    assert (ld_sub / "updates.jsonl").is_file()
    assert (ld_sub / "milestones" / "u2368" / "episodes.jsonl").is_file()
    assert (ld_sub / "milestones" / "u2368" / "summary.json").is_file()

    assert p_x2["pruned"] is True and p_x2["bytes_freed"] > 0
    assert not (x2_root / "checkpoints").exists()
    assert not (x2_root / "result_bundle" / "generations").exists()
    assert not (x2_root / "result_bundle" / "checkpoint_full.zip").exists()
    assert (x2_root / "FINAL_RECEIPT.json").is_file()
    assert (x2_root / "ARM_RESULTS.json").is_file()
    assert (x2_root / "evidence_bundles" / "TRT" / "tier2_capsule" / "probes.npz").is_file()
    assert (x2_root / "result_bundle" / "result.zip").is_file()

    assert p_lp["pruned"] is False and p_lp["bytes_freed"] == 0
    assert partial_opt.is_file()

    out_dir = tmp_path / "out_results_only"
    delivery = camp.export_archives(out_dir, prefix="ELENDRIL_V8_X2", status="PARTIAL", results_only=True)
    assert "full" not in delivery["archives"]
    assert not (out_dir / "ELENDRIL_V8_X2_FULL.zip").exists()
    a_zip = Path(delivery["archives"]["analysis"])
    assert a_zip.is_file()

    with zipfile.ZipFile(a_zip, "r") as zf:
        names = set(zf.namelist())
    assert f"jobs/{ld_jid}/jobs/{ld_jid}/job.json" in names
    assert f"jobs/{ld_jid}/jobs/{ld_jid}/updates.jsonl" in names
    assert f"jobs/{ld_jid}/jobs/{ld_jid}/milestones/u2368/episodes.jsonl" in names
    assert f"jobs/{ld_jid}/jobs/{ld_jid}/milestones/u2368/summary.json" in names
    assert f"jobs/{x2_jid}/FINAL_RECEIPT.json" in names
    assert f"jobs/{x2_jid}/ARM_RESULTS.json" in names
    assert f"jobs/{x2_jid}/evidence_bundles/TRT/tier2_capsule/probes.npz" in names
    assert f"jobs/{x2_jid}/result_bundle/result.zip" in names
    assert "specs/sample_spec.json" in names
    assert not any(n.endswith(".bin") or n.endswith("checkpoint_full.zip") for n in names)

    discovered = discover_kaggle_input_receipts(out_dir, tmp_path / "ext_ro", camp.registry)
    res_rec = build_verified_resume_set(
        registry=camp.registry, work_dir=tmp_path / "res_ro", imported_receipts=discovered
    )
    assert set(res_rec["verified_completed_ids"]) == {ld_jid, x2_jid}

    plan_10h = camp.admit_slice(
        DEFAULT_X2_SESSIONS,
        measured_seconds_per_update=0.65,
        measured_per_job_disk_bytes=32 * 1024 * 1024,
        parallel_workers=2,
        available_disk_bytes=15 * 1024**3,
    )
    assert len(plan_10h["admitted_job_ids"]) == 43

