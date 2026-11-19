"""Campaign orchestration, dual-source routing, 6 failure-injection recovery tests, and notebook checks for X2."""

from __future__ import annotations

import ast
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

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
    materialize_x2_job_spec,
)
from experiments.x2.registry import X1_LEGACY_SOURCE_SHA, build_x2_design_registry, evaluate_x2_qualification
from tools.build_elendril_x2_notebook import build_notebook_dict, render_notebook_bytes
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

    # Zero-edit published notebook must arm OWNER_AUTHORIZED = True
    cfg_src = "".join(nb_on["cells"][1]["source"])
    assert "OWNER_AUTHORIZED = True" in cfg_src
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
