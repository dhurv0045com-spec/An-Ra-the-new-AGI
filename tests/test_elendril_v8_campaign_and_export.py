"""CPU tests for V8 worker, campaign admission, failure states, exporter verification, and restore."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import shutil
import time
import zipfile
import pytest

from anra_v5.elendril.v8.protocol import OPTIMIZER_DEFAULTS, jobs as protocol_jobs
from tools.elendril_v8_campaign import (
    expected_updates_for_job,
    forecast_campaign_runtime,
    run_campaign,
    validate_job_completion_artifacts,
)
from tools.elendril_v8_export import (
    ANALYSIS_ZIP_NAME,
    FULL_ZIP_NAME,
    extract_verified_analysis,
    restore_checkpoint_archive,
    verify_v8_export,
)
from tools.elendril_v8_worker import (
    prepare_verified_splits,
    rescore_saved_episode_jsonl,
    run_canary,
    run_job_worker,
    run_profile,
)


def test_v8_worker_canary_and_profile_cpu_fixture(tmp_path: Path) -> None:
    src = "d" * 40
    c = run_canary(arm="F", source_commit=src, work_dir=tmp_path / "can", device="cpu", cpu_canary=True)
    assert c["passed"] is True
    assert (tmp_path / "can" / "CANARY_RESULT.json").exists()

    p = run_profile(job_id="V8-P-D1-i7-o7", source_commit=src, work_dir=tmp_path / "prof", device="cpu", cpu_canary=True)
    assert p["resume_parity_at_4_verified"] is True
    assert p["seconds_per_update"] > 0
    assert p["eval_seconds_per_episode"] > 0
    rp = p["recovery_parity"]
    assert rp["weights_match"] is True
    assert rp["optimizer_match"] is True
    assert rp["scheduler_match"] is True
    assert rp["cursor_match"] is True
    assert rp["rng_match"] is True
    assert rp["training_state_match"] is True
    assert rp["all_match"] is True
    assert rp["full_horizon_total_updates"] > 4
    assert p["estimated_full_updates"] == rp["full_horizon_total_updates"]


def test_v8_campaign_guards_failures_and_export_integrity(tmp_path: Path) -> None:
    src = "e" * 40
    repo = Path(__file__).resolve().parents[1]

    # 1. Owner switch disabled -> NOT_RUN
    res_off = run_campaign(
        phase="data-control", source_commit=src, work_root=tmp_path / "w_off", output_dir=tmp_path / "o_off",
        owner_enabled=False, repo_root=repo,
    )
    assert res_off["status"] == "NOT_RUN"
    assert "owner switch is false" in res_off["message"]

    # 2. Phase 0 archive-diagnostic -> COMPLETE + verified ZIPs + tamper rejection
    out_p0 = tmp_path / "o_p0"
    res_p0 = run_campaign(
        phase="archive-diagnostic", source_commit=src, work_root=tmp_path / "w_p0", output_dir=out_p0,
        owner_enabled=True, repo_root=repo,
    )
    assert res_p0["status"] == "COMPLETE"
    assert verify_v8_export(out_p0, expected_source_commit=src)["verified"] is True

    ext = extract_verified_analysis(out_p0 / ANALYSIS_ZIP_NAME, tmp_path / "ext_p0", expected_source_commit=src)
    assert (ext / "METRICS.json").exists()
    rst = restore_checkpoint_archive(out_p0 / FULL_ZIP_NAME, tmp_path / "rst_p0", expected_source_commit=src)
    assert "extracted_files" in rst

    # Wrong source rejected
    with pytest.raises(ValueError, match="source_commit"):
        verify_v8_export(out_p0, expected_source_commit="f" * 40)

    # Stale/tampered ZIP rejected
    bad_zip = tmp_path / "o_bad_zip"
    bad_zip.mkdir()
    for name in ("FINAL_STATUS.json", ANALYSIS_ZIP_NAME, FULL_ZIP_NAME):
        (bad_zip / name).write_bytes((out_p0 / name).read_bytes())
    with zipfile.ZipFile(bad_zip / ANALYSIS_ZIP_NAME, "w") as zf:
        zf.writestr("ARCHIVE_MANIFEST.json", json.dumps({"schema": "wrong"}))
        zf.writestr("FINAL_STATUS.json", (bad_zip / "FINAL_STATUS.json").read_text())
    with pytest.raises(ValueError):
        verify_v8_export(bad_zip, expected_source_commit=src)

    # 3. Storage & time admission denials -> NO_GO
    res_disk = run_campaign(
        phase="head-screen", source_commit=src, work_root=tmp_path / "w_disk", output_dir=tmp_path / "o_disk",
        owner_enabled=True, repo_root=repo, free_bytes_override=1024,
    )
    assert res_disk["status"] == "NO_GO"

    res_time = run_campaign(
        phase="head-screen", source_commit=src, work_root=tmp_path / "w_time", output_dir=tmp_path / "o_time",
        minutes=1.0, owner_enabled=True, repo_root=repo, cpu_smoke=True,
        canary_runner=lambda: {"passed": True, "wall_seconds": 0.1},
        profile_runner=lambda j: {"seconds_per_update": 1.0, "estimated_full_updates": 200, "resume_parity_at_4_verified": True},
    )
    assert res_time["status"] == "NO_GO"

    # 4. Missing report & wrong source -> INVALID_RESULTS
    res_missing = run_campaign(
        phase="head-screen", source_commit=src, work_root=tmp_path / "w_miss", output_dir=tmp_path / "o_miss",
        owner_enabled=True, repo_root=repo, cpu_smoke=True,
        canary_runner=lambda: {"passed": True, "wall_seconds": 0.1},
        profile_runner=lambda j: {"seconds_per_update": 0.01, "estimated_full_updates": 10, "resume_parity_at_4_verified": True},
        job_runner=lambda job, wr: {},
    )
    assert res_missing["status"] == "INVALID_RESULTS"

    # 5. Deadline preemption -> PARTIAL (with valid R2 artifacts for job 1)
    def slow_first_job(job, wr):
        res = run_job_worker(
            job_id=str(job["job_id"]),
            source_commit=src,
            work_dir=wr,
            device="cpu",
            cpu_canary=True,
            updates_override=2,
            dev_worlds_per_cohort=1,
        )
        time.sleep(0.8)
        return res

    res_partial = run_campaign(
        phase="head-screen", source_commit=src, work_root=tmp_path / "w_part", output_dir=tmp_path / "o_part",
        minutes=0.02, owner_enabled=True, repo_root=repo, cpu_smoke=True,
        updates_override=2, dev_worlds_per_cohort=1,
        safety_reserve_seconds=0.01, eval_seconds_per_job=0.0,
        canary_runner=lambda: {"passed": True, "wall_seconds": 0.0},
        profile_runner=lambda j: {"seconds_per_update": 0.0001, "estimated_full_updates": 2, "resume_parity_at_4_verified": True},
        job_runner=slow_first_job,
    )
    assert res_partial["status"] == "PARTIAL"
    assert res_partial["completed_jobs"] == 1
    assert verify_v8_export(tmp_path / "o_part", expected_source_commit=src)["verified"] is True


def test_v8_published_notebook_and_result_template_are_inert_and_pinned(tmp_path: Path) -> None:
    from tools.build_elendril_v8_notebook import REPO_HTTP_URL, verify_notebook_file, write_notebook

    assert REPO_HTTP_URL == "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
    repo = Path(__file__).resolve().parents[1]
    nb_path = repo / "notebooks" / "ELENDRIL_V8_KAGGLE_T4X2.ipynb"
    info = verify_notebook_file(nb_path)
    assert info["verified"] is True

    written_nb = write_notebook(tmp_path / "ELENDRIL_V8_KAGGLE_T4X2.ipynb", source_commit=str(info["source_commit"]))
    written_bytes = written_nb.read_bytes()
    assert b"\r\n" not in written_bytes
    written_all_src = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(written_bytes.decode("utf-8")).get("cells", [])
    )
    assert f'REPO_URL = "{REPO_HTTP_URL}"' in written_all_src
    assert verify_notebook_file(written_nb, expected_source_commit=str(info["source_commit"]))["verified"] is True

    tpl = json.loads((repo / "Experiment Result" / "V8" / "TEMPLATE_METRICS.json").read_text(encoding="utf-8"))
    assert tpl["status"] == "NOT_RUN"
    assert tpl["source_commit"] == info["source_commit"]


def test_v8_r2_stub_rejection_and_missing_or_mutated_artifacts_fail_closed(tmp_path: Path) -> None:
    src = "a" * 40
    repo = Path(__file__).resolve().parents[1]

    # 1. Pass 6 minimal 1-update stub JSON reports into run_campaign -> INVALID_RESULTS
    def _stub_runner(job, wr):
        jid = str(job["job_id"])
        jp = wr / "jobs" / f"{jid}.json"
        jp.parent.mkdir(parents=True, exist_ok=True)
        stub = {
            "job_id": jid,
            "arm": job["arm"],
            "dataset": job["dataset"],
            "condition_id": f"{job['arm']}-{job['dataset']}",
            "init_seed": job["init_seed"],
            "order_seed": job["order_seed"],
            "source_commit": src,
            "completed_updates": 1,
            "target_updates": 1,
        }
        jp.write_text(json.dumps(stub), encoding="utf-8")
        return stub

    res_stub = run_campaign(
        phase="head-screen",
        source_commit=src,
        work_root=tmp_path / "w_stub",
        output_dir=tmp_path / "o_stub",
        minutes=5.0,
        owner_enabled=True,
        repo_root=repo,
        cpu_smoke=True,
        updates_override=2,
        dev_worlds_per_cohort=1,
        safety_reserve_seconds=0.01,
        eval_seconds_per_job=0.0,
        canary_runner=lambda: {"passed": True, "wall_seconds": 0.01},
        profile_runner=lambda j: {"seconds_per_update": 0.001, "estimated_full_updates": 2, "resume_parity_at_4_verified": True},
        job_runner=_stub_runner,
    )
    assert res_stub["status"] == "INVALID_RESULTS"
    assert res_stub["errors"]

    # 2. Clean complete cpu_smoke job -> validate_job_completion_artifacts returns []
    job = protocol_jobs("head-screen")[3]  # V8-F-D1-i7-o7
    job_id = str(job["job_id"])
    clean_dir = tmp_path / "clean_job"
    clean_report = run_job_worker(
        job_id=job_id,
        source_commit=src,
        work_dir=clean_dir,
        device="cpu",
        cpu_canary=True,
        updates_override=4,
        dev_worlds_per_cohort=1,
    )
    split_sha = str(clean_report["split_audit_sha256"])
    assert validate_job_completion_artifacts(
        job,
        clean_report,
        clean_dir,
        source_commit=src,
        expected_updates=4,
        expected_split_audit_sha256=split_sha,
    ) == []

    def _clone_valid(name: str) -> tuple[Path, dict[str, object]]:
        dest = tmp_path / f"mut_{name}"
        shutil.copytree(clean_dir, dest)
        rep = json.loads(json.dumps(clean_report))
        rep["episodes_file"] = str(dest / "episodes" / f"{job_id}.jsonl")
        rep["steps_file"] = str(dest / "steps" / f"{job_id}.jsonl")
        rep["milestones_dir"] = str(dest / "milestones" / job_id)
        assert validate_job_completion_artifacts(
            job, rep, dest, source_commit=src, expected_updates=4, expected_split_audit_sha256=split_sha,
        ) == []
        return dest, rep

    def _assert_invalid(name: str, mutate_fn) -> None:
        dest, rep = _clone_valid(name)
        mutate_fn(dest, rep)
        errs = validate_job_completion_artifacts(
            job, rep, dest, source_commit=src, expected_updates=4, expected_split_audit_sha256=split_sha,
        )
        assert errs, f"expected fail-closed violation for {name}"

    # (a) checkpoints/<job_id>/ file deleted or checkpoint_sha256 mutated
    def _del_ckpt_obj(d: Path, _r: dict[str, object]) -> None:
        latest = str(_r["checkpoint_sha256"])
        (d / "checkpoints" / job_id / "objects" / latest / "model.bin").unlink()

    _assert_invalid("a_del_ckpt_obj", _del_ckpt_obj)
    _assert_invalid("a_bad_ckpt_sha", lambda _d, r: r.__setitem__("checkpoint_sha256", "0" * 64))

    # (b) steps/<job_id>.jsonl missing or duplicate update
    _assert_invalid("b_del_steps", lambda d, _r: (d / "steps" / f"{job_id}.jsonl").unlink())

    def _dup_step(d: Path, _r: dict[str, object]) -> None:
        sp = d / "steps" / f"{job_id}.jsonl"
        lines = sp.read_text(encoding="utf-8").splitlines()
        sp.write_text("\n".join(lines + [lines[-1]]) + "\n", encoding="utf-8")

    _assert_invalid("b_dup_step", _dup_step)

    # (c) milestones/<job_id>/initial.json missing
    _assert_invalid("c_del_initial", lambda d, _r: (d / "milestones" / job_id / "initial.json").unlink())

    # (d) milestones/<job_id>/epoch-4.json missing
    _assert_invalid("d_del_epoch4", lambda d, _r: (d / "milestones" / job_id / "epoch-4.json").unlink())

    # (e) episodes/<job_id>.jsonl missing, missing mode, missing cohort, tampered exact_match
    _assert_invalid("e_del_episodes", lambda d, _r: (d / "episodes" / f"{job_id}.jsonl").unlink())

    def _drop_mode(d: Path, _r: dict[str, object]) -> None:
        ep = d / "episodes" / f"{job_id}.jsonl"
        kept = [ln for ln in ep.read_text(encoding="utf-8").splitlines() if ln.strip() and json.loads(ln)["mode"] != "raw"]
        ep.write_text("\n".join(kept) + "\n", encoding="utf-8")

    def _drop_cohort(d: Path, _r: dict[str, object]) -> None:
        ep = d / "episodes" / f"{job_id}.jsonl"
        kept = [ln for ln in ep.read_text(encoding="utf-8").splitlines() if ln.strip() and json.loads(ln)["cohort"] != "R09"]
        ep.write_text("\n".join(kept) + "\n", encoding="utf-8")

    def _tamper_exact_match(d: Path, _r: dict[str, object]) -> None:
        ep = d / "episodes" / f"{job_id}.jsonl"
        rows = [json.loads(ln) for ln in ep.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rows[0]["exact_match"] = not bool(rows[0]["exact_match"])
        ep.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    _assert_invalid("e_drop_mode", _drop_mode)
    _assert_invalid("e_drop_cohort", _drop_cohort)
    _assert_invalid("e_tamper_exact", _tamper_exact_match)

    # (f) empty episode steps
    def _empty_ep_steps(d: Path, _r: dict[str, object]) -> None:
        ep = d / "episodes" / f"{job_id}.jsonl"
        rows = [json.loads(ln) for ln in ep.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rows[0]["steps"] = []
        ep.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    _assert_invalid("f_empty_steps", _empty_ep_steps)

    # (g) missing SPLIT_AUDIT.json / mismatched split_audit_sha256
    _assert_invalid("g_del_split_audit", lambda d, _r: (d / "splits" / "SPLIT_AUDIT.json").unlink())
    _assert_invalid("g_bad_split_sha", lambda _d, r: r.__setitem__("split_audit_sha256", "f" * 64))

    # (h) missing family in final_probe
    def _drop_probe_family(_d: Path, r: dict[str, object]) -> None:
        fp = dict(r["final_probe"])  # type: ignore[arg-type]
        fp["families_covered"] = list(fp["families_covered"])[:-1]
        r["final_probe"] = fp

    _assert_invalid("h_missing_family", _drop_probe_family)

    # Verify run_campaign itself fails closed as INVALID_RESULTS if a worker's milestone is missing
    def _tampered_worker(j, wr):
        res = run_job_worker(
            job_id=str(j["job_id"]),
            source_commit=src,
            work_dir=wr,
            device="cpu",
            cpu_canary=True,
            updates_override=2,
            dev_worlds_per_cohort=1,
        )
        (wr / "milestones" / str(j["job_id"]) / "epoch-4.json").unlink()
        return res

    res_campaign_fail = run_campaign(
        phase="head-screen",
        source_commit=src,
        work_root=tmp_path / "w_mut_camp",
        output_dir=tmp_path / "o_mut_camp",
        minutes=5.0,
        owner_enabled=True,
        repo_root=repo,
        cpu_smoke=True,
        updates_override=2,
        dev_worlds_per_cohort=1,
        safety_reserve_seconds=0.01,
        eval_seconds_per_job=0.0,
        canary_runner=lambda: {"passed": True, "wall_seconds": 0.01},
        profile_runner=lambda j: {"seconds_per_update": 0.001, "estimated_full_updates": 2, "resume_parity_at_4_verified": True},
        job_runner=_tampered_worker,
    )
    assert res_campaign_fail["status"] == "INVALID_RESULTS"


def test_v8_r3_weight_free_replay_archive_and_rescore(tmp_path: Path) -> None:
    src = "b" * 40
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "o_r3"
    work_dir = tmp_path / "w_r3"

    res = run_campaign(
        phase="head-screen",
        source_commit=src,
        work_root=work_dir,
        output_dir=out_dir,
        minutes=10.0,
        owner_enabled=True,
        repo_root=repo,
        cpu_smoke=True,
        updates_override=2,
        dev_worlds_per_cohort=1,
        safety_reserve_seconds=0.01,
        eval_seconds_per_job=0.0,
        canary_runner=lambda arm, source_commit, work_dir, device, cpu_canary, lr: {
            "passed": True, "arm": arm, "wall_seconds": 0.01, "exact_action_accuracy": 1.0,
        },
        profile_runner=lambda j: {
            "seconds_per_update": 0.001, "estimated_full_updates": 2, "resume_parity_at_4_verified": True,
        },
        job_runner=lambda j, wr: run_job_worker(
            job_id=str(j["job_id"]),
            source_commit=src,
            work_dir=wr,
            device="cpu",
            cpu_canary=True,
            updates_override=2,
            dev_worlds_per_cohort=1,
        ),
    )
    assert res["status"] == "COMPLETE"
    assert verify_v8_export(out_dir, expected_source_commit=src)["verified"] is True

    ext = extract_verified_analysis(out_dir / ANALYSIS_ZIP_NAME, tmp_path / "ext_r3", expected_source_commit=src)
    assert not (ext / "checkpoints").exists()
    weight_files = [
        p for p in ext.rglob("*")
        if p.is_file() and p.suffix in (".bin", ".pt", ".pth", ".safetensors")
    ]
    assert weight_files == []

    for req_file in (
        "splits/SPLIT_AUDIT.json",
        "DIAGNOSTICS.json",
        "COST_VIEWS.json",
        "PAIRWISE_COMPARISONS.json",
        "SCENARIO_LEDGER.json",
        "METRICS.json",
        "RESULTS.md",
        "FINAL_STATUS.json",
    ):
        assert (ext / req_file).exists(), f"missing {req_file} in extracted analysis archive"

    for req_dir in ("jobs", "steps", "milestones", "episodes"):
        d = ext / req_dir
        assert d.is_dir() and any(d.rglob("*")), f"missing or empty {req_dir}/ in extracted analysis archive"

    ep_files = sorted((ext / "episodes").glob("*.jsonl"))
    assert len(ep_files) == 6
    for ep_file in ep_files:
        rescored = rescore_saved_episode_jsonl(ep_file)
        assert rescored["verified"] is True
        hist = rescored["first_divergence_histogram"]
        assert isinstance(hist, dict) and hist
        assert sum(hist.values()) == rescored["episode_count"]


def test_v8_r4_b0_canary_gpu_lr_and_campaign_active_arm_canaries(tmp_path: Path) -> None:
    src = "c" * 40
    repo = Path(__file__).resolve().parents[1]

    c_b0 = run_canary(arm="B0", source_commit=src, work_dir=tmp_path / "can_b0", device="cpu", cpu_canary=True)
    assert c_b0["passed"] is True
    assert c_b0["arm"] == "B0"
    assert c_b0["exact_action_accuracy"] is not None

    assert OPTIMIZER_DEFAULTS["peak_lr"] == pytest.approx(1e-4)
    assert inspect.signature(run_canary).parameters["lr"].default is None
    assert 'OPTIMIZER_DEFAULTS["peak_lr"]' in inspect.getsource(run_canary)

    # Verify run_campaign invokes canary_runner for B0 with GPU default lr=1e-4 when cpu_smoke=False
    dc_calls: list[tuple[str, float | None, bool]] = []

    def _fake_split_prep(work_root: Path):
        return prepare_verified_splits(work_root, cpu_smoke=True, dev_worlds_per_cohort=1, conf_worlds_per_cohort=1)

    def _spy_canary_gpu(arm: str, lr: float | None, cpu_canary: bool):
        dc_calls.append((arm, lr, cpu_canary))
        return {"passed": False, "wall_seconds": 0.0}

    res_dc = run_campaign(
        phase="data-control",
        source_commit=src,
        work_root=tmp_path / "w_dc_can",
        output_dir=tmp_path / "o_dc_can",
        minutes=500.0,
        owner_enabled=True,
        repo_root=repo,
        cpu_smoke=False,
        free_bytes_override=10**12,
        split_preparer=_fake_split_prep,
        canary_runner=_spy_canary_gpu,
    )
    assert res_dc["status"] == "NO_GO"
    assert dc_calls == [("B0", 1e-4, False)]

    # Verify run_campaign invokes canary_runner for EVERY active arm in head-screen (F and J)
    hs_arms: list[str] = []

    def _spy_canary_hs(arm: str):
        hs_arms.append(arm)
        return {"passed": True, "wall_seconds": 0.01, "exact_action_accuracy": 1.0}

    res_hs = run_campaign(
        phase="head-screen",
        source_commit=src,
        work_root=tmp_path / "w_hs_can",
        output_dir=tmp_path / "o_hs_can",
        minutes=5.0,
        owner_enabled=True,
        repo_root=repo,
        cpu_smoke=True,
        dev_worlds_per_cohort=1,
        canary_runner=_spy_canary_hs,
        profile_runner=lambda j: {"resume_parity_at_4_verified": False, "seconds_per_update": 0.01},
    )
    assert res_hs["status"] == "NO_GO"
    assert hs_arms == sorted({str(j["arm"]) for j in protocol_jobs("head-screen")}) == ["F", "J"]


def test_v8_r6_measured_forecast_2x_margin_and_no_go(tmp_path: Path) -> None:
    sample_jobs = protocol_jobs("head-screen")[:2]
    fc = forecast_campaign_runtime(
        phase_jobs=sample_jobs,
        seconds_per_update=2.0,
        updates_per_job=[10, 10],
        canary_seconds=5.0,
        profile_seconds=10.0,
        lanes=2,
        eval_seconds_per_episode=0.5,
        episodes_per_job=20,
        phase_budget_seconds=100.0,
    )
    assert fc["work_margin"] == pytest.approx(2.0)
    assert fc["total_train_seconds_serial"] == pytest.approx(40.0)
    assert fc["total_eval_seconds_serial"] == pytest.approx(20.0)
    assert fc["scheduled_parallel_work_seconds"] == pytest.approx(30.0)
    assert fc["margined_parallel_work_seconds"] == pytest.approx(60.0)
    assert fc["total_forecast_seconds"] == pytest.approx(60.0 + 5.0 + 10.0 + 600.0)
    assert fc["feasible"] is False

    # Verify run_campaign rejects an infeasible phase upfront as NO_GO (RESOURCE_INFEASIBLE)
    src = "d" * 40
    repo = Path(__file__).resolve().parents[1]
    job_runner_called = False

    def _never_called(_j, _wr):
        nonlocal job_runner_called
        job_runner_called = True
        return {}

    reg_u = expected_updates_for_job(sample_jobs[0], cpu_smoke=True, updates_override=2)
    assert reg_u == 2
    res_nogo = run_campaign(
        phase="head-screen",
        source_commit=src,
        work_root=tmp_path / "w_r6_nogo",
        output_dir=tmp_path / "o_r6_nogo",
        minutes=2.0,
        owner_enabled=True,
        repo_root=repo,
        cpu_smoke=True,
        updates_override=2,
        dev_worlds_per_cohort=1,
        canary_runner=lambda: {"passed": True, "wall_seconds": 0.1},
        profile_runner=lambda j: {
            "seconds_per_update": 5.0,
            "eval_seconds_per_episode": 1.0,
            "estimated_full_updates": 20,
            "resume_parity_at_4_verified": True,
        },
        job_runner=_never_called,
    )
    assert res_nogo["status"] == "NO_GO"
    assert res_nogo["reason_code"] == "RESOURCE_INFEASIBLE"
    assert res_nogo["forecast"]["work_margin"] == pytest.approx(2.0)
    assert job_runner_called is False
