"""Comprehensive verification suite for Planthir / Palantiri Level-6 upgrades."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from anra_v5.planthir import (
    PLANTHIR_LEVEL,
    PLANTHIR_LEVEL_NAME,
    CampaignLedgerVerifier,
    CampaignOrchestrator,
    ConfirmLock,
    Palantiri,
    SessionJob,
    StaticPromptKernel,
    SubgoalProbeSpec,
    UniversalAction,
    UniversalObservation,
    agent_info,
    build_pinned_kaggle_notebook,
    collect_minimal_checkpoint_files,
    exact_mcnemar_test,
    export_session_archives,
    holm_bonferroni_adjust,
    inspect_campaign_status,
    level_info,
    mirror_to_working_dir,
    planthir_doctor,
    prune_completed_job_artifacts,
    render_campaign_dashboard,
    run_representation_linear_probe,
    run_subgoal_hop_decomposition,
    scaffold_experiment_module,
    serialize_pinned_notebook,
    synthesize_scientific_interpretation,
    verify_campaign_ledger,
    verify_pinned_notebook_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_level6_facade_metadata_and_alias_parity() -> None:
    import anra_v5.palantiri as pal
    import anra_v5.planter as plt
    import anra_v5.planthir as pla

    assert PLANTHIR_LEVEL == 6
    assert "Level 6" in PLANTHIR_LEVEL_NAME
    assert Palantiri.level == 6
    assert list(pal.__all__) == list(pla.__all__) == list(plt.__all__)
    assert pal.Palantiri is pla.Palantiri
    assert plt.ConfirmLock is pla.ConfirmLock

    info = level_info()
    assert info["schema"] == "anra-planthir-level6-v1"
    assert info["level"] == 6
    assert set(info["subsystems"]) >= {
        "archives",
        "custody",
        "notebooks",
        "reporting",
        "runtime",
        "causal_science",
        "studio",
    }
    assert Palantiri.doctor()["platform_level"] == 6
    assert agent_info()["platform_level"] == 6


def test_confirm_lock_custody_and_run_registered_plan_integration(tmp_path: Path) -> None:
    lock_path = tmp_path / "CONFIRM_LOCK.json"
    lock = ConfirmLock(lock_path)

    missing_rep = lock.inspect(["j1"])
    assert missing_rep["exists"] is False
    assert missing_rep["custody_status"] == "CUSTODY_LIMITATION_MISSING_ORIGINAL_LOCK"
    assert missing_rep["unlocked"] is False

    sealed = lock.seal(["j2", "j1"], source_commit="a" * 40, protocol_sha256="b" * 64)
    assert sealed["status"] == "SEALED"
    assert sealed["required_job_ids"] == ["j1", "j2"]
    assert lock.seal(["j1", "j2"]) == sealed  # idempotent across sessions

    with pytest.raises(ValueError, match="CONFIRM_LOCK_MUTATION_FORBIDDEN"):
        lock.seal(["j1", "j3"])

    # Partial completion refuses unlock cleanly
    refused = lock.unlock_if_ready(["j1"])
    assert refused["unlocked"] is False
    assert refused["status"] == "SEALED"
    assert refused["pending_confirm_jobs"] == ["j2"]

    # Complete prerequisite cohort unlocks cleanly
    opened = lock.unlock_if_ready(["j1", "j2"])
    assert opened["unlocked"] is True
    assert opened["status"] == "UNLOCKED"
    assert opened["pending_confirm_jobs"] == []
    assert opened["unlock_receipt"]["original_lock_sha256"] == sealed["lock_sha256"]

    # Tampering detection fails closed
    raw_doc = json.loads(lock_path.read_text(encoding="utf-8"))
    raw_doc["required_job_ids"] = ["j1"]
    lock_path.write_text(json.dumps(raw_doc), encoding="utf-8")
    tampered = lock.inspect(["j1", "j2"])
    assert "CONFIRM_LOCK_TAMPERED" in tampered["defects"]
    assert tampered["unlocked"] is False
    with pytest.raises(ValueError, match="CONFIRM_LOCK_TAMPERED"):
        lock.seal(["j1"])

    # Integration with CampaignOrchestrator.run_registered_plan (complete vs partial lock + pruning)
    def _runner_with_ckpt(job: SessionJob, wroot: Path, device: str) -> dict:
        lin = wroot / "ckpt" / "main"
        sha_old = "1" * 64
        sha_cur = "2" * 64
        (lin / "objects" / sha_old).mkdir(parents=True, exist_ok=True)
        (lin / "objects" / sha_cur).mkdir(parents=True, exist_ok=True)
        (lin / "objects" / sha_old / "model.bin").write_bytes(b"old-weights" * 16)
        (lin / "objects" / sha_cur / "model.bin").write_bytes(b"new-weights" * 16)
        (lin / "LATEST").write_text(sha_cur + "\n", encoding="utf-8")
        (lin / "PREVIOUS").write_text(sha_old + "\n", encoding="utf-8")
        return {"exit_code": 0, "status": "COMPLETE"}

    jobs = [
        SessionJob("job_a", "P", 7, 7, 10, device="cuda:0"),
        SessionJob("job_b", "R", 7, 7, 10, device="cuda:1"),
    ]
    full_dir = tmp_path / "orch_full"
    res_full = CampaignOrchestrator(
        campaign_id="L6-FULL",
        source_commit="c" * 40,
        work_dir=full_dir,
        max_minutes=120.0,
        reserve_minutes=5.0,
        min_disk_bytes=1024,
    ).run_registered_plan(
        jobs,
        job_runner=_runner_with_ckpt,
        devices=("cuda:0",),
        measured_seconds_per_update=0.5,
        per_job_disk_bytes=1024,
        available_disk_bytes=10**8,
        confirm_lock_path=full_dir / "CONFIRM_LOCK.json",
        prune_completed_checkpoints=True,
    )
    assert res_full["status"] == "COMPLETE"
    assert res_full["confirm_lock"]["unlocked"] is True
    assert res_full["pruned_artifacts"]["pruned_jobs"] == 2
    assert res_full["pruned_artifacts"]["removed_files"] >= 4
    assert not (full_dir / "jobs" / "job_a" / "ckpt" / "main" / "PREVIOUS").exists()
    assert (full_dir / "jobs" / "job_a" / "ckpt" / "main" / "LATEST").is_file()
    assert (full_dir / "jobs" / "job_a" / "JOB_RECEIPT.json").is_file()

    # Partial run keeps ConfirmLock sealed with exact pending_confirm_jobs
    partial_dir = tmp_path / "orch_partial"
    res_partial = CampaignOrchestrator(
        campaign_id="L6-PART",
        source_commit="c" * 40,
        work_dir=partial_dir,
        max_minutes=120.0,
        reserve_minutes=5.0,
        min_disk_bytes=1024,
    ).run_registered_plan(
        jobs,
        job_runner=_runner_with_ckpt,
        devices=("cuda:0",),
        measured_seconds_per_update=60.0,
        per_job_disk_bytes=1024,
        available_disk_bytes=10**8,
        outer_remaining_minutes=16.0,  # 16 - 5 = 11m = 660s -> fits 1 job (600s), defers job_b
        confirm_lock_path=partial_dir / "CONFIRM_LOCK.json",
    )
    assert res_partial["status"] == "PARTIAL"
    assert res_partial["confirm_lock"]["unlocked"] is False
    assert res_partial["confirm_lock"]["status"] == "SEALED"
    assert res_partial["confirm_lock"]["pending_confirm_jobs"] == ["job_b"]


def test_campaign_ledger_verifier(tmp_path: Path) -> None:
    lock_p = tmp_path / "CONFIRM_LOCK.json"
    ConfirmLock(lock_p).seal(["j1", "j2"])
    ConfirmLock(lock_p).unlock_if_ready(["j1", "j2"])

    f_res = tmp_path / "RESULTS.md"
    f_res.write_text("# OK\n", encoding="utf-8")
    exp = export_session_archives(
        tmp_path,
        tmp_path / "exports",
        prefix="L6",
        summary_files={"RESULTS.md": f_res},
        source_commit="a" * 40,
        write_full_zip=False,
    )
    sum_zip = Path(exp["archives"]["summary"])

    receipts = {
        "j1": {"status": "COMPLETE", "checkpoint_sha256": "1" * 64, "parent_checkpoint_sha256": None},
        "j2": {"status": "VERIFIED", "checkpoint_sha256": "2" * 64, "parent_checkpoint_sha256": "1" * 64},
    }
    ok = CampaignLedgerVerifier.verify(
        expected_job_ids=["j1", "j2"],
        job_receipts=receipts,
        study_requirements={"S1": ["j1", "j2"]},
        confirm_lock_path=lock_p,
        archive_paths=[sum_zip],
        expected_source_commit="a" * 40,
        require_archive_sidecars=True,
        require_v6_manifest=True,
    )
    assert ok["verified"] is True
    assert ok["custody_status"] == "VERIFIED_COMPLETE"
    assert ok["verified_studies"] == ["S1"]

    partial = verify_campaign_ledger(
        expected_job_ids=["j1", "j2"],
        job_receipts={"j1": receipts["j1"]},
        study_requirements={"S1": ["j1", "j2"]},
        confirm_lock_path=lock_p,
    )
    assert partial["verified"] is False
    assert partial["custody_status"] == "VERIFIED_PARTIAL"
    assert partial["missing_jobs"] == ["j2"]
    assert len(partial["unverified_studies"]) == 1

    no_lock = verify_campaign_ledger(
        expected_job_ids=["j1"],
        job_receipts={"j1": receipts["j1"]},
        confirm_lock_path=tmp_path / "MISSING_LOCK.json",
    )
    assert no_lock["custody_status"] == "CUSTODY_LIMITATION_MISSING_ORIGINAL_LOCK"

    # Bad receipt + invalid SHA + tampered sidecar trigger DEFECTS_DETECTED
    (sum_zip.parent / f"{sum_zip.name}.sha256").write_text("f" * 64 + f"  {sum_zip.name}\n", encoding="utf-8")
    bad = verify_campaign_ledger(
        expected_job_ids=["j1", "j2"],
        job_receipts={
            "j1": {"status": "COMPLETE", "checkpoint_sha256": ""},
            "j2": {"status": "PARTIAL"},
        },
        archive_paths=[sum_zip],
    )
    assert bad["custody_status"] == "DEFECTS_DETECTED"
    assert len(bad["defects"]) >= 3


def test_streaming_archives_minimal_collection_pruning_and_mirror(tmp_path: Path, monkeypatch) -> None:
    lin = tmp_path / "job" / "ckpt" / "main"
    sha_ms = "a" * 64
    sha_prev = "b" * 64
    sha_lat = "c" * 64
    for s in (sha_ms, sha_prev, sha_lat):
        (lin / "objects" / s).mkdir(parents=True, exist_ok=True)
        (lin / "objects" / s / "model.bin").write_bytes(s.encode("ascii") * 64)
    (lin / "LATEST").write_text(sha_lat + "\n", encoding="utf-8")
    (lin / "PREVIOUS").write_text(sha_prev + "\n", encoding="utf-8")
    (lin / "MILESTONES").write_text(json.dumps({"u10": sha_ms}), encoding="utf-8")
    receipt_f = tmp_path / "job" / "JOB_RECEIPT.json"
    receipt_f.write_text('{"status": "COMPLETE"}\n', encoding="utf-8")

    minimal = collect_minimal_checkpoint_files(tmp_path / "job" / "ckpt", arc_prefix="ckpt")
    assert "ckpt/main/LATEST" in minimal
    assert "ckpt/main/MILESTONES" in minimal
    assert f"ckpt/main/objects/{sha_lat}/model.bin" in minimal
    assert f"ckpt/main/objects/{sha_ms}/model.bin" in minimal
    assert "ckpt/main/PREVIOUS" not in minimal
    assert f"ckpt/main/objects/{sha_prev}/model.bin" not in minimal

    # Verify streaming export NEVER calls Path.read_bytes on source model files
    def _forbid_read_bytes(self: Path, *args, **kwargs):
        if self.name == "model.bin":
            raise AssertionError("export_session_archives should stream model.bin via zf.write, not read_bytes()")
        return Path.open(self, "rb").read()

    monkeypatch.setattr(Path, "read_bytes", _forbid_read_bytes)
    exp = export_session_archives(
        tmp_path / "job",
        tmp_path / "exports",
        prefix="STRM",
        summary_files={"JOB_RECEIPT.json": receipt_f},
        full_files={**minimal, "JOB_RECEIPT.json": receipt_f},
        source_commit="d" * 40,
    )
    assert exp["verified"] is True and exp["v6_manifest_embedded"] is True
    monkeypatch.undo()

    # Pruning removes PREVIOUS + sha_prev while keeping LATEST, MILESTONES & receipts
    pruned = prune_completed_job_artifacts(tmp_path / "job", keep_latest=True, keep_milestones=True)
    assert pruned["removed_files"] >= 2
    assert pruned["freed_bytes"] > 0
    assert not (lin / "PREVIOUS").exists()
    assert not (lin / "objects" / sha_prev).exists()
    assert (lin / "LATEST").is_file()
    assert (lin / "objects" / sha_lat / "model.bin").is_file()
    assert (lin / "objects" / sha_ms / "model.bin").is_file()
    assert receipt_f.is_file()

    # Zero-copy hardlink mirror + OSError copy fallback
    work_dir = tmp_path / "working"
    res_mirror = mirror_to_working_dir([Path(exp["archives"]["summary"])], work_dir)
    assert len(res_mirror) == 1
    assert res_mirror[0]["mode"] in ("hardlink", "copy")
    dst_f = Path(res_mirror[0]["dst"])
    assert dst_f.is_file()
    if res_mirror[0]["mode"] == "hardlink":
        assert os.stat(exp["archives"]["summary"]).st_ino == os.stat(dst_f).st_ino


def test_notebook_preflight_order_zero_copy_and_byte_verification(tmp_path: Path) -> None:
    nb = build_pinned_kaggle_notebook(
        source_commit="e" * 40,
        title="Level 6 Preflight Gate Order",
        launch_command="python -m experiments.demo.module --mode FULL",
        owner_authorized=True,
        preflight_canary_backend="x1_rd2",
        preflight_commands=["python -m experiments.demo.smoke --u0000"],
        mirror_outputs_to_working=True,
        require_dual_gpu=True,
        emit_file_links=True,
    )
    assert len(nb["cells"]) == 4
    cell1_src = "".join(nb["cells"][1]["source"])
    gate_src = "".join(nb["cells"][2]["source"])

    # Preflight-before-launch guarantee: LAUNCH_CMD must NOT run inside Cell 1 when gate cell is present
    assert "subprocess.check_call(launch_cmd" not in cell1_src
    # Gate cell runs doctor + preflight commands + preflight canary BEFORE launching main campaign
    assert gate_src.index("PREFLIGHT_COMMANDS") < gate_src.index("subprocess.check_call(launch_cmd")
    assert gate_src.index("'doctor'") < gate_src.index("subprocess.check_call(launch_cmd")
    assert "os.link(p, dst)" in gate_src

    nb_path = tmp_path / "L6_NB.ipynb"
    nb_path.write_bytes(serialize_pinned_notebook(nb))
    ver_ok = verify_pinned_notebook_bytes(nb_path, nb)
    assert ver_ok["match"] is True and ver_ok["parsed_match"] is True

    nb_path.write_bytes(nb_path.read_bytes() + b" ")
    ver_bad = verify_pinned_notebook_bytes(nb_path, nb)
    assert ver_bad["match"] is False


def test_level6_scientific_rigor_stats_nonlinear_probe_nhop_and_convergence_guard() -> None:
    # 1. Exact McNemar & Holm-Bonferroni
    mc = exact_mcnemar_test([True] * 10 + [False] * 10, [True] * 10 + [True] * 9 + [False])
    assert mc["discordant"] == 9 and mc["c_treatment_only"] == 9 and mc["b_baseline_only"] == 0
    assert abs(mc["p_value"] - (2.0 / 512.0)) < 1e-12
    assert mc["significant_at_005"] is True

    hb = holm_bonferroni_adjust({"h1": 0.01, "h2": 0.04, "h3": 0.20}, alpha=0.05)
    assert abs(hb["adjusted_p_values"]["h1"] - 0.03) < 1e-12
    assert abs(hb["adjusted_p_values"]["h2"] - 0.08) < 1e-12
    assert hb["rejected"] == {"h1": True, "h2": False, "h3": False}

    # 2. General 3-hop subgoal decomposition
    class _ThreeHopKernel(StaticPromptKernel):
        def decompose_subgoals(self, world):
            if int(world.get("hops", 1)) < 3:
                return ()
            return (
                SubgoalProbeSpec(
                    subgoal_id=f"{world['world_id']}:h1",
                    stage_name="hop_1",
                    hop_index=1,
                    modified_world={"world_id": f"{world['world_id']}:h1", "prompt": "stage=1", "expected": "A"},
                    expected_operation="ANSWER",
                    expected_argument="A",
                ),
                SubgoalProbeSpec(
                    subgoal_id=f"{world['world_id']}:h2",
                    stage_name="hop_2",
                    hop_index=2,
                    modified_world={"world_id": f"{world['world_id']}:h2", "prompt": "stage=2", "expected": "A"},
                    expected_operation="ANSWER",
                    expected_argument="A",
                ),
                SubgoalProbeSpec(
                    subgoal_id=f"{world['world_id']}:h3",
                    stage_name="hop_3",
                    hop_index=3,
                    modified_world={"world_id": f"{world['world_id']}:h3", "prompt": "stage=3", "expected": "A"},
                    expected_operation="ANSWER",
                    expected_argument="A",
                ),
            )

    class _SubgoalOnlyPolicy:
        policy_id = "subgoal-only"

        def reset_episode(self) -> None:
            return None

        def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
            ans = "A" if "stage=" in obs.rendered_text else "B"
            return UniversalAction("ANSWER", ans, -0.05, {"A": 0.9, "B": 0.1}, "raw", "ANSWER", ans)

        def extract_representations(self, obs_batch):
            return {"decision_hidden": torch.zeros((len(obs_batch), 4))}

    k3 = _ThreeHopKernel(action_labels=("A", "B"))
    w3 = [
        {"world_id": f"mh3_{i}", "hops": 3, "prompt": "full chain", "hidden": {"expected": "A"}}
        for i in range(4)
    ]
    decomp = run_subgoal_hop_decomposition(_SubgoalOnlyPolicy(), k3, w3)
    assert decomp["worlds_decomposed"] == 4
    assert decomp["max_hop_count"] == 3
    assert decomp["per_stage_accuracy"] == {"hop_1": 1.0, "hop_2": 1.0, "hop_3": 1.0}
    assert decomp["joint_individual_hops_solvable"] == 1.0
    assert decomp["full_chain_accuracy"] == 0.0
    assert decomp["composition_gap"] == 1.0

    # 3. Nonlinear XOR manifold probe vs linear ridge probe
    def _xor_worlds(prefix: str, reps_per_quadrant: int) -> list[dict]:
        rows: list[dict] = []
        quads = [(-1.0, -1.0, "A"), (1.0, 1.0, "A"), (-1.0, 1.0, "B"), (1.0, -1.0, "B")]
        for q_idx, (x0, x1, tgt) in enumerate(quads):
            for r in range(reps_per_quadrant):
                idx = q_idx * reps_per_quadrant + r
                jitter = (r % 3 - 1) * 0.02
                rows.append(
                    {
                        "world_id": f"{prefix}_{idx}",
                        "sig": f"{prefix}_{idx}",
                        "prompt": f"xor={x0 + jitter:.3f},{x1 - jitter:.3f}",
                        "xor_vec": [x0 + jitter, x1 - jitter, 0.0, 0.0],
                        "hidden": {"expected": tgt},
                    }
                )
        return rows

    class _XorPolicy:
        policy_id = "xor-hidden-bad-head"

        def reset_episode(self) -> None:
            return None

        def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
            # Always emits 'A' so head accuracy on balanced XOR is 0.50
            return UniversalAction("ANSWER", "A", -0.2, {"A": 0.5, "B": 0.5}, "raw", "ANSWER", "A")

        def extract_representations(self, obs_batch):
            vecs = []
            for o in obs_batch:
                raw = o.rendered_text.split("xor=", 1)[1].splitlines()[0]
                a, b = [float(v) for v in raw.split(",")]
                vecs.append([a, b, 0.0, 0.0])
            return {"decision_hidden": torch.tensor(vecs, dtype=torch.float32)}

    kx = StaticPromptKernel(action_labels=("A", "B"))
    probe_res = run_representation_linear_probe(
        _XorPolicy(),
        kx,
        _xor_worlds("tr", 6),
        _xor_worlds("ev", 2),
        include_nonlinear_probe=True,
        nonlinear_hidden_dim=16,
        nonlinear_steps=160,
        nonlinear_seed=17,
    )
    assert probe_res["probe_minus_head_gap"] == probe_res["head_minus_probe_gap"]
    assert probe_res["nonlinear_probe_final_symbol_accuracy"] == 1.0
    assert probe_res["linear_vs_nonlinear_probe_gap"] > 0.25

    # 4. Epistemic convergence guard metadata on near-zero baseline accuracy
    interp = synthesize_scientific_interpretation(
        experiment_id="L6-GUARD",
        instrument_validation={"passed": True, "status": "OK"},
        process_funnel={"overall_accuracy": 0.0, "funnel": {"stage3_single_hop_grounding": 0.0}},
    )
    assert interp.metadata["convergence_guard"]["low_accuracy_unconverged"] is True
    assert interp.metadata["convergence_guard"]["max_observed_baseline_accuracy"] == 0.0


def test_cli_level_status_verify_archive_and_scaffold_subprocess(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    def _run_cli(args: list[str], extra_py: Path | None = None, expect_rc: int = 0) -> subprocess.CompletedProcess[str]:
        run_env = dict(env)
        if extra_py is not None:
            run_env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{extra_py}"
        proc = subprocess.run(
            [sys.executable, "-m", "anra_v5.palantiri", *args],
            cwd=str(REPO_ROOT),
            env=run_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == expect_rc, f"args={args}\nrc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        return proc

    # 1. `level` CLI
    p_lvl = _run_cli(["level"])
    lvl_doc = json.loads(p_lvl.stdout)
    assert lvl_doc["level"] == 6 and lvl_doc["schema"] == "anra-planthir-level6-v1"

    # 2. `scaffold` CLI -> runnable `--backend-module` doctor + init + validate
    pkg_root = tmp_path / "scaffolds"
    p_scaf = _run_cli(["scaffold", "--name", "demo_l6_exp", "--out-dir", str(pkg_root)])
    scaf_doc = json.loads(p_scaf.stdout)
    assert scaf_doc["backend_id"] == "demo_l6_exp"
    assert len(scaf_doc["created_files"]) == 3

    p_doc = _run_cli(["--backend-module", "demo_l6_exp.module", "doctor"], extra_py=pkg_root)
    doc_json = json.loads(p_doc.stdout)
    assert doc_json["green_signal"] is True
    assert "demo_l6_exp" in doc_json["checks"]["backends_and_contracts"]["registered"]

    spec_p = tmp_path / "demo_spec.json"
    _run_cli(["--backend-module", "demo_l6_exp.module", "init", "--backend", "demo_l6_exp", "--out", str(spec_p)], extra_py=pkg_root)
    val_p = _run_cli(["--backend-module", "demo_l6_exp.module", "validate", "--spec", str(spec_p), "--canary"], extra_py=pkg_root)
    assert json.loads(val_p.stdout)["valid"] is True

    # 3. `status` CLI (JSON + --markdown) & `verify-archive` CLI (pass & exit 2 on tamper)
    wroot = tmp_path / "camp_status"
    wroot.mkdir()
    (wroot / "CAMPAIGN_STATUS.json").write_text(
        json.dumps({"campaign_id": "STAT-01", "status": "PARTIAL", "total_jobs": 2, "completed_count": 1, "completed_jobs": ["j1"]}),
        encoding="utf-8",
    )
    (wroot / "PENDING_QUEUE.json").write_text(json.dumps({"pending_job_ids": ["j2"]}), encoding="utf-8")
    ConfirmLock(wroot / "CONFIRM_LOCK.json").seal(["j1", "j2"])

    stat_json = json.loads(_run_cli(["status", "--work-root", str(wroot)]).stdout)
    assert stat_json["status"] == "PARTIAL"
    assert stat_json["counts"] == {"total": 2, "completed": 1, "pending": 1, "blocked": 0, "failed": 0}
    assert stat_json["confirm_lock"]["pending_confirm_jobs"] == ["j2"]

    stat_md = _run_cli(["status", "--work-root", str(wroot), "--markdown"]).stdout
    assert "Palantiri Level-6 Campaign Dashboard" in stat_md
    assert "['j2']" in stat_md
    assert render_campaign_dashboard(stat_json).strip() == stat_md.strip()

    rep_f = tmp_path / "RES.md"
    rep_f.write_text("# Hi\n", encoding="utf-8")
    exp = export_session_archives(
        tmp_path,
        tmp_path / "arch",
        prefix="VA",
        summary_files={"RES.md": rep_f},
        source_commit="f" * 40,
        write_full_zip=False,
    )
    sum_p = exp["archives"]["summary"]
    ver_ok = json.loads(_run_cli(["verify-archive", "--archive", sum_p, "--expected-commit", "f" * 40]).stdout)
    assert ver_ok["verified"] is True

    ver_bad_p = _run_cli(["verify-archive", "--archive", sum_p, "--expected-commit", "0" * 40], expect_rc=2)
    assert json.loads(ver_bad_p.stdout)["verified"] is False
