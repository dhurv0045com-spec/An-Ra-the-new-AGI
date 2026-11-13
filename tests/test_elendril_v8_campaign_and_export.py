"""CPU tests for V8 worker, campaign admission, failure states, exporter verification, and restore."""

from __future__ import annotations

import json
from pathlib import Path
import time
import zipfile
import pytest

from tools.elendril_v8_campaign import run_campaign
from tools.elendril_v8_export import (
    ANALYSIS_ZIP_NAME,
    FULL_ZIP_NAME,
    extract_verified_analysis,
    restore_checkpoint_archive,
    verify_v8_export,
)
from tools.elendril_v8_worker import run_canary, run_profile


def test_v8_worker_canary_and_profile_cpu_fixture(tmp_path: Path) -> None:
    src = "d" * 40
    c = run_canary(arm="F", source_commit=src, work_dir=tmp_path / "can", device="cpu", cpu_canary=True)
    assert c["passed"] is True
    assert (tmp_path / "can" / "CANARY_RESULT.json").exists()

    p = run_profile(job_id="V8-P-D1-i7-o7", source_commit=src, work_dir=tmp_path / "prof", device="cpu", cpu_canary=True)
    assert p["resume_parity_at_4_verified"] is True
    assert p["seconds_per_update"] > 0


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

    # 5. Deadline preemption -> PARTIAL
    def slow_first_job(job, wr):
        jid = str(job["job_id"])
        jp = wr / "jobs" / f"{jid}.json"
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps({
            "job_id": jid, "arm": job["arm"], "source_commit": src,
            "completed_updates": 2, "target_updates": 2, "wall_seconds": 0.05,
        }), encoding="utf-8")
        time.sleep(0.35)
        return {}

    res_partial = run_campaign(
        phase="head-screen", source_commit=src, work_root=tmp_path / "w_part", output_dir=tmp_path / "o_part",
        minutes=0.005, owner_enabled=True, repo_root=repo, cpu_smoke=True,
        safety_reserve_seconds=0.01, eval_seconds_per_job=0.0,
        canary_runner=lambda: {"passed": True, "wall_seconds": 0.0},
        profile_runner=lambda j: {"seconds_per_update": 0.0001, "estimated_full_updates": 2, "resume_parity_at_4_verified": True},
        job_runner=slow_first_job,
    )
    assert res_partial["status"] == "PARTIAL"


def test_v8_published_notebook_and_result_template_are_inert_and_pinned() -> None:
    from tools.build_elendril_v8_notebook import verify_notebook_file

    repo = Path(__file__).resolve().parents[1]
    nb_path = repo / "notebooks" / "ELENDRIL_V8_KAGGLE_T4X2.ipynb"
    info = verify_notebook_file(nb_path)
    assert info["verified"] is True
    tpl = json.loads((repo / "Experiment Result" / "V8" / "TEMPLATE_METRICS.json").read_text(encoding="utf-8"))
    assert tpl["status"] == "NOT_RUN"
    assert tpl["source_commit"] == info["source_commit"]
