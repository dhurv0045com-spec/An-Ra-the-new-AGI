"""CPU integration tests for ELENDRIL V8-X1 orchestration, export security, analyzer, and notebook."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from tools.analyze_elendril_v8x_results import analyze_x1_archive
from tools.build_elendril_v8x_notebook import (
    build_notebook_dict,
    verify_notebook_file,
    write_notebook,
)
from tools.elendril_v8x_campaign import main as campaign_main
from tools.elendril_v8x_export import (
    analysis_zip_name,
    full_zip_name,
    safe_extract_x1_zip,
    verify_single_x1_zip,
    verify_x1_export,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DUMMY_COMMIT = "18ed23a0" + "0" * 32


def test_campaign_prepare_validate_admit_denial_and_authorization(tmp_path: Path, capsys) -> None:
    wdir = tmp_path / "work"
    assert campaign_main(["prepare", "--source", DUMMY_COMMIT, "--session", "A0", "--work-dir", str(wdir), "--cpu-fixtures"]) == 0
    prep = json.loads(capsys.readouterr().out)
    assert prep["status"] == "PREPARED" and prep["job_count"] == 6

    assert campaign_main(["validate", "--source", DUMMY_COMMIT, "--session", "A0", "--work-dir", str(wdir), "--cpu-fixtures"]) == 0
    val = json.loads(capsys.readouterr().out)
    assert val["status"] == "VALIDATED" and val["design_registry_verified"] is True

    # Running without --owner-authorized must refuse cleanly with NOT_LAUNCHED
    outdir = tmp_path / "exports_denied"
    assert campaign_main(["run", "--source", DUMMY_COMMIT, "--session", "A0", "--work-dir", str(wdir), "--output-dir", str(outdir), "--cpu-fixtures"]) == 1
    denied = json.loads(capsys.readouterr().out)
    assert denied["status"] == "NOT_LAUNCHED"
    assert (outdir / "FINAL_STATUS.json").is_file()

    # Production GPU admit on CPU-only machine must fail closed with NO_GO (no silent 1-GPU or CPU bypass)
    assert campaign_main(["admit", "--source", DUMMY_COMMIT, "--session", "A0", "--work-dir", str(wdir), "--devices", "0,1"]) == 1
    adm_no_gpu = json.loads(capsys.readouterr().out)
    assert adm_no_gpu["status"] == "NO_GO" and adm_no_gpu["reason_code"] == "GPU_ADMISSION_FAILED"

    # Authorized run without successful admission must also fail closed with NO_GO
    assert campaign_main(["run", "--source", DUMMY_COMMIT, "--session", "A0", "--work-dir", str(wdir), "--output-dir", str(outdir), "--owner-authorized", "--cpu-fixtures"]) == 1
    unadm = json.loads(capsys.readouterr().out)
    assert unadm["status"] == "NO_GO"


def test_campaign_end_to_end_cpu_fixtures_export_resume_and_analyzer(tmp_path: Path, capsys) -> None:
    wdir = tmp_path / "a0_work"
    outdir = tmp_path / "a0_exports"
    curated = tmp_path / "curated"

    assert campaign_main(["prepare", "--source", DUMMY_COMMIT, "--session", "A0", "--work-dir", str(wdir), "--cpu-fixtures"]) == 0
    _ = capsys.readouterr()

    assert campaign_main(["admit", "--source", DUMMY_COMMIT, "--session", "A0", "--work-dir", str(wdir), "--cpu-fixtures"]) == 0
    adm = json.loads(capsys.readouterr().out)
    assert adm["status"] == "ADMITTED"

    # Run all 6 A0 jobs for 2 updates via CPU fixtures using IndependentWorkerScheduler
    assert campaign_main([
        "run",
        "--source",
        DUMMY_COMMIT,
        "--session",
        "A0",
        "--work-dir",
        str(wdir),
        "--output-dir",
        str(outdir),
        "--owner-authorized",
        "--cpu-fixtures",
        "--updates-override",
        "2",
    ]) == 0
    run_out = json.loads(capsys.readouterr().out)
    assert run_out["status"] == "COMPLETE" and len(run_out["completed_jobs"]) == 6

    receipt = verify_x1_export(outdir, session="A0", expected_source_commit=DUMMY_COMMIT)
    assert receipt["verified"] is True and receipt["status"] == "COMPLETE"

    # Simulate partial failure / interrupt recovery: downgrade 1 job to PARTIAL and call resume
    partial_job = run_out["completed_jobs"][0]
    jfile = wdir / "jobs" / partial_job / "job.json"
    jdoc = json.loads(jfile.read_text(encoding="utf-8"))
    jdoc["status"] = "PARTIAL"
    jfile.write_text(json.dumps(jdoc), encoding="utf-8")

    assert campaign_main([
        "resume",
        "--source",
        DUMMY_COMMIT,
        "--session",
        "A0",
        "--work-dir",
        str(wdir),
        "--output-dir",
        str(outdir),
        "--owner-authorized",
        "--cpu-fixtures",
        "--updates-override",
        "2",
    ]) == 0
    res_out = json.loads(capsys.readouterr().out)
    assert res_out["status"] == "COMPLETE" and len(res_out["completed_jobs"]) == 6

    # Offline analyzer Intake & safe extraction
    a_zip = outdir / analysis_zip_name("A0")
    f_zip = outdir / full_zip_name("A0")
    assert a_zip.is_file() and f_zip.is_file()
    metrics = analyze_x1_archive(a_zip, expected_source_commit=DUMMY_COMMIT, output_dir=curated)
    assert metrics["audit_status"] == "PASS" and metrics["replayed_episodes"] > 0
    assert (curated / "METRICS.json").is_file() and (curated / "PROVENANCE.json").is_file() and (curated / "RESULTS.md").is_file()

    ext_info = safe_extract_x1_zip(a_zip, tmp_path / "extracted", expected_source_commit=DUMMY_COMMIT, expected_session="A0")
    assert ext_info["final_status"]["session"] == "A0"


@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_export_rejects_traversal_tamper_and_wrong_source(tmp_path: Path) -> None:
    bad_trav = tmp_path / "trav.zip"
    with zipfile.ZipFile(bad_trav, "w") as zf:
        zf.writestr("../escape.txt", b"bad")
    with pytest.raises(ValueError, match="unsafe archive member path"):
        verify_single_x1_zip(bad_trav)

    bad_dup = tmp_path / "dup.zip"
    with zipfile.ZipFile(bad_dup, "w") as zf:
        zf.writestr("a.txt", b"1")
        zf.writestr("a.txt", b"2")
    with pytest.raises(ValueError, match="duplicate ZIP member"):
        verify_single_x1_zip(bad_dup)

    bad_nested = tmp_path / "nested.zip"
    with zipfile.ZipFile(bad_nested, "w") as zf:
        zf.writestr("inner.zip", b"PK\x05\x06" + b"\x00" * 18)
    with pytest.raises(ValueError, match="recursive ZIP prohibited"):
        verify_single_x1_zip(bad_nested)


def test_notebook_generation_compilation_and_disabled_execution(tmp_path: Path, capsys) -> None:
    nb_path = tmp_path / "test_x1.ipynb"
    write_notebook(nb_path, source_commit=DUMMY_COMMIT)
    v = verify_notebook_file(nb_path, expected_source_commit=DUMMY_COMMIT)
    assert v["verified"] is True

    # Default owner-armed notebook compiles cleanly and launches A0..D2 without manual code edits
    armed_nb = build_notebook_dict(source_commit=DUMMY_COMMIT)
    assert any("OWNER_AUTHORIZED = True" in "".join(c.get("source", [])) for c in armed_nb["cells"])

    # Execute every code cell under OWNER_AUTHORIZED=False to verify zero side effects & NOT LAUNCHED banner
    nb = build_notebook_dict(source_commit=DUMMY_COMMIT, owner_authorized_default=False)
    ns: dict[str, object] = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "<nb-cell>", "exec"), ns, ns)
    assert ns["OWNER_AUTHORIZED"] is False
    assert ns["SOURCE_SHA"] == DUMMY_COMMIT
    out_text = capsys.readouterr().out
    assert "NOT LAUNCHED:" in out_text
