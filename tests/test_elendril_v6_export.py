from __future__ import annotations
import json
from pathlib import Path
import zipfile

import pytest
from tools import elendril_v6_export as bundle

SOURCE = "a" * 40
PROTOCOL = "b" * 64

def test_export_contains_every_receipt_checksums_and_rescore_failure(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "partial.json").write_text('{"status":"PARTIAL","failed":["arm-2"]}')
    result = bundle.export_bundle(tmp_path, source=SOURCE, protocol_sha=PROTOCOL,
        campaign={"status":"PARTIAL", "failed_arms":["arm-2"]},
        rescore=lambda *_: (_ for _ in ()).throw(RuntimeError("scorer unavailable")), reserve_bytes=0)
    assert result["status"] == "EXPORT_VERIFIED"
    assert result["scientific_rescore"]["status"] == "FAILED"
    with zipfile.ZipFile(result["zip_path"]) as zf:
        members = set(zf.namelist())
        assert {"README_RESULTS.md", "SHA256SUMS", "metrics/partial.json", "BUNDLE_MANIFEST.json"} <= members
        assert "arm-2" in zf.read("metrics/partial.json").decode()
        sums = zf.read("SHA256SUMS").decode()
        assert "metrics/partial.json" in sums and "SHA256SUMS" not in sums


def test_export_verification_failure_preserves_previous_zip(tmp_path, monkeypatch):
    report = bundle.export_bundle(tmp_path, source=SOURCE, protocol_sha=PROTOCOL,
                                  campaign={"status":"FAILED"}, reserve_bytes=0)
    destination = Path(report["zip_path"])
    old = destination.read_bytes()
    monkeypatch.setattr(bundle, "verify_zip", lambda *a, **k: (_ for _ in ()).throw(ValueError("injected corruption")))
    with pytest.raises(ValueError, match="injected corruption"):
        bundle.export_bundle(tmp_path, source=SOURCE, protocol_sha=PROTOCOL,
                             campaign={"status":"PARTIAL"}, reserve_bytes=0)
    assert destination.read_bytes() == old
    assert len(list((tmp_path / "exports").glob(".pending-*.zip"))) == 0


def test_disk_admission_failure_preserves_previous_zip(tmp_path, monkeypatch):
    first = bundle.export_bundle(tmp_path, source=SOURCE, protocol_sha=PROTOCOL,
                                 campaign={"status":"PARTIAL"}, reserve_bytes=0)
    destination = Path(first["zip_path"])
    old = destination.read_bytes()
    monkeypatch.setattr(bundle.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 0})())
    with pytest.raises(ValueError, match="NO-GO export disk"):
        bundle.export_bundle(tmp_path, source=SOURCE, protocol_sha=PROTOCOL,
                             campaign={"status":"FAILED"}, reserve_bytes=1)
    assert destination.read_bytes() == old


def test_inventory_rejects_symlinks_and_verifier_rejects_traversal(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.write_text("secret")
    link = tmp_path / "escape"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="linked"):
        bundle.inventory(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        bundle._member_name("../secret")
    with pytest.raises(ValueError, match="unsafe"):
        bundle._member_name("C:/secret")
import builtins
import json
from pathlib import Path
import subprocess

from tools.build_elendril_v6_notebook import build


def test_generated_notebook_is_registry_bound_and_all_gates_default_off():
    notebook = build("a" * 40)
    assert notebook["nbformat"] == 4
    config = "".join(notebook["cells"][1]["source"])
    assert "EXPECTED_REGISTRY_SHA256 = \"" + notebook["metadata"]["registry_sha256"] + "\"" in config
    assert "RUN_PREFLIGHT = False" in config and "RUN_CAMPAIGN = False" in config and "RUN_EXPORT = False" in config
    rendered = "".join(notebook["cells"][3]["source"])
    assert "RENDERED_REGISTRY_SHA256" in rendered
    assert not any(line.startswith("EXPECTED_REGISTRY_SHA256 =") for line in rendered.splitlines())
    assert "[512,512,640]" in rendered
    assert all(not cell.get("outputs") and cell.get("execution_count") is None
               for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "FileLink" in "".join(notebook["cells"][9]["source"])


def test_all_disabled_cells_have_no_network_gpu_or_filesystem_side_effects(monkeypatch):
    notebook = build("a" * 40)
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled notebook attempted an external side effect")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    state = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), state)
    assert state["RUN_CAMPAIGN"] is False
    assert state["ENVIRONMENT"] is None and state["v6"] is None


def test_inconsistent_owner_switches_fail_in_config_before_preflight():
    notebook = build("a" * 40)
    config = "".join(notebook["cells"][1]["source"])
    config = config.replace("RUN_RECOVERY = False", "RUN_RECOVERY = True")
    scope = {}
    try:
        exec(config, scope)
    except RuntimeError as error:
        assert "NO-GO" in str(error)
    else:
        raise AssertionError("inconsistent switches should fail in the configuration cell")
    assert "ENVIRONMENT" not in scope

def _make_small_zip(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "metrics").mkdir()
    (source / "metrics" / "partial.json").write_text('{"status":"PARTIAL"}')
    report = bundle.export_bundle(source, source=SOURCE, protocol_sha=PROTOCOL,
                                  campaign={"status":"PARTIAL"}, reserve_bytes=0)
    return source, Path(report["zip_path"])


def test_extract_verified_streams_and_verifies_to_new_outside_directory(tmp_path):
    source, archive = _make_small_zip(tmp_path)
    destination = tmp_path / "restored"
    report = bundle.extract_verified(archive, destination, reserve_bytes=0)
    assert report["status"] == "EXTRACTION_VERIFIED"
    assert report["verified_root"] == str(destination)
    assert (destination / "metrics" / "partial.json").read_bytes() == (source / "metrics" / "partial.json").read_bytes()
    assert not (destination / ".EXTRACTION_UNCOMMITTED.json").exists()
    assert bundle.verify_zip(archive)["manifest_sha256"] == report["manifest_sha256"]


def test_extract_verified_refuses_overlapping_or_insufficient_destination(tmp_path, monkeypatch):
    _, archive = _make_small_zip(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        bundle.extract_verified(archive, archive.parent / "nested", reserve_bytes=0)
    dest = tmp_path / "no-space"
    monkeypatch.setattr(bundle.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 0})())
    with pytest.raises(ValueError, match="NO-GO extraction disk"):
        bundle.extract_verified(archive, dest, reserve_bytes=0)
    assert not dest.exists()


def test_extract_verified_keeps_failed_partial_directory_marked_uncommitted(tmp_path, monkeypatch):
    _, archive = _make_small_zip(tmp_path)
    destination = tmp_path / "partial-extraction"
    original = bundle._file_sha
    calls = {"count": 0}
    def injected(path):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected readback failure")
        return original(path)
    monkeypatch.setattr(bundle, "_file_sha", injected)
    with pytest.raises(RuntimeError, match="uncommitted partial retained"):
        bundle.extract_verified(archive, destination, reserve_bytes=0)
    marker = json.loads((destination / ".EXTRACTION_UNCOMMITTED.json").read_text())
    assert marker["status"] == "FAILED_UNCOMMITTED"
    assert "injected readback failure" in marker["error"]
    assert list(destination.rglob("*.json"))

def test_extract_verified_rejects_traversal_before_creating_destination(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "not extracted")
    destination = tmp_path / "new-root"
    with pytest.raises(ValueError, match="unsafe"):
        bundle.extract_verified(archive, destination, reserve_bytes=0)
    assert not destination.exists()
    assert not (tmp_path.parent / "escape.txt").exists()


def test_extract_verified_rejects_zip_symlink_member_before_creating_destination(tmp_path):
    _, valid_archive = _make_small_zip(tmp_path)
    archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(valid_archive, "r") as original, zipfile.ZipFile(archive, "w") as output:
        for info in original.infolist():
            if info.filename == "metrics/partial.json":
                info.create_system = 3
                info.external_attr = (0o120777 << 16)
            output.writestr(info, original.read(info.filename))
    destination = tmp_path / "new-root"
    with pytest.raises(ValueError, match="symbolic-link"):
        bundle.extract_verified(archive, destination, reserve_bytes=0)
    assert not destination.exists()

def test_extract_verified_rejects_oversized_uncompressed_archive_before_destination(tmp_path, monkeypatch):
    archive = tmp_path / "oversized.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("README_RESULTS.md", "")
        output.writestr("SHA256SUMS", "")
        output.writestr("BUNDLE_MANIFEST.json", "{}")
        output.writestr("payload.bin", bytes(17 << 20))
    monkeypatch.setattr(bundle, "MAX_EXTRACT_BYTES", 1)
    destination = tmp_path / "new-root"
    with pytest.raises(ValueError, match="uncompressed verification bound"):
        bundle.extract_verified(archive, destination, reserve_bytes=0)
    assert not destination.exists()

def test_cpu_only_export_retry_is_separate_from_gpu_launch():
    """The owner must be able to re-export saved data after GPU quota expires."""
    notebook = build("a" * 40)
    config = "".join(notebook["cells"][1]["source"]).replace("RUN_EXPORT = False", "RUN_EXPORT = True")
    state = {}
    exec(config, state)
    assert state["RUN_EXPORT"] is True and state["RUN_PREFLIGHT"] is False
    source = "".join(notebook["cells"][5]["source"])
    assert "if RUN_PREFLIGHT or RUN_EXPORT:" in source
    assert "if RUN_PREFLIGHT:" in source and "        import torch" in source
    assert "v6.restore_output(PRIOR_CAMPAIGN_OUTPUT, OUTPUT" in source
    assert "saved output source, protocol, or run ID differs" in source
    assert "EXPORT_ONLY_SOURCE_AND_OUTPUT_BOUND" in source
    assert "if PRIOR_CAMPAIGN_OUTPUT is not None:" in source
    assert "independent_rescore_and_export" in "".join(notebook["cells"][11]["source"])
