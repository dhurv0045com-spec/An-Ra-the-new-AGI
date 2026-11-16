import json
import zipfile

import pytest
import torch

from anra_v5.elendril.training import _dump
from anra_v5.planthir.checkpoint_intelligence import CheckpointCatalog
from anra_v5.planthir.evidence import (
    _build_fp16_model_payload, _safe_arcname, _write_verified_zip,
    export_result_bundle, verify_single_planthir_zip,
)


def _archive(path, *, kind="checkpoint_full", entries=()):
    files = {
        "FINAL_STATUS.json": json.dumps({"status": "COMPLETE", "source_commit": "abc"}).encode(),
        "CHECKPOINT_INDEX.json": json.dumps({"source_commit": "abc", "entries": list(entries)}).encode(),
        "RESULT_OVERVIEW.md": b"overview",
    }
    _write_verified_zip(path, files, meta={"kind": kind, "source_commit": "abc"})


def test_duplicate_zip_member_is_rejected(tmp_path):
    path = tmp_path / "result.zip"
    _archive(path)
    with pytest.warns(UserWarning), zipfile.ZipFile(path, "a") as zf:
        zf.writestr("FINAL_STATUS.json", b"{}")
    with pytest.raises(ValueError, match="duplicate"):
        verify_single_planthir_zip(path)


@pytest.mark.parametrize("name", ["C:/outside", "foo\\bar", "foo//bar", "foo/../bar"])
def test_archive_names_are_portable_and_canonical(name):
    with pytest.raises(ValueError, match="unsafe"):
        _safe_arcname(name)


def test_fp16_overflow_is_rejected():
    with pytest.raises(ValueError, match="overflows"):
        _build_fp16_model_payload(_dump({"weight": torch.tensor([1e10])}))


def test_failed_size_cap_preserves_previous_result(tmp_path):
    previous = tmp_path / "result.zip"
    previous.write_bytes(b"previous result")
    with pytest.raises(ValueError, match="exceeds cap"):
        export_result_bundle(tmp_path, campaign_id="x", source_commit="abc", status="COMPLETE", experiment_bundles={}, max_result_zip_bytes=1)
    assert previous.read_bytes() == b"previous result"


def test_restore_rejects_non_full_and_preserves_destination(tmp_path):
    archive = tmp_path / "result.zip"
    _archive(archive, kind="result")
    dest = tmp_path / "existing"
    dest.mkdir()
    (dest / "valuable").write_text("keep")
    with pytest.raises(ValueError, match="not a full"):
        CheckpointCatalog.restore_full_checkpoint_zip(archive, dest)
    assert (dest / "valuable").read_text() == "keep"


def test_invalid_checkpoint_fails_before_publishing_restore(tmp_path):
    archive = tmp_path / "full.zip"
    _archive(archive, entries=[{"lineage_id": "lineage", "latest_checkpoint_sha256": "a" * 64}])
    dest = tmp_path / "restore"
    with pytest.raises(ValueError, match="LATEST mismatch"):
        CheckpointCatalog.restore_full_checkpoint_zip(archive, dest)
    assert not dest.exists()


def test_archive_rejects_case_aliases_and_resource_limits(tmp_path):
    path = tmp_path / "full.zip"
    _archive(path)
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("final_status.json", b"alias")
    with pytest.raises(ValueError, match="aliases"):
        verify_single_planthir_zip(path)
    _archive(path)
    with pytest.raises(ValueError, match="member count"):
        verify_single_planthir_zip(path, max_members=1)
    with pytest.raises(ValueError, match="uncompressed size"):
        verify_single_planthir_zip(path, max_uncompressed_bytes=1)


def test_capsule_probe_is_exploratory_and_supports_groups(tmp_path):
    import numpy as np
    from anra_v5.planthir.evidence import write_tier2_diagnostic_capsule, probe_from_capsule
    write_tier2_diagnostic_capsule(
        tmp_path, experiment_id="x", diagnostics={},
        probe_arrays={"decision_hidden": np.eye(6), "y_op": np.array([0, 0, 1, 1, -1, 0])},
    )
    result = probe_from_capsule(tmp_path, group_ids=["a", "a", "b", "b", "c", "c"])
    assert result["qualification"] == "exploratory_only"
    assert result["independent_evaluation"] is False
    assert result["split_method"] == "group_order"
    with pytest.raises(ValueError, match="at least two groups"):
        probe_from_capsule(tmp_path, group_ids=["a"] * 6)


def test_zip_writer_streams_path_entries_and_binds_written_bytes(tmp_path, monkeypatch):
    source = tmp_path / "payload"
    source.write_bytes(b"abc" * 100_000)
    archive = tmp_path / "archive.zip"
    original = type(source).read_bytes
    def reject_bulk_read(path):
        if path == source:
            raise AssertionError("bulk read of archive source")
        return original(path)
    monkeypatch.setattr(type(source), "read_bytes", reject_bulk_read)
    manifest = _write_verified_zip(archive, {"payload": source}, meta={})
    import hashlib
    assert manifest["files"][0]["sha256"] == hashlib.sha256(b"abc" * 100_000).hexdigest()
    with zipfile.ZipFile(archive) as zf:
        assert zf.read("payload") == b"abc" * 100_000


def test_oversized_capsule_samples_and_full_archive_preserves_raw(tmp_path):
    import numpy as np
    from anra_v5.planthir.evidence import write_tier2_diagnostic_capsule
    exp = tmp_path / "experiment"
    arrays = {"decision_hidden": np.random.default_rng(3).normal(size=(500, 100)).astype("float32"),
              "y_op": np.arange(500)}
    diagnostic = {"traces": [{"row": i, "text": "x" * 300} for i in range(500)]}
    t2 = write_tier2_diagnostic_capsule(exp / "tier2_capsule", experiment_id="exp", probe_arrays=arrays, diagnostics=diagnostic, max_bytes=5000)
    manifest = json.loads((exp / "tier2_capsule" / "CAPSULE_MANIFEST.json").read_text())
    assert manifest["sampling"]["original_rows"] == 500
    assert 0 < manifest["sampling"]["selected_rows"] < 500
    raw = exp / "raw_evidence"
    (raw / "tier1_full.json").write_text("{}")
    summary = exp / "tier1_summary"
    summary.mkdir()
    (summary / "SCIENTIFIC_SUMMARY.json").write_text('{}')
    (summary / "SCIENTIFIC_SUMMARY.md").write_text('summary')
    result = export_result_bundle(tmp_path / "bundle", campaign_id="c", source_commit="abc", status="COMPLETE", experiment_bundles={"exp": {"tier1_summary": {"json": str(summary / "SCIENTIFIC_SUMMARY.json"), "markdown": str(summary / "SCIENTIFIC_SUMMARY.md")}, "tier2_capsule": t2, "raw_evidence_dir": str(raw)}})
    with zipfile.ZipFile(result["checkpoint_full_zip"]) as zf:
        assert json.loads(zf.read("raw_evidence/exp/diagnostics_full.json")) == diagnostic
        import io
        with np.load(io.BytesIO(zf.read("raw_evidence/exp/representations_full.npz"))) as saved:
            np.testing.assert_array_equal(saved["decision_hidden"], arrays["decision_hidden"])
    with zipfile.ZipFile(result["result_zip"]) as zf:
        assert not any(n.startswith("raw_evidence/") for n in zf.namelist())
