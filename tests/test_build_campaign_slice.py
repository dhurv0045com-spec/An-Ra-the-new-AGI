from __future__ import annotations

import json
from pathlib import Path

from scripts.build_campaign_slice import build_campaign_slice


def _write_source(path: Path, lines: int, tag: str) -> Path:
    path.write_text(
        "\n".join(f"{tag} line {i}: the quick brown fox jumps over {i}" for i in range(lines))
        + "\n",
        encoding="utf-8",
    )
    return path


def test_slice_splits_deterministically_and_disjointly(tmp_path: Path) -> None:
    src = _write_source(tmp_path / "fineweb.txt", 500, "fw")
    out = tmp_path / "slice"

    first = build_campaign_slice({"fineweb_edu": src}, out, min_slice_mb=0.0)
    second = build_campaign_slice({"fineweb_edu": src}, out, min_slice_mb=0.0)

    assert first["train_sha256"] == second["train_sha256"]
    entry = first["sources"]["fineweb_edu"]
    assert entry["status"] == "sliced"
    assert entry["train_lines"] > 0
    assert entry["heldout_lines"] > 0
    assert entry["heldout_disjoint"] is True
    assert first["all_heldout_disjoint"] is True
    # Held-out file exists and is content-addressed.
    heldout = Path(entry["heldout_path"])
    assert heldout.is_file()


def test_slice_reports_min_mb_gate(tmp_path: Path) -> None:
    src = _write_source(tmp_path / "small.txt", 50, "s")
    manifest = build_campaign_slice({"fineweb_edu": src}, tmp_path / "o", min_slice_mb=50.0)
    assert manifest["meets_min_slice"] is False
    assert manifest["train_mb"] < 50.0


def test_slice_handles_missing_source(tmp_path: Path) -> None:
    present = _write_source(tmp_path / "a.txt", 100, "a")
    manifest = build_campaign_slice(
        {"fineweb_edu": present, "permissive_code": tmp_path / "missing.txt"},
        tmp_path / "o",
        min_slice_mb=0.0,
    )
    assert manifest["sources"]["permissive_code"]["status"] == "missing"
    assert manifest["sources"]["fineweb_edu"]["status"] == "sliced"
    assert manifest["sources_sliced"] == 1


def test_slice_manifest_is_written(tmp_path: Path) -> None:
    src = _write_source(tmp_path / "a.txt", 100, "a")
    out = tmp_path / "slice"
    build_campaign_slice({"fineweb_edu": src}, out, min_slice_mb=0.0)
    manifest_path = out / "campaign_slice_manifest.json"
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert "heldout_split_rule" in payload
