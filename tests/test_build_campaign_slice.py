from __future__ import annotations

import json
from pathlib import Path

from scripts import build_campaign_slice as campaign_slice
from scripts.build_campaign_slice import (
    _record_text_and_key,
    build_campaign_slice,
    build_streaming_campaign_slice,
)


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


def test_unverified_dfc_cannot_enter_verified_campaign_bucket() -> None:
    unverified = json.dumps({"text": "candidate", "verified": False})
    text, key = _record_text_and_key(unverified, "verified_dfc")
    assert text == ""
    assert key == "unclassified"

    verified = json.dumps(
        {"text": "verified candidate", "verifier_status": "verified"}
    )
    text, key = _record_text_and_key(verified, "verified_dfc")
    assert text == "verified candidate\n"
    assert key == "verified_dfc"


def test_streaming_slice_explicitly_replay_weights_small_identity_source(
    tmp_path: Path,
) -> None:
    keys = (
        "fineweb_edu",
        "permissive_code",
        "finemath",
        "science_technical",
        "verified_instruction",
        "verified_dfc",
        "identity_replay",
    )
    sources: dict[str, Path] = {}
    for key in keys:
        path = tmp_path / f"{key}.txt"
        lines = 8 if key == "identity_replay" else 500
        if key == "verified_dfc":
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "text": f"{key} verified row {index} " + "evidence " * 8,
                            "verifier_status": "verified",
                        }
                    )
                    + "\n"
                    for index in range(lines)
                ),
                encoding="utf-8",
            )
        else:
            _write_source(path, lines, key)
        sources[key] = path

    manifest = build_streaming_campaign_slice(
        sources,
        tmp_path / "slice",
        min_slice_mb=0.0,
        max_train_mb=0.02,
    )

    identity = manifest["sources"]["identity_replay"]
    assert identity["replayed_bytes"] > 0
    assert identity["replayed_lines"] > 0
    assert manifest["all_required_sources_present"] is True
    assert manifest["campaign_mix_verified"] is True


def test_default_sources_exclude_legacy_corpus_when_native_manifest_exists(
    tmp_path: Path, monkeypatch
) -> None:
    training_data = tmp_path / "training_data"
    training_data.mkdir()
    native = training_data / "foundation_records.jsonl"
    native.write_text("{}\n", encoding="utf-8")
    (training_data / "anra_training.txt").write_text(
        "legacy unbound corpus\n", encoding="utf-8"
    )
    monkeypatch.setattr(campaign_slice, "ROOT", tmp_path)
    monkeypatch.setattr(campaign_slice, "get_identity_file", lambda: None)

    sources = campaign_slice._default_sources()

    assert sources == {"native_foundation": native}
