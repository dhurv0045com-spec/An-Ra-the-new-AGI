from __future__ import annotations

import json

import pytest

from scripts import execute_stream_b as stream_b


def _write_json(path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_native_foundation_gate_requires_matching_complete_source_audit(
    tmp_path, monkeypatch
) -> None:
    corpus = tmp_path / "foundation.jsonl"
    corpus.write_text('{"text":"verified"}\n', encoding="utf-8")
    audit = tmp_path / "audit.json"
    status = tmp_path / "status.json"
    source_stats = {
        source: {"bytes": 1, "documents": 1}
        for source in stream_b.EXPECTED_NATIVE_SOURCES
    }
    _write_json(
        audit,
        {
            "resume_safe": True,
            "target_complete": True,
            "corpus_size_bytes": corpus.stat().st_size,
            "source_stats": source_stats,
            "failures": {"invalid_json": 0, "disallowed_licenses": 0},
        },
    )
    _write_json(
        status,
        {
            "status": "complete",
            "requested_buckets": ["base"],
            "buckets": [{"bucket": "base"}],
        },
    )
    monkeypatch.setattr(stream_b, "FOUNDATION_CORPUS", corpus)
    monkeypatch.setattr(stream_b, "FOUNDATION_AUDIT", audit)
    monkeypatch.setattr(stream_b, "DOWNLOAD_STATUS", status)

    evidence = stream_b.validate_native_foundation()

    assert evidence["corpus_bytes"] == corpus.stat().st_size
    assert set(evidence["sources"]) == stream_b.EXPECTED_NATIVE_SOURCES


def test_native_foundation_gate_rejects_missing_source_class(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "foundation.jsonl"
    corpus.write_text('{"text":"verified"}\n', encoding="utf-8")
    audit = tmp_path / "audit.json"
    status = tmp_path / "status.json"
    _write_json(
        audit,
        {
            "resume_safe": True,
            "target_complete": True,
            "corpus_size_bytes": corpus.stat().st_size,
            "source_stats": {"FineWeb-Edu": {"bytes": 1}},
            "failures": {"invalid_json": 0},
        },
    )
    _write_json(
        status,
        {
            "status": "complete",
            "requested_buckets": ["base"],
            "buckets": [{"bucket": "base"}],
        },
    )
    monkeypatch.setattr(stream_b, "FOUNDATION_CORPUS", corpus)
    monkeypatch.setattr(stream_b, "FOUNDATION_AUDIT", audit)
    monkeypatch.setattr(stream_b, "DOWNLOAD_STATUS", status)

    with pytest.raises(RuntimeError, match="source coverage failed"):
        stream_b.validate_native_foundation()


def test_native_foundation_gate_rejects_reasoning_only_status(
    tmp_path, monkeypatch
) -> None:
    corpus = tmp_path / "foundation.jsonl"
    corpus.write_text('{"text":"verified"}\n', encoding="utf-8")
    audit = tmp_path / "audit.json"
    status = tmp_path / "status.json"
    _write_json(
        audit,
        {
            "resume_safe": True,
            "target_complete": True,
            "corpus_size_bytes": corpus.stat().st_size,
            "source_stats": {},
            "failures": {},
        },
    )
    _write_json(
        status,
        {
            "status": "complete",
            "requested_buckets": ["reasoning"],
            "buckets": [{"bucket": "reasoning"}],
        },
    )
    monkeypatch.setattr(stream_b, "FOUNDATION_CORPUS", corpus)
    monkeypatch.setattr(stream_b, "FOUNDATION_AUDIT", audit)
    monkeypatch.setattr(stream_b, "DOWNLOAD_STATUS", status)

    with pytest.raises(RuntimeError, match="completed base bucket"):
        stream_b.validate_native_foundation()


def test_partial_immutable_shard_family_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(stream_b, "DATA_MANIFEST_DIR", tmp_path)
    (tmp_path / "native_foundation_v4" / "30gb").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Partial immutable V4"):
        stream_b._existing_family_inventory("v4", "30gb")


def test_existing_shard_family_requires_verified_campaign_sampling(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(stream_b, "DATA_MANIFEST_DIR", tmp_path)
    family = tmp_path / "native_foundation_v4" / "30gb"
    family.mkdir(parents=True)
    _write_json(
        family / "token_inventory.json",
        {
            "tokenizer_family": "v4",
            "campaign_sampling_verified": False,
        },
    )

    with pytest.raises(RuntimeError, match="campaign sampling"):
        stream_b._existing_family_inventory("v4", "30gb")


def test_existing_shard_family_validates_all_bound_artifacts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(stream_b, "DATA_MANIFEST_DIR", tmp_path)
    family = tmp_path / "native_foundation_v4" / "30gb"
    tokenizer_sha256 = "a" * 64
    manifests = {}
    for split, directory in {
        "manifest": family,
        "validation_manifest": family / "validation",
        "test_manifest": family / "test",
    }.items():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokens.npy").write_bytes(b"tokens")
        payload = {
            "tokenizer_sha256": tokenizer_sha256,
            "total_tokens": 7,
            "shards": [{"path": "tokens.npy"}],
        }
        if split == "manifest":
            payload["campaign_sampling_verified"] = True
        path = directory / "manifest.json"
        _write_json(path, payload)
        manifests[split] = str(path)
    _write_json(
        family / "token_inventory.json",
        {
            "tokenizer_family": "v4",
            "tokenizer_sha256": tokenizer_sha256,
            "campaign_sampling_verified": True,
            "licensed_tokens": 7,
            **manifests,
        },
    )

    inventory = stream_b._existing_family_inventory("v4", "30gb")

    assert inventory is not None
    assert inventory["licensed_tokens"] == 7
