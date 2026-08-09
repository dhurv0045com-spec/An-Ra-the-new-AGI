from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.sft_source_registry import download_sft_sources, load_sft_source_registry


def _registry(path: Path, *, url: str, digest: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "anra-sft-source-registry/v1",
                "sources": [
                    {
                        "source_id": "fixture-source",
                        "url": url,
                        "filename": "source.jsonl",
                        "sha256": digest,
                        "license": "Apache-2.0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_sft_source_registry_downloads_only_hash_verified_explicit_sources(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"prompt":"hi","answer":"hello"}\n', encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    registry = tmp_path / "registry.json"
    _registry(registry, url=source.as_uri(), digest=digest)

    report = download_sft_sources(registry, tmp_path / "output", allow_local_sources=True)
    receipt = report["sources"][0]
    assert receipt["status"] == "downloaded_verified"
    assert Path(str(receipt["path"])).read_bytes() == source.read_bytes()
    assert (
        download_sft_sources(registry, tmp_path / "output", allow_local_sources=True)["sources"][0][
            "status"
        ]
        == "already_verified"
    )


def test_sft_source_registry_rejects_unknown_license_and_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("source\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    _registry(registry, url=source.as_uri(), digest="0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        download_sft_sources(registry, tmp_path / "output", allow_local_sources=True)
    raw = json.loads(registry.read_text(encoding="utf-8"))
    raw["sources"][0]["license"] = "unknown"
    registry.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="license"):
        load_sft_source_registry(registry, allow_local_sources=True)
