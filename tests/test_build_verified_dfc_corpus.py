from __future__ import annotations

import json
from pathlib import Path

from scripts.build_verified_dfc_corpus import build_verified_dfc_corpus
from training.data_pipeline import validate_dfc


def test_verified_dfc_builder_emits_only_unique_verified_rows(tmp_path: Path) -> None:
    output = tmp_path / "verified.jsonl"
    manifest = tmp_path / "manifest.json"
    report = build_verified_dfc_corpus(
        output,
        manifest_path=manifest,
        target_bytes=32_000,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["status"] == "complete"
    assert report["bytes"] >= 32_000
    assert report["records"] == len(rows)
    assert report["unique_records"] == len(rows)
    assert report["all_verified"] is True
    assert set(report["verifier_counts"]) == {"constraint_json", "formal_proof"}
    assert all(row["verified"] is True for row in rows)
    assert all(row["verifier_status"] == "verified" for row in rows)
    assert all(validate_dfc(row["text"]) for row in rows)
    assert len({row["document_sha256"] for row in rows}) == len(rows)
    assert manifest.is_file()
