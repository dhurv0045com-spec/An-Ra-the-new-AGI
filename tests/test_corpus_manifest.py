from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from training import corpus_manifest as cm


def test_campaign_manifest_is_valid_and_normalized() -> None:
    report = cm.build_corpus_manifest()

    assert report.valid is True
    assert report.violations == []
    assert report.weight_normalized is True
    assert report.all_licenses_allowed is True
    assert report.all_revisions_pinned is True
    assert report.unique_keys is True
    assert report.total_weight == 1.0
    assert report.target_clean_text_gb >= 120.0
    assert len(report.manifest_sha256) == 64


def test_manifest_hash_is_stable_and_content_addressed() -> None:
    first = cm.build_corpus_manifest().manifest_sha256
    second = cm.build_corpus_manifest().manifest_sha256
    assert first == second
    # Changing any pin changes the hash.
    mutated = tuple(
        replace(source, revision="deadbeef" * 5) if source.key == "fineweb_edu" else source
        for source in cm.CAMPAIGN_CORPUS_SOURCES
    )
    assert cm.build_corpus_manifest(mutated).manifest_sha256 != first


def test_manifest_rejects_unlicensed_source() -> None:
    bad = tuple(
        replace(source, license="GPL-3.0") if source.key == "fineweb_edu" else source
        for source in cm.CAMPAIGN_CORPUS_SOURCES
    )
    report = cm.build_corpus_manifest(bad)
    assert report.valid is False
    assert any("license" in violation for violation in report.violations)


def test_manifest_rejects_unpinned_revision() -> None:
    bad = tuple(
        replace(source, revision="latest") if source.key == "finemath" else source
        for source in cm.CAMPAIGN_CORPUS_SOURCES
    )
    report = cm.build_corpus_manifest(bad)
    assert report.valid is False
    assert any("pinned" in violation for violation in report.violations)


def test_manifest_rejects_unnormalized_weights() -> None:
    bad = tuple(
        replace(source, weight=0.99) if source.key == "fineweb_edu" else source
        for source in cm.CAMPAIGN_CORPUS_SOURCES
    )
    report = cm.build_corpus_manifest(bad)
    assert report.valid is False
    assert any("weights sum" in violation for violation in report.violations)


def test_write_manifest_emits_sorted_json(tmp_path: Path) -> None:
    target = tmp_path / "upstream.json"
    report = cm.write_corpus_manifest(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["manifest_sha256"] == report.manifest_sha256
    assert payload["schema_version"] == cm.MANIFEST_SCHEMA_VERSION
    assert len(payload["sources"]) == len(cm.CAMPAIGN_CORPUS_SOURCES)


def test_per_record_license_defers_but_is_accepted() -> None:
    code = next(s for s in cm.CAMPAIGN_CORPUS_SOURCES if s.key == "permissive_code")
    assert cm.normalize_license(code.license) == "per-record"
    assert code.license_ok() is True
