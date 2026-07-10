from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_foundation_records import audit_foundation_records


def _record(text: str, source: str = "FineWeb-Edu") -> dict[str, object]:
    return {
        "text": text,
        "source": source,
        "license": "ODC-By",
        "source_revision": "unit@" + "a" * 40,
        "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "quality_checks": {
            "pii_redacted": True,
            "minhash_deduplicated": True,
            "language_detected": True,
            "benchmark_contamination_checked": True,
        },
    }


def _write(path: Path, rows: list[dict[str, object]], *, trailing_newline: bool = True) -> None:
    content = "\n".join(json.dumps(row) for row in rows)
    if trailing_newline:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def test_foundation_audit_builds_disk_index_and_source_totals(tmp_path: Path) -> None:
    corpus = tmp_path / "foundation.jsonl"
    _write(corpus, [_record("alpha"), _record("beta", "FineMath-4+")])

    report = audit_foundation_records(
        corpus,
        report_path=tmp_path / "report.json",
        index_path=tmp_path / "index.sqlite3",
        target_bytes=1,
    )

    assert report["structurally_valid"] is True
    assert report["resume_safe"] is True
    assert report["valid_records"] == 2
    assert report["minhash_signatures"] == 2
    assert report["source_stats"]["FineWeb-Edu"]["documents"] == 1
    assert Path(report["index_path"]).is_file()
    assert len(report["report_sha256"]) == 64


def test_foundation_audit_rejects_duplicate_and_partial_tail(tmp_path: Path) -> None:
    corpus = tmp_path / "foundation.jsonl"
    row = _record("duplicate")
    _write(corpus, [row, row], trailing_newline=False)

    report = audit_foundation_records(
        corpus,
        report_path=tmp_path / "report.json",
        index_path=tmp_path / "index.sqlite3",
    )

    assert report["structurally_valid"] is False
    assert report["resume_safe"] is False
    assert report["failures"]["duplicate_records"] == 1
    assert report["failures"]["missing_trailing_newline"] == 1
