"""Streaming integrity audit for resumable native-foundation JSONL artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import BinaryIO

from anra.anra_paths import OUTPUT_V2_DIR, ROOT

from scripts.download_training_data import MinHashDeduplicator

DEFAULT_CORPUS = ROOT / "training_data" / "foundation_records.jsonl"
DEFAULT_REPORT = OUTPUT_V2_DIR / "foundation_records_audit.json"
DEFAULT_INDEX = OUTPUT_V2_DIR / "foundation_records_index.sqlite3"
ALLOWED_LICENSE_MARKERS = ("odc-by", "mit", "apache", "bsd", "isc", "mpl")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _open_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS documents ("
        "document_sha256 TEXT PRIMARY KEY, source TEXT NOT NULL, line_bytes INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS minhash_signatures ("
        "document_sha256 TEXT PRIMARY KEY, signature TEXT NOT NULL)"
    )
    return connection


def _last_byte(stream: BinaryIO) -> bytes:
    stream.seek(0, 2)
    size = stream.tell()
    if size == 0:
        return b""
    stream.seek(-1, 2)
    return stream.read(1)


def audit_foundation_records(
    corpus_path: str | Path,
    *,
    report_path: str | Path,
    index_path: str | Path,
    target_bytes: int = 30 * 1024**3,
) -> dict[str, object]:
    corpus = Path(corpus_path)
    report_file = Path(report_path)
    index_file = Path(index_path)
    if not corpus.is_file():
        raise FileNotFoundError(corpus)

    # Rebuild into a fresh database so stale rows can never make a corrupt file
    # appear complete. The final replace occurs only after the scan succeeds.
    temporary_index = index_file.with_suffix(index_file.suffix + ".tmp")
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(temporary_index) + suffix)
        if candidate.exists():
            candidate.unlink()
    connection = _open_index(temporary_index)
    source_stats: dict[str, dict[str, int]] = {}
    invalid_json = 0
    invalid_utf8 = 0
    missing_fields = 0
    hash_mismatches = 0
    duplicate_records = 0
    disallowed_licenses = 0
    quality_contract_failures = 0
    valid_records = 0
    minhash_records = 0
    scanned_bytes = 0
    started = time.time()

    with corpus.open("rb") as stream:
        trailing_newline = _last_byte(stream) == b"\n"
        stream.seek(0)
        for raw_line in stream:
            scanned_bytes += len(raw_line)
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                invalid_utf8 += 1
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            if not isinstance(item, dict):
                invalid_json += 1
                continue
            text = str(item.get("text", ""))
            source = str(item.get("source", "")).strip()
            license_name = str(item.get("license", "")).strip()
            declared_hash = str(item.get("document_sha256", "")).strip().lower()
            revision = str(item.get("source_revision", "")).strip()
            if not text or not source or not license_name or not declared_hash or not revision:
                missing_fields += 1
                continue
            computed_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if computed_hash != declared_hash:
                hash_mismatches += 1
                continue
            normalized_license = license_name.lower().replace("_", "-")
            if not any(marker in normalized_license for marker in ALLOWED_LICENSE_MARKERS):
                disallowed_licenses += 1
                continue
            quality = item.get("quality_checks", {})
            required_quality = (
                "pii_redacted",
                "minhash_deduplicated",
                "language_detected",
                "benchmark_contamination_checked",
            )
            if not isinstance(quality, dict) or not all(
                quality.get(key) is True for key in required_quality
            ):
                quality_contract_failures += 1
                continue
            try:
                connection.execute(
                    "INSERT INTO documents(document_sha256, source, line_bytes) VALUES (?, ?, ?)",
                    (computed_hash, source, len(raw_line)),
                )
            except sqlite3.IntegrityError:
                duplicate_records += 1
                continue
            if minhash_records < 500_000:
                signature = MinHashDeduplicator.signature(text)
                connection.execute(
                    "INSERT INTO minhash_signatures(document_sha256, signature) VALUES (?, ?)",
                    (computed_hash, json.dumps(signature, separators=(",", ":"))),
                )
                minhash_records += 1
            valid_records += 1
            row = source_stats.setdefault(source, {"documents": 0, "bytes": 0})
            row["documents"] += 1
            row["bytes"] += len(raw_line)
            if valid_records % 10_000 == 0:
                connection.commit()

    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('corpus_path', ?)",
        (str(corpus.resolve()),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('corpus_size_bytes', ?)",
        (str(corpus.stat().st_size),),
    )
    connection.commit()
    connection.close()
    if index_file.exists():
        index_file.unlink()
    temporary_index.replace(index_file)

    failures = {
        "invalid_json": invalid_json,
        "invalid_utf8": invalid_utf8,
        "missing_fields": missing_fields,
        "hash_mismatches": hash_mismatches,
        "duplicate_records": duplicate_records,
        "disallowed_licenses": disallowed_licenses,
        "quality_contract_failures": quality_contract_failures,
        "missing_trailing_newline": int(not trailing_newline),
    }
    structurally_valid = not any(failures.values())
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": time.time(),
        "corpus_path": str(corpus.resolve()),
        "corpus_size_bytes": corpus.stat().st_size,
        "target_bytes": int(target_bytes),
        "target_completion": corpus.stat().st_size / max(1, int(target_bytes)),
        "valid_records": valid_records,
        "minhash_signatures": minhash_records,
        "scanned_bytes": scanned_bytes,
        "source_stats": dict(sorted(source_stats.items())),
        "failures": failures,
        "structurally_valid": structurally_valid,
        "target_complete": corpus.stat().st_size >= int(target_bytes * 0.98),
        "resume_safe": structurally_valid,
        "index_path": str(index_file.resolve()),
        "elapsed_seconds": time.time() - started,
    }
    payload["report_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _atomic_json(report_file, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--target-gb", type=float, default=30.0)
    args = parser.parse_args()
    report = audit_foundation_records(
        args.corpus,
        report_path=args.report,
        index_path=args.index,
        target_bytes=int(args.target_gb * 1024**3),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["structurally_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
