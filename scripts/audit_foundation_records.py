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
AUDIT_SCHEMA_VERSION = 2
COMMIT_EVERY_RECORDS = 10_000
FINGERPRINT_WINDOW_BYTES = 1024 * 1024


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


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in connection.execute("SELECT key, value FROM metadata")
    }


def _set_metadata(connection: sqlite3.Connection, values: dict[str, object]) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        [(key, str(value)) for key, value in values.items()],
    )


def _corpus_fingerprint(path: Path) -> dict[str, object]:
    """Return a cheap identity strong enough to bind an interrupted scan."""
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        prefix = stream.read(FINGERPRINT_WINDOW_BYTES)
        digest.update(prefix)
        if stat.st_size > FINGERPRINT_WINDOW_BYTES:
            stream.seek(max(0, stat.st_size - FINGERPRINT_WINDOW_BYTES))
            digest.update(stream.read(FINGERPRINT_WINDOW_BYTES))
    return {
        "corpus_path": str(path.resolve()),
        "corpus_size_bytes": stat.st_size,
        "corpus_mtime_ns": stat.st_mtime_ns,
        "corpus_edge_sha256": digest.hexdigest(),
    }


def _remove_sqlite_family(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _remove_sqlite_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _checkpoint_progress(
    connection: sqlite3.Connection,
    *,
    fingerprint: dict[str, object],
    scanned_bytes: int,
    failures: dict[str, int],
    progress_file: Path,
    started: float,
) -> None:
    _set_metadata(
        connection,
        {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            **fingerprint,
            "scan_offset_bytes": scanned_bytes,
            "failure_counts_json": json.dumps(failures, sort_keys=True),
        },
    )
    connection.commit()
    valid_records = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    _atomic_json(
        progress_file,
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": "scanning",
            "corpus_path": fingerprint["corpus_path"],
            "corpus_size_bytes": fingerprint["corpus_size_bytes"],
            "scanned_bytes": scanned_bytes,
            "completion": scanned_bytes / max(1, int(fingerprint["corpus_size_bytes"])),
            "valid_records": valid_records,
            "failures": failures,
            "elapsed_seconds_this_process": time.time() - started,
            "updated_at": time.time(),
        },
    )


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
    progress_path: str | Path | None = None,
    restart: bool = False,
) -> dict[str, object]:
    corpus = Path(corpus_path)
    report_file = Path(report_path)
    index_file = Path(index_path)
    progress_file = (
        Path(progress_path)
        if progress_path is not None
        else report_file.with_suffix(report_file.suffix + ".progress.json")
    )
    if not corpus.is_file():
        raise FileNotFoundError(corpus)

    fingerprint = _corpus_fingerprint(corpus)
    # A partial database is reusable only when its complete corpus identity and
    # committed byte boundary match. Otherwise stale rows are discarded.
    temporary_index = index_file.with_suffix(index_file.suffix + ".tmp")
    resumed = False
    if restart:
        _remove_sqlite_family(temporary_index)
    elif temporary_index.exists():
        probe = _open_index(temporary_index)
        metadata = _metadata(probe)
        expected = {
            "audit_schema_version": str(AUDIT_SCHEMA_VERSION),
            **{key: str(value) for key, value in fingerprint.items()},
        }
        resumed = all(metadata.get(key) == value for key, value in expected.items())
        probe.close()
        if not resumed:
            _remove_sqlite_family(temporary_index)
    connection = _open_index(temporary_index)
    metadata = _metadata(connection)
    scanned_bytes = int(metadata.get("scan_offset_bytes", "0")) if resumed else 0
    failures: dict[str, int] = {
        "invalid_json": 0,
        "invalid_utf8": 0,
        "missing_fields": 0,
        "hash_mismatches": 0,
        "duplicate_records": 0,
        "disallowed_licenses": 0,
        "quality_contract_failures": 0,
    }
    if resumed:
        saved_failures = json.loads(metadata.get("failure_counts_json", "{}"))
        failures.update({key: int(value) for key, value in saved_failures.items()})
    minhash_records = int(
        connection.execute("SELECT COUNT(*) FROM minhash_signatures").fetchone()[0]
    )
    records_since_commit = 0
    started = time.time()

    def save_progress() -> None:
        try:
            _checkpoint_progress(
                connection,
                fingerprint=fingerprint,
                scanned_bytes=scanned_bytes,
                failures=failures,
                progress_file=progress_file,
                started=started,
            )
        except BaseException:
            # A caught interruption must not retain a Windows file lock. A hard
            # process kill is handled by SQLite's WAL recovery on the next run.
            connection.close()
            raise

    with corpus.open("rb") as stream:
        trailing_newline = _last_byte(stream) == b"\n"
        stream.seek(scanned_bytes)
        for raw_line in stream:
            scanned_bytes += len(raw_line)
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                failures["invalid_utf8"] += 1
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                failures["invalid_json"] += 1
                continue
            if not isinstance(item, dict):
                failures["invalid_json"] += 1
                continue
            text = str(item.get("text", ""))
            source = str(item.get("source", "")).strip()
            license_name = str(item.get("license", "")).strip()
            declared_hash = str(item.get("document_sha256", "")).strip().lower()
            revision = str(item.get("source_revision", "")).strip()
            if not text or not source or not license_name or not declared_hash or not revision:
                failures["missing_fields"] += 1
                continue
            computed_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if computed_hash != declared_hash:
                failures["hash_mismatches"] += 1
                continue
            normalized_license = license_name.lower().replace("_", "-")
            if not any(marker in normalized_license for marker in ALLOWED_LICENSE_MARKERS):
                failures["disallowed_licenses"] += 1
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
                failures["quality_contract_failures"] += 1
                continue
            try:
                connection.execute(
                    "INSERT INTO documents(document_sha256, source, line_bytes) VALUES (?, ?, ?)",
                    (computed_hash, source, len(raw_line)),
                )
            except sqlite3.IntegrityError:
                failures["duplicate_records"] += 1
                continue
            if minhash_records < 500_000:
                signature = MinHashDeduplicator.signature(text)
                connection.execute(
                    "INSERT INTO minhash_signatures(document_sha256, signature) VALUES (?, ?)",
                    (computed_hash, json.dumps(signature, separators=(",", ":"))),
                )
                minhash_records += 1
            records_since_commit += 1
            if records_since_commit >= COMMIT_EVERY_RECORDS:
                save_progress()
                records_since_commit = 0

    failures["missing_trailing_newline"] = int(not trailing_newline)
    save_progress()
    source_stats = {
        str(source): {"documents": int(documents), "bytes": int(source_bytes)}
        for source, documents, source_bytes in connection.execute(
            "SELECT source, COUNT(*), SUM(line_bytes) FROM documents GROUP BY source"
        )
    }
    valid_records = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    # Fold the WAL into the main database before atomically publishing it.
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.close()
    _remove_sqlite_sidecars(temporary_index)
    _remove_sqlite_family(index_file)
    temporary_index.replace(index_file)

    structurally_valid = not any(failures.values())
    payload: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
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
        "resumed_partial_audit": resumed,
        "index_path": str(index_file.resolve()),
        "elapsed_seconds": time.time() - started,
    }
    payload["report_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _atomic_json(report_file, payload)
    if progress_file.exists():
        progress_file.unlink()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--target-gb", type=float, default=30.0)
    parser.add_argument("--progress", default=None)
    parser.add_argument(
        "--restart", action="store_true", help="Discard a matching partial audit index."
    )
    args = parser.parse_args()
    report = audit_foundation_records(
        args.corpus,
        report_path=args.report,
        index_path=args.index,
        target_bytes=int(args.target_gb * 1024**3),
        progress_path=args.progress,
        restart=args.restart,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["structurally_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
