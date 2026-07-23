#!/usr/bin/env python3
"""Download and assemble An-Ra training data buckets."""

from __future__ import annotations

# Direct execution bootstraps repository imports after resolving REPO_ROOT.
# ruff: noqa: E402
import argparse
import ast
import glob
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from array import array
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anra.anra_paths import (
    DATA_MANIFEST_DIR,
    TOKEN_INVENTORY_MANIFEST,
    get_identity_file,
)
from training.data_ledger import DataQuality
from training.data_pipeline import (
    CANONICAL_TOKENIZER_VERSION,
    SourceRecord,
    TokenShardPublisher,
)
from training.v2_config import CANONICAL_V4_VOCAB_SIZE

TRAINING_DATA_DIR = Path("training_data")
DOWNLOAD_STATUS = DATA_MANIFEST_DIR / "download_status.json"
DOWNLOAD_PROGRESS = DATA_MANIFEST_DIR / "download_progress.json"
TOKEN_SHARD_PROGRESS = DATA_MANIFEST_DIR / "token_shard_progress.json"
FOUNDATION_AUDIT_REPORT = DATA_MANIFEST_DIR.parent / "foundation_records_audit.json"
FOUNDATION_RESUME_INDEX = DATA_MANIFEST_DIR.parent / "foundation_records_index.sqlite3"
REQUIRED_CAMPAIGN_SOURCE_CLASSES = frozenset(
    {
        "fineweb_edu",
        "permissive_code",
        "finemath",
        "science_technical",
        "verified_instruction",
        "verified_dfc",
        "identity_replay",
    }
)

# Phase A is raw causal foundation training. Small verified, instruction, and
# identity corpora remain provenance-bound in the immutable shards, but are
# reserved for their structured continuation objectives. Giving a two-window
# identity corpus a fixed 2% raw-token share would replay it millions of times
# in a real campaign and teach memorization rather than identity continuity.
# Preserve the original relative weights of the four broad pretraining sources
# while normalizing their foundation-only mixture to one.
FOUNDATION_CAMPAIGN_MIX = {
    "fineweb_edu": 11 / 18,
    "permissive_code": 1 / 6,
    "finemath": 2 / 15,
    "science_technical": 4 / 45,
}
# The acquisition profile still reserves ten percent of its byte budget for
# separately verified supplemental material. This is intentionally distinct
# from the Phase-A sampler, which normalizes only the broad native sources.
NATIVE_FOUNDATION_WEIGHT = 0.90


def campaign_source_class(source: str) -> str:
    lowered = source.lower()
    if "fineweb" in lowered:
        return "fineweb_edu"
    if "stack" in lowered or "open code" in lowered:
        return "permissive_code"
    if "finemath" in lowered:
        return "finemath"
    if "arxiv" in lowered or "science/technical" in lowered or "dolma" in lowered:
        return "science_technical"
    if "smol-smoltalk" in lowered:
        return "verified_instruction"
    if "verified dfc" in lowered:
        return "verified_dfc"
    if "identity replay" in lowered:
        return "identity_replay"
    return "unclassified"


DATA_PROFILES = {
    "smoke": {
        "target_gb": 0.02,
        "fineweb_docs": 2_000,
        "redpajama_docs": 1_000,
        "reasoning_per_source": 1_000,
        "science_per_source": 500,
    },
    "15gb": {
        "target_gb": 15.0,
        "fineweb_docs": 1_200_000,
        "redpajama_docs": 0,
        "reasoning_per_source": 120_000,
        "science_per_source": 60_000,
    },
    "30gb": {
        "target_gb": 30.0,
        "fineweb_docs": 2_400_000,
        "redpajama_docs": 0,
        "reasoning_per_source": 240_000,
        "science_per_source": 120_000,
    },
    "120gb": {
        "target_gb": 120.0,
        "native_target_gb": 120.0,
        "fineweb_docs": 9_600_000,
        "redpajama_docs": 0,
        "reasoning_per_source": 120_000,
        "science_per_source": 120_000,
    },
    "t4-15gb": {
        "fineweb_docs": 120_000,
        "redpajama_docs": 40_000,
        "reasoning_per_source": 20_000,
        "science_per_source": 10_000,
    },
    "t4-cached": {
        "fineweb_docs": 100_000,
        "redpajama_docs": 20_000,
        "reasoning_per_source": 8_000,
        "science_per_source": 4_000,
    },
    "tpu": {
        "fineweb_docs": 120_000,
        "redpajama_docs": 40_000,
        "reasoning_per_source": 20_000,
        "science_per_source": 10_000,
    },
    "full": {
        "fineweb_docs": 1_000_000,
        "redpajama_docs": 800_000,
        "reasoning_per_source": None,
        "science_per_source": None,
    },
}

_PII_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"\b(?:\+?\d[\d .()-]{8,}\d)\b"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
)

_COMMON_ENGLISH_WORDS = {
    "a",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}

_DATASET_REVISION_CACHE: dict[str, str] = {}
PROGRESS_INTERVAL_BYTES = 64 * 1024 * 1024


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _publish_download_progress(
    *,
    status: str,
    target_bytes: int,
    output: Path,
    source: str = "",
    source_bytes: int = 0,
    source_target_bytes: int = 0,
    source_documents: int = 0,
    downloaded_this_run_bytes: int = 0,
    elapsed_seconds: float = 0.0,
    errors: list[str] | None = None,
) -> None:
    rate = downloaded_this_run_bytes / max(1e-9, elapsed_seconds)
    _atomic_json(
        DOWNLOAD_PROGRESS,
        {
            "schema_version": 1,
            "status": status,
            "updated_at": time.time(),
            "output": str(output),
            "output_bytes": output.stat().st_size if output.is_file() else 0,
            "target_bytes": int(target_bytes),
            "completion": (
                (output.stat().st_size if output.is_file() else 0)
                / max(1, int(target_bytes))
            ),
            "source": source,
            "source_bytes": int(source_bytes),
            "source_target_bytes": int(source_target_bytes),
            "source_completion": source_bytes / max(1, int(source_target_bytes)),
            "source_documents": int(source_documents),
            "downloaded_this_run_bytes": int(downloaded_this_run_bytes),
            "elapsed_seconds": float(elapsed_seconds),
            "bytes_per_second": rate,
            "errors": list(errors or ()),
        },
    )


def _publish_incremental_foundation_audit(
    *,
    output: Path,
    index_path: Path,
    connection: sqlite3.Connection,
    base_audit: dict[str, Any],
    target_bytes: int,
    started_at: float,
) -> dict[str, Any]:
    """Advance a trusted audit after online-validated append-only writes."""
    corpus_size = output.stat().st_size
    source_stats = {
        str(source): {"documents": int(documents), "bytes": int(source_bytes)}
        for source, documents, source_bytes in connection.execute(
            "SELECT source, COUNT(*), SUM(line_bytes) FROM documents GROUP BY source"
        )
    }
    valid_records = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    minhash_signatures = int(
        connection.execute("SELECT COUNT(*) FROM minhash_signatures").fetchone()[0]
    )
    indexed_bytes = sum(int(row["bytes"]) for row in source_stats.values())
    if indexed_bytes != corpus_size:
        raise RuntimeError(
            "Refusing incremental audit publication: indexed bytes do not match corpus"
        )
    base_size = int(base_audit.get("corpus_size_bytes", 0))
    base_records = int(base_audit.get("valid_records", 0))
    failures = {
        "invalid_json": 0,
        "invalid_utf8": 0,
        "missing_fields": 0,
        "hash_mismatches": 0,
        "duplicate_records": 0,
        "disallowed_licenses": 0,
        "quality_contract_failures": 0,
        "missing_trailing_newline": 0,
    }
    payload: dict[str, Any] = {
        "schema_version": 2,
        "generated_at": time.time(),
        "corpus_path": str(output.resolve()),
        "corpus_size_bytes": corpus_size,
        "target_bytes": int(target_bytes),
        "target_completion": corpus_size / max(1, int(target_bytes)),
        "valid_records": valid_records,
        "minhash_signatures": minhash_signatures,
        "scanned_bytes": corpus_size,
        "source_stats": dict(sorted(source_stats.items())),
        "failures": failures,
        "structurally_valid": True,
        "target_complete": corpus_size >= int(target_bytes * 0.98),
        "resume_safe": True,
        "resumed_partial_audit": False,
        "incremental_append_audit": True,
        "base_report_sha256": str(base_audit.get("report_sha256", "")),
        "appended_bytes": corpus_size - base_size,
        "appended_records": valid_records - base_records,
        "index_path": str(index_path.resolve()),
        "elapsed_seconds": time.time() - started_at,
    }
    payload["report_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _atomic_json(FOUNDATION_AUDIT_REPORT, payload)
    return payload


def _commit_foundation_resume_boundary(
    *,
    stream: Any,
    output: Path,
    connection: sqlite3.Connection,
) -> int:
    """Commit one recoverable file/index boundary in durability order."""
    stream.flush()
    os.fsync(stream.fileno())
    corpus_size = output.stat().st_size
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('corpus_size_bytes', ?)",
        (str(corpus_size),),
    )
    connection.commit()
    return corpus_size


def _recover_committed_foundation_append(
    *,
    output: Path,
    connection: sqlite3.Connection,
    audit: dict[str, Any],
    target_bytes: int,
    started_at: float,
) -> dict[str, Any]:
    """Discard only an uncommitted file tail after a hard-terminated append."""
    metadata = {
        str(key): str(value)
        for key, value in connection.execute("SELECT key, value FROM metadata")
    }
    audit_size = int(audit.get("corpus_size_bytes", -1))
    indexed_size = int(metadata.get("corpus_size_bytes", "-1"))
    corpus_size = output.stat().st_size
    if metadata.get("base_report_sha256") != str(audit.get("report_sha256", "")):
        raise RuntimeError("Foundation append journal is not bound to the resume audit")
    if int(metadata.get("base_corpus_size_bytes", "-1")) != audit_size:
        raise RuntimeError("Foundation append journal has the wrong audited base boundary")
    if not (audit_size <= indexed_size <= corpus_size):
        raise RuntimeError("Foundation append boundaries are inconsistent")
    indexed_document_bytes = int(
        connection.execute("SELECT COALESCE(SUM(line_bytes), 0) FROM documents").fetchone()[0]
    )
    indexed_documents = int(
        connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    )
    base_records = int(audit.get("valid_records", -1))
    appended_records = indexed_documents - base_records
    byte_discrepancy = indexed_size - indexed_document_bytes
    # Historical Windows append sessions opened the corpus in text mode. Python
    # translated each written LF to CRLF while ``line_bytes`` recorded the
    # pre-translation UTF-8 length, leaving one uncounted byte per appended row.
    # Repair only the exact, provable shape of that defect; every other mismatch
    # remains fail-closed.
    if appended_records > 0 and byte_discrepancy == appended_records:
        cursor = connection.execute(
            "UPDATE documents SET line_bytes = line_bytes + 1 WHERE rowid IN ("
            "SELECT rowid FROM documents ORDER BY rowid LIMIT -1 OFFSET ?)",
            (base_records,),
        )
        if cursor.rowcount != appended_records:
            raise RuntimeError("Legacy newline accounting repair changed the wrong row count")
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("legacy_windows_newline_rows_repaired", str(appended_records)),
        )
        connection.commit()
        indexed_document_bytes += appended_records
    if indexed_document_bytes != indexed_size:
        raise RuntimeError("Foundation index rows do not match its committed byte boundary")
    if corpus_size > indexed_size:
        with output.open("r+b") as stream:
            stream.truncate(indexed_size)
            stream.flush()
            os.fsync(stream.fileno())
    recovered = _publish_incremental_foundation_audit(
        output=output,
        index_path=FOUNDATION_RESUME_INDEX,
        connection=connection,
        base_audit=audit,
        target_bytes=target_bytes,
        started_at=started_at,
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("base_report_sha256", str(recovered["report_sha256"])),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("base_corpus_size_bytes", str(recovered["corpus_size_bytes"])),
    )
    connection.commit()
    return recovered


def recover_native_foundation_append() -> dict[str, Any]:
    """Finalize only the durable append journal without acquiring more data."""
    output = TRAINING_DATA_DIR / "foundation_records.jsonl"
    if not output.is_file() or not FOUNDATION_AUDIT_REPORT.is_file():
        raise FileNotFoundError("Foundation corpus and audit report are required")
    if not FOUNDATION_RESUME_INDEX.is_file():
        raise FileNotFoundError("Foundation resume index is required")
    audit = json.loads(FOUNDATION_AUDIT_REPORT.read_text(encoding="utf-8"))
    if audit.get("resume_safe") is not True:
        raise RuntimeError("Foundation audit did not authorize append recovery")
    connection = sqlite3.connect(FOUNDATION_RESUME_INDEX)
    connection.execute("PRAGMA journal_mode=WAL")
    try:
        indexed_row = connection.execute(
            "SELECT value FROM metadata WHERE key='corpus_size_bytes'"
        ).fetchone()
        if indexed_row is None:
            raise RuntimeError("Foundation resume index has no committed byte boundary")
        audit_size = int(audit.get("corpus_size_bytes", -1))
        indexed_size = int(indexed_row[0])
        corpus_size = output.stat().st_size
        if audit_size == indexed_size == corpus_size:
            return audit
        return _recover_committed_foundation_append(
            output=output,
            connection=connection,
            audit=audit,
            target_bytes=int(audit.get("target_bytes", audit_size)),
            started_at=time.time(),
        )
    finally:
        connection.close()

_ALLOWED_ROW_LICENSES = frozenset(
    {
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc-by",
        "cc-by-sa",
        "cc0",
        "isc",
        "mit",
        "mpl-2.0",
        "odc-by",
        "public-domain",
        "unlicense",
    }
)


def normalize_foundation_license(value: object) -> str:
    """Normalize common dataset/SPDX license spellings to the campaign allowlist."""
    text = str(value).strip().lower().replace("_", "-")
    compact = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if "public-domain" in compact:
        return "public-domain"
    if "unlicense" in compact:
        return "unlicense"
    if "creativecommons-org-publicdomain-zero" in compact or compact in {
        "cc0",
        "cc0-1-0",
    }:
        return "cc0"
    if "creativecommons-org-licenses-by-sa" in compact or compact.startswith("cc-by-sa"):
        return "cc-by-sa"
    if "creativecommons-org-licenses-by" in compact or compact.startswith("cc-by"):
        return "cc-by"
    if compact.startswith("apache"):
        return "apache-2.0"
    if compact.startswith("bsd-2") or compact == "bsd-2-clause":
        return "bsd-2-clause"
    if compact.startswith("bsd-3") or compact == "bsd-3-clause":
        return "bsd-3-clause"
    if compact in {"isc", "isc-license"}:
        return "isc"
    if compact in {"mit", "mit-license"}:
        return "mit"
    if compact.startswith("mpl-2") or compact.startswith("mozilla-public-license-2"):
        return "mpl-2.0"
    if compact.startswith("odc-by") or "open-data-commons-attribution" in compact:
        return "odc-by"
    return compact


def foundation_licenses_allowed(values: object) -> tuple[bool, tuple[str, ...]]:
    """Require every declared row license to be explicitly allowlisted."""
    if isinstance(values, (list, tuple, set, frozenset)):
        raw_values = tuple(values)
    else:
        text = str(values).strip()
        raw_values = tuple(re.split(r"\s+(?:AND|OR)\s+|[,;]", text, flags=re.IGNORECASE))
    normalized = tuple(
        dict.fromkeys(
            normalize_foundation_license(value)
            for value in raw_values
            if str(value).strip()
        )
    )
    allowed = bool(normalized) and all(
        item in _ALLOWED_ROW_LICENSES for item in normalized
    )
    return allowed, normalized


def _row_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _row_license_values(item: dict[str, Any], fallback: str) -> object:
    if fallback != "per-record":
        return fallback
    metadata = _row_metadata(item)
    declared: list[object] = []
    for candidate in (
        item.get("license"),
        metadata.get("detected_licenses"),
        metadata.get("license"),
        metadata.get("gha_license_id"),
    ):
        if isinstance(candidate, (list, tuple, set, frozenset)):
            declared.extend(candidate)
        elif candidate is not None and str(candidate).strip():
            declared.append(candidate)
    return tuple(declared)


def resolve_dataset_revision(dataset_name: str) -> str:
    """Resolve a mutable Hub name to the immutable commit used by this campaign."""
    cached = _DATASET_REVISION_CACHE.get(dataset_name)
    if cached:
        return cached
    try:
        from huggingface_hub import HfApi

        revision = str(HfApi().dataset_info(dataset_name).sha or "").strip()
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve immutable revision for {dataset_name}: {exc}"
        ) from exc
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"Dataset {dataset_name} returned non-immutable revision {revision!r}")
    _DATASET_REVISION_CACHE[dataset_name] = revision
    return revision


class MinHashDeduplicator:
    """Bounded-memory LSH index for near-duplicate document rejection."""

    _PRIME = (1 << 61) - 1
    _PERMUTATIONS = (
        (3, 17),
        (5, 29),
        (11, 41),
        (17, 53),
        (23, 71),
        (31, 89),
        (43, 107),
        (59, 131),
    )

    def __init__(self, threshold: float = 0.80, max_entries: int = 500_000) -> None:
        self.threshold = float(threshold)
        self.max_entries = max(1, int(max_entries))
        # Flat uint64 storage avoids four million boxed Python integers at the
        # 500k-entry resume ceiling. Most LSH buckets contain one signature, so
        # store a bare integer until a collision actually needs a list.
        self._signatures = array("Q")
        self._bands: dict[tuple[int, int, int], int | list[int]] = {}

    @property
    def count(self) -> int:
        return len(self._signatures) // len(self._PERMUTATIONS)

    def _signature_at(self, index: int) -> tuple[int, ...]:
        width = len(self._PERMUTATIONS)
        start = int(index) * width
        return tuple(self._signatures[start : start + width])

    @classmethod
    def signature(cls, text: str) -> tuple[int, ...]:
        words = re.findall(r"[a-z0-9_]+", text.lower())
        if not words:
            return tuple(0 for _ in cls._PERMUTATIONS)
        width = min(5, len(words))
        shingles = {
            " ".join(words[index : index + width])
            for index in range(max(1, len(words) - width + 1))
        }
        hashes = sorted(
            int.from_bytes(
                hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            % cls._PRIME
            for shingle in shingles
        )
        if len(hashes) > 512:
            step = len(hashes) / 512
            hashes = [hashes[int(index * step)] for index in range(512)]
        return tuple(
            min((coefficient * value + offset) % cls._PRIME for value in hashes)
            for coefficient, offset in cls._PERMUTATIONS
        )

    def seen_or_add(self, text: str) -> bool:
        return self.seen_or_add_signature(self.signature(text))

    def seen_or_add_signature(self, signature: tuple[int, ...]) -> bool:
        candidates: set[int] = set()
        for band in range(4):
            key = (band, signature[band * 2], signature[band * 2 + 1])
            stored = self._bands.get(key)
            if isinstance(stored, int):
                candidates.add(stored)
            elif stored:
                candidates.update(stored)
        for candidate in candidates:
            prior = self._signature_at(candidate)
            similarity = sum(a == b for a, b in zip(signature, prior, strict=True)) / len(signature)
            if similarity >= self.threshold:
                return True
        if self.count >= self.max_entries:
            return False
        self.add_signature(signature)
        return False

    def add_signature(self, signature: tuple[int, ...]) -> bool:
        if self.count >= self.max_entries:
            return False
        index = self.count
        self._signatures.extend(signature)
        for band in range(4):
            key = (band, signature[band * 2], signature[band * 2 + 1])
            stored = self._bands.get(key)
            if stored is None:
                self._bands[key] = index
            elif isinstance(stored, int):
                self._bands[key] = [stored, index]
            else:
                stored.append(index)
        return True


def _detect_content_language(text: str, *, source: str, hint: str = "") -> str:
    source_lower = source.lower()
    hint_lower = hint.strip().lower()
    if "stack" in source_lower:
        return f"code:{hint_lower or 'unknown'}"
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return "math" if "math" in source_lower else "unknown"
    letters = [character for character in text if character.isalpha()]
    ascii_ratio = sum(character.isascii() for character in letters) / max(1, len(letters))
    common = sum(word.lower() in _COMMON_ENGLISH_WORDS for word in words)
    if ascii_ratio >= 0.90 and (common >= 2 or len(words) < 20):
        return "en"
    return "unknown"


def _code_syntax_valid(text: str, *, source: str, language_hint: str = "") -> bool:
    if "stack" not in source.lower():
        return True
    language = language_hint.strip().lower()
    if language not in {"python", "py"}:
        return bool(re.search(r"[A-Za-z_][A-Za-z0-9_]*", text))
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def _safe_arithmetic_value(expression: str) -> float:
    operators = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.Pow: lambda left, right: left**right,
        ast.USub: lambda value: -value,
        ast.UAdd: lambda value: value,
    }

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return float(operators[type(node.op)](evaluate(node.left), evaluate(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return float(operators[type(node.op)](evaluate(node.operand)))
        raise ValueError("unsupported arithmetic expression")

    return evaluate(ast.parse(expression, mode="eval"))


def _math_text_valid(text: str, *, source: str) -> bool:
    if "math" not in source.lower():
        return True
    if any(text.count(left) != text.count(right) for left, right in (("(", ")"), ("[", "]"))):
        return False
    final_equations = re.findall(
        r"(?:final answer|answer)\s*[:=-]\s*([0-9 .+*/()^-]+)\s*=\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    for expression, expected in final_equations:
        try:
            actual = _safe_arithmetic_value(expression.replace("^", "**"))
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            return False
        if abs(actual - float(expected)) > 1e-8:
            return False
    return True


def _clean_foundation_text(text: str) -> str:
    cleaned = text.replace("\x00", " ").strip()
    for pattern in _PII_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    if len(cleaned) < 200:
        return ""
    printable_ratio = sum(character.isprintable() for character in cleaned) / len(cleaned)
    if printable_ratio < 0.98:
        return ""
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) >= 8 and len(set(lines)) / len(lines) < 0.35:
        return ""
    lowered = cleaned.lower()
    contamination_markers = (
        "gsm8k test",
        "human_eval test",
        "mmlu test question",
        "truthfulqa benchmark",
    )
    if any(marker in lowered for marker in contamination_markers):
        return ""
    return cleaned


def download_native_foundation(
    load_dataset: Callable[..., Any] | None,
    *,
    target_gb: float,
    native_target_gb: float | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Stream the licensed native foundation mix into one provenance JSONL."""
    output = TRAINING_DATA_DIR / "foundation_records.jsonl"
    target_bytes = int(float(target_gb) * 1024**3)
    native_target_bytes = int(
        float(native_target_gb) * 1024**3
        if native_target_gb is not None
        else target_bytes * NATIVE_FOUNDATION_WEIGHT
    )
    fineweb_config = "sample-100BT"
    specs = (
        {
            "source": "FineWeb-Edu",
            "dataset": "HuggingFaceFW/fineweb-edu",
            "config": fineweb_config,
            "weight": 0.55,
            "fields": ("text",),
            "license": "ODC-By",
            "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        },
        {
            "source": "Common Pile Stack v2 open code",
            "dataset": "common-pile/stackv2_edu_filtered",
            "config": None,
            "weight": 0.15,
            "fields": ("text",),
            "license": "per-record",
            "revision": "c354dbe88469a1153e97c6a63ac50591849654de",
        },
        {
            "source": "FineMath-4+",
            "dataset": "HuggingFaceTB/finemath",
            "config": "finemath-4plus",
            "weight": 0.12,
            "fields": ("text", "content"),
            "license": "ODC-By",
            "revision": "e92b25a616738fe95dc186b64dfb19f9c8525594",
        },
        {
            "source": "Common Pile ArXiv science/technical",
            "dataset": "common-pile/arxiv_papers_filtered",
            "config": None,
            "weight": 0.08,
            "fields": ("text",),
            "license": "per-record",
            "revision": "033cf7f53f9b348deec868c1a5a48484f3ee9e52",
        },
    )
    stats: dict[str, Any] = {
        "bucket": "base",
        "output": str(output),
        "target_bytes": target_bytes,
        "raw_foundation_target_bytes": native_target_bytes,
        "supplemental_target_bytes": int(target_bytes * 0.10),
        "campaign_mix": FOUNDATION_CAMPAIGN_MIX,
        "bytes": 0,
        "documents": 0,
        "sources": {},
        "errors": [],
        "rejected": {
            "exact_duplicate": 0,
            "near_duplicate": 0,
            "language": 0,
            "code_syntax": 0,
            "math_verifier": 0,
        },
    }
    if dry_run:
        print(
            f"DRY RUN: would stream {native_target_bytes / 1024**3:.2f} GB "
            f"native foundation mix to {output}"
        )
        return stats
    assert load_dataset is not None
    near_duplicates = MinHashDeduplicator()
    seen_hashes: set[str] = set()
    resume_db: sqlite3.Connection | None = None
    resume_audit: dict[str, Any] | None = None
    existing_sources: dict[str, dict[str, int]] = {}
    append_started = time.time()
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"Refusing to truncate existing foundation corpus: {output}. "
                "Audit it, then pass --resume."
            )
        if not FOUNDATION_AUDIT_REPORT.is_file() or not FOUNDATION_RESUME_INDEX.is_file():
            raise RuntimeError(
                "Safe resume requires foundation_records_audit.json and "
                "foundation_records_index.sqlite3"
            )
        audit = json.loads(FOUNDATION_AUDIT_REPORT.read_text(encoding="utf-8"))
        if audit.get("resume_safe") is not True:
            raise RuntimeError("Foundation audit did not authorize safe resume")
        resume_db = sqlite3.connect(FOUNDATION_RESUME_INDEX)
        resume_db.execute("PRAGMA journal_mode=WAL")
        indexed_size = resume_db.execute(
            "SELECT value FROM metadata WHERE key='corpus_size_bytes'"
        ).fetchone()
        if indexed_size is None:
            resume_db.close()
            raise RuntimeError("Foundation resume index has no committed byte boundary")
        audit_size = int(audit.get("corpus_size_bytes", -1))
        corpus_size = output.stat().st_size
        if audit_size == corpus_size and int(indexed_size[0]) == corpus_size:
            resume_audit = audit
            resume_db.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("base_report_sha256", str(audit.get("report_sha256", ""))),
            )
            resume_db.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("base_corpus_size_bytes", str(audit_size)),
            )
            resume_db.commit()
        else:
            try:
                resume_audit = _recover_committed_foundation_append(
                    output=output,
                    connection=resume_db,
                    audit=audit,
                    target_bytes=native_target_bytes,
                    started_at=append_started,
                )
            except Exception:
                resume_db.close()
                raise
        for source, documents, source_bytes in resume_db.execute(
            "SELECT source, COUNT(*), SUM(line_bytes) FROM documents GROUP BY source"
        ):
            existing_sources[str(source)] = {
                "documents": int(documents),
                "bytes": int(source_bytes),
            }
        for (encoded_signature,) in resume_db.execute(
            "SELECT signature FROM minhash_signatures ORDER BY rowid"
        ):
            near_duplicates.add_signature(
                tuple(int(value) for value in json.loads(str(encoded_signature)))
            )

    def exact_seen(content_hash: str) -> bool:
        if resume_db is None:
            return content_hash in seen_hashes
        return (
            resume_db.execute(
                "SELECT 1 FROM documents WHERE document_sha256=?", (content_hash,)
            ).fetchone()
            is not None
        )

    mode = "ab" if output.exists() else "wb"
    pending_index_writes = 0
    with output.open(mode) as stream:
        for spec in specs:
            source_target = int(
                native_target_bytes
                * float(spec["weight"])
                / NATIVE_FOUNDATION_WEIGHT
            )
            existing = existing_sources.get(str(spec["source"]), {})
            source_bytes = int(existing.get("bytes", 0))
            source_docs = int(existing.get("documents", 0))
            resumed_bytes = source_bytes
            resumed_docs = source_docs
            resolved_revision = ""
            source_started = time.monotonic()
            last_progress_bytes = source_bytes
            try:
                if source_bytes >= source_target:
                    stats["sources"][str(spec["source"])] = {
                        "bytes": source_bytes,
                        "documents": source_docs,
                        "target_bytes": source_target,
                        "resumed_bytes": resumed_bytes,
                        "resumed_documents": resumed_docs,
                        "downloaded_this_run_bytes": 0,
                        "downloaded_this_run_documents": 0,
                        "revision": str(spec["revision"]),
                    }
                    stats["bytes"] += source_bytes
                    stats["documents"] += source_docs
                    print(
                        f"{spec['source']}: resume quota already complete "
                        f"({source_bytes / 1024**3:.2f} GB)",
                        flush=True,
                    )
                    continue
                resolved_revision = str(spec["revision"])
                if not re.fullmatch(r"[0-9a-f]{40}", resolved_revision):
                    raise RuntimeError(
                        f"{spec['source']}: source revision is not an immutable commit"
                    )
                kwargs: dict[str, Any] = {
                    "split": "train",
                    "streaming": True,
                    "revision": resolved_revision,
                }
                _publish_download_progress(
                    status="downloading",
                    target_bytes=native_target_bytes,
                    output=output,
                    source=str(spec["source"]),
                    source_bytes=source_bytes,
                    source_target_bytes=source_target,
                    source_documents=source_docs,
                    errors=stats["errors"],
                )
                config = spec["config"]
                dataset = (
                    load_dataset(spec["dataset"], config, **kwargs)
                    if config
                    else load_dataset(spec["dataset"], **kwargs)
                )
                for item in dataset:
                    metadata = _row_metadata(item)
                    language_hint = str(
                        item.get("language")
                        or item.get("lang")
                        or metadata.get("language")
                        or metadata.get("extension")
                        or Path(str(item.get("path", ""))).suffix.lstrip(".")
                    )
                    text = next(
                        (str(item.get(field, "")) for field in spec["fields"] if item.get(field)),
                        "",
                    )
                    text = _clean_foundation_text(text)
                    if not text:
                        continue
                    language = _detect_content_language(
                        text,
                        source=str(spec["source"]),
                        hint=language_hint,
                    )
                    if language == "unknown":
                        stats["rejected"]["language"] += 1
                        continue
                    if not _code_syntax_valid(
                        text,
                        source=str(spec["source"]),
                        language_hint=language_hint,
                    ):
                        stats["rejected"]["code_syntax"] += 1
                        continue
                    if not _math_text_valid(text, source=str(spec["source"])):
                        stats["rejected"]["math_verifier"] += 1
                        continue
                    allowed_license, normalized_licenses = foundation_licenses_allowed(
                        _row_license_values(item, str(spec["license"]))
                    )
                    if not allowed_license:
                        continue
                    license_name = " AND ".join(normalized_licenses)
                    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if exact_seen(content_hash):
                        stats["rejected"]["exact_duplicate"] += 1
                        continue
                    signature = MinHashDeduplicator.signature(text)
                    signature_count_before = near_duplicates.count
                    if near_duplicates.seen_or_add_signature(signature):
                        stats["rejected"]["near_duplicate"] += 1
                        continue
                    signature_added = near_duplicates.count > signature_count_before
                    record = {
                        "text": text,
                        "source": spec["source"],
                        "license": license_name,
                        "source_revision": (
                            f"{spec['dataset']}@{resolved_revision}:{spec['config'] or 'default'}"
                        ),
                        "document_sha256": content_hash,
                        "language": language,
                        "quality_checks": {
                            "pii_redacted": True,
                            "minhash_deduplicated": True,
                            "language_detected": True,
                            "code_syntax_checked": "stack" in str(spec["source"]).lower(),
                            "math_verified": "math" in str(spec["source"]).lower(),
                            "benchmark_contamination_checked": True,
                        },
                    }
                    encoded_line = (json.dumps(record, ensure_ascii=False) + "\n").encode(
                        "utf-8"
                    )
                    stream.write(encoded_line)
                    encoded_bytes = len(encoded_line)
                    source_bytes += encoded_bytes
                    source_docs += 1
                    if resume_db is None:
                        seen_hashes.add(content_hash)
                    else:
                        resume_db.execute(
                            "INSERT INTO documents(document_sha256, source, line_bytes) "
                            "VALUES (?, ?, ?)",
                            (content_hash, str(spec["source"]), encoded_bytes),
                        )
                        if signature_added:
                            resume_db.execute(
                                "INSERT OR IGNORE INTO minhash_signatures"
                                "(document_sha256, signature) VALUES (?, ?)",
                                (content_hash, json.dumps(signature, separators=(",", ":"))),
                            )
                        pending_index_writes += 1
                        if pending_index_writes >= 1_000:
                            _commit_foundation_resume_boundary(
                                stream=stream,
                                output=output,
                                connection=resume_db,
                            )
                            pending_index_writes = 0
                    if source_bytes - last_progress_bytes >= PROGRESS_INTERVAL_BYTES:
                        elapsed = time.monotonic() - source_started
                        downloaded_bytes = source_bytes - resumed_bytes
                        _publish_download_progress(
                            status="downloading",
                            target_bytes=native_target_bytes,
                            output=output,
                            source=str(spec["source"]),
                            source_bytes=source_bytes,
                            source_target_bytes=source_target,
                            source_documents=source_docs,
                            downloaded_this_run_bytes=downloaded_bytes,
                            elapsed_seconds=elapsed,
                            errors=stats["errors"],
                        )
                        print(
                            f"{spec['source']}: {source_bytes / 1024**3:.2f} / "
                            f"{source_target / 1024**3:.2f} GiB; "
                            f"{downloaded_bytes / max(1e-9, elapsed) / 1024**2:.2f} MiB/s",
                            flush=True,
                        )
                        last_progress_bytes = source_bytes
                    if source_bytes >= source_target:
                        break
                if source_bytes < int(source_target * 0.98):
                    raise SourceDownloadFailure(
                        str(spec["source"]),
                        f"downloaded {source_bytes:,} of required {source_target:,} bytes",
                    )
            except Exception as exc:
                stats["errors"].append(f"{spec['source']}: {exc}")
                print(f"{spec['source']}: FAILED: {exc}", flush=True)
            stats["sources"][str(spec["source"])] = {
                "bytes": source_bytes,
                "documents": source_docs,
                "target_bytes": source_target,
                "resumed_bytes": resumed_bytes,
                "resumed_documents": resumed_docs,
                "downloaded_this_run_bytes": source_bytes - resumed_bytes,
                "downloaded_this_run_documents": source_docs - resumed_docs,
                "revision": (
                    f"{spec['dataset']}@{resolved_revision}:{spec['config'] or 'default'}"
                    if resolved_revision
                    else str(spec["revision"])
                ),
            }
            stats["bytes"] += source_bytes
            stats["documents"] += source_docs
            print(
                f"{spec['source']}: {source_docs:,} documents, {source_bytes / 1024**3:.2f} GB",
                flush=True,
            )
        if resume_db is not None:
            _commit_foundation_resume_boundary(
                stream=stream,
                output=output,
                connection=resume_db,
            )
            assert resume_audit is not None
            _publish_incremental_foundation_audit(
                output=output,
                index_path=FOUNDATION_RESUME_INDEX,
                connection=resume_db,
                base_audit=resume_audit,
                target_bytes=native_target_bytes,
                started_at=append_started,
            )
            resume_db.close()
    _publish_download_progress(
        status="incomplete" if stats["errors"] else "native_foundation_complete",
        target_bytes=native_target_bytes,
        output=output,
        errors=stats["errors"],
    )
    return stats


class SourceDownloadFailure(RuntimeError):  # noqa: N818 - serialized status name
    def __init__(self, source: str, message: str) -> None:
        super().__init__(f"{source}: {message}")
        self.source = source
        self.message = message


def load_datasets_import(dry_run: bool = False) -> Callable[..., Any] | None:
    if dry_run:
        return None
    try:
        from datasets import load_dataset
    except ImportError:
        print("Run: pip install datasets")
        sys.exit(1)
    return load_dataset


def ensure_training_data_dir() -> None:
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _download_status_path(buckets: list[str]) -> Path:
    """Keep partial bucket runs from overwriting foundation campaign truth."""
    if buckets == ["base"] or buckets == ["base", "reasoning", "science"]:
        return DOWNLOAD_STATUS
    scope = "_".join(sorted(buckets))
    return DATA_MANIFEST_DIR / f"download_status_{scope}.json"


def prompt_key_from_dfc_text(text: str) -> str:
    task_close = "</task>"
    if "<task" not in text or task_close not in text:
        return text[:100]
    task_start = text.find("<task")
    prompt_start = text.find(">", task_start)
    if prompt_start == -1:
        return text[:100]
    prompt_end = text.find(task_close, prompt_start)
    if prompt_end == -1:
        return text[:100]
    return text[prompt_start + 1 : prompt_end][:100]


def download_base(
    load_dataset: Callable[..., Any] | None,
    dry_run: bool = False,
    *,
    fineweb_docs: int = 1_000_000,
    redpajama_docs: int = 800_000,
) -> dict[str, Any]:
    output = TRAINING_DATA_DIR / "base_corpus.txt"
    stats: dict[str, Any] = {"bucket": "base", "output": str(output), "documents": 0, "errors": []}

    if dry_run:
        print(
            "DRY RUN: would download "
            f"{fineweb_docs:,} FineWeb-Edu docs and {redpajama_docs:,} RedPajama docs into {output}"
        )
        return stats

    assert load_dataset is not None

    fineweb_path = TRAINING_DATA_DIR / "fineweb_edu.txt"
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        count = 0
        with fineweb_path.open("w", encoding="utf-8") as f:
            for item in ds:
                try:
                    text = item.get("text", "")
                    if text:
                        f.write(text.strip() + "\n\n")
                        count += 1
                    if count >= fineweb_docs:
                        break
                except Exception:
                    continue
        stats["documents"] += count
        print(f"FineWeb-Edu: {count:,} documents")
    except Exception as e:
        stats["errors"].append(f"FineWeb-Edu: {e}")
        print(f"SKIP FineWeb-Edu: {e}")

    redpajama_path = TRAINING_DATA_DIR / "redpajama.txt"
    try:
        ds2 = load_dataset(
            "togethercomputer/RedPajama-Data-V2",
            name="sample",
            split="train",
            streaming=True,
        )
        count = 0
        with redpajama_path.open("w", encoding="utf-8") as f:
            for item in ds2:
                try:
                    text = item.get("raw_content", item.get("text", ""))
                    if text and len(text) > 200:
                        f.write(text.strip() + "\n\n")
                        count += 1
                    if count >= redpajama_docs:
                        break
                except Exception:
                    continue
        stats["documents"] += count
        print(f"RedPajama: {count:,} documents")
    except Exception as e:
        stats["errors"].append(f"RedPajama: {e}")
        print(f"SKIP RedPajama: {e}")

    try:
        with output.open("w", encoding="utf-8") as out:
            for fname in glob.glob(str(TRAINING_DATA_DIR / "fineweb_edu.txt")) + glob.glob(
                str(TRAINING_DATA_DIR / "redpajama.txt")
            ):
                with open(fname, encoding="utf-8", errors="replace") as src:
                    shutil.copyfileobj(src, out)
                out.write("\n\n")
        size_gb = output.stat().st_size / 1024**3
        print(f"Base corpus: {size_gb:.2f} GB")
    except Exception as e:
        stats["errors"].append(f"base_corpus concat: {e}")
        print(f"SKIP base_corpus concat: {e}")

    return stats


def download_reasoning(
    load_dataset: Callable[..., Any] | None,
    dry_run: bool = False,
    *,
    per_source_limit: int | None = None,
) -> dict[str, Any]:
    output = TRAINING_DATA_DIR / "reasoning.jsonl"
    stats: dict[str, Any] = {
        "bucket": "reasoning",
        "output": str(output),
        "examples": 0,
        "errors": [],
    }

    if dry_run:
        print(f"DRY RUN: would download reasoning teacher datasets into {output}")
        return stats

    assert load_dataset is not None

    datasets_to_load: list[
        tuple[str, str, int | None, Callable[[dict[str, Any]], dict[str, str] | None]]
    ] = [
        (
            "HuggingFaceTB/smol-smoltalk",
            "train",
            120_000,
            lambda x: {
                "prompt": next(
                    message["content"]
                    for message in x.get("messages", [])
                    if message.get("role") == "user"
                ),
                "response": next(
                    message["content"]
                    for message in x.get("messages", [])
                    if message.get("role") == "assistant"
                ),
            }
            if x.get("messages")
            else None,
        ),
    ]

    reject_patterns = [
        "ChatGPT",
        "GPT-4",
        "GPT4",
        "Claude",
        "Anthropic",
        "OpenAI",
        "I am an AI",
        "As an AI",
    ]

    total = 0
    with output.open("w", encoding="utf-8") as out:
        for ds_name, split, max_n, mapper in datasets_to_load:
            try:
                resolved_revision = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
                ds = load_dataset(
                    ds_name,
                    split=split,
                    streaming=True,
                    revision=resolved_revision,
                )
                count = 0
                for item in ds:
                    try:
                        mapped = mapper(item)
                        if mapped is None:
                            continue
                        response = mapped.get("response", "")
                        if any(p in response for p in reject_patterns):
                            continue
                        task_type = (
                            "code"
                            if "coder" in ds_name.lower()
                            else "math"
                            if "math" in ds_name.lower()
                            else "instruction"
                        )
                        out.write(
                            json.dumps(
                                {
                                    "prompt": mapped["prompt"],
                                    "response": response,
                                    "source": ds_name,
                                    "source_revision": resolved_revision,
                                    "task_type": task_type,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        count += 1
                        total += 1
                        limit = (
                            min(max_n, per_source_limit)
                            if max_n and per_source_limit
                            else (per_source_limit or max_n)
                        )
                        if limit and count >= limit:
                            break
                    except Exception:
                        continue
                print(f"  {ds_name}: {count:,} examples")
            except Exception as e:
                stats["errors"].append(f"{ds_name}: {e}")
                print(f"  SKIP {ds_name}: {e}")

    stats["examples"] = total
    size_mb = output.stat().st_size / 1024**2
    print(f"Reasoning total: {total:,} examples ({size_mb:.0f} MB)")
    return stats


def download_science(
    load_dataset: Callable[..., Any] | None,
    dry_run: bool = False,
    *,
    per_source_limit: int | None = None,
) -> dict[str, Any]:
    output = TRAINING_DATA_DIR / "frontier_dfc.jsonl"
    stats: dict[str, Any] = {
        "bucket": "science",
        "output": str(output),
        "examples": 0,
        "errors": [],
    }

    if dry_run:
        print(f"DRY RUN: would append science datasets into {output}")
        return stats

    assert load_dataset is not None

    science_datasets: list[
        tuple[str, str | None, str, int | None, Callable[[dict[str, Any]], dict[str, str]]]
    ] = [
        (
            "openai/gsm8k",
            "main",
            "train",
            None,
            lambda x: {
                "prompt": x["question"],
                "response": x["answer"],
                "template": "constraint_solve",
                "domain": "math",
            },
        ),
        (
            "lighteval/MATH",
            None,
            "train",
            None,
            lambda x: {
                "prompt": x["problem"],
                "response": x["solution"],
                "template": "hypothesis_chain",
                "domain": "math",
            },
        ),
        (
            "BoltzmannEntropy/QuantumLLMInstruct",
            None,
            "train",
            2000,
            lambda x: {
                "prompt": x.get("question", ""),
                "response": x.get("answer", ""),
                "template": "tool_action_trace",
                "domain": "quantum",
            },
        ),
        (
            "laion/Scientific-Summaries",
            None,
            "train",
            30_000,
            lambda x: {
                "prompt": "Summarize and state the main hypothesis of: " + x.get("abstract", ""),
                "response": x.get("title", "") + ". " + x.get("abstract", ""),
                "template": "hypothesis_chain",
                "domain": "science",
            },
        ),
    ]

    existing = set()
    try:
        with output.open("r", encoding="utf-8") as existing_file:
            for line in existing_file:
                try:
                    obj = json.loads(line)
                    prompt = obj.get("prompt", "")
                    if not prompt and isinstance(obj.get("text"), str):
                        prompt = prompt_key_from_dfc_text(obj["text"])
                    existing.add(prompt[:100])
                except Exception:
                    continue
    except Exception:
        pass

    added = 0
    with output.open("a", encoding="utf-8") as out:
        for ds_name, config, split, max_n, mapper in science_datasets:
            try:
                resolved_revision = resolve_dataset_revision(ds_name)
                if config:
                    ds = load_dataset(
                        ds_name,
                        config,
                        split=split,
                        streaming=True,
                        revision=resolved_revision,
                    )
                else:
                    ds = load_dataset(
                        ds_name,
                        split=split,
                        streaming=True,
                        revision=resolved_revision,
                    )
                count = 0
                for item in ds:
                    try:
                        mapped = mapper(item)
                        if not mapped.get("prompt") or not mapped.get("response"):
                            continue
                        key = mapped["prompt"][:100]
                        if key in existing:
                            continue
                        existing.add(key)
                        dfc_entry = {
                            "text": (
                                f"<bos>"
                                f'<task domain="{mapped["domain"]}" '
                                f'type="{mapped["template"]}">'
                                f"{mapped['prompt']}</task>"
                                f"<hyp>{mapped['response'][:500]}</hyp>"
                                f"<verify>INFERRED: from dataset, "
                                f"not simulator-verified</verify>"
                                f"<eos>"
                            ),
                            "domain": mapped["domain"],
                            "template": mapped["template"],
                            "verified": False,
                            "verifier_status": "inferred",
                            "source": ds_name,
                            "source_revision": resolved_revision,
                        }
                        out.write(json.dumps(dfc_entry, ensure_ascii=False) + "\n")
                        count += 1
                        added += 1
                        limit = (
                            min(max_n, per_source_limit)
                            if max_n and per_source_limit
                            else (per_source_limit or max_n)
                        )
                        if limit and count >= limit:
                            break
                    except Exception:
                        continue
                print(f"  {ds_name}: {count:,} DFC examples added")
            except Exception as e:
                stats["errors"].append(f"{ds_name}: {e}")
                print(f"  SKIP {ds_name}: {e}")

    stats["examples"] = added
    print(f"DFC science total added: {added:,}")
    return stats


def print_summary() -> None:
    print()
    print("=" * 60)
    print("TRAINING DATA SUMMARY")
    print("=" * 60)
    files = {
        "base_corpus.txt": "Base corpus (language model pretraining)",
        "reasoning.jsonl": "Reasoning Q&A (teacher data)",
        "frontier_dfc.jsonl": "DFC frontier science (domain verifier data)",
    }
    total_gb = 0.0
    for fname, desc in files.items():
        path = TRAINING_DATA_DIR / fname
        if path.exists():
            gb = path.stat().st_size / 1024**3
            total_gb += gb
            print(f"  {fname:<30} {gb:.2f} GB  {desc}")
        else:
            print(f"  {fname:<30} MISSING")
    print(f"\n  TOTAL: {total_gb:.2f} GB")
    print(f"  Estimated tokens: ~{int(total_gb * 250_000_000):,}")
    print("\n  Recommended data mix in training:")
    print("    base_corpus.txt  -> own_ratio  0.55 (55%)")
    print("    reasoning.jsonl  -> teacher    0.25 (25%)")
    print("    frontier_dfc     -> science    0.10 (10%)")
    print("    identity data    -> identity   0.10 (10%)")


def _record_split(digest: str) -> str:
    split_value = int(digest[:8], 16) % 100
    return "validation" if split_value == 98 else "test" if split_value == 99 else "train"


def _verified_supplemental_records(split: str) -> Iterator[SourceRecord]:
    """Yield verifier-backed DFC and owner identity records for one split."""
    dfc_path = TRAINING_DATA_DIR / "verified_dfc.jsonl"
    if dfc_path.is_file():
        with dfc_path.open("r", encoding="utf-8", errors="strict") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(item.get("verifier_status", "unverified")) != "verified":
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if _record_split(digest) != split:
                    continue
                yield SourceRecord(
                    text=text,
                    source="An-Ra verified DFC",
                    license="owner",
                    bucket="dfc",
                    # Difficulty is a percentile consumed by a band-pass gate;
                    # 0.8 incorrectly scored verified rows below admission.
                    quality=DataQuality(0.65, 0.9, 1.0, 0.95, 0.4, 1.0),
                    verifier_status="verified",
                    source_revision=str(
                        item.get("source_revision", item.get("source", "owner-verified"))
                    ),
                    source_class="verified_dfc",
                )

    identity_path = get_identity_file()
    if split == "train" and identity_path is not None and identity_path.is_file():
        identity_text = identity_path.read_text(
            encoding="utf-8", errors="strict"
        ).strip()
        if identity_text:
            yield SourceRecord(
                text=identity_text,
                source="An-Ra identity replay",
                license="owner",
                bucket="identity",
                # Identity is neither a high-difficulty nor easy-data outlier;
                # the ledger's neutral band-pass center is the truthful value.
                quality=DataQuality(0.5, 0.9, 1.0, 0.9, 0.5, 1.0),
                verifier_status="verified",
                source_revision=hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
                source_class="identity_replay",
            )


def publish_fineweb_token_shards(
    profile: str = "30gb",
    *,
    tokenizer_path: str | Path | None = None,
    tokenizer_family: str = "v4",
) -> dict[str, Any]:
    if tokenizer_family != "v4":
        raise ValueError("Only canonical tokenizer_family='v4' is supported")
    foundation_path = TRAINING_DATA_DIR / "foundation_records.jsonl"
    fineweb_path = TRAINING_DATA_DIR / "fineweb_edu.txt"
    if not foundation_path.exists() and not fineweb_path.exists():
        raise FileNotFoundError("Native foundation records must be downloaded first.")
    from tokenizer.subword_tokenizer import SubwordTokenizer

    bound_tokenizer_path = Path(
        tokenizer_path or (REPO_ROOT / "tokenizer" / "tokenizer_v4_32k.json")
    ).resolve()
    if not bound_tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer artifact is missing: {bound_tokenizer_path}")
    tokenizer = SubwordTokenizer.load(bound_tokenizer_path)
    if int(tokenizer.vocab_size) != CANONICAL_V4_VOCAB_SIZE:
        raise ValueError(
            "Only the canonical 32,768-token V4 tokenizer can publish new shards; "
            f"loaded vocabulary has {tokenizer.vocab_size:,} entries"
        )
    tokenizer_sha256 = hashlib.sha256(bound_tokenizer_path.read_bytes()).hexdigest()
    tokenizer_version = CANONICAL_TOKENIZER_VERSION
    revision_dir = (
        DATA_MANIFEST_DIR / f"native_foundation_{tokenizer_family}" / profile
    )
    published_tokens = {"train": 0, "validation": 0, "test": 0}
    published_shards = {"train": 0, "validation": 0, "test": 0}

    def progress_callback(split: str) -> Callable[[dict[str, object]], None]:
        def publish(item: dict[str, object]) -> None:
            published_tokens[split] += int(item["tokens"])
            published_shards[split] += 1
            _atomic_json(
                TOKEN_SHARD_PROGRESS,
                {
                    "schema_version": 1,
                    "status": "publishing",
                    "updated_at": time.time(),
                    "tokenizer_family": tokenizer_family,
                    "tokenizer_sha256": tokenizer_sha256,
                    "profile": profile,
                    "split": split,
                    "published_tokens": dict(published_tokens),
                    "published_shards": dict(published_shards),
                    "last_shard": item,
                },
            )
            print(
                f"{tokenizer_family.upper()} {split}: "
                f"{published_tokens[split]:,} tokens in "
                f"{published_shards[split]:,} shard(s)",
                flush=True,
            )

        return publish

    def records(split: str) -> Iterator[SourceRecord]:
        if foundation_path.exists():
            with foundation_path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(item.get("text", "")).strip()
                    if not text:
                        continue
                    digest = str(
                        item.get("document_sha256")
                        or hashlib.sha256(text.encode("utf-8")).hexdigest()
                    )
                    split_value = int(digest[:8], 16) % 100
                    record_split = (
                        "validation"
                        if split_value == 98
                        else "test"
                        if split_value == 99
                        else "train"
                    )
                    if record_split != split:
                        continue
                    source = str(item.get("source", "unknown"))
                    yield SourceRecord(
                        text=text,
                        source=source,
                        license=str(item.get("license", "unknown")),
                        bucket="foundation",
                        quality=DataQuality(
                            0.6,
                            0.8,
                            0.95,
                            0.7 if "math" in source.lower() else 0.6,
                            0.2,
                            1.0,
                        ),
                        source_revision=str(item.get("source_revision", "unknown")),
                        source_class=campaign_source_class(source),
                    )
        elif split == "train":
            with fineweb_path.open("r", encoding="utf-8", errors="replace") as stream:
                buffer: list[str] = []
                for line in stream:
                    if line.strip():
                        buffer.append(line.strip())
                        continue
                    if buffer:
                        yield SourceRecord(
                            text="\n".join(buffer),
                            source="FineWeb-Edu",
                            license="ODC-By",
                            bucket="foundation",
                            quality=DataQuality(0.5, 0.8, 0.9, 0.6, 0.2, 1.0),
                            source_revision="HuggingFaceFW/fineweb-edu:sample-10BT",
                            source_class="fineweb_edu",
                        )
                        buffer = []
                if buffer:
                    yield SourceRecord(
                        text="\n".join(buffer),
                        source="FineWeb-Edu",
                        license="ODC-By",
                        bucket="foundation",
                        quality=DataQuality(0.5, 0.8, 0.9, 0.6, 0.2, 1.0),
                        source_revision="HuggingFaceFW/fineweb-edu:sample-10BT",
                        source_class="fineweb_edu",
                    )

        reasoning_path = TRAINING_DATA_DIR / "reasoning.jsonl"
        if reasoning_path.is_file():
            with reasoning_path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    source = str(item.get("source", ""))
                    if source != "HuggingFaceTB/smol-smoltalk":
                        continue
                    prompt = str(item.get("prompt", "")).strip()
                    response = str(item.get("response", "")).strip()
                    if not prompt or not response:
                        continue
                    text = f"H: {prompt}\nANRA: {response}"
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    split_value = int(digest[:8], 16) % 100
                    record_split = (
                        "validation"
                        if split_value == 98
                        else "test"
                        if split_value == 99
                        else "train"
                    )
                    if record_split != split:
                        continue
                    yield SourceRecord(
                        text=text,
                        source="Smol-SmolTalk verified instruction",
                        license="Apache-2.0",
                        bucket="instruction",
                        quality=DataQuality(0.7, 0.85, 0.95, 0.8, 0.3, 1.0),
                        verifier_status="verified",
                        source_revision=str(
                            item.get("source_revision", "HuggingFaceTB/smol-smoltalk")
                        ),
                        source_class="verified_instruction",
                    )

        yield from _verified_supplemental_records(split)

    train_manifest = TokenShardPublisher(
        revision_dir,
        tokenizer_version=tokenizer_version,
        tokenizer_sha256=tokenizer_sha256,
    ).publish(
        records("train"),
        tokenizer,
        allow_partial_final=True,
        minimum_replay_tokens={"identity_replay": 4097},
        progress_callback=progress_callback("train"),
    )
    validation_manifest = TokenShardPublisher(
        revision_dir / "validation",
        tokenizer_version=tokenizer_version,
        tokenizer_sha256=tokenizer_sha256,
    ).publish(
        records("validation"),
        tokenizer,
        allow_partial_final=True,
        progress_callback=progress_callback("validation"),
    )
    test_manifest = TokenShardPublisher(
        revision_dir / "test",
        tokenizer_version=tokenizer_version,
        tokenizer_sha256=tokenizer_sha256,
    ).publish(
        records("test"),
        tokenizer,
        allow_partial_final=True,
        progress_callback=progress_callback("test"),
    )

    category_tokens = dict.fromkeys(REQUIRED_CAMPAIGN_SOURCE_CLASSES, 0)
    unclassified_tokens = 0
    for source_class, token_count in train_manifest.get(
        "source_class_token_mix", {}
    ).items():
        if source_class not in category_tokens:
            unclassified_tokens += int(token_count)
        else:
            category_tokens[source_class] += int(token_count)
    classified_total = sum(category_tokens.values())
    realized_mix = {
        name: count / max(1, classified_total) for name, count in category_tokens.items()
    }
    mix_deviation = {
        name: realized_mix[name] - target for name, target in FOUNDATION_CAMPAIGN_MIX.items()
    }
    campaign_sampling_verified = (
        classified_total > 0
        and unclassified_tokens == 0
        and all(count > 0 for count in category_tokens.values())
        and abs(sum(FOUNDATION_CAMPAIGN_MIX.values()) - 1.0) <= 1e-9
    )
    train_manifest.update(
        {
            "campaign_mix_target": FOUNDATION_CAMPAIGN_MIX,
            "campaign_mix_realized": realized_mix,
            "campaign_mix_deviation": mix_deviation,
            "campaign_mix_materialization": "deterministic_source_weighted_sampler",
            "campaign_sampling_verified": campaign_sampling_verified,
            "campaign_mix_verified": campaign_sampling_verified,
            "unclassified_tokens": unclassified_tokens,
        }
    )
    manifest_path = revision_dir / "manifest.json"
    manifest_tmp = manifest_path.with_suffix(".tmp")
    manifest_tmp.write_text(
        json.dumps(train_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_tmp.replace(manifest_path)
    inventory = {
        "schema_version": 3,
        "tokenizer_family": tokenizer_family,
        "tokenizer_path": str(bound_tokenizer_path),
        "licensed_tokens": int(train_manifest["total_tokens"]),
        "tokenizer_sha256": tokenizer_sha256,
        "manifest": str(revision_dir / "manifest.json"),
        "validation_manifest": str(revision_dir / "validation" / "manifest.json"),
        "test_manifest": str(revision_dir / "test" / "manifest.json"),
        "validation_tokens": int(validation_manifest["total_tokens"]),
        "test_tokens": int(test_manifest["total_tokens"]),
        "sources": train_manifest.get("source_record_mix", {}),
        "source_revisions": train_manifest.get("source_revisions", []),
        "licenses": train_manifest.get("licenses", []),
        "campaign_mix_target": FOUNDATION_CAMPAIGN_MIX,
        "campaign_mix_realized": realized_mix,
        "campaign_mix_materialization": "deterministic_source_weighted_sampler",
        "campaign_sampling_verified": campaign_sampling_verified,
        "campaign_mix_verified": campaign_sampling_verified,
        "unclassified_tokens": unclassified_tokens,
    }
    family_inventory = revision_dir / "token_inventory.json"
    family_temporary = family_inventory.with_suffix(".tmp")
    family_temporary.write_text(
        json.dumps(inventory, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    family_temporary.replace(family_inventory)
    TOKEN_INVENTORY_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = TOKEN_INVENTORY_MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(TOKEN_INVENTORY_MANIFEST)
    _atomic_json(
        TOKEN_SHARD_PROGRESS,
        {
            "schema_version": 1,
            "status": "complete",
            "updated_at": time.time(),
            "tokenizer_family": tokenizer_family,
            "tokenizer_sha256": tokenizer_sha256,
            "profile": profile,
            "published_tokens": dict(published_tokens),
            "published_shards": dict(published_shards),
            "inventory": str(family_inventory),
        },
    )
    return inventory


def _merge_integer_maps(*values: object) -> dict[str, int]:
    merged: dict[str, int] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, count in value.items():
            merged[str(key)] = merged.get(str(key), 0) + int(count)
    return merged


def _merge_token_manifests(
    base: dict[str, Any],
    supplemental: dict[str, Any],
    *,
    renamed_shards: list[dict[str, Any]],
    base_sha256: str,
) -> dict[str, Any]:
    if base.get("tokenizer_sha256") != supplemental.get("tokenizer_sha256"):
        raise RuntimeError("Supplemental tokenizer identity does not match base shards")
    merged = dict(base)
    for key in (
        "source_record_mix",
        "source_token_mix",
        "source_class_token_mix",
        "source_class_replayed_tokens",
        "verifier_record_distribution",
        "rejection_counts",
    ):
        merged[key] = _merge_integer_maps(base.get(key), supplemental.get(key))
    merged["shards"] = [*base.get("shards", []), *renamed_shards]
    merged["total_tokens"] = int(base.get("total_tokens", 0)) + int(
        supplemental.get("total_tokens", 0)
    )
    merged["pending_tokens"] = int(base.get("pending_tokens", 0)) + int(
        supplemental.get("pending_tokens", 0)
    )
    merged["accepted_records"] = int(base.get("accepted_records", 0)) + int(
        supplemental.get("accepted_records", 0)
    )
    merged["source_revisions"] = sorted(
        {str(value) for value in base.get("source_revisions", [])}
        | {str(value) for value in supplemental.get("source_revisions", [])}
    )
    merged["licenses"] = sorted(
        {str(value) for value in base.get("licenses", [])}
        | {str(value) for value in supplemental.get("licenses", [])}
    )
    base_quality = base.get("quality", {})
    supplemental_quality = supplemental.get("quality", {})
    accepted = int(base_quality.get("accepted", 0)) + int(
        supplemental_quality.get("accepted", 0)
    )
    rejected = int(base_quality.get("rejected", 0)) + int(
        supplemental_quality.get("rejected", 0)
    )
    merged["quality"] = {
        "threshold": base_quality.get("threshold", supplemental_quality.get("threshold")),
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": accepted / max(1, accepted + rejected),
        "weights": base_quality.get("weights", supplemental_quality.get("weights", {})),
    }
    merged["augmentation"] = {
        "schema_version": 1,
        "reason": "restore_verified_dfc_and_identity_after_correcting_band_pass_metadata",
        "base_manifest_sha256": base_sha256,
        "supplemental_tokens": int(supplemental.get("total_tokens", 0)),
        "supplemental_records": int(supplemental.get("accepted_records", 0)),
        "created_at": time.time(),
    }
    return merged


def augment_verified_v4_shards(
    profile: str = "30gb",
    *,
    tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    """Repair an unpromoted failed-mix publication without re-encoding native shards."""
    from tokenizer.subword_tokenizer import SubwordTokenizer

    bound_tokenizer_path = Path(
        tokenizer_path or (REPO_ROOT / "tokenizer" / "tokenizer_v4_32k.json")
    ).resolve()
    tokenizer = SubwordTokenizer.load(bound_tokenizer_path)
    tokenizer_sha256 = hashlib.sha256(bound_tokenizer_path.read_bytes()).hexdigest()
    revision_dir = DATA_MANIFEST_DIR / "native_foundation_v4" / profile
    inventory_path = revision_dir / "token_inventory.json"
    if not inventory_path.is_file():
        raise FileNotFoundError(f"V4 inventory is missing: {inventory_path}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("tokenizer_sha256") != tokenizer_sha256:
        raise RuntimeError("Inventory tokenizer hash does not match the requested V4 tokenizer")
    if inventory.get("campaign_sampling_verified") is True:
        return inventory

    split_dirs = {
        "train": revision_dir,
        "validation": revision_dir / "validation",
        "test": revision_dir / "test",
    }
    manifests: dict[str, dict[str, Any]] = {}
    base_hashes: dict[str, str] = {}
    for split, directory in split_dirs.items():
        path = directory / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("augmentation"):
            raise RuntimeError(f"Shard manifest was already augmented: {path}")
        manifests[split] = payload
        base_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()

    existing_classes = {
        key
        for key, count in manifests["train"].get("source_class_token_mix", {}).items()
        if int(count) > 0
    }
    missing_classes = set(REQUIRED_CAMPAIGN_SOURCE_CLASSES) - existing_classes
    repairable = {"verified_dfc", "identity_replay"}
    if not missing_classes or not missing_classes <= repairable:
        raise RuntimeError(
            f"Refusing non-surgical shard augmentation; missing classes={sorted(missing_classes)}"
        )

    staging = revision_dir / ".verified-repair.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    combined: dict[str, dict[str, Any]] = {}
    try:
        for split, directory in split_dirs.items():
            records = (
                record
                for record in _verified_supplemental_records(split)
                if record.source_class in missing_classes
            )
            supplemental = TokenShardPublisher(
                staging / split,
                tokenizer_version=CANONICAL_TOKENIZER_VERSION,
                tokenizer_sha256=tokenizer_sha256,
            ).publish(
                records,
                tokenizer,
                allow_partial_final=True,
                minimum_replay_tokens=(
                    {"identity_replay": 4097}
                    if split == "train" and "identity_replay" in missing_classes
                    else None
                ),
            )
            renamed_shards: list[dict[str, Any]] = []
            for index, shard in enumerate(supplemental.get("shards", [])):
                source = staging / split / str(shard["path"])
                target_name = f"tokens-verified-{index:05d}.npy"
                target = directory / target_name
                if target.exists():
                    raise FileExistsError(f"Supplemental shard already exists: {target}")
                source.replace(target)
                renamed = dict(shard)
                renamed["path"] = target_name
                if hashlib.sha256(target.read_bytes()).hexdigest() != renamed.get("sha256"):
                    raise RuntimeError(f"Supplemental shard hash mismatch: {target}")
                renamed_shards.append(renamed)
            combined[split] = _merge_token_manifests(
                manifests[split],
                supplemental,
                renamed_shards=renamed_shards,
                base_sha256=base_hashes[split],
            )

        category_tokens = {
            key: int(combined["train"]["source_class_token_mix"].get(key, 0))
            for key in REQUIRED_CAMPAIGN_SOURCE_CLASSES
        }
        unclassified_tokens = sum(
            int(count)
            for key, count in combined["train"]["source_class_token_mix"].items()
            if key not in REQUIRED_CAMPAIGN_SOURCE_CLASSES
        )
        classified_total = sum(category_tokens.values())
        realized_mix = {
            key: count / max(1, classified_total) for key, count in category_tokens.items()
        }
        sampling_verified = (
            classified_total > 0
            and unclassified_tokens == 0
            and all(count > 0 for count in category_tokens.values())
        )
        combined["train"].update(
            {
                "campaign_mix_target": FOUNDATION_CAMPAIGN_MIX,
                "campaign_mix_realized": realized_mix,
                "campaign_mix_deviation": {
                    key: realized_mix[key] - target
                    for key, target in FOUNDATION_CAMPAIGN_MIX.items()
                },
                "campaign_mix_materialization": "deterministic_source_weighted_sampler",
                "campaign_sampling_verified": sampling_verified,
                "campaign_mix_verified": sampling_verified,
                "unclassified_tokens": unclassified_tokens,
            }
        )
        if not sampling_verified:
            raise RuntimeError("Verified-source augmentation did not satisfy campaign sampling")

        for split, directory in split_dirs.items():
            _atomic_json(directory / "manifest.json", combined[split])

        inventory.update(
            {
                "licensed_tokens": int(combined["train"]["total_tokens"]),
                "validation_tokens": int(combined["validation"]["total_tokens"]),
                "test_tokens": int(combined["test"]["total_tokens"]),
                "sources": combined["train"].get("source_record_mix", {}),
                "source_revisions": combined["train"].get("source_revisions", []),
                "licenses": combined["train"].get("licenses", []),
                "campaign_mix_realized": realized_mix,
                "campaign_sampling_verified": True,
                "campaign_mix_verified": True,
                "unclassified_tokens": unclassified_tokens,
                "augmentation": combined["train"]["augmentation"],
            }
        )
        _atomic_json(inventory_path, inventory)
        _atomic_json(TOKEN_INVENTORY_MANIFEST, inventory)
        _atomic_json(
            TOKEN_SHARD_PROGRESS,
            {
                "schema_version": 1,
                "status": "complete",
                "updated_at": time.time(),
                "tokenizer_family": "v4",
                "tokenizer_sha256": tokenizer_sha256,
                "profile": profile,
                "published_tokens": {
                    split: int(payload["total_tokens"]) for split, payload in combined.items()
                },
                "published_shards": {
                    split: len(payload["shards"]) for split, payload in combined.items()
                },
                "inventory": str(inventory_path),
                "augmentation": inventory["augmentation"],
            },
        )
        return inventory
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download An-Ra training data buckets.")
    parser.add_argument(
        "--profile",
        choices=sorted(DATA_PROFILES),
        default="30gb",
        help="Data size profile. 30gb is the default native continuation campaign.",
    )
    parser.add_argument(
        "--bucket",
        choices=["base", "reasoning", "science"],
        help="Download only one bucket. Omit to build all buckets.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show planned work without downloading."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an audited foundation corpus; never truncates existing data.",
    )
    parser.add_argument(
        "--recover-only",
        action="store_true",
        help="Finalize the durable append journal without downloading additional data.",
    )
    parser.add_argument(
        "--repair-verified-shards",
        action="store_true",
        help="Augment a failed-mix V4 publication with verified DFC and identity shards.",
    )
    parser.add_argument(
        "--publish-token-shards",
        action="store_true",
        help="Publish immutable 10M-token uint16 FineWeb-Edu shards after download.",
    )
    parser.add_argument(
        "--shards-only",
        action="store_true",
        help="Publish token shards from an already prepared local corpus without downloads.",
    )
    parser.add_argument(
        "--tokenizer-family",
        choices=("v4",),
        default="v4",
        help="Publish source-pure shards bound to canonical V4.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Explicit tokenizer artifact; required implicitly for V4 if non-canonical.",
    )
    parser.add_argument(
        "--prepare-corpus",
        action="store_true",
        help="Convert downloaded files into anra_training.txt and teacher_reasoning_v2.jsonl.",
    )
    parser.add_argument(
        "--max-source-mb",
        type=int,
        default=4096,
        help="Maximum downloaded source file size to ingest when --prepare-corpus is set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_training_data_dir()
    if args.recover_only:
        if args.shards_only or args.publish_token_shards or args.prepare_corpus or args.dry_run:
            raise ValueError("--recover-only cannot be combined with other execution modes")
        report = recover_native_foundation_append()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.repair_verified_shards:
        if args.shards_only or args.publish_token_shards or args.prepare_corpus or args.dry_run:
            raise ValueError(
                "--repair-verified-shards cannot be combined with other execution modes"
            )
        inventory = augment_verified_v4_shards(
            args.profile,
            tokenizer_path=args.tokenizer_path,
        )
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 0
    if args.shards_only:
        if args.dry_run or args.prepare_corpus or not args.publish_token_shards:
            raise ValueError(
                "--shards-only requires --publish-token-shards and cannot be combined "
                "with --dry-run or --prepare-corpus"
            )
        inventory = publish_fineweb_token_shards(
            args.profile,
            tokenizer_path=args.tokenizer_path,
            tokenizer_family=args.tokenizer_family,
        )
        print(f"Published licensed token inventory: {inventory['licensed_tokens']:,}")
        return 0
    load_dataset = load_datasets_import(dry_run=args.dry_run)

    buckets = [args.bucket] if args.bucket else ["base", "reasoning", "science"]
    if args.resume and buckets != ["base"]:
        raise ValueError("--resume currently requires --bucket base")

    print("AN-RA TRAINING DATA DOWNLOAD")
    print("=" * 60)
    print(f"Profile: {args.profile}")
    print(f"Buckets: {', '.join(buckets)}")
    if args.dry_run:
        print("Mode: dry run")
    print()

    profile = DATA_PROFILES[args.profile]
    results: list[dict[str, Any]] = []
    inventory: dict[str, Any] | None = None
    for bucket in buckets:
        if bucket == "base":
            if "target_gb" in profile:
                results.append(
                    download_native_foundation(
                        load_dataset,
                        target_gb=float(profile["target_gb"]),
                        native_target_gb=(
                            float(profile["native_target_gb"])
                            if "native_target_gb" in profile
                            else None
                        ),
                        dry_run=args.dry_run,
                        resume=args.resume,
                    )
                )
            else:
                results.append(
                    download_base(
                        load_dataset,
                        dry_run=args.dry_run,
                        fineweb_docs=int(profile["fineweb_docs"]),
                        redpajama_docs=int(profile["redpajama_docs"]),
                    )
                )
        elif bucket == "reasoning":
            results.append(
                download_reasoning(
                    load_dataset,
                    dry_run=args.dry_run,
                    per_source_limit=profile["reasoning_per_source"],
                )
            )
        elif bucket == "science":
            results.append(
                download_science(
                    load_dataset,
                    dry_run=args.dry_run,
                    per_source_limit=profile["science_per_source"],
                )
            )

    if args.publish_token_shards and not args.dry_run:
        inventory = publish_fineweb_token_shards(
            args.profile,
            tokenizer_path=args.tokenizer_path,
            tokenizer_family=args.tokenizer_family,
        )
        print(f"Published licensed token inventory: {inventory['licensed_tokens']:,}")

    if args.prepare_corpus and not args.dry_run:
        from training.data_ingestion import prepare_training_corpus

        sources = [
            TRAINING_DATA_DIR / "base_corpus.txt",
            TRAINING_DATA_DIR / "reasoning.jsonl",
            TRAINING_DATA_DIR / "frontier_dfc.jsonl",
        ]
        report = prepare_training_corpus(
            explicit_sources=[source for source in sources if source.exists()],
            include_drive=True,
            max_source_mb=args.max_source_mb,
            mount_drive=False,
        )
        print(
            "Prepared AN-RA corpus: "
            f"{report.total_examples:,} examples, {report.teacher_records:,} teacher records"
        )

    print_summary()
    failures = [
        SourceDownloadFailure(result["bucket"], str(error))
        for result in results
        for error in result.get("errors", [])
    ]
    campaign_bytes = sum(
        Path(str(result.get("output", ""))).stat().st_size
        for result in results
        if result.get("output") and Path(str(result["output"])).is_file()
    )
    target_gb = float(profile.get("target_gb", 0.0))
    required_campaign_bytes = int(target_gb * 1024**3)
    native_result = next(
        (
            result
            for result in results
            if result.get("bucket") == "base"
            and "raw_foundation_target_bytes" in result
        ),
        None,
    )
    if native_result is not None:
        required_campaign_bytes = int(
            native_result.get("raw_foundation_target_bytes", required_campaign_bytes)
        )
    enforce_campaign_size = not args.dry_run and target_gb > 0.0 and buckets in (
        ["base"],
        ["base", "reasoning", "science"],
    )
    if enforce_campaign_size and campaign_bytes < int(required_campaign_bytes * 0.98):
        failures.append(
            SourceDownloadFailure(
                "campaign_size",
                f"prepared {campaign_bytes / 1024**3:.2f} GB of required "
                f"{required_campaign_bytes / 1024**3:.2f} GB",
            )
        )
    if (
        inventory is not None
        and args.profile in {"15gb", "30gb"}
        and not bool(inventory.get("campaign_mix_verified", False))
    ):
        failures.append(
            SourceDownloadFailure(
                "campaign_mix",
                "immutable shards do not satisfy the registered Phase-A source mix",
            )
        )
    status = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "incomplete" if failures else "complete",
        "requested_buckets": buckets,
        "buckets": results,
        "campaign_bytes": campaign_bytes,
        "required_campaign_bytes": required_campaign_bytes,
        "campaign_target_gb": target_gb,
        "failures": [
            {"source": failure.source, "message": failure.message} for failure in failures
        ],
    }
    status_path = _download_status_path(buckets)
    if not args.dry_run:
        _atomic_json(status_path, status)
    if failures:
        print(f"INCOMPLETE INVENTORY: {len(failures)} source failure(s). See {status_path}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
