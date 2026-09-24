"""Shared first-party corpus/tokenizer loading for canaries and audits.

Binds the frozen 24,576-entry tokenizer artifact and a pinned allowlist of
first-party research documents. The same manifest works in Kaggle snapshots
without Git metadata and ignores all workspace files that are not listed.
Lives in the data plane so data-plane consumers do not import the training
plane; training-plane modules re-export these helpers.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

from v5_data.manifest import Document
from v5_tokenizer.adapter import FrozenTokenizer, TokenizerIdentity


MAX_CORPUS_FILES = 48
MAX_DOCUMENT_TOKENS = 20_000
MAX_DOCUMENT_BYTES = 128_000
MAX_CORPUS_BYTES = 500_000
MAX_MANIFEST_BYTES = 64_000
CORPUS_MANIFEST_SCHEMA = "anra-v5-first-party-corpus-manifest/v1"
CORPUS_MANIFEST_RELATIVE_PATH = "v5_data/first_party_corpus_manifest.json"
CORPUS_MANIFEST_SHA256 = "5bf48c3e9f2b3a14719a1483cd281d4eaf7d3d2150e35e3eb327c1337f675d55"
_ALLOWED_CORPUS_PREFIXES = (
    PurePosixPath("docs/research"),
    PurePosixPath("docs/cymek/research"),
    PurePosixPath("docs/signac_100m"),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in corpus manifest: {key}")
        result[key] = value
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_manifest_path(root: Path, raw_path: object) -> tuple[str, Path]:
    if not isinstance(raw_path, str) or "\\" in raw_path:
        raise ValueError("corpus manifest paths must be repository-relative POSIX paths")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != raw_path
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.suffix != ".md"
        or not any(relative.is_relative_to(prefix) for prefix in _ALLOWED_CORPUS_PREFIXES)
    ):
        raise ValueError(f"unsafe or out-of-scope corpus path: {raw_path!r}")

    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"corpus manifest paths may not traverse symlinks: {raw_path}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"corpus document is missing or escapes the repository: {raw_path}") from exc
    if not resolved.is_file():
        raise ValueError(f"corpus entry is not a regular file: {raw_path}")
    return raw_path, resolved


class _HFTokenizerBackend:
    """Adapter exposing encode(text)->ids and decode(ids)->str."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def encode(self, text: str) -> list[int]:
        return list(self._backend.encode(text).ids)

    def decode(self, ids: list[int]) -> str:
        return self._backend.decode(ids, skip_special_tokens=False)


def _load_tokenizer(repo: Path) -> tuple[FrozenTokenizer, dict[str, object]]:
    artifact = repo / "artifacts/e1/local_tournament/tokenizer-24576.json.gz"
    result = json.loads((repo / "artifacts/e1/local_tournament/result.json").read_text("utf-8"))
    row = next(r for r in result["candidate_rows"] if r["vocabulary_size"] == 24_576)
    if row["artifact_sha256"] != _sha256_file(artifact):
        raise ValueError("committed tokenizer artifact does not match its tournament receipt")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("the optional `tokenizers` package is required") from exc
    backend = _HFTokenizerBackend(
        Tokenizer.from_str(gzip.decompress(artifact.read_bytes()).decode("utf-8"))
    )
    trainer_config = json.dumps(result["trainer"], sort_keys=True).encode("utf-8")
    identity = TokenizerIdentity(
        schema="anra-v5-tokenizer-identity/v1",
        vocabulary_size=24_576,
        special_token_ids={"pad": 0, "unk": 1, "bos": 2, "eos": 3},
        artifact_sha256=row["artifact_sha256"],
        trainer_config_sha256=hashlib.sha256(trainer_config).hexdigest(),
        corpus_manifest_sha256=result["corpus_manifest_sha256"],
    )
    return FrozenTokenizer(identity=identity, backend=backend), dict(row["evaluation"])


def _load_corpus(repo: Path, tokenizer: FrozenTokenizer) -> list[Document]:
    """Load only pinned, hash-verified research documents from the manifest.

    Git is intentionally not consulted: Kaggle snapshots often omit it. The
    manifest digest and each raw file digest are checked before tokenization;
    unlisted workspace, test, evaluator, and notebook files are never scanned.
    """

    root = repo.resolve(strict=True)
    manifest_path = root / CORPUS_MANIFEST_RELATIVE_PATH
    manifest_cursor = root
    for part in PurePosixPath(CORPUS_MANIFEST_RELATIVE_PATH).parts:
        manifest_cursor = manifest_cursor / part
        if manifest_cursor.is_symlink():
            raise ValueError("corpus manifest path may not traverse symlinks")
    try:
        manifest_size = manifest_path.stat().st_size
        if manifest_size > MAX_MANIFEST_BYTES:
            raise ValueError("first-party corpus manifest exceeds its byte limit")
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError("pinned first-party corpus manifest is missing") from exc
    if len(manifest_bytes) != manifest_size:
        raise ValueError("first-party corpus manifest changed while it was being read")
    if hashlib.sha256(manifest_bytes).hexdigest() != CORPUS_MANIFEST_SHA256:
        raise ValueError("first-party corpus manifest digest does not match the pinned identity")
    try:
        manifest = json.loads(
            manifest_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("first-party corpus manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "acquired_date", "corpus_id", "documents", "license", "provenance", "schema",
        "usage_scope",
    }:
        raise ValueError("first-party corpus manifest has an invalid top-level schema")
    if (
        manifest["schema"] != CORPUS_MANIFEST_SCHEMA
        or manifest["corpus_id"] != "signac-first-party-research-canary-v1"
        or manifest["license"] != "MIT"
        or manifest["usage_scope"] != "development-canary-only"
        or not isinstance(manifest["provenance"], str)
        or not manifest["provenance"].strip()
    ):
        raise ValueError("first-party corpus manifest identity or provenance is invalid")
    try:
        date.fromisoformat(manifest["acquired_date"])
    except (TypeError, ValueError) as exc:
        raise ValueError("first-party corpus manifest acquired_date is invalid") from exc

    entries = manifest["documents"]
    if not isinstance(entries, list) or not 8 <= len(entries) <= MAX_CORPUS_FILES:
        raise ValueError("first-party corpus manifest has an invalid document count")
    required_entry_keys = {
        "acquired_date", "authorization_category", "domain", "doc_id", "family",
        "license", "path", "provenance", "sha256",
    }
    documents: list[Document] = []
    total_bytes = 0
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    previous_path = ""
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != required_entry_keys:
            raise ValueError("first-party corpus manifest entry has an invalid schema")
        relative, path = _safe_manifest_path(root, entry["path"])
        doc_id = entry["doc_id"]
        if not isinstance(doc_id, str) or doc_id != relative:
            raise ValueError("corpus document id must equal its canonical manifest path")
        if doc_id in seen_ids or relative in seen_paths or relative <= previous_path:
            raise ValueError("corpus manifest entries must be unique and sorted by path")
        seen_ids.add(doc_id)
        seen_paths.add(relative)
        previous_path = relative
        if (
            entry["domain"] != "prose"
            or entry["family"] != "natural"
            or entry["authorization_category"] != "first-party-development-only"
            or entry["license"] != "MIT"
            or not isinstance(entry["provenance"], str)
            or not entry["provenance"].strip()
        ):
            raise ValueError(f"corpus entry has invalid provenance or classification: {relative}")
        try:
            date.fromisoformat(entry["acquired_date"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"corpus entry acquired_date is invalid: {relative}") from exc
        if not _is_sha256(entry["sha256"]):
            raise ValueError(f"corpus entry has an invalid raw SHA-256: {relative}")
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise ValueError(f"cannot stat pinned corpus document: {relative}") from exc
        if file_size > MAX_DOCUMENT_BYTES:
            raise ValueError(f"corpus document exceeds its byte limit: {relative}")
        if total_bytes + file_size > MAX_CORPUS_BYTES:
            raise ValueError("pinned first-party corpus exceeds its aggregate byte limit")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read pinned corpus document: {relative}") from exc
        if len(raw) != file_size:
            raise ValueError(f"corpus document changed while it was being read: {relative}")
        raw_source_sha256 = hashlib.sha256(raw).hexdigest()
        if raw_source_sha256 != entry["sha256"]:
            raise ValueError(f"corpus document digest does not match the pinned manifest: {relative}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"corpus document is not valid UTF-8: {relative}") from exc
        if not text.strip():
            raise ValueError(f"pinned corpus document is empty: {relative}")
        encoded = tokenizer.encode(text)
        if len(encoded) > MAX_DOCUMENT_TOKENS:
            raise ValueError(f"corpus document exceeds its token limit; refusing truncation: {relative}")
        total_bytes += file_size
        documents.append(
            Document(
                doc_id=doc_id,
                text=text,
                source_id=doc_id,
                domain=entry["domain"],
                family=entry["family"],
                authorization_category=entry["authorization_category"],
                acquired_date=entry["acquired_date"],
                raw_source_sha256=raw_source_sha256,
            )
        )
    if len(documents) < 8:
        raise ValueError("miniature corpus failed to bind enough verified real documents")
    return documents


__all__ = [
    "CORPUS_MANIFEST_RELATIVE_PATH",
    "CORPUS_MANIFEST_SCHEMA",
    "CORPUS_MANIFEST_SHA256",
    "MAX_CORPUS_BYTES",
    "MAX_CORPUS_FILES",
    "MAX_DOCUMENT_BYTES",
    "MAX_DOCUMENT_TOKENS",
    "MAX_MANIFEST_BYTES",
    "_HFTokenizerBackend",
    "_canonical_json",
    "_load_corpus",
    "_load_tokenizer",
    "_sha256_file",
]
