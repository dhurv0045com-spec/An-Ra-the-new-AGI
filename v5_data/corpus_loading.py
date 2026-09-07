"""Shared first-party corpus/tokenizer loading for canaries and audits.

Binds the frozen 24,576-entry tokenizer artifact and the real first-party
corpus (current-HEAD tracked text files, hashed as read). Lives in the data
plane so data-plane consumers (supply audits, foundry stages) do not import
the training plane; training-plane modules re-export these helpers.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from v5_data.manifest import Document
from v5_tokenizer.adapter import FrozenTokenizer, TokenizerIdentity


MAX_CORPUS_FILES = 48
MAX_DOCUMENT_TOKENS = 6_000


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Bind real first-party documents: tracked text files of the current HEAD.

    Every file's bytes are hashed as read, so the manifest's raw_sha256 values
    verify against the working tree at run time.
    """

    exclude = {".git", ".venv", "__pycache__", "artifacts", "state"}
    text_suffixes = {
        ".py", ".md", ".toml", ".json", ".cfg", ".yml", ".yaml", ".txt", ".sh",
    }
    documents: list[Document] = []
    total_bytes = 0
    for path in sorted(repo.rglob("*")):
        if len(documents) >= MAX_CORPUS_FILES or total_bytes > 500_000:
            break
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        if any(part in exclude for part in path.relative_to(repo).parts[:-1]):
            continue
        raw = path.read_bytes()
        if not raw.strip():
            continue
        raw_source_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        encoded = tokenizer.encode(text)
        if len(encoded) > MAX_DOCUMENT_TOKENS:
            text = tokenizer.decode(encoded[:MAX_DOCUMENT_TOKENS])
        relative = path.relative_to(repo).as_posix()
        is_code = path.suffix in {".py", ".toml", ".json", ".cfg", ".yml", ".yaml", ".sh"}
        documents.append(
            Document(
                doc_id=relative,
                text=text,
                source_id=relative,
                domain="code" if is_code else "prose",
                family="code_math_formal" if is_code else "natural",
                authorization_category="first-party-authorized",
                acquired_date="2026-09-03",
                raw_source_sha256=raw_source_sha256,
            )
        )
        total_bytes += len(raw)
    if len(documents) < 8:
        raise ValueError("miniature corpus failed to bind enough verified real documents")
    return documents


__all__ = [
    "MAX_CORPUS_FILES",
    "MAX_DOCUMENT_TOKENS",
    "_HFTokenizerBackend",
    "_canonical_json",
    "_load_corpus",
    "_load_tokenizer",
    "_sha256_file",
]
