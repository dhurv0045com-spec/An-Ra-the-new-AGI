"""Canonical loading of the real V5 tokenizer artifact.

Exactly one loader exists. It verifies the artifact bytes against the frozen
SHA-256, checks vocabulary size, special IDs, and the no-normalization
contract, then hands the backend to ``FrozenTokenizer``. Anything else fails
closed: no silent substitution, no byte-fallback impostors, no guessed IDs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from .adapter import SPECIAL_TOKEN_IDS, FrozenTokenizer, TokenizerIdentity


ARTIFACT_SCHEMA = "anra-v5-tokenizer-artifact/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_tokenizer(
    artifact_path: Path,
    *,
    expected_sha256: str,
    vocabulary_size: int,
    trainer_config_sha256: str,
    corpus_manifest_sha256: str,
) -> tuple[Any, TokenizerIdentity]:
    """Load a gzipped ``tokenizers`` artifact after verifying every identity."""

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("loading a real artifact requires the tokenizers package") from exc
    actual = sha256_file(artifact_path)
    if actual != expected_sha256:
        raise ValueError("tokenizer artifact bytes do not match the frozen SHA-256")
    payload = gzip.open(artifact_path, "rt", encoding="utf-8").read()
    backend = Tokenizer.from_str(payload)
    if backend.get_vocab_size() != vocabulary_size:
        raise ValueError("artifact vocabulary size contradicts the frozen identity")
    for name, index in SPECIAL_TOKEN_IDS.items():
        if backend.token_to_id(f"<{name}>") != index:
            raise ValueError(f"artifact special token <{name}> is not id {index}")
    spec = json.loads(payload)
    if spec.get("normalizer") is not None:
        raise ValueError("V5 forbids tokenizer normalization")
    identity = TokenizerIdentity(
        schema="anra-v5-tokenizer-identity/v1",
        vocabulary_size=vocabulary_size,
        special_token_ids=dict(SPECIAL_TOKEN_IDS),
        artifact_sha256=actual,
        trainer_config_sha256=trainer_config_sha256,
        corpus_manifest_sha256=corpus_manifest_sha256,
    )
    identity.assert_valid()
    return backend, identity


def load_frozen(
    artifact_path: Path,
    *,
    expected_sha256: str,
    vocabulary_size: int,
    trainer_config_sha256: str,
    corpus_manifest_sha256: str,
) -> FrozenTokenizer:
    """Load and wrap the verified artifact in one canonical step."""

    backend, identity = load_verified_tokenizer(
        artifact_path,
        expected_sha256=expected_sha256,
        vocabulary_size=vocabulary_size,
        trainer_config_sha256=trainer_config_sha256,
        corpus_manifest_sha256=corpus_manifest_sha256,
    )
    return FrozenTokenizer(identity=identity, backend=backend)


__all__ = ["ARTIFACT_SCHEMA", "load_frozen", "load_verified_tokenizer", "sha256_file"]
