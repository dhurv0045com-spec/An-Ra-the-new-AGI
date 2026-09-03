"""Frozen byte-BPE tokenizer interface and identity for V5.

The interface is byte-level BPE with byte fallback: exactly 24,576 entries in
the V5-A center, no Unicode normalization, case folding, whitespace rewrite,
prefix-space insertion, or tokenizer dropout. Reserved IDs are PAD 0, UNK 1,
BOS 2, EOS 3. Freezing requires an exact artifact hash, a trainer-config
hash, a corpus-manifest hash, a passing round-trip probe, and zero unknowns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from v5_contracts.data_spec import TokenizerReceipt
from v5_contracts.lineage import ArtifactIdentity


SPECIAL_TOKEN_IDS = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}


@dataclass(frozen=True, slots=True)
class TokenizerIdentity:
    schema: str
    vocabulary_size: int
    special_token_ids: Mapping[str, int]
    artifact_sha256: str
    trainer_config_sha256: str
    corpus_manifest_sha256: str

    def assert_valid(self) -> None:
        if self.schema != "anra-v5-tokenizer-identity/v1":
            raise ValueError("unsupported tokenizer-identity schema")
        if self.vocabulary_size <= 256:
            raise ValueError("vocabulary is too small for the subword contract")
        if dict(self.special_token_ids) != SPECIAL_TOKEN_IDS:
            raise ValueError("V5 reserves exactly PAD 0, UNK 1, BOS 2, EOS 3")
        for name in ("artifact_sha256", "trainer_config_sha256", "corpus_manifest_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")

    def freeze(
        self,
        *,
        artifact: ArtifactIdentity,
        identity_roundtrip_passed: bool,
        unknown_rate: float,
    ) -> TokenizerReceipt:
        """Bind this identity to artifact provenance and probe evidence."""

        self.assert_valid()
        if artifact.sha256 != self.artifact_sha256:
            raise ValueError("artifact provenance does not match the frozen identity")
        receipt = TokenizerReceipt(
            schema="anra-v5-tokenizer-receipt/v1",
            artifact=artifact,
            vocabulary_size=self.vocabulary_size,
            special_token_ids=dict(self.special_token_ids),
            trainer_config_sha256=self.trainer_config_sha256,
            corpus_manifest_sha256=self.corpus_manifest_sha256,
            identity_roundtrip_passed=identity_roundtrip_passed,
            unknown_rate=unknown_rate,
        )
        receipt.assert_valid()
        return receipt


class FrozenTokenizer:
    """Encode/decode facade over an opaque backend (duck-typed).

    The backend must expose ``encode(text) -> list[int]`` and
    ``decode(ids) -> str`` (the ``tokenizers`` library Tokenizer satisfies
    this). This facade adds range validation, BOS/EOS helpers, and an unknown
    audit; it never invents IDs.
    """

    def __init__(self, *, identity: TokenizerIdentity, backend: Any) -> None:
        identity.assert_valid()
        if not callable(getattr(backend, "encode", None)) or not callable(
            getattr(backend, "decode", None)
        ):
            raise ValueError("backend must expose encode(text) and decode(ids)")
        self._identity = identity
        self._backend = backend

    @property
    def identity(self) -> TokenizerIdentity:
        return self._identity

    @property
    def vocabulary_size(self) -> int:
        return self._identity.vocabulary_size

    def encode(self, text: str) -> list[int]:
        ids = [int(token) for token in self._backend.encode(text)]
        for token in ids:
            if not 0 <= token < self._identity.vocabulary_size:
                raise ValueError("backend emitted an out-of-vocabulary id")
        return ids

    def decode(self, ids: list[int]) -> str:
        for token in ids:
            if not 0 <= token < self._identity.vocabulary_size:
                raise ValueError("cannot decode an out-of-vocabulary id")
        return str(self._backend.decode(list(ids)))

    def audit(self, texts: list[str]) -> dict[str, object]:
        """Encode a probe corpus; report round-trip and unknown statistics."""

        total_unknown = 0
        total_tokens = 0
        roundtrip_ok = True
        unk = self._identity.special_token_ids["unk"]
        for text in texts:
            ids = self.encode(text)
            total_tokens += len(ids)
            total_unknown += sum(1 for token in ids if token == unk)
            if self.decode(ids) != text:
                roundtrip_ok = False
        return {
            "texts": len(texts),
            "tokens": total_tokens,
            "unknowns": total_unknown,
            "unknown_rate": (total_unknown / total_tokens) if total_tokens else 0.0,
            "identity_roundtrip_passed": roundtrip_ok and total_tokens > 0,
        }

    def segment(self, token_ids: list[int]) -> list[int]:
        """Wrap content IDs as a BOS-content-EOS training segment."""

        specials = self._identity.special_token_ids
        for token in token_ids:
            if not 0 <= token < self._identity.vocabulary_size:
                raise ValueError("cannot segment an out-of-vocabulary id")
            if token in (specials["pad"], specials["bos"], specials["eos"]):
                raise ValueError("content must not carry boundary markers")
        return [specials["bos"], *token_ids, specials["eos"]]


__all__ = [
    "SPECIAL_TOKEN_IDS",
    "FrozenTokenizer",
    "TokenizerIdentity",
]
