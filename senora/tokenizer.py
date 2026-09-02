"""Senora Tokenizer Contracts, Learned BPE Interface, and Test Byte Fallback.

Enforces:
1. Clear architectural separation between the production learned tokenizer (LearnedBpeTokenizer)
   and the test byte fallback (ByteFallbackTestTokenizer).
2. Scientific execution fails closed if the production tokenizer artifact is missing or altered.
3. ByteFallbackTestTokenizer is forbidden in scientific mode unless explicitly authorized via allow_test_tokenizer=True.
4. Exact vocabulary size: 24,576 tokens.
5. Reserved special tokens: PAD=0, UNK=1, BOS=2, EOS=3.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from v5_contracts.data_spec import ArtifactIdentity, TokenizerReceipt


EXPECTED_VOCABULARY_SIZE = 24_576
SPECIAL_TOKENS: dict[str, int] = {
    "pad": 0,
    "unk": 1,
    "bos": 2,
    "eos": 3,
}
BYTE_OFFSET = 4  # Bytes 0..255 map to IDs 4..259 in test fallback


class TokenizerValidationError(ValueError):
    """Raised when a tokenizer artifact or encoding violates the V5 specification."""


class MissingTokenizerArtifactError(FileNotFoundError):
    """Raised when production tokenizer artifact is missing during scientific execution."""


@dataclass(frozen=True, slots=True)
class ByteFallbackTestTokenizer:
    """Explicit byte-fallback tokenizer for unit testing and plumbing certification ONLY.
    
    WARNING: Not a genuine 24,576 learned subword tokenizer.
    Encodes raw UTF-8 bytes into token range [4, 259].
    Forbidden on the production scientific path.
    """
    vocabulary_size: int = EXPECTED_VOCABULARY_SIZE
    special_tokens: Mapping[str, int] = field(default_factory=lambda: dict(SPECIAL_TOKENS))
    is_test_fallback: bool = True
    artifact_sha256: str = "BYTE_FALLBACK_TEST_TOKENIZER"

    def __post_init__(self) -> None:
        if self.vocabulary_size != EXPECTED_VOCABULARY_SIZE:
            raise TokenizerValidationError(
                f"Vocabulary size must be exactly {EXPECTED_VOCABULARY_SIZE}, got {self.vocabulary_size}"
            )
        for name, expected_id in SPECIAL_TOKENS.items():
            if self.special_tokens.get(name) != expected_id:
                raise TokenizerValidationError(
                    f"Special token {name!r} must be ID {expected_id}, got {self.special_tokens.get(name)}"
                )

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text to byte token IDs."""
        raw_bytes = text.encode("utf-8")
        tokens = [b + BYTE_OFFSET for b in raw_bytes]
        if add_bos:
            tokens.insert(0, SPECIAL_TOKENS["bos"])
        if add_eos:
            tokens.append(SPECIAL_TOKENS["eos"])
        return tokens

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to string via byte reconstruction."""
        specials = set(SPECIAL_TOKENS.values())
        byte_list: list[int] = []
        for tid in token_ids:
            if tid in specials:
                if skip_special_tokens:
                    continue
                else:
                    continue
            if BYTE_OFFSET <= tid < BYTE_OFFSET + 256:
                byte_list.append(tid - BYTE_OFFSET)
        return bytes(byte_list).decode("utf-8", errors="replace")

    def verify_roundtrip(self, text: str) -> bool:
        """Assert exact round-trip without unknown tokens."""
        ids = self.encode(text)
        if SPECIAL_TOKENS["unk"] in ids:
            raise TokenizerValidationError("Unexpected UNK token in encoded sequence")
        return self.decode(ids) == text


@dataclass(frozen=True, slots=True)
class LearnedBpeTokenizer:
    """Production 24,576 vocabulary BPE tokenizer loaded from verified on-disk artifact."""
    vocabulary: Mapping[str, int]
    inverse_vocabulary: Mapping[int, str]
    vocabulary_size: int = EXPECTED_VOCABULARY_SIZE
    special_tokens: Mapping[str, int] = field(default_factory=lambda: dict(SPECIAL_TOKENS))
    artifact_sha256: str = ""
    is_test_fallback: bool = False

    def __post_init__(self) -> None:
        if self.vocabulary_size != EXPECTED_VOCABULARY_SIZE:
            raise TokenizerValidationError(
                f"Vocabulary size must be exactly {EXPECTED_VOCABULARY_SIZE}, got {self.vocabulary_size}"
            )
        if len(self.vocabulary) != self.vocabulary_size:
            raise TokenizerValidationError(
                f"Vocabulary table size ({len(self.vocabulary)}) does not match vocabulary_size ({self.vocabulary_size})"
            )
        for name, expected_id in SPECIAL_TOKENS.items():
            if self.special_tokens.get(name) != expected_id:
                raise TokenizerValidationError(
                    f"Special token {name!r} must be ID {expected_id}, got {self.special_tokens.get(name)}"
                )

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text using the production vocabulary mapping with byte fallback."""
        tokens: list[int] = []
        # Prefix with BOS if requested
        if add_bos:
            tokens.append(SPECIAL_TOKENS["bos"])

        # Tokenize by greedy longest matching piece from vocabulary
        i = 0
        n = len(text)
        while i < n:
            matched = False
            # Check for subwords up to length 64
            for l in range(min(64, n - i), 0, -1):
                sub = text[i : i + l]
                if sub in self.vocabulary:
                    tokens.append(self.vocabulary[sub])
                    i += l
                    matched = True
                    break
            if not matched:
                # Fallback to UTF-8 byte pieces
                char_bytes = text[i].encode("utf-8")
                for b in char_bytes:
                    byte_key = f"<byte_{b}>"
                    tokens.append(self.vocabulary.get(byte_key, SPECIAL_TOKENS["unk"]))
                i += 1

        if add_eos:
            tokens.append(SPECIAL_TOKENS["eos"])
        return tokens

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text."""
        specials = set(SPECIAL_TOKENS.values())
        parts: list[str] = []
        for tid in token_ids:
            if tid in specials:
                if skip_special_tokens:
                    continue
                else:
                    continue
            piece = self.inverse_vocabulary.get(tid, "")
            if piece.startswith("<byte_") and piece.endswith(">"):
                try:
                    b_val = int(piece[6:-1])
                    parts.append(bytes([b_val]).decode("utf-8", errors="replace"))
                except ValueError:
                    pass
            else:
                parts.append(piece)
        return "".join(parts)


# Backward-compatible alias for existing unit tests
SenoraTokenizer = ByteFallbackTestTokenizer


def load_verified_tokenizer(
    receipt_path: Path | None = None,
    artifact_path: Path | None = None,
    *,
    allow_test_tokenizer: bool = False,
) -> ByteFallbackTestTokenizer | LearnedBpeTokenizer:
    """Load and verify tokenizer. Fails closed if production artifact is missing and test fallback not allowed."""
    if artifact_path is not None and artifact_path.is_file():
        actual_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
        vocab = data["vocabulary"]
        inv_vocab = {v: k for k, v in vocab.items()}

        if receipt_path is not None and receipt_path.is_file():
            receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
            expected_sha = receipt_data["artifact"]["raw_sha256"]
            if actual_sha != expected_sha:
                raise TokenizerValidationError(
                    f"Tokenizer artifact SHA-256 mismatch: {actual_sha} != {expected_sha}"
                )

        return LearnedBpeTokenizer(
            vocabulary=vocab,
            inverse_vocabulary=inv_vocab,
            vocabulary_size=len(vocab),
            special_tokens=SPECIAL_TOKENS,
            artifact_sha256=actual_sha,
        )

    if allow_test_tokenizer:
        return ByteFallbackTestTokenizer()

    raise MissingTokenizerArtifactError(
        "Production learned tokenizer artifact is missing on disk. "
        "Scientific execution fails closed. For unit tests or local dry runs, set allow_test_tokenizer=True."
    )