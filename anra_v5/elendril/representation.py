"""Versioned text boundary; the historical symbolic codec remains immutable.

``UTF8`` is a lossless byte reference, not a trained subword tokenizer. It has
no corpus dependency or UNK token, performs no Unicode normalization and costs
one content token per UTF-8 byte. Special tokens can only be inserted through
the typed API. A literal spelling such as ``<EOS>`` is ordinary text.

Offsets are half-open positions in the original text. Each byte of a Unicode
codepoint shares its character interval and has its own byte interval. Prompt
and answer offsets are local to their respective input strings. Loss masks
mark target positions; a causal objective must shift them exactly once.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
import math
import re
from typing import Any

from . import codec


MAX_TEXT_BYTES = 16 * 1024 * 1024


class SpecialToken(Enum):
    PAD = "PAD"
    BOS = "BOS"
    EOS = "EOS"
    ANSWER = "ANSWER"


@dataclass(frozen=True, slots=True)
class TokenizerManifest:
    version: str
    vocabulary_size: int
    vocabulary_sha256: str
    normalizer_sha256: str
    pretokenizer_sha256: str
    config_sha256: str
    special_tokens: tuple[tuple[str, int], ...]
    checkpoint_identity: str | None = None

    @property
    def identity(self) -> str:
        # Old checkpoints already bind this exact identity. The new adapter
        # must never reinterpret that field as the hash of a different record.
        return self.checkpoint_identity or codec.digest(asdict(self))

    def record(self) -> dict:
        return {**asdict(self), "identity": self.identity}


@dataclass(frozen=True, slots=True)
class TokenOffset:
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int


@dataclass(frozen=True, slots=True)
class Encoding:
    ids: tuple[int, ...]
    offsets: tuple[TokenOffset, ...]


@dataclass(frozen=True, slots=True)
class FramedExample:
    ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    loss_mask: tuple[bool, ...]
    segment_ids: tuple[int, ...]
    prompt_span: tuple[int, int]
    answer_span: tuple[int, int]  # Includes EOS, excludes ANSWER delimiter.
    offsets: tuple[TokenOffset | None, ...]
    dropped_prompt_characters: int
    tokenizer_sha256: str


def _text(text: str) -> bytes:
    if not isinstance(text, str):
        raise TypeError("text must be a Unicode string")
    # Cheap upper bound prevents allocating enormous UTF-8 temporaries.
    if len(text) > MAX_TEXT_BYTES:
        raise ValueError("text exceeds byte-tokenizer safety limit")
    try:
        raw = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("text contains an unpaired Unicode surrogate") from exc
    if len(raw) > MAX_TEXT_BYTES:
        raise ValueError("text exceeds byte-tokenizer safety limit")
    return raw


def _limit(value: int | None, *, name: str, minimum: int = 0) -> None:
    if value is not None and (type(value) is not int or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, slots=True, init=False)
class Tokenizer:
    """Small immutable implementation shared by two explicitly distinct paths."""

    _manifest: TokenizerManifest
    _byte_level: bool

    def __init__(self, *, byte_level: bool):
        if type(byte_level) is not bool:
            raise TypeError("byte_level must be a bool")
        object.__setattr__(self, "_byte_level", byte_level)
        specials = tuple((item.value, index) for index, item in enumerate(SpecialToken)
                         if byte_level or item is not SpecialToken.ANSWER)
        vocabulary = (["PAD", "BOS", "EOS", "ANSWER"] + list(range(256)) if byte_level
                      else ["PAD", "BOS", "EOS", *codec.ALPHABET])
        object.__setattr__(self, "_manifest", TokenizerManifest(
            version="utf8-byte-v1" if byte_level else "legacy-symbol-v1",
            vocabulary_size=len(vocabulary),
            vocabulary_sha256=codec.digest(vocabulary),
            normalizer_sha256=codec.digest({"normalization": "identity", "surrogates": "reject"}),
            pretokenizer_sha256=codec.digest({"unit": "utf8-byte" if byte_level else "symbol"}),
            config_sha256=codec.digest({"schema": 1, "special_insertion": "typed-only",
                                       "max_text_bytes": MAX_TEXT_BYTES,
                                       "byte_offset": 4 if byte_level else None}),
            special_tokens=specials,
            checkpoint_identity=None if byte_level else codec.CODEC_SHA,
        ))

    @property
    def manifest(self) -> TokenizerManifest:
        return self._manifest

    @property
    def identity(self) -> str:
        return self.manifest.identity

    @property
    def vocabulary_size(self) -> int:
        return self.manifest.vocabulary_size

    @property
    def pad_id(self) -> int:
        return self.special(SpecialToken.PAD)

    @property
    def bos_id(self) -> int:
        return self.special(SpecialToken.BOS)

    @property
    def eos_id(self) -> int:
        return self.special(SpecialToken.EOS)

    def special(self, token: SpecialToken) -> int:
        if not isinstance(token, SpecialToken):
            raise TypeError("special token insertion requires SpecialToken")
        try:
            return dict(self.manifest.special_tokens)[token.value]
        except KeyError as exc:
            raise ValueError("special token unsupported by this representation") from exc

    def require_binding(self, tokenizer_sha256: str, vocabulary_size: int) -> None:
        if (type(vocabulary_size) is not int or vocabulary_size != self.vocabulary_size
                or tokenizer_sha256 != self.identity):
            raise ValueError("checkpoint tokenizer/vocabulary mismatch; explicit migration required")

    def require_manifest(self, record: dict) -> None:
        """Validate the complete persisted configuration, including old adapters.

        Comparing only an untrusted record's ``identity`` field would miss a
        changed vocabulary/normalizer carrying a copied historical identity.
        Canonical JSON also accepts the tuple-to-list conversion from storage.
        """
        if not isinstance(record, dict) or codec.canonical(record) != codec.canonical(self.manifest.record()):
            raise ValueError("tokenizer manifest does not match its supported implementation")

    def encode(self, text: str, *, max_tokens: int | None = None) -> tuple[int, ...]:
        _limit(max_tokens, name="max_tokens")
        raw = _text(text)
        count = len(raw) if self._byte_level else len(text)
        if max_tokens is not None and count > max_tokens:
            raise ValueError("content exceeds max_tokens; no implicit truncation")
        return tuple(value + 4 for value in raw) if self._byte_level else codec.encode(text)

    def encode_with_offsets(self, text: str, *, max_tokens: int | None = None) -> Encoding:
        ids = self.encode(text, max_tokens=max_tokens)
        offsets = []
        byte = 0
        for char, value in enumerate(text):
            size = len(value.encode("utf-8"))
            for index in range(size if self._byte_level else 1):
                offsets.append(TokenOffset(char, char + 1, byte + index, byte + index + 1))
            byte += size
        return Encoding(ids, tuple(offsets))

    def decode(self, ids: Iterable[int]) -> str:
        materialized = []
        for index, value in enumerate(ids):
            if index >= MAX_TEXT_BYTES:
                raise ValueError("content exceeds byte-tokenizer safety limit")
            materialized.append(value)
        ids = tuple(materialized)
        if not self._byte_level:
            return codec.decode(ids)
        if any(type(value) is not int or not 4 <= value < self.vocabulary_size for value in ids):
            raise ValueError("expected content tokens only")
        try:
            return bytes(value - 4 for value in ids).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("content is not a complete valid UTF-8 sequence") from exc

    def incremental_encoder(self, *, max_tokens: int = MAX_TEXT_BYTES) -> IncrementalEncoder:
        return IncrementalEncoder(self, max_tokens=max_tokens)

    def frame(self, prompt: str, answer: str, *, max_length: int | None = None,
              pad_to: int | None = None, truncate_prompt: bool = False) -> FramedExample:
        """Frame a supervised pair, never truncating the answer or its EOS.

        Explicit prompt truncation removes complete leading Unicode characters.
        Historical framing inserts no new token and matches ``serialize``.
        Byte framing inserts typed ANSWER after the prompt. Padding is trailing,
        never eligible for loss, with segment -1 for packed-layout compatibility.
        """
        _limit(max_length, name="max_length", minimum=2)
        _limit(pad_to, name="pad_to", minimum=2)
        if type(truncate_prompt) is not bool:
            raise TypeError("truncate_prompt must be a bool")
        source = self.encode_with_offsets(prompt)
        target = self.encode_with_offsets(answer)
        delimiter = (self.special(SpecialToken.ANSWER),) if self._byte_level else ()
        fixed = len(target.ids) + 2 + len(delimiter)
        if max_length is not None and fixed > max_length:
            raise ValueError("answer and required special tokens exceed max_length")
        drop = 0
        dropped_chars = 0
        if max_length is not None and fixed + len(source.ids) > max_length:
            if not truncate_prompt:
                raise ValueError("example exceeds max_length; enable explicit prompt truncation")
            drop = fixed + len(source.ids) - max_length
            # Do not start a byte sequence in the middle of a codepoint.
            while drop < len(source.ids) and source.offsets[drop].char_start == source.offsets[drop - 1].char_start:
                drop += 1
            dropped_chars = source.offsets[drop - 1].char_end
        prompt_ids, prompt_offsets = source.ids[drop:], source.offsets[drop:]
        answer_start = 1 + len(prompt_ids) + len(delimiter)
        ids = (self.bos_id, *prompt_ids, *delimiter, *target.ids, self.eos_id)
        length = len(ids)
        if pad_to is not None and (pad_to < length or (max_length is not None and pad_to > max_length)):
            raise ValueError("pad_to must cover the example and stay within max_length")
        padding = (pad_to or length) - length
        return FramedExample(
            ids=ids + (self.pad_id,) * padding,
            attention_mask=(True,) * length + (False,) * padding,
            loss_mask=(False,) * answer_start + (True,) * (len(target.ids) + 1) + (False,) * padding,
            segment_ids=(0,) * length + (-1,) * padding,
            prompt_span=(1, 1 + len(prompt_ids)), answer_span=(answer_start, length),
            offsets=(None, *prompt_offsets, *((None,) * len(delimiter)), *target.offsets,
                     None, *((None,) * padding)),
            dropped_prompt_characters=dropped_chars, tokenizer_sha256=self.identity,
        )


class IncrementalEncoder:
    """Encode valid Unicode chunks with offsets in the concatenated text.

    There is no subword lookahead or normalization state. A rejected chunk
    does not advance counters. Feed strings, not split UTF-8 byte sequences.
    """

    def __init__(self, tokenizer: Tokenizer, *, max_tokens: int):
        if type(max_tokens) is not int or not 0 <= max_tokens <= MAX_TEXT_BYTES:
            raise ValueError("incremental max_tokens must be within safety limit")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.characters = self.bytes = self.tokens = 0

    def feed(self, text: str) -> Encoding:
        encoded = self.tokenizer.encode_with_offsets(text, max_tokens=self.max_tokens - self.tokens)
        byte_count = len(_text(text))
        if self.bytes + byte_count > MAX_TEXT_BYTES:
            raise ValueError("incremental text exceeds safety limit")
        offsets = tuple(TokenOffset(item.char_start + self.characters, item.char_end + self.characters,
                                    item.byte_start + self.bytes, item.byte_end + self.bytes)
                        for item in encoded.offsets)
        self.characters += len(text)
        self.bytes += byte_count
        self.tokens += len(encoded.ids)
        return Encoding(encoded.ids, offsets)


LEGACY = Tokenizer(byte_level=False)
UTF8 = Tokenizer(byte_level=True)


def tokenizer_from_manifest(record: dict) -> Tokenizer:
    """Restore a known tokenizer definition, never executable tokenizer code."""
    if not isinstance(record, dict):
        raise ValueError("tokenizer manifest must be a mapping")
    available = {tokenizer.manifest.version: tokenizer for tokenizer in (LEGACY, UTF8)}
    if not isinstance(record.get("version"), str) or record["version"] not in available:
        raise ValueError("unknown tokenizer manifest version")
    tokenizer = available[record["version"]]
    tokenizer.require_manifest(record)
    return tokenizer


@dataclass(frozen=True, slots=True)
class VocabularyMigration:
    """Explicit one-way tied-embedding migration; exact resume is forbidden."""

    source_tokenizer_sha256: str
    destination_tokenizer_sha256: str
    source_vocabulary_size: int
    destination_vocabulary_size: int
    source_to_destination: tuple[int, ...]
    seed: int
    initialization_std: float = 0.02
    version: str = "legacy-symbol-to-utf8-tied-embedding-v1"
    optimizer_policy: str = "reset-required"

    @property
    def identity(self) -> str:
        return codec.digest(asdict(self))

    def validate(self) -> None:
        LEGACY.require_binding(self.source_tokenizer_sha256, self.source_vocabulary_size)
        UTF8.require_binding(self.destination_tokenizer_sha256, self.destination_vocabulary_size)
        expected = (0, 1, 2, *(ord(char) + 4 for char in codec.ALPHABET))
        if (self.version != "legacy-symbol-to-utf8-tied-embedding-v1"
                or self.optimizer_policy != "reset-required"
                or self.source_to_destination != expected
                or any(type(value) is not int for value in self.source_to_destination)
                or type(self.seed) is not int or not 0 <= self.seed < 2**32
                or type(self.initialization_std) not in (int, float)
                or not math.isfinite(self.initialization_std) or self.initialization_std <= 0):
            raise ValueError("invalid vocabulary migration recipe")


def legacy_to_utf8_migration(*, seed: int) -> VocabularyMigration:
    recipe = VocabularyMigration(
        LEGACY.identity, UTF8.identity, LEGACY.vocabulary_size, UTF8.vocabulary_size,
        (0, 1, 2, *(ord(char) + 4 for char in codec.ALPHABET)), seed,
    )
    recipe.validate()
    return recipe


def migrate_tied_embedding(weight: Any, recipe: VocabularyMigration, *,
                           parent_model_sha256: str) -> tuple[Any, dict]:
    """Return a fresh table and migration receipt without mutating the parent.

    The caller must build a new model spec/vocab identity, install this one tied
    table, reset the optimizer and publish a *migration* lineage. This is not a
    whole-model loader and does not adapt legacy-symbol-specific copy heads.
    Copied rows are exact; enlarged-softmax predictions are not equivalent.
    """
    import torch

    recipe.validate()
    if not isinstance(parent_model_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", parent_model_sha256) is None:
        raise ValueError("parent model identity must be lowercase SHA-256")
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
            or weight.shape[0] != recipe.source_vocabulary_size or weight.shape[1] <= 0
            or not weight.is_floating_point() or not torch.isfinite(weight).all().item()):
        raise ValueError("migration requires a finite source-vocabulary floating embedding")
    generator = torch.Generator(device="cpu").manual_seed(recipe.seed)
    fresh = torch.empty((recipe.destination_vocabulary_size, weight.shape[1]), dtype=torch.float32)
    fresh.normal_(std=recipe.initialization_std, generator=generator)
    fresh = fresh.to(device=weight.device, dtype=weight.dtype)
    if not torch.isfinite(fresh).all().item():
        raise ValueError("initialization is not finite in destination dtype")
    with torch.no_grad():
        fresh[list(recipe.source_to_destination)] = weight.detach()
    from .backend import tensor_sha256
    embedding_sha = tensor_sha256(fresh)
    identity = codec.digest({"parent": parent_model_sha256, "migration": recipe.identity,
                             "embedding_sha256": embedding_sha,
                             "shape": list(fresh.shape), "dtype": str(fresh.dtype)})
    receipt = {"kind": "migration", "parent_model_sha256": parent_model_sha256,
               "model_sha256": identity, "recipe": asdict(recipe), "recipe_sha256": recipe.identity,
               "embedding_sha256": embedding_sha, "optimizer_policy": recipe.optimizer_policy,
               "prediction_equivalence": False}
    return fresh, receipt
