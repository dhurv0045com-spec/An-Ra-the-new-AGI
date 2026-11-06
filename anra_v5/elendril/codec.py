"""Reversible character vocabulary for controlled symbolic tasks."""

from dataclasses import dataclass
from hashlib import sha256
import json


PAD, BOS, EOS = range(3)
ALPHABET = "!#-:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
VOCAB_SIZE = len(ALPHABET) + 3
TOKEN_IDS = {char: index for index, char in enumerate(ALPHABET, 3)}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return sha256(canonical(value)).hexdigest()


CODEC_SHA = digest(["elendril-symbol-v1", PAD, BOS, EOS, ALPHABET])


def encode(text: str) -> tuple[int, ...]:
    try:
        return tuple(TOKEN_IDS[char] for char in text)
    except KeyError as exc:
        raise ValueError(f"unsupported symbol {exc.args[0]!r}") from exc


def decode(ids: tuple[int, ...]) -> str:
    if any(type(token) is not int or not 3 <= token < VOCAB_SIZE for token in ids):
        raise ValueError("expected content tokens only")
    return "".join(ALPHABET[token - 3] for token in ids)


@dataclass(frozen=True, slots=True)
class Prediction:
    task_id: str
    tokens: tuple[int, ...]


def answer_tokens(answer: str) -> tuple[int, ...]:
    if not answer:
        raise ValueError("answer must contain a content symbol")
    return (*encode(answer), EOS)
