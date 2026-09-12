"""Public event codec (B02): deterministic byte serialization for the learner.

One byte codec serves language rows and goal-conditioned trajectories. Public
payloads are canonicalized (sorted keys, explicit role tag, fixed end-of-event
marker), so key insertion order cannot change tokens while meaningful event or
list order does. Provenance never enters this module: callers keep task IDs,
seeds, splits, generator hashes and collection-policy identities in sidecars.

Physical vocabulary: 256 byte values plus four declared structural tokens,
matching ``bramastra_lab.research.config.BYTE_TOKENIZER_SPECIALS``.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from bramastra_lab.research.contracts.core import content_identity

SPECIAL_PAD = 256
SPECIAL_EOS = 257
SPECIAL_END_OF_EVENT = 258
SPECIAL_BOUNDARY = 259

BYTE_TOKENS = 256
STRUCTURAL_TOKENS = frozenset({SPECIAL_PAD, SPECIAL_EOS, SPECIAL_END_OF_EVENT, SPECIAL_BOUNDARY})

DEFAULT_MAX_EVENT_BYTES = 512

EVENT_ROLES = frozenset({
    "goal", "observation", "action", "feedback", "budget", "instruction", "candidate",
})

CODEC_IDENTITY_PAYLOAD = {
    "name": "bramastra-public-byte-codec/v1",
    "byte_tokens": BYTE_TOKENS,
    "specials": {"pad": SPECIAL_PAD, "eos": SPECIAL_EOS,
                 "end_of_event": SPECIAL_END_OF_EVENT, "boundary": SPECIAL_BOUNDARY},
    "event_serialization": "role + ':' + canonical-json(content) + end_of_event",
    "canonicalization": "utf-8, sorted keys, separators (',',':'), allow_nan=False",
}


class EncodingError(ValueError):
    """A public payload cannot be encoded under the declared codec rules."""


def codec_identity() -> str:
    """Stable identity of the codec; recorded in manifests and checkpoints."""
    return content_identity(CODEC_IDENTITY_PAYLOAD)


def canonical_event_bytes(content: Any) -> bytes:
    """Canonical JSON bytes with sorted keys and finite numbers only."""
    try:
        return json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EncodingError(f"content is not canonically serializable: {exc}") from exc


def encode_text(text: str) -> list[int]:
    """Encode raw UTF-8 text as byte tokens (no structural markers)."""
    if not isinstance(text, str):
        raise EncodingError("text must be a str")
    return list(text.encode("utf-8"))


def encode_event(role: str, content: Any, *, max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES) -> list[int]:
    """Encode one event: role tag, canonical content bytes, end-of-event marker.

    The role must be one of the declared public roles. Equivalent key order
    produces identical tokens; changing the content or the role changes them.
    Oversized events raise instead of truncating.
    """
    if not isinstance(role, str) or role not in EVENT_ROLES:
        raise EncodingError(f"event role must be one of {sorted(EVENT_ROLES)}")
    body = canonical_event_bytes(content)
    if len(body) > max_event_bytes:
        raise EncodingError(
            f"event content is {len(body)} bytes; max_event_bytes is {max_event_bytes} "
            "(oversized events are errors, not silent truncations)")
    payload = role.encode("utf-8") + b":" + body
    if len(payload) + 1 > max_event_bytes:
        raise EncodingError(
            f"event with role overhead exceeds max_event_bytes={max_event_bytes}")
    return list(payload) + [SPECIAL_END_OF_EVENT]


def decode_to_bytes(tokens: list[int]) -> bytes:
    """Inverse of the byte layer for verification; structural tokens are
    skipped, out-of-range tokens reject."""
    out = bytearray()
    for token in tokens:
        if not isinstance(token, int) or isinstance(token, bool):
            raise EncodingError("tokens must be integers")
        if token in STRUCTURAL_TOKENS:
            continue
        if not 0 <= token < BYTE_TOKENS:
            raise EncodingError(f"token {token} is outside the byte vocabulary")
        out.append(token)
    return bytes(out)


def token_identity(tokens: list[int]) -> str:
    """Content identity of a token sequence (dtype-free, order-sensitive)."""
    return content_identity({"codec": codec_identity(), "tokens": list(tokens)})
