"""Minimal explicit normalization, versioned per transform (M6).

Allowed: UTF-8 decode repair, canonical newline handling, stripping of
clearly non-content transport wrappers. Forbidden by default: lowercasing,
punctuation removal, math/code rewriting, significant-whitespace changes.
Every applied transform is recorded on the document so the raw-to-processed
chain stays auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


NORMALIZE_VERSION = "v5-normalize/v1"

_WHITESPACE_RUN = re.compile(r"[ \t]+")


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    doc_id: str
    text: str
    applied: tuple[str, ...]
    normalize_version: str


def normalize_text(
    doc_id: str,
    text: str,
    *,
    strip_surrounding_whitespace: bool = True,
    collapse_inner_whitespace: bool = False,
    canonical_newlines: bool = True,
) -> NormalizedDocument:
    """Normalize with an explicit, recorded transform set."""

    if not isinstance(text, str):
        raise ValueError("normalization input must be text")
    applied: list[str] = []
    if canonical_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        applied.append("canonical_newlines")
    if strip_surrounding_whitespace:
        text = text.strip()
        applied.append("strip_surrounding_whitespace")
    if collapse_inner_whitespace:
        text = _WHITESPACE_RUN.sub(" ", text)
        applied.append("collapse_inner_whitespace")
    return NormalizedDocument(
        doc_id=doc_id,
        text=text,
        applied=tuple(applied),
        normalize_version=NORMALIZE_VERSION,
    )


__all__ = ["NORMALIZE_VERSION", "NormalizedDocument", "normalize_text"]
