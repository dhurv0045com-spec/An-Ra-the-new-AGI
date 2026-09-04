"""Transparent quality filtering and cheap domain tagging (M7/M8).

No magical quality score: every signal is computed and persisted, and each
document resolves to KEEP, DROP, or QUARANTINE with explicit reasons.
Domain tags use reliable cheap signals (code fences, math symbols, dialogue
roles) with recorded uncertainty; there is no giant classifier.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


QUALITY_VERSION = "v5-quality/v1"

_HTML_TAG = re.compile(r"</?(?:div|span|nav|header|footer|script|style|a|ul|li|table)[^>]*>", re.IGNORECASE)
_URL = re.compile(r"https?://\S+|www\.\S+")
_CODE_FENCE = re.compile(r"```|~~~")
_MATH_SYMBOL = re.compile(r"[∑∫∂√∞≈≠≤≥±×÷α-ωΑ-Ω]|\\(?:frac|sum|int|sqrt|alpha|beta|theta|lambda|pi)\b")
_DIALOGUE_ROLE = re.compile(r"^(USER|ASSISTANT|HUMAN|SYSTEM)\s*:", re.MULTILINE)


def _signals(text: str) -> dict[str, float]:
    length = max(len(text), 1)
    letters = sum(1 for c in text if c.isalpha())
    alpha_density = letters / length
    alnum_space = sum(1 for c in text if c.isalnum() or c.isspace())
    printable_ratio = alnum_space / length
    lines = text.split("\n")
    nonempty = [line for line in lines if line.strip()]
    line_counts: dict[str, int] = {}
    for line in nonempty:
        line_counts[line.strip()] = line_counts.get(line.strip(), 0) + 1
    repeated = sum(count - 1 for count in line_counts.values() if count > 1)
    line_repetition = repeated / max(len(nonempty), 1)
    longest_run = 1
    run = 1
    for first, second in zip(text, text[1:]):
        if second == first:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 1
    char_repetition = longest_run / length
    words = re.findall(r"[A-Za-z]+", text)
    entropy = 0.0
    if words:
        total = len(words)
        frequencies: dict[str, int] = {}
        for word in words:
            key = word[:12]
            frequencies[key] = frequencies.get(key, 0) + 1
        entropy = -sum((count / total) * math.log2(count / total) for count in frequencies.values())
    return {
        "length": float(len(text)),
        "alpha_density": alpha_density,
        "printable_ratio": printable_ratio,
        "line_repetition": line_repetition,
        "char_repetition": char_repetition,
        "word_entropy": entropy,
        "html_tags": float(len(_HTML_TAG.findall(text))),
        "urls": float(len(_URL.findall(text))),
        "code_fences": float(len(_CODE_FENCE.findall(text))),
        "math_symbols": float(len(_MATH_SYMBOL.findall(text))),
        "dialogue_roles": float(len(_DIALOGUE_ROLE.findall(text))),
        "nonempty_lines": float(len(nonempty)),
    }


@dataclass(frozen=True, slots=True)
class QualityVerdict:
    doc_id: str
    decision: str
    reasons: tuple[str, ...]
    signals: tuple[tuple[str, float], ...]
    domain: str
    domain_uncertain: bool
    quality_version: str


def judge(doc_id: str, text: str) -> QualityVerdict:
    """Compute signals, tag domain, and resolve KEEP/DROP/QUARANTINE."""

    signals = _signals(text)
    reasons: list[str] = []
    if len(text) < 200:
        reasons.append("very_short_fragment")
    if signals["printable_ratio"] < 0.80:
        reasons.append("binary_or_garbled")
    if signals["alpha_density"] < 0.30:
        reasons.append("very_low_information_density")
    if signals["char_repetition"] > 0.05:
        reasons.append("extreme_character_repetition")
    if signals["line_repetition"] > 0.50 and signals["nonempty_lines"] > 4:
        reasons.append("pathological_line_repetition")
    if signals["html_tags"] > 10:
        reasons.append("html_boilerplate")
    if signals["urls"] > 10:
        reasons.append("very_high_url_density")
    if signals["word_entropy"] < 2.0 and len(text) > 500:
        reasons.append("machine_generated_garbage_indicator")
    domain, uncertain = _domain(text, signals)
    if len(reasons) >= 3:
        decision = "DROP"
    elif reasons:
        decision = "QUARANTINE"
    else:
        decision = "KEEP"
    return QualityVerdict(
        doc_id=doc_id,
        decision=decision,
        reasons=tuple(reasons),
        signals=tuple(sorted(signals.items())),
        domain=domain,
        domain_uncertain=uncertain,
        quality_version=QUALITY_VERSION,
    )


def _domain(text: str, signals: dict[str, float]) -> tuple[str, bool]:
    if signals["code_fences"] >= 1:
        return "code", False
    stripped = text.lstrip()
    if stripped.startswith(("def ", "class ", "import ", "from ", "#include", "function ", "const ")):
        return "code", False
    if signals["math_symbols"] >= 3:
        return "math", False
    if signals["dialogue_roles"] >= 2:
        return "dialogue", False
    if signals["alpha_density"] >= 0.55:
        return "general_prose", True
    return "general_prose", True


__all__ = ["QUALITY_VERSION", "QualityVerdict", "judge"]
