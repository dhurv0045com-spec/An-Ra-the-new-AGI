"""Small committed canaries; serious E1 corpora remain external and hash-bound."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizerProbe:
    probe_id: str
    domain: str
    text: str
    tags: tuple[str, ...]


PROBES = (
    TokenizerProbe("natural-01", "natural", "A quiet river crossed the old stone bridge.", ("english",)),
    TokenizerProbe("unicode-01", "natural", "नदी के पास एक छोटा प्रयोग हुआ।", ("unicode", "hindi")),
    TokenizerProbe("unicode-02", "natural", "Δx = 3.2×10⁻⁷; café; 東京; 🧪", ("unicode", "scientific")),
    TokenizerProbe("code-01", "code", "def bind(entity_id: str) -> tuple[int, str]:", ("identifier", "python")),
    TokenizerProbe("code-02", "code", "checkpoint.global_step += 1", ("identifier", "punctuation")),
    TokenizerProbe("number-01", "math", "-0.00314159 20260829 1,048,576 6.022e23", ("numbers",)),
    TokenizerProbe("formal-01", "formal", "∀x∈S: P(x) ⇒ ∃y R(x,y)", ("symbols",)),
    TokenizerProbe("nonce-01", "cognition", "dv-vek-7-419 maps to DV8042-G.", ("nonce", "binding")),
    TokenizerProbe("nonce-02", "cognition", "FR9921-C|FR1048-A", ("nonce", "composition")),
    TokenizerProbe("legal-01", "legal", "Clause 4(b)(iii) supersedes Annex A-17 only after countersignature.", ("structured",)),
    TokenizerProbe("spacing-01", "code", "x\t:=\tvalue\nnext_line()", ("whitespace", "roundtrip")),
    TokenizerProbe("answer-01", "cognition", "Context value: ZX-4819.\nAnswer: ZX-4819", ("context-answer-consistency",)),
)


def probe_map() -> dict[str, TokenizerProbe]:
    return {probe.probe_id: probe for probe in PROBES}
