"""Speculative decoding acceptance and promotion measurements."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeculativeBenchmark:
    proposed_tokens: int
    accepted_tokens: int
    baseline_seconds: float
    speculative_seconds: float

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(1, self.proposed_tokens)

    @property
    def speedup(self) -> float:
        return self.baseline_seconds / max(self.speculative_seconds, 1e-9)

    @property
    def promotion_allowed(self) -> bool:
        return self.acceptance_rate >= 0.30 and self.speedup >= 1.5


def accept_draft_prefix(draft_tokens: list[int], target_tokens: list[int]) -> int:
    accepted = 0
    for draft, target in zip(draft_tokens, target_tokens, strict=False):
        if draft != target:
            break
        accepted += 1
    return accepted
