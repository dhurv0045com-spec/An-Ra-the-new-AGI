"""Token accounting and mixture planning with explicit reuse (M16-M19).

Accounting distinguishes AVAILABLE UNIQUE TOKENS from TARGET CONSUMPTION
per source: documents, raw/processed bytes, raw tokens, unique-after-dedup
tokens, and TRAIN/DEV/SEALED/FRESH splits. The mixture planner takes
qualified inventory, a token budget, desired proportions, and reuse limits,
and returns FEASIBLE with a frozen mixture receipt or INFEASIBLE with the
blocking shortfall. Silent recycling is refused: every repeated exposure is
planned, capped, and recorded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping


ACCOUNTING_SCHEMA = "anra-v5-token-accounting/v1"
MIXTURE_SCHEMA = "anra-v5-mixture-plan/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_str(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceAccounting:
    source_id: str
    source_class: str
    documents: int
    raw_bytes: int
    processed_bytes: int
    raw_tokens: int
    unique_tokens: int
    train_tokens: int
    dev_tokens: int
    sealed_tokens: int
    fresh_tokens: int

    def assert_valid(self) -> None:
        if not self.source_id or not self.source_class:
            raise ValueError("source identity and class are required")
        for name in ("documents", "raw_bytes", "processed_bytes", "raw_tokens",
                     "unique_tokens", "train_tokens", "dev_tokens",
                     "sealed_tokens", "fresh_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"accounting count is negative: {name}")
        if self.unique_tokens > self.raw_tokens:
            raise ValueError("unique tokens cannot exceed raw tokens")
        splits = self.train_tokens + self.dev_tokens + self.sealed_tokens + self.fresh_tokens
        if splits != self.unique_tokens:
            raise ValueError("split tokens must equal unique tokens exactly")


def account_source(
    *,
    source_id: str,
    source_class: str,
    documents: list[tuple[str, str, str]],
    encode: Callable[[str], int],
    split_of: Mapping[str, str],
) -> SourceAccounting:
    """Account one source: byte/token counts with exact split attribution.

    ``documents`` carries ``(doc_id, raw_text, processed_text)``; ``encode``
    returns the token COUNT for processed text; ``split_of`` maps doc IDs to
    TRAIN/DEV/SEALED/FRESH. Unique counts treat each doc_id once (callers
    deduplicate first).
    """

    raw_bytes = 0
    processed_bytes = 0
    raw_tokens = 0
    per_split: dict[str, int] = {"TRAIN": 0, "DEV": 0, "SEALED": 0, "FRESH": 0}
    seen: set[str] = set()
    unique = 0
    for doc_id, raw_text, processed_text in documents:
        if doc_id in seen:
            raise ValueError(f"duplicate doc_id inside source accounting: {doc_id}")
        seen.add(doc_id)
        raw_bytes += len(raw_text.encode("utf-8"))
        processed_bytes += len(processed_text.encode("utf-8"))
        count = int(encode(processed_text))
        if count < 0:
            raise ValueError("token counts cannot be negative")
        raw_tokens += count
        unique += count
        split = split_of.get(doc_id)
        if split not in per_split:
            raise ValueError(f"document lacks a valid split: {doc_id}")
        per_split[split] += count
    accounting = SourceAccounting(
        source_id=source_id, source_class=source_class,
        documents=len(documents), raw_bytes=raw_bytes,
        processed_bytes=processed_bytes, raw_tokens=raw_tokens,
        unique_tokens=unique, train_tokens=per_split["TRAIN"],
        dev_tokens=per_split["DEV"], sealed_tokens=per_split["SEALED"],
        fresh_tokens=per_split["FRESH"],
    )
    accounting.assert_valid()
    return accounting


@dataclass(frozen=True, slots=True)
class MixturePlan:
    schema: str
    token_budget: int
    target_proportions: tuple[tuple[str, float], ...]
    planned_consumed: tuple[tuple[str, int], ...]
    expected_reuse: tuple[tuple[str, float], ...]
    max_reuse: float

    def assert_valid(self) -> None:
        if self.schema != MIXTURE_SCHEMA:
            raise ValueError("unsupported mixture-plan schema")
        if self.token_budget <= 0:
            raise ValueError("token budget must be positive")
        if abs(sum(fraction for _, fraction in self.target_proportions) - 1.0) > 1e-9:
            raise ValueError("target proportions must sum to one")
        if sum(count for _, count in self.planned_consumed) != self.token_budget:
            raise ValueError("planned consumption must equal the token budget")
        if self.max_reuse < 1.0:
            raise ValueError("maximum reuse cannot go below one pass")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha256_str(
            _canonical_json(
                {
                    "schema": self.schema,
                    "token_budget": self.token_budget,
                    "target_proportions": [list(item) for item in self.target_proportions],
                    "planned_consumed": [list(item) for item in self.planned_consumed],
                    "expected_reuse": [list(item) for item in self.expected_reuse],
                    "max_reuse": self.max_reuse,
                }
            )
        )


def plan_mixture(
    inventory: Mapping[str, int],
    *,
    token_budget: int,
    target_proportions: Mapping[str, float],
    max_reuse: float,
) -> dict[str, object]:
    """Return FEASIBLE with a frozen plan or INFEASIBLE with shortfalls."""

    if token_budget <= 0:
        raise ValueError("token budget must be positive")
    if abs(sum(target_proportions.values()) - 1.0) > 1e-9:
        raise ValueError("target proportions must sum to one")
    if max_reuse < 1.0:
        raise ValueError("maximum reuse cannot go below one pass")
    shortfalls: dict[str, dict[str, float]] = {}
    for source_class in sorted(target_proportions):
        want = token_budget * target_proportions[source_class]
        have = float(inventory.get(source_class, 0))
        capacity = have * max_reuse
        if want - capacity > 1e-9:
            shortfalls[source_class] = {
                "wanted": want, "unique_available": have,
                "max_reuse": max_reuse, "capacity": capacity,
            }
    if shortfalls:
        return {
            "schema": MIXTURE_SCHEMA,
            "feasible": False,
            "token_budget": token_budget,
            "shortfalls": shortfalls,
            "status": "INFEASIBLE",
        }
    from .mixture import allocate

    planned = allocate(token_budget, dict(target_proportions))
    for source_class, count in planned.items():
        capacity = float(inventory.get(source_class, 0)) * max_reuse
        if count - capacity > 1.0:
            shortfalls[source_class] = {
                "wanted": token_budget * target_proportions[source_class],
                "unique_available": float(inventory.get(source_class, 0)),
                "max_reuse": max_reuse,
                "capacity": capacity,
            }
    if shortfalls:
        return {
            "schema": MIXTURE_SCHEMA,
            "feasible": False,
            "token_budget": token_budget,
            "shortfalls": shortfalls,
            "status": "INFEASIBLE",
        }
    reuse = {
        source_class: (planned[source_class] / float(inventory.get(source_class, 0)))
        if float(inventory.get(source_class, 0)) > 0 else 0.0
        for source_class in planned
    }
    plan = MixturePlan(
        schema=MIXTURE_SCHEMA,
        token_budget=token_budget,
        target_proportions=tuple(sorted(target_proportions.items())),
        planned_consumed=tuple(sorted(planned.items())),
        expected_reuse=tuple(sorted(reuse.items())),
        max_reuse=max_reuse,
    )
    plan.assert_valid()
    return {
        "schema": MIXTURE_SCHEMA,
        "feasible": True,
        "plan_sha256": plan.sha256(),
        "token_budget": token_budget,
        "target_proportions": dict(plan.target_proportions),
        "planned_consumed": dict(plan.planned_consumed),
        "expected_reuse": dict(plan.expected_reuse),
        "status": "FEASIBLE",
    }


__all__ = [
    "ACCOUNTING_SCHEMA",
    "MIXTURE_SCHEMA",
    "MixturePlan",
    "SourceAccounting",
    "account_source",
    "plan_mixture",
]
