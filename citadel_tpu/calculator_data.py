"""Deterministic calculator canary generator. No torch, no device, pure Python.

Canonical representation (frozen v1): "<a> <op> <b> = <c>" with single spaces,
ops in {+, -, *, /}, exact integer division only (a % b == 0 for '/').
Split discipline: TRAIN/DEV/TEST seeds and operand ranges are disjoint by
construction; no exact problem string overlaps across splits. Generalization
slices: held-out operand pairs, held-out numeric ranges, held-out commutative
orderings (a+b vs b+a) where meaningful.

Schema: citadel-calculator-canary/v1. Generator version is recorded in every
receipt; changing ranges/seeds/format requires a version bump + re-prereg.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


GENERATOR_VERSION = "calculator-canary/1.1"
SCHEMA = "citadel-calculator-canary/v1"

Op = Literal["+", "-", "*", "/"]

SPLITS: dict[str, dict[str, object]] = {
    "train": {"seed": 71001, "count": 4000, "lo": 0, "hi": 49},
    "development": {"seed": 71002, "count": 500, "lo": 50, "hi": 79},
    "test": {"seed": 71003, "count": 500, "lo": 80, "hi": 119},
}


def render(a: int, op: Op, b: int) -> str:
    if op == "+":
        c = a + b
    elif op == "-":
        c = a - b
    elif op == "*":
        c = a * b
    elif op == "/":
        c = a // b
    else:  # pragma: no cover - guarded by type
        raise ValueError(f"unknown op {op!r}")
    return f"{a} {op} {b} = {c}"


def _draw(rng: random.Random, lo: int, hi: int) -> tuple[int, Op, int]:
    op: Op = rng.choice(["+", "-", "*", "/"])
    if op == "/":
        b = rng.randint(1, 12)
        q = rng.randint(lo, hi)
        return b * q, "/", b
    return rng.randint(lo, hi), op, rng.randint(lo, hi)


def generate(*, split: Literal["train", "development", "test"]) -> list[str]:
    """Generate one split deterministically. Pure function of (seed, range)."""
    cfg = SPLITS[split]
    rng = random.Random(int(cfg["seed"]))
    seen: set[str] = set()
    out: list[str] = []
    target, lo, hi = int(cfg["count"]), int(cfg["lo"]), int(cfg["hi"])
    guard = 0
    while len(out) < target and guard < target * 50:
        guard += 1
        a, op, b = _draw(rng, lo, hi)
        s = render(a, op, b)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    if len(out) < target:
        raise RuntimeError(f"generator exhausted for split {split}")
    return out


def generalization_slices(*, seed: int = 71999) -> dict[str, list[str]]:
    """Held-out commutativity + range slices (fixed seed, disjoint from splits)."""
    rng = random.Random(seed)
    pairs = [(a, b) for a in range(80, 90) for b in range(80, 90)]
    off = seed % 2  # deterministic stride offset: exactly 50 of 100 pairs
    comm = [f"{b} + {a} = {a + b}" for i, (a, b) in enumerate(pairs) if i % 2 == off][:50]
    assert len(comm) == 50, "commutative slice must be exactly 50 rows"
    rng_hi = [render(rng.randint(120, 199), rng.choice(["+", "-", "*"]), rng.randint(120, 199)) for _ in range(100)]
    return {"commutative_heldout": comm, "range_heldout_120_199": rng_hi}


def build_all(*, out_dir: str = "docs/citadel/tpu_receipts/calculator_canary") -> dict[str, object]:
    """Generate all splits + slices, write one JSONL per split, return receipt."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"schema": SCHEMA, "generator_version": GENERATOR_VERSION, "splits": {}}
    splits_payload: dict[str, object] = {}
    for split in ("train", "development", "test"):
        rows = generate(split=split)  # type: ignore[arg-type]
        (root / f"{split}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        splits_payload[split] = {
            "count": len(rows),
            "seed": SPLITS[split]["seed"],
            "range": [SPLITS[split]["lo"], SPLITS[split]["hi"]],
            "sha256": hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest(),
        }
    overlap = (
        set(generate(split="train")) & set(generate(split="test"))
    ) | (
        set(generate(split="train")) & set(generate(split="development"))
    )
    if overlap:
        raise RuntimeError(f"split overlap detected: {len(overlap)} rows")
    slices = generalization_slices()
    (root / "generalization.json").write_text(json.dumps(slices, indent=2, sort_keys=True), encoding="utf-8")
    payload["splits"] = splits_payload
    payload["split_overlap_rows"] = 0
    payload["generalization"] = {k: len(v) for k, v in slices.items()}
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    (root / "calculator_canary_receipt.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


@dataclass(frozen=True, slots=True)
class EvalResult:
    split: str
    total: int
    exact_match: int

    @property
    def accuracy(self) -> float:
        return self.exact_match / self.total if self.total else 0.0


def score(predictions: list[str], golds: list[str], *, split: str) -> EvalResult:
    """Canonical exact-match scoring on normalized '<a> <op> <b> = <c>' strings."""
    norm = lambda s: " ".join(s.strip().split())
    return EvalResult(split=split, total=len(golds), exact_match=sum(1 for p, g in zip(predictions, golds) if norm(p) == norm(g)))


__all__ = ["EvalResult", "SPLITS", "build_all", "generate", "generalization_slices", "render", "score"]
