# ruff: noqa: E402
"""Week 1 tokenizer fertility measurement (MASTER_UPGRADE v2, Layer 3).

Measures per-source fertility of the live tokenizer against the V4 migration
gates and runs the append-only audit, producing the evidence the V4 decision
requires:

- English prose: tokens per whitespace word. V4 target <= 1.35; a reading
  >= 1.5 confirms the V3 compute tax the plan predicts.
- Code: tokens per character. V4 target <= 0.45.
- Math/DFC: tokens per character (informational; no gate yet).

English held-out text is selected by the same deterministic hash-split rule
as scripts/build_tokenizer_recovery.py (sha256(line)[0] < 13), so measured
lines are the ones the recovery pipeline would also hold out. Code fertility
is measured on this repository's own Python sources, which were never part of
the tokenizer's training corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anra.anra_paths import OUTPUT_V2_DIR, ROOT, V3_TOKENIZER_FILE
from tokenizer.subword_tokenizer import SubwordTokenizer
from tokenizer.validate_tokenizer_v3 import audit_token_fertility

ENGLISH_TOK_PER_WORD_GATE = 1.35
CODE_TOK_PER_CHAR_GATE = 0.45


def measure_fertility(tokenizer: SubwordTokenizer, text: str) -> dict[str, float | int]:
    if not text:
        raise ValueError("Fertility measurement requires non-empty text")
    ids = tokenizer.encode(text)
    words = text.split()
    unknown = sum(1 for token_id in ids if token_id == tokenizer.unk_token_id)
    return {
        "sampled_chars": len(text),
        "sampled_words": len(words),
        "tokens": len(ids),
        "tok_per_word": round(len(ids) / max(1, len(words)), 6),
        "tok_per_char": round(len(ids) / len(text), 6),
        "chars_per_token": round(len(text) / max(1, len(ids)), 6),
        "unk_rate": round(unknown / max(1, len(ids)), 8),
    }


def heldout_english(path: Path, *, max_chars: int) -> str:
    """Deterministic held-out slice: same hash rule as build_tokenizer_recovery."""
    lines: list[str] = []
    consumed = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if consumed >= max_chars:
                break
            if hashlib.sha256(line.encode("utf-8")).digest()[0] < 13:
                lines.append(line)
                consumed += len(line)
    return "".join(lines)


def repo_code_sample(root: Path, *, max_chars: int) -> str:
    """Deterministic sample of this repo's Python sources (unseen by V3 training)."""
    chunks: list[str] = []
    consumed = 0
    for directory in ("anra", "training", "tokenizer", "agents"):
        for path in sorted((root / directory).rglob("*.py")):
            if consumed >= max_chars:
                break
            text = path.read_text(encoding="utf-8", errors="replace")[: max_chars - consumed]
            chunks.append(text)
            consumed += len(text)
    return "\n".join(chunks)


def dfc_math_sample(path: Path, *, max_chars: int) -> str:
    chunks: list[str] = []
    consumed = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if consumed >= max_chars:
                break
            try:
                text = str(json.loads(line).get("text", ""))
            except json.JSONDecodeError:
                continue
            chunks.append(text)
            consumed += len(text)
    return "\n".join(chunks)


def build_report(
    tokenizer_json: Path,
    *,
    english_source: Path,
    dfc_source: Path | None,
    max_chars: int = 500_000,
    audit_sources: list[Path] | None = None,
    max_units: int = 1_000_000,
) -> dict[str, object]:
    tokenizer = SubwordTokenizer.load(tokenizer_json)
    sources: dict[str, dict[str, object]] = {}

    english = measure_fertility(tokenizer, heldout_english(english_source, max_chars=max_chars))
    english["gate"] = ENGLISH_TOK_PER_WORD_GATE
    english["gate_metric"] = "tok_per_word"
    english["gate_pass"] = bool(english["tok_per_word"] <= ENGLISH_TOK_PER_WORD_GATE)
    sources["english_prose"] = english

    code = measure_fertility(tokenizer, repo_code_sample(REPO_ROOT, max_chars=max_chars))
    code["gate"] = CODE_TOK_PER_CHAR_GATE
    code["gate_metric"] = "tok_per_char"
    code["gate_pass"] = bool(code["tok_per_char"] <= CODE_TOK_PER_CHAR_GATE)
    sources["code"] = code

    if dfc_source is not None and dfc_source.is_file():
        math = measure_fertility(tokenizer, dfc_math_sample(dfc_source, max_chars=max_chars))
        math["gate"] = None
        math["gate_metric"] = "tok_per_char"
        math["gate_pass"] = None
        sources["math_dfc"] = math

    audit_paths = [p for p in (audit_sources or [english_source, dfc_source]) if p and p.is_file()]
    audit = audit_token_fertility(tokenizer_json, audit_paths, max_units=max_units)
    audit.pop("candidate_tokens", None)

    v3_tax_confirmed = bool(sources["english_prose"]["tok_per_word"] >= 1.5)
    return {
        "schema_version": 1,
        "tokenizer": str(tokenizer_json),
        "backend": tokenizer.backend,
        "vocab_size": tokenizer.vocab_size,
        "sources": sources,
        "append_audit": audit,
        "v3_tax_confirmed": v3_tax_confirmed,
        "v4_migration_justified": v3_tax_confirmed
        or not (sources["english_prose"]["gate_pass"] and sources["code"]["gate_pass"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure per-source tokenizer fertility")
    parser.add_argument("--tokenizer", default=str(V3_TOKENIZER_FILE))
    parser.add_argument("--max-chars", type=int, default=500_000)
    parser.add_argument("--max-units", type=int, default=1_000_000)
    parser.add_argument("--output", default=str(OUTPUT_V2_DIR / "fertility_week1.json"))
    args = parser.parse_args()

    report = build_report(
        Path(args.tokenizer),
        english_source=ROOT / "training_data" / "anra_training.txt",
        dfc_source=ROOT / "training_data" / "frontier_dfc.jsonl",
        max_chars=args.max_chars,
        max_units=args.max_units,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
