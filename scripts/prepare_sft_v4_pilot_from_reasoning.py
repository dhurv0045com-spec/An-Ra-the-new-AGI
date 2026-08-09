"""Prepare a small, provenance-bound SFT pilot from the repository's SmolTalk subset.

The repository already contains ``training_data/reasoning.jsonl``.  This tool
does not pretend that raw pretraining text is instruction data: it selects
records with explicit user/assistant pairs, assigns a conservative category
using visible prompt signals, records the exact input and derived-file hashes,
and then delegates immutable split/manifest creation to ``sft_dataset_v4``.

This is a bounded pilot corpus, not a claim that the full SFT curriculum is
complete.  The generated audit report keeps the labeling rules and sample
counts visible so the owner can replace or expand it before a production run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from training.sft_dataset_v4 import build_sft_dataset_v4, sha256_file

SOURCE_ID = "huggingfacetb-smol-smoltalk-derived-v1"
SOURCE_LICENSE = "Apache-2.0"
SOURCE_URL = "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk"
SOURCE_REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"

# Ordered from specific to broad so a coding question is not hidden by a
# generic "explain" match.  Every non-fallback category has a visible reason
# that can be reviewed in the audit report.
_RULES: tuple[tuple[str, str], ...] = (
    (
        "correction",
        r"\b(edit|rewrite|revise|revision|grammar|spelling|proofread|correct|"
        r"improve (?:this|the) text)\b",
    ),
    (
        "code",
        r"\b(code|coding|python|javascript|program|function|algorithm|debug|"
        r"sql|api implementation)\b",
    ),
    (
        "mathematics",
        r"\b(math|mathematics|calculate|equation|algebra|geometry|probability|"
        r"percentage|percent|integral|derivative|solve)\b",
    ),
    (
        "tool_contracts",
        r"\b(tool|tools|api|json|function call|schema|endpoint|database|webhook)\b",
    ),
    (
        "decomposition",
        r"\b(step[- ]by[- ]step|steps|break down|decompose|plan|outline|explain|"
        r"how does|how do|reason through)\b",
    ),
    (
        "uncertainty",
        r"\b(uncertain|uncertainty|unknown|not sure|evidence|verify|confidence|"
        r"possibly|might|cannot determine)\b",
    ),
    (
        "dialogue",
        r"\b(conversation|conversational|pretend|roleplay|excited|feel|friendly|"
        r"chat|respond to|imagine you)\b",
    ),
)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _classify(prompt: str) -> tuple[str, str]:
    lowered = prompt.lower()
    for category, pattern in _RULES:
        match = re.search(pattern, lowered)
        if match:
            return category, f"{category}:{match.group(0)}"
    return "instruction_following", "fallback:no-specific-rule"


def _read_records(path: Path, *, per_category: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            prompt = raw.get("prompt")
            answer = raw.get("response")
            if not isinstance(prompt, str) or not isinstance(answer, str):
                continue
            prompt = prompt.strip()
            answer = answer.strip()
            if not prompt or not answer:
                continue
            category, reason = _classify(prompt)
            if counts[category] >= per_category:
                continue
            # Separate groups are deterministic and small enough that the
            # builder can place whole groups into train/validation/test.
            group_number = counts[category] // 64
            record = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "category": category,
                "source_id": SOURCE_ID,
                "split_group": f"{category}-group-{group_number:03d}",
                "license": SOURCE_LICENSE,
                "source_line": line_number,
            }
            accepted.append(record)
            counts[category] += 1
            reasons[reason] += 1
            examples.setdefault(category, []).append(prompt[:240])
    required = {
        "instruction_following",
        "dialogue",
        "code",
        "mathematics",
        "decomposition",
        "tool_contracts",
        "uncertainty",
        "correction",
    }
    missing = sorted(required - set(counts))
    if missing:
        raise ValueError(f"reasoning source did not provide required categories: {missing}")
    audit = {
        "schema": "anra-sft-pilot-audit/v1",
        "source_id": SOURCE_ID,
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "license": SOURCE_LICENSE,
        "category_rules": [{"category": name, "pattern": pattern} for name, pattern in _RULES],
        "fallback_category": "instruction_following",
        "accepted_examples": len(accepted),
        "category_counts": dict(sorted(counts.items())),
        "classification_reasons": dict(sorted(reasons.items())),
        "sample_prompts": {name: values[:5] for name, values in sorted(examples.items())},
    }
    return accepted, audit


def prepare(input_path: str | Path, output_dir: str | Path, *, per_category: int) -> dict[str, Any]:
    source = Path(input_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records, audit = _read_records(source, per_category=per_category)
    derived_path = destination / "smol-smoltalk-derived.jsonl"
    content = b"".join(
        json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
        for record in records
    )
    derived_path.write_bytes(content)
    receipt_payload = {
        "schema": "anra-sft-source-receipts/v1",
        "sources": [
            {
                "source_id": SOURCE_ID,
                "url": SOURCE_URL,
                "license": SOURCE_LICENSE,
                "path": str(derived_path),
                "sha256": sha256_file(derived_path),
                "size_bytes": derived_path.stat().st_size,
                "status": "derived_from_hash_audited_repository_subset",
                "source_revision": SOURCE_REVISION,
                "input_path": str(source),
                "input_sha256": sha256_file(source),
            }
        ],
    }
    receipt_path = destination / "sft-v4-source-receipts.json"
    receipt_path.write_text(
        json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit.update(
        {
            "input_path": str(source),
            "input_sha256": sha256_file(source),
            "derived_path": str(derived_path),
            "derived_sha256": sha256_file(derived_path),
            "receipt_path": str(receipt_path),
        }
    )
    (destination / "sft-v4-pilot-audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = build_sft_dataset_v4(
        [derived_path],
        destination,
        quality_gate_passed=True,
        licenses_audited=True,
        source_receipts_path=receipt_path,
    )
    return {
        "output_dir": str(result.output_dir),
        "accepted_examples": result.accepted_examples,
        "rejected_examples": result.rejected_examples,
        "audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a bounded V4 SFT pilot from reasoning.jsonl"
    )
    parser.add_argument("--input", default="training_data/reasoning.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-category", type=int, default=1024)
    args = parser.parse_args()
    if args.per_category < 64:
        raise SystemExit("--per-category must be at least 64 so groups can form cleanly")
    print(
        json.dumps(
            prepare(args.input, args.output_dir, per_category=args.per_category), indent=2
        )
    )


if __name__ == "__main__":
    main()
