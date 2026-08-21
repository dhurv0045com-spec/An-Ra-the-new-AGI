"""Build a conservative, balanced V4 SFT corpus for the 181M model.

The input is the pinned Apache-2.0 Smol-SmolTalk subset intended for small
models.  This program keeps only the newly generated Apache-2.0 components,
rejects malformed/collapsed/oversized conversations, assigns reviewable
capability labels, and delegates immutable split/manifest creation to the
canonical V4 dataset builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.posttraining_contract import REQUIRED_SFT_CATEGORIES
from training.sft_dataset_v4 import build_sft_dataset_v4, sha256_file

SOURCE_ID = "huggingfacetb-smol-smoltalk-small-model-apache-v1"
SOURCE_REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
SOURCE_FILE_SHA256 = "be6773dcce145f3918ff14237b1f765affa427b0b13f6a02d397e665ac908b9a"
SOURCE_URL = (
    "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/"
    f"{SOURCE_REVISION}/data/test-00000-of-00001.parquet"
)
LICENSE = "Apache-2.0"
ALLOWED_COMPONENTS = frozenset(
    {
        "smol-magpie-ultra-short",
        "smol-contraints",
        "smollm-rewrite-30k",
        "smol-summarize-20k",
        "smol-summarize-5k",
    }
)

_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("correction", r"\b(rewrite|revise|edit|proofread|correct|improve the (?:text|writing))\b"),
    (
        "tool_contracts",
        r"\b(json|api|schema|function call|tool call|endpoint|request body|"
        r"response format|yaml|xml|output format|return only|structured output)\b|"
        r"\brespond (?:with|in)\b|\bformat (?:the|your|as)\b",
    ),
    (
        "code",
        r"\b(python|javascript|typescript|java|c\+\+|code|function|class|sql|debug|algorithm)\b",
    ),
    (
        "mathematics",
        r"\b(calculate|compute|equation|algebra|geometry|probability|percentage|"
        r"integer|fraction|solve for)\b|\d\s*[+*/=]\s*\d",
    ),
    (
        "uncertainty",
        r"\b(uncertain|unknown|insufficient information|cannot determine|verify|"
        r"evidence|confidence|ambiguous|not enough information|if you do not know|"
        r"fact[- ]check)\b",
    ),
    (
        "decomposition",
        r"\b(step[- ]by[- ]step|break (?:it )?down|plan|outline|procedure|workflow|"
        r"explain how|reason through)\b",
    ),
    (
        "dialogue",
        r"\b(conversation|dialogue|roleplay|respond politely|friendly reply|chat|"
        r"what would you say|customer)\b",
    ),
)

_BAD_PHRASES = (
    "as an ai language model",
    "i cannot fulfill this request",
    "i'm unable to assist with that",
    "lorem ipsum",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _repetition_ratio(text: str, n: int = 4) -> float:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def _classify(prompt: str, component: str) -> tuple[str, str]:
    lowered = prompt.lower()
    if component == "smollm-rewrite-30k":
        return "correction", "component:rewrite"
    for category, pattern in _CATEGORY_RULES:
        match = re.search(pattern, lowered)
        if match:
            return category, f"rule:{match.group(0)[:80]}"
    if component.startswith("smol-summarize"):
        return "decomposition", "component:summarize"
    if component == "smol-contraints":
        return "instruction_following", "component:constraints"
    return "instruction_following", "fallback:general-instruction"


def _normalize_messages(raw: object) -> tuple[list[dict[str, str]] | None, str]:
    if not isinstance(raw, list) or not 2 <= len(raw) <= 10:
        return None, "message_count"
    messages: list[dict[str, str]] = []
    last_role = ""
    for item in raw:
        if not isinstance(item, dict):
            return None, "message_shape"
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).replace("\r\n", "\n").replace("\r", "\n").strip()
        if role not in {"system", "user", "assistant"} or not content:
            return None, "role_or_empty"
        if role == last_role and role != "system":
            return None, "nonalternating_roles"
        if any(ord(char) < 32 and char not in {"\n", "\t"} for char in content):
            return None, "control_character"
        messages.append({"role": role, "content": content})
        last_role = role
    if messages[-1]["role"] != "assistant" or not any(
        message["role"] == "user" for message in messages[:-1]
    ):
        return None, "conversation_boundary"
    prompt = next(message["content"] for message in messages if message["role"] == "user")
    answer = messages[-1]["content"]
    total_chars = sum(len(message["content"]) for message in messages)
    if not 8 <= len(prompt) <= 2_500 or not 24 <= len(answer) <= 6_000:
        return None, "turn_length"
    if total_chars > 9_000:
        return None, "context_length"
    lowered_answer = answer.lower()
    if any(phrase in lowered_answer for phrase in _BAD_PHRASES):
        return None, "generic_refusal"
    if _repetition_ratio(answer) > 0.18:
        return None, "answer_repetition"
    if answer.count("```") % 2:
        return None, "unclosed_code_fence"
    if len(set(re.findall(r"\w+", answer.lower()))) < min(8, max(3, len(answer.split()) // 4)):
        return None, "low_lexical_diversity"
    return messages, "accepted"


def prepare(
    source_path: str | Path, output_dir: str | Path, *, per_category: int
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Install pyarrow only for dataset preparation: pip install pyarrow"
        ) from error

    source = Path(source_path).resolve()
    destination = Path(output_dir).resolve()
    if sha256_file(source) != SOURCE_FILE_SHA256:
        raise ValueError("Smol-SmolTalk source SHA-256 does not match the pinned reviewed file")
    table = parquet.read_table(source, columns=["messages", "source"])
    candidates: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    components: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    seen: set[str] = set()
    for row_number, row in enumerate(table.to_pylist(), start=1):
        component = str(row.get("source", ""))
        if component not in ALLOWED_COMPONENTS:
            rejected["unreviewed_component_license"] += 1
            continue
        messages, disposition = _normalize_messages(row.get("messages"))
        if messages is None:
            rejected[disposition] += 1
            continue
        identity = hashlib.sha256(_canonical(messages)).hexdigest()
        if identity in seen:
            rejected["duplicate_conversation"] += 1
            continue
        seen.add(identity)
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        category, reason = _classify(prompt, component)
        record = {
            "messages": messages,
            "category": category,
            "source_id": SOURCE_ID,
            "license": LICENSE,
            "split_group": f"{category}-{component}-{row_number // 32:05d}",
            "upstream_component": component,
            "upstream_row": row_number,
        }
        candidates[category].append((identity, record))
        components[component] += 1
        reasons[f"{category}:{reason}"] += 1

    selected: list[dict[str, Any]] = []
    shortfalls: dict[str, int] = {}
    for category in REQUIRED_SFT_CATEGORIES:
        ordered = sorted(candidates.get(category, []), key=lambda item: item[0])
        keep = ordered[:per_category]
        selected.extend(record for _, record in keep)
        if len(keep) < per_category:
            shortfalls[category] = per_category - len(keep)
    # Rare but important capabilities are retained without duplicating examples.
    # Thirty-two distinct examples is the fail-closed floor; shortfalls remain
    # visible in the signed audit instead of being hidden by oversampling.
    minimum = 32
    below_minimum = {
        category: len(candidates.get(category, []))
        for category in REQUIRED_SFT_CATEGORIES
        if len(candidates.get(category, [])) < minimum
    }
    if below_minimum:
        raise ValueError(f"source lacks minimum high-quality category coverage: {below_minimum}")

    destination.mkdir(parents=True, exist_ok=True)
    derived = destination / "sft-v4-high-quality-source.jsonl"
    derived.write_bytes(b"".join(_canonical(record) + b"\n" for record in selected))
    receipt = destination / "sft-v4-source-receipts.json"
    receipt_payload = {
        "schema": "anra-sft-source-receipts/v1",
        "sources": [
            {
                "source_id": SOURCE_ID,
                "url": SOURCE_URL,
                "license": LICENSE,
                "path": str(derived),
                "sha256": sha256_file(derived),
                "size_bytes": derived.stat().st_size,
                "status": "derived_from_pinned_hash_verified_small_model_subset",
                "source_revision": SOURCE_REVISION,
                "input_sha256": SOURCE_FILE_SHA256,
                "allowed_components": sorted(ALLOWED_COMPONENTS),
            }
        ],
    }
    receipt.write_text(
        json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    build = build_sft_dataset_v4(
        [derived],
        destination,
        quality_gate_passed=True,
        licenses_audited=True,
        source_receipts_path=receipt,
    )
    report = {
        "schema": "anra-sft-quality-audit/v2",
        "purpose": "bounded high-quality SFT candidate for An-Ra V4 181M",
        "source": {
            "id": SOURCE_ID,
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "sha256": SOURCE_FILE_SHA256,
            "license": LICENSE,
            "allowed_components": sorted(ALLOWED_COMPONENTS),
        },
        "selected_examples": len(selected),
        "selected_category_counts": dict(sorted(Counter(r["category"] for r in selected).items())),
        "available_category_counts": {name: len(rows) for name, rows in sorted(candidates.items())},
        "category_shortfalls_against_target": shortfalls,
        "upstream_component_counts_before_balance": dict(sorted(components.items())),
        "classification_evidence_counts": dict(sorted(reasons.items())),
        "rejections": dict(sorted(rejected.items())),
        "derived_sha256": sha256_file(derived),
        "accepted_splits": build.accepted_examples,
        "promotion_status": "pilot_required; quality is not claimed until held-out behavior passes",
    }
    report_path = destination / "sft-v4-quality-audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="pinned Smol-SmolTalk parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-category", type=int, default=1_000)
    args = parser.parse_args()
    if not 64 <= args.per_category <= 4_000:
        raise SystemExit("--per-category must be in [64, 4000]")
    print(
        json.dumps(prepare(args.source, args.output_dir, per_category=args.per_category), indent=2)
    )


if __name__ == "__main__":
    main()
