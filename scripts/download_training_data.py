#!/usr/bin/env python3
"""Download and assemble An-Ra training data buckets."""

from __future__ import annotations

# Direct execution bootstraps repository imports after resolving REPO_ROOT.
# ruff: noqa: E402
import argparse
import ast
import glob
import hashlib
import json
import re
import shutil
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anra.anra_paths import DATA_MANIFEST_DIR, TOKEN_INVENTORY_MANIFEST
from training.data_ledger import DataQuality
from training.data_pipeline_v3 import SourceRecord, TokenShardPublisher

TRAINING_DATA_DIR = Path("training_data")
DOWNLOAD_STATUS = DATA_MANIFEST_DIR / "download_status.json"
DATA_PROFILES = {
    "smoke": {
        "target_gb": 0.02,
        "fineweb_docs": 2_000,
        "redpajama_docs": 1_000,
        "reasoning_per_source": 1_000,
        "science_per_source": 500,
    },
    "15gb": {
        "target_gb": 15.0,
        "fineweb_docs": 1_200_000,
        "redpajama_docs": 0,
        "reasoning_per_source": 120_000,
        "science_per_source": 60_000,
    },
    "30gb": {
        "target_gb": 30.0,
        "fineweb_docs": 2_400_000,
        "redpajama_docs": 0,
        "reasoning_per_source": 240_000,
        "science_per_source": 120_000,
    },
    "t4-15gb": {
        "fineweb_docs": 120_000,
        "redpajama_docs": 40_000,
        "reasoning_per_source": 20_000,
        "science_per_source": 10_000,
    },
    "t4-cached": {
        "fineweb_docs": 100_000,
        "redpajama_docs": 20_000,
        "reasoning_per_source": 8_000,
        "science_per_source": 4_000,
    },
    "tpu": {
        "fineweb_docs": 120_000,
        "redpajama_docs": 40_000,
        "reasoning_per_source": 20_000,
        "science_per_source": 10_000,
    },
    "full": {
        "fineweb_docs": 1_000_000,
        "redpajama_docs": 800_000,
        "reasoning_per_source": None,
        "science_per_source": None,
    },
}

_PII_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"\b(?:\+?\d[\d .()-]{8,}\d)\b"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
)

_COMMON_ENGLISH_WORDS = {
    "a",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}


class MinHashDeduplicator:
    """Bounded-memory LSH index for near-duplicate document rejection."""

    _PRIME = (1 << 61) - 1
    _PERMUTATIONS = (
        (3, 17),
        (5, 29),
        (11, 41),
        (17, 53),
        (23, 71),
        (31, 89),
        (43, 107),
        (59, 131),
    )

    def __init__(self, threshold: float = 0.80, max_entries: int = 500_000) -> None:
        self.threshold = float(threshold)
        self.max_entries = max(1, int(max_entries))
        self._signatures: list[tuple[int, ...]] = []
        self._bands: dict[tuple[int, tuple[int, ...]], list[int]] = {}

    @classmethod
    def signature(cls, text: str) -> tuple[int, ...]:
        words = re.findall(r"[a-z0-9_]+", text.lower())
        if not words:
            return tuple(0 for _ in cls._PERMUTATIONS)
        width = min(5, len(words))
        shingles = {
            " ".join(words[index : index + width])
            for index in range(max(1, len(words) - width + 1))
        }
        hashes = sorted(
            int.from_bytes(
                hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            % cls._PRIME
            for shingle in shingles
        )
        if len(hashes) > 512:
            step = len(hashes) / 512
            hashes = [hashes[int(index * step)] for index in range(512)]
        return tuple(
            min((coefficient * value + offset) % cls._PRIME for value in hashes)
            for coefficient, offset in cls._PERMUTATIONS
        )

    def seen_or_add(self, text: str) -> bool:
        signature = self.signature(text)
        candidates: set[int] = set()
        for band in range(4):
            key = (band, signature[band * 2 : band * 2 + 2])
            candidates.update(self._bands.get(key, ()))
        for candidate in candidates:
            prior = self._signatures[candidate]
            similarity = sum(a == b for a, b in zip(signature, prior, strict=True)) / len(signature)
            if similarity >= self.threshold:
                return True
        if len(self._signatures) >= self.max_entries:
            return False
        index = len(self._signatures)
        self._signatures.append(signature)
        for band in range(4):
            key = (band, signature[band * 2 : band * 2 + 2])
            self._bands.setdefault(key, []).append(index)
        return False


def _detect_content_language(text: str, *, source: str, hint: str = "") -> str:
    source_lower = source.lower()
    hint_lower = hint.strip().lower()
    if "stack" in source_lower:
        return f"code:{hint_lower or 'unknown'}"
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return "math" if "math" in source_lower else "unknown"
    letters = [character for character in text if character.isalpha()]
    ascii_ratio = sum(character.isascii() for character in letters) / max(1, len(letters))
    common = sum(word.lower() in _COMMON_ENGLISH_WORDS for word in words)
    if ascii_ratio >= 0.90 and (common >= 2 or len(words) < 20):
        return "en"
    return "unknown"


def _code_syntax_valid(text: str, *, source: str, language_hint: str = "") -> bool:
    if "stack" not in source.lower():
        return True
    language = language_hint.strip().lower()
    if language not in {"python", "py"}:
        return bool(re.search(r"[A-Za-z_][A-Za-z0-9_]*", text))
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def _safe_arithmetic_value(expression: str) -> float:
    operators = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.Pow: lambda left, right: left**right,
        ast.USub: lambda value: -value,
        ast.UAdd: lambda value: value,
    }

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return float(operators[type(node.op)](evaluate(node.left), evaluate(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return float(operators[type(node.op)](evaluate(node.operand)))
        raise ValueError("unsupported arithmetic expression")

    return evaluate(ast.parse(expression, mode="eval"))


def _math_text_valid(text: str, *, source: str) -> bool:
    if "math" not in source.lower():
        return True
    if any(text.count(left) != text.count(right) for left, right in (("(", ")"), ("[", "]"))):
        return False
    final_equations = re.findall(
        r"(?:final answer|answer)\s*[:=-]\s*([0-9 .+*/()^-]+)\s*=\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    for expression, expected in final_equations:
        try:
            actual = _safe_arithmetic_value(expression.replace("^", "**"))
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            return False
        if abs(actual - float(expected)) > 1e-8:
            return False
    return True


def _clean_foundation_text(text: str) -> str:
    cleaned = text.replace("\x00", " ").strip()
    for pattern in _PII_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    if len(cleaned) < 200:
        return ""
    printable_ratio = sum(character.isprintable() for character in cleaned) / len(cleaned)
    if printable_ratio < 0.98:
        return ""
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) >= 8 and len(set(lines)) / len(lines) < 0.35:
        return ""
    lowered = cleaned.lower()
    contamination_markers = (
        "gsm8k test",
        "human_eval test",
        "mmlu test question",
        "truthfulqa benchmark",
    )
    if any(marker in lowered for marker in contamination_markers):
        return ""
    return cleaned


def download_native_foundation(
    load_dataset: Callable[..., Any] | None,
    *,
    target_gb: float,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stream the licensed native foundation mix into one provenance JSONL."""
    output = TRAINING_DATA_DIR / "foundation_records.jsonl"
    target_bytes = int(float(target_gb) * 1024**3)
    specs = (
        {
            "source": "FineWeb-Edu",
            "dataset": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",
            "weight": 0.55,
            "fields": ("text",),
            "license": "ODC-By",
            "revision": "HuggingFaceFW/fineweb-edu:sample-10BT",
        },
        {
            "source": "The Stack v2 dedup",
            "dataset": "bigcode/the-stack-v2-dedup",
            "config": None,
            "weight": 0.15,
            "fields": ("content", "text"),
            "license": "per-record",
            "revision": "bigcode/the-stack-v2-dedup:train",
        },
        {
            "source": "FineMath-4+",
            "dataset": "HuggingFaceTB/finemath",
            "config": "finemath-4plus",
            "weight": 0.12,
            "fields": ("text", "content"),
            "license": "ODC-By",
            "revision": "HuggingFaceTB/finemath:finemath-4plus",
        },
        {
            "source": "Dolma science/technical",
            "dataset": "allenai/dolma",
            "config": "v1_7-sample",
            "weight": 0.08,
            "fields": ("text",),
            "license": "ODC-By",
            "revision": "allenai/dolma:v1_7-sample",
        },
    )
    stats: dict[str, Any] = {
        "bucket": "base",
        "output": str(output),
        "target_bytes": target_bytes,
        "bytes": 0,
        "documents": 0,
        "sources": {},
        "errors": [],
        "rejected": {
            "exact_duplicate": 0,
            "near_duplicate": 0,
            "language": 0,
            "code_syntax": 0,
            "math_verifier": 0,
        },
    }
    if dry_run:
        print(f"DRY RUN: would stream {target_gb:.2f} GB native foundation mix to {output}")
        return stats
    assert load_dataset is not None
    seen_hashes: set[str] = set()
    near_duplicates = MinHashDeduplicator()
    with output.open("w", encoding="utf-8") as stream:
        for spec in specs:
            source_target = int(target_bytes * float(spec["weight"]))
            source_bytes = 0
            source_docs = 0
            try:
                kwargs: dict[str, Any] = {
                    "split": "train",
                    "streaming": True,
                    "trust_remote_code": True,
                }
                config = spec["config"]
                dataset = (
                    load_dataset(spec["dataset"], config, **kwargs)
                    if config
                    else load_dataset(spec["dataset"], **kwargs)
                )
                for item in dataset:
                    language_hint = str(
                        item.get("language")
                        or item.get("lang")
                        or Path(str(item.get("path", ""))).suffix.lstrip(".")
                    )
                    text = next(
                        (str(item.get(field, "")) for field in spec["fields"] if item.get(field)),
                        "",
                    )
                    text = _clean_foundation_text(text)
                    if not text:
                        continue
                    language = _detect_content_language(
                        text,
                        source=str(spec["source"]),
                        hint=language_hint,
                    )
                    if language == "unknown":
                        stats["rejected"]["language"] += 1
                        continue
                    if not _code_syntax_valid(
                        text,
                        source=str(spec["source"]),
                        language_hint=language_hint,
                    ):
                        stats["rejected"]["code_syntax"] += 1
                        continue
                    if not _math_text_valid(text, source=str(spec["source"])):
                        stats["rejected"]["math_verifier"] += 1
                        continue
                    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if content_hash in seen_hashes:
                        stats["rejected"]["exact_duplicate"] += 1
                        continue
                    if near_duplicates.seen_or_add(text):
                        stats["rejected"]["near_duplicate"] += 1
                        continue
                    seen_hashes.add(content_hash)
                    license_name = str(
                        item.get("license", spec["license"])
                        if spec["license"] == "per-record"
                        else spec["license"]
                    )
                    normalized_license = license_name.lower().replace("_", "-")
                    if spec["license"] == "per-record" and not any(
                        allowed in normalized_license
                        for allowed in ("mit", "apache", "bsd", "isc", "mpl")
                    ):
                        continue
                    record = {
                        "text": text,
                        "source": spec["source"],
                        "license": license_name,
                        "source_revision": spec["revision"],
                        "document_sha256": content_hash,
                        "language": language,
                        "quality_checks": {
                            "pii_redacted": True,
                            "minhash_deduplicated": True,
                            "language_detected": True,
                            "code_syntax_checked": "stack" in str(spec["source"]).lower(),
                            "math_verified": "math" in str(spec["source"]).lower(),
                            "benchmark_contamination_checked": True,
                        },
                    }
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    stream.write(line)
                    encoded_bytes = len(line.encode("utf-8"))
                    source_bytes += encoded_bytes
                    source_docs += 1
                    if source_bytes >= source_target:
                        break
                if source_bytes < int(source_target * 0.98):
                    raise SourceDownloadFailure(
                        str(spec["source"]),
                        f"downloaded {source_bytes:,} of required {source_target:,} bytes",
                    )
            except Exception as exc:
                stats["errors"].append(f"{spec['source']}: {exc}")
            stats["sources"][str(spec["source"])] = {
                "bytes": source_bytes,
                "documents": source_docs,
                "target_bytes": source_target,
                "revision": spec["revision"],
            }
            stats["bytes"] += source_bytes
            stats["documents"] += source_docs
            print(
                f"{spec['source']}: {source_docs:,} documents, {source_bytes / 1024**3:.2f} GB",
                flush=True,
            )
    return stats


class SourceDownloadFailure(RuntimeError):  # noqa: N818 - serialized status name
    def __init__(self, source: str, message: str) -> None:
        super().__init__(f"{source}: {message}")
        self.source = source
        self.message = message


def load_datasets_import(dry_run: bool = False) -> Callable[..., Any] | None:
    if dry_run:
        return None
    try:
        from datasets import load_dataset
    except ImportError:
        print("Run: pip install datasets")
        sys.exit(1)
    return load_dataset


def ensure_training_data_dir() -> None:
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)


def prompt_key_from_dfc_text(text: str) -> str:
    task_close = "</task>"
    if "<task" not in text or task_close not in text:
        return text[:100]
    task_start = text.find("<task")
    prompt_start = text.find(">", task_start)
    if prompt_start == -1:
        return text[:100]
    prompt_end = text.find(task_close, prompt_start)
    if prompt_end == -1:
        return text[:100]
    return text[prompt_start + 1 : prompt_end][:100]


def download_base(
    load_dataset: Callable[..., Any] | None,
    dry_run: bool = False,
    *,
    fineweb_docs: int = 1_000_000,
    redpajama_docs: int = 800_000,
) -> dict[str, Any]:
    output = TRAINING_DATA_DIR / "base_corpus.txt"
    stats: dict[str, Any] = {"bucket": "base", "output": str(output), "documents": 0, "errors": []}

    if dry_run:
        print(
            "DRY RUN: would download "
            f"{fineweb_docs:,} FineWeb-Edu docs and {redpajama_docs:,} RedPajama docs into {output}"
        )
        return stats

    assert load_dataset is not None

    fineweb_path = TRAINING_DATA_DIR / "fineweb_edu.txt"
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        count = 0
        with fineweb_path.open("w", encoding="utf-8") as f:
            for item in ds:
                try:
                    text = item.get("text", "")
                    if text:
                        f.write(text.strip() + "\n\n")
                        count += 1
                    if count >= fineweb_docs:
                        break
                except Exception:
                    continue
        stats["documents"] += count
        print(f"FineWeb-Edu: {count:,} documents")
    except Exception as e:
        stats["errors"].append(f"FineWeb-Edu: {e}")
        print(f"SKIP FineWeb-Edu: {e}")

    redpajama_path = TRAINING_DATA_DIR / "redpajama.txt"
    try:
        ds2 = load_dataset(
            "togethercomputer/RedPajama-Data-V2",
            name="sample",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        count = 0
        with redpajama_path.open("w", encoding="utf-8") as f:
            for item in ds2:
                try:
                    text = item.get("raw_content", item.get("text", ""))
                    if text and len(text) > 200:
                        f.write(text.strip() + "\n\n")
                        count += 1
                    if count >= redpajama_docs:
                        break
                except Exception:
                    continue
        stats["documents"] += count
        print(f"RedPajama: {count:,} documents")
    except Exception as e:
        stats["errors"].append(f"RedPajama: {e}")
        print(f"SKIP RedPajama: {e}")

    try:
        with output.open("w", encoding="utf-8") as out:
            for fname in glob.glob(str(TRAINING_DATA_DIR / "fineweb_edu.txt")) + glob.glob(
                str(TRAINING_DATA_DIR / "redpajama.txt")
            ):
                with open(fname, encoding="utf-8", errors="replace") as src:
                    shutil.copyfileobj(src, out)
                out.write("\n\n")
        size_gb = output.stat().st_size / 1024**3
        print(f"Base corpus: {size_gb:.2f} GB")
    except Exception as e:
        stats["errors"].append(f"base_corpus concat: {e}")
        print(f"SKIP base_corpus concat: {e}")

    return stats


def download_reasoning(
    load_dataset: Callable[..., Any] | None,
    dry_run: bool = False,
    *,
    per_source_limit: int | None = None,
) -> dict[str, Any]:
    output = TRAINING_DATA_DIR / "reasoning.jsonl"
    stats: dict[str, Any] = {
        "bucket": "reasoning",
        "output": str(output),
        "examples": 0,
        "errors": [],
    }

    if dry_run:
        print(f"DRY RUN: would download reasoning teacher datasets into {output}")
        return stats

    assert load_dataset is not None

    datasets_to_load: list[
        tuple[str, str, int | None, Callable[[dict[str, Any]], dict[str, str] | None]]
    ] = [
        (
            "HuggingFaceTB/smol-smoltalk",
            "train",
            120_000,
            lambda x: {
                "prompt": next(
                    message["content"]
                    for message in x.get("messages", [])
                    if message.get("role") == "user"
                ),
                "response": next(
                    message["content"]
                    for message in x.get("messages", [])
                    if message.get("role") == "assistant"
                ),
            }
            if x.get("messages")
            else None,
        ),
        (
            "teknium/OpenHermes-2.5",
            "train",
            80_000,
            lambda x: {
                "prompt": x["conversations"][0]["value"],
                "response": x["conversations"][1]["value"],
            }
            if len(x.get("conversations", [])) >= 2
            else None,
        ),
        (
            "HuggingFaceH4/ultrachat_200k",
            "train_sft",
            60_000,
            lambda x: {
                "prompt": x["messages"][0]["content"],
                "response": x["messages"][1]["content"],
            }
            if len(x.get("messages", [])) >= 2
            else None,
        ),
        (
            "microsoft/orca-math-word-problems-200k",
            "train",
            50_000,
            lambda x: {"prompt": x["question"], "response": x["answer"]},
        ),
        (
            "meta-math/MetaMathQA",
            "train",
            100_000,
            lambda x: {"prompt": x["query"], "response": x["response"]},
        ),
        (
            "WizardLMTeam/WizardCoder_evol_instruct_110k",
            "train",
            None,
            lambda x: {"prompt": x["instruction"], "response": x["output"]},
        ),
    ]

    reject_patterns = [
        "ChatGPT",
        "GPT-4",
        "GPT4",
        "Claude",
        "Anthropic",
        "OpenAI",
        "I am an AI",
        "As an AI",
    ]

    total = 0
    with output.open("w", encoding="utf-8") as out:
        for ds_name, split, max_n, mapper in datasets_to_load:
            try:
                ds = load_dataset(ds_name, split=split, streaming=True, trust_remote_code=True)
                count = 0
                for item in ds:
                    try:
                        mapped = mapper(item)
                        if mapped is None:
                            continue
                        response = mapped.get("response", "")
                        if any(p in response for p in reject_patterns):
                            continue
                        task_type = (
                            "code"
                            if "coder" in ds_name.lower()
                            else "math"
                            if "math" in ds_name.lower()
                            else "instruction"
                        )
                        out.write(
                            json.dumps(
                                {
                                    "prompt": mapped["prompt"],
                                    "response": response,
                                    "source": ds_name,
                                    "task_type": task_type,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        count += 1
                        total += 1
                        limit = (
                            min(max_n, per_source_limit)
                            if max_n and per_source_limit
                            else (per_source_limit or max_n)
                        )
                        if limit and count >= limit:
                            break
                    except Exception:
                        continue
                print(f"  {ds_name}: {count:,} examples")
            except Exception as e:
                stats["errors"].append(f"{ds_name}: {e}")
                print(f"  SKIP {ds_name}: {e}")

    stats["examples"] = total
    size_mb = output.stat().st_size / 1024**2
    print(f"Reasoning total: {total:,} examples ({size_mb:.0f} MB)")
    return stats


def download_science(
    load_dataset: Callable[..., Any] | None,
    dry_run: bool = False,
    *,
    per_source_limit: int | None = None,
) -> dict[str, Any]:
    output = TRAINING_DATA_DIR / "frontier_dfc.jsonl"
    stats: dict[str, Any] = {
        "bucket": "science",
        "output": str(output),
        "examples": 0,
        "errors": [],
    }

    if dry_run:
        print(f"DRY RUN: would append science datasets into {output}")
        return stats

    assert load_dataset is not None

    science_datasets: list[
        tuple[str, str | None, str, int | None, Callable[[dict[str, Any]], dict[str, str]]]
    ] = [
        (
            "openai/gsm8k",
            "main",
            "train",
            None,
            lambda x: {
                "prompt": x["question"],
                "response": x["answer"],
                "template": "constraint_solve",
                "domain": "math",
            },
        ),
        (
            "lighteval/MATH",
            None,
            "train",
            None,
            lambda x: {
                "prompt": x["problem"],
                "response": x["solution"],
                "template": "hypothesis_chain",
                "domain": "math",
            },
        ),
        (
            "BoltzmannEntropy/QuantumLLMInstruct",
            None,
            "train",
            2000,
            lambda x: {
                "prompt": x.get("question", ""),
                "response": x.get("answer", ""),
                "template": "tool_action_trace",
                "domain": "quantum",
            },
        ),
        (
            "laion/Scientific-Summaries",
            None,
            "train",
            30_000,
            lambda x: {
                "prompt": "Summarize and state the main hypothesis of: " + x.get("abstract", ""),
                "response": x.get("title", "") + ". " + x.get("abstract", ""),
                "template": "hypothesis_chain",
                "domain": "science",
            },
        ),
    ]

    existing = set()
    try:
        with output.open("r", encoding="utf-8") as existing_file:
            for line in existing_file:
                try:
                    obj = json.loads(line)
                    prompt = obj.get("prompt", "")
                    if not prompt and isinstance(obj.get("text"), str):
                        prompt = prompt_key_from_dfc_text(obj["text"])
                    existing.add(prompt[:100])
                except Exception:
                    continue
    except Exception:
        pass

    added = 0
    with output.open("a", encoding="utf-8") as out:
        for ds_name, config, split, max_n, mapper in science_datasets:
            try:
                if config:
                    ds = load_dataset(
                        ds_name, config, split=split, streaming=True, trust_remote_code=True
                    )
                else:
                    ds = load_dataset(ds_name, split=split, streaming=True, trust_remote_code=True)
                count = 0
                for item in ds:
                    try:
                        mapped = mapper(item)
                        if not mapped.get("prompt") or not mapped.get("response"):
                            continue
                        key = mapped["prompt"][:100]
                        if key in existing:
                            continue
                        existing.add(key)
                        dfc_entry = {
                            "text": (
                                f"<bos>"
                                f'<task domain="{mapped["domain"]}" '
                                f'type="{mapped["template"]}">'
                                f"{mapped['prompt']}</task>"
                                f"<hyp>{mapped['response'][:500]}</hyp>"
                                f"<verify>INFERRED: from dataset, "
                                f"not simulator-verified</verify>"
                                f"<eos>"
                            ),
                            "domain": mapped["domain"],
                            "template": mapped["template"],
                            "verified": False,
                            "verifier_status": "inferred",
                            "source": ds_name,
                        }
                        out.write(json.dumps(dfc_entry, ensure_ascii=False) + "\n")
                        count += 1
                        added += 1
                        limit = (
                            min(max_n, per_source_limit)
                            if max_n and per_source_limit
                            else (per_source_limit or max_n)
                        )
                        if limit and count >= limit:
                            break
                    except Exception:
                        continue
                print(f"  {ds_name}: {count:,} DFC examples added")
            except Exception as e:
                stats["errors"].append(f"{ds_name}: {e}")
                print(f"  SKIP {ds_name}: {e}")

    stats["examples"] = added
    print(f"DFC science total added: {added:,}")
    return stats


def print_summary() -> None:
    print()
    print("=" * 60)
    print("TRAINING DATA SUMMARY")
    print("=" * 60)
    files = {
        "base_corpus.txt": "Base corpus (language model pretraining)",
        "reasoning.jsonl": "Reasoning Q&A (teacher data)",
        "frontier_dfc.jsonl": "DFC frontier science (domain verifier data)",
    }
    total_gb = 0.0
    for fname, desc in files.items():
        path = TRAINING_DATA_DIR / fname
        if path.exists():
            gb = path.stat().st_size / 1024**3
            total_gb += gb
            print(f"  {fname:<30} {gb:.2f} GB  {desc}")
        else:
            print(f"  {fname:<30} MISSING")
    print(f"\n  TOTAL: {total_gb:.2f} GB")
    print(f"  Estimated tokens: ~{int(total_gb * 250_000_000):,}")
    print("\n  Recommended data mix in training:")
    print("    base_corpus.txt  -> own_ratio  0.55 (55%)")
    print("    reasoning.jsonl  -> teacher    0.25 (25%)")
    print("    frontier_dfc     -> science    0.10 (10%)")
    print("    identity data    -> identity   0.10 (10%)")


def publish_fineweb_token_shards(profile: str = "30gb") -> dict[str, Any]:
    foundation_path = TRAINING_DATA_DIR / "foundation_records.jsonl"
    fineweb_path = TRAINING_DATA_DIR / "fineweb_edu.txt"
    if not foundation_path.exists() and not fineweb_path.exists():
        raise FileNotFoundError("Native foundation records must be downloaded first.")
    from training.v2_runtime import load_or_build_v2_tokenizer

    tokenizer = load_or_build_v2_tokenizer(dataset_path=TRAINING_DATA_DIR / "anra_training.txt")
    tokenizer_manifest = json.loads(
        (DATA_MANIFEST_DIR / "tokenizer_v3.json").read_text(encoding="utf-8")
    )
    tokenizer_sha256 = str(tokenizer_manifest["tokenizer_sha256"])
    revision_dir = DATA_MANIFEST_DIR / "native_foundation_v3" / profile

    def records(split: str) -> Iterator[SourceRecord]:
        if foundation_path.exists():
            with foundation_path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(item.get("text", "")).strip()
                    if not text:
                        continue
                    digest = str(
                        item.get("document_sha256")
                        or hashlib.sha256(text.encode("utf-8")).hexdigest()
                    )
                    split_value = int(digest[:8], 16) % 100
                    record_split = (
                        "validation"
                        if split_value == 98
                        else "test"
                        if split_value == 99
                        else "train"
                    )
                    if record_split != split:
                        continue
                    source = str(item.get("source", "unknown"))
                    yield SourceRecord(
                        text=text,
                        source=source,
                        license=str(item.get("license", "unknown")),
                        bucket="foundation",
                        quality=DataQuality(
                            0.6,
                            0.8,
                            0.95,
                            0.7 if "math" in source.lower() else 0.6,
                            0.2,
                            1.0,
                        ),
                        source_revision=str(item.get("source_revision", "unknown")),
                    )
            return
        if split != "train":
            return
        with fineweb_path.open("r", encoding="utf-8", errors="replace") as stream:
            buffer: list[str] = []
            for line in stream:
                if line.strip():
                    buffer.append(line.strip())
                    continue
                if buffer:
                    yield SourceRecord(
                        text="\n".join(buffer),
                        source="FineWeb-Edu",
                        license="ODC-By",
                        bucket="foundation",
                        quality=DataQuality(0.5, 0.8, 0.9, 0.6, 0.2, 1.0),
                        source_revision="HuggingFaceFW/fineweb-edu:sample-10BT",
                    )
                    buffer = []
            if buffer:
                yield SourceRecord(
                    text="\n".join(buffer),
                    source="FineWeb-Edu",
                    license="ODC-By",
                    bucket="foundation",
                    quality=DataQuality(0.5, 0.8, 0.9, 0.6, 0.2, 1.0),
                    source_revision="HuggingFaceFW/fineweb-edu:sample-10BT",
                )

    train_manifest = TokenShardPublisher(
        revision_dir,
        tokenizer_version="v3",
        tokenizer_sha256=tokenizer_sha256,
    ).publish(records("train"), tokenizer, allow_partial_final=True)
    validation_manifest = TokenShardPublisher(
        revision_dir / "validation",
        tokenizer_version="v3",
        tokenizer_sha256=tokenizer_sha256,
    ).publish(records("validation"), tokenizer, allow_partial_final=True)
    test_manifest = TokenShardPublisher(
        revision_dir / "test",
        tokenizer_version="v3",
        tokenizer_sha256=tokenizer_sha256,
    ).publish(records("test"), tokenizer, allow_partial_final=True)
    inventory = {
        "schema_version": 3,
        "licensed_tokens": int(train_manifest["total_tokens"]),
        "tokenizer_sha256": tokenizer_sha256,
        "manifest": str(revision_dir / "manifest.json"),
        "validation_manifest": str(revision_dir / "validation" / "manifest.json"),
        "test_manifest": str(revision_dir / "test" / "manifest.json"),
        "validation_tokens": int(validation_manifest["total_tokens"]),
        "test_tokens": int(test_manifest["total_tokens"]),
        "sources": train_manifest.get("source_record_mix", {}),
        "source_revisions": train_manifest.get("source_revisions", []),
        "licenses": train_manifest.get("licenses", []),
    }
    TOKEN_INVENTORY_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = TOKEN_INVENTORY_MANIFEST.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(inventory, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(TOKEN_INVENTORY_MANIFEST)
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download An-Ra training data buckets.")
    parser.add_argument(
        "--profile",
        choices=sorted(DATA_PROFILES),
        default="30gb",
        help="Data size profile. 30gb is the default native continuation campaign.",
    )
    parser.add_argument(
        "--bucket",
        choices=["base", "reasoning", "science"],
        help="Download only one bucket. Omit to build all buckets.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show planned work without downloading."
    )
    parser.add_argument(
        "--publish-token-shards",
        action="store_true",
        help="Publish immutable 10M-token uint16 FineWeb-Edu shards after download.",
    )
    parser.add_argument(
        "--prepare-corpus",
        action="store_true",
        help="Convert downloaded files into anra_training.txt and teacher_reasoning_v2.jsonl.",
    )
    parser.add_argument(
        "--max-source-mb",
        type=int,
        default=4096,
        help="Maximum downloaded source file size to ingest when --prepare-corpus is set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_training_data_dir()
    load_dataset = load_datasets_import(dry_run=args.dry_run)

    buckets = [args.bucket] if args.bucket else ["base", "reasoning", "science"]

    print("AN-RA TRAINING DATA DOWNLOAD")
    print("=" * 60)
    print(f"Profile: {args.profile}")
    print(f"Buckets: {', '.join(buckets)}")
    if args.dry_run:
        print("Mode: dry run")
    print()

    profile = DATA_PROFILES[args.profile]
    results: list[dict[str, Any]] = []
    for bucket in buckets:
        if bucket == "base":
            if "target_gb" in profile:
                results.append(
                    download_native_foundation(
                        load_dataset,
                        target_gb=float(profile["target_gb"]),
                        dry_run=args.dry_run,
                    )
                )
            else:
                results.append(
                    download_base(
                        load_dataset,
                        dry_run=args.dry_run,
                        fineweb_docs=int(profile["fineweb_docs"]),
                        redpajama_docs=int(profile["redpajama_docs"]),
                    )
                )
        elif bucket == "reasoning":
            results.append(
                download_reasoning(
                    load_dataset,
                    dry_run=args.dry_run,
                    per_source_limit=profile["reasoning_per_source"],
                )
            )
        elif bucket == "science":
            results.append(
                download_science(
                    load_dataset,
                    dry_run=args.dry_run,
                    per_source_limit=profile["science_per_source"],
                )
            )

    if args.publish_token_shards and not args.dry_run:
        inventory = publish_fineweb_token_shards(args.profile)
        print(f"Published licensed token inventory: {inventory['licensed_tokens']:,}")

    if args.prepare_corpus and not args.dry_run:
        from training.data_ingestion import prepare_training_corpus

        sources = [
            TRAINING_DATA_DIR / "base_corpus.txt",
            TRAINING_DATA_DIR / "reasoning.jsonl",
            TRAINING_DATA_DIR / "frontier_dfc.jsonl",
        ]
        report = prepare_training_corpus(
            explicit_sources=[source for source in sources if source.exists()],
            include_drive=True,
            max_source_mb=args.max_source_mb,
            mount_drive=False,
        )
        print(
            "Prepared AN-RA corpus: "
            f"{report.total_examples:,} examples, {report.teacher_records:,} teacher records"
        )

    print_summary()
    failures = [
        SourceDownloadFailure(result["bucket"], str(error))
        for result in results
        for error in result.get("errors", [])
    ]
    status = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "incomplete" if failures else "complete",
        "buckets": results,
        "failures": [
            {"source": failure.source, "message": failure.message} for failure in failures
        ],
    }
    DOWNLOAD_STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = DOWNLOAD_STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(DOWNLOAD_STATUS)
    if failures:
        print(f"INCOMPLETE INVENTORY: {len(failures)} source failure(s). See {DOWNLOAD_STATUS}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
