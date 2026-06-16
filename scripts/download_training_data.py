#!/usr/bin/env python3
"""Download and assemble An-Ra training data buckets."""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

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
        "fineweb_docs": 2_000,
        "redpajama_docs": 1_000,
        "reasoning_per_source": 1_000,
        "science_per_source": 500,
    },
    "t4-15gb": {
        "fineweb_docs": 120_000,
        "redpajama_docs": 40_000,
        "reasoning_per_source": 20_000,
        "science_per_source": 10_000,
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


class SourceDownloadFailure(RuntimeError):
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
                with open(fname, "r", encoding="utf-8", errors="replace") as src:
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
    stats: dict[str, Any] = {"bucket": "reasoning", "output": str(output), "examples": 0, "errors": []}

    if dry_run:
        print(f"DRY RUN: would download reasoning teacher datasets into {output}")
        return stats

    assert load_dataset is not None

    datasets_to_load: list[tuple[str, str, int | None, Callable[[dict[str, Any]], dict[str, str] | None]]] = [
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
                        out.write(
                            json.dumps(
                                {
                                    "prompt": mapped["prompt"],
                                    "response": response,
                                    "source": ds_name,
                                    "task_type": "reasoning",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        count += 1
                        total += 1
                        limit = min(max_n, per_source_limit) if max_n and per_source_limit else (per_source_limit or max_n)
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
    stats: dict[str, Any] = {"bucket": "science", "output": str(output), "examples": 0, "errors": []}

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
                    ds = load_dataset(ds_name, config, split=split, streaming=True, trust_remote_code=True)
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
                                f"<task domain=\"{mapped['domain']}\" "
                                f"type=\"{mapped['template']}\">"
                                f"{mapped['prompt']}</task>"
                                f"<hyp>{mapped['response'][:500]}</hyp>"
                                f"<verify>INFERRED: from dataset, "
                                f"not simulator-verified</verify>"
                                f"<eos>"
                            ),
                            "domain": mapped["domain"],
                            "template": mapped["template"],
                            "verified": False,
                            "source": ds_name,
                        }
                        out.write(json.dumps(dfc_entry, ensure_ascii=False) + "\n")
                        count += 1
                        added += 1
                        limit = min(max_n, per_source_limit) if max_n and per_source_limit else (per_source_limit or max_n)
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


def publish_fineweb_token_shards() -> dict[str, Any]:
    fineweb_path = TRAINING_DATA_DIR / "fineweb_edu.txt"
    if not fineweb_path.exists():
        raise FileNotFoundError(
            "FineWeb-Edu must be downloaded before token-shard publication."
        )
    from training.v2_runtime import load_or_build_v2_tokenizer

    tokenizer = load_or_build_v2_tokenizer(
        dataset_path=TRAINING_DATA_DIR / "anra_training.txt"
    )
    revision_dir = DATA_MANIFEST_DIR / "fineweb_edu_v3"

    def records():
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

    manifest = TokenShardPublisher(
        revision_dir,
        tokenizer_version="v3",
    ).publish(records(), tokenizer)
    inventory = {
        "schema_version": 3,
        "licensed_tokens": int(manifest["total_tokens"]),
        "sources": [
            {
                "name": "FineWeb-Edu",
                "revision": "HuggingFaceFW/fineweb-edu:sample-10BT",
                "license": "ODC-By",
                "manifest": str(revision_dir / "manifest.json"),
            }
        ],
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
        default="t4-15gb",
        help="Data size profile. t4-15gb is the default Colab/Drive-friendly profile.",
    )
    parser.add_argument(
        "--bucket",
        choices=["base", "reasoning", "science"],
        help="Download only one bucket. Omit to build all buckets.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned work without downloading.")
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

    if args.bucket:
        buckets = [args.bucket]
    else:
        buckets = ["base", "reasoning", "science"]

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
        inventory = publish_fineweb_token_shards()
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
            {"source": failure.source, "message": failure.message}
            for failure in failures
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
