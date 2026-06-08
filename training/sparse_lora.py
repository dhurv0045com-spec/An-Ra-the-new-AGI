from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


SPARSE_LORA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SparseLoRAEstimateConfig:
    mode: str = "logging_only"
    keep_ratio: float = 0.65
    protect_first_tokens: int = 4
    protect_last_tokens: int = 8
    pad_token_id: int = 0
    special_token_ids: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_examples: int = 256


def _stable_text_token_ids(texts: Sequence[str]) -> list[list[int]]:
    vocab: dict[str, int] = {}
    rows: list[list[int]] = []
    for text in texts:
        pieces = re.findall(r"\w+|[^\w\s]", text.lower())
        row: list[int] = []
        for piece in pieces:
            if piece not in vocab:
                vocab[piece] = len(vocab) + 8
            row.append(vocab[piece])
        rows.append(row)
    return rows


def _token_scores(sequence: Sequence[int], cfg: SparseLoRAEstimateConfig) -> list[tuple[int, float]]:
    active = [(idx, int(token)) for idx, token in enumerate(sequence) if int(token) != cfg.pad_token_id]
    if not active:
        return []

    counts: dict[int, int] = {}
    for _, token in active:
        counts[token] = counts.get(token, 0) + 1

    last_idx = len(sequence) - 1
    scores: list[tuple[int, float]] = []
    for idx, token in active:
        score = 1.0
        protected = idx < cfg.protect_first_tokens or idx > last_idx - cfg.protect_last_tokens
        if token in cfg.special_token_ids:
            score *= 0.35
        if idx > 0 and int(sequence[idx - 1]) == token:
            score *= 0.45
        if counts[token] >= max(4, len(active) // 5):
            score *= 0.65
        if protected:
            score = max(score, 0.95)
        scores.append((idx, score))
    return scores


def estimate_sparse_lora_from_sequences(
    sequences: Iterable[Sequence[int]],
    *,
    config: SparseLoRAEstimateConfig | None = None,
) -> dict[str, object]:
    cfg = config or SparseLoRAEstimateConfig()
    total_active = 0
    total_kept = 0
    total_skipped = 0
    examples_analyzed = 0
    samples: list[dict[str, object]] = []

    for example_idx, sequence in enumerate(sequences):
        scores = _token_scores(sequence, cfg)
        active_tokens = len(scores)
        if active_tokens == 0:
            continue
        examples_analyzed += 1

        keep_count = max(1, math.ceil(active_tokens * cfg.keep_ratio))
        ranked = sorted(scores, key=lambda item: (-item[1], item[0]))
        kept_positions = {idx for idx, _ in ranked[:keep_count]}
        skipped = active_tokens - len(kept_positions)

        total_active += active_tokens
        total_kept += len(kept_positions)
        total_skipped += skipped

        if len(samples) < 8:
            samples.append(
                {
                    "example_index": example_idx,
                    "active_tokens": active_tokens,
                    "kept_tokens": len(kept_positions),
                    "skipped_tokens_estimate": skipped,
                    "skip_ratio": round(skipped / active_tokens, 4),
                }
            )

    skip_ratio = total_skipped / total_active if total_active else 0.0
    return {
        "mode": cfg.mode,
        "active_tokens": total_active,
        "kept_tokens": total_kept,
        "skipped_tokens_estimate": total_skipped,
        "estimated_skip_ratio": round(skip_ratio, 4),
        "estimated_context_token_compute_saved": round(skip_ratio, 4),
        "examples_analyzed": examples_analyzed,
        "sample_estimates": samples,
        "config": {
            **asdict(cfg),
            "special_token_ids": list(cfg.special_token_ids),
        },
    }


def estimate_sparse_lora_from_texts(
    texts: Sequence[str],
    *,
    config: SparseLoRAEstimateConfig | None = None,
) -> dict[str, object]:
    return estimate_sparse_lora_from_sequences(_stable_text_token_ids(texts), config=config)


def sample_training_texts(path: Path, *, max_examples: int) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            rows.append(line)
        if len(rows) >= max_examples:
            break
    return rows


def build_sparse_lora_report(
    texts: Sequence[str],
    *,
    source: str,
    config: SparseLoRAEstimateConfig | None = None,
) -> dict[str, object]:
    cfg = config or SparseLoRAEstimateConfig()
    estimate = estimate_sparse_lora_from_texts(texts, config=cfg)
    return {
        "schema_version": SPARSE_LORA_SCHEMA_VERSION,
        "generated_at": time.time(),
        "technology": "SparseLoRA-style contextual sparsity",
        "source": source,
        "training_enabled": False,
        "decision": "measure_only_until_eval_beats_lora_baseline",
        "estimate": estimate,
        "notes": [
            "This report estimates context-token gradient work that could be skipped.",
            "No token skipping is applied to training until a LoRA/QLoRA baseline comparison passes eval gates.",
        ],
    }


def write_sparse_lora_report_from_dataset(
    data_path: Path,
    *,
    config: SparseLoRAEstimateConfig | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    cfg = config or SparseLoRAEstimateConfig()
    texts = sample_training_texts(data_path, max_examples=cfg.max_examples)
    report = build_sparse_lora_report(texts, source=str(data_path), config=cfg)
    if not texts:
        report["decision"] = "blocked_no_dataset_sample"
    if output_path is None:
        from training.v2_runtime import v2_report_path, write_json

        output_path = v2_report_path("sparse_lora_report")
    else:
        from training.v2_runtime import write_json

    report["report_path"] = str(output_path)
    write_json(output_path, report)
    return report
