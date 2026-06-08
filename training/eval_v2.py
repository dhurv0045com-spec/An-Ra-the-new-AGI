from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from engine.metric_bus import instrument
from training.v2_runtime import append_jsonl, generate_text, v2_report_path, write_json


try:
    from symbolic_bridge import query_logic, query_math
except Exception:
    query_logic = query_math = None  # type: ignore[assignment]


EVAL_SUITE = [
    {
        "id": "identity_self",
        "category": "identity",
        "prompt": "H: Who are you?\nANRA:",
        "keywords": ["an-ra"],
    },
    {
        "id": "identity_purpose",
        "category": "identity",
        "prompt": "H: What is your purpose?\nANRA:",
        "keywords": ["purpose", "an-ra"],
    },
    {
        "id": "continuity",
        "category": "continuity",
        "prompt": "H: Remember this key: cobalt-19. Now tell me the key and why context matters.\nANRA:",
        "keywords": ["cobalt-19", "context"],
    },
    {
        "id": "reasoning_consistency",
        "category": "reasoning",
        "prompt": "H: Explain the difference between strong consistency and eventual consistency in two or three sentences.\nANRA:",
        "keywords": ["strong consistency", "eventual consistency"],
    },
    {
        "id": "instruction_debug",
        "category": "instruction",
        "prompt": "H: Give me three short steps to debug a failing Python test.\nANRA:",
        "keywords": ["test", "debug", "assert"],
    },
    {
        "id": "symbolic_math",
        "category": "symbolic",
        "prompt": "H: Differentiate x^2 + 3*x.\nANRA:",
        "verifier": "math",
    },
    {
        "id": "symbolic_logic",
        "category": "symbolic",
        "prompt": "H: Is (A->B) and (B->C) -> (A->C) valid? Explain briefly.\nANRA:",
        "verifier": "logic",
    },
]

GOLDEN_EVAL_SCHEMA_VERSION = 1
GOLDEN_EVAL_THRESHOLDS = {
    "overall_min": 0.60,
    "identity_min": 0.70,
    "symbolic_min": 0.60,
    "reasoning_min": 0.60,
}


@instrument("evaluation")
def quick_eval_loss(model, dataset, *, device: torch.device, max_examples: int = 100, batch_size: int = 8, pad_id: int = 0) -> dict:
    """Mean CE loss over up to max_examples validation examples."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, min(len(dataset), max_examples), batch_size):
            rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset), max_examples))]
            if not rows:
                break
            xb = torch.stack([row[0] for row in rows]).to(device)
            yb = torch.stack([row[1] for row in rows]).to(device)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=pad_id)
            losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("[eval_v2] quick_eval_loss received an empty validation dataset")
    loss_value = float(sum(losses) / len(losses))
    return {
        "score": max(0.0, 1.0 - loss_value / 10.0),
        "loss": loss_value,
        "n_examples": len(losses),
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _keyword_score(text: str, keywords: list[str]) -> float:
    lowered = _normalize(text)
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    if not keywords:
        return 0.0
    return hits / len(keywords)


def _verified_score(verifier: str, prompt: str, response: str) -> tuple[float, str]:
    response_norm = _normalize(response)
    if verifier == "math" and query_math is not None:
        result = query_math(prompt)
        answer_text = getattr(result, "answer_text", str(result))
        expected_norm = _normalize(answer_text)
        score = 1.0 if expected_norm and expected_norm in response_norm else 0.0
        return score, answer_text
    if verifier == "logic" and query_logic is not None:
        result = query_logic(prompt)
        answer_text = getattr(result, "answer_text", str(result))
        expected_norm = _normalize(answer_text)
        score = 1.0 if expected_norm and expected_norm in response_norm else 0.0
        return score, answer_text
    return 0.0, ""


def build_golden_eval_baseline(
    summary: dict[str, object],
    *,
    source: str = "compact_eval",
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    active_thresholds = dict(GOLDEN_EVAL_THRESHOLDS)
    if thresholds:
        active_thresholds.update({key: float(value) for key, value in thresholds.items()})

    category_scores_raw = summary.get("category_scores", {})
    category_scores = category_scores_raw if isinstance(category_scores_raw, dict) else {}
    overall_score = float(summary.get("overall_score", 0.0) or 0.0)

    gates = {
        "overall": overall_score >= active_thresholds["overall_min"],
        "identity": float(category_scores.get("identity", 0.0) or 0.0) >= active_thresholds["identity_min"],
        "symbolic": float(category_scores.get("symbolic", 0.0) or 0.0) >= active_thresholds["symbolic_min"],
        "reasoning": float(category_scores.get("reasoning", 0.0) or 0.0) >= active_thresholds["reasoning_min"],
    }

    results_raw = summary.get("results", [])
    results = results_raw if isinstance(results_raw, list) else []
    tasks: list[dict[str, object]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        tasks.append(
            {
                "id": result.get("id", ""),
                "category": result.get("category", ""),
                "prompt": result.get("prompt", ""),
                "response": result.get("response", ""),
                "score": float(result.get("score", 0.0) or 0.0),
                "reason": result.get("reason", ""),
                "expected": result.get("expected", ""),
            }
        )

    generated_at = float(summary.get("generated_at", time.time()) or time.time())
    return {
        "schema_version": GOLDEN_EVAL_SCHEMA_VERSION,
        "baseline_id": f"golden-{int(generated_at)}-{overall_score:.4f}",
        "generated_at": generated_at,
        "source": source,
        "suite_size": len(tasks),
        "overall_score": round(overall_score, 4),
        "category_scores": {
            str(category): round(float(score), 4)
            for category, score in category_scores.items()
        },
        "thresholds": active_thresholds,
        "promotion_gates": gates,
        "promotion_allowed": all(gates.values()),
        "tasks": tasks,
    }


def write_golden_eval_baseline(
    summary: dict[str, object],
    *,
    source: str = "compact_eval",
    output_path: Path | None = None,
) -> dict[str, object]:
    baseline = build_golden_eval_baseline(summary, source=source)
    write_json(output_path or v2_report_path("golden_eval_baseline"), baseline)
    return baseline


@instrument("evaluation")
def run_compact_eval(
    model,
    tokenizer,
    *,
    device: torch.device,
    output: bool = True,
    golden: bool = False,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    category_scores: dict[str, list[float]] = defaultdict(list)

    for item in EVAL_SUITE:
        response = generate_text(
            model,
            tokenizer,
            item["prompt"],
            device=device,
            max_new_tokens=96,
            temperature=0.8,
            top_k=40,
        )
        if "verifier" in item:
            score, expected = _verified_score(str(item["verifier"]), str(item["prompt"]), response)
            reason = f"verified against {item['verifier']} reference"
        else:
            score = _keyword_score(response, list(item.get("keywords", [])))
            expected = ""
            reason = "keyword coverage"
        category_scores[str(item["category"])].append(score)
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "response": response,
                "score": round(float(score), 4),
                "reason": reason,
                "expected": expected,
            }
        )

    averages = {
        category: round(sum(scores) / max(1, len(scores)), 4)
        for category, scores in category_scores.items()
    }
    overall = round(sum(averages.values()) / max(1, len(averages)), 4)
    summary = {
        "generated_at": time.time(),
        "overall_score": overall,
        "category_scores": averages,
        "results": results,
    }
    if output:
        write_json(v2_report_path("eval_summary"), summary)
        append_jsonl(v2_report_path("eval_history"), {"ts": summary["generated_at"], "overall_score": overall, "category_scores": averages})
        if golden:
            write_golden_eval_baseline(summary, source="train_unified_eval")
    return summary


if __name__ == "__main__":
    print(json.dumps({"suite_size": len(EVAL_SUITE)}, indent=2))
