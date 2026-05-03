from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from anra_paths import DRIVE_V2_CHECKPOINTS
from training.v2_runtime import canonical_v2_checkpoint, v2_report_path, write_json


logger = logging.getLogger(__name__)

FIXED_CAPABILITY_PROMPTS = [
    {"id": "reasoning_01", "kind": "reasoning", "prompt": "If A implies B and B implies C, does A imply C?"},
    {"id": "reasoning_02", "kind": "reasoning", "prompt": "A train leaves at 3pm and travels for 2.5 hours. When does it arrive?"},
    {"id": "math_01", "kind": "math", "prompt": "Solve x^2 - 4 = 0."},
    {"id": "math_02", "kind": "math", "prompt": "What is 17 * 19?"},
    {"id": "code_01", "kind": "code", "prompt": "Find the bug: def tail(xs): return xs[0:len(xs)-1]"},
    {"id": "code_02", "kind": "code", "prompt": "Explain why mutable default arguments are risky in Python."},
    {"id": "identity_01", "kind": "identity", "prompt": "Who are you?"},
    {"id": "identity_02", "kind": "identity", "prompt": "Who created An-Ra?"},
    {"id": "verification_01", "kind": "verification", "prompt": "How would you check a symbolic math answer?"},
    {"id": "honesty_01", "kind": "honesty", "prompt": "What should you do when you are uncertain?"},
]


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read sovereignty input JSON %s: %s", path, exc)
        return None


def _metric_float(data: dict, *names: str) -> float | None:
    for name in names:
        value = data.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _capability_metrics(eval_summary: dict) -> dict[str, object]:
    prompt_scores = eval_summary.get("fixed_prompt_scores", {})
    if isinstance(prompt_scores, list):
        prompt_scores = {
            str(item.get("id", idx)): float(item.get("score", 0.0) or 0.0)
            for idx, item in enumerate(prompt_scores)
            if isinstance(item, dict)
        }
    if not isinstance(prompt_scores, dict):
        prompt_scores = {}
    fixed_scores = {
        prompt["id"]: float(prompt_scores.get(prompt["id"], 0.0) or 0.0)
        for prompt in FIXED_CAPABILITY_PROMPTS
    }
    return {
        "perplexity": _metric_float(eval_summary, "val_perplexity", "perplexity", "ppl"),
        "overall_score": _metric_float(eval_summary, "overall_score", "score"),
        "fixed_prompt_scores": fixed_scores,
        "fixed_prompt_mean": sum(fixed_scores.values()) / max(1, len(fixed_scores)),
        "fixed_prompt_rubric": {
            "0.0": "wrong, evasive, or identity-breaking",
            "0.5": "partially correct but incomplete",
            "1.0": "correct, concise, and identity-consistent",
        },
    }


def _wins_capability_gate(current: dict[str, object], previous: dict[str, object]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    prev_cap = previous.get("capability_metrics", {}) if isinstance(previous.get("capability_metrics"), dict) else {}
    current_ppl = current.get("perplexity")
    prev_ppl = prev_cap.get("perplexity")
    current_fixed = float(current.get("fixed_prompt_mean", 0.0) or 0.0)
    prev_fixed = float(prev_cap.get("fixed_prompt_mean", 0.0) or 0.0)
    current_score = float(current.get("overall_score", 0.0) or 0.0)
    prev_score = float(previous.get("best_score", 0.0) or 0.0)

    if current_ppl is None:
        reasons.append("missing candidate perplexity")
    elif prev_ppl is not None and float(current_ppl) > float(prev_ppl):
        reasons.append(f"perplexity regressed: {current_ppl} > {prev_ppl}")

    if current_fixed <= 0.0:
        reasons.append("missing fixed-prompt benchmark scores")
    elif current_fixed < prev_fixed:
        reasons.append(f"fixed-prompt score regressed: {current_fixed:.3f} < {prev_fixed:.3f}")

    if current_score < prev_score and current_fixed <= prev_fixed:
        reasons.append(f"overall score did not improve: {current_score:.3f} < {prev_score:.3f}")

    return not reasons, reasons


def run_sovereignty_audit() -> dict[str, object]:
    eval_summary = _load_json(v2_report_path("eval_summary")) or {}
    improvement = _load_json(v2_report_path("improvement_report")) or {}
    previous = _load_json(v2_report_path("audit_report")) or {}
    capability = _capability_metrics(eval_summary)
    current_score = float(capability.get("overall_score", 0.0) or 0.0)
    best_score = max(current_score, float(previous.get("best_score", 0.0) or 0.0))
    checkpoint = canonical_v2_checkpoint("ouroboros")
    if not checkpoint.exists():
        checkpoint = canonical_v2_checkpoint("identity")
    if not checkpoint.exists():
        checkpoint = canonical_v2_checkpoint("brain")

    promoted, gate_reasons = _wins_capability_gate(capability, previous)
    best_checkpoint = checkpoint.parent / "best_v2_checkpoint.pt"
    if promoted and checkpoint.exists():
        shutil.copy2(checkpoint, best_checkpoint)
        try:
            DRIVE_V2_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_checkpoint, DRIVE_V2_CHECKPOINTS / best_checkpoint.name)
        except Exception as exc:
            logger.warning("Could not mirror promoted checkpoint to Drive: %s", exc)

    report = {
        "generated_at": time.time(),
        "stage": "sovereignty_audit_v2",
        "checkpoint": str(checkpoint),
        "current_score": current_score,
        "best_score": best_score,
        "capability_metrics": capability,
        "promoted": promoted,
        "best_checkpoint": str(best_checkpoint) if promoted else str(previous.get("best_checkpoint", best_checkpoint)),
        "decision": "promote" if promoted else "hold",
        "gate_reasons": gate_reasons,
        "recommendations": improvement.get("recommendations", []),
    }
    write_json(v2_report_path("audit_report"), report)
    return report


def main() -> None:
    print(json.dumps(run_sovereignty_audit(), indent=2))


run_sovereignty_audit_v2 = run_sovereignty_audit


if __name__ == "__main__":
    main()
