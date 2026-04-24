from __future__ import annotations

import json
import time
from pathlib import Path

from training.v2_runtime import v2_report_path, write_json


def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _recommendations(eval_summary: dict, hard_examples: list[dict], mix_report: dict) -> list[str]:
    recs: list[str] = []
    category_scores = eval_summary.get("category_scores", {}) if isinstance(eval_summary, dict) else {}
    if float(category_scores.get("symbolic", 0.0) or 0.0) < 0.6:
        recs.append("Increase verified symbolic/code samples in the next milestone run.")
    if float(category_scores.get("identity", 0.0) or 0.0) < 0.7:
        recs.append("Bias the next identity fine-tune toward self-description, purpose, and worldview prompts.")
    if float(category_scores.get("continuity", 0.0) or 0.0) < 0.6:
        recs.append("Replay more continuity-heavy hard examples through ghost_memory-style curriculum.")
    previews = " ".join(str(item.get("preview", "")) for item in hard_examples[:8]).lower()
    if any(token in previews for token in ["debug", "python", "code", "test"]):
        recs.append("Add more teacher-generated code-debug examples with verified fixes.")
    if int(mix_report.get("teacher_external_used", 0) or 0) == 0:
        recs.append("Provide training_data/teacher_reasoning_v2.jsonl to strengthen the teacher bucket beyond symbolic fallbacks.")
    if not recs:
        recs.append("Keep the current ratios; no single weak area dominated the latest session.")
    return recs


def run_self_improvement() -> dict[str, object]:
    eval_summary = _load_json(v2_report_path("eval_summary")) or {}
    hard_blob = _load_json(v2_report_path("hard_examples")) or {}
    mix_report = _load_json(v2_report_path("mix_report")) or {}
    hard_examples = hard_blob.get("examples", []) if isinstance(hard_blob, dict) else []

    report = {
        "generated_at": time.time(),
        "stage": "self_improvement_v2",
        "eval_summary_path": str(v2_report_path("eval_summary")),
        "hard_examples_path": str(v2_report_path("hard_examples")),
        "mix_report_path": str(v2_report_path("mix_report")),
        "recommendations": _recommendations(
            eval_summary if isinstance(eval_summary, dict) else {},
            hard_examples if isinstance(hard_examples, list) else [],
            mix_report if isinstance(mix_report, dict) else {},
        ),
    }
    write_json(v2_report_path("improvement_report"), report)
    return report


def main() -> None:
    print(json.dumps(run_self_improvement(), indent=2))


run_self_improvement_v2 = run_self_improvement


if __name__ == "__main__":
    main()
