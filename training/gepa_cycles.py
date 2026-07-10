"""Evidence-preserving GEPA cycle runner; it never applies proposals itself."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path

from training.gepa import build_gepa_report


def run_gepa_cycles(
    *,
    cycles: int,
    evidence_for_cycle: Callable[[int], dict[str, object]],
    review_candidate: Callable[[int, Mapping[str, object]], tuple[bool, str]],
    score_floor: float = 4.0,
) -> dict[str, object]:
    """Run proposal-only GEPA cycles and require at least one justified rejection."""
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    rows: list[dict[str, object]] = []
    rejected = 0
    for cycle in range(1, cycles + 1):
        evidence = evidence_for_cycle(cycle)
        report = build_gepa_report(
            eval_summary=dict(evidence.get("eval_summary", {})),
            hard_examples=evidence.get("hard_examples", []),
            rlvr_report=dict(evidence.get("rlvr_report", {})),
        )
        decisions = list(report["scores"])
        reviews = []
        for decision in decisions:
            approved, reason = review_candidate(cycle, decision)
            reviews.append(
                {
                    "candidate_id": decision.get("candidate_id"),
                    "approved": bool(approved),
                    "reason": str(reason),
                    "score": float(decision.get("score", 0.0)),
                }
            )
        cycle_rejected = any(
            review["approved"] is False and float(review["score"]) >= score_floor
            for review in reviews
        )
        rejected += int(cycle_rejected)
        rows.append(
            {
                "cycle": cycle,
                "candidate_count": len(report["candidates"]),
                "decisions": decisions,
                "reviews": reviews,
                "correctly_rejected": cycle_rejected,
            }
        )
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "cycles": cycles,
        "rejected_cycles": rejected,
        "ten_cycle_gate": cycles >= 10 and rejected >= 1,
        "auto_apply_enabled": False,
        "rows": rows,
    }


def write_gepa_cycle_report(
    report: dict[str, object], output_path: str | Path
) -> dict[str, object]:
    from training.v2_runtime import write_json

    path = Path(output_path)
    report["report_path"] = str(path)
    write_json(path, report)
    return report
