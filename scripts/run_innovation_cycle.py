from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import ROOT
from innovation.gap_scanner import scan
from innovation.schema import CapabilityGap, Hypothesis
from innovation.scoreboard import score_hypothesis, write_report


def _hyp_id(gap: CapabilityGap) -> str:
    return hashlib.sha1(f"{gap.gap_id}|{gap.description}".encode("utf-8")).hexdigest()[:12]


def gap_to_hypothesis(gap: CapabilityGap) -> Hypothesis:
    description = f"Close gap: {gap.description} in {gap.detected_in}"
    return Hypothesis(
        hyp_id=_hyp_id(gap),
        gap_id=gap.gap_id,
        description=description,
        falsifier="The gap remains detectable by innovation.scan() or the targeted pytest/verifier still fails.",
        predicted_delta={
            "capability_gap_count": -1,
            "verified_behavior": "+1 targeted passing check",
            "regression_budget": "no existing tests fail",
        },
        constraints=[
            "No checkpoint format changes",
            "No mandatory new pip dependencies",
            "Do not edit history or checkpoint files",
            "Use the smallest experiment that proves the behavior",
        ],
        smallest_experiment=f"Patch {gap.detected_in}, then run a focused pytest or verifier check for the affected behavior.",
        verifier_path="python -m pytest tests/ -x -q",
        created_at=time.time(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the An-Ra innovation cycle.")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    gaps = scan(args.repo_root)[: max(1, int(args.top))]
    hyps = [gap_to_hypothesis(gap) for gap in gaps]
    scores = [score_hypothesis(hyp) for hyp in hyps]
    score_by_id = {score.hyp_id: score for score in scores}

    report_path = args.repo_root / "state" / "reports" / f"innovation_{time.strftime('%Y%m%d')}.json"
    write_report(scores, report_path)
    context_path = report_path.with_name(report_path.stem + "_context.json")
    payload = {"gaps": [gap.to_dict() for gap in gaps], "hypotheses": [hyp.to_dict() for hyp in hyps]}
    context_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ranked = sorted(hyps, key=lambda hyp: score_by_id[hyp.hyp_id].total, reverse=True)
    print("Rank | Score | Decision         | Hypothesis")
    print("-----|-------|------------------|-----------")
    for idx, hyp in enumerate(ranked, start=1):
        score = score_by_id[hyp.hyp_id]
        print(f"{idx:>4} | {score.total:>5.1f} | {score.decision:<16} | {hyp.description}")
    winners = [hyp for hyp in ranked if score_by_id[hyp.hyp_id].total >= 80]
    if winners:
        print(f"**IMPLEMENT: {winners[0].description}**")
    print(f"Report: {report_path}")
    print(f"Context: {context_path}")


if __name__ == "__main__":
    main()
