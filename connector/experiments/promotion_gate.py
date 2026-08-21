"""Promotion gate: evaluation is the only authority that promotes a checkpoint.

Compares a candidate checkpoint against the measured baseline on the P1-P6
probe battery (nonce items, both protocols) and issues a promote/reject
verdict from explicit rules — never from vibes:

  PROMOTE requires ALL of:
    R1  P1 nonce knowledge >= 4/5 in BOTH protocols (the binding deficit is fixed)
    R2  no family regresses below baseline by more than 1 item
    R3  P3 natural-language copying >= 4/5 (the one skill the baseline has)

Anything else is REJECT with the measured numbers attached. The verdict JSON
is standalone evidence: it embeds both probe reports.

Run:
  py -3 -m connector.experiments.promotion_gate \
      --candidate checkpoints/anra-v4-20k-sft-context-binding.pt \
      --baseline-probe output/probe_full_resume.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anra_core.executor import CoreExecutor
from connector.experiments.cognitive_credit.capability_probe import run_probe

THRESHOLD = 4  # items of 5


def _hits(report: dict, family: str, protocol: str) -> int:
    block = report.get(family, {})
    if isinstance(block, dict) and protocol in block:
        return int(str(block[protocol]).split("/")[0])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-probe", required=True,
                        help="probe JSON of the current active checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="output/promotion_gate_verdict.json")
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline_probe).read_text(encoding="utf-8"))
    print(f"[gate] evaluating candidate: {args.candidate}", flush=True)
    candidate = run_probe(args.candidate, args.device)

    families = [
        "P1_nonce_knowledge_use",
        "P2_plan_following_no_arithmetic",
        "P3_verbatim_copy",
        "P4_tool_result_use",
    ]
    checks, regressions = [], []
    for family in families:
        for protocol in ("nl", "tag"):
            base = _hits(baseline, family, protocol)
            cand = _hits(candidate, family, protocol)
            checks.append({
                "family": family, "protocol": protocol,
                "baseline": f"{base}/5", "candidate": f"{cand}/5",
                "delta": cand - base,
            })
            if cand < base - 1:
                regressions.append(f"{family}:{protocol} {base}-> {cand}")

    p1_ok = all(
        _hits(candidate, "P1_nonce_knowledge_use", p) >= THRESHOLD
        for p in ("nl", "tag"))
    p3_nl_kept = _hits(candidate, "P3_verbatim_copy", "nl") >= THRESHOLD
    no_regress = not regressions

    promote = p1_ok and p3_nl_kept and no_regress
    verdict = {
        "verdict": "PROMOTE" if promote else "REJECT",
        "candidate": args.candidate,
        "candidate_global_step": candidate.get("global_step"),
        "rules": {
            "R1_p1_binding_both_protocols_ge_4": p1_ok,
            "R2_no_family_regression_gt_1": no_regress,
            "R3_p3_nl_copy_ge_4": p3_nl_kept,
        },
        "regressions": regressions,
        "comparison": checks,
        "candidate_probe": candidate,
        "baseline_probe": baseline,
        "note": "Training proposes; evaluation disposes. Connector never "
                "mutates weights; this gate is the promotion authority.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(json.dumps({k: verdict[k] for k in
                      ("verdict", "rules", "regressions")}, indent=2))
    print("\n| family | proto | baseline | candidate |")
    print("|---|---|---|---|")
    for c in checks:
        print(f"| {c['family']} | {c['protocol']} | {c['baseline']} "
              f"| {c['candidate']} ({c['delta']:+d}) |")
    print(f"\nwrote {args.out}")
    return 0 if promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
