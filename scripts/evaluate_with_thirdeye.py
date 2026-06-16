"""Run AN-RA's one-click ThirdEye inventory and evidence-gap evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.thirdeye_adapter import run_one_click
from training.v2_runtime import build_frontier_model, model_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AN-RA with ThirdEye")
    parser.add_argument(
        "--profile",
        choices=["quick", "standard", "exhaustive", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--without-model",
        action="store_true",
        help="Skip construction of the 900M frontier model and report only artifact probes.",
    )
    args = parser.parse_args()
    model = None
    if not args.without_model:
        model = build_frontier_model()
        summary = model_summary(model)
        if not 850_000_000 <= int(summary["parameters"]) <= 1_000_000_000:
            raise RuntimeError(f"Unexpected 900M-class frontier parameter count: {summary}")
    result = run_one_click(profile=args.profile, model=model)
    print(
        json.dumps(
            {
                "project": result["project"]["project_id"],
                "features": len(result["features"]),
                "recommended_experiments": len(result["recommended_experiments"]),
                "activation_snapshot": result["activation_snapshot"],
                "reports": result["report_paths"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
