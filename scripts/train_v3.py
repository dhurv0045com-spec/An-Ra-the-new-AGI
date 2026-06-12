#!/usr/bin/env python3
"""Operator entrypoint for resumable AN-RA V3 training campaigns.

Run as: python -m scripts.train_v3 status
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from training.stages import CampaignState, TrainingStage


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or inspect an AN-RA V3 campaign")
    parser.add_argument("action", choices=["status", "run"])
    parser.add_argument("--state", default="output/v3/campaign.json")
    parser.add_argument("--config", default="config/anra_frontier.yaml")
    parser.add_argument("--stage", choices=[stage.value for stage in TrainingStage])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state = CampaignState(ROOT / args.state)
    if args.action == "status":
        print(json.dumps(state.manifest(), indent=2, default=str))
        return 0

    config = state.next_stage() if args.stage is None else next(
        item for item in state.stages if item.stage.value == args.stage
    )
    if config is None:
        print("Campaign already complete.")
        return 0
    command = [
        sys.executable,
        "-m",
        "scripts.train",
        "--config",
        str(ROOT / args.config),
        "--max_steps",
        str(args.max_steps or config.max_steps),
        "--optimizer",
        args.optimizer,
        "--device",
        args.device,
        "--filter_dfc",
    ]
    print(" ".join(command))
    if args.dry_run:
        return 0
    state.update(config.stage, step=0, status="running", checkpoint=None)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    state.update(
        config.stage,
        step=args.max_steps or config.max_steps,
        status="complete" if completed.returncode == 0 else "blocked",
        checkpoint=None,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
