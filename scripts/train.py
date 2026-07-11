#!/usr/bin/env python3
"""Fail-closed migration shim for the removed legacy standalone trainer.

The old implementation used an adjacent-token validation split, could invent a
synthetic corpus, could switch to an incompatible character tokenizer, and
wrote checkpoints outside the canonical schema. Keeping that path runnable
would make it too easy to reproduce the failed legacy training conditions.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-migration", action="store_true")
    args, unknown = parser.parse_known_args()
    guidance = {
        "status": "legacy_trainer_removed",
        "reason": (
            "This entrypoint cannot satisfy immutable data, tokenizer, validation, "
            "checkpoint-boundary, and campaign-phase contracts."
        ),
        "campaign_status": "python -m scripts.campaign_status --phase pilot",
        "canonical_gpu_training": "python -m training.train_unified --help",
        "direct_phase_trainer": "python -m scripts.build_brain --help",
        "received_legacy_arguments": unknown,
    }
    print(json.dumps(guidance, indent=2))
    if args.show_migration:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
