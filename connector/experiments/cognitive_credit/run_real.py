"""Run the cognitive credit experiment against a real Core checkpoint.

Usage (from repo root):
    py -3.14 -m connector.experiments.cognitive_credit.run_real \
        --checkpoint "C:\\path\\to\\anra-v4-tpu-latest.pt" [--device cpu]

The completer executes real Core generation. The diagnostician never sees
HiddenGroundTruth; only scoring does.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from connector.experiments.cognitive_credit.case import Attempt
from connector.experiments.cognitive_credit.runner import (
    Completer,
    render_table,
    run_experiment,
)


def make_core_completer(executor, tokenizer) -> tuple[Completer, dict[str, int]]:
    """Real Core execution with per-call error capture for diagnosis."""
    from anra_core.errors import CoreError

    stats = {"executions": 0, "core_errors": 0}

    def complete(attempt: Attempt) -> tuple[bool, str]:
        from anra_core.generate import generate

        stats["executions"] += 1
        try:
            text = generate(
                executor,
                tokenizer,
                attempt.render(),
                max_new_tokens=attempt.decode.max_new_tokens,
                temperature=attempt.decode.temperature,
                top_p=attempt.decode.top_p,
                seed=attempt.decode.seed,
            )
        except CoreError as exc:
            stats["core_errors"] += 1
            return False, f"<core-error {type(exc).__name__}>"
        return False, text  # success decided by verifier upstream

    return complete, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from anra_core.executor import CoreExecutor

    print(f"loading checkpoint: {args.checkpoint}", flush=True)
    t0 = time.time()
    executor = CoreExecutor.from_checkpoint(args.checkpoint, device=args.device)
    tokenizer = executor.tokenizer
    assert tokenizer is not None
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    complete, stats = make_core_completer(executor, tokenizer)
    t1 = time.time()
    summary = run_experiment(complete)
    wall = time.time() - t1

    payload = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "wall_seconds": round(wall, 1),
        "completer_stats": stats,
        **summary,
    }
    text = json.dumps(payload, indent=2)
    print(render_table(summary))
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
