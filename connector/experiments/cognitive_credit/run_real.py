"""Run the cognitive credit experiment against a real Core checkpoint.

Usage (from repo root):
    py -3.14 -m connector.experiments.cognitive_credit.run_real \
        --checkpoint "C:\\path\\to\\anra-v4-tpu-latest.pt" [--device cpu]

Contract: the completer returns raw CompletionResult outputs only. It executes
every candidate an honest best-of-N policy requests and invokes the real tool
adapter, feeding its actual output into the prompt. Success is decided solely
by the runner's verifier.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from connector.experiments.cognitive_credit.case import Attempt, CompletionResult
from connector.experiments.cognitive_credit.capability_probe import (
    family_gates,
    run_probe,
)
from connector.experiments.cognitive_credit.runner import (
    render_table,
    run_experiment,
)


def make_core_completer(executor, tokenizer):
    """Real Core execution: outputs only, honest candidate counts.

    Executes ``PreparedExecution`` — the exact resolved prompt — and passes
    every decode-policy parameter explicitly to generate(). Nothing behavior-
    changing is left as an invisible default.
    """
    from anra_core.errors import CoreError
    from anra_core.generate import generate

    from connector.experiments.cognitive_credit.case import (
        PreparedExecution,
        ToolUnavailableError,
    )

    stats = {"generations": 0, "core_errors": 0, "tool_calls": 0, "tool_failures": 0}

    def complete(attempt: Attempt) -> CompletionResult:
        prepared = PreparedExecution.from_attempt(attempt)
        prompt = prepared.prompt
        policy = prepared.decode
        n = max(1, policy.candidates) if policy.temperature > 0 else 1
        texts: list[str] = []
        for index in range(n):
            try:
                texts.append(
                    generate(
                        executor,
                        tokenizer,
                        prompt,
                        max_new_tokens=policy.max_new_tokens,
                        temperature=policy.temperature,
                        top_p=policy.top_p,
                        seed=policy.seed + index,
                        repetition_penalty=policy.repetition_penalty,
                        no_repeat_ngram_size=policy.no_repeat_ngram_size,
                    )
                )
                stats["generations"] += 1
            except CoreError as exc:
                stats["core_errors"] += 1
                return CompletionResult(
                    texts=(),
                    n_executions=max(1, len(texts)),
                    error=type(exc).__name__,
                )
        return CompletionResult(texts=tuple(texts), n_executions=n)

    return complete, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="")
    parser.add_argument("--skip-gate", action="store_true",
                        help="run even if the capability floor fails")
    args = parser.parse_args()

    from anra_core.executor import CoreExecutor

    print(f"loading checkpoint: {args.checkpoint}", flush=True)
    t0 = time.time()
    executor = CoreExecutor.from_checkpoint(args.checkpoint, device=args.device)
    tokenizer = executor.tokenizer
    assert tokenizer is not None
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    # Capability gate (probe v2): primitives first, family-specific. If a
    # family's prerequisite primitive has no evidence, that family is not
    # runnable — scoring it would produce unattributable results.
    print("running capability probe (v2, raw profile)...", flush=True)
    probe = run_probe(args.checkpoint, args.device, profile="raw")
    gates = family_gates(probe)
    runnable = [family for family, ok in gates.items() if ok]
    blocked = [family for family, ok in gates.items() if not ok]
    if not runnable and not args.skip_gate:
        payload = {
            "verdict": "substrate below experimental floor",
            "probe": probe,
            "family_gates": gates,
            "note": "no cognitive-credit family is runnable on this substrate; "
                    "interventions would not be interpretable",
        }
        text = json.dumps(payload, indent=2)
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(text)
        return 2

    complete, stats = make_core_completer(executor, tokenizer)
    t1 = time.time()
    summary = run_experiment(complete)
    wall = time.time() - t1

    payload = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "capability_probe": probe,
        "family_gates": gates,
        "runnable_families": runnable,
        "blocked_families": blocked,
        "decode_profile": "assisted (generate defaults: penalty 1.15, ngram 4)",
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
