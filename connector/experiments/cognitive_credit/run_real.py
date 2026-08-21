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
from connector.experiments.cognitive_credit.capability_probe import run_probe
from connector.experiments.cognitive_credit.runner import (
    render_table,
    run_experiment,
)


def make_core_completer(executor, tokenizer):
    """Real Core execution: outputs only, honest candidate counts, live tools."""
    from anra_core.errors import CoreError
    from anra_core.generate import generate

    from connector.experiments.cognitive_credit.case import ToolUnavailableError

    stats = {"generations": 0, "core_errors": 0, "tool_calls": 0, "tool_failures": 0}

    def complete(attempt: Attempt) -> CompletionResult:
        prompt = attempt.render()
        if attempt.tool is not None:
            try:
                stats["tool_calls"] += 1
                output = attempt.tool.run()
                prompt += f"\n<tool_output>{output}</tool_output>"
            except ToolUnavailableError as exc:
                stats["tool_failures"] += 1
                prompt += f"\n<tool_output>ERROR: {exc}</tool_output>"

        policy = attempt.decode
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

    # Capability gate: primitives first. If the substrate cannot use supplied
    # knowledge, follow plans, copy values, or use tool results — in the
    # experiment's own tag protocol — the cognitive experiment cannot be
    # interpreted: report the floor and stop.
    print("running capability probe...", flush=True)
    probe = run_probe(executor=executor)

    def passes(score: object) -> bool:
        tag = score["tag"] if isinstance(score, dict) else score
        return int(str(tag).split("/")[0]) >= 4

    required = (
        "P1_nonce_knowledge_use",
        "P2_plan_following_no_arithmetic",
        "P4_tool_result_use",
    )
    failed = [name for name in required if not passes(probe[name])]
    if failed and not args.skip_gate:
        payload = {
            "verdict": "substrate below experimental floor",
            "failed_probes": failed,
            "probe": probe,
            "note": "cognitive-credit experiment skipped; interventions would "
                    "not be interpretable on this substrate",
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
