from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from anra.anra_paths import OUTPUT_V2_DIR, ROOT

SMOKE_PROMPTS = [
    "H: Who are you?\nANRA:",
    "H: Explain your current training state in three short sentences.\nANRA:",
    "H: Give three steps to debug a failing Python test.\nANRA:",
    "H: Differentiate x^2 + 3*x.\nANRA:",
    "H: Remember cobalt-19. What key did I give you and why does context matter?\nANRA:",
]


def _jsonl_append(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _trace_payload(prompt: str, trace, info: dict[str, object], session_id: str) -> dict[str, object]:
    entropy_avg = sum(trace.entropy_curve) / max(1, len(trace.entropy_curve))
    max_prob_avg = sum(trace.max_prob_curve) / max(1, len(trace.max_prob_curve))
    return {
        "ts": time.time(),
        "session_id": session_id,
        "checkpoint": info.get("checkpoint"),
        "profile": info.get("profile"),
        "device": info.get("device"),
        "param_count": info.get("param_count"),
        "prompt": prompt,
        "response": trace.output,
        "strategy": trace.strategy,
        "tokens_generated": trace.tokens_generated,
        "time_ms": trace.time_ms,
        "entropy_avg": entropy_avg,
        "max_prob_avg": max_prob_avg,
        "stopped_by": trace.stopped_by,
        "repeated_ngrams_detected": trace.repeated_ngrams_detected,
        "ghost_state_loaded": trace.ghost_state_loaded,
    }


def _run_prompt(prompt: str, *, cfg, session_id: str, output: Path) -> None:
    from generate import generate_traced, get_model_info

    info = get_model_info()
    trace = generate_traced(prompt, cfg, session_id=session_id)
    payload = _trace_payload(prompt, trace, info, session_id)
    _jsonl_append(output, payload)
    print("\nH/PROMPT:")
    print(prompt)
    print("\nANRA:")
    print(trace.output)
    print(
        f"\n[trace] tokens={trace.tokens_generated} "
        f"time_ms={trace.time_ms:.1f} stopped_by={trace.stopped_by} "
        f"repetition={trace.repeated_ngrams_detected}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the iterate500 frontier runtime.")
    parser.add_argument("--checkpoint", default="", help="Optional trained checkpoint path.")
    parser.add_argument("--profile", default="frontier", choices=["frontier", "auto"])
    parser.add_argument("--suite", choices=["smoke"], default=None)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--session-id", default="frontier_manual")
    parser.add_argument("--strategy", default="nucleus")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument(
        "--output",
        default=str(OUTPUT_V2_DIR / "frontier_chat_traces.jsonl"),
    )
    args = parser.parse_args()

    if args.profile == "frontier":
        os.environ["ANRA_MODEL_PROFILE"] = "frontier"
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        os.environ["ANRA_CHECKPOINT_PATH"] = str(checkpoint)

    from generate import GenerationConfig

    cfg = GenerationConfig(
        strategy=args.strategy,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    if args.prompt:
        _run_prompt(args.prompt, cfg=cfg, session_id=args.session_id, output=output)
        return
    if args.suite == "smoke":
        for prompt in SMOKE_PROMPTS:
            _run_prompt(prompt, cfg=cfg, session_id=args.session_id, output=output)
        return
    if args.interactive:
        print("Frontier chat. Ctrl+C or an empty line exits.")
        while True:
            try:
                message = input("\nH: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not message:
                return
            prompt = message if message.startswith("H:") else f"H: {message}\nANRA:"
            _run_prompt(prompt, cfg=cfg, session_id=args.session_id, output=output)
        return
    parser.error("choose --prompt, --suite smoke, or --interactive")


if __name__ == "__main__":
    main()
