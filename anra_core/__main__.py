from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .brain import Brain, ThoughtPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the inference-only An-Ra V4 dense core")
    parser.add_argument("--checkpoint", default=os.environ.get("ANRA_CHECKPOINT_PATH"))
    parser.add_argument(
        "--tokenizer",
        default=str(Path(__file__).parent / "assets" / "tokenizer_v4_32k.json"),
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("direct", "deliberate"), default="direct")
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("provide --checkpoint or ANRA_CHECKPOINT_PATH")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    brain = Brain.from_checkpoint(args.checkpoint, args.tokenizer, device=device)
    if args.describe:
        print(brain.describe())
    thought = brain.think(
        args.prompt,
        ThoughtPolicy(
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            candidates=args.candidates,
            seed=args.seed,
        ),
    )
    print(thought.text)
    print(
        f"\n[mode={thought.mode} self_likelihood={thought.self_likelihood:.4f} "
        f"step={thought.checkpoint_step}]",
        file=__import__("sys").stderr,
    )


if __name__ == "__main__":
    main()
