from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .checkpoint import load_core_checkpoint
from .generate import generate
from .tokenizer import V4Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the inference-only An-Ra V4 dense core")
    parser.add_argument("--checkpoint", default=os.environ.get("ANRA_CHECKPOINT_PATH"))
    parser.add_argument("--tokenizer", default=str(Path(__file__).parents[1] / "tokenizer" / "tokenizer_v4_32k.json"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("provide --checkpoint or ANRA_CHECKPOINT_PATH")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model, metadata = load_core_checkpoint(args.checkpoint)
    model.to(device)
    tokenizer = V4Tokenizer.load(args.tokenizer)
    print(
        generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        )
    )
    step = metadata.get("global_step", metadata.get("step"))
    if step is not None:
        print(f"\n[checkpoint step {step}]", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
