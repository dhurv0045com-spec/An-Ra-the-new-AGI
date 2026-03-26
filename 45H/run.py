"""
================================================================================
FILE: run.py
PROJECT: Transformer Language Model — 45H Final Phase
PURPOSE: CLI entry point — train, generate, or evaluate from the command line
================================================================================

Examples:

  # Generate text (no checkpoint needed — uses random weights for demo)
  python run.py --prompt "Once upon a time" --max_tokens 100

  # Generate with a checkpoint
  python run.py --prompt "The future of AI" \\
                --checkpoint best_model.npy \\
                --config config/small.yaml \\
                --temperature 0.8 \\
                --top_p 0.95 \\
                --max_tokens 200

  # Train from scratch
  python run.py --mode train --config config/tiny.yaml

  # Resume training
  python run.py --mode train --config config/small.yaml \\
                --train.resume_from checkpoints/step_010000.npy

  # Evaluate
  python run.py --mode evaluate --config config/small.yaml \\
                --checkpoint best_model.npy \\
                --eval_data data/test.txt
================================================================================
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Ensure parent directory is on path (for library usage without pip install)
sys.path.insert(0, os.path.dirname(__file__))

from model         import LanguageModel
from config_loader import ConfigError
from logger        import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Transformer Language Model — train, generate, or evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --prompt "Hello world" --max_tokens 50
  python run.py --mode train --config config/tiny.yaml
  python run.py --mode evaluate --checkpoint best_model.npy --eval_data test.txt
        """,
    )

    # Mode
    parser.add_argument(
        "--mode", choices=["generate", "train", "evaluate"], default="generate",
        help="Operation mode (default: generate)"
    )

    # Config
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config file. Default: config/tiny.yaml"
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to model checkpoint (.npy file)"
    )

    # Generation args
    parser.add_argument("--prompt",      default="The ", help="Input prompt for generation")
    parser.add_argument("--max_tokens",  type=int,   default=None, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top_k",       type=int,   default=None, help="Top-k filter")
    parser.add_argument("--top_p",       type=float, default=None, help="Nucleus sampling threshold")
    parser.add_argument("--strategy",    default=None,
                        choices=["greedy", "temperature", "top_k", "top_p"],
                        help="Generation strategy")
    parser.add_argument("--repetition_penalty", type=float, default=None,
                        help="Repetition penalty (1.0 = none)")

    # Evaluation
    parser.add_argument("--eval_data",  default=None, help="Path to evaluation text file")
    parser.add_argument("--eval_batches", type=int, default=100, help="Max eval batches")

    # Training
    parser.add_argument("--resume",     default=None, help="Checkpoint path to resume from")

    # Logging
    parser.add_argument("--log_level",  default=None, choices=["DEBUG","INFO","WARNING","ERROR"])
    parser.add_argument("--no_log_file", action="store_true", help="Disable file logging")

    # Config overrides (passed through as "section.key=value")
    parser.add_argument("overrides", nargs="*",
                        help="Config overrides: section.field=value (e.g. train.lr=1e-3)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Apply log_level override to overrides list
    overrides = list(args.overrides)
    if args.log_level:
        overrides.append(f"logging.level={args.log_level}")
    if args.no_log_file:
        overrides.append("logging.log_to_file=false")

    # Build model
    try:
        lm = LanguageModel(
            config_path=args.config or "config/tiny.yaml",
            overrides=overrides if overrides else None,
            checkpoint=args.checkpoint,
        )
    except ConfigError as e:
        print(f"\n[CONFIG ERROR]\n{e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[FILE NOT FOUND]\n{e}", file=sys.stderr)
        sys.exit(1)

    log = logging.getLogger(__name__)

    # ── Dispatch ──────────────────────────────────────────────────────────

    if args.mode == "generate":
        prompt = args.prompt
        if not prompt or not prompt.strip():
            print("Error: --prompt cannot be empty.", file=sys.stderr)
            sys.exit(1)

        print(f"\n{'─'*60}")
        print(f"  Model:  {lm}")
        print(f"  Prompt: {prompt!r}")
        print(f"{'─'*60}\n")

        gen_kwargs = {}
        if args.max_tokens  is not None: gen_kwargs["max_new_tokens"]  = args.max_tokens
        if args.temperature is not None: gen_kwargs["temperature"]     = args.temperature
        if args.top_k       is not None: gen_kwargs["top_k"]           = args.top_k
        if args.top_p       is not None: gen_kwargs["top_p"]           = args.top_p
        if args.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = args.repetition_penalty

        output = lm.generate(prompt, **gen_kwargs)
        print(output)
        print(f"\n{'─'*60}")
        print(f"  Generated {len(output) - len(prompt)} chars beyond prompt")
        print(f"{'─'*60}\n")

    elif args.mode == "train":
        log.info("Starting training...")
        results = lm.train(resume=args.resume)
        print(f"\n{'─'*60}")
        print(f"  Training complete")
        print(f"  Final loss:    {results['final_loss']:.4f}" if results["final_loss"] else "")
        print(f"  Best val loss: {results['best_val_loss']:.4f}")
        print(f"  Total steps:   {results['total_steps']:,}")
        print(f"  Elapsed:       {results['elapsed_seconds']:.0f}s")
        print(f"{'─'*60}\n")

    elif args.mode == "evaluate":
        eval_path = args.eval_data
        if not eval_path:
            cfg_path = lm.cfg.get("train.val_dataset_path", "data/val.txt")
            eval_path = cfg_path
            log.info(f"No --eval_data specified, using config path: {eval_path}")

        results = lm.evaluate(eval_path, max_batches=args.eval_batches)
        print(f"\n{'─'*60}")
        print(f"  Evaluation Results")
        print(f"  Loss:        {results['loss']:.4f}")
        print(f"  Perplexity:  {results['perplexity']:.2f}")
        print(f"  Tokens:      {results['num_tokens']:,}")
        print(f"  Batches:     {results['num_batches']}")
        print(f"  Time:        {results['elapsed_seconds']:.2f}s")
        print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
