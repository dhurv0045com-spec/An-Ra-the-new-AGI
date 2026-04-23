from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from anra_brain import CausalTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="anra_brain.pt")
    ap.add_argument("--output", default="anra_ouroboros.pt")
    ap.add_argument("--steps", type=int, default=5000)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    with open(root / "tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)

    model = CausalTransformer(tok.vocab_size, 256, 4, 4, 128)
    state = torch.load(args.base_model, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)

    # Placeholder wiring: preserves compatibility and emits staged-training intent.
    payload = {
        "model_state_dict": model.state_dict(),
        "global_step": 0,
        "phase": "A/B/C schedule delegated to phase3/ouroboros (45O)/train_ouroboros.py",
        "requested_steps": args.steps,
    }
    torch.save(payload, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
