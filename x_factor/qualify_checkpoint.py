"""Single entry point for future stronger checkpoints (Mission 10).

Usage (from repo root):
  python x_factor/qualify_checkpoint.py --checkpoint PATH [--level quick|standard]

1. verifies provenance + arch/tokenizer compat
2. runs software canaries (import/shape)
3. runs bounded readiness probes (difficulty ladder, raw+oracle)
4. identifies PARTIAL region
5. returns READY_FOR_BINDING_CAUSAL_RESEARCH or NOT_READY_* with reasons.

FAILs loudly on missing checkpoint (no silent fallback).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys as _sys

_XF = Path(__file__).resolve().parent
if str(_XF) not in _sys.path:
    _sys.path.insert(0, str(_XF))

from checkpoint_identity import load_core, resolve_checkpoint  # noqa: E402
from readiness.gate import run_gate  # noqa: E402
from readiness.ladder import RUNGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--level", default="quick", choices=["quick", "standard"])
    ap.add_argument("--seed", type=int, default=42424)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="output/readiness_receipt.json")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = str(resolve_checkpoint(args.checkpoint))  # fails loudly if absent
    # compat canary
    model, tok, payload = load_core(ckpt, device)
    cfg = payload["model_config"]
    assert cfg.get("vocab_size") == 32768, "tokenizer/vocab mismatch"
    assert cfg.get("n_layers") == 18 and cfg.get("d_model") == 896, "arch mismatch"
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.level == "quick":
        rungs, n = ("B0", "B1", "B2", "B3"), 12
    else:
        rungs, n = tuple(RUNGS), 16
    receipt = run_gate(ckpt, args.seed, n, device, rungs=rungs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"classification": receipt["classification"],
                      "partial_rungs": receipt["partial_rungs"],
                      "blockers": receipt["blockers"],
                      "permitted": receipt["permitted_next_experiments"]}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
