"""Evaluate the native draft proof before allowing frontier rescue."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tokenizer.subword_tokenizer import SubwordTokenizer
from training.eval_v2 import run_compact_eval
from training.v2_config import V2ModelConfig
from training.v2_runtime import build_model_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = V2ModelConfig(**checkpoint["model_config"])
    tokenizer = SubwordTokenizer.from_pretrained(args.tokenizer)
    model = build_model_from_config(
        config,
        block_size=config.block_size,
        vocab_size=tokenizer.vocab_size,
    )
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"draft checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    evaluation = run_compact_eval(model, tokenizer, device=device, output=False, seed=0)
    coherence = float(evaluation.get("coherence_rate", 0.0))
    validation_loss = float(checkpoint.get("validation_loss", float("inf")))
    report = {
        "schema_version": 2,
        "checkpoint": str(args.checkpoint),
        "coherence_rate": coherence,
        "coherence_basis": evaluation.get("coherence_basis", "unknown"),
        "repetition_failure_rate": float(evaluation.get("repetition_failure_rate", 0.0)),
        "decoding": evaluation.get("decoding", {}),
        "validation_loss": validation_loss,
        "finite_validation": math.isfinite(validation_loss),
        "evaluation": evaluation,
        "passed": math.isfinite(validation_loss) and coherence >= 0.60,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
