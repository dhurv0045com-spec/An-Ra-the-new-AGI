#!/usr/bin/env python3
"""Run one real M102 forward/backward/update on a selected target device.

This is a bounded engineering smoke using deterministic synthetic token IDs.
It verifies the executable 100M-class model and optimizer path; its receipt is
not data, capability, or production-training evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _candidate_specs() -> dict[str, Any]:
    from signac_100m.spec import (
        MODEL_SPEC,
        TPU_DEPTH_PRESERVING_CHALLENGER,
        TPU_TILED_CHALLENGER,
    )

    return {
        "m102_primary": MODEL_SPEC,
        "tpu_depth_preserving_challenger": TPU_DEPTH_PRESERVING_CHALLENGER,
        "tpu_tiled_challenger": TPU_TILED_CHALLENGER,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(_candidate_specs()), default="m102_primary")
    parser.add_argument("--device", choices=("cpu", "cuda", "xla"), default="cpu")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73_011)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--verify-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or args.batch_size <= 0 or args.sequence_length < 2 or args.threads <= 0:
        parser.error("seed, batch size, sequence length, and thread count must be positive/in range")

    import torch

    torch.set_num_threads(args.threads)
    from v5_training.target_canary import run_target_canary

    spec = _candidate_specs()[args.candidate]
    if args.sequence_length > spec.context_length:
        parser.error("sequence length exceeds the selected candidate's native context")
    receipt = run_target_canary(
        model_spec=spec,
        device=args.device,
        workdir=args.workdir,
        seed=args.seed,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        verify_checkpoint=args.verify_checkpoint,
        verify_continuation=args.verify_checkpoint,
        torch_module=torch,
    )
    receipt["candidate"] = args.candidate
    receipt["synthetic_input"] = True
    receipt["production_training_authorized"] = False
    receipt["claim_ceiling"] = (
        "Executable model/update plumbing only; no qualified corpus, capability, TPU-fit, or AGI result."
    )
    output_path = args.workdir / "signac_model_smoke_receipt.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "candidate": args.candidate,
        "parameter_count": receipt["parameter_count"],
        "stages": receipt["stages"],
        "receipt": str(output_path.resolve()),
    }, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
