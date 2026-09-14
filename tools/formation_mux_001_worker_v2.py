"""FORMATION-MUX-001 Amendment-1 worker: one arm, one process, one GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXIT_OK = 0
EXIT_ARM_LOCAL = 3
EXIT_GLOBAL = 4
EXIT_USAGE = 2

GLOBAL_MARKERS = (
    "identity mismatch",
    "identity drift",
    "manifest hash",
    "shortcut screen",
    "not registered",
    "not preregistered",
    "R0 token ids missing",
    "production-tokenizer surface invalid",
    "latent scientific token escaped",
    "protocol",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=("CS-MECH-002", "REP-FORM-003A"))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed-bundle", type=int, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--engineering-only", action="store_true")
    parser.add_argument("--a-updates-override", type=int, default=None)
    parser.add_argument("--b-token-budget-override", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.monotonic()
    try:
        import torch
        from anra_v5 import formation_mux_train_v2 as train
        from v5_experiments import formation_mux_protocol_v2 as proto
        from v5_experiments.formation_mux_data import load_surface

        valid_arms = proto.ARMS_A if args.experiment == proto.EXPERIMENT_A else proto.ARMS_B
        if args.arm not in valid_arms:
            raise RuntimeError(f"arm not registered: {args.experiment}/{args.arm}")
        if args.seed_bundle not in proto.SEED_BUNDLES and not args.engineering_only:
            raise RuntimeError(f"seed bundle not preregistered: {args.seed_bundle}")
        if (args.a_updates_override is not None or args.b_token_budget_override is not None) and not args.engineering_only:
            raise RuntimeError("protocol override is allowed only in engineering-only mode")

        surface = load_surface(args.surface)
        device = torch.device(args.device)
        body = train.train_arm(
            experiment=args.experiment,
            arm=args.arm,
            seed_bundle=args.seed_bundle,
            surface=surface,
            out_dir=args.out,
            torch=torch,
            device=device,
            engineering_only=args.engineering_only,
            a_updates_override=args.a_updates_override,
            b_token_budget_override=args.b_token_budget_override,
            progress=lambda m: print(m, flush=True),
        )
        receipt = {
            "schema": "anra.formation-mux-worker/v2",
            "experiment": args.experiment,
            "arm": args.arm,
            "seed_bundle": args.seed_bundle,
            "engineering_only": bool(args.engineering_only),
            "device": str(device),
            "visible_devices_inside_worker": torch.cuda.device_count(),
            "status": body["status"],
            "updates": body["updates"],
            "processed_tokens": body["processed_tokens"],
            "timing": body["timing"],
            "wall_seconds": time.monotonic() - started,
        }
        print("WORKER_RECEIPT " + json.dumps(receipt), flush=True)
        return EXIT_OK
    except RuntimeError as exc:
        message = str(exc)
        classification = EXIT_GLOBAL if any(marker in message for marker in GLOBAL_MARKERS) else EXIT_ARM_LOCAL
        payload = {
            "schema": "anra.formation-mux-worker-failure/v2",
            "class": "GLOBAL_INTEGRITY_FAILURE" if classification == EXIT_GLOBAL else "ARM_LOCAL_ENGINEERING_FAILURE",
            "exception": type(exc).__name__,
            "message": message,
            "traceback": traceback.format_exc(),
            "wall_seconds": time.monotonic() - started,
        }
        print(("GLOBAL_FAILURE " if classification == EXIT_GLOBAL else "ARM_LOCAL_FAILURE ") + json.dumps(payload), flush=True)
        return classification
    except Exception as exc:  # fail closed on unknown failures
        payload = {
            "schema": "anra.formation-mux-worker-failure/v2",
            "class": "GLOBAL_INTEGRITY_FAILURE",
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "wall_seconds": time.monotonic() - started,
        }
        print("GLOBAL_FAILURE " + json.dumps(payload), flush=True)
        return EXIT_GLOBAL


if __name__ == "__main__":
    raise SystemExit(main())
