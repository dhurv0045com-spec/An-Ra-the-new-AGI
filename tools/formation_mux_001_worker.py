"""FORMATION-MUX-001 worker: one arm, one GPU, one process.

The Kaggle operator spawns exactly one worker per physical T4 with
``CUDA_VISIBLE_DEVICES`` pinned by the parent, so a worker can only ever
see its own GPU (no cross-GPU allocation is possible from inside).

Exit codes:
    0  arm COMPLETE (or SKIPPED_COMPLETED_ARM)
    3  arm-local ENGINEERING_FAILURE (receipt preserved; the matched
       comparison is incomplete, unrelated work continues)
    4  GLOBAL integrity violation (campaign must fail closed)
    2  usage/argument error
"""

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True,
                        choices=("CS-MECH-002", "REP-FORM-003A"))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed-bundle", type=int, required=True)
    parser.add_argument("--surface", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--updates-override", type=int, default=None,
                        help="ENGINEERING_ONLY tiny-run override; official "
                             "runs must not pass this")
    parser.add_argument("--engineering-only", action="store_true",
                        help="label the run ENGINEERING_ONLY (incapable of "
                             "an official verdict)")
    parser.add_argument("--device", default="cuda",
                        help="cuda (default; local cuda:0 after the parent "
                             "pins CUDA_VISIBLE_DEVICES) or cpu for "
                             "qualification only")
    parser.add_argument("--deadline-minutes", type=float, default=None)
    args = parser.parse_args(argv)

    started = time.monotonic()
    try:
        import torch
        from anra_v5 import formation_mux_train as train
        from v5_experiments import formation_mux_protocol as proto
        from v5_experiments.formation_mux_data import load_surface

        surface = load_surface(args.surface)
        valid_arms = (proto.ARMS_A if args.experiment == proto.EXPERIMENT_A
                      else proto.ARMS_B)
        if args.arm not in valid_arms:
            print("GLOBAL_FAILURE " + json.dumps(
                {"reason": f"arm {args.arm} not registered for "
                           f"{args.experiment}"}))
            return EXIT_GLOBAL
        if args.seed_bundle not in proto.SEED_BUNDLES:
            print("GLOBAL_FAILURE " + json.dumps(
                {"reason": f"seed bundle {args.seed_bundle} not "
                           "preregistered"}))
            return EXIT_GLOBAL
        device = torch.device(args.device)
        deadline = (time.monotonic() + args.deadline_minutes * 60.0
                    if args.deadline_minutes else None)
        body = train.train_arm(
            experiment=args.experiment, arm=args.arm,
            seed_bundle=args.seed_bundle, surface=surface, out_dir=args.out,
            torch=torch, device=device,
            updates=args.updates_override, deadline=deadline,
            progress=lambda m: print(m, flush=True))
        receipt = {
            "schema": "anra.formation-mux-worker/v1",
            "experiment": args.experiment, "arm": args.arm,
            "seed_bundle": args.seed_bundle,
            "engineering_only": bool(args.engineering_only
                                     or args.updates_override is not None
                                     or args.device == "cpu"),
            "device": str(device), "visible_devices":
                __import__("torch").cuda.device_count(),
            "status": body["status"],
            "wall_seconds": round(time.monotonic() - started, 1)}
        print("WORKER_RECEIPT " + json.dumps(receipt), flush=True)
        return EXIT_OK
    except RuntimeError as exc:
        message = str(exc)
        global_markers = ("identity mismatch", "identity drift", "hash mismatch",
                          "manifest hash", "not registered", "not preregistered",
                          "shortcut screen", "protocol")
        if any(marker in message for marker in global_markers):
            print("GLOBAL_FAILURE " + json.dumps(
                {"exception": type(exc).__name__, "message": message,
                 "traceback": traceback.format_exc()}), flush=True)
            return EXIT_GLOBAL
        failure = {"schema": "anra.formation-mux-arm-failure/v1",
                   "class": "ARM_LOCAL_ENGINEERING_FAILURE",
                   "exception": type(exc).__name__, "message": message,
                   "traceback": traceback.format_exc(),
                   "wall_seconds": round(time.monotonic() - started, 1)}
        try:
            root = Path(args.out) / args.experiment / args.arm / \
                f"S{args.seed_bundle}"
            root.mkdir(parents=True, exist_ok=True)
            (root / "FAILURE.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8")
        except Exception:
            pass
        print("ARM_LOCAL_FAILURE " + json.dumps(failure), flush=True)
        return EXIT_ARM_LOCAL
    except Exception as exc:  # noqa: BLE001 - fail closed with evidence
        print("GLOBAL_FAILURE " + json.dumps(
            {"exception": type(exc).__name__, "message": str(exc),
             "traceback": traceback.format_exc()}), flush=True)
        return EXIT_GLOBAL


if __name__ == "__main__":
    raise SystemExit(main())
