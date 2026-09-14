"""FORMATION-MUX-001 Science-S5 worker: one arm, one GPU, public data only."""
from __future__ import annotations
import argparse, json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXIT_OK, EXIT_ARM_LOCAL, EXIT_GLOBAL = 0, 3, 4
GLOBAL_MARKERS = (
    "identity mismatch", "identity drift", "manifest hash", "shortcut screen",
    "not registered", "not preregistered", "R0 token ids missing",
    "production-tokenizer surface invalid", "latent scientific token escaped",
    "protocol", "SEALED_FIREWALL_BREACH", "public surface", "S5 public",
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, choices=("CS-MECH-002", "REP-FORM-003A"))
    p.add_argument("--arm", required=True)
    p.add_argument("--seed-bundle", type=int, required=True)
    p.add_argument("--surface", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--engineering-only", action="store_true")
    p.add_argument("--a-updates-override", type=int, default=None)
    p.add_argument("--b-token-budget-override", type=int, default=None)
    args = p.parse_args(argv)
    started = time.monotonic()
    try:
        import torch
        from anra_v5 import formation_mux_train_v5 as train
        from v5_experiments import formation_mux_protocol_v5 as proto
        from v5_experiments.formation_mux_surface_v5 import load_public_surface
        arms = proto.ARMS_A if args.experiment == proto.EXPERIMENT_A else proto.ARMS_B
        if args.arm not in arms:
            raise RuntimeError(f"arm not registered: {args.experiment}/{args.arm}")
        if args.seed_bundle not in proto.SEED_BUNDLES and not args.engineering_only:
            raise RuntimeError(f"seed bundle not preregistered: {args.seed_bundle}")
        if (args.a_updates_override is not None or args.b_token_budget_override is not None) and not args.engineering_only:
            raise RuntimeError("protocol override is allowed only in engineering-only mode")
        surface = load_public_surface(args.surface)
        if "sealed" in surface.get("splits", {}):
            raise RuntimeError("SEALED_FIREWALL_BREACH: worker-visible manifest contains sealed rows")
        body = train.train_arm(
            experiment=args.experiment, arm=args.arm, seed_bundle=args.seed_bundle,
            surface=surface, out_dir=args.out, torch=torch,
            device=torch.device(args.device), engineering_only=args.engineering_only,
            a_updates_override=args.a_updates_override,
            b_token_budget_override=args.b_token_budget_override,
            progress=lambda m: print(m, flush=True),
        )
        print("WORKER_RECEIPT " + json.dumps({
            "schema": "anra.formation-mux-worker/v5",
            "experiment": args.experiment, "arm": args.arm,
            "seed_bundle": args.seed_bundle,
            "engineering_only": bool(args.engineering_only),
            "visible_devices_inside_worker": torch.cuda.device_count(),
            "sealed_rows_visible": False, "status": body["status"],
            "updates": body["updates"], "processed_tokens": body["processed_tokens"],
            "timing": body["timing"], "wall_seconds": time.monotonic() - started,
        }), flush=True)
        return EXIT_OK
    except RuntimeError as exc:
        message = str(exc)
        code = EXIT_GLOBAL if any(x in message for x in GLOBAL_MARKERS) else EXIT_ARM_LOCAL
        print(("GLOBAL_FAILURE " if code == EXIT_GLOBAL else "ARM_LOCAL_FAILURE ") + json.dumps({
            "schema": "anra.formation-mux-worker-failure/v5",
            "class": "GLOBAL_INTEGRITY_FAILURE" if code == EXIT_GLOBAL else "ARM_LOCAL_ENGINEERING_FAILURE",
            "exception": type(exc).__name__, "message": message,
            "traceback": traceback.format_exc(),
        }), flush=True)
        return code
    except Exception as exc:
        print("GLOBAL_FAILURE " + json.dumps({
            "exception": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        }), flush=True)
        return EXIT_GLOBAL


if __name__ == "__main__":
    raise SystemExit(main())
