"""One-GPU worker for the preregistered TIE-ROLE frontier extension."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXIT_OK, EXIT_ARM_LOCAL, EXIT_GLOBAL = 0, 3, 4
GLOBAL_MARKERS = (
    "identity mismatch", "identity drift", "manifest hash", "shortcut screen",
    "not registered", "not preregistered", "production-tokenizer surface invalid",
    "latent scientific token escaped", "protocol", "SEALED_FIREWALL_BREACH",
    "public surface", "TIE_ROLE_PILOT_NO_GO",
)


def main(argv=None) -> int:
    from v5_experiments import tie_role_protocol_v1 as proto

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, choices=proto.EXPERIMENTS)
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
        proto.assert_frontier_launch_allowed()
        import torch
        from anra_v5 import tie_role_train_v1 as train
        from v5_experiments.formation_mux_surface_v5 import load_public_surface
        # Architecture: 1 proc : 1 GPU via CUDA_VISIBLE_DEVICES (see S5 worker).
        if str(args.device).startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU required; select Kaggle GPU T4 x2")
            if torch.cuda.device_count() != 1:
                raise RuntimeError(
                    "worker must see exactly one pinned CUDA device; observed "
                    f"{torch.cuda.device_count()} (parent must pin CUDA_VISIBLE_DEVICES)"
                )
            torch.cuda.set_device(0)

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
            experiment=args.experiment,
            arm=args.arm,
            seed_bundle=args.seed_bundle,
            surface=surface,
            out_dir=args.out,
            torch=torch,
            device=torch.device(args.device),
            engineering_only=args.engineering_only,
            a_updates_override=args.a_updates_override,
            b_token_budget_override=args.b_token_budget_override,
            progress=lambda m: print(m, flush=True),
        )
        print("TIE_ROLE_WORKER_RECEIPT " + json.dumps({
            "schema": "anra.tie-role-worker/v1",
            "experiment": args.experiment,
            "arm": args.arm,
            "seed_bundle": args.seed_bundle,
            "engineering_only": bool(args.engineering_only),
            "visible_devices_inside_worker": torch.cuda.device_count(),
            "sealed_rows_visible": False,
            "status": body["status"],
            "updates": body["updates"],
            "processed_tokens": body["processed_tokens"],
            "timing": body["timing"],
            "wall_seconds": time.monotonic() - started,
        }), flush=True)
        return EXIT_OK
    except RuntimeError as exc:
        message = str(exc)
        code = EXIT_GLOBAL if any(x in message for x in GLOBAL_MARKERS) else EXIT_ARM_LOCAL
        print(("GLOBAL_FAILURE " if code == EXIT_GLOBAL else "ARM_LOCAL_FAILURE ") + json.dumps({
            "schema": "anra.tie-role-worker-failure/v1",
            "class": "GLOBAL_INTEGRITY_FAILURE" if code == EXIT_GLOBAL else "ARM_LOCAL_ENGINEERING_FAILURE",
            "exception": type(exc).__name__,
            "message": message,
            "traceback": traceback.format_exc(),
        }), flush=True)
        return code
    except Exception as exc:
        print("GLOBAL_FAILURE " + json.dumps({
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }), flush=True)
        return EXIT_GLOBAL


if __name__ == "__main__":
    raise SystemExit(main())
