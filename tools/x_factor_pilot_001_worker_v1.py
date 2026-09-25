from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anra_v5 import x_factor_pilot_train_v1 as train
from v5_experiments import x_factor_pilot_protocol_v1 as protocol

EXIT_OK = 0
EXIT_ARM_LOCAL = 3
EXIT_GLOBAL = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("official", "control", "canary"), required=True)
    parser.add_argument("--arm", choices=protocol.ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-updates", type=int, default=None)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--deadline-epoch", type=float, default=None)
    return parser


def _load_surface(mode: str, path: Path) -> dict:
    if mode == "official":
        from v5_experiments.formation_mux_surface_v5 import load_public_surface
        return load_public_surface(path)
    return protocol.load_positive_control_surface(path)


def _receipt_path(out: Path, mode: str, arm: str, seed: int) -> Path:
    return out / mode / arm / protocol.seed_label(seed) / "WORKER_RECEIPT.json"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.monotonic()
    try:
        import torch
        if args.mode == "canary":
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends, "cuda"):
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
        if args.device != "cuda" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("worker requires exactly one visible CUDA device")
        if "T4" not in torch.cuda.get_device_name(0).upper():
            raise RuntimeError("worker requires an NVIDIA T4 device")
        if not 14 * 1024 ** 3 <= int(torch.cuda.get_device_properties(0).total_memory) <= 17 * 1024 ** 3:
            raise RuntimeError("worker requires a T4-sized device")
        if args.mode == "official" and args.seed not in protocol.MODEL_SEEDS:
            raise RuntimeError("official seed is not registered")
        if args.mode == "control" and args.seed != protocol.CONTROL_SEED:
            raise RuntimeError("control seed is not registered")
        if args.mode == "canary" and args.seed != protocol.CALIBRATION_SEED:
            raise RuntimeError("canary seed is not registered")
        if args.mode == "official" and (args.target_updates is not None or args.stop_after is not None):
            raise RuntimeError("official exposure cannot be overridden")
        if args.mode == "control" and (args.target_updates is not None or args.stop_after is not None or args.deadline_epoch is not None):
            raise RuntimeError("control exposure cannot be overridden")
        if args.mode == "canary" and args.deadline_epoch is not None:
            raise RuntimeError("canary exposure cannot use a session deadline")
        surface = _load_surface(args.mode, args.surface)
        result = train.train_arm(
            mode=args.mode,
            arm=args.arm,
            seed=args.seed,
            surface=surface,
            out_dir=args.out,
            torch=torch,
            device=torch.device(args.device),
            target_updates=args.target_updates,
            stop_after=args.stop_after,
            deadline_epoch=args.deadline_epoch,
            progress=lambda message: print(message, flush=True),
        )
        receipt = {
            "schema": "anra.x-factor-pilot-worker/v1",
            "mode": args.mode,
            "arm": args.arm,
            "seed": args.seed,
            "protocol_sha256": protocol.protocol_sha256(),
            "device": torch.cuda.get_device_name(0),
            "visible_device_count": int(torch.cuda.device_count()),
            "sealed_rows_passed": False,
            "status": result["status"],
            "updates": result["updates"],
            "processed_tokens": result["processed_tokens"],
            "deadline_epoch": args.deadline_epoch,
            "wall_seconds": time.monotonic() - started,
        }
        train.atomic_json(_receipt_path(args.out, args.mode, args.arm, args.seed), receipt)
        return EXIT_OK
    except RuntimeError as exc:
        message = str(exc)
        global_markers = ("SEALED_ROWS_PRESENT", "protocol", "not registered", "official exposure", "control exposure", "CUDA", "T4", "checkpoint identity", "initial model")
        code = EXIT_GLOBAL if any(marker in message for marker in global_markers) else EXIT_ARM_LOCAL
        payload = {
            "schema": "anra.x-factor-pilot-worker-failure/v1",
            "class": "GLOBAL_INTEGRITY_FAILURE" if code == EXIT_GLOBAL else "ARM_LOCAL_ENGINEERING_FAILURE",
            "mode": args.mode,
            "arm": args.arm,
            "seed": args.seed,
            "exception": type(exc).__name__,
            "message": message,
            "traceback": traceback.format_exc(),
        }
        try:
            train.atomic_json(_receipt_path(args.out, args.mode, args.arm, args.seed), payload)
        except Exception:
            pass
        print(json.dumps(payload), flush=True)
        return code
    except Exception as exc:
        payload = {
            "schema": "anra.x-factor-pilot-worker-failure/v1",
            "class": "GLOBAL_INTEGRITY_FAILURE",
            "mode": args.mode,
            "arm": args.arm,
            "seed": args.seed,
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            train.atomic_json(_receipt_path(args.out, args.mode, args.arm, args.seed), payload)
        except Exception:
            pass
        print(json.dumps(payload), flush=True)
        return EXIT_GLOBAL


if __name__ == "__main__":
    raise SystemExit(main())
