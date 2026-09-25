from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import x_factor_pilot_001_kaggle_operator_v1 as base
from v5_experiments import x_factor_pilot_protocol_v1 as protocol


def single_gpu_receipt(torch: Any) -> dict[str, Any]:
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": torch.cuda.get_device_name(index), "vram_bytes": int(properties.total_memory)})
    passed = len(devices) == 1 and "T4" in devices[0]["name"].upper() and 14 * 1024 ** 3 <= devices[0]["vram_bytes"] <= 17 * 1024 ** 3
    return {"schema": "anra.x-factor-pilot-colab-hardware/v1", "torch": torch.__version__, "cuda": torch.version.cuda, "devices": devices, "required": "one visible NVIDIA T4", "passed": passed}


def require_single_t4(torch: Any) -> dict[str, Any]:
    receipt = single_gpu_receipt(torch)
    if not receipt["passed"]:
        raise RuntimeError(f"COLAB_T4_REQUIRED: select one T4 GPU in Runtime Settings; observed {receipt['devices']}")
    return receipt


def one_gpu_jobs() -> list[dict[str, Any]]:
    return [{**job, "gpu": 0} for job in base._official_jobs()]


def import_resume(resume_from: Path | None, out: Path) -> str | None:
    if resume_from is None:
        return None
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("COLAB_RESUME_TARGET_NOT_EMPTY")
    if not resume_from.exists() or not (resume_from / "CAMPAIGN_STATE.json").exists():
        raise RuntimeError("COLAB_RESUME_SOURCE_INVALID")
    shutil.copytree(resume_from, out, dirs_exist_ok=True)
    return str(resume_from)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path("/content/X_FACTOR_PILOT_001"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--skip-sealed", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    out = args.out.resolve()
    started = time.monotonic()
    try:
        import torch
        if not repo.exists():
            raise RuntimeError("COLAB_REPOSITORY_MISSING")
        out.mkdir(parents=True, exist_ok=True)
        resumed = import_resume(args.resume_from, out)
        if (out / "COLAB_GLOBAL_FAILURE.json").exists():
            raise RuntimeError("UNRESOLVED_COLAB_GLOBAL_FAILURE")
        frozen = base.verify_frozen_files(repo)
        hardware = require_single_t4(torch)
        base.atomic_json(out / "COLAB_ENVIRONMENT.json", {"execution_mode": "single_gpu_colab", "hardware": hardware, "resumed_from": resumed, "head": base._git(repo, "rev-parse", "HEAD")})
        public_path, control_path, surface_receipt = base.build_surfaces(repo, out)
        qualification = base.qualify(repo, out, control_path)
        base.storage_preflight(out, repo)
        state_path = out / "CAMPAIGN_STATE.json"
        gate = base.positive_control(out, repo, control_path, state_path, started)
        if gate.get("status") != "PASS":
            base.package(out, repo)
            print(f"X-FACTOR-PILOT-001 COLAB STATUS: {gate.get('status')}", flush=True)
            return 0
        public_sha = json.loads(public_path.read_text(encoding="utf-8"))["sha256"]
        state = base.run_jobs(repo=repo, out=out, surface=public_path, surface_sha=public_sha, mode="official", jobs=one_gpu_jobs(), state_path=state_path, started=started)
        if state.get("status") != "ARMS_COMPLETE":
            base.package(out, repo)
            print(f"X-FACTOR-PILOT-001 COLAB STATUS: {state.get('status')}", flush=True)
            return 0
        base.development_aggregate(out)
        if not args.skip_sealed:
            base.finalize_sealed(out, public_path, repo, torch)
        final_state = {"schema": "anra.x-factor-pilot-campaign-state/v1", "campaign": protocol.CAMPAIGN, "protocol_sha256": protocol.protocol_sha256(), "status": "COMPLETE", "execution_mode": "single_gpu_colab", "positive_control": gate["status"], "official_arms": len(protocol.ARMS) * len(protocol.MODEL_SEEDS), "sealed_consumed": not args.skip_sealed, "head": base._git(repo, "rev-parse", "HEAD"), "frozen_files": frozen, "hardware": hardware, "surface_receipt": surface_receipt, "qualification": qualification["status"], "resumed_from": resumed}
        base.atomic_json(state_path, final_state)
        bundle = base.package(out, repo)
        print("X-FACTOR-PILOT-001 COLAB STATUS: COMPLETE", flush=True)
        print("RESULT BUNDLE:", json.dumps(bundle), flush=True)
        return 0
    except Exception as exc:
        failure = {"schema": "anra.x-factor-pilot-colab-global-failure/v1", "protocol_sha256": protocol.protocol_sha256(), "exception": type(exc).__name__, "message": str(exc), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            base.atomic_json(out / "COLAB_GLOBAL_FAILURE.json", failure)
            base.package(out, repo)
        except Exception:
            pass
        print("X-FACTOR-PILOT-001 COLAB GLOBAL FAIL-CLOSED:", repr(exc), file=sys.stderr, flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
