"""Operator runner for CYR-GPU-013 / R1B replicated vocabulary response curve."""
from __future__ import annotations

import gc
import hashlib
import json
import time
import traceback
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from anra_v5 import cyr_gpu011_run as inherited
from anra_v5 import cyr_gpu012_r1_run as r1run
from anra_v5.cyr_gpu011_optimizer_compat import canonical_optimizer_compat
from v5_experiments import cyr_gpu011 as base
from v5_experiments import cyr_gpu012_r1 as r1core
from v5_experiments import cyr_gpu013_r1b as core

BUNDLE_NAME = "CYMEK_R1B_VOCAB_RESPONSE_CURVE_RESULTS.zip"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _environment(torch: Any, device: Any) -> dict[str, Any]:
    return {
        "schema": "anra-cyr-gpu013-r1b-environment/v1",
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }


@contextmanager
def _patched_r1_globals() -> Iterator[None]:
    """Reuse audited R1 arm machinery with the R1B fixed endpoint and seeds."""
    old = {
        "SCREEN_UPDATES": r1core.SCREEN_UPDATES,
        "SCREEN_ROW_PRESENTATIONS": r1core.SCREEN_ROW_PRESENTATIONS,
        "MODEL_SEEDS": r1core.MODEL_SEEDS,
        "ORDER_SEEDS": r1core.ORDER_SEEDS,
    }
    r1core.SCREEN_UPDATES = core.UPDATES
    r1core.SCREEN_ROW_PRESENTATIONS = core.ROW_PRESENTATIONS
    r1core.MODEL_SEEDS = core.MODEL_SEEDS
    r1core.ORDER_SEEDS = core.ORDER_SEEDS
    try:
        yield
    finally:
        for key, value in old.items():
            setattr(r1core, key, value)


def calibrate_all(*, data: Mapping[str, Any], torch: Any, device: Any) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    with canonical_optimizer_compat():
        for vocab in core.VOCABS:
            tok = r1core.tokenizer_for(vocab)
            spec = r1core.spec_for(vocab)
            receipts[f"V{vocab}"] = inherited._calibrate_one(
                regime=f"R1B_CHAR_V{vocab}",
                spec=spec,
                tokenizer=tok,
                special=tok.special,
                batch_rows=core.BATCH_ROWS,
                train_rows=list(data["train"]),
                eval_rows=list(data["dev_controller"]),
                torch=torch,
                device=device,
            )
    return receipts


def _package(out: Path, campaign: Mapping[str, Any], preregistration: Mapping[str, Any],
             failure: Mapping[str, Any] | None) -> dict[str, Any]:
    bundle = out / BUNDLE_NAME
    payload = {
        "SESSION_MANIFEST.json": {
            "experiment": core.EXPERIMENT,
            "status": campaign.get("status"),
            "wall_seconds": campaign.get("wall_seconds"),
        },
        "PREREGISTRATION.json": dict(preregistration),
        "ENVIRONMENT.json": campaign.get("environment", {}),
        "CALIBRATION.json": campaign.get("calibrations", {}),
        "RESOLVED.json": campaign.get("resolved", {}),
        "DATA_RECEIPT.json": campaign.get("data_receipt", {}),
        "MATCHED_INIT_RECEIPTS.json": campaign.get("matched_init_receipts", []),
        "ARMS.json": campaign.get("arms", {}),
        "DECISION.json": campaign.get("decision", {}),
    }
    if failure is not None:
        payload["FAILURE.json"] = dict(failure)
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in payload.items():
            zf.writestr(name, json.dumps(body, indent=2, sort_keys=True, default=str))
    return {"path": str(bundle), "sha256": _sha256(bundle), "entries": sorted(payload)}


def run_campaign(*, repo: Path, out: Path, preregistration: Mapping[str, Any],
                 resolved: Mapping[str, Any], calibrations: Mapping[str, Any],
                 torch: Any, device: Any,
                 progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available() or getattr(device, "type", None) != "cuda":
        raise RuntimeError("R1B requires CUDA")
    if int(resolved.get("curves_to_run", 0)) < 2:
        raise RuntimeError("R1B refuses execution without two complete prospective curves")
    if list(resolved.get("vocabs", [])) != list(core.VOCABS):
        raise RuntimeError("R1B vocabulary grid drift")
    if int(resolved.get("updates_per_arm", -1)) != core.UPDATES:
        raise RuntimeError("R1B endpoint drift")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    hard_deadline = started + core.WALL_MINUTES * 60.0
    science_deadline = hard_deadline - core.PACKAGING_RESERVE_MINUTES * 60.0
    campaign: dict[str, Any] = {
        "schema": "anra-cyr-gpu013-r1b-campaign/v1",
        "experiment": core.EXPERIMENT,
        "status": "RUNNING",
        "environment": _environment(torch, device),
        "resolved": dict(resolved),
        "calibrations": dict(calibrations),
        "arms": {},
        "matched_init_receipts": [],
        "claim_ceiling": "CONTROLLED_DEVELOPMENT_MECHANISM_ONLY",
    }
    failure: dict[str, Any] | None = None
    try:
        manifest_path = Path(repo) / "docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json"
        data = base.load_ark002b_manifest(manifest_path)
        expected_split = preregistration.get("data", {}).get("split_sha256")
        expected_blob = preregistration.get("data", {}).get("manifest_blob_sha")
        if data["source_split_sha256"] != expected_split or data["source_blob_sha"] != expected_blob:
            raise RuntimeError("R1B frozen data identity mismatch")
        battery = base.make_reasoning_battery(data)
        campaign["data_receipt"] = {
            "source_split_sha256": data["source_split_sha256"],
            "source_blob_sha": data["source_blob_sha"],
            "role_sha256": data["role_sha256"],
            "train": len(data["train"]),
            "dev_controller": len(data["dev_controller"]),
            "dev_measurement": len(data["dev_measurement"]),
            "sealed_reserved_not_primary": len(data["sealed_reserved"]),
        }

        curves_to_run = int(resolved["curves_to_run"])
        with _patched_r1_globals():
            for seed_index in range(curves_to_run):
                curve_order = core.ARM_ORDERS[seed_index]
                need = sum(float(resolved["estimated_arm_seconds"][f"V{v}"]) for v in curve_order)
                remaining = science_deadline - time.monotonic()
                if remaining < need:
                    if seed_index < 2:
                        raise RuntimeError(
                            f"mandatory R1B curve {seed_index + 1} no longer fits safely: "
                            f"need={need:.1f}s remaining={remaining:.1f}s"
                        )
                    if progress:
                        progress("R1B optional third curve skipped by wall protection")
                    break
                if progress:
                    progress(f"R1B starting complete curve seed_index={seed_index + 1} order={curve_order}")
                for vocab in curve_order:
                    arm = r1run._run_or_reuse_arm(
                        out=out,
                        seed_index=seed_index,
                        vocab=vocab,
                        data=data,
                        battery=battery,
                        torch=torch,
                        device=device,
                        deadline=science_deadline - core.MIN_FINALIZE_SECONDS,
                        build_receipts=campaign["matched_init_receipts"],
                        progress=progress,
                    )
                    campaign["arms"][core.arm_label(seed_index, vocab)] = arm
                    _write_json(out / "partial_campaign.json", campaign)

        complete_curves = 0
        for i in range(curves_to_run):
            if all(core.arm_label(i, v) in campaign["arms"] for v in core.VOCABS):
                complete_curves += 1
        campaign["complete_curves"] = complete_curves
        campaign["decision"] = core.decision(campaign["arms"], complete_curves)
        campaign["status"] = "COMPLETE" if complete_curves >= 2 else "INCOMPLETE"
        if complete_curves < 2:
            raise RuntimeError("R1B ended without two complete mandatory curves")
    except Exception as exc:
        failure = {
            "schema": "anra-cyr-gpu013-r1b-failure/v1",
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "wall_seconds": time.monotonic() - started,
        }
        campaign["status"] = "FAILED"
        campaign.setdefault("decision", {
            "verdict": "INCONCLUSIVE_RUNTIME_FAILURE",
            "pre500m_authorized": False,
            "training_500m_authorized": False,
        })
    finally:
        campaign["wall_seconds"] = time.monotonic() - started
        campaign["bundle"] = _package(out, campaign, preregistration, failure)
        _write_json(out / "campaign_receipt.json", campaign)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if failure is not None:
        raise RuntimeError(f"R1B failed after packaging: {failure['message']}")
    return campaign


__all__ = ["BUNDLE_NAME", "calibrate_all", "run_campaign"]
