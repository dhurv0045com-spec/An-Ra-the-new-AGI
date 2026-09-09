"""CYR-GPU-009 progressive one-shot Colab runner.

Unlike 006-008, this runner does not reject a healthy GPU because an entire
ideal campaign cannot be guaranteed before training. It follows the useful
Arkenstone Discovery V7 pattern: fixed global wall budget, sequential matched
units, conservative per-unit launch gates, partial receipts after each unit,
and packaging in finally.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

from anra_v5 import cyr_gpu006_run as legacy
from anra_v5.cyr_gpu008_run import calibrate_candidates
from v5_experiments import cyr_gpu009 as core

BUNDLE_NAME = "CYMEK_GPU_RESEARCH_V9_RESULTS.zip"
production_tokenizer = legacy.production_tokenizer
write_json = legacy.write_json
read_json = legacy.read_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(torch: Any, device: Any) -> dict[str, Any]:
    return {
        "schema": "anra-cyr-gpu009-environment/v1",
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }


def _package(
    out: Path,
    *,
    campaign: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    bundle = out / BUNDLE_NAME
    payload = {
        "SESSION_MANIFEST.json": {
            "experiment": core.CYR9_ID,
            "status": campaign.get("status"),
            "wall_seconds": campaign.get("wall_seconds"),
        },
        "ENVIRONMENT.json": campaign.get("environment", {}),
        "PREREGISTRATION.json": dict(preregistration),
        "RESOLVED_PREREGISTRATION.json": campaign.get("resolved", {}),
        "CALIBRATION.json": campaign.get("calibrations", {}),
        "DATA_MANIFEST.json": campaign.get("data_manifest", {}),
        "SPLIT_MANIFEST.json": campaign.get("split_manifest", {}),
        "PARENT_UNITS.json": campaign.get("parent_runs", []),
        "DECISION.json": campaign.get("decision", {}),
        "SEALED.json": campaign.get("sealed", {}),
    }
    if failure is not None:
        payload["FAILURE.json"] = dict(failure)
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in payload.items():
            archive.writestr(name, json.dumps(body, indent=2, sort_keys=True, default=str))
    return {
        "path": str(bundle),
        "sha256": _sha256(bundle),
        "entries": sorted(payload),
    }


def run_campaign(
    *,
    repo: Path,
    out: Path,
    preregistration: Mapping[str, Any],
    resolved: Mapping[str, Any],
    calibrations: Mapping[str, Any],
    torch: Any = None,
    device: Any = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if torch is None:
        import torch as torch_module
        torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError("CYR-GPU-009 requires a CUDA Colab GPU")
    if device is None:
        device = torch.device("cuda")
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError("CYR-GPU-009 refuses non-CUDA scientific execution")
    resolved = core.validate_resolved(resolved)
    selected = calibrations.get(resolved["proxy"], {})
    if selected.get("status") != "PASS":
        raise ValueError("selected CYR-GPU-009 proxy lacks a passing calibration")

    repo = Path(repo)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    hard_deadline = started + float(resolved["wall_budget_minutes"]) * 60.0
    science_deadline = hard_deadline - float(resolved["packaging_reserve_minutes"]) * 60.0
    campaign: dict[str, Any] = {
        "schema": "anra-cyr-gpu009-campaign/v1",
        "experiment": core.CYR9_ID,
        "status": "RUNNING",
        "resolved": dict(resolved),
        "calibrations": dict(calibrations),
        "environment": _environment(torch, device),
        "arkenstone_audited_sha": core.ARKENSTONE_AUDITED_SHA,
        "execution_technique": core.ARKENSTONE_TECHNIQUE,
        "resume_semantics": (
            "completed acquisition/arm receipts are reused; incomplete units restart "
            "from their deterministic seed or immutable G90 checkpoint"
        ),
    }
    failure: dict[str, Any] | None = None
    try:
        tokenizer, identity = production_tokenizer(repo)
        expected_tok = preregistration.get("tokenizer", {}).get("artifact_sha256")
        if expected_tok and expected_tok != identity["artifact_sha256"]:
            raise ValueError("runtime tokenizer differs from CYR-GPU-009 preregistration")
        special = {
            "pad_id": identity["pad_id"],
            "bos_id": identity["bos_id"],
            "eos_id": identity["eos_id"],
        }

        splits = core.render_t2_worlds()
        manifest = core.build_data_manifest(splits)
        core.assert_manifest_sha(manifest)
        expected_manifest = preregistration.get("data", {}).get("data_manifest_sha256_full")
        if expected_manifest and expected_manifest != manifest["sha256"]:
            raise ValueError("runtime rendered-data manifest differs from preregistration")
        leak = core.commutation_audit(splits, tv_bound=0.20)
        if not leak["commutation_free"]:
            raise ValueError(f"data leak audit failed: {leak['findings']}")
        campaign["data_manifest"] = manifest
        campaign["split_manifest"] = {
            name: [row["world_id"] for row in rows] for name, rows in splits.items()
        }
        campaign["leak_audit"] = leak

        registry = core.proxy_registry(vocab_size=identity["vocabulary_size"])
        spec = registry[resolved["proxy"]]["spec"]
        campaign["proxy_registry"] = {
            name: {"parameters": entry["parameters"], "role": entry["role"]}
            for name, entry in registry.items()
        }

        parent_runs: list[dict[str, Any]] = []
        seeds = [int(seed) for seed in resolved["parent_seeds"]]
        for parent_index, seed in enumerate(seeds):
            now = time.monotonic()
            remaining_science = max(0.0, science_deadline - now)
            remaining_units = len(seeds) - parent_index
            launch_floor = float(resolved["minimum_parent_launch_minutes"]) * 60.0
            if remaining_science < launch_floor:
                record = {
                    "seed": seed,
                    "status": "BUDGET_BLOCKED",
                    "parent_status": "NOT_ATTEMPTED",
                    "reason": "remaining global science budget below frozen parent launch gate",
                    "remaining_science_seconds": remaining_science,
                    "arms": {},
                    "parent_equivalence": {"identical": False},
                    "future_tail": {"identical": False},
                }
                parent_runs.append(record)
                write_json(out / f"parent-{seed}" / "parent-run-v9.json", record)
                continue

            # Arkenstone-style progressive allocation: reserve an equal share of
            # remaining wall for every still-unstarted matched parent unit.
            unit_budget = remaining_science / max(remaining_units, 1)
            unit_deadline = min(science_deadline, now + unit_budget)
            acquisition_deadline = min(unit_deadline, now + unit_budget * 0.68)
            stream = core.build_future_stream(
                seed=seed,
                world_count=len(splits["train"]),
                prefix_rows=max(64, int(resolved["target_actual_tokens_acquisition"] // 128)),
                tail_rows=max(64, int(resolved["target_actual_tokens_continuation"] // 128)),
            )
            if progress:
                progress(
                    f"parent {seed}: unit={unit_budget/60:.1f}m, "
                    f"acquisition window={(acquisition_deadline-now)/60:.1f}m"
                )
            parent = legacy.acquire_parent(
                seed=seed,
                spec=spec,
                proxy_name=resolved["proxy"],
                tokenizer=tokenizer,
                torch=torch,
                device=device,
                special=special,
                train_rows=splits["train"],
                controller_rows=splits["dev_controller"],
                probe_rows=splits["train"][:16],
                target_actual_tokens=int(resolved["target_actual_tokens_acquisition"]),
                stream=stream,
                store_root=out,
                stage_deadline=acquisition_deadline,
                eval_interval_tokens=int(resolved["acquisition_eval_interval_tokens"]),
                progress=progress,
            )
            if parent.get("parent_status") != "G90_CONFIRMED":
                record = {
                    "seed": seed,
                    "status": "NO_QUALIFIED_PARENT",
                    "parent_status": parent.get("parent_status"),
                    "parent": parent,
                    "arms": {},
                    "parent_equivalence": {"identical": False},
                    "future_tail": {"identical": False},
                }
                parent_runs.append(record)
                write_json(out / f"parent-{seed}" / "parent-run-v9.json", record)
                continue

            equivalence = legacy.verify_parent_equivalence(
                parent=parent, spec=spec, torch=torch, device=device
            )
            arms: dict[str, Any] = {}
            order = list(core.CYR9_ARMS)
            if parent_index % 2:
                order.reverse()
            for arm_index, arm in enumerate(order):
                now_arm = time.monotonic()
                if now_arm >= unit_deadline:
                    arms[arm] = {"status": "TIMEBOX", "arm": arm, "actual_real_tokens": 0}
                    continue
                arms_left = len(order) - arm_index
                arm_deadline = min(
                    unit_deadline,
                    now_arm + (unit_deadline - now_arm) / max(arms_left, 1),
                )
                arms[arm] = legacy.continuation_arm(
                    arm=arm,
                    parent=parent,
                    spec=spec,
                    tokenizer=tokenizer,
                    torch=torch,
                    device=device,
                    special=special,
                    train_rows=splits["train"],
                    controller_rows=splits["dev_controller"],
                    measurement_rows=splits["dev_measurement"],
                    stream=stream,
                    target_actual_tokens=int(resolved["target_actual_tokens_continuation"]),
                    store_root=out,
                    stage_deadline=arm_deadline,
                    eval_interval_tokens=int(resolved["continuation_eval_interval_tokens"]),
                    progress=progress,
                )

            complete_receipts = [r for r in arms.values() if r.get("consumed_batch_shas")]
            compared = min((len(r["consumed_batch_shas"]) for r in complete_receipts), default=0)
            if compared and len(complete_receipts) == len(core.CYR9_ARMS):
                future_tail = core.assert_future_tail_equality({
                    arm: arms[arm]["consumed_batch_shas"][:compared]
                    for arm in core.CYR9_ARMS
                })
            else:
                future_tail = {"identical": False, "batches_compared": compared}
            record = {
                "seed": seed,
                "status": "MATCHED_UNIT_COMPLETE" if all(
                    arms.get(arm, {}).get("status") == "COMPLETE" for arm in core.CYR9_ARMS
                ) else "MATCHED_UNIT_PARTIAL",
                "parent_status": parent["parent_status"],
                "parent": parent,
                "arms": arms,
                "parent_equivalence": equivalence,
                "future_tail": future_tail,
                "unit_wall_seconds": unit_budget,
            }
            parent_runs.append(record)
            write_json(out / f"parent-{seed}" / "parent-run-v9.json", record)

        campaign["parent_runs"] = parent_runs
        decision = core.decide(parent_runs, resolved)
        campaign["decision"] = decision

        # Sealed measurement is post-decision and optional under the wall. It
        # never turns a partial matched pair into valid evidence.
        sealed: dict[str, Any] = {"status": "NOT_MEASURED_BUDGET", "arms": {}}
        if time.monotonic() < science_deadline - 60.0:
            sealed["status"] = "MEASURED_AFTER_DECISION"
            from v5_model.core import initialize
            from v5_training.optimizer import build_adamw_optimizer
            for parent_run in parent_runs:
                seed = int(parent_run.get("seed", -1))
                for arm, receipt in parent_run.get("arms", {}).items():
                    if receipt.get("status") != "COMPLETE" or not receipt.get("final_checkpoint"):
                        continue
                    model = initialize(spec, seed, torch_module=torch).to(device)
                    optimizer = build_adamw_optimizer(model, torch_module=torch)
                    legacy._load_checkpoint(
                        Path(receipt["final_checkpoint"]), model=model, optimizer=optimizer, torch=torch
                    )
                    sealed["arms"][f"{seed}/{arm}"] = legacy.generate_rates_batched(
                        model,
                        tokenizer,
                        splits["sealed_reserved"],
                        torch=torch,
                        device=device,
                        special=special,
                        batch_size=32,
                    )
                    del model, optimizer
                    if time.monotonic() >= science_deadline - 30.0:
                        break
        campaign["sealed"] = sealed
        campaign["status"] = "COMPLETE_OR_BUDGETED_PARTIAL"
    except Exception as exc:
        failure = {
            "schema": "anra-cyr-gpu009-failure/v1",
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "wall_seconds": time.monotonic() - started,
        }
        campaign["status"] = "FAILED"
        campaign["decision"] = campaign.get(
            "decision",
            {
                "verdict": "INCONCLUSIVE_RUNTIME_FAILURE",
                "production_promotion_authorized": False,
                "pre500m_authorized": False,
                "training_500m_authorized": False,
            },
        )
    finally:
        campaign["wall_seconds"] = time.monotonic() - started
        bundle = _package(
            out,
            campaign=campaign,
            preregistration=preregistration,
            failure=failure,
        )
        campaign["bundle"] = bundle
        write_json(out / "campaign_receipt.json", campaign)
    if failure is not None:
        raise RuntimeError(
            f"CYR-GPU-009 failed after packaging partial evidence: {failure['message']}"
        )
    return campaign
