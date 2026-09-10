"""Operator runner for CYR-GPU-012 / R1 representation causal screen."""
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
from anra_v5.cyr_gpu011_optimizer_compat import canonical_optimizer_compat
from v5_experiments import cyr_gpu011 as base
from v5_experiments import cyr_gpu012_r1 as core

BUNDLE_NAME = "CYMEK_R1_REPRESENTATION_CAUSAL_RESULTS.zip"
write_json = inherited.write_json
read_json = inherited.read_json


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_sha(named: list[tuple[str, Any]], *, active_embedding_rows: int | None = None) -> str:
    h = hashlib.sha256()
    for name, tensor in named:
        t = tensor.detach().float().cpu().contiguous()
        if name == "embedding.weight" and active_embedding_rows is not None:
            t = t[:active_embedding_rows]
        h.update(name.encode("utf-8") + b"\0")
        h.update(str(tuple(t.shape)).encode("ascii") + b"\0")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _environment(torch: Any, device: Any) -> dict[str, Any]:
    return {
        "schema": "anra-cyr-gpu012-r1-environment/v1",
        "torch": torch.__version__, "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }


def calibrate_all(*, data: Mapping[str, Any], torch: Any, device: Any) -> dict[str, Any]:
    """Calibrate fixed batch64 only; batch is a causal invariant, not a resolver knob."""
    receipts: dict[str, Any] = {}
    with canonical_optimizer_compat():
        for vocab in (19, core.OPTIONAL_VOCAB, 24_576):
            tok = core.tokenizer_for(vocab)
            spec = core.spec_for(vocab)
            rec = inherited._calibrate_one(
                regime=f"R1_CHAR_V{vocab}", spec=spec, tokenizer=tok, special=tok.special,
                batch_rows=core.BATCH_ROWS, train_rows=list(data["train"]),
                eval_rows=list(data["dev_controller"]), torch=torch, device=device,
            )
            receipts[f"V{vocab}"] = rec
    return receipts


@contextmanager
def _screen_scope(*, seed: int, build_receipts: list[dict[str, Any]]) -> Iterator[None]:
    """Force exactly 8k updates and matched non-vocabulary initialization."""
    original_build = inherited._build_model
    old_max = base.CYR11_MAX_UPDATES
    old_g90 = base.CYR11_G90

    def matched_build(spec: Any, model_seed: int, *, torch: Any, device: Any):
        if int(model_seed) != int(seed):
            raise ValueError("R1 model-seed drift inside matched initialization scope")
        target = original_build(spec, model_seed, torch=torch, device=device)
        vocab = int(spec.vocabulary_size)
        if vocab == 19:
            shared = _tensor_sha(list(target.named_parameters()), active_embedding_rows=19)
            build_receipts.append({
                "vocabulary_size": vocab, "model_seed": int(model_seed),
                "parameters": sum(p.numel() for p in target.parameters()),
                "shared_core_plus_active_embedding_sha256": shared,
                "extra_embedding_rows_sha256": None,
                "matched_init": "REFERENCE",
            })
            return target

        reference = original_build(core.spec_for(19), model_seed, torch=torch, device=device)
        ref_named = dict(reference.named_parameters())
        tgt_named = dict(target.named_parameters())
        if set(ref_named) != set(tgt_named):
            raise ValueError("R1 parameter-name drift across vocabulary intervention")
        with torch.no_grad():
            for name, param in tgt_named.items():
                ref = ref_named[name]
                if name == "embedding.weight":
                    if param.shape[1:] != ref.shape[1:] or param.shape[0] < 19:
                        raise ValueError("R1 embedding shape mismatch")
                    param[:19].copy_(ref)
                else:
                    if tuple(param.shape) != tuple(ref.shape):
                        raise ValueError(f"R1 shared parameter shape drift: {name}")
                    param.copy_(ref)
        ref_shared = _tensor_sha(list(reference.named_parameters()), active_embedding_rows=19)
        tgt_shared = _tensor_sha(list(target.named_parameters()), active_embedding_rows=19)
        if ref_shared != tgt_shared:
            raise AssertionError("R1 shared initialization hash mismatch")
        extra = hashlib.sha256(target.embedding.weight[19:].detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
        build_receipts.append({
            "vocabulary_size": vocab, "model_seed": int(model_seed),
            "parameters": sum(p.numel() for p in target.parameters()),
            "shared_core_plus_active_embedding_sha256": tgt_shared,
            "reference_shared_sha256": ref_shared,
            "extra_embedding_rows_sha256": extra,
            "matched_init": "ALL_SHARED_TENSORS_AND_ACTIVE_EMBEDDING_ROWS_COPIED_FROM_V19_REFERENCE",
        })
        del reference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return target

    inherited._build_model = matched_build
    base.CYR11_MAX_UPDATES = core.SCREEN_UPDATES
    # The inherited runner normally stops early at qualified G90. R1 requires a
    # fixed 512k-row endpoint for every arm, so make G90 unreachable during the
    # screen while preserving all raw controller/measurement trajectories.
    base.CYR11_G90 = 1.01
    try:
        yield
    finally:
        inherited._build_model = original_build
        base.CYR11_MAX_UPDATES = old_max
        base.CYR11_G90 = old_g90


def _arm_wrapper_path(out: Path, label: str) -> Path:
    return out / label / "R1_ARM.json"


def _valid_reuse(body: Mapping[str, Any], *, label: str, vocab: int, model_seed: int, order_seed: int) -> bool:
    acq = body.get("acquisition", {})
    return bool(
        body.get("schema") == "anra-cyr-gpu012-r1-arm/v1"
        and body.get("label") == label and int(body.get("vocabulary_size", -1)) == int(vocab)
        and int(body.get("model_seed", -1)) == int(model_seed)
        and int(body.get("order_seed", -1)) == int(order_seed)
        and int(body.get("target_updates", -1)) == core.SCREEN_UPDATES
        and int(acq.get("updates", -1)) == core.SCREEN_UPDATES
        and int(acq.get("row_presentations", -1)) == core.SCREEN_ROW_PRESENTATIONS
    )


def _run_or_reuse_arm(*, out: Path, seed_index: int, vocab: int, data: Mapping[str, Any],
                      battery: Mapping[str, Any], torch: Any, device: Any, deadline: float,
                      build_receipts: list[dict[str, Any]], progress: Callable[[str], None] | None) -> dict[str, Any]:
    label = core.arm_label(seed_index, vocab)
    model_seed = core.MODEL_SEEDS[seed_index]
    order_seed = core.ORDER_SEEDS[seed_index]
    wrapper = _arm_wrapper_path(out, label)
    if wrapper.exists():
        body = read_json(wrapper)
        if _valid_reuse(body, label=label, vocab=vocab, model_seed=model_seed, order_seed=order_seed):
            init_receipt = body.get("matched_initialization")
            if isinstance(init_receipt, Mapping):
                build_receipts.append(dict(init_receipt))
            if progress:
                progress(f"R1 reuse complete arm: {label}")
            return dict(body)
        raise RuntimeError(f"existing R1 arm is incompatible and will not be overwritten: {wrapper}")

    tok = core.tokenizer_for(vocab)
    spec = core.spec_for(vocab)
    arm_out = out / label
    arm_out.mkdir(parents=True, exist_ok=True)
    local_builds: list[dict[str, Any]] = []
    with canonical_optimizer_compat(), _screen_scope(seed=model_seed, build_receipts=local_builds):
        acq = inherited.run_acquisition(
            label=label, model_seed=model_seed, order_seed=order_seed,
            spec=spec, tokenizer=tok, special=tok.special,
            batch_rows=core.BATCH_ROWS, data=data, battery=battery,
            torch=torch, device=device, deadline=deadline, out=arm_out,
            include_verbal=False, progress=progress,
        )
    if int(acq.get("updates", -1)) != core.SCREEN_UPDATES:
        raise RuntimeError(f"{label} did not reach fixed R1 endpoint before wall: {acq.get('updates')}")
    if len(local_builds) != 1:
        raise AssertionError(f"{label}: expected exactly one matched-init build receipt")
    build_receipts.extend(local_builds)
    body = {
        "schema": "anra-cyr-gpu012-r1-arm/v1", "label": label,
        "vocabulary_size": int(vocab), "active_token_ids": list(range(19)),
        "model_seed": int(model_seed), "order_seed": int(order_seed),
        "batch_rows": core.BATCH_ROWS, "target_updates": core.SCREEN_UPDATES,
        "target_row_presentations": core.SCREEN_ROW_PRESENTATIONS,
        "forced_fixed_endpoint": True,
        "matched_initialization": local_builds[0],
        "acquisition": acq,
    }
    write_json(wrapper, body)
    return body


def _package(out: Path, campaign: Mapping[str, Any], preregistration: Mapping[str, Any],
             failure: Mapping[str, Any] | None) -> dict[str, Any]:
    bundle = out / BUNDLE_NAME
    payload = {
        "SESSION_MANIFEST.json": {"experiment": core.CYR12_ID, "status": campaign.get("status"),
                                  "wall_seconds": campaign.get("wall_seconds")},
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
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for name, body in payload.items():
            z.writestr(name, json.dumps(body, indent=2, sort_keys=True, default=str))
    return {"path": str(bundle), "sha256": _sha256(bundle), "entries": sorted(payload)}


def run_campaign(*, repo: Path, out: Path, preregistration: Mapping[str, Any],
                 resolved: Mapping[str, Any], calibrations: Mapping[str, Any],
                 torch: Any = None, device: Any = None,
                 progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    if torch is None:
        import torch as torch_module
        torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError("CYR-GPU-012 R1 requires CUDA")
    if device is None:
        device = torch.device("cuda")
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError("CYR-GPU-012 R1 refuses non-CUDA scientific execution")
    if int(resolved.get("batch_rows", -1)) != core.BATCH_ROWS:
        raise ValueError("R1 fixed batch invariant violated")
    if int(resolved.get("screen_row_presentations", -1)) != core.SCREEN_ROW_PRESENTATIONS:
        raise ValueError("R1 endpoint drift")

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    hard_deadline = started + core.WALL_MINUTES * 60.0
    science_deadline = hard_deadline - core.PACKAGING_RESERVE_MINUTES * 60.0
    campaign: dict[str, Any] = {
        "schema": "anra-cyr-gpu012-r1-campaign/v1", "experiment": core.CYR12_ID,
        "status": "RUNNING", "environment": _environment(torch, device),
        "resolved": dict(resolved), "calibrations": dict(calibrations),
        "arms": {}, "matched_init_receipts": [],
        "claim_ceiling": "CONTROLLED_DEVELOPMENT_MECHANISM_ONLY",
    }
    failure = None
    try:
        manifest_path = Path(repo) / "docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json"
        data = base.load_ark002b_manifest(manifest_path)
        expected_split = preregistration.get("data", {}).get("split_sha256")
        expected_blob = preregistration.get("data", {}).get("manifest_blob_sha")
        if expected_split != data["source_split_sha256"] or expected_blob != data["source_blob_sha"]:
            raise ValueError("R1 frozen ARK-002B data identity mismatch")
        battery = base.make_reasoning_battery(data)
        campaign["data_receipt"] = {
            "source_split_sha256": data["source_split_sha256"], "source_blob_sha": data["source_blob_sha"],
            "role_sha256": data["role_sha256"], "train": len(data["train"]),
            "dev_controller": len(data["dev_controller"]), "dev_measurement": len(data["dev_measurement"]),
            "sealed_reserved_not_primary": len(data["sealed_reserved"]),
            "note": "The CYR11 holdout has already been consumed; R1 is an explicitly developmental causal follow-up.",
        }

        seeds_to_run = int(resolved["seeds_to_run"])
        for seed_index in range(seeds_to_run):
            for vocab in core.PRIMARY_VOCABS:
                key = f"V{vocab}"
                need = float(resolved["estimated_arm_seconds"][key])
                remaining = science_deadline - time.monotonic()
                if remaining < need:
                    if seed_index == 0:
                        raise RuntimeError(f"primary R1 pair cannot complete within wall before {core.arm_label(seed_index, vocab)}")
                    campaign["arms"][core.arm_label(seed_index, vocab)] = {
                        "status": "NOT_RUN_WALL_PROTECTION", "estimated_seconds_needed": need,
                        "remaining_seconds": max(0.0, remaining),
                    }
                    break
                arm = _run_or_reuse_arm(
                    out=out, seed_index=seed_index, vocab=vocab, data=data, battery=battery,
                    torch=torch, device=device, deadline=science_deadline - core.MIN_FINALIZE_SECONDS,
                    build_receipts=campaign["matched_init_receipts"], progress=progress,
                )
                campaign["arms"][core.arm_label(seed_index, vocab)] = arm
            if core.arm_label(seed_index, 24_576) not in campaign["arms"] or \
                    campaign["arms"][core.arm_label(seed_index, 24_576)].get("schema") != "anra-cyr-gpu012-r1-arm/v1":
                break

        # Optional dose point never displaces a second primary seed. It launches
        # only if the resolved plan selected one seed or actual runtime leaves a
        # safely projected window after all primary pairs.
        opt_key = f"V{core.OPTIONAL_VOCAB}"
        opt_need = float(resolved.get("estimated_arm_seconds", {}).get(opt_key, 1e30))
        remaining = science_deadline - time.monotonic()
        primary_complete_seeds = sum(
            1 for i in range(seeds_to_run)
            if campaign["arms"].get(core.arm_label(i, 19), {}).get("schema") == "anra-cyr-gpu012-r1-arm/v1"
            and campaign["arms"].get(core.arm_label(i, 24_576), {}).get("schema") == "anra-cyr-gpu012-r1-arm/v1"
        )
        if primary_complete_seeds >= 1 and remaining >= opt_need:
            arm = _run_or_reuse_arm(
                out=out, seed_index=0, vocab=core.OPTIONAL_VOCAB, data=data, battery=battery,
                torch=torch, device=device, deadline=science_deadline - core.MIN_FINALIZE_SECONDS,
                build_receipts=campaign["matched_init_receipts"], progress=progress,
            )
            campaign["arms"][core.arm_label(0, core.OPTIONAL_VOCAB)] = arm
        else:
            campaign["arms"].setdefault(core.arm_label(0, core.OPTIONAL_VOCAB), {
                "status": "NOT_RUN_OPTIONAL_WALL_PRIORITY", "estimated_seconds_needed": opt_need,
                "remaining_seconds": max(0.0, remaining),
            })

        acquisitions = {k: v["acquisition"] for k, v in campaign["arms"].items()
                        if isinstance(v, Mapping) and "acquisition" in v}
        campaign["decision"] = core.decision(acquisitions, seeds_to_run)
        campaign["status"] = "COMPLETE"
    except Exception as exc:
        failure = {
            "schema": "anra-cyr-gpu012-r1-failure/v1", "exception": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "wall_seconds": time.monotonic() - started,
        }
        campaign["status"] = "FAILED"
        campaign.setdefault("decision", {
            "verdict": "INCONCLUSIVE_RUNTIME_FAILURE", "pre500m_authorized": False,
            "training_500m_authorized": False,
        })
    finally:
        campaign["wall_seconds"] = time.monotonic() - started
        campaign["bundle"] = _package(out, campaign, preregistration, failure)
        write_json(out / "campaign_receipt.json", campaign)
    if failure is not None:
        raise RuntimeError(f"CYR-GPU-012 R1 failed after packaging: {failure['message']}")
    return campaign


__all__ = ["BUNDLE_NAME", "calibrate_all", "run_campaign"]
