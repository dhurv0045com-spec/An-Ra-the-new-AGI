"""Custody receipts and isolated independent bundle replay verification from checkpoints and archives."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import torch

from .checkpoint_intelligence import CheckpointCatalog
from .evidence import (
    BUNDLE_POINTER_NAME,
    DEFAULT_RESULT_ZIP_MAX_BYTES,
    FULL_CHECKPOINT_ZIP_NAME,
    RESULT_ZIP_NAME,
    verify_result_bundle,
)
from .execution import _atomic_write, _sync_dir


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_target_bundle_dir(bundle_dir: Path | str) -> tuple[Path, dict[str, Any] | None]:
    root = Path(bundle_dir)
    ptr_path = root / BUNDLE_POINTER_NAME
    if ptr_path.is_file():
        from .evidence import resolve_published_generation

        ptr = resolve_published_generation(root, verify=False)
        return Path(ptr["generation_dir"]), ptr
    return root, None


def build_custody_receipt(
    bundle_dir: Path | str,
    *,
    expected_source_commit: str | None = None,
    max_result_zip_bytes: int = DEFAULT_RESULT_ZIP_MAX_BYTES,
    raw_evidence_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Verify full chain-of-custody across bundle pointer, archive SHA-256 sidecars, and capsule links."""

    target_dir, pointer = _resolve_target_bundle_dir(bundle_dir)
    verified = verify_result_bundle(
        target_dir,
        expected_source_commit=expected_source_commit,
        max_result_zip_bytes=max_result_zip_bytes,
    )

    defects: list[str] = []
    result_zip = target_dir / RESULT_ZIP_NAME
    full_zip = target_dir / FULL_CHECKPOINT_ZIP_NAME
    r_sha = _sha256_file(result_zip)
    f_sha = _sha256_file(full_zip)

    for zname, expected_sha in ((RESULT_ZIP_NAME, r_sha), (FULL_CHECKPOINT_ZIP_NAME, f_sha)):
        sidecar = target_dir / f"{zname}.sha256"
        if not sidecar.is_file():
            defects.append(f"MISSING_SHA256_SIDECAR:{zname}")
        else:
            recorded = sidecar.read_text(encoding="utf-8").strip().split()[0]
            if recorded != expected_sha:
                defects.append(f"SIDECAR_SHA256_MISMATCH:{zname}")

    if pointer is not None:
        if pointer.get("result_manifest_sha256") != verified["result_manifest_sha256"]:
            defects.append("POINTER_RESULT_MANIFEST_MISMATCH")
        if pointer.get("full_manifest_sha256") != verified["full_manifest_sha256"]:
            defects.append("POINTER_FULL_MANIFEST_MISMATCH")

    capsule_links: dict[str, Any] = {}
    with zipfile.ZipFile(result_zip, "r") as zf:
        for arc in sorted(zf.namelist()):
            if arc.startswith("capsules/") and arc.endswith("/CAPSULE_MANIFEST.json"):
                cap = json.loads(zf.read(arc).decode("utf-8"))
                link = cap.get("raw_evidence_link")
                if link is not None:
                    capsule_links[arc] = link
                    if raw_evidence_dir is not None:
                        from .evidence import verify_raw_evidence_chunks

                        raw_res = verify_raw_evidence_chunks(raw_evidence_dir)
                        if link.get("raw_manifest_sha256") != raw_res["manifest_sha256"]:
                            defects.append(f"CAPSULE_RAW_MANIFEST_MISMATCH:{arc}")

    custody_ok = len(defects) == 0
    receipt = {
        "schema": "anra-planthir-custody-receipt-v1",
        "verified": custody_ok,
        "status": "CUSTODY_VERIFIED" if custody_ok else "CUSTODY_DEFECT",
        "bundle_root": str(Path(bundle_dir)),
        "resolved_generation_dir": str(target_dir),
        "generation_id": pointer.get("current_generation") if pointer else None,
        "source_commit": verified["source_commit"],
        "bundle_status": verified["status"],
        "result_zip_sha256": r_sha,
        "result_zip_bytes": int(result_zip.stat().st_size),
        "result_manifest_sha256": verified["result_manifest_sha256"],
        "checkpoint_full_zip_sha256": f_sha,
        "checkpoint_full_zip_bytes": int(full_zip.stat().st_size),
        "full_manifest_sha256": verified["full_manifest_sha256"],
        "capsule_raw_links": capsule_links,
        "defects": defects,
    }
    if output_path is not None:
        op = Path(output_path)
        _atomic_write(op, (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        _sync_dir(op.parent)
    if not custody_ok:
        raise ValueError(f"bundle custody verification failed: {defects}")
    return receipt


def _build_kernel_for_backend(backend_id: str):
    try:
        from .backend import get_backend

        be = get_backend(backend_id)
        if callable(getattr(be, "create_kernel", None)):
            return be.create_kernel()  # type: ignore[attr-defined]
    except Exception:
        pass
    if backend_id == "x1_rd2":
        from .adapters.x1_rd2_adapter import X1RD2EnvironmentKernel

        return X1RD2EnvironmentKernel()
    if backend_id == "x1_s20":
        from anra_v5.elendril.v8x.evaluation import X1S20EnvironmentKernel

        return X1S20EnvironmentKernel()
    from .adapters.v8_adapter import V8EnvironmentKernel

    return V8EnvironmentKernel()


def independent_replay_bundle(
    bundle_dir: Path | str,
    *,
    expected_source_commit: str | None = None,
    worlds: Sequence[Mapping[str, Any]] | None = None,
    raw_evidence_dir: Path | str | None = None,
    device: str = "cpu",
    max_result_zip_bytes: int = DEFAULT_RESULT_ZIP_MAX_BYTES,
    accuracy_tolerance: float = 0.0,
    output_path: Path | str | None = None,
    independent_environment: bool = False,
    reference_predictions: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Restore an exported bundle into an isolated temp workspace and independently replay checkpoints & checks."""

    custody = build_custody_receipt(
        bundle_dir,
        expected_source_commit=expected_source_commit,
        max_result_zip_bytes=max_result_zip_bytes,
        raw_evidence_dir=raw_evidence_dir,
    )
    target_dir = Path(custody["resolved_generation_dir"])
    full_zip = target_dir / FULL_CHECKPOINT_ZIP_NAME
    result_zip = target_dir / RESULT_ZIP_NAME

    defects: list[str] = []
    lineage_reports: list[dict[str, Any]] = []

    with zipfile.ZipFile(result_zip, "r") as rz:
        recorded_summaries: dict[str, dict[str, Any]] = {}
        for name in sorted(rz.namelist()):
            if name.startswith("experiments/") and name.endswith("/SCIENTIFIC_SUMMARY.json"):
                doc = json.loads(rz.read(name).decode("utf-8"))
                recorded_summaries[str(doc.get("experiment_id", name))] = doc

        capsule_checks: dict[str, Any] = {}
        for name in sorted(rz.namelist()):
            if name.startswith("capsules/") and name.endswith("/representations.npz"):
                cap_prefix = name[: -len("representations.npz")]
                raw_npz = rz.read(name)
                with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
                    tf.write(raw_npz)
                    tpath = Path(tf.name)
                try:
                    with np.load(tpath, allow_pickle=False) as npz:
                        arrs = {k: np.asarray(npz[k]) for k in npz.files}
                    finite_ok = all(bool(np.isfinite(a).all()) for a in arrs.values())
                    capsule_checks[cap_prefix.rstrip("/")] = {
                        "keys": sorted(arrs),
                        "shapes": {k: list(v.shape) for k, v in arrs.items()},
                        "all_finite": finite_ok,
                    }
                    if not finite_ok:
                        defects.append(f"NONFINITE_CAPSULE_ARRAY:{cap_prefix}")
                finally:
                    tpath.unlink(missing_ok=True)

    if raw_evidence_dir is not None:
        from .evidence import verify_raw_evidence_chunks

        verify_raw_evidence_chunks(raw_evidence_dir)

    with tempfile.TemporaryDirectory(prefix="planthir-indep-replay-") as iso_tmp:
        restored = CheckpointCatalog.restore_full_checkpoint_zip(
            full_zip,
            Path(iso_tmp) / "restored_store",
            expected_source_commit=expected_source_commit,
        )
        if not restored.get("verified"):
            defects.append("RESTORE_FULL_CHECKPOINT_ZIP_UNVERIFIED")

        # Keep only final record per lineage
        latest_by_lineage: dict[str, Any] = {}
        for rec in restored["catalog"].records:
            prev = latest_by_lineage.get(rec.lineage_id)
            if prev is None or int(rec.global_update) >= int(prev.global_update):
                latest_by_lineage[rec.lineage_id] = rec

        if not latest_by_lineage:
            defects.append("NO_CHECKPOINT_LINEAGES_RESTORED")

        eval_worlds = list(worlds) if worlds is not None else []
        if not eval_worlds:
            from anra_v5.elendril.v8.data import D1_TRAIN_SEED, build_d1_training_dataset

            eval_worlds = list(
                w
                for w in build_d1_training_dataset(seed=D1_TRAIN_SEED + 11, quota_per_family=16).worlds
                if int(w.get("renderer_template", 0)) == 0
            )[:4]

        agreement_statuses: list[str] = []
        for lineage_id, rec in sorted(latest_by_lineage.items()):
            loaded = CheckpointCatalog.load_checkpoint_for_causal_probing(
                rec,
                mode="raw",
                device=device,
            )
            try:
                model = loaded["model"]
                policy = loaded["policy"]
                backend_id = str(loaded.get("backend_id") or "")
                if not backend_id:
                    prog_fam = str(loaded.get("program_family", ""))
                    if hasattr(model, "x1_spec"):
                        backend_id = "x1_s20" if getattr(getattr(model, "x1_spec", None), "base_spec", None) is not None else "x1_rd2"
                    elif prog_fam.startswith("V8") or not prog_fam:
                        backend_id = "v8"
                    else:
                        backend_id = prog_fam
                kernel = _build_kernel_for_backend(backend_id)
                obs_list = [kernel.observe(kernel.initial_state(w)) for w in eval_worlds]
                props = [policy.propose_with_telemetry(o) for o in obs_list]
                reps = policy.extract_representations(obs_list)["decision_hidden"].detach().cpu()
                finite_hidden = bool(torch.isfinite(reps).all().item())
                if not finite_hidden:
                    defects.append(f"NONFINITE_REPLAY_HIDDEN:{lineage_id}")

                correct = 0
                replayed_cases: dict[str, dict[str, Any]] = {}
                world_by_id = {str(w.get("world_id", idx)): w for idx, w in enumerate(eval_worlds)}
                for idx, w in enumerate(eval_worlds):
                    wid = str(w.get("world_id", idx))
                    st = kernel.initial_state(w)
                    init_obs = kernel.observe(st)
                    init_sha = sha256(init_obs.rendered_text.encode("utf-8")).hexdigest()
                    actions: list[tuple[str, str | None]] = []
                    for _ in range(4):
                        if getattr(st, "terminal", False):
                            break
                        obs = kernel.observe(st)
                        act = policy.propose_with_telemetry(obs)
                        actions.append((act.operation, act.argument))
                        st = kernel.step(st, act.operation, act.argument)
                    res = kernel.verify(w, actions)
                    if res.get("correct"):
                        correct += 1
                    first_op, first_arg = actions[0] if actions else ("INVALID", None)
                    replayed_cases[wid] = {
                        "world_id": wid,
                        "rendered_input_sha256": init_sha,
                        "operation": first_op,
                        "argument": first_arg,
                        "correct": bool(res.get("correct")),
                    }
                replay_acc = float(correct / max(len(eval_worlds), 1))

                ref_rows: Sequence[Mapping[str, Any]] | None = None
                if reference_predictions is not None:
                    ref_rows = reference_predictions.get(lineage_id)
                    if ref_rows is None:
                        for exp_doc in recorded_summaries.values():
                            if str(exp_doc.get("lineage_id", "")) == lineage_id:
                                ref_rows = reference_predictions.get(str(exp_doc.get("experiment_id", "")))
                if ref_rows is None:
                    for exp_doc in recorded_summaries.values():
                        if exp_doc.get("reference_predictions") and (
                            str(exp_doc.get("checkpoint_sha256", "")) == rec.checkpoint_sha256
                            or str(exp_doc.get("lineage_id", "")) == lineage_id
                        ):
                            ref_rows = exp_doc["reference_predictions"]
                            break

                if ref_rows is not None:
                    case_mismatches: list[dict[str, Any]] = []
                    for ref in ref_rows:
                        r_wid = str(ref.get("world_id", ""))
                        if r_wid not in replayed_cases:
                            if r_wid in world_by_id:
                                case_mismatches.append({"world_id": r_wid, "reason": "UNREPLAYED_CASE"})
                            else:
                                case_mismatches.append({"world_id": r_wid, "reason": "MISSING_REFERENCE_CASE"})
                            continue
                        got_c = replayed_cases[r_wid]
                        if ref.get("rendered_input_sha256") and ref["rendered_input_sha256"] != got_c["rendered_input_sha256"]:
                            case_mismatches.append({"world_id": r_wid, "reason": "RENDERED_INPUT_HASH_MISMATCH"})
                        elif (str(ref.get("operation")), ref.get("argument")) != (got_c["operation"], got_c["argument"]):
                            case_mismatches.append({
                                "world_id": r_wid,
                                "reason": "PREDICTION_MISMATCH",
                                "expected": [ref.get("operation"), ref.get("argument")],
                                "observed": [got_c["operation"], got_c["argument"]],
                            })
                    orig_agree = {
                        "status": "MATCHED" if not case_mismatches else "MISMATCH",
                        "matched": len(case_mismatches) == 0,
                        "checked_cases": len(ref_rows),
                        "mismatches": case_mismatches,
                    }
                    if case_mismatches:
                        defects.append(f"ORIGINAL_PREDICTION_MISMATCH:{lineage_id}")
                        agreement_statuses.append("MISMATCH")
                    else:
                        agreement_statuses.append("MATCHED")
                else:
                    orig_agree = {
                        "status": "PROBE_ONLY_NO_REFERENCE",
                        "matched": None,
                        "checked_cases": 0,
                        "mismatches": [],
                    }
                    agreement_statuses.append("PROBE_ONLY_NO_REFERENCE")

                actual_params = sum(int(p.numel()) for p in model.parameters())
                lineage_reports.append(
                    {
                        "lineage_id": lineage_id,
                        "checkpoint_sha256": rec.checkpoint_sha256,
                        "global_update": int(rec.global_update),
                        "backend_id": backend_id,
                        "architecture": rec.architecture,
                        "parameter_count": actual_params,
                        "restore_verified": bool(restored.get("verified")),
                        "probe_worlds": len(obs_list),
                        "operations": [p.operation for p in props],
                        "hidden_shape": list(reps.shape),
                        "finite_hidden": finite_hidden,
                        "replay_probe_accuracy": replay_acc,
                        "probe_accuracy": replay_acc,
                        "original_prediction_agreement": orig_agree,
                    }
                )
            finally:
                del loaded
                if torch.device(device).type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if any(s == "MISMATCH" for s in agreement_statuses):
        overall_agreement = "MISMATCH"
    elif agreement_statuses and all(s == "MATCHED" for s in agreement_statuses):
        overall_agreement = "MATCHED"
    else:
        overall_agreement = "PROBE_ONLY_NO_REFERENCE"

    verified = len(defects) == 0
    receipt = {
        "schema": "anra-planthir-replay-receipt-v1",
        "verified": verified,
        "status": "REPLAY_VERIFIED" if verified else "REPLAY_FAILED",
        "restore_verified": bool(not any(d == "RESTORE_FULL_CHECKPOINT_ZIP_UNVERIFIED" for d in defects)),
        "prediction_agreement_status": overall_agreement,
        "independent_environment": bool(independent_environment),
        "accuracy_tolerance": float(accuracy_tolerance),
        "bundle_root": str(Path(bundle_dir)),
        "resolved_generation_dir": str(target_dir),
        "custody": custody,
        "recorded_experiment_count": len(recorded_summaries),
        "capsule_checks": capsule_checks,
        "restored_lineages": lineage_reports,
        "defects": defects,
    }
    if output_path is not None:
        op = Path(output_path)
        _atomic_write(op, (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        _sync_dir(op.parent)
    if not verified:
        raise ValueError(f"independent bundle replay failed: {defects}")
    return receipt


__all__ = [
    "build_custody_receipt",
    "independent_replay_bundle",
]
