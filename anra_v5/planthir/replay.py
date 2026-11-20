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
    from .backend import get_backend

    try:
        be = get_backend(backend_id)
    except (KeyError, ValueError):
        be = None
    if be is not None and callable(getattr(be, "create_kernel", None)):
        return be.create_kernel()  # type: ignore[attr-defined]
    if backend_id == "x1_rd2":
        from .adapters.x1_rd2_adapter import X1RD2EnvironmentKernel

        return X1RD2EnvironmentKernel()
    if backend_id == "x1_s20":
        from anra_v5.elendril.v8x.evaluation import X1S20EnvironmentKernel

        return X1S20EnvironmentKernel()
    if backend_id == "v8":
        from .adapters.v8_adapter import V8EnvironmentKernel

        return V8EnvironmentKernel()
    raise ValueError(
        f"MISSING_BACKEND_PLUGIN:{backend_id}: Backend {backend_id!r} is not registered. "
        "Set PLANTHIR_BACKEND_MODULES or register the backend plugin before running replay."
    )


def _canonical_digest(obj: Any) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _rollout_case(
    kernel: Any,
    policy: Any,
    world: Mapping[str, Any],
    *,
    world_idx: int,
    checkpoint_sha256: str,
    decoder: str,
    steps_limit: int,
) -> dict[str, Any]:
    wid = str(world.get("world_id", world_idx))
    if hasattr(policy, "reset_episode"):
        policy.reset_episode()
    st = kernel.initial_state(world)
    init_obs = kernel.observe(st)
    init_sha = sha256(init_obs.rendered_text.encode("utf-8")).hexdigest()
    actions: list[tuple[str, str | None]] = []
    history: list[dict[str, Any]] = []
    for step_idx in range(steps_limit):
        if (st.get("terminal", False) if isinstance(st, Mapping) else getattr(st, "terminal", False)):
            break
        obs = kernel.observe(st)
        obs_sha = sha256(obs.rendered_text.encode("utf-8")).hexdigest()
        act = policy.propose_with_telemetry(obs)
        actions.append((act.operation, act.argument))
        history.append(
            {
                "step_index": step_idx,
                "rendered_input_sha256": obs_sha,
                "operation": act.operation,
                "argument": act.argument,
            }
        )
        st = kernel.step(st, act.operation, act.argument)
    res = kernel.verify(world, actions)
    first_op, first_arg = actions[0] if actions else ("INVALID", None)
    return {
        "world_id": wid,
        "checkpoint_sha256": checkpoint_sha256,
        "decoder": decoder,
        "renderer": world.get("renderer_template", world.get("renderer", 0)),
        "rendered_input_sha256": init_sha,
        "operation": first_op,
        "argument": first_arg,
        "correct": bool(res.get("correct")),
        "actions": [[op, arg] for op, arg in actions],
        "history": history,
    }


def independent_replay_bundle(
    bundle_dir: Path | str,
    *,
    expected_source_commit: str | None = None,
    expected_backend: str | None = None,
    worlds: Sequence[Mapping[str, Any]] | None = None,
    raw_evidence_dir: Path | str | None = None,
    device: str = "cpu",
    max_result_zip_bytes: int = DEFAULT_RESULT_ZIP_MAX_BYTES,
    accuracy_tolerance: float = 0.0,
    output_path: Path | str | None = None,
    independent_environment: bool = False,
    reference_predictions: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    use_archived_reference: bool = True,
    require_archived_reference: bool = False,
    require_reference: bool = False,
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
    archived_ref: dict[str, Any] | None = None
    legacy_v8_compatibility = False
    insufficient_evidence = False
    supported_recovery_path: str | None = None

    with zipfile.ZipFile(result_zip, "r") as rz:
        if use_archived_reference and "REPLAY_REFERENCE.json" in rz.namelist():
            archived_ref = json.loads(rz.read("REPLAY_REFERENCE.json").decode("utf-8"))
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

    archived_worlds: list[Mapping[str, Any]] | None = None
    if archived_ref is not None:
        if archived_ref.get("schema") != "anra-planthir-replay-reference-v1":
            defects.append(f"INVALID_REPLAY_REFERENCE_SCHEMA:{archived_ref.get('schema')}")
        ref_eval_worlds = archived_ref.get("eval_worlds")
        if not isinstance(ref_eval_worlds, list) or not ref_eval_worlds:
            defects.append("EMPTY_REPLAY_REFERENCE_WORLDS")
        else:
            ordered_ids = [str(w.get("world_id", i)) for i, w in enumerate(ref_eval_worlds)]
            if archived_ref.get("ordered_world_ids") != ordered_ids or archived_ref.get("cohort_sha256") != _canonical_digest(ordered_ids):
                defects.append("REPLAY_COHORT_HASH_MISMATCH")
            if archived_ref.get("worlds_sha256") != _canonical_digest(ref_eval_worlds):
                defects.append("REPLAY_WORLDS_HASH_MISMATCH")
            archived_worlds = list(ref_eval_worlds)

    if raw_evidence_dir is not None:
        from .evidence import verify_raw_evidence_chunks

        verify_raw_evidence_chunks(raw_evidence_dir)

    with tempfile.TemporaryDirectory(prefix="planthir-indep-replay-", ignore_cleanup_errors=True) as iso_tmp:
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

        agreement_statuses: list[str] = []
        archived_lineages = (
            archived_ref.get("lineages", {})
            if isinstance(archived_ref, Mapping) and isinstance(archived_ref.get("lineages"), Mapping)
            else {}
        )
        for lineage_id, rec in sorted(latest_by_lineage.items()):
            archived_lin = archived_lineages.get(lineage_id) if isinstance(archived_lineages, Mapping) else None
            target_mode = (
                str(archived_lin.get("decoder", "raw"))
                if isinstance(archived_lin, Mapping) and archived_lin.get("decoder")
                else "raw"
            )
            loaded = CheckpointCatalog.load_checkpoint_for_causal_probing(
                rec,
                mode=target_mode,
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

                # Validate backend consistency against expected_backend and archived REPLAY_REFERENCE.json BEFORE running worlds
                if expected_backend is not None and str(expected_backend) != backend_id:
                    defects.append(f"BACKEND_MISMATCH:{lineage_id}:expected={expected_backend}:ckpt={backend_id}")
                if archived_ref is not None:
                    ref_bid = str(archived_ref.get("backend_id") or "")
                    if ref_bid and ref_bid != backend_id:
                        defects.append(f"BACKEND_MISMATCH:{lineage_id}:ref={ref_bid}:ckpt={backend_id}")
                    if isinstance(archived_lin, Mapping):
                        lin_bid = str(archived_lin.get("backend_id") or "")
                        if lin_bid and lin_bid != backend_id:
                            defects.append(f"BACKEND_MISMATCH:{lineage_id}:lineage_ref={lin_bid}:ckpt={backend_id}")

                kernel = _build_kernel_for_backend(backend_id)

                # Resolve evaluation worlds AFTER resolving backend_id (never silently feed V8 worlds to non-V8 backends)
                if worlds is not None:
                    eval_worlds = list(worlds)
                elif archived_worlds is not None:
                    eval_worlds = list(archived_worlds)
                elif backend_id == "v8" and not require_archived_reference:
                    from anra_v5.elendril.v8.data import D1_TRAIN_SEED, build_d1_training_dataset

                    eval_worlds = list(
                        w
                        for w in build_d1_training_dataset(seed=D1_TRAIN_SEED + 11, quota_per_family=16).worlds
                        if int(w.get("renderer_template", 0)) == 0
                    )[:4]
                    legacy_v8_compatibility = True
                    supported_recovery_path = (
                        "Legacy V8 bundle lacked REPLAY_REFERENCE.json; replayed with canonical V8 D1 compatibility probe. "
                        "Re-export the bundle with REPLAY_REFERENCE.json or supply explicit worlds/reference_predictions for exact agreement."
                    )
                else:
                    insufficient_evidence = True
                    supported_recovery_path = (
                        f"Bundle for backend {backend_id!r} lacks REPLAY_REFERENCE.json and no explicit eval worlds were supplied. "
                        "Re-export the bundle using Planthir with REPLAY_REFERENCE.json archiving, or call "
                        "independent_replay_bundle(..., worlds=..., reference_predictions=...) with the backend's recorded evaluation cohort."
                    )
                    defects.append(f"INSUFFICIENT_REPLAY_REFERENCE:{lineage_id}:{backend_id}")
                    lineage_reports.append(
                        {
                            "lineage_id": lineage_id,
                            "checkpoint_sha256": rec.checkpoint_sha256,
                            "global_update": int(rec.global_update),
                            "backend_id": backend_id,
                            "architecture": rec.architecture,
                            "parameter_count": sum(int(p.numel()) for p in model.parameters()),
                            "restore_verified": bool(restored.get("verified")),
                            "probe_worlds": 0,
                            "operations": [],
                            "hidden_shape": [],
                            "finite_hidden": True,
                            "replay_probe_accuracy": 0.0,
                            "probe_accuracy": 0.0,
                            "original_prediction_agreement": {
                                "status": "INSUFFICIENT_REPLAY_EVIDENCE",
                                "matched": False,
                                "join_keys": ["checkpoint_sha256", "rendered_input_sha256", "world_id", "decoder"],
                                "checked_cases": 0,
                                "mismatches": [
                                    {
                                        "backend_id": backend_id,
                                        "reason": "INSUFFICIENT_REPLAY_REFERENCE",
                                    }
                                ],
                            },
                        }
                    )
                    agreement_statuses.append("INSUFFICIENT_REPLAY_EVIDENCE")
                    continue

                if not eval_worlds:
                    defects.append(f"EMPTY_EVAL_WORLDS:{lineage_id}:{backend_id}")
                    agreement_statuses.append("MISMATCH")
                    continue

                obs_list = [kernel.observe(kernel.initial_state(w)) for w in eval_worlds]
                props = []
                for o in obs_list:
                    if hasattr(policy, "reset_episode"):
                        policy.reset_episode()
                    props.append(policy.propose_with_telemetry(o))
                reps = policy.extract_representations(obs_list)["decision_hidden"].detach().cpu()
                finite_hidden = bool(torch.isfinite(reps).all().item())
                if not finite_hidden:
                    defects.append(f"NONFINITE_REPLAY_HIDDEN:{lineage_id}")

                # Resolve reference rows before rollout so multi-step cap covers full recorded history
                ref_rows: Sequence[Mapping[str, Any]] | None = None
                matched_summary_acc: float | None = None
                if reference_predictions is not None:
                    ref_rows = reference_predictions.get(lineage_id)
                    if ref_rows is None:
                        for exp_doc in recorded_summaries.values():
                            if str(exp_doc.get("lineage_id", "")) == lineage_id:
                                ref_rows = reference_predictions.get(str(exp_doc.get("experiment_id", "")))
                if ref_rows is None and isinstance(archived_lin, Mapping):
                    ref_rows = archived_lin.get("reference_predictions")
                    if archived_lin.get("overall_accuracy") is not None:
                        matched_summary_acc = float(archived_lin["overall_accuracy"])
                    if not ref_rows:
                        defects.append(f"EMPTY_ARCHIVED_REFERENCE:{lineage_id}")
                elif ref_rows is None and archived_ref is not None:
                    defects.append(f"MISSING_ARCHIVED_REFERENCE:{lineage_id}")

                for exp_doc in recorded_summaries.values():
                    if (
                        str(exp_doc.get("checkpoint_sha256", "")) == rec.checkpoint_sha256
                        or str(exp_doc.get("lineage_id", "")) == lineage_id
                    ):
                        if matched_summary_acc is None and "overall_accuracy" in exp_doc and exp_doc["overall_accuracy"] is not None:
                            matched_summary_acc = float(exp_doc["overall_accuracy"])
                        if ref_rows is None and exp_doc.get("reference_predictions"):
                            ref_rows = exp_doc["reference_predictions"]
                        break

                max_ref_hist = 0
                if ref_rows:
                    for rr in ref_rows:
                        hist = rr.get("history")
                        if isinstance(hist, Sequence):
                            max_ref_hist = max(max_ref_hist, len(hist))
                steps_limit = max(4, int(getattr(kernel, "max_actions", 8) or 8), max_ref_hist)

                correct = 0
                replayed_cases: dict[str, dict[str, Any]] = {}
                world_by_id = {str(w.get("world_id", idx)): (idx, w) for idx, w in enumerate(eval_worlds)}
                extra_mode_policies: dict[str, Any] = {target_mode: policy}
                extra_mode_loaded: list[dict[str, Any]] = []
                for idx, w in enumerate(eval_worlds):
                    case_rec = _rollout_case(
                        kernel,
                        policy,
                        w,
                        world_idx=idx,
                        checkpoint_sha256=rec.checkpoint_sha256,
                        decoder=target_mode,
                        steps_limit=steps_limit,
                    )
                    if case_rec["correct"]:
                        correct += 1
                    replayed_cases[case_rec["world_id"]] = case_rec
                replay_acc = float(correct / max(len(eval_worlds), 1))

                if ref_rows is not None:
                    case_mismatches: list[dict[str, Any]] = []
                    for ref in ref_rows:
                        r_wid = str(ref.get("world_id", ""))
                        r_dec = str(ref.get("decoder", ref.get("mode", target_mode)))
                        r_ckpt = str(ref.get("checkpoint_sha256", rec.checkpoint_sha256))
                        if r_ckpt != rec.checkpoint_sha256:
                            case_mismatches.append(
                                {"world_id": r_wid, "backend_id": backend_id, "decoder": r_dec, "reason": "CHECKPOINT_SHA_MISMATCH"}
                            )
                            continue
                        if r_dec != target_mode:
                            if r_wid not in world_by_id:
                                case_mismatches.append(
                                    {"world_id": r_wid, "backend_id": backend_id, "decoder": r_dec, "reason": "MISSING_REFERENCE_CASE"}
                                )
                                continue
                            if r_dec not in extra_mode_policies:
                                mod_loaded = CheckpointCatalog.load_checkpoint_for_causal_probing(
                                    rec, mode=r_dec, device=device
                                )
                                extra_mode_loaded.append(mod_loaded)
                                extra_mode_policies[r_dec] = mod_loaded["policy"]
                            mpol = extra_mode_policies[r_dec]
                            w_idx_m, w_m = world_by_id[r_wid]
                            got_c = _rollout_case(
                                kernel,
                                mpol,
                                w_m,
                                world_idx=w_idx_m,
                                checkpoint_sha256=rec.checkpoint_sha256,
                                decoder=r_dec,
                                steps_limit=steps_limit,
                            )
                        else:
                            if r_wid not in replayed_cases:
                                reason = "UNREPLAYED_CASE" if r_wid in world_by_id else "MISSING_REFERENCE_CASE"
                                case_mismatches.append(
                                    {"world_id": r_wid, "backend_id": backend_id, "decoder": r_dec, "reason": reason}
                                )
                                continue
                            got_c = replayed_cases[r_wid]

                        if "renderer" in ref and ref["renderer"] is not None and ref["renderer"] != got_c.get("renderer"):
                            case_mismatches.append(
                                {
                                    "world_id": r_wid,
                                    "backend_id": backend_id,
                                    "decoder": r_dec,
                                    "reason": "RENDERER_MISMATCH",
                                    "expected": ref["renderer"],
                                    "observed": got_c.get("renderer"),
                                }
                            )
                        elif ref.get("rendered_input_sha256") and ref["rendered_input_sha256"] != got_c["rendered_input_sha256"]:
                            case_mismatches.append(
                                {
                                    "world_id": r_wid,
                                    "backend_id": backend_id,
                                    "decoder": r_dec,
                                    "reason": "RENDERED_INPUT_HASH_MISMATCH",
                                    "expected": ref["rendered_input_sha256"],
                                    "observed": got_c["rendered_input_sha256"],
                                }
                            )
                        elif (str(ref.get("operation")), ref.get("argument")) != (got_c["operation"], got_c["argument"]):
                            case_mismatches.append(
                                {
                                    "world_id": r_wid,
                                    "backend_id": backend_id,
                                    "decoder": r_dec,
                                    "step_index": 0,
                                    "reason": "PREDICTION_MISMATCH",
                                    "expected": [ref.get("operation"), ref.get("argument")],
                                    "observed": [got_c["operation"], got_c["argument"]],
                                }
                            )
                        elif isinstance(ref.get("history"), Sequence):
                            ref_hist = list(ref["history"])
                            got_hist = list(got_c.get("history", []))
                            if len(ref_hist) != len(got_hist):
                                case_mismatches.append(
                                    {
                                        "world_id": r_wid,
                                        "backend_id": backend_id,
                                        "decoder": r_dec,
                                        "reason": "HISTORY_LENGTH_MISMATCH",
                                        "expected_steps": len(ref_hist),
                                        "observed_steps": len(got_hist),
                                    }
                                )
                            else:
                                for s_idx, (rs, gs) in enumerate(zip(ref_hist, got_hist)):
                                    if rs.get("rendered_input_sha256") and rs["rendered_input_sha256"] != gs.get("rendered_input_sha256"):
                                        case_mismatches.append(
                                            {
                                                "world_id": r_wid,
                                                "backend_id": backend_id,
                                                "decoder": r_dec,
                                                "step_index": s_idx,
                                                "reason": "STEP_INPUT_HASH_MISMATCH",
                                                "expected": rs["rendered_input_sha256"],
                                                "observed": gs.get("rendered_input_sha256"),
                                            }
                                        )
                                        break
                                    if (str(rs.get("operation")), rs.get("argument")) != (gs.get("operation"), gs.get("argument")):
                                        case_mismatches.append(
                                            {
                                                "world_id": r_wid,
                                                "backend_id": backend_id,
                                                "decoder": r_dec,
                                                "step_index": s_idx,
                                                "reason": "STEP_PREDICTION_MISMATCH",
                                                "expected": [rs.get("operation"), rs.get("argument")],
                                                "observed": [gs.get("operation"), gs.get("argument")],
                                            }
                                        )
                                        break
                    if matched_summary_acc is not None and abs(replay_acc - matched_summary_acc) > float(accuracy_tolerance):
                        case_mismatches.append(
                            {
                                "backend_id": backend_id,
                                "reason": "AGGREGATE_ACCURACY_OUT_OF_TOLERANCE",
                                "expected": matched_summary_acc,
                                "observed": replay_acc,
                                "tolerance": float(accuracy_tolerance),
                            }
                        )
                    orig_agree = {
                        "status": "MATCHED" if not case_mismatches else "MISMATCH",
                        "matched": len(case_mismatches) == 0,
                        "join_keys": ["checkpoint_sha256", "rendered_input_sha256", "world_id", "decoder"],
                        "checked_cases": len(ref_rows),
                        "mismatches": case_mismatches,
                    }
                    if case_mismatches:
                        defects.append(f"ORIGINAL_PREDICTION_MISMATCH:{lineage_id}")
                        agreement_statuses.append("MISMATCH")
                    else:
                        agreement_statuses.append("MATCHED")
                elif require_reference:
                    insufficient_evidence = True
                    supported_recovery_path = (
                        f"Replay for backend {backend_id!r} required reference predictions but none were available. "
                        "Supply reference_predictions or enable use_archived_reference=True with REPLAY_REFERENCE.json."
                    )
                    defects.append(f"INSUFFICIENT_REPLAY_REFERENCE:{lineage_id}:{backend_id}")
                    orig_agree = {
                        "status": "INSUFFICIENT_REPLAY_EVIDENCE",
                        "matched": False,
                        "join_keys": ["checkpoint_sha256", "rendered_input_sha256", "world_id", "decoder"],
                        "checked_cases": 0,
                        "mismatches": [{"backend_id": backend_id, "reason": "INSUFFICIENT_REPLAY_REFERENCE"}],
                    }
                    agreement_statuses.append("INSUFFICIENT_REPLAY_EVIDENCE")
                else:
                    orig_agree = {
                        "status": "PROBE_ONLY_NO_REFERENCE",
                        "matched": None,
                        "join_keys": ["checkpoint_sha256", "rendered_input_sha256", "world_id", "decoder"],
                        "checked_cases": 0,
                        "mismatches": [],
                    }
                    agreement_statuses.append("PROBE_ONLY_NO_REFERENCE")
                del extra_mode_loaded

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

    if any(s == "INSUFFICIENT_REPLAY_EVIDENCE" for s in agreement_statuses):
        overall_agreement = "INSUFFICIENT_REPLAY_EVIDENCE"
    elif any(s == "MISMATCH" for s in agreement_statuses):
        overall_agreement = "MISMATCH"
    elif agreement_statuses and all(s == "MATCHED" for s in agreement_statuses):
        overall_agreement = "MATCHED"
    elif any(s == "MATCHED" for s in agreement_statuses):
        overall_agreement = "PARTIAL_MATCH"
    else:
        overall_agreement = "PROBE_ONLY_NO_REFERENCE"

    mean_probe_acc = (
        float(sum(r["probe_accuracy"] for r in lineage_reports) / len(lineage_reports))
        if lineage_reports
        else 0.0
    )
    all_agree_mismatches = [
        {"lineage_id": r["lineage_id"], **m}
        for r in lineage_reports
        for m in r["original_prediction_agreement"]["mismatches"]
    ]
    verified = len(defects) == 0
    if verified:
        status = "REPLAY_VERIFIED"
    elif insufficient_evidence:
        status = "INSUFFICIENT_REPLAY_EVIDENCE"
    else:
        status = "REPLAY_FAILED"
    receipt = {
        "schema": "anra-planthir-replay-receipt-v1",
        "verified": verified,
        "status": status,
        "restore_verified": bool(not any(d == "RESTORE_FULL_CHECKPOINT_ZIP_UNVERIFIED" for d in defects)),
        "archived_reference_used": bool(archived_ref is not None),
        "legacy_v8_compatibility": bool(legacy_v8_compatibility),
        "supported_recovery_path": supported_recovery_path,
        "probe_accuracy": mean_probe_acc,
        "prediction_agreement_status": overall_agreement,
        "original_prediction_agreement": {
            "status": overall_agreement,
            "matched": (True if overall_agreement == "MATCHED" else (False if overall_agreement in ("MISMATCH", "INSUFFICIENT_REPLAY_EVIDENCE") else None)),
            "join_keys": ["checkpoint_sha256", "rendered_input_sha256", "world_id", "decoder"],
            "checked_cases": sum(r["original_prediction_agreement"]["checked_cases"] for r in lineage_reports),
            "mismatches": all_agree_mismatches,
        },
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
        if insufficient_evidence:
            raise ValueError(f"INSUFFICIENT_REPLAY_EVIDENCE: {defects}; recovery: {supported_recovery_path}")
        raise ValueError(f"independent bundle replay failed: {defects}")
    return receipt


__all__ = [
    "build_custody_receipt",
    "independent_replay_bundle",
]
