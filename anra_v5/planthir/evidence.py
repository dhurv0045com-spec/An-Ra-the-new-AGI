"""Three-Tier Evidence Architecture: Tier-1 Summary, Tier-2 Diagnostic Capsule, Tier-3 Binding."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np
import torch

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.training import _dump, _load
from v5_training.checkpoint import CheckpointStore, MILESTONES_FILENAME

from .causal import _eval_ridge_classifier, _fit_ridge_classifier
from .contracts import ExperimentRegistration
from .interpretation import ScientificInterpretationReport
from .observatory import compute_effective_rank

TIER1_MAX_BYTES = 100 * 1024
TIER2_MAX_BYTES = 2 * 1024 * 1024
RESULT_ZIP_NAME = "result.zip"
FULL_CHECKPOINT_ZIP_NAME = "checkpoint_full.zip"
DEFAULT_RESULT_ZIP_MAX_BYTES = 20 * 1024 * 1024
ALLOWED_BUNDLE_STATUSES = ("COMPLETE", "PARTIAL", "NOT_RUN", "NO_GO", "INVALID_RESULTS")


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def _write_json(path: Path, payload: Mapping[str, Any]) -> bytes:
    clean = _sanitize_json(dict(payload))
    text = json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n"
    raw = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def write_tier1_summary(
    output_dir: Path,
    *,
    registration: ExperimentRegistration,
    instrument_validation: Mapping[str, Any],
    process_funnel: Mapping[str, Any] | None,
    causal_diagnostics: Mapping[str, Any] | None,
    interpretation: ScientificInterpretationReport,
    recovery_binding: Mapping[str, Any] | None = None,
    max_bytes: int = TIER1_MAX_BYTES,
) -> dict[str, Path]:
    """Write lightweight Tier-1 scientific summary (< 100 KB) safe for direct Git commit."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Compact funnel without per-episode trace dump (which belongs in Tier 2)
    compact_funnel = None
    if process_funnel is not None:
        compact_funnel = {k: v for k, v in process_funnel.items() if k != "traces"}

    summary_payload = {
        "schema": "anra-planthir-tier1-summary-v1",
        "experiment_id": registration.experiment_id,
        "registration_sha256": registration.sha256(),
        "registration": registration.canonical_dict(),
        "instrument_validation": {
            "passed": instrument_validation.get("passed"),
            "status": instrument_validation.get("status"),
            "kernel_id": instrument_validation.get("kernel_id"),
            "checks": instrument_validation.get("checks"),
            "defect_count": len(instrument_validation.get("defects", ())),
        },
        "process_funnel": compact_funnel,
        "causal_diagnostics": dict(causal_diagnostics) if causal_diagnostics else None,
        "interpretation": interpretation.to_dict(),
        "recovery_binding": dict(recovery_binding) if recovery_binding else None,
    }
    json_path = out / "SCIENTIFIC_SUMMARY.json"
    raw_json = _write_json(json_path, summary_payload)
    if len(raw_json) > max_bytes:
        raise ValueError(f"Tier-1 SCIENTIFIC_SUMMARY.json ({len(raw_json)} B) exceeds limit {max_bytes} B")

    interp = interpretation
    lines = [
        f"# {registration.experiment_id} — Tier-1 Scientific Summary",
        "",
        f"- **Program:** `{registration.program}` | **Architecture:** `{registration.architecture_family}` ({registration.parameter_count:,} params)",
        f"- **Source Commit:** `{registration.source_commit}` | **Registration SHA:** `{registration.sha256()[:16]}`",
        f"- **Instrument Validation:** `{instrument_validation.get('status')}`",
        f"- **Interpretation Status:** `{interp.status}`",
        "",
    ]
    if compact_funnel and "funnel" in compact_funnel:
        f = compact_funnel["funnel"]
        lines.extend([
            "## 4-Stage Process Funnel",
            "",
            f"- **Overall Accuracy:** `{compact_funnel.get('overall_accuracy'):.3f}`",
            f"- **Stage 1 (Info Seeking):** `{f.get('stage1_info_seeking')}`",
            f"- **Stage 2 (Tool Discipline):** `{f.get('stage2_tool_discipline')}`",
            f"- **Stage 3 (1-Hop Grounding):** `{f.get('stage3_single_hop_grounding')}`",
            f"- **Stage 4 (Multi-Hop Composition):** `{f.get('stage4_multi_hop_composition')}`",
            f"- **Stage 4 (Epistemic Abstention):** `{f.get('stage4_epistemic_abstention')}`",
            "",
        ])

    for title, items in (
        ("Confirmed Causes", interp.confirmed_causes),
        ("Suspected Mechanisms", interp.suspected_mechanisms),
        ("Falsified Hypotheses", interp.falsified_hypotheses),
        ("Unresolved Questions", interp.unresolved_questions),
    ):
        lines.append(f"## {title}")
        lines.append("")
        if not items:
            lines.append("- None recorded under strict epistemic rules.")
        else:
            for item in items:
                lines.append(f"- **`{item.claim_id}`**: {item.statement}")
        lines.append("")

    if interp.recommended_next_experiments:
        lines.extend(["## Recommended Next Experiments", ""])
        for rec in interp.recommended_next_experiments:
            lines.append(
                f"{rec.rank}. **{rec.title}** (`{rec.cost_tier}`, EIG/GPUh=`{rec.expected_info_gain_per_gpu_hour:.1f}`) — {rec.rationale}"
            )
        lines.append("")

    md_path = out / "SCIENTIFIC_SUMMARY.md"
    md_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    md_path.write_bytes(md_bytes)
    return {"json": json_path, "markdown": md_path}


def write_tier2_diagnostic_capsule(
    capsule_dir: Path,
    *,
    experiment_id: str,
    probe_arrays: Mapping[str, torch.Tensor | np.ndarray],
    diagnostics: Mapping[str, Any],
    max_bytes: int = TIER2_MAX_BYTES,
) -> dict[str, Any]:
    """Write compact Tier-2 Diagnostic Capsule (.json + float16 .npz) with SHA-256 manifest."""

    cdir = Path(capsule_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    np_arrays: dict[str, np.ndarray] = {}
    for key, val in probe_arrays.items():
        arr = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float16)
        elif np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int64)
        np_arrays[key] = arr

    npz_path = cdir / "representations.npz"
    np.savez_compressed(npz_path, **np_arrays)

    diag_path = cdir / "diagnostics.json"
    _write_json(diag_path, diagnostics)

    members = {}
    total_bytes = 0
    for fname in ("representations.npz", "diagnostics.json"):
        raw = (cdir / fname).read_bytes()
        total_bytes += len(raw)
        members[fname] = {"byte_size": len(raw), "sha256": sha256(raw).hexdigest()}

    if total_bytes > max_bytes:
        raise ValueError(f"Tier-2 Diagnostic Capsule size {total_bytes} B exceeds limit {max_bytes} B")

    manifest = {
        "schema": "anra-planthir-tier2-capsule-v1",
        "experiment_id": experiment_id,
        "total_bytes": total_bytes,
        "array_shapes": {k: list(v.shape) for k, v in np_arrays.items()},
        "array_dtypes": {k: str(v.dtype) for k, v in np_arrays.items()},
        "members": members,
    }
    manifest_path = cdir / "CAPSULE_MANIFEST.json"
    _write_json(manifest_path, manifest)
    return {
        "capsule_dir": str(cdir),
        "manifest_path": str(manifest_path),
        "total_bytes": total_bytes,
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
    }


def verify_tier2_capsule(capsule_dir: Path) -> dict[str, Any]:
    """Fail-closed SHA-256 and size verification of a Tier-2 Diagnostic Capsule."""

    cdir = Path(capsule_dir)
    manifest_path = cdir / "CAPSULE_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing CAPSULE_MANIFEST.json in {cdir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "anra-planthir-tier2-capsule-v1":
        raise ValueError("unsupported Tier-2 capsule schema")
    members = manifest.get("members", {})
    if set(members) != {"representations.npz", "diagnostics.json"}:
        raise ValueError("invalid Tier-2 capsule member inventory")
    for fname, meta in members.items():
        fpath = cdir / fname
        if not fpath.is_file():
            raise ValueError(f"missing Tier-2 capsule member: {fname}")
        raw = fpath.read_bytes()
        if len(raw) != meta["byte_size"] or sha256(raw).hexdigest() != meta["sha256"]:
            raise ValueError(f"Tier-2 capsule corruption detected in {fname}")
    return manifest


def probe_from_capsule(
    capsule_dir: Path,
    *,
    train_fraction: float = 0.6,
    l2_reg: float = 1.0,
) -> dict[str, Any]:
    """Run offline representation rank and linear probes purely from a Tier-2 capsule (no Tier-3 weights)."""

    manifest = verify_tier2_capsule(capsule_dir)
    cdir = Path(capsule_dir)
    with np.load(cdir / "representations.npz") as archive:
        if "decision_hidden" not in archive:
            raise ValueError("capsule representations.npz missing 'decision_hidden'")
        hidden = torch.from_numpy(archive["decision_hidden"].astype(np.float32))
        results: dict[str, Any] = {
            "schema": "anra-planthir-capsule-probe-v1",
            "experiment_id": manifest["experiment_id"],
            "samples": int(hidden.shape[0]),
            "hidden_dim": int(hidden.shape[1]),
            "effective_rank": compute_effective_rank(hidden),
            "probes": {},
        }
        split = max(2, min(hidden.shape[0] - 2, int(hidden.shape[0] * train_fraction)))
        x_tr, x_ev = hidden[:split], hidden[split:]
        for key in ("y_op", "y_hop1", "y_final"):
            if key not in archive:
                continue
            y = torch.from_numpy(archive[key].astype(np.int64))
            num_classes = int(y.max().item()) + 1
            w = _fit_ridge_classifier(x_tr, y[:split], num_classes, l2_reg=l2_reg)
            acc = _eval_ridge_classifier(w, x_ev, y[split:])
            results["probes"][key] = acc
    return results


def bind_tier3_recovery_archive(
    checkpoint_root: Path,
    lineage: str,
) -> dict[str, Any]:
    """Bind Tier-1/2 artifacts to an existing Tier-3 CheckpointStore without copying weights."""

    store = CheckpointStore(Path(checkpoint_root), lineage)
    latest = store.latest_sha256()
    if latest is None:
        return {
            "tier": "TIER3_RECOVERY_ARCHIVE",
            "status": "NO_CHECKPOINT_COMMITTED",
            "checkpoint_root": str(checkpoint_root),
            "lineage": lineage,
            "latest_checkpoint_sha256": None,
        }
    manifest_path = store.objects / latest / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    total_bytes = sum(int(c["byte_size"]) for c in manifest.get("components", []))
    return {
        "tier": "TIER3_RECOVERY_ARCHIVE",
        "status": "BOUND_VERIFIED",
        "checkpoint_root": str(checkpoint_root),
        "lineage": lineage,
        "latest_checkpoint_sha256": latest,
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "total_checkpoint_bytes": total_bytes,
    }


def package_three_tier_evidence(
    experiment_dir: Path,
    *,
    registration: ExperimentRegistration,
    instrument_validation: Mapping[str, Any],
    process_funnel: Mapping[str, Any] | None,
    causal_diagnostics: Mapping[str, Any] | None,
    interpretation: ScientificInterpretationReport,
    probe_arrays: Mapping[str, torch.Tensor | np.ndarray],
    learning_telemetry: Sequence[Mapping[str, Any]] = (),
    checkpoint_root: Path | None = None,
    checkpoint_lineage: str | None = None,
) -> dict[str, Any]:
    """Produce complete Tier-1 summary, Tier-2 capsule, and optional Tier-3 custody binding."""

    exp_dir = Path(experiment_dir)
    recovery_binding = None
    if checkpoint_root is not None and checkpoint_lineage is not None:
        recovery_binding = bind_tier3_recovery_archive(checkpoint_root, checkpoint_lineage)

    t1 = write_tier1_summary(
        exp_dir / "tier1_summary",
        registration=registration,
        instrument_validation=instrument_validation,
        process_funnel=process_funnel,
        causal_diagnostics=causal_diagnostics,
        interpretation=interpretation,
        recovery_binding=recovery_binding,
    )
    t2 = write_tier2_diagnostic_capsule(
        exp_dir / "tier2_capsule",
        experiment_id=registration.experiment_id,
        probe_arrays=probe_arrays,
        diagnostics={
            "traces": (process_funnel or {}).get("traces", []),
            "learning_telemetry": list(learning_telemetry),
            "causal_diagnostics": dict(causal_diagnostics) if causal_diagnostics else {},
        },
    )
    return {
        "tier1_summary": {k: str(v) for k, v in t1.items()},
        "tier2_capsule": t2,
        "tier3_binding": recovery_binding,
    }


def _safe_arcname(name: str) -> str:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or any(part == "" for part in p.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return p.as_posix()


def _bytes_sha256(raw: bytes) -> tuple[str, int]:
    return sha256(raw).hexdigest(), len(raw)


def _file_sha256(path: Path) -> tuple[str, int]:
    h = sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _zip_member_sha256(zf: zipfile.ZipFile, arc: str) -> tuple[str, int]:
    h = sha256()
    size = 0
    with zf.open(arc, "r") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _build_fp16_model_payload(model_bin_bytes: bytes) -> bytes:
    weights = _load(model_bin_bytes)
    if not isinstance(weights, Mapping):
        raise ValueError("model.bin payload must deserialize to a state_dict mapping")
    fp16_weights: dict[str, Any] = {}
    for key, val in weights.items():
        if isinstance(val, torch.Tensor) and torch.is_floating_point(val):
            fp16_weights[key] = val.detach().cpu().to(dtype=torch.float16)
        elif isinstance(val, torch.Tensor):
            fp16_weights[key] = val.detach().cpu()
        else:
            fp16_weights[key] = val
    return _dump(fp16_weights)


def _write_verified_zip(
    zip_path: Path,
    entries: Mapping[str, Path | bytes],
    *,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []
    sorted_items = sorted((_safe_arcname(arc), src) for arc, src in entries.items())
    for arc, src in sorted_items:
        if isinstance(src, (bytes, bytearray)):
            file_sha, size = _bytes_sha256(bytes(src))
        else:
            file_sha, size = _file_sha256(Path(src))
        manifest_entries.append({"path": arc, "sha256": file_sha, "bytes": size})

    manifest = {
        "schema": "anra-planthir-archive-manifest-v1",
        **dict(meta),
        "files": manifest_entries,
        "manifest_sha256": digest(manifest_entries),
    }
    manifest_bytes = (
        json.dumps(_sanitize_json(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    fd, tmp_name = tempfile.mkstemp(dir=zip_path.parent, prefix=f".{zip_path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for arc, src in sorted_items:
                if isinstance(src, (bytes, bytearray)):
                    zf.writestr(arc, bytes(src))
                else:
                    zf.write(Path(src), arcname=arc)
            zf.writestr("ARCHIVE_MANIFEST.json", manifest_bytes)
        os.replace(tmp_path, zip_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return manifest


def _resolve_experiment_artifacts(
    exp_id: str,
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Any]]:
    result_files: dict[str, Path] = {}
    capsule_files: dict[str, Path] = {}
    summary_data: dict[str, Any] = {}

    t1 = bundle.get("tier1_summary", {})
    if isinstance(t1, Mapping):
        for val in t1.values():
            p = Path(str(val))
            if p.is_file():
                result_files[p.name] = p
                if p.name == "SCIENTIFIC_SUMMARY.json":
                    try:
                        summary_data = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        summary_data = {}

    t2 = bundle.get("tier2_capsule", {})
    cdir_raw = t2.get("capsule_dir") if isinstance(t2, Mapping) else bundle.get("capsule_dir")
    if cdir_raw:
        cdir = Path(str(cdir_raw))
        if cdir.is_dir():
            for fname in ("CAPSULE_MANIFEST.json", "representations.npz", "diagnostics.json"):
                fp = cdir / fname
                if fp.is_file():
                    capsule_files[fname] = fp

    return result_files, capsule_files, summary_data


def _collect_checkpoint_lineages(checkpoint_root: Path | None) -> dict[str, dict[str, Any]]:
    lineages: dict[str, dict[str, Any]] = {}
    if checkpoint_root is None or not Path(checkpoint_root).exists():
        return lineages
    ckpt_root = Path(checkpoint_root)
    candidate_dirs: list[Path] = [p for p in sorted(ckpt_root.iterdir()) if p.is_dir()]
    for ldir in candidate_dirs:
        if not ((ldir / "LATEST").is_file() or (ldir / MILESTONES_FILENAME).is_file()):
            continue
        store = CheckpointStore(ckpt_root, ldir.name)
        latest_sha = store.latest_sha256()
        if latest_sha is None:
            continue
        # Verify committed checkpoint integrity before packaging
        state, payloads = store.restore(latest_sha)
        fp16_model = _build_fp16_model_payload(payloads["model.bin"])
        sched = json.loads(payloads["scheduler.json"])

        compact_files: dict[str, Path | bytes] = {
            f"checkpoints/{ldir.name}/LATEST": ldir / "LATEST",
        }
        fp16_files: dict[str, Path | bytes] = {
            f"checkpoints/{ldir.name}/objects/{latest_sha}/model_fp16.bin": fp16_model,
        }
        if (ldir / MILESTONES_FILENAME).is_file():
            compact_files[f"checkpoints/{ldir.name}/{MILESTONES_FILENAME}"] = ldir / MILESTONES_FILENAME
        obj_dir = store.objects / latest_sha
        for fname in (
            "manifest.json",
            "training_state.json",
            "scheduler.json",
            "cursor.json",
            "ledger.json",
            "rng.bin",
        ):
            fp = obj_dir / fname
            if fp.is_file():
                compact_files[f"checkpoints/{ldir.name}/objects/{latest_sha}/{fname}"] = fp

        latest_full_files: dict[str, Path | bytes] = {
            f"checkpoints/{ldir.name}/objects/{latest_sha}/model.bin": obj_dir / "model.bin",
            f"checkpoints/{ldir.name}/objects/{latest_sha}/optimizer.bin": obj_dir / "optimizer.bin",
        }

        all_store_files: dict[str, Path] = {}
        for item in sorted(ldir.rglob("*")):
            if item.is_file():
                arc = f"checkpoints/{ldir.name}/{item.relative_to(ldir).as_posix()}"
                all_store_files[arc] = item

        lineages[ldir.name] = {
            "lineage_id": ldir.name,
            "latest_sha256": latest_sha,
            "global_update": int(state.global_update),
            "cumulative_tokens": int(state.cumulative_tokens),
            "curriculum_phase": str(state.curriculum_phase),
            "model_spec_sha256": str(state.identities.model_spec_sha256),
            "architecture": str(sched.get("metadata", {}).get("v8_model_spec", {}).get("architecture", "unknown")),
            "compact_files": compact_files,
            "fp16_files": fp16_files,
            "latest_full_files": latest_full_files,
            "all_store_files": all_store_files,
        }
    return lineages


def export_result_bundle(
    output_dir: Path,
    *,
    campaign_id: str,
    source_commit: str,
    status: str,
    experiment_bundles: Mapping[str, Mapping[str, Any]],
    checkpoint_root: Path | None = None,
    max_result_zip_bytes: int = DEFAULT_RESULT_ZIP_MAX_BYTES,
    extra_root_files: Mapping[str, Path | bytes] | None = None,
) -> dict[str, Any]:
    """Export clean, self-describing `result.zip` (compact <=20 MB) and full `checkpoint_full.zip`."""

    if status not in ALLOWED_BUNDLE_STATUSES:
        raise ValueError(f"unsupported bundle status: {status!r}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lineages = _collect_checkpoint_lineages(checkpoint_root)
    base_result_entries: dict[str, Path | bytes] = {}
    full_entries: dict[str, Path | bytes] = {}
    index_entries: list[dict[str, Any]] = []

    reserved_root_names = {
        "ARCHIVE_MANIFEST.json",
        "FINAL_STATUS.json",
        "CHECKPOINT_INDEX.json",
        "RESULT_OVERVIEW.md",
    }
    standalone_extras: dict[str, bytes] = {}
    if extra_root_files:
        for raw_arc, src in extra_root_files.items():
            arc = _safe_arcname(raw_arc)
            if arc in reserved_root_names:
                raise ValueError(f"cannot override reserved archive file {arc!r}")
            raw_bytes = bytes(src) if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
            base_result_entries[arc] = raw_bytes
            full_entries[arc] = raw_bytes
            if "/" not in arc:
                standalone_extras[arc] = raw_bytes

    for exp_id in sorted(experiment_bundles):
        bundle = experiment_bundles[exp_id]
        res_files, cap_files, summary_data = _resolve_experiment_artifacts(exp_id, bundle)
        for fname, fpath in res_files.items():
            arc = f"results/{exp_id}/{fname}"
            base_result_entries[arc] = fpath
            full_entries[arc] = fpath
        for fname, fpath in cap_files.items():
            arc = f"capsules/{exp_id}/{fname}"
            base_result_entries[arc] = fpath
            full_entries[arc] = fpath

        reg = summary_data.get("registration", {})
        rb = bundle.get("tier3_binding") or summary_data.get("recovery_binding") or {}
        lineage_id = bundle.get("lineage_id") or rb.get("lineage")
        sha = bundle.get("checkpoint_sha256") or rb.get("latest_checkpoint_sha256")
        if lineage_id and lineage_id in lineages and sha is None:
            sha = lineages[lineage_id]["latest_sha256"]

        funnel = summary_data.get("process_funnel", {}) or {}
        interp = summary_data.get("interpretation", {}) or {}
        lin_meta = lineages.get(str(lineage_id), {}) if lineage_id else {}
        arch = reg.get("architecture_family") or lin_meta.get("architecture") or "unknown"

        entry = {
            "experiment_id": exp_id,
            "program": reg.get("program", campaign_id),
            "architecture": arch,
            "parameter_count": reg.get("parameter_count"),
            "lineage_id": lineage_id,
            "latest_checkpoint_sha256": sha,
            "global_update": lin_meta.get("global_update"),
            "overall_accuracy": funnel.get("overall_accuracy"),
            "interpretation_status": interp.get("status"),
            "results_prefix": f"results/{exp_id}/" if res_files else None,
            "capsule_prefix": f"capsules/{exp_id}/" if cap_files else None,
            "checkpoint_prefix": f"checkpoints/{lineage_id}/" if lineage_id else None,
        }
        index_entries.append(entry)

    # Track any standalone lineages under checkpoint_root not already referenced
    referenced_lineages = {e["lineage_id"] for e in index_entries if e.get("lineage_id")}
    for lin_id, linfo in sorted(lineages.items()):
        if lin_id not in referenced_lineages:
            index_entries.append({
                "experiment_id": None,
                "program": campaign_id,
                "architecture": linfo.get("architecture"),
                "parameter_count": None,
                "lineage_id": lin_id,
                "latest_checkpoint_sha256": linfo["latest_sha256"],
                "global_update": linfo["global_update"],
                "overall_accuracy": None,
                "interpretation_status": None,
                "results_prefix": None,
                "capsule_prefix": None,
                "checkpoint_prefix": f"checkpoints/{lin_id}/",
            })

    for linfo in lineages.values():
        full_entries.update(linfo["all_store_files"])

    def _build_metadata_files(embedded_full: bool) -> tuple[bytes, bytes, bytes]:
        status_doc = {
            "schema": "anra-planthir-bundle-status-v1",
            "campaign_id": campaign_id,
            "source_commit": source_commit,
            "status": status,
            "experiment_count": len(experiment_bundles),
            "checkpoint_count": len(lineages),
            "whole_checkpoint_embedded": embedded_full,
        }
        status_bytes = (
            json.dumps(_sanitize_json(status_doc), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")

        index_doc = {
            "schema": "anra-planthir-checkpoint-index-v1",
            "campaign_id": campaign_id,
            "source_commit": source_commit,
            "whole_checkpoint_embedded": embedded_full,
            "checkpoint_mode_in_result_zip": "full_embedded" if embedded_full else "compact_fp16",
            "entries": index_entries,
        }
        index_bytes = (
            json.dumps(_sanitize_json(index_doc), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")

        lines = [
            f"# {campaign_id} — Clear Result & Checkpoint Overview",
            "",
            f"- **Status:** `{status}` | **Source Commit:** `{source_commit}`",
            f"- **Checkpoints Indexed:** `{len(lineages)}` | **`result.zip` Checkpoint Mode:** `{'full_embedded (model.bin + optimizer.bin)' if embedded_full else 'compact_fp16 (model_fp16.bin + verified metadata; full optimizer/FP32 in checkpoint_full.zip)'}`",
            "",
            "| Experiment ID | Arch | Params | Lineage ID | Checkpoint SHA-256 | Updates | Overall Acc | Interp Status |",
            "| :--- | :--- | ---: | :--- | :--- | ---: | ---: | :--- |",
        ]
        for row in index_entries:
            p_str = f"{row['parameter_count']:,}" if row["parameter_count"] is not None else "n/a"
            sha_short = f"`{row['latest_checkpoint_sha256'][:12]}`" if row["latest_checkpoint_sha256"] else "none"
            upd = row["global_update"] if row["global_update"] is not None else "n/a"
            acc = f"{row['overall_accuracy']:.3f}" if row["overall_accuracy"] is not None else "n/a"
            lines.append(
                f"| `{row['experiment_id'] or 'unlinked'}` | `{row['architecture']}` | {p_str} | "
                f"`{row['lineage_id'] or 'none'}` | {sha_short} | {upd} | {acc} | `{row['interpretation_status'] or 'n/a'}` |"
            )
        lines.extend([
            "",
            "## Archive Layout",
            "",
            "- `FINAL_STATUS.json`, `CHECKPOINT_INDEX.json`, `RESULT_OVERVIEW.md`, `ARCHIVE_MANIFEST.json`",
            "- `results/<experiment_id>/`: Tier-1 `SCIENTIFIC_SUMMARY.json` + `SCIENTIFIC_SUMMARY.md`",
            "- `capsules/<experiment_id>/`: SHA-256 verified Tier-2 `representations.npz` + `diagnostics.json`",
            "- `checkpoints/<lineage_id>/`: Committed `CheckpointStore` files for zero-retraining causal probing & recovery",
            "",
        ])
        overview_bytes = ("\n".join(lines)).encode("utf-8")
        return status_bytes, index_bytes, overview_bytes

    # Candidate result.zip: embed clean CheckpointStore latest object (model.bin + optimizer.bin) if <= budget,
    # otherwise fall back to compact_fp16 (model_fp16.bin + verified store metadata).
    result_zip_path = out / RESULT_ZIP_NAME
    full_zip_path = out / FULL_CHECKPOINT_ZIP_NAME
    compact_entries = dict(base_result_entries)
    embedded_entries = dict(base_result_entries)
    for linfo in lineages.values():
        compact_entries.update(linfo["compact_files"])
        compact_entries.update(linfo["fp16_files"])
        embedded_entries.update(linfo["compact_files"])
        embedded_entries.update(linfo["latest_full_files"])

    meta_common = {"campaign_id": campaign_id, "source_commit": source_commit, "status": status}

    status_bytes, index_bytes, overview_bytes = _build_metadata_files(embedded_full=True)
    for fname, raw in (
        ("FINAL_STATUS.json", status_bytes),
        ("CHECKPOINT_INDEX.json", index_bytes),
        ("RESULT_OVERVIEW.md", overview_bytes),
    ):
        embedded_entries[fname] = raw

    _write_verified_zip(result_zip_path, embedded_entries, meta={**meta_common, "kind": "result"})
    whole_checkpoint_embedded = result_zip_path.stat().st_size <= max_result_zip_bytes

    if not whole_checkpoint_embedded:
        status_bytes, index_bytes, overview_bytes = _build_metadata_files(embedded_full=False)
        for fname, raw in (
            ("FINAL_STATUS.json", status_bytes),
            ("CHECKPOINT_INDEX.json", index_bytes),
            ("RESULT_OVERVIEW.md", overview_bytes),
        ):
            compact_entries[fname] = raw
        _write_verified_zip(result_zip_path, compact_entries, meta={**meta_common, "kind": "result"})
        if result_zip_path.stat().st_size > max_result_zip_bytes:
            raise ValueError(
                f"compact {RESULT_ZIP_NAME} size {result_zip_path.stat().st_size} B exceeds cap {max_result_zip_bytes} B"
            )

    # Always write full CheckpointStore archive containing matching status/index/overview metadata
    for fname, raw in (
        ("FINAL_STATUS.json", status_bytes),
        ("CHECKPOINT_INDEX.json", index_bytes),
        ("RESULT_OVERVIEW.md", overview_bytes),
    ):
        full_entries[fname] = raw
    _write_verified_zip(full_zip_path, full_entries, meta={**meta_common, "kind": "checkpoint_full"})

    # Write standalone readable files in output_dir matching the ZIPs byte-for-byte
    (out / "FINAL_STATUS.json").write_bytes(status_bytes)
    (out / "CHECKPOINT_INDEX.json").write_bytes(index_bytes)
    (out / "RESULT_OVERVIEW.md").write_bytes(overview_bytes)
    for fname, raw_bytes in standalone_extras.items():
        (out / fname).write_bytes(raw_bytes)

    verified = verify_result_bundle(
        out,
        expected_source_commit=source_commit,
        max_result_zip_bytes=max_result_zip_bytes,
    )
    return {
        "schema": "anra-planthir-bundle-receipt-v1",
        "output_dir": str(out),
        "result_zip": str(result_zip_path),
        "checkpoint_full_zip": str(full_zip_path),
        "result_zip_bytes": int(result_zip_path.stat().st_size),
        "checkpoint_full_zip_bytes": int(full_zip_path.stat().st_size),
        "whole_checkpoint_embedded": whole_checkpoint_embedded,
        "checkpoint_count": len(lineages),
        "experiment_count": len(experiment_bundles),
        **verified,
    }


def verify_single_planthir_zip(
    zip_path: Path | str,
    *,
    expected_source_commit: str | None = None,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Fail-closed SHA-256 verification of `result.zip` or `checkpoint_full.zip`."""

    zp = Path(zip_path)
    if not zp.is_file():
        raise ValueError(f"missing Planthir archive: {zp}")
    zip_size = int(zp.stat().st_size)
    if max_bytes is not None and zip_size > max_bytes:
        raise ValueError(f"archive {zp.name} ({zip_size} B) exceeds size limit {max_bytes} B")

    with zipfile.ZipFile(zp, "r") as zf:
        names = set(zf.namelist())
        required = {"ARCHIVE_MANIFEST.json", "FINAL_STATUS.json", "CHECKPOINT_INDEX.json", "RESULT_OVERVIEW.md"}
        missing_req = sorted(required - names)
        if missing_req:
            raise ValueError(f"archive {zp.name} missing required metadata: {missing_req}")
        for name in names:
            _safe_arcname(name)

        manifest = json.loads(zf.read("ARCHIVE_MANIFEST.json").decode("utf-8"))
        if manifest.get("schema") != "anra-planthir-archive-manifest-v1":
            raise ValueError("unsupported Planthir archive manifest schema")
        if expected_source_commit is not None and manifest.get("source_commit") != expected_source_commit:
            raise ValueError(
                f"archive source_commit mismatch: got {manifest.get('source_commit')!r}, expected {expected_source_commit!r}"
            )
        files = manifest.get("files", [])
        if digest(files) != manifest.get("manifest_sha256"):
            raise ValueError("archive manifest digest mismatch")
        declared = {str(entry["path"]): entry for entry in files}
        if set(declared) | {"ARCHIVE_MANIFEST.json"} != names:
            raise ValueError("archive contents differ from ARCHIVE_MANIFEST.json")

        for arc, entry in declared.items():
            actual_sha, actual_size = _zip_member_sha256(zf, arc)
            if actual_size != int(entry["bytes"]) or actual_sha != str(entry["sha256"]):
                raise ValueError(f"checksum/size mismatch in {zp.name} member: {arc}")

        status_doc = json.loads(zf.read("FINAL_STATUS.json").decode("utf-8"))
        index_doc = json.loads(zf.read("CHECKPOINT_INDEX.json").decode("utf-8"))
        if status_doc.get("status") not in ALLOWED_BUNDLE_STATUSES:
            raise ValueError(f"invalid FINAL_STATUS status: {status_doc.get('status')!r}")
        if expected_source_commit is not None and status_doc.get("source_commit") != expected_source_commit:
            raise ValueError("FINAL_STATUS source_commit mismatch")
        if index_doc.get("source_commit") != status_doc.get("source_commit"):
            raise ValueError("CHECKPOINT_INDEX source_commit disagrees with FINAL_STATUS")

        # Verify embedded Tier-2 capsules inside archive match their internal CAPSULE_MANIFEST.json
        for arc in sorted(declared):
            if arc.startswith("capsules/") and arc.endswith("/CAPSULE_MANIFEST.json"):
                cap_prefix = arc[: -len("CAPSULE_MANIFEST.json")]
                cap_manifest = json.loads(zf.read(arc).decode("utf-8"))
                for cname, cmeta in cap_manifest.get("members", {}).items():
                    member_arc = f"{cap_prefix}{cname}"
                    if member_arc not in declared:
                        raise ValueError(f"capsule member missing in archive: {member_arc}")
                    m_sha, m_size = _zip_member_sha256(zf, member_arc)
                    if m_size != int(cmeta["byte_size"]) or m_sha != str(cmeta["sha256"]):
                        raise ValueError(f"capsule corruption detected inside archive: {member_arc}")

    return {
        "zip_path": str(zp),
        "zip_bytes": zip_size,
        "manifest": manifest,
        "final_status": status_doc,
        "checkpoint_index": index_doc,
    }


def verify_result_bundle(
    output_dir: Path | str,
    *,
    expected_source_commit: str | None = None,
    max_result_zip_bytes: int = DEFAULT_RESULT_ZIP_MAX_BYTES,
) -> dict[str, Any]:
    """Verify standalone metadata, compact `result.zip`, and full `checkpoint_full.zip` consistency."""

    out = Path(output_dir)
    for fname in ("FINAL_STATUS.json", "CHECKPOINT_INDEX.json", "RESULT_OVERVIEW.md"):
        if not (out / fname).is_file():
            raise ValueError(f"missing {fname} in {out}")

    r_info = verify_single_planthir_zip(
        out / RESULT_ZIP_NAME,
        expected_source_commit=expected_source_commit,
        max_bytes=max_result_zip_bytes,
    )
    f_info = verify_single_planthir_zip(
        out / FULL_CHECKPOINT_ZIP_NAME,
        expected_source_commit=expected_source_commit,
        max_bytes=None,
    )

    for zpath in (out / RESULT_ZIP_NAME, out / FULL_CHECKPOINT_ZIP_NAME):
        with zipfile.ZipFile(zpath, "r") as zf:
            for fname in ("FINAL_STATUS.json", "CHECKPOINT_INDEX.json", "RESULT_OVERVIEW.md"):
                if zf.read(fname) != (out / fname).read_bytes():
                    raise ValueError(f"{fname} inside {zpath.name} differs from standalone file")

    return {
        "verified": True,
        "status": r_info["final_status"]["status"],
        "source_commit": r_info["final_status"]["source_commit"],
        "result_manifest_sha256": r_info["manifest"]["manifest_sha256"],
        "full_manifest_sha256": f_info["manifest"]["manifest_sha256"],
    }


__all__ = [
    "DEFAULT_RESULT_ZIP_MAX_BYTES",
    "FULL_CHECKPOINT_ZIP_NAME",
    "RESULT_ZIP_NAME",
    "TIER1_MAX_BYTES",
    "TIER2_MAX_BYTES",
    "bind_tier3_recovery_archive",
    "export_result_bundle",
    "package_three_tier_evidence",
    "probe_from_capsule",
    "verify_result_bundle",
    "verify_single_planthir_zip",
    "verify_tier2_capsule",
    "write_tier1_summary",
    "write_tier2_diagnostic_capsule",
]
