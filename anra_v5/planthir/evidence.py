"""Three-Tier Evidence Architecture: Tier-1 Summary, Tier-2 Diagnostic Capsule, Tier-3 Binding."""

from __future__ import annotations

from hashlib import sha256
from dataclasses import asdict
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import torch

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.training import _dump, _load
from v5_training.checkpoint import (CheckpointStore, MILESTONES_FILENAME, MANIFEST_SCHEMA,
                                    REQUIRED_COMPONENTS, _canonical_json)
from v5_training.state import TrainingState

from .causal import _eval_ridge_classifier, _fit_ridge_classifier
from .contracts import ExperimentRegistration
from .execution import _atomic_write, _sync_dir, trigger_fault_point
from .interpretation import ScientificInterpretationReport
from .observatory import compute_effective_rank

TIER1_MAX_BYTES = 100 * 1024
TIER2_MAX_BYTES = 2 * 1024 * 1024
RESULT_ZIP_NAME = "result.zip"
FULL_CHECKPOINT_ZIP_NAME = "checkpoint_full.zip"
DEFAULT_RESULT_ZIP_MAX_BYTES = 20 * 1024 * 1024
ALLOWED_BUNDLE_STATUSES = ("COMPLETE", "PARTIAL", "NOT_RUN", "NO_GO", "INVALID_RESULTS")
BUNDLE_POINTER_SCHEMA = "anra-planthir-bundle-pointer-v1"
BUNDLE_POINTER_NAME = "CURRENT.json"
GENERATIONS_DIR_NAME = "generations"
QUARANTINE_DIR_NAME = ".quarantine"


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


def _compact_evidence(value: Any, reference: str, pointer: str = "") -> Any:
    """Replace bulky fields with explicit JSON-pointer references to preserved evidence."""
    if isinstance(value, dict):
        return {str(k): _compact_evidence(v, reference, pointer + "/" + str(k).replace("~", "~0").replace("/", "~1"))
                if k != "evidence_basis" else {"artifact": reference, "json_pointer": pointer + "/evidence_basis"}
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 16:
            return {"artifact": reference, "json_pointer": pointer, "item_count": len(value)}
        return [_compact_evidence(v, reference, pointer + "/" + str(i)) for i, v in enumerate(value)]
    if isinstance(value, str) and len(value) > 1024:
        return {"artifact": reference, "json_pointer": pointer, "characters": len(value)}
    return value


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
    raw_dir = out.parent / "raw_evidence"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_summary = dict(summary_payload)
    raw_summary["process_funnel"] = dict(process_funnel) if process_funnel is not None else None
    _write_json(raw_dir / "tier1_full.json", raw_summary)
    reference = f"raw_evidence/{registration.experiment_id}/tier1_full.json"
    summary_payload = _compact_evidence(summary_payload, reference)
    summary_payload["raw_evidence"] = reference
    json_path = out / "SCIENTIFIC_SUMMARY.json"
    raw_json = _write_json(json_path, summary_payload)
    if len(raw_json) > max_bytes:
        for section in ("causal_diagnostics", "interpretation", "process_funnel", "instrument_validation"):
            summary_payload[section] = {"artifact": reference, "json_pointer": "/" + section}
            raw_json = _write_json(json_path, summary_payload)
            if len(raw_json) <= max_bytes:
                break
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
    if len(md_bytes) > max_bytes:
        md_bytes = (f"# {registration.experiment_id} - Tier-1 Scientific Summary\n\n"
                    f"Interpretation status: `{interp.status}`. The full claims and evidence are in "
                    f"`{reference}`; see SCIENTIFIC_SUMMARY.json for the bounded summary.\n").encode("utf-8")
    if len(md_bytes) > max_bytes:
        raise ValueError("Tier-1 Markdown metadata exceeds limit")
    md_path.write_bytes(md_bytes)
    return {"json": json_path, "markdown": md_path}


def write_tier2_diagnostic_capsule(
    capsule_dir: Path,
    *,
    experiment_id: str,
    probe_arrays: Mapping[str, torch.Tensor | np.ndarray],
    diagnostics: Mapping[str, Any],
    max_bytes: int = TIER2_MAX_BYTES,
    raw_evidence_link: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write compact Tier-2 Diagnostic Capsule (.json + float16 .npz) with SHA-256 manifest."""

    cdir = Path(capsule_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    raw_dir = cdir.parent / "raw_evidence"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_arrays = {key: (val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val))
                  for key, val in probe_arrays.items()}
    if any(arr.dtype.hasobject for arr in raw_arrays.values()):
        raise ValueError("object arrays cannot be preserved safely in capsule evidence")
    np.savez_compressed(raw_dir / "representations_full.npz", **raw_arrays)
    _write_json(raw_dir / "diagnostics_full.json", diagnostics)
    raw_prefix = f"raw_evidence/{experiment_id}"
    compact_diagnostics = _compact_evidence(dict(diagnostics), raw_prefix + "/diagnostics_full.json")
    diag_path = cdir / "diagnostics.json"
    _write_json(diag_path, compact_diagnostics)
    if diag_path.stat().st_size > max_bytes // 2:
        _write_json(diag_path, {"artifact": raw_prefix + "/diagnostics_full.json", "json_pointer": ""})

    rows = len(raw_arrays["decision_hidden"]) if "decision_hidden" in raw_arrays else 0
    selected = rows
    npz_path = cdir / "representations.npz"
    while True:
        np_arrays = {}
        for key, arr in raw_arrays.items():
            # Keep aligned row arrays together; other arrays remain in the raw sidecar.
            if arr.ndim == 0 or len(arr) != rows:
                continue
            sample = arr[:selected]
            if np.issubdtype(sample.dtype, np.floating):
                sample = sample.astype(np.float16)
            elif np.issubdtype(sample.dtype, np.integer):
                sample = sample.astype(np.int64)
            np_arrays[key] = sample
        np.savez_compressed(npz_path, **np_arrays)
        total_bytes = npz_path.stat().st_size + diag_path.stat().st_size
        if total_bytes <= max_bytes:
            break
        if selected <= 1:
            raise ValueError(f"Tier-2 capsule metadata or one sample exceeds limit {max_bytes} B")
        selected = max(1, selected // 2)

    members = {}
    for fname in ("representations.npz", "diagnostics.json"):
        file_sha, size = _file_sha256(cdir / fname)
        members[fname] = {"byte_size": size, "sha256": file_sha}
    selected_indices = list(range(selected))
    omitted_fields = sorted(set(raw_arrays) - set(np_arrays))
    if raw_evidence_link is not None:
        if raw_evidence_link.get("selected_row_indices") is not None:
            selected_indices = [int(i) for i in raw_evidence_link["selected_row_indices"]]
        if raw_evidence_link.get("omitted_fields"):
            omitted_fields = sorted(set(omitted_fields) | {str(x) for x in raw_evidence_link["omitted_fields"]})

    manifest = {
        "schema": "anra-planthir-tier2-capsule-v1", "experiment_id": experiment_id,
        "total_bytes": total_bytes,
        "array_shapes": {k: list(v.shape) for k, v in np_arrays.items()},
        "array_dtypes": {k: str(v.dtype) for k, v in np_arrays.items()},
        "sampling": {"method": "first_rows", "original_rows": rows, "selected_rows": selected,
                     "omitted_arrays": sorted(set(raw_arrays) - set(np_arrays))},
        "raw_evidence": {"arrays": raw_prefix + "/representations_full.npz",
                         "diagnostics": raw_prefix + "/diagnostics_full.json"},
        "raw_evidence_link": {
            "raw_manifest_sha256": raw_evidence_link.get("raw_manifest_sha256") if raw_evidence_link else None,
            "selected_row_indices": selected_indices,
            "omitted_fields": omitted_fields,
            "lossy_quantization": {
                "float_cast": "float16",
                "source_dtypes": {k: str(v.dtype) for k, v in raw_arrays.items()},
                "quantized_keys": sorted(k for k, v in np_arrays.items() if str(v.dtype) == "float16"),
            },
        },
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
    group_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run exploratory capsule probes; these do not qualify independent generalization.

    Optional group IDs keep related rows together; callers must supply meaningful
    groups and arrange an independent evaluation for scientific qualification.
    """

    manifest = verify_tier2_capsule(capsule_dir)
    cdir = Path(capsule_dir)
    with np.load(cdir / "representations.npz") as archive:
        if "decision_hidden" not in archive:
            raise ValueError("capsule representations.npz missing 'decision_hidden'")
        hidden = torch.from_numpy(archive["decision_hidden"].astype(np.float32))
        if hidden.ndim != 2 or len(hidden) < 4 or not 0 < train_fraction < 1:
            raise ValueError("capsule probe requires at least four rows and 0 < train_fraction < 1")
        results: dict[str, Any] = {
            "schema": "anra-planthir-capsule-probe-v1",
            "qualification": "exploratory_only",
            "independent_evaluation": False,
            "experiment_id": manifest["experiment_id"],
            "samples": int(hidden.shape[0]),
            "hidden_dim": int(hidden.shape[1]),
            "effective_rank": compute_effective_rank(hidden),
            "probes": {},
        }
        if group_ids is None:
            split = max(2, min(hidden.shape[0] - 2, int(hidden.shape[0] * train_fraction)))
            train_mask = torch.arange(len(hidden)) < split
            results["split_method"] = "row_order"
        else:
            if len(group_ids) != len(hidden) or any(not isinstance(g, str) or not g for g in group_ids):
                raise ValueError("group_ids must contain one nonempty string per row")
            groups = list(dict.fromkeys(group_ids))
            if len(groups) < 2:
                raise ValueError("group split requires at least two groups")
            train_groups = set()
            target = len(hidden) * train_fraction
            for group in groups[:-1]:
                if train_groups and sum(g in train_groups for g in group_ids) >= target:
                    break
                train_groups.add(group)
            train_mask = torch.tensor([g in train_groups for g in group_ids], dtype=torch.bool)
            results["split_method"] = "group_order"
        eval_mask = ~train_mask
        x_tr, x_ev = hidden[train_mask], hidden[eval_mask]
        for key in ("y_op", "y_hop1", "y_final"):
            if key not in archive:
                continue
            y = torch.from_numpy(archive[key].astype(np.int64))
            if y.ndim != 1 or len(y) != len(hidden):
                raise ValueError(f"invalid capsule label shape for {key}")
            y_tr, y_ev = y[train_mask], y[eval_mask]
            train_valid, eval_valid = y_tr >= 0, y_ev >= 0
            if not train_valid.any() or not eval_valid.any():
                results["probes"][key] = None
                continue
            num_classes = int(y_tr[train_valid].max().item()) + 1
            if (y_ev[eval_valid] >= num_classes).any():
                num_classes = int(y.max().item()) + 1
            w = _fit_ridge_classifier(x_tr[train_valid], y_tr[train_valid], num_classes, l2_reg=l2_reg)
            acc = _eval_ridge_classifier(w, x_ev[eval_valid], y_ev[eval_valid])
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
    _verify_local_checkpoint_streaming(store, latest)
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
        "raw_evidence_dir": str(exp_dir / "raw_evidence"),
    }


def _safe_arcname(name: str) -> str:
    p = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or p.is_absolute()
            or ".." in p.parts or p.as_posix() != name):
        raise ValueError(f"unsafe archive member path: {name!r}")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(part.rstrip(" .") != part or part.split(".")[0].upper() in reserved for part in p.parts):
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
            converted = val.detach().cpu().to(dtype=torch.float16)
            if torch.isfinite(val).all() and not torch.isfinite(converted).all():
                raise ValueError(f"FP16 conversion overflows for {key!r}")
            fp16_weights[key] = converted
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
    sorted_items = sorted((_safe_arcname(arc), src) for arc, src in entries.items())
    fd, tmp_name = tempfile.mkstemp(dir=zip_path.parent, prefix=f".{zip_path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        manifest_entries: list[dict[str, Any]] = []
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for arc, src in sorted_items:
                h, size = sha256(), 0
                with zf.open(arc, "w", force_zip64=True) as dst:
                    if isinstance(src, (bytes, bytearray)):
                        raw = bytes(src)
                        dst.write(raw)
                        h.update(raw)
                        size = len(raw)
                    else:
                        with Path(src).open("rb") as inp:
                            while chunk := inp.read(1024 * 1024):
                                dst.write(chunk)
                                h.update(chunk)
                                size += len(chunk)
                manifest_entries.append({"path": arc, "sha256": h.hexdigest(), "bytes": size})
            manifest = {"schema": "anra-planthir-archive-manifest-v1", **dict(meta),
                        "files": manifest_entries, "manifest_sha256": digest(manifest_entries)}
            manifest_bytes = (json.dumps(_sanitize_json(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
            zf.writestr("ARCHIVE_MANIFEST.json", manifest_bytes)
        os.replace(tmp_path, zip_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
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


def _verify_local_checkpoint_streaming(store: CheckpointStore, sha: str) -> TrainingState:
    """Verify a local v1 checkpoint without materializing model or optimizer bytes."""
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("invalid checkpoint SHA-256")
    obj = store.objects / sha
    manifest_path = obj / "manifest.json"
    if obj.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("checkpoint manifest is missing")
    raw = manifest_path.read_bytes()
    if sha256(raw).hexdigest() != sha:
        raise ValueError("checkpoint manifest hash mismatch")
    manifest = json.loads(raw)
    if set(manifest) != {"schema", "lineage_id", "state_sha256", "state", "components", "durability"} or manifest["schema"] != MANIFEST_SCHEMA or manifest["lineage_id"] != store.lineage_id:
        raise ValueError("unsupported checkpoint schema or lineage")
    components = manifest["components"]
    if len(components) != len(REQUIRED_COMPONENTS) or {item["name"] for item in components} != REQUIRED_COMPONENTS:
        raise ValueError("checkpoint component inventory is incomplete")
    if {p.name for p in obj.iterdir()} != set(REQUIRED_COMPONENTS) | {"manifest.json"}:
        raise ValueError("checkpoint directory contains missing or untracked components")
    for item in components:
        if set(item) != {"name", "sha256", "byte_size"}:
            raise ValueError("invalid checkpoint component identity")
        path = obj / item["name"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint component is missing")
        actual_sha, size = _file_sha256(path)
        if size != item["byte_size"] or actual_sha != item["sha256"]:
            raise ValueError(f"checkpoint component is corrupt: {item['name']}")
    state = TrainingState.from_dict(manifest["state"])
    if state.lineage_id != store.lineage_id or state.sha256() != manifest["state_sha256"]:
        raise ValueError("checkpoint training state mismatch")
    if (obj / "training_state.json").read_bytes() != _canonical_json(state.canonical()):
        raise ValueError("training state payload mismatch")
    if json.loads((obj / "cursor.json").read_bytes()) != json.loads(_canonical_json(asdict(state.cursor))) or json.loads((obj / "ledger.json").read_bytes()) != dict(state.tokens_by_source):
        raise ValueError("checkpoint cursor or ledger disagrees with state")
    return state


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
        state = _verify_local_checkpoint_streaming(store, latest_sha)
        obj_dir = store.objects / latest_sha
        model_payload = obj_dir / "model.bin"
        sched = json.loads((obj_dir / "scheduler.json").read_bytes())

        compact_files: dict[str, Path | bytes] = {
            f"checkpoints/{ldir.name}/LATEST": ldir / "LATEST",
        }
        fp16_files: dict[str, Path | bytes] = {
            f"checkpoints/{ldir.name}/objects/{latest_sha}/model_fp16.bin": model_payload,
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
            "architecture": str(
                sched.get("metadata", {}).get("v8_model_spec", {}).get("architecture")
                or sched.get("metadata", {}).get("x1_model_spec", {}).get("architecture", "unknown")
            ),
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

        raw_dir_value = bundle.get("raw_evidence_dir")
        if raw_dir_value:
            raw_dir = Path(str(raw_dir_value))
            required_raw = ("tier1_full.json", "diagnostics_full.json", "representations_full.npz")
            for fname in required_raw:
                fpath = raw_dir / fname
                if not fpath.is_file():
                    raise ValueError(f"missing raw evidence for {exp_id}: {fname}")
                full_entries[f"raw_evidence/{exp_id}/{fname}"] = fpath

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

    def _build_metadata_files(embedded_full: bool, evidence_only: bool = False) -> tuple[bytes, bytes, bytes]:
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
            "checkpoint_mode_in_result_zip": "full_embedded" if embedded_full else ("sidecar_only" if evidence_only else "compact_fp16"),
            "full_checkpoint_sidecar": FULL_CHECKPOINT_ZIP_NAME,
            "entries": index_entries,
        }
        index_bytes = (
            json.dumps(_sanitize_json(index_doc), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")

        lines = [
            f"# {campaign_id} — Clear Result & Checkpoint Overview",
            "",
            f"- **Status:** `{status}` | **Source Commit:** `{source_commit}`",
            f"- **Checkpoints Indexed:** `{len(lineages)}` | **`result.zip` Checkpoint Mode:** `{'full_embedded' if embedded_full else ('sidecar_only (checkpoint_full.zip required)' if evidence_only else 'compact_fp16 (lossy; full checkpoint in checkpoint_full.zip)')}`",
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
        # Conversion is deferred until the full candidate exceeds the size cap.
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

    candidate_path = out / ("." + RESULT_ZIP_NAME + ".candidate")
    _write_verified_zip(candidate_path, embedded_entries, meta={**meta_common, "kind": "result"})
    whole_checkpoint_embedded = candidate_path.stat().st_size <= max_result_zip_bytes

    if not whole_checkpoint_embedded:
        model_files = [(arc, path) for linfo in lineages.values() for arc, path in linfo["fp16_files"].items()]
        # Bound aggregate conversion and serialization memory; larger models use the full sidecar.
        skip_compact = sum(path.stat().st_size for _, path in model_files) > max_result_zip_bytes * 2
        if not skip_compact:
            for arc, model_path in model_files:
                compact_entries[arc] = _build_fp16_model_payload(model_path.read_bytes())
        status_bytes, index_bytes, overview_bytes = _build_metadata_files(embedded_full=False)
        for fname, raw in (
            ("FINAL_STATUS.json", status_bytes),
            ("CHECKPOINT_INDEX.json", index_bytes),
            ("RESULT_OVERVIEW.md", overview_bytes),
        ):
            compact_entries[fname] = raw
        if not skip_compact:
            _write_verified_zip(candidate_path, compact_entries, meta={**meta_common, "kind": "result"})
        if skip_compact or candidate_path.stat().st_size > max_result_zip_bytes:
            status_bytes, index_bytes, overview_bytes = _build_metadata_files(embedded_full=False, evidence_only=True)
            evidence_entries = dict(base_result_entries)
            evidence_entries.update({"FINAL_STATUS.json": status_bytes, "CHECKPOINT_INDEX.json": index_bytes, "RESULT_OVERVIEW.md": overview_bytes})
            _write_verified_zip(candidate_path, evidence_entries, meta={**meta_common, "kind": "result"})
            if candidate_path.stat().st_size > max_result_zip_bytes:
                candidate_size = candidate_path.stat().st_size
                candidate_path.unlink()
                raise ValueError(f"evidence-only {RESULT_ZIP_NAME} size {candidate_size} B exceeds cap {max_result_zip_bytes} B")
    os.replace(candidate_path, result_zip_path)
    result_zip_sha256, _ = _file_sha256(result_zip_path)
    (out / "result.zip.sha256").write_text(
        f"{result_zip_sha256}  {RESULT_ZIP_NAME}\n", encoding="ascii"
    )

    # Always write full CheckpointStore archive containing matching status/index/overview metadata
    for fname, raw in (
        ("FINAL_STATUS.json", status_bytes),
        ("CHECKPOINT_INDEX.json", index_bytes),
        ("RESULT_OVERVIEW.md", overview_bytes),
    ):
        full_entries[fname] = raw
    _write_verified_zip(full_zip_path, full_entries, meta={**meta_common, "kind": "checkpoint_full"})
    full_zip_sha256, _ = _file_sha256(full_zip_path)
    (out / "checkpoint_full.zip.sha256").write_text(
        f"{full_zip_sha256}  {FULL_CHECKPOINT_ZIP_NAME}\n", encoding="ascii"
    )

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
        "result_zip_sha256": result_zip_sha256,
        "checkpoint_full_zip_bytes": int(full_zip_path.stat().st_size),
        "checkpoint_full_zip_sha256": full_zip_sha256,
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
    max_uncompressed_bytes: int = 128 * 1024**3,
    max_members: int = 100_000,
) -> dict[str, Any]:
    """Fail-closed SHA-256 verification of `result.zip` or `checkpoint_full.zip`."""

    zp = Path(zip_path)
    if not zp.is_file():
        raise ValueError(f"missing Planthir archive: {zp}")
    zip_size = int(zp.stat().st_size)
    if max_bytes is not None and zip_size > max_bytes:
        raise ValueError(f"archive {zp.name} ({zip_size} B) exceeds size limit {max_bytes} B")

    with zipfile.ZipFile(zp, "r") as zf:
        member_names = zf.namelist()
        if len(member_names) != len(set(member_names)):
            raise ValueError("archive contains duplicate member names")
        if len(member_names) > max_members:
            raise ValueError("archive exceeds member count limit")
        if sum(info.file_size for info in zf.infolist()) > max_uncompressed_bytes:
            raise ValueError("archive exceeds uncompressed size limit")
        if len({name.casefold() for name in member_names}) != len(member_names):
            raise ValueError("archive contains case-insensitive member aliases")
        names = set(member_names)
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
        if len(declared) != len(files) or "ARCHIVE_MANIFEST.json" in declared:
            raise ValueError("archive manifest contains duplicate or reserved paths")
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
                if (cap_manifest.get("schema") != "anra-planthir-tier2-capsule-v1"
                        or set(cap_manifest.get("members", {})) != {"representations.npz", "diagnostics.json"}):
                    raise ValueError(f"invalid capsule manifest: {arc}")
                for cname, cmeta in cap_manifest["members"].items():
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


def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f".tmp-{dst.name}-")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle, src.open("rb") as inp:
            shutil.copyfileobj(inp, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, dst)
        _sync_dir(dst.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def quarantine_partial_generations(bundle_dir: Path | str) -> list[str]:
    """Move incomplete staging/backup/temp bundle entries into `.quarantine` without altering CURRENT.json."""
    bdir = Path(bundle_dir)
    if not bdir.exists():
        return []
    qdir = bdir / QUARANTINE_DIR_NAME
    quarantined: list[str] = []
    for child in sorted(bdir.iterdir()):
        if child.name.startswith((".staging-", ".backup-", ".tmp-")):
            qdir.mkdir(parents=True, exist_ok=True)
            dest = qdir / child.name
            if dest.exists():
                dest = qdir / f"{child.name}-{os.getpid()}"
            os.replace(str(child), str(dest))
            quarantined.append(child.name)
    if quarantined:
        _sync_dir(qdir)
        _sync_dir(bdir)
    return quarantined


def resolve_published_generation(
    bundle_dir: Path | str,
    *,
    expected_source_commit: str | None = None,
    max_result_zip_bytes: int = DEFAULT_RESULT_ZIP_MAX_BYTES,
    verify: bool = True,
) -> dict[str, Any]:
    """Resolve and optionally verify the currently committed generation pointed to by `CURRENT.json`."""
    bdir = Path(bundle_dir)
    ptr_path = bdir / BUNDLE_POINTER_NAME
    if not ptr_path.is_file():
        raise ValueError(f"missing bundle pointer: {ptr_path}")
    ptr = json.loads(ptr_path.read_text(encoding="utf-8"))
    if ptr.get("schema") != BUNDLE_POINTER_SCHEMA:
        raise ValueError(f"unsupported bundle pointer schema: {ptr.get('schema')!r}")
    rel = str(ptr.get("generation_relpath", ""))
    posix_rel = PurePosixPath(rel.replace("\\", "/"))
    if not rel or posix_rel.is_absolute() or ".." in posix_rel.parts:
        raise ValueError(f"unsafe generation_relpath: {rel!r}")
    gen_dir = (bdir / posix_rel).resolve()
    bdir_resolved = bdir.resolve()
    if not gen_dir.is_dir() or bdir_resolved not in gen_dir.parents:
        raise ValueError(f"published generation directory missing or escapes root: {gen_dir}")
    result: dict[str, Any] = {
        **ptr,
        "bundle_dir": str(bdir),
        "generation_dir": str(gen_dir),
    }
    if verify:
        verified = verify_result_bundle(
            gen_dir,
            expected_source_commit=expected_source_commit,
            max_result_zip_bytes=max_result_zip_bytes,
        )
        if (
            ptr.get("result_manifest_sha256") != verified["result_manifest_sha256"]
            or ptr.get("full_manifest_sha256") != verified["full_manifest_sha256"]
        ):
            raise ValueError("CURRENT.json manifest digest disagrees with published generation")
        result["verification"] = verified
    return result


def publish_bundle_generation(
    bundle_dir: Path | str,
    *,
    campaign_id: str,
    source_commit: str,
    status: str,
    experiment_bundles: Mapping[str, Mapping[str, Any]],
    checkpoint_root: Path | str | None = None,
    max_result_zip_bytes: int = DEFAULT_RESULT_ZIP_MAX_BYTES,
    extra_root_files: Mapping[str, bytes] | None = None,
    generation_id: str | None = None,
    export_fn: Callable[..., Mapping[str, Any]] | None = None,
    verify_fn: Callable[..., Mapping[str, Any]] | None = None,
    enable_fault_injection: bool = True,
) -> dict[str, Any]:
    """Stage, verify, commit `generations/<gen_id>`, and atomically flip `CURRENT.json`."""
    bdir = Path(bundle_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    quarantined = quarantine_partial_generations(bdir)

    gens_root = bdir / GENERATIONS_DIR_NAME
    if generation_id is None:
        existing_nums: list[int] = []
        if gens_root.is_dir():
            for d in gens_root.iterdir():
                if d.is_dir() and d.name.startswith("gen-"):
                    try:
                        existing_nums.append(int(d.name.split("-", 1)[1]))
                    except ValueError:
                        pass
        next_num = (max(existing_nums) + 1) if existing_nums else 1
        gen_id = f"gen-{next_num:04d}"

    staging = bdir / f".staging-{gen_id}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    _export = export_fn or export_result_bundle
    _verify = verify_fn or verify_result_bundle
    receipt = dict(
        _export(
            staging,
            campaign_id=campaign_id,
            source_commit=source_commit,
            status=status,
            experiment_bundles=experiment_bundles,
            checkpoint_root=checkpoint_root,
            max_result_zip_bytes=max_result_zip_bytes,
            extra_root_files=extra_root_files,
        )
    )
    _sync_dir(staging)
    if enable_fault_injection:
        trigger_fault_point("publication_stage")

    gens_root.mkdir(parents=True, exist_ok=True)
    gen_dir = gens_root / gen_id
    if gen_dir.exists():
        raise FileExistsError(f"generation directory already exists: {gen_dir}")
    os.replace(str(staging), str(gen_dir))
    _sync_dir(gens_root)

    try:
        verified = dict(
            _verify(
                gen_dir,
                expected_source_commit=source_commit,
                max_result_zip_bytes=max_result_zip_bytes,
            )
        )
    except BaseException:
        qdir = bdir / QUARANTINE_DIR_NAME
        qdir.mkdir(parents=True, exist_ok=True)
        qdest = qdir / f"unverified-{gen_id}-{os.getpid()}"
        if gen_dir.exists():
            os.replace(str(gen_dir), str(qdest))
        raise
    if enable_fault_injection:
        trigger_fault_point("publication_pointer")

    pointer_doc = {
        "schema": BUNDLE_POINTER_SCHEMA,
        "campaign_id": campaign_id,
        "source_commit": source_commit,
        "current_generation": gen_id,
        "generation_relpath": f"{GENERATIONS_DIR_NAME}/{gen_id}",
        "status": str(verified["status"]),
        "result_manifest_sha256": str(verified["result_manifest_sha256"]),
        "full_manifest_sha256": str(verified["full_manifest_sha256"]),
        "result_zip_bytes": int(receipt["result_zip_bytes"]),
        "checkpoint_full_zip_bytes": int(receipt["checkpoint_full_zip_bytes"]),
    }
    pointer_bytes = (
        json.dumps(_sanitize_json(pointer_doc), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(bdir / BUNDLE_POINTER_NAME, pointer_bytes)

    # Atomically refresh backward-compatible top-level bundle files after generation pointer commit
    for entry in sorted(gen_dir.iterdir()):
        if entry.is_file():
            _atomic_copy_file(entry, bdir / entry.name)
    _sync_dir(bdir)

    return {
        **receipt,
        **verified,
        "output_dir": str(bdir),
        "generation_id": gen_id,
        "generation_dir": str(gen_dir),
        "generation_relpath": f"{GENERATIONS_DIR_NAME}/{gen_id}",
        "pointer_path": str(bdir / BUNDLE_POINTER_NAME),
        "result_zip": str(bdir / RESULT_ZIP_NAME),
        "checkpoint_full_zip": str(bdir / FULL_CHECKPOINT_ZIP_NAME),
        "quarantined_staging": quarantined,
    }


RAW_EVIDENCE_STREAM_SCHEMA = "anra-planthir-raw-evidence-stream-v1"
RAW_EVIDENCE_MANIFEST_NAME = "RAW_EVIDENCE_MANIFEST.json"


def _flush_raw_chunk(
    out_dir: Path,
    *,
    chunk_idx: int,
    jsonl_lines: list[bytes],
    arr_cols: dict[str, list[np.ndarray]],
    total_bytes_so_far: int,
    max_total_bytes: int,
) -> dict[str, Any]:
    from .resources import ResourceQuotaExceededError

    chunk_stem = f"chunk-{chunk_idx:04d}"
    jsonl_path = out_dir / f"{chunk_stem}.jsonl"
    jsonl_bytes = b"\n".join(jsonl_lines) + (b"\n" if jsonl_lines else b"")
    npz_path: Path | None = None
    npz_bytes = 0

    try:
        jsonl_path.write_bytes(jsonl_bytes)
        if arr_cols:
            npz_path = out_dir / f"{chunk_stem}.npz"
            stacked = {k: np.stack(v, axis=0) for k, v in arr_cols.items()}
            if any(arr.dtype.hasobject for arr in stacked.values()):
                raise ValueError("object arrays rejected in raw evidence chunk stream")
            np.savez_compressed(npz_path, **stacked)
            npz_bytes = int(npz_path.stat().st_size)

        chunk_bytes = len(jsonl_bytes) + npz_bytes
        if total_bytes_so_far + chunk_bytes > max_total_bytes:
            raise ResourceQuotaExceededError(
                quota_name="max_raw_evidence_bytes",
                limit=max_total_bytes,
                observed=total_bytes_so_far + chunk_bytes,
                phase="raw_evidence_streaming",
            )

        j_sha, j_size = _file_sha256(jsonl_path)
        entry: dict[str, Any] = {
            "chunk_id": chunk_stem,
            "record_count": len(jsonl_lines),
            "jsonl_file": jsonl_path.name,
            "jsonl_bytes": j_size,
            "jsonl_sha256": j_sha,
            "npz_file": None,
            "npz_bytes": 0,
            "npz_sha256": None,
            "array_shapes": {},
        }
        if npz_path is not None:
            n_sha, n_size = _file_sha256(npz_path)
            entry.update(
                {
                    "npz_file": npz_path.name,
                    "npz_bytes": n_size,
                    "npz_sha256": n_sha,
                    "array_shapes": {k: list(np.stack(v, axis=0).shape) for k, v in arr_cols.items()},
                }
            )
        return entry
    except BaseException:
        jsonl_path.unlink(missing_ok=True)
        if npz_path is not None:
            npz_path.unlink(missing_ok=True)
        raise


def stream_raw_evidence_chunks(
    output_dir: Path | str,
    records_iter: Iterable[Mapping[str, Any]],
    *,
    max_chunk_records: int = 64,
    max_chunk_bytes: int = 512 * 1024,
    max_total_bytes: int = 10 * 1024 * 1024,
) -> dict[str, Any]:
    """Stream bounded JSONL/NPZ raw evidence chunks to disk while enforcing write-time byte quotas."""

    if type(max_chunk_records) is not int or max_chunk_records < 1:
        raise ValueError("max_chunk_records must be positive")
    if type(max_chunk_bytes) is not int or max_chunk_bytes < 1:
        raise ValueError("max_chunk_bytes must be positive")
    if type(max_total_bytes) is not int or max_total_bytes < 1:
        raise ValueError("max_total_bytes must be positive")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_meta: list[dict[str, Any]] = []
    jsonl_buf: list[bytes] = []
    arr_buf: dict[str, list[np.ndarray]] = {}
    buf_approx_bytes = 0
    total_bytes = 0
    total_records = 0
    chunk_idx = 0

    for rec in records_iter:
        arrays_field = rec.get("arrays")
        row_payload = {str(k): v for k, v in rec.items() if k != "arrays"}
        line_bytes = json.dumps(_sanitize_json(row_payload), sort_keys=True, allow_nan=False).encode("utf-8")
        jsonl_buf.append(line_bytes)
        buf_approx_bytes += len(line_bytes) + 1

        if isinstance(arrays_field, Mapping):
            for ak, av in arrays_field.items():
                np_arr = av.detach().cpu().numpy() if isinstance(av, torch.Tensor) else np.asarray(av)
                arr_buf.setdefault(str(ak), []).append(np_arr)
                buf_approx_bytes += int(np_arr.nbytes)

        total_records += 1
        if len(jsonl_buf) >= max_chunk_records or buf_approx_bytes >= max_chunk_bytes:
            chunk_idx += 1
            meta = _flush_raw_chunk(
                out_dir,
                chunk_idx=chunk_idx,
                jsonl_lines=jsonl_buf,
                arr_cols=arr_buf,
                total_bytes_so_far=total_bytes,
                max_total_bytes=max_total_bytes,
            )
            chunks_meta.append(meta)
            total_bytes += int(meta["jsonl_bytes"]) + int(meta["npz_bytes"])
            jsonl_buf.clear()
            arr_buf.clear()
            buf_approx_bytes = 0

    if jsonl_buf:
        chunk_idx += 1
        meta = _flush_raw_chunk(
            out_dir,
            chunk_idx=chunk_idx,
            jsonl_lines=jsonl_buf,
            arr_cols=arr_buf,
            total_bytes_so_far=total_bytes,
            max_total_bytes=max_total_bytes,
        )
        chunks_meta.append(meta)
        total_bytes += int(meta["jsonl_bytes"]) + int(meta["npz_bytes"])
        jsonl_buf.clear()
        arr_buf.clear()

    manifest_body = {
        "schema": RAW_EVIDENCE_STREAM_SCHEMA,
        "max_chunk_records": max_chunk_records,
        "max_chunk_bytes": max_chunk_bytes,
        "max_total_bytes": max_total_bytes,
        "total_records": total_records,
        "total_bytes": total_bytes,
        "chunk_count": len(chunks_meta),
        "chunks": chunks_meta,
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": digest(manifest_body),
    }
    manifest_bytes = (
        json.dumps(_sanitize_json(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(out_dir / RAW_EVIDENCE_MANIFEST_NAME, manifest_bytes)
    _sync_dir(out_dir)
    return manifest


def verify_raw_evidence_chunks(output_dir: Path | str) -> dict[str, Any]:
    """Fail-closed SHA-256, quota, and row-count verification of chunked raw evidence."""

    out_dir = Path(output_dir)
    manifest_path = out_dir / RAW_EVIDENCE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"missing {RAW_EVIDENCE_MANIFEST_NAME} in {out_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RAW_EVIDENCE_STREAM_SCHEMA:
        raise ValueError("unsupported raw evidence manifest schema")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if digest(body) != manifest.get("manifest_sha256"):
        raise ValueError("raw evidence manifest_sha256 mismatch")

    max_total = int(manifest["max_total_bytes"])
    total_bytes = 0
    total_records = 0
    for chunk in manifest.get("chunks", []):
        jpath = out_dir / str(chunk["jsonl_file"])
        if not jpath.is_file():
            raise ValueError(f"missing raw evidence chunk jsonl: {jpath.name}")
        j_sha, j_size = _file_sha256(jpath)
        if j_size != int(chunk["jsonl_bytes"]) or j_sha != str(chunk["jsonl_sha256"]):
            raise ValueError(f"checksum mismatch in raw evidence chunk: {jpath.name}")
        raw_lines = [ln for ln in jpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(raw_lines) != int(chunk["record_count"]):
            raise ValueError(f"record count mismatch in raw evidence chunk: {jpath.name}")
        for ln in raw_lines:
            json.loads(ln)
        total_bytes += j_size
        total_records += len(raw_lines)

        npz_name = chunk.get("npz_file")
        if npz_name:
            npath = out_dir / str(npz_name)
            if not npath.is_file():
                raise ValueError(f"missing raw evidence chunk npz: {npz_name}")
            n_sha, n_size = _file_sha256(npath)
            if n_size != int(chunk["npz_bytes"]) or n_sha != str(chunk["npz_sha256"]):
                raise ValueError(f"checksum mismatch in raw evidence chunk: {npz_name}")
            with np.load(npath, allow_pickle=False) as archive:
                for k, arr in archive.items():
                    if not np.isfinite(arr).all():
                        raise ValueError(f"nonfinite array in raw evidence chunk {npz_name}:{k}")
            total_bytes += n_size

    if total_bytes != int(manifest["total_bytes"]) or total_bytes > max_total:
        raise ValueError("raw evidence total_bytes mismatch or quota overflow")
    if total_records != int(manifest["total_records"]):
        raise ValueError("raw evidence total_records mismatch")

    return {
        "verified": True,
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "total_records": total_records,
        "total_bytes": total_bytes,
        "chunk_count": len(manifest.get("chunks", [])),
    }


__all__ = [
    "BUNDLE_POINTER_NAME",
    "BUNDLE_POINTER_SCHEMA",
    "DEFAULT_RESULT_ZIP_MAX_BYTES",
    "FULL_CHECKPOINT_ZIP_NAME",
    "GENERATIONS_DIR_NAME",
    "QUARANTINE_DIR_NAME",
    "RAW_EVIDENCE_MANIFEST_NAME",
    "RAW_EVIDENCE_STREAM_SCHEMA",
    "RESULT_ZIP_NAME",
    "TIER1_MAX_BYTES",
    "TIER2_MAX_BYTES",
    "bind_tier3_recovery_archive",
    "export_result_bundle",
    "package_three_tier_evidence",
    "probe_from_capsule",
    "publish_bundle_generation",
    "quarantine_partial_generations",
    "resolve_published_generation",
    "stream_raw_evidence_chunks",
    "verify_raw_evidence_chunks",
    "verify_result_bundle",
    "verify_single_planthir_zip",
    "verify_tier2_capsule",
    "write_tier1_summary",
    "write_tier2_diagnostic_capsule",
]
