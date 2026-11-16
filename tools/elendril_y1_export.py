"""AN-RA V8-Y1 three-bundle verified export, handoff, and fail-closed restore."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4
import zipfile

from tools import elendril_v6_export as v6
from tools.elendril_v7_release import verify_checkpoint_members
from v5_training.checkpoint import REQUIRED_COMPONENTS
from v5_training.checkpoint_lock import checkpoint_writer


Y1_ARCHIVE_NAMES = {
    "summary": "ELENDRIL_V8_Y1_SUMMARY.zip",
    "evidence": "ELENDRIL_V8_Y1_EVIDENCE.zip",
    "full": "ELENDRIL_V8_Y1_FULL.zip",
}

SUMMARY_MAX_BYTES = 5 * 1024 * 1024  # Target < 5 MiB enforced.
ALLOWED_Y1_HANDOFFS = frozenset({
    ("prepare", "diagnose"),
    ("diagnose", "core"),
    ("core", "followup"),
    ("core", "finalize"),
    ("followup", "finalize"),
})


def _atomic_zip_from_files(
    root: Path,
    rel_paths: list[str],
    *,
    dest_zip: Path,
    source: str,
    protocol_sha: str,
    campaign: dict[str, Any],
    rescore_payload: dict[str, Any],
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic verified ZIP atomically from explicit root-relative regular files."""
    if dest_zip.is_symlink():
        raise ValueError(f"refuse symlink export destination: {dest_zip}")
    root_res = root.resolve(strict=True)
    records = []
    ordered_rels = sorted(
        set(rel_paths) - {v6.MANIFEST_NAME, "README_RESULTS.md", "SHA256SUMS"}
    )
    for rel in ordered_rels:
        rel_clean = v6._member_name(rel.replace("\\", "/"))
        fpath = (root_res / rel_clean)
        if fpath.is_symlink() or not fpath.is_file():
            raise ValueError(f"missing or linked export member: {rel_clean}")
        if fpath.resolve(strict=True) != fpath and root_res not in fpath.resolve(strict=True).parents:
            raise ValueError(f"escape outside export root: {rel_clean}")
        size, fdigest = v6._file_sha(fpath)
        records.append({"name": rel_clean, "byte_size": size, "sha256": fdigest})

    readme_bytes = (
        f"# AN-RA V8-Y1 Evidence Archive ({campaign.get('phase', 'y1')})\n\n"
        f"- Source: `{source}`\n"
        f"- Protocol: `{protocol_sha}`\n"
        f"- Campaign status: `{campaign.get('status', 'UNKNOWN')}`\n"
        f"- Rescore status: `{rescore_payload.get('status', 'UNKNOWN')}`\n"
    ).encode("utf-8")
    records.append({"name": "README_RESULTS.md", "byte_size": len(readme_bytes), "sha256": sha256(readme_bytes).hexdigest()})
    sums_text = "".join(f"{r['sha256']}  {r['name']}\n" for r in sorted(records, key=lambda row: row["name"]))
    sums_bytes = sums_text.encode("utf-8")
    all_records = sorted(
        [*records, {"name": "SHA256SUMS", "byte_size": len(sums_bytes), "sha256": sha256(sums_bytes).hexdigest()}],
        key=lambda row: row["name"],
    )

    manifest = {
        "schema": v6.SCHEMA,
        "source": v6._sha(source, 40),
        "protocol_sha256": v6._sha(protocol_sha, 64),
        "campaign_status": str(campaign.get("status", "UNKNOWN")),
        "rescore": rescore_payload,
        "campaign": campaign,
        "files": all_records,
    }
    manifest_bytes = v6.canonical(manifest)
    manifest_sha = sha256(manifest_bytes).hexdigest()

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = dest_zip.parent / f".pending-{dest_zip.stem}-{uuid4().hex}.zip"
    try:
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(v6.MANIFEST_NAME, manifest_bytes)
            zf.writestr("README_RESULTS.md", readme_bytes)
            zf.writestr("SHA256SUMS", sums_bytes)
            for row in records:
                if row["name"] == "README_RESULTS.md":
                    continue
                zf.write(root_res / row["name"], arcname=row["name"])
        zip_size, zip_sha = v6._file_sha(tmp_zip)
        if max_bytes is not None and zip_size > max_bytes:
            raise ValueError(f"{dest_zip.name} exceeds budget {max_bytes} bytes: {zip_size} bytes")
        checked = v6.verify_zip(tmp_zip)
        if checked["manifest_sha256"] != manifest_sha:
            raise ValueError(f"{dest_zip.name} failed manifest readback verification")
        os.replace(tmp_zip, dest_zip)
        final_size, final_sha = v6._file_sha(dest_zip)
        final_checked = v6.verify_zip(dest_zip)
        return {
            "status": "EXPORT_VERIFIED",
            "zip_path": str(dest_zip),
            "zip_sha256": final_sha,
            "zip_bytes": final_size,
            "expanded_bytes": sum(r["byte_size"] for r in records),
            "member_count": len(records),
            "manifest_sha256": final_checked["manifest_sha256"],
        }
    finally:
        tmp_zip.unlink(missing_ok=True)


def export_y1_bundles(
    output: Path | str,
    *,
    source: str,
    protocol_sha: str,
    campaign: dict[str, Any],
    replay_payload: dict[str, Any],
    include_full: bool = True,
) -> dict[str, Any]:
    """Emit SUMMARY (<5 MiB), EVIDENCE, and FULL ZIPs cleanly, then write FINAL_STATUS.json outside."""
    root = Path(output)
    if root.is_symlink():
        raise ValueError("linked output directory")
    root = root.resolve(strict=True)
    exports = root / "exports"
    if exports.is_symlink():
        raise ValueError("linked exports directory")
    exports.mkdir(exist_ok=True)

    summary_rels: list[str] = []
    evidence_rels: list[str] = []
    checkpoint_rels: list[str] = []

    def add_rel(bucket: list[str], path: Path) -> None:
        if path.is_file() and not path.is_symlink():
            bucket.append(path.relative_to(root).as_posix())

    # Collect summary files (no weights, no per-case logits, no repeated prompt dumps).
    summary_dir = root / "summary"
    if summary_dir.is_dir():
        for p in sorted(summary_dir.rglob("*")):
            if p.is_file():
                add_rel(summary_rels, p)
    for rel_top in (
        "analysis/identity.json",
        "analysis/profile_receipt.json",
        "analysis/forecast_receipt.json",
        "analysis/custody_audit.json",
        "analysis/d0_d1_summary.json",
        "analysis/d2_probe_summary.json",
        "analysis/budget_ledger.json",
        f"analysis/campaign-{campaign['run_id']}.json",
    ):
        add_rel(summary_rels, root / rel_top)
    for sub in ("analysis/gates", "analysis/heuristics", "analysis/jobs"):
        sdir = root / sub
        if sdir.is_dir():
            for p in sorted(sdir.rglob("*.json")):
                # Keep job_spec, run_receipt, metrics_summary (never dense raw prediction lists).
                if p.name in ("job_spec.json", "run_receipt.json", "metrics_summary.json", "t0_gate.json") or sub != "analysis/jobs":
                    add_rel(summary_rels, p)

    # Evidence files = all summary files + deduplicated prepared/ + predictions/ + logits/ + update_logs/.
    evidence_rels.extend(summary_rels)
    for ed_dir in ("prepared", "evidence"):
        pdir = root / ed_dir
        if pdir.is_dir():
            for p in sorted(pdir.rglob("*")):
                if p.is_file():
                    add_rel(evidence_rels, p)
    if (root / "analysis" / "jobs").is_dir():
        for p in sorted((root / "analysis" / "jobs").rglob("*")):
            if p.is_file():
                add_rel(evidence_rels, p)

    # Full files = evidence files + checkpoint objects (never bundling external historical zips).
    cp_dir = root / "checkpoints"
    required_manifest_components: list[str] = []
    if include_full and cp_dir.is_dir():
        for p in sorted(cp_dir.rglob("*")):
            if p.is_file() and ".." not in p.parts:
                rel = p.relative_to(root).as_posix()
                checkpoint_rels.append(rel)
                if rel.endswith("/manifest.json"):
                    prefix = rel.rsplit("/", 1)[0]
                    required_manifest_components.append(rel)
                    required_manifest_components.extend(prefix + "/" + part for part in sorted(REQUIRED_COMPONENTS))

    results: dict[str, Any] = {}
    with checkpoint_writer(root, filename=".pending-y1-export-writer"):
        for kind, rels, limit in (
            ("summary", summary_rels, SUMMARY_MAX_BYTES),
            ("evidence", evidence_rels, None),
            ("full", [*evidence_rels, *checkpoint_rels], None),
        ):
            if kind == "full" and not include_full:
                results[kind] = {"status": "SKIPPED_NOT_REQUESTED"}
                continue
            dest = exports / Y1_ARCHIVE_NAMES[kind]
            try:
                rep = _atomic_zip_from_files(
                    root,
                    rels,
                    dest_zip=dest,
                    source=source,
                    protocol_sha=protocol_sha,
                    campaign=campaign,
                    rescore_payload=replay_payload,
                    max_bytes=limit,
                )
                if kind == "full" and required_manifest_components:
                    with zipfile.ZipFile(dest, "r") as zf:
                        verify_checkpoint_members(zf, required_manifest_components)
                results[kind] = rep
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                results[kind] = {
                    "status": "EXPORT_FAILED",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
    return results


def finalize_y1_campaign(
    output: Path | str,
    *,
    source: str,
    protocol_sha: str,
    campaign: dict[str, Any],
    replay_payload: dict[str, Any],
    include_full: bool = True,
) -> dict[str, Any]:
    """Write identity + campaign marker, emit verified archives, and publish outside FINAL_STATUS.json."""
    root = Path(output).resolve(strict=True)
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)

    v6._write_derived(
        analysis / "identity.json",
        v6.canonical(
            {
                "schema": "anra-v8-y1-archive-identity-v1",
                "source": source,
                "protocol_sha256": protocol_sha,
                "phase": campaign["phase"],
                "run_id": campaign["run_id"],
            }
        )
        + b"\n",
    )
    pre_export = {
        **campaign,
        "independent_replay": replay_payload,
    }
    marker_path = analysis / f"campaign-{campaign['run_id']}.json"
    if not marker_path.exists():
        v6.immutable_json(marker_path, pre_export)
    else:
        v6._write_derived(marker_path, v6.canonical(pre_export) + b"\n")

    exports = export_y1_bundles(
        root,
        source=source,
        protocol_sha=protocol_sha,
        campaign=pre_export,
        replay_payload=replay_payload,
        include_full=include_full,
    )
    required_kinds = ("summary", "evidence", "full") if include_full else ("summary", "evidence")
    delivery_pass = all(exports.get(k, {}).get("status") == "EXPORT_VERIFIED" for k in required_kinds)
    final_status = {
        "schema": "anra-v8-y1-final-status-v1",
        "run_id": campaign["run_id"],
        "phase": campaign["phase"],
        "source": source,
        "protocol_sha256": protocol_sha,
        "status": campaign["status"],
        "scientific_status": campaign.get("scientific_status", "UNEXECUTED"),
        "claim_status": campaign.get("claim_status", "IMPLEMENTED"),
        "delivery_status": "PASS" if delivery_pass else "FAILED",
        "custody_notice": "Local /kaggle/working storage is temporary; save notebook output version or verified durable external copy for durable custody.",
        "exports": exports,
        "independent_replay": replay_payload,
        "errors": campaign.get("errors", []),
    }
    v6._write_derived(root / "exports" / "FINAL_STATUS.json", v6.canonical(final_status) + b"\n")
    return final_status


def inspect_y1_archive_identity(zip_path: Path | str) -> dict[str, Any]:
    """Verify ZIP integrity and return embedded Y1 archive identity."""
    zp = Path(zip_path)
    if zp.is_symlink() or not zp.is_file():
        raise ValueError(f"archive must be an existing regular file: {zp}")
    checked = v6.verify_zip(zp)
    with zipfile.ZipFile(zp, "r") as zf:
        try:
            ident = json.loads(zf.read("analysis/identity.json").decode("utf-8"))
        except KeyError as exc:
            raise ValueError(f"{zp} missing analysis/identity.json") from exc
    return {"zip_check": checked, "identity": ident}


def handoff_y1_archive(
    archive_path: Path | str,
    dest_dir: Path | str,
    *,
    source: str,
    protocol_sha: str,
    from_phase: str,
    to_phase: str,
) -> dict[str, Any]:
    """Selectively verify & unpack non-checkpoint receipts/prepared inputs (+ needed checkpoints when FULL) across Y1 phases."""
    if (from_phase, to_phase) not in ALLOWED_Y1_HANDOFFS:
        raise ValueError(f"disallowed Y1 cross-phase handoff: {from_phase} -> {to_phase}")
    info = inspect_y1_archive_identity(archive_path)
    ident = info["identity"]
    if ident.get("source") != source or ident.get("protocol_sha256") != protocol_sha or ident.get("phase") != from_phase:
        raise ValueError(
            f"handoff identity mismatch: expected ({source}, {protocol_sha}, {from_phase}), got {ident}"
        )

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    transferred: list[str] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        outer = json.loads(zf.read(v6.MANIFEST_NAME).decode("utf-8"))
        file_rows = {r["name"]: r for r in outer["files"]}
        # Selective prefixes: prepared inputs, gates/heuristics/custody/d0_d1 receipts, and (for followup) checkpoints.
        for name, row in sorted(file_rows.items()):
            safe = v6._member_name(name)
            keep = (
                safe.startswith("prepared/")
                or safe.startswith("analysis/gates/")
                or safe.startswith("analysis/heuristics/")
                or safe in ("analysis/custody_audit.json", "analysis/d0_d1_summary.json", "analysis/budget_ledger.json")
                or (to_phase == "followup" and (safe.startswith("checkpoints/") or safe.startswith("analysis/jobs/")))
            )
            if not keep:
                continue
            out = dest / safe
            out.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(safe)
            if sha256(data).hexdigest() != row["sha256"] or len(data) != row["byte_size"]:
                raise ValueError(f"handoff member corrupted during readback: {safe}")
            out.write_bytes(data)
            transferred.append(safe)
    receipt = {
        "schema": "anra-v8-y1-handoff-receipt-v1",
        "from_phase": from_phase,
        "to_phase": to_phase,
        "source": source,
        "protocol_sha256": protocol_sha,
        "source_archive": str(Path(archive_path).resolve()),
        "transferred_count": len(transferred),
    }
    v6._write_derived(dest / "analysis" / "handoff_receipt.json", v6.canonical(receipt) + b"\n")
    return receipt


def restore_y1_archive(
    archive_path: Path | str,
    dest_dir: Path | str,
    *,
    source: str,
    protocol_sha: str,
    phase: str,
) -> dict[str, Any]:
    """Fail-closed same-phase recovery restore of ELENDRIL_V8_Y1_FULL.zip preserving cumulative budget ledger."""
    info = inspect_y1_archive_identity(archive_path)
    ident = info["identity"]
    if ident.get("source") != source or ident.get("protocol_sha256") != protocol_sha or ident.get("phase") != phase:
        raise ValueError(
            f"resume archive mismatch: expected ({source}, {protocol_sha}, {phase}), got {ident}"
        )
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    restored = 0
    with zipfile.ZipFile(archive_path, "r") as zf:
        outer = json.loads(zf.read(v6.MANIFEST_NAME).decode("utf-8"))
        for row in outer["files"]:
            safe = v6._member_name(row["name"])
            if safe in ("README_RESULTS.md", "SHA256SUMS"):
                continue
            out = dest / safe
            out.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(safe)
            if sha256(data).hexdigest() != row["sha256"] or len(data) != row["byte_size"]:
                raise ValueError(f"resume archive member failed SHA verification: {safe}")
            out.write_bytes(data)
            restored += 1
    return {
        "schema": "anra-v8-y1-restore-receipt-v1",
        "phase": phase,
        "source": source,
        "protocol_sha256": protocol_sha,
        "restored_files": restored,
    }
