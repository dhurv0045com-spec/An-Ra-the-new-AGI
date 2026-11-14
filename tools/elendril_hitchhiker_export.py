"""Hitchhiker dual archive export, verified finalize, and fail-closed restore."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4
import zipfile

from tools import elendril_v6_export as v6
from tools import elendril_v7_export as v7
from tools.elendril_v71_export import MAX_EXTRACT_BYTES, extract_verified
from tools.elendril_v7_release import verify_checkpoint_members
from v5_training.checkpoint import REQUIRED_COMPONENTS
from v5_training.checkpoint_lock import checkpoint_writer


HITCHHIKER_ARCHIVE_NAMES = {
    "analysis": "ELENDRIL_HITCHHIKER_ANALYSIS.zip",
    "full": "ELENDRIL_HITCHHIKER_FULL.zip",
}

ALLOWED_HANDOFFS = frozenset({("stage1", "stage2"), ("stage1", "stage3"), ("stage2", "stage3")})
_PHASE_TO_STAGE = {
    "stage0": "stage0_checkpoint_audit",
    "stage1": "stage1_s3_screen",
    "stage2": "stage2_followups",
    "stage3": "stage3_scale_confirmation",
}


def _checkpoint_members(values: list[str] | tuple[str, ...]) -> list[str]:
    required: set[str] = set()
    for value in values:
        name = v6._member_name(str(value).replace("\\", "/"))
        if not name.endswith("/manifest.json"):
            name += "/manifest.json"
        if not name.startswith("checkpoints/") or not re.fullmatch(r"[0-9a-f]{64}", name.split("/")[-2]):
            raise ValueError(f"checkpoint must name an output-relative content-addressed manifest: {value!r}")
        prefix = name.rsplit("/", 1)[0]
        required.add(name)
        required.update(prefix + "/" + part for part in REQUIRED_COMPONENTS)
    return sorted(required)


def export_hitchhiker_results(
    output: Path | str,
    *,
    source: str,
    protocol_sha: str,
    campaign: dict,
    full: bool = True,
    rescore=None,
) -> dict:
    """Serialize Hitchhiker analysis & full bundles reusing V7 verified streaming transport."""
    root = Path(output)
    if root.is_symlink():
        raise ValueError("linked output directory")
    root = root.resolve(strict=True)
    analysis = root / "analysis"
    if analysis.is_symlink():
        raise ValueError("linked analysis directory")
    analysis.mkdir(exist_ok=True)
    v6._write_derived(
        analysis / "README_HITCHHIKER.md",
        (
            "# Elendril Hitchhiker evidence\n\n"
            "Archive verification certifies byte/SHA integrity, not scientific capability.\n"
            "Inspect screening_gate, formation_gate, selection_receipt, and independent_replay receipts.\n"
        ).encode("utf-8"),
    )
    with checkpoint_writer(root, filename=".pending-hitchhiker-export-writer"):
        transport = v7.export_results(
            root,
            source=source,
            protocol_sha=protocol_sha,
            campaign=campaign,
            full=full,
            rescore=rescore,
        )
        results: dict = {}
        for kind in ("analysis", "full"):
            report = transport[kind]
            if report.get("status") != "EXPORT_VERIFIED":
                results[kind] = report
                continue
            try:
                src_path = Path(report["zip_path"])
                if src_path.is_symlink():
                    raise ValueError("linked transport archive")
                size, digest = v6._file_sha(src_path)
                checked = v6.verify_zip(src_path)
                if (
                    size != report["zip_bytes"]
                    or digest != report["zip_sha256"]
                    or checked["manifest_sha256"] != report["manifest_sha256"]
                ):
                    raise ValueError("Hitchhiker transport archive failed verification")
                dest = root / "exports" / HITCHHIKER_ARCHIVE_NAMES[kind]
                os.replace(src_path, dest)
                after = v6.verify_zip(dest)
                if v6._file_sha(dest) != (size, digest) or after["manifest_sha256"] != checked["manifest_sha256"]:
                    raise ValueError("published Hitchhiker archive failed readback")
                results[kind] = {**report, "zip_path": str(dest), "zip_sha256": digest, "zip_bytes": size}
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                results[kind] = {
                    "status": "EXPORT_DEFERRED",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "raw_evidence": str(root),
                    "scientific_rescore": report.get("scientific_rescore"),
                }
        return results


def finalize_hitchhiker(
    output: Path | str,
    *,
    source: str,
    protocol_sha: str,
    campaign: dict,
    required_jobs: list[str] | tuple[str, ...],
    required_checkpoints: list[str] | tuple[str, ...],
    complete: bool,
) -> dict:
    """Publish current-run identity & marker, verify both Hitchhiker archives, and emit FINAL_STATUS.json."""
    v6._sha(source, 40)
    v6._sha(protocol_sha, 64)
    requested = Path(output)
    if requested.is_symlink():
        raise ValueError("linked output directory")
    root = requested.resolve(strict=True)
    if campaign.get("source") != source or campaign.get("protocol_sha256") != protocol_sha:
        raise ValueError("campaign source or protocol identity mismatch")
    run_id = campaign.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,90}", run_id) is None:
        raise ValueError("campaign requires a portable run_id")
    jobs = list(required_jobs)
    if len(set(jobs)) != len(jobs) or any(
        not isinstance(j, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,90}", j) is None for j in jobs
    ):
        raise ValueError("invalid or duplicate required job")
    cp_members = _checkpoint_members(required_checkpoints)

    analysis = root / "analysis"
    if analysis.is_symlink():
        raise ValueError("linked analysis directory")
    analysis.mkdir(exist_ok=True)
    v6._write_derived(
        analysis / "identity.json",
        v6.canonical(
            {
                "schema": "elendril-hitchhiker-archive-identity-v1",
                "source": source,
                "protocol_sha256": protocol_sha,
                "phase": campaign["phase"],
            }
        ),
    )

    token = uuid4().hex
    marker = f"campaign-{run_id}-{token}.json"
    eligibility = (
        complete is True
        and campaign.get("status") == "EXECUTED"
        and campaign.get("independent_replay", {}).get("status") == "PASS"
    )
    v6.immutable_json(
        analysis / marker,
        {
            "source": source,
            "protocol_sha256": protocol_sha,
            "run_id": run_id,
            "phase": campaign["phase"],
            "export_token": token,
            "campaign": campaign,
            "required_jobs": jobs,
            "required_checkpoints": cp_members,
            "requested_complete": bool(complete),
        },
    )

    replay_receipt = campaign.get("independent_replay", {"status": "NOT_RUN"})
    exports = export_hitchhiker_results(
        root,
        source=source,
        protocol_sha=protocol_sha,
        campaign=campaign,
        full=True,
        rescore=lambda _root, _campaign: replay_receipt,
    )

    errors: list[str] = []
    verified: dict = {}
    for kind, filename in HITCHHIKER_ARCHIVE_NAMES.items():
        try:
            report = exports[kind]
            path = root / "exports" / filename
            if report.get("status") != "EXPORT_VERIFIED" or path.is_symlink() or not path.is_file():
                raise ValueError(f"mandatory archive {filename} not verified")
            if Path(report["zip_path"]).resolve() != path.resolve():
                raise ValueError(f"wrong archive path for {filename}")
            size, digest = v6._file_sha(path)
            checked = v6.verify_zip(path)
            if (
                digest != report["zip_sha256"]
                or size != report["zip_bytes"]
                or checked["source"] != source
                or checked["protocol_sha256"] != protocol_sha
                or checked["manifest_sha256"] != report["manifest_sha256"]
                or checked["scientific_rescore"] != replay_receipt
            ):
                raise ValueError(f"archive {filename} hash, identity, or replay differs")
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                prefix = "" if kind == "analysis" else "analysis/"
                if prefix + marker not in names:
                    raise ValueError(f"current-run marker absent in {filename}")
                marker_data = json.loads(archive.read(prefix + marker))
                if marker_data.get("export_token") != token:
                    raise ValueError(f"current-run marker token mismatch in {filename}")
                for job in jobs if eligibility else ():
                    base = prefix + "jobs/" + job + "/"
                    if (base + "run_receipt.json") not in names or (base + "final_eval.json") not in names:
                        raise ValueError(f"required job evaluation files absent in {filename}: {job}")
                if kind == "full" and eligibility:
                    if not set(cp_members) <= names:
                        raise ValueError("required checkpoint components absent in FULL archive")
                    verify_checkpoint_members(archive, cp_members)
            verified[kind] = {
                "zip_path": str(path),
                "zip_sha256": digest,
                "manifest_sha256": checked["manifest_sha256"],
            }
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            errors.append(f"{kind}: {exc}")

    missing_required: list[str] = []
    for job in jobs:
        jdir = analysis / "jobs" / job
        if not (jdir / "run_receipt.json").is_file() or not (jdir / "final_eval.json").is_file():
            missing_required.append(f"job:{job}")
    missing_required.extend(name for name in cp_members if not (root / name).is_file())

    status = "EXPORT_FAILED" if errors else "COMPLETE" if eligibility else "PARTIAL"
    result = {
        "schema": "elendril-hitchhiker-delivery-v1",
        "status": status,
        "run_id": run_id,
        "phase": campaign["phase"],
        "source": source,
        "protocol_sha256": protocol_sha,
        "claim_status": campaign.get("claim_status", "IMPLEMENTED"),
        "expected_jobs": len(jobs),
        "independent_replay": replay_receipt,
        "exports": exports,
        "verified": verified,
        "errors": errors,
        "completion_eligible": eligibility,
        "missing_required": missing_required,
        "restore_limit": "Hitchhiker verified extraction is bounded to 16 GiB uncompressed.",
    }
    receipt = analysis / f"delivery-{run_id}-{token}.json"
    v6.immutable_json(receipt, result)
    result["delivery_receipt"] = str(receipt)
    exports_dir = root / "exports"
    exports_dir.mkdir(exist_ok=True)
    v6._write_derived(exports_dir / "FINAL_STATUS.json", v6.canonical(result))
    return result


def restore_hitchhiker_archive(
    zip_path: Path | str,
    destination: Path | str,
    *,
    source: str,
    phase: str,
) -> dict:
    """Verify archive identity and extract a FULL Hitchhiker resume/handoff archive safely."""
    from anra_v5.elendril.hitchhiker_protocol import PROTOCOL_SHA

    PHASES = ("stage0", "stage1", "stage2", "stage3")

    v6._sha(source, 40)
    if phase not in PHASES:
        raise ValueError(f"invalid restore phase {phase!r}; expected {PHASES}")
    requested = Path(zip_path)
    if requested.is_symlink():
        raise ValueError("linked archive input")
    archive_path = requested.resolve(strict=True)
    before = v6._file_sha(archive_path)
    verified = v6.verify_zip(archive_path, max_bytes=MAX_EXTRACT_BYTES)
    if verified["source"] != source or verified["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("archive source or protocol does not match requested restore")

    expected_identity = {
        "schema": "elendril-hitchhiker-archive-identity-v1",
        "source": source,
        "protocol_sha256": PROTOCOL_SHA,
        "phase": phase,
    }
    with zipfile.ZipFile(archive_path) as archive:
        try:
            entry = archive.getinfo("analysis/identity.json")
        except KeyError as exc:
            raise ValueError("restore requires FULL archive with analysis/identity.json") from exc
        if entry.file_size > 65536:
            raise ValueError("archive identity is oversized")
        actual = json.loads(archive.read(entry))
        if actual != expected_identity:
            raise ValueError("archive identity or phase mismatch")

    if v6._file_sha(archive_path) != before:
        raise ValueError("archive changed during restore admission")
    extracted = extract_verified(archive_path, destination)
    if extracted["zip_sha256"] != before[1]:
        raise RuntimeError("archive changed after admission; extracted output is not admitted")
    # Remove prior bundle wrapper manifests from the restored working root so a subsequent
    # export cycle generates a fresh verified manifest without reserved-member collisions.
    restored_root = Path(extracted["verified_root"])
    for rel in (
        "BUNDLE_MANIFEST.json",
        "SHA256SUMS",
        "analysis/BUNDLE_MANIFEST.json",
        "analysis/SHA256SUMS",
    ):
        (restored_root / rel).unlink(missing_ok=True)
    return {
        **extracted,
        "restore_kind": "hitchhiker_resume",
        "requested_phase": phase,
        "archive_identity": expected_identity,
    }


def _copy_verified_file(src: Path, dst: Path) -> None:
    if src.is_symlink() or not src.is_file():
        raise ValueError(f"expected regular non-symlink handoff file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    v6._write_derived(dst, data)
    if v6._file_sha(src) != v6._file_sha(dst):
        raise ValueError(f"handoff file failed readback verification: {dst}")


def handoff_hitchhiker_archive(
    zip_path: Path | str,
    destination: Path | str,
    *,
    source: str,
    from_phase: str,
    to_phase: str,
    winning_arm: str | None = None,
) -> dict:
    """Verify completed prerequisite FULL archive and stage clean cross-phase handoff into destination."""
    from anra_v5.elendril.hitchhiker_protocol import PROTOCOL_SHA, validate_stage_selection

    v6._sha(source, 40)
    if (from_phase, to_phase) not in ALLOWED_HANDOFFS:
        raise ValueError(
            f"disallowed cross-phase handoff {from_phase!r} -> {to_phase!r}; "
            f"allowed: {sorted(ALLOWED_HANDOFFS)}"
        )

    requested = Path(zip_path)
    if requested.is_symlink():
        raise ValueError("linked archive input")
    archive_path = requested.resolve(strict=True)
    before = v6._file_sha(archive_path)
    verified = v6.verify_zip(archive_path, max_bytes=MAX_EXTRACT_BYTES)
    if verified["source"] != source or verified["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("archive source or protocol does not match requested handoff")
    if verified.get("campaign_status") != "EXECUTED":
        raise ValueError(f"handoff requires completed campaign_status == 'EXECUTED', got {verified.get('campaign_status')!r}")
    if (verified.get("scientific_rescore") or {}).get("status") != "PASS":
        raise ValueError("handoff requires archive independent_replay/rescore status == 'PASS'")

    expected_identity = {
        "schema": "elendril-hitchhiker-archive-identity-v1",
        "source": source,
        "protocol_sha256": PROTOCOL_SHA,
        "phase": from_phase,
    }
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        try:
            id_entry = archive.getinfo("analysis/identity.json")
        except KeyError as exc:
            raise ValueError("handoff requires FULL archive with analysis/identity.json") from exc
        if id_entry.file_size > 65536:
            raise ValueError("archive identity is oversized")
        actual_identity = json.loads(archive.read(id_entry))
        if actual_identity != expected_identity:
            raise ValueError(f"archive identity mismatch: expected {expected_identity}, got {actual_identity}")

        # Verify prerequisite completion inside archive via FINAL_STATUS / delivery receipt / campaign marker.
        completion_verified = False
        if "exports/FINAL_STATUS.json" in names:
            fs_doc = json.loads(archive.read("exports/FINAL_STATUS.json"))
            if (
                fs_doc.get("status") != "COMPLETE"
                or fs_doc.get("completion_eligible") is not True
                or (fs_doc.get("independent_replay") or {}).get("status") != "PASS"
            ):
                raise ValueError("archive FINAL_STATUS.json is not completion-eligible COMPLETE/PASS")
            completion_verified = True

        delivery_members = sorted(n for n in names if n.startswith("analysis/delivery-") and n.endswith(".json"))
        if delivery_members:
            matching_complete = False
            for d_name in delivery_members:
                d_doc = json.loads(archive.read(d_name))
                if (
                    d_doc.get("phase") == from_phase
                    and d_doc.get("status") == "COMPLETE"
                    and d_doc.get("completion_eligible") is True
                    and (d_doc.get("independent_replay") or {}).get("status") == "PASS"
                ):
                    matching_complete = True
            if not matching_complete:
                raise ValueError("archive delivery receipt is not completion-eligible COMPLETE/PASS")
            completion_verified = True

        marker_members = sorted(n for n in names if n.startswith("analysis/campaign-") and n.endswith(".json"))
        complete_markers = []
        for m_name in marker_members:
            m_doc = json.loads(archive.read(m_name))
            m_camp = m_doc.get("campaign") or {}
            if (
                m_doc.get("phase") == from_phase
                and m_doc.get("source") == source
                and m_doc.get("protocol_sha256") == PROTOCOL_SHA
                and m_doc.get("requested_complete") is True
                and m_camp.get("status") == "EXECUTED"
                and (m_camp.get("independent_replay") or {}).get("status") == "PASS"
            ):
                for jid in m_doc.get("required_jobs", []):
                    if (
                        f"analysis/jobs/{jid}/run_receipt.json" not in names
                        or f"analysis/jobs/{jid}/final_eval.json" not in names
                    ):
                        raise ValueError(f"handoff archive missing required job receipts for {jid}")
                complete_markers.append(m_doc)
        if not complete_markers and not completion_verified:
            raise ValueError(f"handoff archive lacks verified completion receipt for phase {from_phase!r}")
        if not complete_markers:
            raise ValueError(f"handoff archive lacks current-run completion marker for phase {from_phase!r}")

        if "analysis/selection_receipt.json" not in names:
            raise ValueError("handoff archive missing analysis/selection_receipt.json")
        selection = json.loads(archive.read("analysis/selection_receipt.json"))
        sel_validation = validate_stage_selection(
            _PHASE_TO_STAGE[to_phase],
            selection,
            winning_arm=winning_arm,
        )
        if not any(n.startswith("prepared/") for n in names):
            raise ValueError("handoff archive missing prepared/ dataset artifacts")

    if v6._file_sha(archive_path) != before:
        raise ValueError("archive changed during handoff admission")

    dest_req = Path(destination)
    if dest_req.is_symlink():
        raise ValueError("linked handoff destination")
    if dest_req.exists() and any(dest_req.iterdir()):
        raise FileExistsError(f"handoff destination must be new or empty: {dest_req}")
    dest_req.mkdir(parents=True, exist_ok=True)
    dest_root = dest_req.resolve(strict=True)
    archive_parent = archive_path.parent
    if dest_root == archive_parent or dest_root.is_relative_to(archive_parent) or archive_parent.is_relative_to(dest_root):
        raise ValueError("handoff destination must not overlap archive directory")

    required_handoff = {"analysis/selection_receipt.json"}
    optional_handoff = {"analysis/gate_receipt.json", "analysis/stage2_gate_receipt.json"}
    with zipfile.ZipFile(archive_path) as archive:
        manifest_raw = archive.read(v6.MANIFEST_NAME)
        manifest_doc = json.loads(manifest_raw)
        if sha256(manifest_raw).hexdigest() != verified["manifest_sha256"]:
            raise ValueError("archive manifest digest changed during handoff")
        expected_entries = {row["name"]: row for row in manifest_doc["files"]}
        selected_names = [
            name
            for name in sorted(expected_entries)
            if name.startswith("prepared/") or name in required_handoff or name in optional_handoff
        ]
        if not any(n.startswith("prepared/") for n in selected_names):
            raise ValueError("extracted handoff archive missing prepared/ artifacts")
        if "analysis/selection_receipt.json" not in selected_names:
            raise ValueError("extracted handoff archive missing selection_receipt.json")
        for name in selected_names:
            v6._member_name(name)
            info = archive.getinfo(name)
            row = expected_entries[name]
            target = dest_root.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            cursor = target.parent
            while cursor != dest_root.parent:
                if cursor.is_symlink() or not cursor.resolve().is_relative_to(dest_root):
                    raise ValueError("handoff destination parent linked or escaped")
                cursor = cursor.parent
            if target.is_symlink() or not target.resolve().is_relative_to(dest_root):
                raise ValueError("handoff destination target linked or escaped")
            count, fp = 0, sha256()
            with archive.open(info) as src, target.open("xb") as dst:
                for chunk in iter(lambda: src.read(v6.CHUNK), b""):
                    count += len(chunk)
                    if count > info.file_size:
                        raise ValueError("archive member exceeded declared size")
                    fp.update(chunk)
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            w_size, w_sha = v6._file_sha(target)
            expected_size = row.get("byte_size", info.file_size)
            if count != info.file_size or count != expected_size or w_size != count or fp.hexdigest() != row["sha256"] or w_sha != row["sha256"]:
                raise ValueError(f"handoff member {name} failed integrity readback")
    if v6._file_sha(archive_path) != before:
        raise RuntimeError("archive changed after handoff extraction; output is not admitted")

    effective_winning_arm = winning_arm if winning_arm is not None else selection.get("winning_arm")
    handoff_receipt = {
        "schema": "elendril-hitchhiker-handoff-v1",
        "source": source,
        "protocol_sha256": PROTOCOL_SHA,
        "from_phase": from_phase,
        "to_phase": to_phase,
        "parent_zip_path": str(archive_path),
        "parent_zip_sha256": before[1],
        "parent_manifest_sha256": verified["manifest_sha256"],
        "winning_arm": effective_winning_arm,
        "selection_validation": sel_validation,
    }
    dest_analysis = dest_root / "analysis"
    dest_analysis.mkdir(parents=True, exist_ok=True)
    handoff_receipt_path = dest_analysis / "handoff_receipt.json"
    v6._write_derived(handoff_receipt_path, v6.canonical(handoff_receipt))
    return {
        "status": "HANDOFF_VERIFIED",
        "restore_kind": "hitchhiker_handoff",
        "from_phase": from_phase,
        "to_phase": to_phase,
        "verified_root": str(dest_root),
        "zip_path": str(archive_path),
        "zip_sha256": before[1],
        "manifest_sha256": verified["manifest_sha256"],
        "source": source,
        "protocol_sha256": PROTOCOL_SHA,
        "winning_arm": effective_winning_arm,
        "selection_validation": sel_validation,
        "handoff_receipt_path": str(handoff_receipt_path),
    }
