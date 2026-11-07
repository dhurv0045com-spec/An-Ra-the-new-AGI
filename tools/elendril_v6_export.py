"""Crash-safe result journal and verified streaming snapshots of V6 outputs.

The source output directory is the primary evidence, not the ZIP. Each result
and journal entry is immutable. A snapshot is a derived, atomically replaced
download artifact; interrupted or racing exports leave the prior verified ZIP.
No promise is made about an abruptly destroyed unsaved Kaggle VM or full disk.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import time
from typing import Callable
from uuid import uuid4
import zipfile

from v5_training.checkpoint_lock import checkpoint_writer

SCHEMA = "elendril-v6-evidence-bundle-v1"
MANIFEST_NAME = "BUNDLE_MANIFEST.json"
CHUNK = 1 << 20
MAX_FILES = 100_000
MAX_BYTES = 32 << 30
MAX_EXTRACT_BYTES = 8 << 30
INTERNAL_PREFIXES = (".pending-", ".staging-", ".LATEST-", ".PREVIOUS-", ".checkpoint-copy-")
INTERNAL_FILES = {".checkpoint-writer", ".elendril-writer", ".v6-journal-writer", ".v6-export-writer"}


class SnapshotChanged(RuntimeError):
    """A worker changed a file during snapshot; raw evidence stays authoritative."""


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      ensure_ascii=False).encode("utf-8") + b"\n"


def _sha(value: str, length: int):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is None:
        raise ValueError("expected an immutable source or protocol hash")


def _member_name(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(c) < 32 for c in value)):
        raise ValueError("unsafe ZIP member name")
    parts = value.split("/")
    if (PurePosixPath(value).is_absolute() or any(part in ("", ".", "..")
            or part.endswith((".", " ")) or PureWindowsPath(part).is_reserved() for part in parts)):
        raise ValueError("unsafe or nonportable ZIP member path")
    return value


def immutable_json(path: Path, value) -> dict:
    """Write once via a hard-link publication; never replace a scientific receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value)
    temporary = path.parent / (".pending-" + uuid4().hex)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.read_bytes() != payload:
            raise ValueError("immutable receipt readback failed")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": sha256(payload).hexdigest(), "bytes": len(payload)}


def append_event(output: Path, event: dict) -> dict:
    """Independent fsynced files avoid torn/shared JSONL records across GPU lanes."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / "journal"
    directory.mkdir(exist_ok=True)
    identity, recorded = uuid4().hex, time.time_ns()
    # UUID publication is collision-safe without a shared counter/lock. Time
    # ordering is descriptive; scientific stage order comes from job/parent IDs.
    record = {"schema": SCHEMA + "/journal", "event_id": identity,
              "recorded_unix_ns": recorded, "event": event}
    return immutable_json(directory / f"{recorded:020d}-{identity}.json", record)


def inventory(output: Path, *, max_files: int = MAX_FILES, max_bytes: int = MAX_BYTES) -> list[dict]:
    root = Path(output).resolve(strict=True)
    if not root.is_dir() or type(max_files) is not int or type(max_bytes) is not int or min(max_files, max_bytes) <= 0:
        raise ValueError("invalid snapshot root or limits")
    records, total, names = [], 0, set()
    for directory, folders, files in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in list(folders):
            path = current / name
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("snapshot refuses linked or escaping directories")
            if (current == root and name == "exports") or name.startswith(INTERNAL_PREFIXES):
                folders.remove(name)
        for name in sorted(files):
            path = current / name
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("snapshot refuses linked or escaping files")
            if name in INTERNAL_FILES or name.startswith(INTERNAL_PREFIXES):
                continue
            member = _member_name(path.relative_to(root).as_posix())
            if member == MANIFEST_NAME or member.casefold() in names:
                raise ValueError("reserved or duplicate portable snapshot member")
            names.add(member.casefold())
            size = path.stat().st_size
            total += size
            if len(records) >= max_files or total > max_bytes:
                raise ValueError("snapshot exceeds registered file/byte limits")
            records.append({"name": member, "byte_size": size})
    return sorted(records, key=lambda row: row["name"])


def export_admission(output: Path, records: list[dict], *, reserve_bytes: int = 256 << 20) -> dict:
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("export reserve must be nonnegative integer bytes")
    payload = sum(row["byte_size"] for row in records)
    # Raw source already occupies disk. A new ZIP needs room alongside the
    # prior ZIP, plus conservative headers/compression overhead and a reserve.
    needed = payload + max(8 << 20, len(records) * 1024) + reserve_bytes
    free = shutil.disk_usage(output).free
    if free < needed:
        raise ValueError(f"NO-GO export disk: need {needed} additional bytes, have {free}; raw evidence retained")
    return {"payload_bytes": payload, "required_additional_bytes": needed,
            "free_bytes": free, "reserve_bytes": reserve_bytes,
            "policy": "raw source plus prior ZIP remain; stage one new ZIP before replacement"}


def _file_sha(path: Path) -> tuple[int, str]:
    size, digest = 0, sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _write_derived(path: Path, payload: bytes) -> None:
    """Replace only generated export indexes; scientific receipts stay immutable."""
    temporary = path.parent / (".pending-" + uuid4().hex)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_readme(root: Path, campaign: dict, score: dict, source: str, protocol_sha: str) -> None:
    status = campaign.get("status", "UNKNOWN")
    body = ("# Elendril V6 results\n\n"
            f"- Campaign status: `{status}`\n- Independent rescore status: `{score.get('status', 'UNKNOWN')}`\n"
            f"- Source commit: `{source}`\n- Protocol SHA-256: `{protocol_sha}`\n\n"
            "This bundle preserves all available successful, failed, partial, and skipped outputs. "
            "Campaign status and rescore status are reported separately; this archive does not promote a model. "
            "See `BUNDLE_MANIFEST.json` for the complete member inventory, per-file hashes, and export custody.\n\n"
            "The output directory is the primary evidence. The ZIP is a verified snapshot at export time; "
            "files written after that snapshot require a later export.\n")
    _write_derived(root / "README_RESULTS.md", body.encode("utf-8"))


def verify_zip(path: Path, *, max_bytes: int = MAX_BYTES, max_files: int = MAX_FILES) -> dict:
    """Read every byte without extracting; reject missing/extra/unsafe members."""
    if type(max_bytes) is not int or type(max_files) is not int or min(max_bytes, max_files) <= 0:
        raise ValueError("invalid ZIP verification limits")
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [_member_name(row.filename) for row in infos]
        if len(names) != len(set(name.casefold() for name in names)) or len(names) > max_files + 1:
            raise ValueError("ZIP has duplicate or excessive members")
        if MANIFEST_NAME not in names or "README_RESULTS.md" not in names or "SHA256SUMS" not in names or any(row.is_dir() for row in infos):
            raise ValueError("ZIP needs manifest, README_RESULTS.md, SHA256SUMS, and file members only")
        for row in infos:
            if ((row.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("ZIP refuses symbolic-link members")
        if sum(row.file_size for row in infos) > max_bytes + (16 << 20):
            raise ValueError("ZIP exceeds uncompressed verification bound")
        manifest_info = archive.getinfo(MANIFEST_NAME)
        if manifest_info.file_size > 16 << 20:
            raise ValueError("oversized ZIP manifest")
        manifest_bytes = archive.read(MANIFEST_NAME)
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("files"), list):
            raise ValueError("unsupported ZIP manifest")
        records = manifest["files"]
        expected = {_member_name(row["name"]): row for row in records}
        if len(expected) != len(records) or set(names) != set(expected) | {MANIFEST_NAME}:
            raise ValueError("ZIP inventory disagrees with manifest")
        for name, record in expected.items():
            if type(record["byte_size"]) is not int or record["byte_size"] < 0:
                raise ValueError("invalid declared ZIP byte size")
            _sha(record["sha256"], 64)
            size, fingerprint = 0, sha256()
            with archive.open(name) as handle:
                for chunk in iter(lambda: handle.read(CHUNK), b""):
                    size += len(chunk)
                    if size > record["byte_size"]:
                        raise ValueError("ZIP member longer than manifest")
                    fingerprint.update(chunk)
            if size != record["byte_size"] or fingerprint.hexdigest() != record["sha256"]:
                raise ValueError("ZIP member integrity mismatch: " + name)
        sums = archive.read("SHA256SUMS").decode("utf-8")
        declared = {}
        for line in sums.splitlines():
            digest, separator, member = line.partition("  ")
            _sha(digest, 64)
            member = _member_name(member)
            if not separator or member == "SHA256SUMS" or member in declared:
                raise ValueError("invalid SHA256SUMS entry")
            declared[member] = digest
        if declared != {name: row["sha256"] for name, row in expected.items() if name != "SHA256SUMS"}:
            raise ValueError("SHA256SUMS does not cover the complete bundle")
    return {"status": "EXPORT_VERIFIED", "manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "files": len(records), "source": manifest["source"],
            "protocol_sha256": manifest["protocol_sha256"], "campaign_status": manifest["campaign_status"],
            "scientific_rescore": manifest["rescore"]}


def extract_verified(zip_path: Path, destination: Path, reserve_bytes: int = 1 << 30) -> dict:
    """Safely materialize a saved evidence ZIP into a new, separate directory.

    Verification completes before creating the destination. Extraction never
    calls ``extractall``; every member is streamed to a contained exclusive
    file and checked against the independently verified manifest. Failed
    mid-stream extractions keep an explicit marker and all bytes already read.
    """
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("extraction reserve must be a nonnegative integer")
    archive_requested = Path(zip_path)
    if archive_requested.is_symlink():
        raise ValueError("refuse symbolic-link ZIP input")
    archive = archive_requested.resolve(strict=True)
    if not archive.is_file():
        raise ValueError("ZIP input must be a regular file")
    # Authenticate all names, file types, sizes, SHA values and CRCs before a
    # destination exists. Hash the outer file too to detect replacement while
    # the second streaming pass is in progress.
    zip_size, zip_sha = _file_sha(archive)
    verified = verify_zip(archive, max_bytes=MAX_EXTRACT_BYTES, max_files=MAX_FILES)
    destination_requested = Path(destination)
    if not destination_requested.name or destination_requested.name in (".", ".."):
        raise ValueError("destination must name a new directory")
    parent_requested = destination_requested.parent
    parent = parent_requested.resolve(strict=True)
    if not parent.is_dir() or any(p.is_symlink() for p in (parent, *parent.parents)):
        raise ValueError("destination parent must be an existing real directory without symlink ancestors")
    root = parent / destination_requested.name
    if root.exists() or root.is_symlink():
        raise FileExistsError("verified extraction requires a new destination directory")
    archive_parent = archive.parent.resolve(strict=True)
    if (root == archive_parent or root.is_relative_to(archive_parent)
            or archive_parent == root or archive_parent.is_relative_to(root)):
        raise ValueError("extraction destination must not overlap the ZIP's containing directory")
    free = shutil.disk_usage(parent).free
    with zipfile.ZipFile(archive, "r") as source_zip:
        manifest_bytes = source_zip.read(MANIFEST_NAME)
        manifest = json.loads(manifest_bytes)
        records = manifest["files"]
        expected = {row["name"]: row for row in records}
        infos = {row.filename: row for row in source_zip.infolist()}
        total = sum(row.file_size for row in source_zip.infolist())
        needed = total + reserve_bytes
        if free < needed:
            raise ValueError(f"NO-GO extraction disk: need {needed} bytes, have {free}")
        names = list(expected) + [MANIFEST_NAME]
        marker_name = ".EXTRACTION_UNCOMMITTED.json"
        if marker_name in names:
            raise ValueError("archive uses reserved extraction marker name")
        for name in names:
            _member_name(name)
        # Reject file/directory prefix collisions before publishing any bytes.
        file_names = set(names)
        for name in names:
            parts = name.split("/")
            if any("/".join(parts[:index]) in file_names for index in range(1, len(parts))):
                raise ValueError("ZIP member collides with a required parent directory: " + name)

        root.mkdir()
        marker = root / marker_name
        state = {"status": "UNCOMMITTED", "zip_path": str(archive), "zip_sha256": zip_sha,
                 "destination": str(root), "verified_manifest_sha256": verified["manifest_sha256"],
                 "reason": "extraction started; this marker is removed only after every file is readback-verified"}
        try:
            with marker.open("xb") as handle:
                handle.write(canonical(state)); handle.flush(); os.fsync(handle.fileno())
            for name in names:
                safe_name = _member_name(name)
                target = root.joinpath(*safe_name.split("/"))
                if not target.resolve().is_relative_to(root) or target.is_symlink():
                    raise ValueError("extraction path escaped or became linked: " + name)
                target.parent.mkdir(parents=True, exist_ok=True)
                parent_cursor = target.parent
                while parent_cursor != root.parent:
                    if parent_cursor.is_symlink() or not parent_cursor.resolve().is_relative_to(root):
                        raise ValueError("extraction parent became linked or escaped: " + name)
                    if parent_cursor == root:
                        break
                    parent_cursor = parent_cursor.parent
                declared_size = infos[name].file_size
                declared_sha = expected[name]["sha256"] if name != MANIFEST_NAME else sha256(manifest_bytes).hexdigest()
                count, fingerprint = 0, sha256()
                with source_zip.open(name, "r") as source_handle, target.open("xb") as target_handle:
                    for chunk in iter(lambda: source_handle.read(CHUNK), b""):
                        count += len(chunk)
                        if count > declared_size:
                            raise ValueError("ZIP member expanded beyond declared size: " + name)
                        fingerprint.update(chunk); target_handle.write(chunk)
                    target_handle.flush(); os.fsync(target_handle.fileno())
                actual_size, actual_sha = _file_sha(target)
                if count != declared_size or actual_size != declared_size or actual_sha != declared_sha:
                    raise ValueError("extracted member failed size/SHA readback: " + name)
            after_size, after_sha = _file_sha(archive)
            if after_size != zip_size or after_sha != zip_sha:
                raise ValueError("ZIP changed during verified extraction")
            marker.unlink()
        except Exception as exc:
            state.update({"status": "FAILED_UNCOMMITTED", "error_type": type(exc).__name__, "error": str(exc)})
            try:
                with marker.open("wb") as handle:
                    handle.write(canonical(state)); handle.flush(); os.fsync(handle.fileno())
            except OSError:
                pass
            raise RuntimeError(f"verified extraction failed; uncommitted partial retained at {root}: {exc}") from exc
    return {"status": "EXTRACTION_VERIFIED", "verified_root": str(root), "zip_path": str(archive),
            "zip_sha256": zip_sha, "manifest_sha256": verified["manifest_sha256"],
            "files": len(records), "uncompressed_bytes": total, "free_bytes": free,
            "reserve_bytes": reserve_bytes, "source": verified["source"],
            "protocol_sha256": verified["protocol_sha256"]}


def export_bundle(output: Path, *, source: str, protocol_sha: str, campaign: dict,
                  rescore: Callable[[Path, dict], dict] | None = None,
                  zip_name: str = "ELENDRIL_V6_RESULTS.zip", reserve_bytes: int = 256 << 20,
                  max_files: int = MAX_FILES, max_bytes: int = MAX_BYTES) -> dict:
    """Publish all stable files, including negatives, with honest rescore status.

    Call only while checkpoint publication is quiescent for a final snapshot.
    An incremental attempt may return SnapshotChanged; callers may defer it,
    preserving the immutable journal and previous ZIP rather than blocking work.
    """
    _sha(source, 40); _sha(protocol_sha, 64)
    _member_name(zip_name)
    if "/" in zip_name or not zip_name.endswith(".zip"):
        raise ValueError("export ZIP must be a single .zip filename")
    requested = Path(output)
    root = requested.resolve(strict=True)
    if requested.is_symlink() or not root.is_dir():
        raise ValueError("export requires an existing output directory")
    with checkpoint_writer(root, filename=".v6-export-writer"):
        if rescore is None:
            score = {"status": "NOT_RUN", "reason": "No independent rescorer supplied; no scientific verification claimed"}
        else:
            try:
                score = rescore(root, campaign)
                if not isinstance(score, dict) or not isinstance(score.get("status"), str):
                    raise ValueError("independent rescorer must return an explicit status")
            except Exception as exc:
                score = {"status": "FAILED", "error_type": type(exc).__name__, "message": str(exc)}
        # This immutable audit is included in the ZIP even if rescoring failed.
        audit = root / "export-audits" / (uuid4().hex + ".json")
        immutable_json(audit, {"source": source, "protocol_sha256": protocol_sha,
                               "campaign_status": campaign.get("status", "UNKNOWN"), "rescore": score})
        _prepare_readme(root, campaign, score, source, protocol_sha)
        records = inventory(root, max_files=max_files, max_bytes=max_bytes)
        sums = "".join(f"{row['sha256']}  {row['name']}\n" for row in (
            {**record, "sha256": _file_sha(root / record["name"])[1]} for record in records
        ) if row["name"] != "SHA256SUMS").encode("utf-8")
        _write_derived(root / "SHA256SUMS", sums)
        records = inventory(root, max_files=max_files, max_bytes=max_bytes)
        admission = export_admission(root, records, reserve_bytes=reserve_bytes)
        exports = root / "exports"
        exports.mkdir(exist_ok=True)
        temporary = exports / (".pending-" + uuid4().hex + ".zip")
        destination = exports / zip_name
        committed = []
        try:
            with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as archive:
                for record in records:
                    path = root / record["name"]
                    if path.is_symlink() or not path.resolve().is_relative_to(root):
                        raise ValueError("snapshot path changed to a linked/escaping file")
                    before = path.stat()
                    fingerprint, size = sha256(), 0
                    with path.open("rb") as source_handle, archive.open(record["name"], "w", force_zip64=True) as target:
                        for chunk in iter(lambda: source_handle.read(CHUNK), b""):
                            size += len(chunk); fingerprint.update(chunk); target.write(chunk)
                    after = path.stat()
                    if (size != record["byte_size"] or before.st_size != after.st_size
                            or before.st_mtime_ns != after.st_mtime_ns or path.is_symlink()):
                        raise SnapshotChanged("worker changed output while exporting: " + record["name"])
                    committed.append({**record, "sha256": fingerprint.hexdigest()})
                manifest = {"schema": SCHEMA, "source": source, "protocol_sha256": protocol_sha,
                            "campaign_status": campaign.get("status", "UNKNOWN"), "rescore": score,
                            "files": committed, "admission": admission,
                            "scope": "complete stable output snapshot; includes failed/partial receipts; not a model promotion",
                            "excluded": "derived exports, writer locks and uncommitted staging/pending paths"}
                archive.writestr(MANIFEST_NAME, canonical(manifest))
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
            report = verify_zip(temporary, max_files=max_files, max_bytes=max_bytes)
            os.replace(temporary, destination)
            digest = sha256()
            with destination.open("rb") as handle:
                for chunk in iter(lambda: handle.read(CHUNK), b""):
                    digest.update(chunk)
            return {**report, "zip_path": str(destination), "zip_sha256": digest.hexdigest(),
                    "zip_bytes": destination.stat().st_size, "admission": admission,
                    "custody": "session-local until Kaggle saved output is independently retrieved"}
        finally:
            temporary.unlink(missing_ok=True)


def checkpoint_export(output: Path, *, source: str, protocol_sha: str, campaign: dict,
                      rescore=None, reserve_bytes: int = 256 << 20) -> dict:
    """Best-effort intermediate refresh; every failure has a durable journal event."""
    try:
        report = export_bundle(output, source=source, protocol_sha=protocol_sha, campaign=campaign,
                               rescore=rescore, reserve_bytes=reserve_bytes)
        append_event(output, {"kind": "EXPORT_VERIFIED", "zip_sha256": report["zip_sha256"],
                              "campaign_status": campaign.get("status", "UNKNOWN")})
        return report
    except (OSError, ValueError, SnapshotChanged, zipfile.BadZipFile) as exc:
        # No erased files, altered metrics, successful-promotion flag or hidden
        # exception: callers can inspect the explicit deferred state and retry.
        append_event(output, {"kind": "EXPORT_DEFERRED", "error_type": type(exc).__name__, "message": str(exc)})
        return {"status": "EXPORT_DEFERRED", "error_type": type(exc).__name__, "message": str(exc),
                "raw_evidence": str(Path(output).resolve())}
