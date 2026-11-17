"""Verified ELENDRIL V8-X1 session exporter, security verifier, and safe archive extractor."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any, Mapping
import zipfile

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v8x.protocol import X1_SESSIONS

FINAL_STATUS_NAME = "FINAL_STATUS.json"
ARCHIVE_MANIFEST_NAME = "ARCHIVE_MANIFEST.json"
ARCHIVE_SCHEMA = "elendril-x1-archive-manifest-v1"
FINAL_STATUS_SCHEMA = "elendril-x1-final-status-v1"
ALLOWED_FINAL_STATUSES = (
    "NOT_RUN",
    "NOT_LAUNCHED",
    "ADMITTED",
    "COMPLETE",
    "PARTIAL",
    "NO_GO",
    "INVALID_RESULTS",
)
DEFAULT_MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024


def analysis_zip_name(session: str) -> str:
    return f"ELENDRIL_X1_{session}_ANALYSIS.zip"


def full_zip_name(session: str) -> str:
    return f"ELENDRIL_X1_{session}_FULL.zip"


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


def _safe_arcname(name: str) -> str:
    if "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe archive member character in path: {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or any(part in ("", ".") for part in p.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return p.as_posix()


def _write_verified_zip(
    zip_path: Path,
    entries: Mapping[str, Path],
    *,
    meta: Mapping[str, object],
) -> dict[str, Any]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    for stale in zip_path.parent.glob(f".{zip_path.name}.*.tmp"):
        stale.unlink(missing_ok=True)

    sorted_items: list[tuple[str, Path]] = []
    seen_arc: set[str] = set()
    for arc, src in sorted((_safe_arcname(k), v) for k, v in entries.items()):
        if arc == ARCHIVE_MANIFEST_NAME:
            continue
        if arc.endswith(".zip"):
            raise ValueError(f"derived archive cannot recursively embed ZIP member: {arc}")
        if arc in seen_arc:
            raise ValueError(f"duplicate archive member path: {arc}")
        seen_arc.add(arc)
        sorted_items.append((arc, src))

    manifest_entries: list[dict[str, object]] = []
    for arc, src in sorted_items:
        file_sha, size = _file_sha256(src)
        manifest_entries.append({"path": arc, "sha256": file_sha, "bytes": size})

    manifest: dict[str, Any] = {
        "schema": ARCHIVE_SCHEMA,
        **dict(meta),
        "files": manifest_entries,
        "manifest_sha256": digest(manifest_entries),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    fd, tmp_name = tempfile.mkstemp(dir=zip_path.parent, prefix=f".{zip_path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for arc, src in sorted_items:
                zf.write(src, arcname=arc)
            zf.writestr(ARCHIVE_MANIFEST_NAME, manifest_bytes)
        os.replace(tmp_path, zip_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return manifest


def verify_single_x1_zip(
    zip_path: Path,
    *,
    expected_source_commit: str | None = None,
    expected_session: str | None = None,
    expected_final_status_bytes: bytes | None = None,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> dict[str, Any]:
    """Fail-closed ZIP verifier checking traversal, symlinks, duplicates, bounds, and SHA-256."""

    if not zip_path.is_file():
        raise ValueError(f"missing archive: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        raw_names = [info.filename for info in infos]
        if len(raw_names) != len(set(raw_names)):
            raise ValueError(f"duplicate ZIP member entries in {zip_path.name}")

        total_uncompressed = 0
        for info in infos:
            _safe_arcname(info.filename)
            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode == stat.S_IFLNK:
                raise ValueError(f"symlink entry prohibited in archive: {info.filename}")
            if info.file_size < 0:
                raise ValueError(f"negative member size in archive: {info.filename}")
            total_uncompressed += int(info.file_size)
            if total_uncompressed > int(max_uncompressed_bytes):
                raise ValueError(f"archive uncompressed bytes exceed bound ({total_uncompressed} > {max_uncompressed_bytes})")
            if info.filename.endswith(".zip"):
                raise ValueError(f"recursive ZIP prohibited inside {zip_path.name}: {info.filename}")

        names = set(raw_names)
        if ARCHIVE_MANIFEST_NAME not in names or FINAL_STATUS_NAME not in names:
            raise ValueError(f"archive missing {ARCHIVE_MANIFEST_NAME} or {FINAL_STATUS_NAME}: {zip_path}")

        manifest = json.loads(zf.read(ARCHIVE_MANIFEST_NAME).decode("utf-8"))
        if manifest.get("schema") != ARCHIVE_SCHEMA:
            raise ValueError(f"unsupported archive manifest schema: {manifest.get('schema')!r}")
        if expected_source_commit is not None and manifest.get("source_commit") != expected_source_commit:
            raise ValueError(f"archive source_commit mismatch: {manifest.get('source_commit')!r} != {expected_source_commit!r}")
        if expected_session is not None and manifest.get("session") != expected_session:
            raise ValueError(f"archive session mismatch: {manifest.get('session')!r} != {expected_session!r}")

        files = manifest.get("files", [])
        if digest(files) != manifest.get("manifest_sha256"):
            raise ValueError("archive manifest_sha256 digest mismatch")

        declared = {str(entry["path"]): entry for entry in files}
        if set(declared) | {ARCHIVE_MANIFEST_NAME} != names:
            raise ValueError("archive member inventory differs from ARCHIVE_MANIFEST.json")

        for arc, entry in declared.items():
            h = sha256()
            measured = 0
            with zf.open(arc, "r") as fh:
                while chunk := fh.read(65536):
                    h.update(chunk)
                    measured += len(chunk)
                    if measured > int(entry["bytes"]):
                        raise ValueError(f"member size exceeds declared size: {arc}")
            if measured != int(entry["bytes"]) or h.hexdigest() != str(entry["sha256"]):
                raise ValueError(f"checksum/size mismatch for archive member: {arc}")

        status_raw = zf.read(FINAL_STATUS_NAME)
        if expected_final_status_bytes is not None and status_raw != expected_final_status_bytes:
            raise ValueError(f"{FINAL_STATUS_NAME} inside {zip_path.name} differs from standalone file")
        status_obj = json.loads(status_raw.decode("utf-8"))
        if status_obj.get("status") not in ALLOWED_FINAL_STATUSES:
            raise ValueError(f"invalid FINAL_STATUS status: {status_obj.get('status')!r}")
        if expected_source_commit is not None and status_obj.get("source_commit") != expected_source_commit:
            raise ValueError("FINAL_STATUS source_commit mismatch")
        if expected_session is not None and status_obj.get("session") != expected_session:
            raise ValueError("FINAL_STATUS session mismatch")

    return {"manifest": manifest, "final_status": status_obj, "zip_sha256": _file_sha256(zip_path)[0]}


def verify_x1_export(
    output_dir: Path,
    *,
    session: str,
    expected_source_commit: str | None = None,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> dict[str, Any]:
    """Verify standalone FINAL_STATUS.json + ELENDRIL_X1_<session>_ANALYSIS.zip + _FULL.zip."""

    if session not in X1_SESSIONS:
        raise ValueError(f"unknown X1 session: {session!r}")
    status_path = output_dir / FINAL_STATUS_NAME
    a_zip = output_dir / analysis_zip_name(session)
    f_zip = output_dir / full_zip_name(session)
    if not status_path.is_file():
        raise ValueError(f"missing {FINAL_STATUS_NAME} in {output_dir}")
    status_bytes = status_path.read_bytes()
    status_obj = json.loads(status_bytes.decode("utf-8"))
    if status_obj.get("status") not in ALLOWED_FINAL_STATUSES:
        raise ValueError(f"invalid FINAL_STATUS status: {status_obj.get('status')!r}")
    if expected_source_commit is not None and status_obj.get("source_commit") != expected_source_commit:
        raise ValueError("standalone FINAL_STATUS source_commit mismatch")
    if status_obj.get("session") != session:
        raise ValueError("standalone FINAL_STATUS session mismatch")

    a_info = verify_single_x1_zip(
        a_zip,
        expected_source_commit=expected_source_commit,
        expected_session=session,
        expected_final_status_bytes=status_bytes,
        max_uncompressed_bytes=max_uncompressed_bytes,
    )
    f_info = verify_single_x1_zip(
        f_zip,
        expected_source_commit=expected_source_commit,
        expected_session=session,
        expected_final_status_bytes=status_bytes,
        max_uncompressed_bytes=max_uncompressed_bytes,
    )
    a_files = {str(e["path"]): str(e["sha256"]) for e in a_info["manifest"]["files"]}
    f_files = {str(e["path"]): str(e["sha256"]) for e in f_info["manifest"]["files"]}
    mismatched = [k for k, v in a_files.items() if f_files.get(k) != v]
    if mismatched:
        raise ValueError(f"{f_zip.name} missing or mismatched analysis members: {mismatched}")

    return {
        "verified": True,
        "session": session,
        "status": status_obj["status"],
        "source_commit": status_obj["source_commit"],
        "analysis_zip": str(a_zip),
        "analysis_zip_sha256": a_info["zip_sha256"],
        "analysis_manifest_sha256": a_info["manifest"]["manifest_sha256"],
        "full_zip": str(f_zip),
        "full_zip_sha256": f_info["zip_sha256"],
        "full_manifest_sha256": f_info["manifest"]["manifest_sha256"],
        "final_status_sha256": sha256(status_bytes).hexdigest(),
    }


def safe_extract_x1_zip(
    zip_path: Path,
    dest_dir: Path,
    *,
    expected_source_commit: str | None = None,
    expected_session: str | None = None,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> dict[str, Any]:
    """Verify integrity and extract ZIP contents cleanly without path traversal or symlinks."""

    info = verify_single_x1_zip(
        zip_path,
        expected_source_commit=expected_source_commit,
        expected_session=expected_session,
        max_uncompressed_bytes=max_uncompressed_bytes,
    )
    dest_root = dest_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for zinfo in zf.infolist():
            safe_rel = _safe_arcname(zinfo.filename)
            target = (dest_root / safe_rel).resolve()
            if target != dest_root and dest_root not in target.parents:
                raise ValueError(f"path escape blocked during extract: {safe_rel}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zinfo, "r") as src, target.open("wb") as dst:
                while chunk := src.read(65536):
                    dst.write(chunk)
    return info


def _collect_entries(work_dir: Path, *, include_full_weights: bool) -> dict[str, Path]:
    entries: dict[str, Path] = {}
    root_files = (
        "identity.json",
        "environment.json",
        "admission.json",
        "registry.json",
        "data/manifest.json",
    )
    for rel in root_files:
        p = work_dir / rel
        if p.is_file():
            entries[rel] = p

    analysis_dirs = ("analysis", "charts", "jobs")
    for dname in analysis_dirs:
        base = work_dir / dname
        if not base.is_dir():
            continue
        for item in sorted(base.rglob("*")):
            if not item.is_file():
                continue
            if item.name.startswith(".") and item.name.endswith(".tmp"):
                continue
            if not include_full_weights and item.suffix == ".bin":
                continue
            arc = _safe_arcname(item.relative_to(work_dir).as_posix())
            entries[arc] = item

    if include_full_weights:
        for dname in ("checkpoints", "shared_backbones", "logs"):
            base = work_dir / dname
            if not base.is_dir():
                continue
            for item in sorted(base.rglob("*")):
                if not item.is_file():
                    continue
                if item.name.startswith(".") and item.name.endswith(".tmp"):
                    continue
                arc = _safe_arcname(item.relative_to(work_dir).as_posix())
                entries[arc] = item

    return entries


def export_x1_archives(
    *,
    work_dir: Path,
    output_dir: Path,
    session: str,
    status: str,
    source_commit: str,
    completed_jobs: list[str],
    expected_jobs: list[str],
    failed_jobs: list[str] | None = None,
    missing_jobs: list[str] | None = None,
    reason: str | None = None,
    extra_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write FINAL_STATUS.json + ANALYSIS.zip + FULL.zip + external DELIVERY_RECEIPT.json."""

    if session not in X1_SESSIONS:
        raise ValueError(f"unknown X1 session: {session!r}")
    if status not in ALLOWED_FINAL_STATUSES:
        raise ValueError(f"invalid status: {status!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    final_status = {
        "schema": FINAL_STATUS_SCHEMA,
        "session": session,
        "status": status,
        "source_commit": source_commit,
        "expected_jobs": list(expected_jobs),
        "completed_jobs": list(completed_jobs),
        "failed_jobs": list(failed_jobs or []),
        "missing_jobs": list(missing_jobs or []),
        "reason": reason,
        **(dict(extra_status or {})),
    }
    status_bytes = (json.dumps(final_status, indent=2, sort_keys=True) + "\n").encode("utf-8")
    status_path = output_dir / FINAL_STATUS_NAME
    status_path.write_bytes(status_bytes)

    work_status_path = work_dir / FINAL_STATUS_NAME
    work_status_path.write_bytes(status_bytes)

    a_entries = _collect_entries(work_dir, include_full_weights=False)
    a_entries[FINAL_STATUS_NAME] = status_path
    a_zip = output_dir / analysis_zip_name(session)
    _write_verified_zip(
        a_zip,
        a_entries,
        meta={"archive_kind": "analysis", "session": session, "source_commit": source_commit},
    )

    f_entries = _collect_entries(work_dir, include_full_weights=True)
    f_entries[FINAL_STATUS_NAME] = status_path
    f_zip = output_dir / full_zip_name(session)
    _write_verified_zip(
        f_zip,
        f_entries,
        meta={"archive_kind": "full", "session": session, "source_commit": source_commit},
    )

    receipt = verify_x1_export(output_dir, session=session, expected_source_commit=source_commit)
    # Write uncycled external delivery receipt AFTER both ZIPs are verified (per Section 13)
    (output_dir / f"DELIVERY_RECEIPT_{session}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export or verify ELENDRIL V8-X1 session archives.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--session", required=True, choices=tuple(X1_SESSIONS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default=None, help="Expected 40-hex source_commit")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    if args.verify_only:
        res = verify_x1_export(args.output_dir, session=args.session, expected_source_commit=args.source)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    id_path = args.work_dir / "identity.json"
    reg_path = args.work_dir / "registry.json"
    if not id_path.is_file() or not reg_path.is_file():
        raise FileNotFoundError(f"work_dir missing identity.json or registry.json: {args.work_dir}")
    ident = json.loads(id_path.read_text(encoding="utf-8"))
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    src_commit = str(args.source or ident["source_commit"])
    expected_jobs = [str(j["job_id"]) for j in reg.get("jobs", [])]

    completed_jobs: list[str] = []
    failed_jobs: list[str] = []
    missing_jobs: list[str] = []
    for j in reg.get("jobs", []):
        jid = str(j["job_id"])
        jpath = args.work_dir / "jobs" / jid / "job.json"
        if not jpath.is_file():
            missing_jobs.append(jid)
            continue
        jdoc = json.loads(jpath.read_text(encoding="utf-8"))
        if jdoc.get("status") == "COMPLETE":
            completed_jobs.append(jid)
        elif jdoc.get("status") == "FAILED":
            failed_jobs.append(jid)
        else:
            missing_jobs.append(jid)

    status = "COMPLETE" if not failed_jobs and not missing_jobs and completed_jobs else ("PARTIAL" if completed_jobs else "NOT_RUN")
    receipt = export_x1_archives(
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        session=args.session,
        status=status,
        source_commit=src_commit,
        completed_jobs=completed_jobs,
        expected_jobs=expected_jobs,
        failed_jobs=failed_jobs,
        missing_jobs=missing_jobs,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ALLOWED_FINAL_STATUSES",
    "ARCHIVE_SCHEMA",
    "FINAL_STATUS_NAME",
    "FINAL_STATUS_SCHEMA",
    "analysis_zip_name",
    "export_x1_archives",
    "full_zip_name",
    "safe_extract_x1_zip",
    "verify_single_x1_zip",
    "verify_x1_export",
]

if __name__ == "__main__":
    raise SystemExit(main())
