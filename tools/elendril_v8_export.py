"""Verified V8 archive exporter, integrity verifier, and safe extractor/restorer."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from anra_v5.elendril.codec import canonical, digest
from v5_training.checkpoint import CheckpointStore

ANALYSIS_ZIP_NAME = "ELENDRIL_V8_ANALYSIS.zip"
FULL_ZIP_NAME = "ELENDRIL_V8_FULL.zip"
FINAL_STATUS_NAME = "FINAL_STATUS.json"
ALLOWED_FINAL_STATUSES = ("NOT_RUN", "COMPLETE", "PARTIAL", "NO_GO", "INVALID_RESULTS")
ANALYSIS_WORK_ROOT_FILES = (
    "splits/SPLIT_AUDIT.json",
    "DIAGNOSTICS.json",
    "COST_VIEWS.json",
    "PAIRWISE_COMPARISONS.json",
    "SCENARIO_LEDGER.json",
    "METRICS.json",
    "RESULTS.md",
)
ANALYSIS_WORK_ROOT_DIRS = ("jobs", "steps", "milestones", "episodes")
FULL_WORK_ROOT_DIRS = ("checkpoints", "logs", "profiles", "canaries")


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
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or any(part == "" for part in p.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return p.as_posix()


def _write_verified_zip(zip_path: Path, entries: dict[str, Path], *, meta: dict[str, object]) -> dict[str, object]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, object]] = []
    sorted_items = sorted((_safe_arcname(arc), path) for arc, path in entries.items())
    for arc, src in sorted_items:
        file_sha, size = _file_sha256(src)
        manifest_entries.append({"path": arc, "sha256": file_sha, "bytes": size})
    manifest = {
        "schema": "elendril-v8-archive-manifest-v1",
        **meta,
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
            zf.writestr("ARCHIVE_MANIFEST.json", manifest_bytes)
        os.replace(tmp_path, zip_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return manifest


def verify_single_zip(
    zip_path: Path,
    *,
    expected_source_commit: str | None = None,
    expected_final_status_bytes: bytes | None = None,
) -> dict[str, object]:
    """Verify every member checksum, manifest hash, source commit, and FINAL_STATUS consistency."""

    if not zip_path.exists():
        raise ValueError(f"missing archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        if "ARCHIVE_MANIFEST.json" not in names or FINAL_STATUS_NAME not in names:
            raise ValueError(f"archive missing manifest or {FINAL_STATUS_NAME}: {zip_path}")
        for name in names:
            _safe_arcname(name)

        manifest = json.loads(zf.read("ARCHIVE_MANIFEST.json").decode("utf-8"))
        if manifest.get("schema") != "elendril-v8-archive-manifest-v1":
            raise ValueError("unsupported archive manifest schema")
        if expected_source_commit is not None and manifest.get("source_commit") != expected_source_commit:
            raise ValueError(
                f"archive source_commit mismatch: got {manifest.get('source_commit')!r}, expected {expected_source_commit!r}"
            )
        files = manifest.get("files", [])
        if digest(files) != manifest.get("manifest_sha256"):
            raise ValueError("archive manifest digest mismatch")
        declared = {str(entry["path"]): entry for entry in files}
        if set(declared) | {"ARCHIVE_MANIFEST.json"} != names:
            raise ValueError("archive contents differ from declared manifest inventory")

        for arc, entry in declared.items():
            raw = zf.read(arc)
            if len(raw) != int(entry["bytes"]) or sha256(raw).hexdigest() != str(entry["sha256"]):
                raise ValueError(f"checksum/size mismatch for archive member: {arc}")

        status_raw = zf.read(FINAL_STATUS_NAME)
        if expected_final_status_bytes is not None and status_raw != expected_final_status_bytes:
            raise ValueError(f"{FINAL_STATUS_NAME} inside {zip_path.name} differs from standalone file")
        status_obj = json.loads(status_raw.decode("utf-8"))
        if status_obj.get("status") not in ALLOWED_FINAL_STATUSES:
            raise ValueError(f"invalid FINAL_STATUS status: {status_obj.get('status')!r}")
        if expected_source_commit is not None and status_obj.get("source_commit") != expected_source_commit:
            raise ValueError("FINAL_STATUS source_commit mismatch")
    return {"manifest": manifest, "final_status": status_obj}


def verify_v8_export(
    output_dir: Path,
    *,
    expected_source_commit: str | None = None,
) -> dict[str, object]:
    """Verify standalone FINAL_STATUS.json, ELENDRIL_V8_ANALYSIS.zip, and ELENDRIL_V8_FULL.zip."""

    status_path = output_dir / FINAL_STATUS_NAME
    analysis_zip = output_dir / ANALYSIS_ZIP_NAME
    full_zip = output_dir / FULL_ZIP_NAME
    if not status_path.exists():
        raise ValueError(f"missing {FINAL_STATUS_NAME} in {output_dir}")
    status_bytes = status_path.read_bytes()
    status_obj = json.loads(status_bytes.decode("utf-8"))
    if status_obj.get("status") not in ALLOWED_FINAL_STATUSES:
        raise ValueError(f"invalid FINAL_STATUS status: {status_obj.get('status')!r}")
    if expected_source_commit is not None and status_obj.get("source_commit") != expected_source_commit:
        raise ValueError("standalone FINAL_STATUS source_commit mismatch")

    a_info = verify_single_zip(
        analysis_zip,
        expected_source_commit=expected_source_commit,
        expected_final_status_bytes=status_bytes,
    )
    f_info = verify_single_zip(
        full_zip,
        expected_source_commit=expected_source_commit,
        expected_final_status_bytes=status_bytes,
    )
    a_files = {str(e["path"]): str(e["sha256"]) for e in a_info["manifest"]["files"]}
    f_files = {str(e["path"]): str(e["sha256"]) for e in f_info["manifest"]["files"]}
    missing_or_mismatched = [k for k, v in a_files.items() if f_files.get(k) != v]
    if missing_or_mismatched:
        raise ValueError(
            f"{FULL_ZIP_NAME} missing or mismatched analysis members: {missing_or_mismatched}"
        )
    return {
        "verified": True,
        "status": status_obj["status"],
        "source_commit": status_obj["source_commit"],
        "analysis_manifest_sha256": a_info["manifest"]["manifest_sha256"],
        "full_manifest_sha256": f_info["manifest"]["manifest_sha256"],
    }


def _collect_work_root_dirs(work_root: Path, rel_dirs: tuple[str, ...]) -> dict[str, Path]:
    collected: dict[str, Path] = {}
    for rel_dir in rel_dirs:
        base = work_root / rel_dir
        if not base.exists():
            continue
        for item in sorted(base.rglob("*")):
            if item.is_file():
                arc = _safe_arcname(item.relative_to(work_root).as_posix())
                collected[arc] = item
    return collected


def export_v8_archives(
    work_root: Path,
    output_dir: Path,
    *,
    source_commit: str,
    phase: str,
    status_payload: dict[str, object],
    analysis_files: dict[str, Path] | None = None,
    include_work_root_evidence: bool = True,
    extra_analysis_files: tuple[str, ...] = ANALYSIS_WORK_ROOT_FILES,
    extra_analysis_dirs: tuple[str, ...] = ANALYSIS_WORK_ROOT_DIRS,
    extra_full_dirs: tuple[str, ...] = FULL_WORK_ROOT_DIRS,
) -> dict[str, object]:
    """Write FINAL_STATUS.json, ELENDRIL_V8_ANALYSIS.zip, and ELENDRIL_V8_FULL.zip and verify immediately."""

    output_dir.mkdir(parents=True, exist_ok=True)
    status_payload = {
        **status_payload,
        "phase": phase,
        "source_commit": source_commit,
    }
    if status_payload.get("status") not in ALLOWED_FINAL_STATUSES:
        raise ValueError(f"invalid FINAL_STATUS status: {status_payload.get('status')!r}")

    status_path = output_dir / FINAL_STATUS_NAME
    tmp_status = output_dir / f".{FINAL_STATUS_NAME}.tmp"
    status_bytes = (json.dumps(status_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp_status.write_bytes(status_bytes)
    os.replace(tmp_status, status_path)

    analysis_entries: dict[str, Path] = {}
    if include_work_root_evidence and work_root.exists():
        for rel_file in extra_analysis_files:
            candidate = work_root / rel_file
            if candidate.exists() and candidate.is_file():
                analysis_entries[_safe_arcname(rel_file)] = candidate
        analysis_entries.update(_collect_work_root_dirs(work_root, extra_analysis_dirs))

    for arc, p in (analysis_files or {}).items():
        if not p.exists():
            raise ValueError(f"required analysis artifact missing: {p}")
        analysis_entries[_safe_arcname(arc)] = p
    analysis_entries[FINAL_STATUS_NAME] = status_path

    meta = {"phase": phase, "source_commit": source_commit, "status": status_payload["status"]}
    analysis_zip = output_dir / ANALYSIS_ZIP_NAME
    _write_verified_zip(analysis_zip, analysis_entries, meta={**meta, "kind": "analysis"})

    full_entries = dict(analysis_entries)
    if work_root.exists():
        full_entries.update(_collect_work_root_dirs(work_root, extra_full_dirs))

    full_zip = output_dir / FULL_ZIP_NAME
    _write_verified_zip(full_zip, full_entries, meta={**meta, "kind": "full"})

    receipt = verify_v8_export(output_dir, expected_source_commit=source_commit)
    return {
        "final_status": str(status_path),
        "analysis_zip": str(analysis_zip),
        "full_zip": str(full_zip),
        **receipt,
    }


def extract_verified_analysis(
    analysis_zip: Path,
    dest_root: Path,
    *,
    expected_source_commit: str | None = None,
) -> Path:
    """Safely verify and extract ELENDRIL_V8_ANALYSIS.zip into dest_root/<campaign_digest>/."""

    info = verify_single_zip(analysis_zip, expected_source_commit=expected_source_commit)
    campaign_id = info["manifest"]["manifest_sha256"][:32]
    target_dir = (dest_root / campaign_id).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(analysis_zip, "r") as zf:
        for member in zf.namelist():
            safe_rel = _safe_arcname(member)
            out_path = (target_dir / safe_rel).resolve()
            if not str(out_path).startswith(str(target_dir)):
                raise ValueError(f"path traversal blocked: {member!r}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(zf.read(member))
    return target_dir


def restore_checkpoint_archive(
    full_zip: Path,
    dest_dir: Path,
    *,
    expected_source_commit: str | None = None,
) -> dict[str, object]:
    """Verify ELENDRIL_V8_FULL.zip, extract checkpoints safely, and verify CheckpointStore lineages."""

    verify_single_zip(full_zip, expected_source_commit=expected_source_commit)
    target = dest_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(full_zip, "r") as zf:
        for member in zf.namelist():
            safe_rel = _safe_arcname(member)
            if not safe_rel.startswith("checkpoints/") and safe_rel not in (FINAL_STATUS_NAME, "ARCHIVE_MANIFEST.json"):
                continue
            out_path = (target / safe_rel).resolve()
            if not str(out_path).startswith(str(target)):
                raise ValueError(f"path traversal blocked: {member!r}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(zf.read(member))
            extracted += 1

    ckpt_root = target / "checkpoints"
    lineages_verified: list[str] = []
    if ckpt_root.exists():
        for lineage_dir in sorted(p for p in ckpt_root.iterdir() if p.is_dir()):
            store = CheckpointStore(ckpt_root, lineage_dir.name)
            latest = store.latest_sha256()
            if latest is not None:
                store.restore(latest)
                lineages_verified.append(lineage_dir.name)
    return {"extracted_files": extracted, "verified_lineages": lineages_verified, "dest_dir": str(target)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ELENDRIL V8 archive verify/extract/restore utility")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ver = sub.add_parser("verify", help="Verify V8 export directory")
    p_ver.add_argument("--dir", type=Path, required=True)
    p_ver.add_argument("--source", default=None)

    p_ext = sub.add_parser("extract-verified", help="Verify and extract analysis zip")
    p_ext.add_argument("--zip", type=Path, required=True)
    p_ext.add_argument("--dest", type=Path, required=True)
    p_ext.add_argument("--source", default=None)

    p_rst = sub.add_parser("restore-archive", help="Verify and restore full checkpoint archive")
    p_rst.add_argument("--zip", type=Path, required=True)
    p_rst.add_argument("--dest", type=Path, required=True)
    p_rst.add_argument("--source", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "verify":
        print(json.dumps(verify_v8_export(args.dir, expected_source_commit=args.source), indent=2))
    elif args.cmd == "extract-verified":
        out = extract_verified_analysis(args.zip, args.dest, expected_source_commit=args.source)
        print(json.dumps({"extracted_dir": str(out)}, indent=2))
    elif args.cmd == "restore-archive":
        print(json.dumps(restore_checkpoint_archive(args.zip, args.dest, expected_source_commit=args.source), indent=2))
    return 0


verify_archive_bundle = verify_single_zip

__all__ = [
    "ANALYSIS_WORK_ROOT_DIRS",
    "ANALYSIS_WORK_ROOT_FILES",
    "ANALYSIS_ZIP_NAME",
    "FINAL_STATUS_NAME",
    "FULL_WORK_ROOT_DIRS",
    "FULL_ZIP_NAME",
    "export_v8_archives",
    "extract_verified_analysis",
    "restore_checkpoint_archive",
    "verify_archive_bundle",
    "verify_single_zip",
    "verify_v8_export",
]

if __name__ == "__main__":
    raise SystemExit(main())
