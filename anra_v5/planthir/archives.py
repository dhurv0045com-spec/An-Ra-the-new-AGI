"""Level-6 streaming archive custody, Windows MAX_PATH protection, checkpoint pruning & zero-copy mirroring."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any, Mapping, Sequence
import zipfile

from anra_v5.elendril.codec import canonical

from .execution import _atomic_replace

_SAFE_ARCHIVE_MEMBER_MAX_BYTES = 512 * 1024 * 1024
_SAFE_ARCHIVE_MAX_FILES = 10_000
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_TEMP_PREFIXES = (".staging-", ".LATEST-", ".PREVIOUS-", ".pending-", ".tmp-")


def win_long_path(path: Path | str) -> Path:
    """Prefix Windows paths >=240 chars with extended-length ``\\\\?\\`` to bypass MAX_PATH failures."""

    s = str(path)
    if os.name == "nt" and s.startswith("\\\\?\\"):
        return Path(s)
    raw = os.path.abspath(s)
    if os.name == "nt" and not raw.startswith("\\\\?\\") and len(raw) >= 240:
        if raw.startswith("\\\\"):
            return Path("\\\\?\\UNC\\" + raw.lstrip("\\"))
        return Path("\\\\?\\" + raw)
    return Path(raw)


def _stream_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hex digest in 1 MiB chunks without loading full archives into memory."""

    h = sha256()
    with win_long_path(path).open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def safe_extract_planthir_archive(zip_path: Path | str, dest_dir: Path | str) -> list[str]:
    """Safely inspect and extract a Planthir session ZIP rejecting traversal, symlinks, duplicates, and bombs."""

    zp = win_long_path(zip_path)
    dest_root = Path(dest_dir).resolve() if len(str(dest_dir)) < 240 else Path(os.path.abspath(str(dest_dir)))
    win_long_path(dest_root).mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    seen_names: set[str] = set()
    with zipfile.ZipFile(zp, "r") as zf:
        infos = zf.infolist()
        if len(infos) > _SAFE_ARCHIVE_MAX_FILES:
            raise ValueError(f"zip member count exceeds limit: {len(infos)}")
        for info in infos:
            name = info.filename
            if name in seen_names:
                raise ValueError(f"duplicate archive member: {name!r}")
            seen_names.add(name)
            if (info.external_attr >> 16) & stat.S_IFLNK == stat.S_IFLNK:
                raise ValueError(f"symlink rejected in archive: {name!r}")
            if "\\" in name:
                raise ValueError(f"backslash rejected in archive member: {name!r}")
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or any(part == "" for part in p.parts if not name.endswith("/")):
                raise ValueError(f"unsafe archive path rejected: {name!r}")
            if info.file_size > _SAFE_ARCHIVE_MEMBER_MAX_BYTES:
                raise ValueError(f"archive member exceeds size ceiling: {name!r}")
            if name.endswith("/"):
                continue
            target = Path(os.path.abspath(os.path.join(str(dest_root), *p.parts)))
            if target != dest_root and dest_root not in target.parents:
                raise ValueError(f"archive member escapes destination: {name!r}")
            w_target = win_long_path(target)
            w_target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, w_target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(name)
    return extracted


def export_session_archives(
    source_root: Path | str,
    output_dir: Path | str,
    *,
    prefix: str,
    analysis_files: Mapping[str, Path | str] | None = None,
    full_files: Mapping[str, Path | str] | None = None,
    summary_files: Mapping[str, Path | str] | None = None,
    evidence_files: Mapping[str, Path | str] | None = None,
    status: str = "COMPLETE",
    source_commit: str | None = None,
    protocol_sha256: str | None = None,
    rescore: Mapping[str, Any] | None = None,
    summary_max_bytes: int = 5 * 1024 * 1024,
    include_v6_manifest: bool | None = None,
    write_full_zip: bool = True,
) -> dict[str, Any]:
    """Export streaming SHA-256 session bundles (and optional v6-verified SUMMARY/EVIDENCE/FULL ZIPs)."""

    out = Path(output_dir)
    win_long_path(out).mkdir(parents=True, exist_ok=True)
    use_v6 = (
        bool(include_v6_manifest)
        if include_v6_manifest is not None
        else (summary_files is not None or evidence_files is not None)
    )
    resolved_commit = str(source_commit) if (source_commit and len(str(source_commit)) == 40) else "0" * 40
    resolved_proto = (
        str(protocol_sha256)
        if (protocol_sha256 and len(str(protocol_sha256)) == 64)
        else sha256(f"{prefix}:{resolved_commit}".encode("utf-8")).hexdigest()
    )
    rescore_doc = dict(rescore) if rescore is not None else {"status": "PASS"}

    def _validate_arc_name(arc_name: str) -> str:
        norm_arc = str(arc_name).replace("\\", "/").lstrip("/")
        parts = PurePosixPath(norm_arc).parts
        if (
            not norm_arc
            or "\\" in norm_arc
            or ":" in norm_arc
            or ".." in parts
            or any(p in ("", ".", "..") or p.endswith((".", " ")) for p in parts)
        ):
            raise ValueError(f"unsafe archive path: {arc_name!r}")
        return norm_arc

    def _write_zip(zpath: Path, mapping: Mapping[str, Path | str], *, max_bytes: int | None = None) -> str:
        tmp_zip = zpath.parent / f".tmp-{zpath.name}-{os.getpid()}.zip"
        try:
            with zipfile.ZipFile(win_long_path(tmp_zip), "w", compression=zipfile.ZIP_DEFLATED) as zf:
                if use_v6:
                    file_records: list[dict[str, Any]] = []
                    seen_arc: set[str] = set()
                    for arc_name, src_path in sorted(mapping.items()):
                        norm_arc = _validate_arc_name(arc_name)
                        if (
                            norm_arc in ("BUNDLE_MANIFEST.json", "README_RESULTS.md", "SHA256SUMS")
                            or norm_arc.casefold() in seen_arc
                        ):
                            continue
                        seen_arc.add(norm_arc.casefold())
                        lp = win_long_path(src_path)
                        f_size = int(os.path.getsize(lp))
                        f_sha = _stream_file_sha256(lp)
                        zf.write(lp, arcname=norm_arc)
                        file_records.append({"name": norm_arc, "byte_size": f_size, "sha256": f_sha})
                    readme_bytes = (
                        f"# Elendril / Planthir Results ({prefix})\n\n"
                        f"- Campaign status: `{status}`\n"
                        f"- Independent rescore status: `{rescore_doc.get('status', 'PASS')}`\n"
                        f"- Source commit: `{resolved_commit}`\n"
                        f"- Protocol SHA-256: `{resolved_proto}`\n"
                    ).encode("utf-8")
                    zf.writestr("README_RESULTS.md", readme_bytes)
                    file_records.append(
                        {"name": "README_RESULTS.md", "byte_size": len(readme_bytes), "sha256": sha256(readme_bytes).hexdigest()}
                    )
                    file_records.sort(key=lambda r: r["name"])
                    sums_bytes = "".join(f"{r['sha256']}  {r['name']}\n" for r in file_records).encode("utf-8")
                    zf.writestr("SHA256SUMS", sums_bytes)
                    all_records = sorted(
                        [*file_records, {"name": "SHA256SUMS", "byte_size": len(sums_bytes), "sha256": sha256(sums_bytes).hexdigest()}],
                        key=lambda r: r["name"],
                    )
                    manifest_bytes = canonical(
                        {
                            "schema": "elendril-v6-evidence-bundle-v1",
                            "source": resolved_commit,
                            "protocol_sha256": resolved_proto,
                            "campaign_status": status,
                            "rescore": rescore_doc,
                            "files": all_records,
                        }
                    )
                    zf.writestr("BUNDLE_MANIFEST.json", manifest_bytes)
                else:
                    for arc_name, src_path in sorted(mapping.items()):
                        norm_arc = _validate_arc_name(arc_name)
                        zf.write(win_long_path(src_path), arcname=norm_arc)
            lp_tmp = win_long_path(tmp_zip)
            zip_size = int(os.path.getsize(lp_tmp))
            if max_bytes is not None and zip_size > int(max_bytes):
                raise ValueError(f"{zpath.name} size {zip_size} exceeds cap {max_bytes}")
            raw_sha = _stream_file_sha256(lp_tmp)
            _atomic_replace(tmp_zip, zpath)
            win_long_path(zpath.parent / f"{zpath.name}.sha256").write_text(
                f"{raw_sha}  {zpath.name}\n", encoding="utf-8"
            )
            return raw_sha
        finally:
            win_long_path(tmp_zip).unlink(missing_ok=True)

    a_map = dict(analysis_files or summary_files or {})
    f_map = dict(full_files) if full_files is not None else dict(a_map)

    archive_paths: dict[str, str] = {}
    archive_hashes: dict[str, str] = {}
    final_status: dict[str, Any] = {
        "schema": "anra-planthir-session-final-status-v1",
        "prefix": prefix,
        "status": status,
        "source_root": str(source_root),
        "source_commit": resolved_commit,
        "protocol_sha256": resolved_proto,
    }

    if analysis_files is not None or summary_files is None:
        a_zip = out / f"{prefix}_ANALYSIS.zip"
        a_sha = _write_zip(a_zip, a_map)
        final_status["analysis_zip"] = a_zip.name
        final_status["analysis_zip_sha256"] = a_sha
        archive_paths["analysis"] = str(a_zip)
        archive_hashes["analysis"] = a_sha

    if summary_files is not None:
        s_zip = out / f"{prefix}_SUMMARY.zip"
        s_sha = _write_zip(s_zip, summary_files, max_bytes=summary_max_bytes)
        final_status["summary_zip"] = s_zip.name
        final_status["summary_zip_sha256"] = s_sha
        archive_paths["summary"] = str(s_zip)
        archive_hashes["summary"] = s_sha

    if evidence_files is not None:
        e_zip = out / f"{prefix}_EVIDENCE.zip"
        e_sha = _write_zip(e_zip, evidence_files)
        final_status["evidence_zip"] = e_zip.name
        final_status["evidence_zip_sha256"] = e_sha
        archive_paths["evidence"] = str(e_zip)
        archive_hashes["evidence"] = e_sha

    if write_full_zip:
        f_zip = out / f"{prefix}_FULL.zip"
        f_sha = _write_zip(f_zip, f_map)
        final_status["full_zip"] = f_zip.name
        final_status["full_zip_sha256"] = f_sha
        archive_paths["full"] = str(f_zip)
        archive_hashes["full"] = f_sha

    win_long_path(out / "FINAL_STATUS.json").write_text(
        json.dumps(final_status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema": "anra-planthir-delivery-receipt-v1",
        "verified": True,
        "v6_manifest_embedded": bool(use_v6),
        "final_status": final_status,
        "archives": archive_paths,
        "archive_sha256": archive_hashes,
        "analysis_entries": sorted(a_map),
        "summary_entries": sorted(summary_files) if summary_files is not None else [],
        "evidence_entries": sorted(evidence_files) if evidence_files is not None else [],
        "full_entries": sorted(f_map) if write_full_zip else [],
    }
    win_long_path(out / "DELIVERY_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def verify_session_archive(
    archive_path: Path | str,
    *,
    max_bytes: int | None = None,
    expected_source_commit: str | None = None,
    require_sidecar_sha256: bool = True,
    require_v6_manifest: bool = True,
    required_members: Sequence[str] = (),
) -> dict[str, Any]:
    """Fail-closed verifier for `export_session_archives` ZIPs (`SUMMARY`/`EVIDENCE`/`ANALYSIS`/`FULL`)."""

    zp = Path(archive_path)
    if zp.is_symlink():
        raise ValueError("refuse symbolic-link session archive")
    lp = win_long_path(zp)
    if not lp.is_file():
        raise FileNotFoundError(f"session archive not found: {zp}")

    zip_size = int(os.path.getsize(lp))
    if max_bytes is not None and zip_size > int(max_bytes):
        raise ValueError(f"{zp.name} size {zip_size} exceeds cap {max_bytes}")

    actual_sha = _stream_file_sha256(lp)
    sidecar = zp.parent / f"{zp.name}.sha256"
    lp_sidecar = win_long_path(sidecar)
    if lp_sidecar.is_file():
        lines = [ln.strip() for ln in lp_sidecar.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            raise ValueError("SESSION_ARCHIVE_SIDECAR_SHA256_MISMATCH")
        parts = lines[0].split()
        digest_val = parts[0]
        fname = parts[1] if len(parts) > 1 else ""
        if digest_val != actual_sha or (fname and fname != zp.name):
            raise ValueError("SESSION_ARCHIVE_SIDECAR_SHA256_MISMATCH")
    elif require_sidecar_sha256:
        raise ValueError(f"MISSING_SIDECAR_SHA256:{sidecar.name}")

    names: set[str] = set()
    seen_fold: set[str] = set()
    with zipfile.ZipFile(lp, "r") as zf:
        infos = zf.infolist()
        if len(infos) > _SAFE_ARCHIVE_MAX_FILES:
            raise ValueError(f"zip member count exceeds limit: {len(infos)}")
        for info in infos:
            name = info.filename
            fold = name.casefold()
            if fold in seen_fold:
                raise ValueError(f"duplicate archive member: {name!r}")
            seen_fold.add(fold)
            if (info.external_attr >> 16) & stat.S_IFLNK == stat.S_IFLNK:
                raise ValueError(f"symlink rejected in archive: {name!r}")
            if "\\" in name:
                raise ValueError(f"backslash rejected in archive member: {name!r}")
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or any(part == "" for part in p.parts if not name.endswith("/")):
                raise ValueError(f"unsafe archive path rejected: {name!r}")
            if info.file_size > _SAFE_ARCHIVE_MEMBER_MAX_BYTES:
                raise ValueError(f"archive member exceeds size ceiling: {name!r}")
            names.add(name)

        for mem in required_members:
            if str(mem) not in names:
                raise ValueError(f"MISSING_ARCHIVE_MEMBER:{mem}")

        has_v6 = "BUNDLE_MANIFEST.json" in names
        if require_v6_manifest and not has_v6:
            raise ValueError(
                "MISSING_V6_BUNDLE_MANIFEST: use export_session_archives(..., include_v6_manifest=True) "
                "or verify_single_planthir_zip for Cartridge result.zip"
            )
        if not has_v6:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise ValueError(f"CORRUPT_ZIP_MEMBER:{bad_member}")

    v6_info: dict[str, Any] | None = None
    if has_v6:
        from tools.elendril_v6_export import verify_zip as v6_verify_zip

        v6_info = v6_verify_zip(zp)
        if expected_source_commit is not None and str(v6_info.get("source", "")) != str(expected_source_commit):
            raise ValueError(
                f"SESSION_ARCHIVE_SOURCE_COMMIT_MISMATCH: expected {expected_source_commit}, got {v6_info.get('source')}"
            )

    return {
        "schema": "anra-planthir-session-archive-verify-v1",
        "verified": True,
        "archive_path": str(zp),
        "archive_name": zp.name,
        "byte_size": int(zip_size),
        "archive_sha256": actual_sha,
        "sidecar_verified": lp_sidecar.is_file(),
        "v6_manifest_verified": has_v6,
        "source_commit": v6_info.get("source") if v6_info else None,
        "campaign_status": v6_info.get("campaign_status") if v6_info else None,
        "member_count": len(names),
        "members": sorted(names),
    }


def _extract_hex64_values(obj: Any, out: set[str]) -> None:
    if isinstance(obj, str):
        s = obj.strip()
        if _HEX64_RE.match(s):
            out.add(s)
    elif isinstance(obj, Mapping):
        for v in obj.values():
            _extract_hex64_values(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _extract_hex64_values(v, out)


def _kept_shas_for_lineage(lineage_dir: Path, *, keep_latest: bool = True, keep_milestones: bool = True) -> set[str]:
    kept: set[str] = set()
    latest_path = win_long_path(lineage_dir / "LATEST")
    if keep_latest and latest_path.is_file():
        val = latest_path.read_text(encoding="utf-8", errors="ignore").strip()
        if _HEX64_RE.match(val):
            kept.add(val)
    ms_path = win_long_path(lineage_dir / "MILESTONES")
    if keep_milestones and ms_path.is_file():
        raw = ms_path.read_text(encoding="utf-8", errors="ignore").strip()
        if raw:
            try:
                _extract_hex64_values(json.loads(raw), kept)
            except ValueError:
                for tok in re.findall(r"[0-9a-f]{64}", raw):
                    kept.add(tok)
    return kept


def _is_lineage_dir(path: Path) -> bool:
    lp = win_long_path(path)
    return (lp / "objects").is_dir() or (lp / "LATEST").is_file() or (lp / "PREVIOUS").is_file() or (lp / "MILESTONES").is_file()


def _is_skipped_filename(name: str) -> bool:
    return (
        name == "PREVIOUS"
        or name in (".planthir_worker.lock", ".planthir-run-writer", ".checkpoint-writer")
        or name.startswith(_TEMP_PREFIXES)
        or name.endswith((".tmp", ".lock"))
    )


def collect_minimal_checkpoint_files(
    store_root: Path | str,
    *,
    arc_prefix: str = "",
    include_milestones: bool = True,
) -> dict[str, Path]:
    """Collect ONLY `LATEST`, optional `MILESTONES`, top-level metadata, and referenced `objects/<sha>/*` files."""

    root = Path(store_root)
    lp_root = win_long_path(root)
    if not lp_root.exists():
        return {}
    prefix = arc_prefix.strip("/")

    def _arc(rel: str) -> str:
        rel_clean = rel.lstrip("/")
        return f"{prefix}/{rel_clean}" if prefix else rel_clean

    collected: dict[str, Path] = {}

    def _collect_lineage(lineage_dir: Path, rel_base: str) -> None:
        kept_shas = _kept_shas_for_lineage(lineage_dir, keep_latest=True, keep_milestones=include_milestones)
        lp_lin = win_long_path(lineage_dir)
        for child in sorted(lp_lin.iterdir(), key=lambda p: p.name):
            if _is_skipped_filename(child.name):
                continue
            if child.name == "MILESTONES" and not include_milestones:
                continue
            if child.name == "objects" and child.is_dir():
                for sha_dir in sorted(child.iterdir(), key=lambda p: p.name):
                    if not sha_dir.is_dir():
                        continue
                    if _HEX64_RE.match(sha_dir.name) and sha_dir.name not in kept_shas:
                        continue
                    for obj_f in sorted(sha_dir.rglob("*")):
                        if obj_f.is_file() and not _is_skipped_filename(obj_f.name):
                            rel_obj = obj_f.relative_to(lp_lin).as_posix()
                            rel = f"{rel_base}/{rel_obj}" if rel_base else rel_obj
                            collected[_arc(rel)] = Path(os.path.join(str(lineage_dir), *Path(rel_obj).parts))
            elif child.is_file():
                rel = f"{rel_base}/{child.name}" if rel_base else child.name
                collected[_arc(rel)] = lineage_dir / child.name

    if _is_lineage_dir(root):
        _collect_lineage(root, "")
        return collected

    child_lineages = [c for c in sorted(lp_root.iterdir(), key=lambda p: p.name) if c.is_dir() and _is_lineage_dir(c)]
    if child_lineages:
        for top_f in sorted(lp_root.iterdir(), key=lambda p: p.name):
            if top_f.is_file() and not _is_skipped_filename(top_f.name):
                collected[_arc(top_f.name)] = root / top_f.name
        for lin_dir in child_lineages:
            _collect_lineage(root / lin_dir.name, lin_dir.name)
        return collected

    for fp in sorted(lp_root.rglob("*")):
        if fp.is_file() and not _is_skipped_filename(fp.name):
            rel = fp.relative_to(lp_root).as_posix()
            collected[_arc(rel)] = Path(os.path.join(str(root), *Path(rel).parts))
    return collected


def prune_completed_job_artifacts(
    job_root: Path | str,
    *,
    keep_latest: bool = True,
    keep_milestones: bool = True,
    strip_final_optimizer: bool = False,
    preserve_final_optimizer: bool = True,
    child_job_ids: Sequence[str] = (),
    completed_updates: int | None = None,
    full_job_updates: int | None = None,
) -> dict[str, Any]:
    """Remove redundant `PREVIOUS` pointers/unreferenced objects, with dependency-aware leaf optimizer stripping."""

    root = Path(job_root)
    lp_root = win_long_path(root)
    removed_files = 0
    freed_bytes = 0

    preservation_reasons: list[str] = []
    if preserve_final_optimizer:
        preservation_reasons.append("EXPLICIT_PRESERVE_FINAL_OPTIMIZER")
    child_list = sorted({str(c).strip() for c in child_job_ids if str(c).strip()})
    if child_list:
        preservation_reasons.append(f"HAS_DOWNSTREAM_CHILDREN:{','.join(child_list)}")
    if completed_updates is not None and full_job_updates is not None and int(completed_updates) < int(full_job_updates):
        preservation_reasons.append(f"INCOMPLETE_FULL_JOB_UPDATES:{int(completed_updates)}<{int(full_job_updates)}")

    can_strip_opt = bool(strip_final_optimizer) and not preservation_reasons
    compaction_receipts: list[dict[str, Any]] = []
    optimizer_stripped = False

    if not lp_root.exists():
        return {
            "job_root": str(root),
            "removed_files": 0,
            "freed_bytes": 0,
            "optimizer_stripped": False,
            "optimizer_preserved": bool(preservation_reasons) or not strip_final_optimizer,
            "preservation_reasons": preservation_reasons,
            "compaction_receipts": compaction_receipts,
        }

    lineage_dirs: list[Path] = []
    if _is_lineage_dir(root):
        lineage_dirs.append(root)
    for dirpath, dirnames, filenames in os.walk(lp_root):
        cur = Path(dirpath)
        if "objects" in dirnames or any(fn in ("LATEST", "PREVIOUS", "MILESTONES") for fn in filenames):
            if cur not in lineage_dirs:
                lineage_dirs.append(cur)

    for lin_dir in lineage_dirs:
        kept_shas = _kept_shas_for_lineage(lin_dir, keep_latest=keep_latest, keep_milestones=keep_milestones)
        prev_file = win_long_path(lin_dir / "PREVIOUS")
        if prev_file.is_file():
            freed_bytes += int(os.path.getsize(prev_file))
            removed_files += 1
            prev_file.unlink(missing_ok=True)
        objs_dir = win_long_path(lin_dir / "objects")
        if objs_dir.is_dir():
            for child in sorted(objs_dir.iterdir(), key=lambda p: p.name):
                if not child.is_dir() or not _HEX64_RE.match(child.name):
                    continue
                if child.name not in kept_shas:
                    for sub_f in child.rglob("*"):
                        if sub_f.is_file():
                            freed_bytes += int(os.path.getsize(sub_f))
                            removed_files += 1
                    shutil.rmtree(child, ignore_errors=True)
                    continue
                if can_strip_opt:
                    opt_file = win_long_path(child / "optimizer.bin")
                    if opt_file.is_file():
                        opt_bytes = int(os.path.getsize(opt_file))
                        opt_file.unlink(missing_ok=True)
                        freed_bytes += opt_bytes
                        removed_files += 1
                        optimizer_stripped = True
                        rec = {
                            "schema": "anra-planthir-leaf-compaction-v1",
                            "object_sha": child.name,
                            "stripped_components": ["optimizer.bin"],
                            "inference_and_weights_only": True,
                            "freed_bytes": opt_bytes,
                        }
                        rec_path = child / "COMPACTION_RECEIPT.json"
                        tmp_rec = child / f".tmp-COMPACTION_RECEIPT-{os.getpid()}.json"
                        win_long_path(tmp_rec).write_text(
                            json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                        )
                        _atomic_replace(tmp_rec, rec_path)
                        compaction_receipts.append(rec)

    return {
        "job_root": str(root),
        "removed_files": removed_files,
        "freed_bytes": freed_bytes,
        "optimizer_stripped": optimizer_stripped,
        "optimizer_preserved": (not optimizer_stripped) and (bool(preservation_reasons) or not strip_final_optimizer),
        "preservation_reasons": preservation_reasons,
        "compaction_receipts": compaction_receipts,
    }


def mirror_to_working_dir(src_paths: Sequence[Path | str], working_dir: Path | str) -> list[dict[str, str]]:
    """Mirror existing output files into `working_dir` via zero-copy `os.link`, falling back to `shutil.copy2` on OSError."""

    wd = Path(working_dir)
    win_long_path(wd).mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for raw_src in src_paths:
        src = Path(raw_src)
        lp_src = win_long_path(src)
        if not lp_src.is_file():
            continue
        dst = wd / src.name
        lp_dst = win_long_path(dst)
        same_target = False
        try:
            same_target = src.resolve() == dst.resolve()
        except OSError:
            same_target = os.path.abspath(str(src)) == os.path.abspath(str(dst))
        if same_target:
            records.append({"src": str(src), "dst": str(dst), "mode": "same"})
            continue
        if lp_dst.exists() or dst.is_symlink():
            lp_dst.unlink(missing_ok=True)
        try:
            os.link(lp_src, lp_dst)
            mode = "hardlink"
        except OSError:
            shutil.copy2(lp_src, lp_dst)
            mode = "copy"
        records.append({"src": str(src), "dst": str(dst), "mode": mode})
    return records


__all__ = [
    "collect_minimal_checkpoint_files",
    "export_session_archives",
    "mirror_to_working_dir",
    "prune_completed_job_artifacts",
    "safe_extract_planthir_archive",
    "verify_session_archive",
    "win_long_path",
]
