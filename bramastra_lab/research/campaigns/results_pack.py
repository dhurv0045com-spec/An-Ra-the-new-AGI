"""Results-only packaging: every experiment result in one small ZIP.

Unlike the full artifact archive (gigabytes of .pt payloads), this pack
contains ONLY results: ledger JSON export, phase outputs, build/xprobe
reports, manifests, logs and summaries. Embedded artifact manifests are
projected to the files actually present in this ZIP and explicitly marked
non-restorable when checkpoint payloads are omitted. Target is ~10MB so it
downloads through the notebook Output panel without a version-save cycle.

Excluded by contract (never results): *.pt, *.bin, *.safetensors,
__pycache__, *.pyc, .git/, working checkpoints bytes.
Included: *.json, *.jsonl, *.md, *.log, *.sqlite (ledger copy), *.csv.

Hard architecture: no locals()/globals(), no bare except, no hardcoded
user paths, explicit allow/deny suffix lists, verified ZIP (testzip),
atomic promote, JSON receipt with sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence

RESULTS_SCHEMA = "bramastra-k8-results-pack/v1"
SAFETY_SCHEMA = "bramastra-k8-safety-snapshot/v1"
SAFETY_KEEP = 4

ALLOW_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".log", ".csv", ".txt", ".sqlite"})
DENY_SUFFIXES = frozenset({".pt", ".bin", ".safetensors", ".pyc", ".pyo"})
DENY_DIRS = frozenset({"__pycache__", ".git", ".pytest_cache", "node_modules"})
DENY_NAMES = frozenset({"artifact_archive.zip", "results-pack.zip"})


class ResultsPackError(RuntimeError):
    """Results packaging refused."""


def _sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a ZIP incrementally instead of loading its whole contents into RAM."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ResultsPackError(f"cannot hash archive {path}: {exc}") from exc
    return digest.hexdigest()


def _allowed(relative: str) -> bool:
    lowered = relative.replace(os.sep, "/").lower()
    parts = lowered.split("/")
    if any(part in DENY_DIRS for part in parts):
        return False
    name = parts[-1]
    if name in DENY_NAMES:
        return False
    suffix = os.path.splitext(name)[1]
    if suffix in DENY_SUFFIXES:
        return False
    return suffix in ALLOW_SUFFIXES


def collect_result_files(sources: Sequence[str | os.PathLike[str]],
                         *, exclude: Sequence[str | os.PathLike[str]] = ()) -> list[tuple[Path, str]]:
    """Walk sources, return (absolute path, archive name) for allowed files.

    Paths under any exclude prefix are skipped (used to keep snapshots out
    of their own source tree).
    """
    excluded = [os.path.abspath(ex) for ex in exclude]
    collected: list[tuple[Path, str]] = []
    for index, source in enumerate(sources):
        root = Path(source)
        if not root.exists():
            continue
        if any(os.path.abspath(root) == ex or os.path.abspath(root).startswith(ex + os.sep)
               for ex in excluded):
            continue
        label = f"source{index}-{root.name}"
        if root.is_file():
            if _allowed(root.name):
                collected.append((root, f"{label}/{root.name}"))
            continue
        for base, dirs, names in os.walk(root):
            if any(os.path.abspath(base) == ex or os.path.abspath(base).startswith(ex + os.sep)
                   for ex in excluded):
                dirs[:] = []
                continue
            dirs[:] = sorted(d for d in dirs if d not in DENY_DIRS)
            for name in sorted(names):
                full = Path(base) / name
                if any(os.path.abspath(full).startswith(ex + os.sep) for ex in excluded):
                    continue
                rel = os.path.relpath(full, root)
                if _allowed(rel):
                    archive_rel = rel.replace(os.sep, "/")
                    collected.append((full, f"{label}/{archive_rel}"))
    return collected


def _archive_name_for_manifest_path(parent: str, relative: str) -> str | None:
    """Map a manifest-relative path into its candidate ZIP member safely."""
    from pathlib import PurePosixPath

    path = PurePosixPath(str(relative).replace("\\", "/"))
    parts = path.parts
    if path.is_absolute() or not parts or any(
            part in ("", ".", "..") for part in parts):
        return None
    return "/".join((parent, *parts)) if parent else "/".join(parts)


def _results_only_metadata(full: Path, arcname: str,
                           archive_names: set[str]) -> bytes | None:
    """Return truthful results-only metadata for K8 artifact records.

    The source export may be restorable, but this ZIP deliberately filters
    checkpoint payloads. Copying its original complete manifest verbatim
    falsely implies that the checkpoint bytes are present in the ZIP.
    """
    if full.name not in {"artifact_manifest.json", "restore_evidence.json"}:
        return None
    try:
        raw = full.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None

    parent = arcname.rpartition("/")[0]
    if full.name == "artifact_manifest.json":
        source_files = document.get("files")
        if not isinstance(source_files, dict):
            return None
        included: dict[str, Any] = {}
        omitted: list[str] = []
        for relative, record in source_files.items():
            candidate = _archive_name_for_manifest_path(parent, str(relative))
            if candidate is not None and candidate in archive_names:
                included[str(relative)] = record
            else:
                omitted.append(str(relative))
        if not omitted:
            return None
        source_complete = bool(document.get("complete", False))
        source_payload_files = int(document.get(
            "payload_files", sum(Path(name).suffix.lower() in DENY_SUFFIXES
                                  for name in source_files)))
        included_payloads = sum(Path(name).suffix.lower() in DENY_SUFFIXES
                                for name in included)
        omitted_payloads = sorted(
            name for name in omitted
            if Path(name).suffix.lower() in DENY_SUFFIXES)
        document["files"] = included
        document["complete"] = False
        document["source_export_complete"] = source_complete
        document["source_payload_files"] = source_payload_files
        document["payload_files"] = included_payloads
        document["source_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        document["results_pack"] = {
            "schema": "bramastra-k8-results-pack-projection/v1",
            "artifact_type": "results-only",
            "restorable_from_archive": not bool(omitted_payloads),
            "omitted_files": sorted(omitted),
            "omitted_payload_files": omitted_payloads,
            "payload_files_in_archive": included_payloads,
        }
        return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
            "utf-8")

    manifest_path = full.with_name("artifact_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        manifest = {}
    manifest_files = manifest.get("files", {}) if isinstance(manifest, dict) else {}
    payload_paths = [
        str(name) for name in manifest_files
        if Path(str(name)).suffix.lower() in DENY_SUFFIXES
    ] if isinstance(manifest_files, dict) else []
    omitted_payloads = sorted(
        name for name in payload_paths
        if (candidate := _archive_name_for_manifest_path(parent, name)) is None
        or candidate not in archive_names)
    source_payload_files = int(document.get("payload_files_exported", 0) or 0)
    if not omitted_payloads and source_payload_files <= 0:
        return None
    document["source_payload_files_exported"] = source_payload_files
    document["payload_files_exported"] = max(
        0, source_payload_files - len(omitted_payloads))
    document["results_pack"] = {
        "schema": "bramastra-k8-results-pack-projection/v1",
        "artifact_type": "results-only",
        "restorable_from_archive": not bool(omitted_payloads),
        "omitted_payload_files": omitted_payloads,
        "payload_files_in_archive": document["payload_files_exported"],
    }
    if omitted_payloads:
        note = str(document.get("note", "")).strip()
        suffix = ("Checkpoint payload bytes are intentionally omitted from this "
                  "results-only ZIP; it cannot restore those checkpoints.")
        document["note"] = f"{note} {suffix}".strip()
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")


def build_results_pack(sources: Sequence[str | os.PathLike[str]], out_zip: str | os.PathLike[str],
                       *, run_id: str = "results") -> dict[str, Any]:
    """Build the verified results ZIP. Returns the receipt dict."""
    return _build(sources, out_zip, run_id=run_id)


def _build(sources: Sequence[str | os.PathLike[str]], out_zip: str | os.PathLike[str],
           *, run_id: str) -> dict:
    destination = Path(out_zip)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ResultsPackError(f"cannot create output dir {destination.parent}: {exc}") from exc
    if destination.exists():
        raise ResultsPackError(f"refusing to overwrite existing pack: {destination}; "
                               "remove it or choose a new path")
    files = collect_result_files(sources)
    if not files:
        raise ResultsPackError("no result files found in sources; nothing to pack")
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".zip",
                                        dir=str(destination.parent))
    os.close(tmp_fd)
    archive_names = {arcname for _, arcname in files}
    try:
        with zipfile.ZipFile(tmp_name, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for full, arcname in files:
                try:
                    projected = _results_only_metadata(full, arcname,
                                                       archive_names)
                    if projected is None:
                        bundle.write(full, arcname)
                    else:
                        bundle.writestr(arcname, projected)
                except OSError as exc:
                    raise ResultsPackError(f"cannot add {full}: {exc}") from exc
        with zipfile.ZipFile(tmp_name) as bundle:
            bad = bundle.testzip()
        if bad is not None:
            raise ResultsPackError(f"pack verification failed at {bad}")
        digest = _sha256_file(tmp_name)
        try:
            shutil.move(tmp_name, destination)
        except OSError as exc:
            raise ResultsPackError(f"cannot promote pack {destination}: {exc}") from exc
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass
    receipt = {
        "schema": RESULTS_SCHEMA,
        "run_id": run_id,
        "archive": destination.name,
        "sha256": digest,
        "bytes": destination.stat().st_size,
        "files": len(files),
    }
    receipt_path = destination.with_suffix(".json")
    try:
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    except OSError as exc:
        raise ResultsPackError(f"cannot write receipt {receipt_path}: {exc}") from exc
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bramastra-package-results",
        description="Pack all experiment results (no payload bytes) into one small ZIP.")
    parser.add_argument("--run-dir", required=True, help="campaign run directory")
    parser.add_argument("--out", required=True, help="NEW output .zip path")
    parser.add_argument("--extra", action="append", default=[],
                        help="additional result file/dir to include (repeatable)")
    parser.add_argument("--run-id", default="results", help="receipt run label")
    return parser


def snapshot_run_dir(run_dir: str | os.PathLike[str], reason: str) -> dict[str, Any]:
    """Best-effort in-run safety snapshot for the last-window guarantee.

    Writes into ``<run_dir>/safety/`` (no path guessing, no working-root
    discovery): timestamped results-only ZIP of the run dir excluding the
    safety dir itself, plus a ``safety-latest.json`` pointer. Keeps the
    newest SAFETY_KEEP snapshots. Raises ResultsPackError only when the
    snapshot cannot be written at all; callers treat failure as advisory
    and never fail the campaign over it.
    """
    import datetime

    root = Path(run_dir)
    if not root.is_dir():
        raise ResultsPackError(f"run dir missing: {run_dir}")
    out_dir = root / "safety"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ResultsPackError(f"cannot create safety dir {out_dir}: {exc}") from exc
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = out_dir / f"safety-{stamp}.zip"
    files = collect_result_files([root], exclude=[out_dir])
    archive_names = {arcname for _, arcname in files}
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".zip", dir=str(out_dir))
    os.close(tmp_fd)
    try:
        with zipfile.ZipFile(tmp_name, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for full, arcname in files:
                try:
                    projected = _results_only_metadata(full, arcname,
                                                       archive_names)
                    if projected is None:
                        bundle.write(full, arcname)
                    else:
                        bundle.writestr(arcname, projected)
                except OSError:
                    continue
        with zipfile.ZipFile(tmp_name) as bundle:
            bad = bundle.testzip()
        if bad is not None:
            raise ResultsPackError(f"safety snapshot verification failed at {bad}")
        digest = _sha256_file(tmp_name)
        try:
            shutil.move(tmp_name, destination)
        except OSError as exc:
            raise ResultsPackError(f"cannot promote snapshot {destination}: {exc}") from exc
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass
    receipt: dict[str, Any] = {
        "schema": SAFETY_SCHEMA,
        "reason": reason,
        "archive": f"safety/{destination.name}",
        "sha256": digest,
        "bytes": destination.stat().st_size,
        "files": len(files),
        "empty": not files,
        "utc": stamp,
    }
    try:
        (out_dir / "safety-latest.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ResultsPackError(f"cannot write safety pointer: {exc}") from exc
    try:
        snaps = sorted(out_dir.glob("safety-*.zip"))
        for stale in snaps[:-SAFETY_KEEP]:
            try:
                stale.unlink()
            except OSError:
                continue
    except OSError:
        pass
    return receipt


def build_safety_snapshot(run_dir: str | os.PathLike[str],
                          working: str | os.PathLike[str],
                          run_id: str, reason: str) -> dict[str, Any]:
    """Timestamped safety snapshot: completed work survives interruption.

    Writes ``<working>/<run-id>-safety-<UTC>.zip`` (results only, same
    allow/deny contract as the final pack) plus a ``<run-id>-safety-latest.json``
    pointer to the newest snapshot. Keeps the newest SAFETY_KEEP snapshots,
    pruning older ones. Never raises for missing inputs: an empty run dir
    yields a snapshot marked ``empty=true`` rather than an error, so the
    last-window sweep always leaves *something* behind. Never overwrites:
    UTC timestamps are unique per call.
    """
    import datetime

    work = Path(working)
    try:
        work.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ResultsPackError(f"cannot create working dir {work}: {exc}") from exc
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_run = "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in run_id) or "run"
    destination = work / f"{safe_run}-safety-{stamp}.zip"
    run_path = Path(run_dir)
    empty = not run_path.is_dir() or not any(run_path.iterdir())
    if empty:
        files: list[tuple[Path, str]] = []
    else:
        files = collect_result_files([run_path])
    archive_names = {arcname for _, arcname in files}
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".zip", dir=str(work))
    os.close(tmp_fd)
    try:
        with zipfile.ZipFile(tmp_name, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for full, arcname in files:
                try:
                    projected = _results_only_metadata(full, arcname,
                                                       archive_names)
                    if projected is None:
                        bundle.write(full, arcname)
                    else:
                        bundle.writestr(arcname, projected)
                except OSError:
                    continue
        with zipfile.ZipFile(tmp_name) as bundle:
            bad = bundle.testzip()
        if bad is not None:
            raise ResultsPackError(f"safety snapshot verification failed at {bad}")
        digest = _sha256_file(tmp_name)
        try:
            shutil.move(tmp_name, destination)
        except OSError as exc:
            raise ResultsPackError(f"cannot promote snapshot {destination}: {exc}") from exc
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass
    receipt: dict[str, Any] = {
        "schema": SAFETY_SCHEMA,
        "run_id": run_id,
        "reason": reason,
        "archive": destination.name,
        "sha256": digest,
        "bytes": destination.stat().st_size,
        "files": len(files),
        "empty": empty,
        "utc": stamp,
    }
    pointer = work / f"{safe_run}-safety-latest.json"
    try:
        pointer.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    except OSError as exc:
        raise ResultsPackError(f"cannot write snapshot pointer {pointer}: {exc}") from exc
    try:
        snaps = sorted(work.glob(f"{safe_run}-safety-*.zip"))
        for stale in snaps[:-SAFETY_KEEP]:
            try:
                stale.unlink()
            except OSError:
                continue
    except OSError:
        pass
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sources: list[str] = [args.run_dir]
    sources.extend(args.extra)
    try:
        receipt = _build(sources, args.out, run_id=args.run_id)
    except ResultsPackError as exc:
        print(json.dumps({"status": "PACK_REFUSED", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "PACKED", **receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
