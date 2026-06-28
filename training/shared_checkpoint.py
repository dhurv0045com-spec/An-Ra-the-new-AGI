from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from anra.anra_paths import DRIVE_DIR, DRIVE_ROOT, DRIVE_V2_CHECKPOINTS, OUTPUT_V2_DIR

GOOGLE_DRIVE_READ_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
GOOGLE_DRIVE_WRITE_SCOPE = "https://www.googleapis.com/auth/drive"
GOOGLE_DRIVE_SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
GOOGLE_DRIVE_FILE_ID_ENV = "ANRA_SHARED_CHECKPOINT_FILE_ID"
REQUIRE_SHARED_MASTER_ENV = "ANRA_REQUIRE_SHARED_MASTER"
CHECKPOINT_ORIGIN_DIR = OUTPUT_V2_DIR / "checkpoint_origins"


def _escape_drive_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _candidate_roots(_filename: str) -> list[Path]:
    roots: list[Path] = []
    override = os.environ.get("ANRA_SHARED_CHECKPOINT_DIR", "").strip()
    if override:
        roots.append(Path(override))
    roots.extend(
        [
            DRIVE_V2_CHECKPOINTS,
            DRIVE_V2_CHECKPOINTS.parent.parent,
            DRIVE_DIR,
            DRIVE_ROOT,
            DRIVE_ROOT / "Shared with me",
        ]
    )
    shared_drives = DRIVE_ROOT.parent / "Shareddrives"
    if shared_drives.exists():
        roots.append(shared_drives)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def find_filesystem_checkpoint(filename: str) -> Path | None:
    """Find the newest checkpoint visible through the mounted Drive filesystem."""
    matches: list[Path] = []
    for root in _candidate_roots(filename):
        candidate = root / filename
        if candidate.is_file():
            matches.append(candidate)
        if root.name == "Shareddrives" and root.exists():
            with contextlib.suppress(OSError):
                matches.extend(path for path in root.rglob(filename) if path.is_file())
    if not matches:
        return None
    # Older notebook versions could leave a stale private copy beside the
    # current shared master. Pick the latest checkpoint rather than silently
    # rewinding an experiment to whichever directory happens to be checked first.
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _drive_service(*, writable: bool = False) -> object | None:
    try:
        from google.colab import auth  # type: ignore

        auth.authenticate_user()
    except Exception as exc:
        print(f"[Drive Shared] Colab API authentication was unavailable: {exc}", flush=True)

    try:
        import google.auth  # type: ignore
        from googleapiclient.discovery import build  # type: ignore
    except Exception as exc:
        print(f"[Drive Shared] Google Drive API dependencies unavailable: {exc}", flush=True)
        return None

    try:
        scope = GOOGLE_DRIVE_WRITE_SCOPE if writable else GOOGLE_DRIVE_READ_SCOPE
        credentials, _project = google.auth.default(scopes=[scope])
        return build("drive", "v3", credentials=credentials, cache_discovery=False)
    except Exception as exc:
        print(f"[Drive Shared] Google Drive API service setup failed: {exc}", flush=True)
        return None


def _find_drive_api_file(
    service: object,
    filename: str,
    *,
    shared_only: bool = False,
) -> dict[str, object] | None:
    explicit_file_id = os.environ.get(GOOGLE_DRIVE_FILE_ID_ENV, "").strip()
    if explicit_file_id:
        try:
            response = (
                service.files()
                .get(
                    fileId=explicit_file_id,
                    supportsAllDrives=True,
                    fields="id,name,mimeType,shortcutDetails(targetId,targetMimeType)",
                )
                .execute()
            )
            return dict(response)
        except Exception as exc:
            print(f"[Drive Shared] configured file id could not be read: {exc}", flush=True)

    escaped = _escape_drive_query(filename)
    queries = [f"sharedWithMe and name = '{escaped}' and trashed = false"]
    if not shared_only:
        queries.insert(0, f"name = '{escaped}' and trashed = false")
    fields = (
        "files(id,name,mimeType,modifiedTime,size,version,shared,ownedByMe,capabilities(canEdit),"
        "shortcutDetails(targetId,targetMimeType),owners(displayName,emailAddress))"
    )
    matches: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for query in queries:
        try:
            response = (
                service.files()
                .list(
                    q=query,
                    spaces="drive",
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                    orderBy="modifiedTime desc",
                    pageSize=10,
                    fields=fields,
                )
                .execute()
            )
        except Exception as exc:
            print(
                f"[Drive Shared] API search failed for checkpoint '{filename}': {exc}", flush=True
            )
            continue
        for file_meta in response.get("files", []):
            item = dict(file_meta)
            file_id = str(item.get("id", ""))
            if file_id and file_id not in seen_ids:
                matches.append(item)
                seen_ids.add(file_id)
    if not matches:
        return None
    return max(matches, key=lambda item: str(item.get("modifiedTime", "")))


def _origin_path(filename: str) -> Path:
    return CHECKPOINT_ORIGIN_DIR / f"{filename}.json"


def _read_origin(filename: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(_origin_path(filename).read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _record_origin(filename: str, payload: dict[str, Any]) -> None:
    target = _origin_path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "filename": filename,
        "recorded_at": time.time(),
        **payload,
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def _record_filesystem_origin(filename: str, source: Path) -> None:
    _record_origin(filename, {"kind": "filesystem", "source_path": str(source)})


def record_filesystem_checkpoint_origin(filename: str, source: Path) -> None:
    """Record a mounted-Drive checkpoint as the save destination for this session."""
    _record_filesystem_origin(filename, source)


def _record_api_origin(filename: str, file_meta: dict[str, Any]) -> None:
    _record_origin(
        filename,
        {
            "kind": "drive_api",
            "file_id": str(file_meta["id"]),
            "version": str(file_meta.get("version", "")),
            "name": str(file_meta.get("name", filename)),
        },
    )


def _copy_checkpoint_to_path(checkpoint: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        shutil.copy2(checkpoint, temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return target


def _download_drive_api_file(service: object, file_id: str, destination: Path) -> bool:
    try:
        from googleapiclient.http import MediaIoBaseDownload  # type: ignore

        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        tmp = destination.with_suffix(destination.suffix + ".shared.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _status, done = downloader.next_chunk()
        tmp.replace(destination)
        return True
    except Exception as exc:
        print(f"[Drive Shared] API checkpoint download failed: {exc}", flush=True)
        return False


def _resolve_api_target(service: object, file_meta: dict[str, Any]) -> dict[str, Any] | None:
    file_id = str(file_meta.get("id", ""))
    if file_meta.get("mimeType") == GOOGLE_DRIVE_SHORTCUT_MIME:
        shortcut = file_meta.get("shortcutDetails", {})
        if isinstance(shortcut, dict):
            file_id = str(shortcut.get("targetId", file_id))
    if not file_id:
        return None
    try:
        fields = "id,name,mimeType,size,version,ownedByMe,capabilities(canEdit)"
        response = (
            service.files()
            .get(
                fileId=file_id,
                supportsAllDrives=True,
                fields=fields,
            )
            .execute()
        )
        return dict(response)
    except Exception as exc:
        print(f"[Drive Shared] could not inspect checkpoint target: {exc}", flush=True)
        return None


def _upload_checkpoint_to_drive_api(checkpoint: Path, origin: dict[str, Any]) -> Path:
    file_id = str(origin.get("file_id", ""))
    if not file_id:
        raise RuntimeError("Shared checkpoint origin has no Google Drive file id.")
    service = _drive_service(writable=True)
    if service is None:
        raise RuntimeError("Google Drive write API is unavailable for the shared checkpoint.")
    target = _resolve_api_target(service, {"id": file_id})
    if target is None:
        raise RuntimeError("The shared checkpoint file is no longer accessible.")
    if not bool(dict(target.get("capabilities", {})).get("canEdit", False)):
        raise PermissionError(
            "The shared checkpoint is read-only for this Google account. "
            "Share it with Editor permission before training."
        )
    expected_version = str(origin.get("version", ""))
    current_version = str(target.get("version", ""))
    if expected_version and current_version and expected_version != current_version:
        raise RuntimeError(
            "The shared checkpoint changed after this session started. "
            "Refusing to overwrite another training session's progress."
        )
    try:
        from googleapiclient.http import MediaFileUpload  # type: ignore

        media = MediaFileUpload(
            str(checkpoint),
            mimetype="application/octet-stream",
            resumable=True,
            chunksize=16 * 1024 * 1024,
        )
        request = service.files().update(
            fileId=file_id,
            media_body=media,
            supportsAllDrives=True,
            fields="id,name,size,version,modifiedTime",
        )
        response = None
        last_percent = -1
        while response is None:
            status, response = request.next_chunk()
            if status is not None:
                percent = int(status.progress() * 100)
                if percent >= last_percent + 10:
                    print(f"[Drive Shared] publishing shared checkpoint: {percent}%", flush=True)
                    last_percent = percent
        result = dict(response)
    except Exception as exc:
        raise RuntimeError(f"Shared checkpoint upload failed: {exc}") from exc
    _record_api_origin(checkpoint.name, result)
    print(
        f"[Drive Shared] shared master checkpoint updated: {result.get('name', checkpoint.name)}",
        flush=True,
    )
    return Path(f"drive-api:{result['id']}")


def sync_checkpoint_to_origin(checkpoint: Path) -> Path:
    """Publish a checkpoint back to the same Drive file it was restored from."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    origin = _read_origin(checkpoint.name)
    if origin and origin.get("kind") == "drive_api":
        return _upload_checkpoint_to_drive_api(checkpoint, origin)
    if origin and origin.get("kind") == "filesystem":
        source = Path(str(origin.get("source_path", "")))
        if source:
            target = _copy_checkpoint_to_path(checkpoint, source)
            print(f"[Drive] checkpoint updated at source: {target}", flush=True)
            return target
    if os.environ.get(REQUIRE_SHARED_MASTER_ENV, "0") == "1":
        raise RuntimeError(
            "No shared master checkpoint origin is recorded. Refusing to create a private "
            "Drive copy. Restore the shared master before training."
        )
    target = DRIVE_V2_CHECKPOINTS / checkpoint.name
    target = _copy_checkpoint_to_path(checkpoint, target)
    _record_filesystem_origin(checkpoint.name, target)
    print(f"[Drive] checkpoint saved to MyDrive: {target}", flush=True)
    return target


def restore_shared_checkpoint(destination: Path, filename: str | None = None) -> Path | None:
    """
    Copy a checkpoint from the owner's My Drive or an editor-visible shared Drive file.

    Returns the source marker/path if a checkpoint was restored. Writes always
    land in the caller's normal local checkpoint path.
    """
    destination = Path(destination)
    filename = filename or destination.name
    print(f"[Drive Shared] looking for checkpoint: {filename}", flush=True)
    service = _drive_service() if os.environ.get("ANRA_SHARED_DRIVE_API", "1") != "0" else None
    if service is not None:
        # The Drive API can see both the owner's My Drive file and an editor's
        # Shared-with-me file. It chooses by modified time, which prevents an
        # old duplicate from rewinding a newer training session.
        master_meta = _find_drive_api_file(service, filename)
        master_target = _resolve_api_target(service, master_meta) if master_meta else None
        if master_target is not None:
            print(f"[Drive Shared] downloading newest master checkpoint {filename}", flush=True)
            if _download_drive_api_file(service, str(master_target["id"]), destination):
                _record_api_origin(filename, master_target)
                return Path(f"drive-api:{master_target['id']}")

    candidate = find_filesystem_checkpoint(filename)
    if candidate is not None:
        if destination.resolve() != candidate.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
        print(
            f"[Drive Shared] restored newest mounted checkpoint {candidate} -> {destination}",
            flush=True,
        )
        _record_filesystem_origin(filename, candidate)
        return candidate

    if os.environ.get(REQUIRE_SHARED_MASTER_ENV, "0") == "1":
        print(
            "[Drive Shared] required master checkpoint was not found in My Drive "
            "or Shared with me; "
            "private Drive copies are disabled.",
            flush=True,
        )
        return None

    if service is None:
        print(
            "[Drive Shared] API lookup disabled and no mounted Drive checkpoint was found.",
            flush=True,
        )
        return None
    file_meta = _find_drive_api_file(service, filename)
    target = _resolve_api_target(service, file_meta) if file_meta else None
    if target is None:
        print(f"[Drive Shared] no checkpoint named '{filename}' was found.", flush=True)
        return None
    print(f"[Drive Shared] downloading checkpoint {filename}", flush=True)
    if _download_drive_api_file(service, str(target["id"]), destination):
        _record_api_origin(filename, target)
        return Path(f"drive-api:{target['id']}")
    return None
