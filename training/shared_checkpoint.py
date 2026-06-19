from __future__ import annotations

import os
import shutil
from pathlib import Path

from anra.anra_paths import DRIVE_DIR, DRIVE_ROOT, DRIVE_V2_CHECKPOINTS


GOOGLE_DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
GOOGLE_DRIVE_SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
GOOGLE_DRIVE_FILE_ID_ENV = "ANRA_SHARED_CHECKPOINT_FILE_ID"


def _escape_drive_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _candidate_roots(filename: str) -> list[Path]:
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
    """Find a checkpoint visible through the mounted Drive filesystem."""
    for root in _candidate_roots(filename):
        candidate = root / filename
        if candidate.is_file():
            return candidate
        if root.name == "Shareddrives" and root.exists():
            try:
                matches = sorted(
                    (path for path in root.rglob(filename) if path.is_file()),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            except OSError:
                matches = []
            if matches:
                return matches[0]
    return None


def _drive_service():
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
        credentials, _project = google.auth.default(scopes=[GOOGLE_DRIVE_FILE_SCOPE])
        return build("drive", "v3", credentials=credentials, cache_discovery=False)
    except Exception as exc:
        print(f"[Drive Shared] Google Drive API service setup failed: {exc}", flush=True)
        return None


def _find_drive_api_file(service, filename: str) -> dict[str, object] | None:
    explicit_file_id = os.environ.get(GOOGLE_DRIVE_FILE_ID_ENV, "").strip()
    if explicit_file_id:
        try:
            response = service.files().get(
                fileId=explicit_file_id,
                supportsAllDrives=True,
                fields="id,name,mimeType,shortcutDetails(targetId,targetMimeType)",
            ).execute()
            return dict(response)
        except Exception as exc:
            print(f"[Drive Shared] configured file id could not be read: {exc}", flush=True)

    escaped = _escape_drive_query(filename)
    queries = [
        f"sharedWithMe and name = '{escaped}' and trashed = false",
        f"name = '{escaped}' and trashed = false",
    ]
    fields = (
        "files(id,name,mimeType,modifiedTime,size,shared,"
        "shortcutDetails(targetId,targetMimeType),owners(displayName,emailAddress))"
    )
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
            print(f"[Drive Shared] API search failed for checkpoint '{filename}': {exc}", flush=True)
            continue
        files = response.get("files", [])
        if files:
            return dict(files[0])
    return None


def _cache_in_my_drive(checkpoint: Path) -> None:
    """Cache a restored shared checkpoint in the current user's MyDrive once."""
    target = DRIVE_V2_CHECKPOINTS / checkpoint.name
    try:
        if target.resolve() == checkpoint.resolve():
            return
    except OSError:
        pass
    try:
        if target.exists() and target.stat().st_size == checkpoint.stat().st_size:
            print(f"[Drive Shared] MyDrive checkpoint cache already exists: {target}", flush=True)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Drive Shared] caching checkpoint in MyDrive: {target}", flush=True)
        shutil.copy2(checkpoint, target)
        print(f"[Drive Shared] MyDrive checkpoint cache ready: {target}", flush=True)
    except Exception as exc:
        print(f"[Drive Shared] could not cache checkpoint in MyDrive: {exc}", flush=True)


def _download_drive_api_file(service, file_id: str, destination: Path) -> bool:
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


def restore_shared_checkpoint(destination: Path, filename: str | None = None) -> Path | None:
    """
    Copy a checkpoint from visible shared Drive locations or Shared-with-me API.

    Returns the source marker/path if a checkpoint was restored. Writes always
    land in the caller's normal local checkpoint path.
    """
    destination = Path(destination)
    filename = filename or destination.name
    if destination.exists():
        return destination

    print(f"[Drive Shared] looking for checkpoint: {filename}", flush=True)
    candidate = find_filesystem_checkpoint(filename)
    if candidate is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, destination)
        print(f"[Drive Shared] restored {candidate} -> {destination}", flush=True)
        _cache_in_my_drive(destination)
        return candidate

    if os.environ.get("ANRA_SHARED_DRIVE_API", "1") == "0":
        print("[Drive Shared] API lookup disabled and no mounted Drive checkpoint was found.", flush=True)
        return None

    service = _drive_service()
    if service is None:
        print("[Drive Shared] no mounted or API-visible checkpoint was found.", flush=True)
        return None
    file_meta = _find_drive_api_file(service, filename)
    if file_meta is None:
        print(f"[Drive Shared] no Shared-with-me checkpoint named '{filename}' was found.", flush=True)
        return None

    file_id = str(file_meta.get("id", ""))
    if file_meta.get("mimeType") == GOOGLE_DRIVE_SHORTCUT_MIME:
        shortcut = file_meta.get("shortcutDetails", {})
        if isinstance(shortcut, dict):
            file_id = str(shortcut.get("targetId", file_id))
    if not file_id:
        return None

    print(f"[Drive Shared] downloading Shared-with-me checkpoint {filename}", flush=True)
    if _download_drive_api_file(service, file_id, destination):
        _cache_in_my_drive(destination)
        return Path(f"drive-api:{file_id}")
    return None
