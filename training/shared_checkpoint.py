from __future__ import annotations

import os
import shutil
from pathlib import Path

from anra.anra_paths import DRIVE_DIR, DRIVE_ROOT, DRIVE_V2_CHECKPOINTS


GOOGLE_DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
GOOGLE_DRIVE_SHORTCUT_MIME = "application/vnd.google-apps.shortcut"


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
    except Exception:
        pass

    try:
        import google.auth  # type: ignore
        from googleapiclient.discovery import build  # type: ignore
    except Exception:
        return None

    try:
        credentials, _project = google.auth.default(scopes=[GOOGLE_DRIVE_FILE_SCOPE])
        return build("drive", "v3", credentials=credentials, cache_discovery=False)
    except Exception:
        return None


def _find_drive_api_file(service, filename: str) -> dict[str, object] | None:
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
        except Exception:
            continue
        files = response.get("files", [])
        if files:
            return dict(files[0])
    return None


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

    candidate = find_filesystem_checkpoint(filename)
    if candidate is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, destination)
        print(f"[Drive Shared] restored {candidate} -> {destination}", flush=True)
        return candidate

    if os.environ.get("ANRA_SHARED_DRIVE_API", "1") == "0":
        return None

    service = _drive_service()
    if service is None:
        return None
    file_meta = _find_drive_api_file(service, filename)
    if file_meta is None:
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
        return Path(f"drive-api:{file_id}")
    return None
