"""Resolve the canonical Colab training home across mounted Drive locations.

Google Colab mounts ``MyDrive`` and shared-drive/shortcut targets, but it does
not expose the web UI's ``Shared with me`` list as an ordinary directory. The
operator shares one ``ANRA_T4_TRAINING_HOME`` folder with Editor access and
adds a shortcut to the currently mounted account. Every asset needed to resume
training lives directly in that folder, so a worker cannot silently combine a
checkpoint, data pack, and signing identity from different Drive locations.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PACK_PARTS = (
    "v4_phase_a_170m_seed1301.tar.gz.part00",
    "v4_phase_a_170m_seed1301.tar.gz.part01",
)
DEFAULT_SIGNING_KEY_NAMES = (
    "anra-v4-recovery-signing-keys.json",
    "training-signing-keys.json",
)
TRAINING_HOME_NAME = "ANRA_T4_TRAINING_HOME"
CURRENT_FULL_RESUME_NAME = "anra-v4-current-full-resume.pt"
CURRENT_FULL_RESUME_METADATA_NAME = "anra-v4-current-full-resume.json"


@dataclass(frozen=True)
class ColabTrainingAssets:
    training_home: Path
    vault_root: Path
    pack_parts: tuple[Path, ...]
    signing_key: Path | None
    vault_step: int


def _bounded_directories(parent: Path, *, limit: int = 256) -> list[Path]:
    if not parent.is_dir():
        return []
    directories: list[Path] = []
    try:
        with os.scandir(parent) as entries:
            for entry in entries:
                if len(directories) >= limit:
                    break
                try:
                    if entry.is_dir(follow_symlinks=True):
                        directories.append(Path(entry.path))
                except OSError:
                    continue
    except OSError:
        return []
    return directories


def mounted_training_roots(
    mount_root: str | Path = "/content/drive",
) -> tuple[Path, ...]:
    """Return bounded, deterministic roots that Colab can actually read.

    This deliberately avoids recursively walking a user's whole Drive.  It
    searches the canonical path, MyDrive root, direct MyDrive folders,
    shortcut targets, and Shared Drives.
    """

    mount = Path(mount_root)
    # Colab normally calls this ``MyDrive``, while some Drive mounts expose
    # ``My Drive``.  More importantly, a folder shortcut is not required to
    # preserve the source folder's name: the owner can rename it when adding
    # it to My Drive.  Treat direct folders as discovery roots and validate
    # their *contents* later, instead of relying on the shortcut label.
    my_drives = (mount / "MyDrive", mount / "My Drive")
    roots: list[Path] = []
    for my_drive in my_drives:
        roots.extend(
            [
                my_drive / "AnRa" / "cluster",
                my_drive / "AnRa",
                my_drive,
            ]
        )
        roots.extend(_bounded_directories(my_drive))

    # Depending on the Drive FUSE version, shortcut targets may be exposed
    # beside MyDrive or inside it.  Probe both layouts.  These opaque target
    # directories deliberately bypass name matching below.
    shortcut_targets: list[Path] = []
    for shortcut_root in (
        mount / ".shortcut-targets-by-id",
        *(my_drive / ".shortcut-targets-by-id" for my_drive in my_drives),
    ):
        shortcut_targets.extend(_bounded_directories(shortcut_root))
    roots.extend(shortcut_targets)
    for target in shortcut_targets:
        roots.extend(
            [
                target / "cluster",
                target / "AnRa" / "cluster",
            ]
        )

    shared_drives = mount / "Shareddrives"
    for shared in _bounded_directories(shared_drives):
        roots.extend([shared, shared / "AnRa" / "cluster", shared / "cluster"])

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = os.path.normcase(os.path.abspath(root))
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return tuple(unique)


def _named_candidates(name: str, roots: Iterable[Path]) -> list[Path]:
    matches: list[Path] = []
    for root in roots:
        if root.name == name and root.exists():
            matches.append(root)
        direct = root / name
        if direct.exists():
            matches.append(direct)
        nested = root / "cluster" / name
        if nested.exists():
            matches.append(nested)
    return matches


def _training_home_candidates(
    mount_root: str | Path,
    roots: Iterable[Path],
) -> tuple[Path, ...]:
    mount = Path(mount_root)
    my_drives = (mount / "MyDrive", mount / "My Drive")
    candidates: list[Path] = []
    for my_drive in my_drives:
        candidates.extend(
            [
                my_drive / TRAINING_HOME_NAME,
                my_drive / "AnRa" / TRAINING_HOME_NAME,
            ]
        )
    for root in roots:
        if root.name == TRAINING_HOME_NAME:
            candidates.append(root)
        candidates.append(root / TRAINING_HOME_NAME)
        # A direct My Drive folder can be a renamed Drive shortcut.  It is
        # safe to try it because a candidate is only accepted after all
        # required checkpoint, pack, and signing-key checks pass.
        if root.name not in {
            "MyDrive",
            "My Drive",
            "AnRa",
            "cluster",
            ".shortcut-targets-by-id",
        }:
            candidates.append(root)
        normalised = root.as_posix().lower()
        if "/.shortcut-targets-by-id/" in normalised:
            # A shortcut target is mounted under its opaque Drive ID rather
            # than under the shared folder's human-readable name.
            candidates.append(root)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def _vault_step(path: Path) -> int | None:
    if not path.is_dir():
        return None
    current = path / CURRENT_FULL_RESUME_NAME
    current_metadata = path / CURRENT_FULL_RESUME_METADATA_NAME
    if current.is_file() and current.stat().st_size > 0 and current_metadata.is_file():
        try:
            payload = json.loads(current_metadata.read_text(encoding="utf-8"))
            step = int(payload["global_step"])
            expected_size = int(payload["size_bytes"])
            if step >= 0 and current.stat().st_size == expected_size:
                return step
            if step >= 0:
                # Google Drive's folder-shortcut FUSE can expose a completed
                # replacement checkpoint before its tiny metadata file catches
                # up.  The Colab notebook hashes and structurally loads this
                # file before it repairs that stale record; refusing discovery
                # here would make a recoverable metadata drift look like a
                # missing shared folder.
                return step
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    portable_steps: list[int] = []
    prefix = "anra-v4-step-"
    suffix = "-full-resume.pt"
    for checkpoint in path.glob(f"{prefix}*{suffix}"):
        step_text = checkpoint.name[len(prefix) : -len(suffix)]
        if checkpoint.is_file() and checkpoint.stat().st_size > 0 and step_text.isdigit():
            portable_steps.append(int(step_text))
    if portable_steps:
        return max(portable_steps)

    # Compatibility with the retired content-addressed Drive layout.  It is
    # accepted only so the next run can resume once and publish the first
    # portable single-file checkpoint.
    pointer_path = path / "canonical.json"
    manifests = path / "manifests"
    chunks = path / "chunks"
    if not (pointer_path.is_file() and manifests.is_dir() and chunks.is_dir()):
        return None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        snapshot_id = str(pointer["snapshot_id"])
        manifest = manifests / f"{snapshot_id}.json"
        if not manifest.is_file():
            return None
        return int(pointer.get("global_step", 0))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _require_writable_directory(path: Path) -> None:
    probe = path / ".anra-write-probe"
    try:
        with probe.open("xb") as handle:
            handle.write(b"ok")
        probe.unlink()
    except OSError as exc:
        probe.unlink(missing_ok=True)
        raise PermissionError(
            f"Checkpoint vault is not writable from this Drive account: {path}. "
            "Share the folder with Editor access, not Viewer access."
        ) from exc


def _home_preference(path: Path) -> int:
    normalised = path.as_posix().lower()
    if "/.shortcut-targets-by-id/" in normalised or "/shareddrives/" in normalised:
        return 2
    if normalised.endswith(f"/mydrive/{TRAINING_HOME_NAME.lower()}"):
        return 1
    return 0


def resolve_colab_training_assets(
    mount_root: str | Path = "/content/drive",
    *,
    pack_names: Iterable[str] = DEFAULT_PACK_PARTS,
    require_writable_vault: bool = True,
    require_pack_parts: bool = True,
    discovery_timeout_seconds: float = 20.0,
) -> ColabTrainingAssets:
    # Drive FUSE can mount before it has indexed newly added folder shortcuts.
    # Refresh a bounded set of locations for a short period instead of making
    # the operator restart Colab or repeatedly rerun a cell.
    deadline = time.monotonic() + max(0.0, float(discovery_timeout_seconds))
    homes: list[tuple[int, Path, tuple[Path, ...], Path | None]] = []
    last_candidates: tuple[Path, ...] = ()
    while True:
        roots = mounted_training_roots(mount_root)
        last_candidates = _training_home_candidates(mount_root, roots)
        homes = []
        for candidate in last_candidates:
            step = _vault_step(candidate)
            if step is None:
                continue
            parts = tuple(candidate / name for name in pack_names)
            if require_pack_parts and not all(path.is_file() for path in parts):
                continue
            if not require_pack_parts:
                parts = tuple(path for path in parts if path.is_file())
            signing_key = next(
                (
                    candidate / key_name
                    for key_name in DEFAULT_SIGNING_KEY_NAMES
                    if (candidate / key_name).is_file()
                ),
                None,
            )
            # A canonical training session cannot safely create or publish a
            # checkpoint without its signing identity.  Reject an otherwise
            # plausible folder here so a wrong Drive shortcut produces one
            # clear discovery error rather than failing later in the notebook.
            if require_pack_parts and signing_key is None:
                continue
            homes.append((step, candidate, parts, signing_key))
        if homes or time.monotonic() >= deadline:
            break
        time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))

    if not homes:
        visible = [str(path) for path in last_candidates if path.exists()][:12]
        visible_text = ", ".join(visible) if visible else "none"
        raise FileNotFoundError(
            f"No complete {TRAINING_HOME_NAME} folder is visible in mounted Drive. "
            "The folder must directly contain one portable full-resume checkpoint, "
            "both V4 data-pack parts, and the campaign signing key. Share that one "
            "folder with Editor access and add its shortcut to My Drive. "
            f"After waiting {max(0.0, float(discovery_timeout_seconds)):.0f}s, "
            f"visible candidate paths were: {visible_text}."
        )
    vault_step, training_home, pack_parts, signing_key = max(
        homes,
        key=lambda item: (item[0], _home_preference(item[1]), str(item[1])),
    )
    if require_writable_vault:
        _require_writable_directory(training_home)

    return ColabTrainingAssets(
        training_home=training_home,
        vault_root=training_home,
        pack_parts=pack_parts,
        signing_key=signing_key,
        vault_step=vault_step,
    )
