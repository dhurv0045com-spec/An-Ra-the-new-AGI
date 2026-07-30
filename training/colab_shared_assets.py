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
    my_drive = mount / "MyDrive"
    roots: list[Path] = [
        my_drive / "AnRa" / "cluster",
        my_drive / "AnRa",
        my_drive,
    ]
    roots.extend(_bounded_directories(my_drive))

    shortcut_root = mount / ".shortcut-targets-by-id"
    shortcut_targets = _bounded_directories(shortcut_root)
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
    my_drive = mount / "MyDrive"
    candidates: list[Path] = [
        my_drive / TRAINING_HOME_NAME,
        my_drive / "AnRa" / TRAINING_HOME_NAME,
    ]
    for root in roots:
        if root.name == TRAINING_HOME_NAME:
            candidates.append(root)
        candidates.append(root / TRAINING_HOME_NAME)
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
) -> ColabTrainingAssets:
    roots = mounted_training_roots(mount_root)
    homes: list[tuple[int, Path, tuple[Path, ...], Path | None]] = []
    for candidate in _training_home_candidates(mount_root, roots):
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
        homes.append((step, candidate, parts, signing_key))

    if not homes:
        raise FileNotFoundError(
            f"No complete {TRAINING_HOME_NAME} folder is visible in mounted Drive. "
            "The folder must directly contain one portable full-resume checkpoint, "
            "both V4 data-pack parts, and the campaign signing key. Share that one "
            "folder with Editor access and add its shortcut to My Drive."
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
