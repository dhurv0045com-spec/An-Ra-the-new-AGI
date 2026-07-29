"""Resolve canonical Colab training assets across mounted Drive locations.

Google Colab mounts ``MyDrive`` and shared-drive/shortcut targets, but it does
not expose the web UI's ``Shared with me`` list as an ordinary directory.  The
operator shares one training folder with Editor access and adds a shortcut to
the currently mounted account.  This resolver then finds the same vault from
the owner's account, a secondary Gmail account, or a Shared Drive without
hard-coding an account-specific path.
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


@dataclass(frozen=True)
class ColabTrainingAssets:
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


def _vault_preference(path: Path) -> int:
    normalised = path.as_posix().lower()
    if "/.shortcut-targets-by-id/" in normalised or "/shareddrives/" in normalised:
        return 2
    if normalised.endswith("/mydrive/anra/cluster/checkpoint-vault"):
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
    vaults: list[tuple[int, Path]] = []
    vault_candidates = [*roots, *_named_candidates("checkpoint-vault", roots)]
    for candidate in vault_candidates:
        step = _vault_step(candidate)
        if step is not None:
            vaults.append((step, candidate))
    if not vaults:
        raise FileNotFoundError(
            "No valid checkpoint-vault directory is visible in mounted Drive. "
            "A compressed checkpoint-vault file is not sufficient. Share the "
            "real folder with Editor access and add a shortcut to My Drive."
        )
    vault_step, vault_root = max(
        vaults,
        key=lambda item: (item[0], _vault_preference(item[1]), str(item[1])),
    )
    if require_writable_vault:
        _require_writable_directory(vault_root)

    pack_parts: list[Path] = []
    for name in pack_names:
        matches = [path for path in _named_candidates(name, roots) if path.is_file()]
        if not matches:
            if not require_pack_parts:
                continue
            raise FileNotFoundError(
                f"Missing training asset {name!r}. Put it beside the shared "
                "checkpoint vault or in the mounted account's MyDrive root. "
                "The Colab notebook can use its authenticated Drive API "
                "fallback when the file is only visible in Shared with me."
            )
        pack_parts.append(matches[0])

    key_candidates: list[Path] = []
    for key_name in DEFAULT_SIGNING_KEY_NAMES:
        key_candidates.extend(
            [
                Path(mount_root) / "MyDrive" / "AnRa" / "private" / key_name,
                *_named_candidates(key_name, roots),
            ]
        )
    signing_key = next((path for path in key_candidates if path.is_file()), None)
    return ColabTrainingAssets(
        vault_root=vault_root,
        pack_parts=tuple(pack_parts),
        signing_key=signing_key,
        vault_step=vault_step,
    )
