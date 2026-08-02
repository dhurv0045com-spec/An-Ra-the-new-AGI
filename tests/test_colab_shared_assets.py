from __future__ import annotations

from pathlib import Path

import pytest

from training.colab_shared_assets import (
    CURRENT_FULL_RESUME_METADATA_NAME,
    CURRENT_FULL_RESUME_NAME,
    TRAINING_HOME_NAME,
    resolve_colab_training_assets,
)


PACKS = (
    "v4_phase_a_170m_seed1301.tar.gz.part00",
    "v4_phase_a_170m_seed1301.tar.gz.part01",
)


def _complete_home(path: Path, step: int) -> Path:
    path.mkdir(parents=True)
    (path / f"anra-v4-step-{step:012d}-full-resume.pt").write_bytes(b"checkpoint")
    for name in PACKS:
        (path / name).write_bytes(b"pack")
    (path / "anra-v4-recovery-signing-keys.json").write_text(
        "{}",
        encoding="utf-8",
    )
    return path


def test_resolves_complete_training_home_from_shortcut_target(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    my_drive.mkdir()
    old = _complete_home(my_drive / TRAINING_HOME_NAME, 100)
    newest = _complete_home(
        tmp_path / ".shortcut-targets-by-id" / "shared-folder-id",
        600,
    )

    assets = resolve_colab_training_assets(tmp_path)

    assert assets.training_home == newest
    assert assets.vault_root == newest
    assert assets.vault_root != old
    assert assets.vault_step == 600
    assert tuple(path.name for path in assets.pack_parts) == PACKS
    assert assets.signing_key == newest / "anra-v4-recovery-signing-keys.json"


def test_rejects_assets_scattered_outside_training_home(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    my_drive.mkdir()
    (my_drive / "anra-v4-step-000000000600-full-resume.pt").write_bytes(b"checkpoint")
    for name in PACKS:
        (my_drive / name).write_bytes(b"pack")

    with pytest.raises(FileNotFoundError, match=TRAINING_HOME_NAME):
        resolve_colab_training_assets(tmp_path)


def test_can_inspect_training_home_without_pack_parts(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    home = my_drive / TRAINING_HOME_NAME
    home.mkdir(parents=True)
    (home / "anra-v4-step-000000000253-full-resume.pt").write_bytes(b"checkpoint")

    assets = resolve_colab_training_assets(
        tmp_path,
        require_pack_parts=False,
    )

    assert assets.training_home == home
    assert assets.vault_root == home
    assert assets.pack_parts == ()


def test_resolves_single_portable_checkpoint_training_home(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    home = _complete_home(my_drive / TRAINING_HOME_NAME, 400)

    assets = resolve_colab_training_assets(tmp_path)

    assert assets.training_home == home
    assert assets.vault_root == home
    assert assets.vault_step == 400


def test_prefers_recovery_signing_identity_on_recovered_lineage(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    home = _complete_home(my_drive / TRAINING_HOME_NAME, 600)
    legacy = home / "training-signing-keys.json"
    legacy.write_text("legacy", encoding="utf-8")
    recovery = home / "anra-v4-recovery-signing-keys.json"
    recovery.write_text("recovery", encoding="utf-8")

    assets = resolve_colab_training_assets(tmp_path)

    assert assets.signing_key == recovery


def test_resolves_stable_current_checkpoint_with_verified_metadata(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    home = _complete_home(my_drive / TRAINING_HOME_NAME, 100)
    for legacy in home.glob("anra-v4-step-*-full-resume.pt"):
        legacy.unlink()
    current = home / CURRENT_FULL_RESUME_NAME
    current.write_bytes(b"checkpoint-current")
    (home / CURRENT_FULL_RESUME_METADATA_NAME).write_text(
        '{"global_step": 1657, "size_bytes": 18}',
        encoding="utf-8",
    )

    assets = resolve_colab_training_assets(tmp_path)

    assert assets.training_home == home
    assert assets.vault_step == 1657
