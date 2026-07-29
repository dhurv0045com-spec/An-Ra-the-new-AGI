from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.colab_shared_assets import resolve_colab_training_assets


PACKS = (
    "v4_phase_a_170m_seed1301.tar.gz.part00",
    "v4_phase_a_170m_seed1301.tar.gz.part01",
)


def _vault(path: Path, step: int) -> Path:
    path.mkdir(parents=True)
    (path / "chunks").mkdir()
    (path / "manifests").mkdir()
    snapshot = f"step-{step:012d}-test"
    (path / "canonical.json").write_text(
        json.dumps({"snapshot_id": snapshot, "global_step": step}),
        encoding="utf-8",
    )
    (path / "manifests" / f"{snapshot}.json").write_text("{}", encoding="utf-8")
    return path


def test_resolves_newest_writable_vault_from_shortcut_target(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    my_drive.mkdir()
    old = _vault(my_drive / "AnRa" / "cluster" / "checkpoint-vault", 100)
    target = tmp_path / ".shortcut-targets-by-id" / "shared-vault-id"
    newest = _vault(target, 253)
    for name in PACKS:
        (my_drive / name).write_bytes(b"pack")
    key = my_drive / "training-signing-keys.json"
    key.write_text("{}", encoding="utf-8")

    assets = resolve_colab_training_assets(tmp_path)

    assert assets.vault_root == newest
    assert assets.vault_root != old
    assert assets.vault_step == 253
    assert tuple(path.name for path in assets.pack_parts) == PACKS
    assert assets.signing_key == key


def test_rejects_compressed_vault_file(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    my_drive.mkdir()
    (my_drive / "checkpoint-vault").write_bytes(b"zip")
    for name in PACKS:
        (my_drive / name).write_bytes(b"pack")

    with pytest.raises(FileNotFoundError, match="compressed checkpoint-vault"):
        resolve_colab_training_assets(tmp_path)


def test_can_resolve_vault_before_shared_pack_api_fallback(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    vault = _vault(my_drive / "checkpoint-vault", 253)

    assets = resolve_colab_training_assets(
        tmp_path,
        require_pack_parts=False,
    )

    assert assets.vault_root == vault
    assert assets.pack_parts == ()


def test_resolves_single_portable_checkpoint_vault(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    vault = my_drive / "AnRa" / "cluster" / "checkpoint-vault"
    vault.mkdir(parents=True)
    (vault / "anra-v4-step-000000000400-full-resume.pt").write_bytes(
        b"portable-checkpoint"
    )
    for name in PACKS:
        (my_drive / name).write_bytes(b"pack")

    assets = resolve_colab_training_assets(tmp_path)

    assert assets.vault_root == vault
    assert assets.vault_step == 400


def test_prefers_recovery_signing_identity_on_recovered_lineage(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    _vault(my_drive / "checkpoint-vault", 3)
    for name in PACKS:
        (my_drive / name).write_bytes(b"pack")
    legacy = my_drive / "training-signing-keys.json"
    legacy.write_text("legacy", encoding="utf-8")
    recovery = my_drive / "anra-v4-recovery-signing-keys.json"
    recovery.write_text("recovery", encoding="utf-8")

    assets = resolve_colab_training_assets(tmp_path)

    assert assets.signing_key == recovery
