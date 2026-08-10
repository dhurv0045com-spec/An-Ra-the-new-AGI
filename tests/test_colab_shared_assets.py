from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path

import pytest

from training.colab_shared_assets import (
    CURRENT_FULL_RESUME_METADATA_NAME,
    CURRENT_FULL_RESUME_NAME,
    TRAINING_HOME_NAME,
    resolve_colab_training_assets,
    stage_verified_checkpoint,
)


PACKS = (
    "v4_phase_a_170m_seed1301.tar.gz.part00",
    "v4_phase_a_170m_seed1301.tar.gz.part01",
)
ROOT = Path(__file__).resolve().parents[1]


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


def test_resolves_renamed_direct_my_drive_shortcut(tmp_path: Path) -> None:
    """Drive shortcuts may have an owner-chosen label instead of our name."""
    my_drive = tmp_path / "MyDrive"
    shortcut = _complete_home(my_drive / "Training checkpoint shortcut", 700)

    assets = resolve_colab_training_assets(tmp_path)

    assert assets.training_home == shortcut
    assert assets.vault_step == 700


def test_resolves_shortcut_targets_inside_mydrive_layout(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    target = _complete_home(
        my_drive / ".shortcut-targets-by-id" / "shared-folder-id",
        800,
    )

    assets = resolve_colab_training_assets(tmp_path)

    assert assets.training_home == target
    assert assets.vault_step == 800


def test_rejects_assets_scattered_outside_training_home(tmp_path: Path) -> None:
    my_drive = tmp_path / "MyDrive"
    my_drive.mkdir()
    (my_drive / "anra-v4-step-000000000600-full-resume.pt").write_bytes(b"checkpoint")
    for name in PACKS:
        (my_drive / name).write_bytes(b"pack")

    with pytest.raises(FileNotFoundError, match=TRAINING_HOME_NAME):
        resolve_colab_training_assets(tmp_path, discovery_timeout_seconds=0)


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


def test_discovers_current_checkpoint_when_drive_metadata_is_stale(tmp_path: Path) -> None:
    """The trainer verifies bytes; its next writer-leased save refreshes metadata."""
    my_drive = tmp_path / "MyDrive"
    home = _complete_home(my_drive / TRAINING_HOME_NAME, 100)
    for legacy in home.glob("anra-v4-step-*-full-resume.pt"):
        legacy.unlink()
    current = home / CURRENT_FULL_RESUME_NAME
    current.write_bytes(b"checkpoint-current-replaced")
    (home / CURRENT_FULL_RESUME_METADATA_NAME).write_text(
        '{"global_step": 1700, "size_bytes": 18, "sha256": "stale"}',
        encoding="utf-8",
    )

    assets = resolve_colab_training_assets(tmp_path)

    assert assets.training_home == home
    assert assets.vault_step == 1700


def test_checkpoint_staging_recovers_after_drive_transport_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import training.colab_shared_assets as shared_assets

    stale_source = tmp_path / "stale" / CURRENT_FULL_RESUME_NAME
    refreshed_source = tmp_path / "remounted" / CURRENT_FULL_RESUME_NAME
    refreshed_source.parent.mkdir()
    payload = b"verified full resume"
    refreshed_source.write_bytes(payload)
    destination = tmp_path / "scratch" / "resume-source.pt"
    real_copyfile = shared_assets.shutil.copyfile
    copy_attempts = 0

    def disconnect_once(source: Path, target: Path) -> None:
        nonlocal copy_attempts
        copy_attempts += 1
        if copy_attempts == 1:
            raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
        real_copyfile(source, target)

    recoveries = 0

    def recover_source() -> Path:
        nonlocal recoveries
        recoveries += 1
        return refreshed_source

    monkeypatch.setattr(shared_assets.shutil, "copyfile", disconnect_once)

    successful_source = stage_verified_checkpoint(
        stale_source,
        destination,
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        recover_source=recover_source,
        retry_delay_seconds=0,
    )

    assert successful_source == refreshed_source
    assert destination.read_bytes() == payload
    assert recoveries == 1
    assert copy_attempts == 2
    assert not destination.with_suffix(".pt.tmp").exists()


def test_checkpoint_staging_never_promotes_unverified_bytes(tmp_path: Path) -> None:
    source = tmp_path / CURRENT_FULL_RESUME_NAME
    source.write_bytes(b"changed checkpoint")
    destination = tmp_path / "resume-source.pt"
    destination.write_bytes(b"previous verified checkpoint")

    with pytest.raises(RuntimeError, match="integrity verification"):
        stage_verified_checkpoint(
            source,
            destination,
            expected_size=source.stat().st_size,
            expected_sha256=hashlib.sha256(b"expected checkpoint").hexdigest(),
        )

    assert destination.read_bytes() == b"previous verified checkpoint"
    assert not destination.with_suffix(".pt.tmp").exists()


def test_persistent_drive_disconnect_exhausts_retries_without_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import training.colab_shared_assets as shared_assets

    source = tmp_path / CURRENT_FULL_RESUME_NAME
    source.write_bytes(b"checkpoint")
    destination = tmp_path / "resume-source.pt"
    destination.write_bytes(b"previous verified checkpoint")
    recoveries = 0

    def disconnected_copy(source: Path, target: Path) -> None:
        target.write_bytes(b"partial")
        raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")

    def recover_source() -> Path:
        nonlocal recoveries
        recoveries += 1
        return source

    monkeypatch.setattr(shared_assets.shutil, "copyfile", disconnected_copy)

    with pytest.raises(OSError, match="Transport endpoint"):
        stage_verified_checkpoint(
            source,
            destination,
            expected_size=source.stat().st_size,
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            recover_source=recover_source,
            max_attempts=3,
            retry_delay_seconds=0,
        )

    assert recoveries == 2
    assert destination.read_bytes() == b"previous verified checkpoint"
    assert not destination.with_suffix(".pt.tmp").exists()


def test_protected_notebook_defaults_to_a_sequential_canonical_handoff() -> None:
    notebook_path = ROOT / "notebooks" / "AN_RA_T4_PROTECTED_TRAINER_V4.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    final_cell = "".join(notebook["cells"][-1].get("source", []))

    assert 'WORKER_ROLE = "canonical_trainer"' in source
    assert "Verification complete" not in final_cell
    assert "SystemExit" not in final_cell
    assert "ANRA_DURABLE_CHECKPOINT_STEPS'] = '200'" in final_cell
    assert "latest_training_failure.log" in final_cell
    assert "PACK_CATALOG" in source
    assert "stage_verified_checkpoint(" in source
    assert "drive.flush_and_unmount()" in source
    assert "shutil.copyfile(source_checkpoint, temporary)" not in source
    assert "select_continuation_pack(phase_a_tokens_seen, PACK_CATALOG)" in source
    assert "v4_phase_a_cont_170m_to_500m_seed1301.tar.gz" in source
    assert "phase_A_tokens={phase_a_tokens_seen:,}" in source
    assert "TRAINER FAILURE — LAST 160 LINES" in final_cell
