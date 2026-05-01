from __future__ import annotations

from pathlib import Path

from runtime.drive_session_manager import DriveSessionManager


def test_corruption_fallback_uses_previous_version(tmp_path: Path) -> None:
    mgr = DriveSessionManager(tmp_path / "drive", session_id="t1")
    src = tmp_path / "brain.pt"

    src.write_text("v1", encoding="utf-8")
    mgr.save_family("training_state", src)
    src.write_text("v2", encoding="utf-8")
    mgr.save_family("training_state", src)

    manifest = mgr._load_manifest()
    latest_rel = manifest["families"]["training_state"]["versions"][0]["path"]
    latest = mgr.session_dir / latest_rel
    latest.write_text("corrupted", encoding="utf-8")

    restored = tmp_path / "restored.pt"
    assert mgr.load_family("training_state", restored)
    assert restored.read_text(encoding="utf-8") == "v1"


def test_restart_restore_behavior(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    src = tmp_path / "ghost.json"
    src.write_text('{"ok": true}', encoding="utf-8")

    mgr_a = DriveSessionManager(drive_root, session_id="restart")
    mgr_a.save_family("ghost", src)

    mgr_b = DriveSessionManager(drive_root, session_id="restart")
    dst = tmp_path / "ghost_restored.json"
    assert mgr_b.load_family("ghost", dst)
    assert dst.read_text(encoding="utf-8") == '{"ok": true}'
