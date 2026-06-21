from __future__ import annotations

import json
from pathlib import Path

from scripts.colab_prepare_data import CACHE_FILES, cache_is_valid, copy_cached_files, write_manifest


ROOT = Path(__file__).resolve().parents[1]


def _write_cache_files(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(CACHE_FILES):
        (root / name).write_bytes(f"prepared-{index}".encode("utf-8"))


def test_drive_data_cache_round_trip(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    local = tmp_path / "local"
    _write_cache_files(drive)
    write_manifest(drive, "t4-cached")

    assert cache_is_valid(drive, "t4-cached")
    copy_cached_files(drive, local)

    assert cache_is_valid(local, "t4-cached")
    for name in CACHE_FILES:
        assert (local / name).read_bytes() == (drive / name).read_bytes()


def test_t4_notebook_uses_fast_bootstrap_and_persistent_data_cache() -> None:
    notebook = json.loads((ROOT / "notebooks" / "AN_RA_T4_TRAINING.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "scripts/colab_prepare_data.py" in source
    assert "DATA_PROFILE = 't4-cached'" in source
    assert "os.environ['ANRA_DATA_PROFILE'] = DATA_PROFILE" in source
    assert "ANRA_TRAINING_DATA_LAYOUT" in source
    assert "ANRA_REQUIRE_SHARED_MASTER" in source
    assert "scripts/download_training_data.py --profile $DATA_PROFILE" not in source
    assert "PIP_DISABLE_PIP_VERSION_CHECK" in source


def test_colab_bootstrap_keeps_existing_cuda_torch() -> None:
    source = (ROOT / "scripts" / "colab_bootstrap.py").read_text(encoding="utf-8")

    assert '"--no-deps", "-e", str(repo)' in source
    assert 'f"{repo}[evidence]"' not in source
    assert "full preflight skipped" in source


def test_frontier_trainer_has_no_legacy_brain_autosave() -> None:
    source = (ROOT / "scripts" / "build_brain.py").read_text(encoding="utf-8")

    assert 'sync_to_drive("brain")' not in source
    assert "sync_v2_artifacts" not in source
    assert "DRIVE_SESSION_MANAGER" not in source
    assert "compact evaluation failed after checkpoint save" in source
