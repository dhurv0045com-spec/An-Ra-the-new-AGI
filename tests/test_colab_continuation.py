from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from training.colab_continuation import (
    materialize_continuation_pack,
    parse_pack_catalog,
    select_continuation_pack,
)


def _row(name: str, start: int, end: int) -> dict[str, object]:
    payload = b"archive"
    return {
        "name": name,
        "start_token": start,
        "end_token": end,
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "files": [
            {
                "name": f"{name}.tar.gz",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }


def test_selects_pack_containing_checkpoint_boundary() -> None:
    catalog = [_row("p0", 0, 170), _row("p1", 170, 200), _row("p2", 200, 500)]

    assert select_continuation_pack(180, catalog).name == "p1"
    assert select_continuation_pack(200, catalog).name == "p2"
    with pytest.raises(RuntimeError, match="reached catalog boundary"):
        select_continuation_pack(500, catalog)


def test_rejects_catalog_gap() -> None:
    with pytest.raises(ValueError, match="gap or overlap"):
        parse_pack_catalog([_row("p0", 0, 170), _row("p1", 180, 200)])


def test_materializes_only_selected_verified_archive(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    scratch = tmp_path / "scratch"
    parent = tmp_path / "packs"
    name = "continuation"
    archive = home / f"{name}.tar.gz"
    manifest = {
        "data_window_start_token": 170,
        "cumulative_phase_tokens": 200,
        "training_tokens_requested": 30,
    }
    with tarfile.open(archive, "w:gz") as bundle:
        payload = json.dumps(manifest).encode("utf-8")
        info = tarfile.TarInfo(f"{name}/pack_manifest.json")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    row = {
        "name": name,
        "start_token": 170,
        "end_token": 200,
        "archive_sha256": digest,
        "files": [{"name": archive.name, "size": archive.stat().st_size, "sha256": digest}],
    }

    selected = select_continuation_pack(180, [row])
    root = materialize_continuation_pack(
        training_home=home,
        scratch_root=scratch,
        pack_parent=parent,
        pack=selected,
    )

    assert root == (parent / name).resolve()
    assert (root / "pack_manifest.json").is_file()
