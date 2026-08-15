"""Select and materialize the next immutable Colab foundation-data window."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContinuationPack:
    name: str
    start_token: int
    end_token: int
    archive_sha256: str
    files: tuple[tuple[str, int, str], ...]


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_pack_catalog(rows: Iterable[Mapping[str, object]]) -> tuple[ContinuationPack, ...]:
    packs: list[ContinuationPack] = []
    for row in rows:
        files_value = row.get("files", [])
        if not isinstance(files_value, list) or not files_value:
            raise ValueError("continuation pack requires at least one archive file")
        files: list[tuple[str, int, str]] = []
        for item in files_value:
            if not isinstance(item, Mapping):
                raise ValueError("continuation pack file declaration must be an object")
            files.append(
                (
                    str(item["name"]),
                    int(item["size"]),
                    str(item["sha256"]),
                )
            )
        pack = ContinuationPack(
            name=str(row["name"]),
            start_token=int(row["start_token"]),
            end_token=int(row["end_token"]),
            archive_sha256=str(row["archive_sha256"]),
            files=tuple(files),
        )
        if pack.start_token < 0 or pack.end_token <= pack.start_token:
            raise ValueError(f"invalid continuation token range for {pack.name}")
        if len(pack.archive_sha256) != 64:
            raise ValueError(f"invalid archive digest for {pack.name}")
        packs.append(pack)
    packs.sort(key=lambda item: (item.start_token, item.end_token))
    for previous, current in zip(packs, packs[1:], strict=False):
        if current.start_token != previous.end_token:
            raise ValueError(
                "continuation pack catalog has a gap or overlap: "
                f"{previous.end_token:,} -> {current.start_token:,}"
            )
    return tuple(packs)


def select_continuation_pack(
    phase_tokens_seen: int,
    catalog: Iterable[Mapping[str, object]],
) -> ContinuationPack:
    """Select the single pack containing the checkpoint's next token."""

    seen = max(0, int(phase_tokens_seen))
    packs = parse_pack_catalog(catalog)
    for pack in packs:
        if pack.start_token <= seen < pack.end_token:
            return pack
    final = packs[-1].end_token if packs else 0
    if seen >= final and final > 0:
        raise RuntimeError(
            f"Foundation checkpoint reached catalog boundary {final:,}; publish the next "
            "immutable continuation pack before training again."
        )
    raise RuntimeError(f"No signed continuation pack covers checkpoint token {seen:,}")


def materialize_continuation_pack(
    *,
    training_home: Path,
    scratch_root: Path,
    pack_parent: Path,
    pack: ContinuationPack,
) -> Path:
    """Verify, reconstruct, safely extract, and validate one selected pack."""

    scratch_root.mkdir(parents=True, exist_ok=True)
    archive = scratch_root / f"{pack.name}.tar.gz"
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    try:
        with temporary.open("wb") as target:
            for filename, expected_size, expected_hash in pack.files:
                source = (training_home / filename).resolve()
                if source.parent != training_home.resolve():
                    raise ValueError(f"continuation asset escaped training home: {source}")
                if not source.is_file():
                    raise FileNotFoundError(f"missing continuation data asset: {source}")
                if source.stat().st_size != expected_size or _sha256(source) != expected_hash:
                    raise ValueError(f"continuation data asset failed verification: {source}")
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, target, 8 * 1024 * 1024)
        if _sha256(temporary) != pack.archive_sha256:
            raise ValueError(f"continuation archive hash mismatch: {pack.name}")
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    pack_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(pack_parent, filter="data")
    root = (pack_parent / pack.name).resolve()
    manifest_path = root / "pack_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"extracted continuation manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_start = int(
        manifest.get(
            "data_window_start_token",
            int(manifest["cumulative_phase_tokens"])
            - int(manifest["training_tokens_requested"]),
        )
    )
    if declared_start != pack.start_token:
        raise ValueError("extracted continuation pack start boundary changed")
    if int(manifest["cumulative_phase_tokens"]) != pack.end_token:
        raise ValueError("extracted continuation pack end boundary changed")
    return root
