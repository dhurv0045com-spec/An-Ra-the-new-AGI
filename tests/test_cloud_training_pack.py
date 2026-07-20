from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import build_cloud_training_pack as cloud_pack
from training.curriculum_sampler import PERMUTATION_SAMPLER_ALGORITHM


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest(root: Path, tokenizer_hash: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    shards = []
    for index, source in enumerate(cloud_pack.PHASE_A_MIX):
        values = (np.arange(2048 * 80 + 1, dtype=np.uint32) + index) % 32_000
        path = root / f"{source}.npy"
        np.save(path, values.astype(np.uint16), allow_pickle=False)
        shards.append(
            {
                "path": path.name,
                "sha256": _sha(path),
                "tokens": len(values),
                "source_class": source,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "tokenizer_schema_version": 3,
                "tokenizer_version": "v4-32768",
                "tokenizer_sha256": tokenizer_hash,
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )


def test_cloud_pack_is_compact_mixed_and_immutable(tmp_path, monkeypatch) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer = tokenizer_dir / "tokenizer_v4_32k.json"
    tokenizer.write_text('{"schema_version":3}', encoding="utf-8")
    tokenizer.with_suffix(tokenizer.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "backend": "native_append_v4",
                "vocab_size": 32_768,
            }
        ),
        encoding="utf-8",
    )
    tokenizer_hash = _sha(tokenizer)
    source = tmp_path / "source"
    _source_manifest(source, tokenizer_hash)
    _source_manifest(source / "validation", tokenizer_hash)
    monkeypatch.setattr(cloud_pack, "ROOT", tmp_path)

    output = tmp_path / "pack"
    report = cloud_pack.build_cloud_pack(
        source_root=source,
        output_root=output,
        training_tokens=2048 * 64,
        validation_tokens=2048 * 32,
        block_size=2048,
        seed=1301,
    )
    train = json.loads((output / "train" / "manifest.json").read_text())

    assert report["training_tokens_effective"] == 2048 * 64
    assert train["sampling_policy"] == PERMUTATION_SAMPLER_ALGORITHM
    assert train["total_training_windows"] == 64
    assert train["campaign_mix_verified"] is True
    assert report["tokenizer_metadata_path"] == "tokenizer_v4_32k.json.meta.json"
    assert _sha(output / report["tokenizer_metadata_path"]) == report[
        "tokenizer_metadata_sha256"
    ]
    assert sum(item["training_windows"] for item in train["shards"]) == 64
    assert all(_sha(output / item["path"]) == item["sha256"] for item in report["files"])

    try:
        cloud_pack.build_cloud_pack(source_root=source, output_root=output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable cloud pack was overwritten")
