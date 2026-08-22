"""Tests for the core-vnext training path: WSD schedule + pack verification."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training.pack_verify import PackVerificationError, build_manifest, verify_pack
from training.wsd_scheduler import (
    PackWsdSchedule,
    build_wsd_schedule,
    phase_for_step,
    steps_for_tokens,
    wsd_multiplier,
)


# --------------------------------------------------------------------------
# WSD schedule
# --------------------------------------------------------------------------


def test_wsd_warmup_rises_then_stable_then_decays() -> None:
    warmup, total = 20, 1_000
    assert wsd_multiplier(0, warmup_steps=warmup, total_steps=total) == 0.0
    assert wsd_multiplier(warmup // 2, warmup_steps=warmup, total_steps=total) == pytest.approx(0.5)
    assert wsd_multiplier(warmup, warmup_steps=warmup, total_steps=total) == 1.0
    assert wsd_multiplier(total // 2, warmup_steps=warmup, total_steps=total) == 1.0
    final = wsd_multiplier(total - 1, warmup_steps=warmup, total_steps=total)
    assert 0.09 <= final <= 0.11  # decays to min ratio ~0.1


def test_phase_reporting() -> None:
    assert phase_for_step(5, warmup_steps=20, total_steps=1_000).name == "warmup"
    assert phase_for_step(500, warmup_steps=20, total_steps=1_000).name == "stable"
    assert phase_for_step(950, warmup_steps=20, total_steps=1_000).name == "decay"
    assert phase_for_step(950, warmup_steps=20, total_steps=1_000).decay_started


def test_steps_for_tokens() -> None:
    assert steps_for_tokens(500_000_000, tokens_per_step=131_072) == 3_814
    with pytest.raises(ValueError):
        steps_for_tokens(100, tokens_per_step=0)


def test_schedule_integrates_with_optimizer() -> None:
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=2e-4)
    scheduler = build_wsd_schedule(optimizer, total_steps=100, warmup_steps=10)
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == 0.0
    assert lrs[50] == pytest.approx(2e-4)
    assert lrs[99] < lrs[50]  # decay engaged


def test_pack_schedule_round_trips_and_reaches_exact_floor() -> None:
    schedule = PackWsdSchedule(
        base_lr=2e-4,
        total_steps=100,
        warmup_steps=0,
        min_lr_ratio=0.1,
        decay_fraction=0.1,
    )
    restored = PackWsdSchedule.from_dict(schedule.to_dict())
    assert restored == schedule
    assert restored.lr_at(0) == pytest.approx(2e-4)
    assert restored.lr_at(99) == pytest.approx(2e-5)


# --------------------------------------------------------------------------
# Pack verification: fail-closed
# --------------------------------------------------------------------------


@pytest.fixture()
def pack_dir(tmp_path: Path) -> Path:
    root = tmp_path / "pack"
    (root / "train").mkdir(parents=True)
    rng = np.random.default_rng(0)
    shards = []
    total = 0
    for index in range(2):
        tokens = rng.integers(0, 32_768, size=5_000, dtype=np.int16)
        path = root / "train" / f"shard_{index:05d}.npy"
        np.save(path, tokens)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        shards.append({"file": f"train/shard_{index:05d}.npy", "tokens": 5_000, "sha256": digest})
        total += 5_000
    (root / "manifest.json").write_text(
        json.dumps({"schema": "anra-token-pack/v1", "block_size": 256, "total_tokens": total, "shards": shards}),
        encoding="utf-8",
    )
    return root


def test_verified_pack_passes(pack_dir: Path) -> None:
    pack = verify_pack(pack_dir)
    assert pack.total_tokens == 10_000
    assert pack.total_windows == 2 * ((5_000 - 1) // 256)
    assert pack.manifest_sha256 == hashlib.sha256(
        (pack_dir / "manifest.json").read_bytes()
    ).hexdigest()
    assert len(pack.shard_paths) == 2


def test_campaign_v3_pack_manifest_passes(pack_dir: Path) -> None:
    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("schema")
    manifest["schema_version"] = 3
    manifest["pack_schema_version"] = 1
    for shard in manifest["shards"]:
        shard["path"] = shard.pop("file")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pack = verify_pack(pack_dir)
    assert pack.total_tokens == 10_000
    assert pack.total_windows == 2 * ((5_000 - 1) // 256)


def test_manifest_cannot_escape_pack_root(pack_dir: Path) -> None:
    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["file"] = "../outside.npy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackVerificationError, match="escapes pack root"):
        verify_pack(pack_dir)


def test_missing_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(PackVerificationError, match="no manifest"):
        verify_pack(tmp_path)


def test_corrupted_shard_fails_closed(pack_dir: Path) -> None:
    shard = pack_dir / "train" / "shard_00000.npy"
    data = bytearray(shard.read_bytes())
    data[100] ^= 0xFF
    shard.write_bytes(bytes(data))
    with pytest.raises(PackVerificationError, match="hash mismatch"):
        verify_pack(pack_dir)


def test_missing_shard_fails_closed(pack_dir: Path) -> None:
    (pack_dir / "train" / "shard_00001.npy").unlink()
    with pytest.raises(PackVerificationError, match="missing"):
        verify_pack(pack_dir)


def test_wrong_schema_fails_closed(pack_dir: Path) -> None:
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema"] = "something/else"
    (pack_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackVerificationError, match="schema"):
        verify_pack(pack_dir)


def test_build_manifest_round_trips(pack_dir: Path) -> None:
    manifest = build_manifest(pack_dir, block_size=256)
    (pack_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pack = verify_pack(pack_dir)
    assert pack.total_tokens == 10_000


def test_shard_dataset_windows_align(tmp_path: Path) -> None:
    from training.train_xla import TokenShardDataset

    tokens = np.arange(1_000, dtype=np.int16)
    array_path = tmp_path / "shard.npy"
    np.save(array_path, tokens)
    dataset = TokenShardDataset(tmp_path, block_size=64)
    x, y = dataset[0]
    assert x.shape == (64,) and y.shape == (64,)
    assert int(x[0]) == 0 and int(y[0]) == 1
    assert int(x[-1]) == 63 and int(y[-1]) == 64


def test_deprecated_train_tpu_refuses_use() -> None:
    """The old trainer must be a hard failure, not a second production path."""
    import importlib

    with pytest.raises(RuntimeError, match="training.train_xla"):
        importlib.import_module("training.train_tpu").anything
