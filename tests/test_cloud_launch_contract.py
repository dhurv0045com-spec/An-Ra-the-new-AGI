from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.create_cloud_launch import create_cloud_launch
from training.train_unified import assert_launch_checkout


def test_continuation_pack_resets_only_its_local_sampler(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    metadata = tmp_path / "tokenizer.json.meta.json"
    train = tmp_path / "train-manifest.json"
    validation = tmp_path / "validation-manifest.json"
    tokenizer.write_bytes(b"v4-tokenizer")
    metadata.write_text("{}", encoding="utf-8")
    train.write_text("{}", encoding="utf-8")
    validation.write_text("{}", encoding="utf-8")
    parent = tmp_path / "parent.pt"
    torch.save({"continuation_token_counts": {"A": 175_000_000}}, parent)
    files = []
    for path in (tokenizer, metadata, train, validation):
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pack = {
        "builder_commit": "commit-1",
        "files": files,
        "tokenizer_path": tokenizer.name,
        "tokenizer_metadata_path": metadata.name,
        "train_manifest": train.name,
        "validation_manifest": validation.name,
        "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        "tokenizer_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "training_tokens_requested": 50_000_000,
        "cumulative_phase_tokens": 220_000_000,
        "seed": 1301,
    }
    (tmp_path / "pack_manifest.json").write_text(
        json.dumps(pack),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "scripts.create_cloud_launch.subprocess.check_output",
        lambda *_args, **_kwargs: "commit-1\n",
    )

    def capture_manifest(**kwargs):
        captured.update(kwargs)
        return dict(kwargs)

    monkeypatch.setattr(
        "scripts.create_cloud_launch.build_launch_manifest",
        capture_manifest,
    )
    monkeypatch.setattr(
        "scripts.create_cloud_launch.sign_manifest",
        lambda manifest, _output: manifest,
    )

    create_cloud_launch(
        pack_root=tmp_path,
        output=tmp_path / "launch.json",
        artifact_path=str(tmp_path / "child.pt"),
        checkpoint_source=str(parent),
        worker_id="worker-1",
        runtime_estimate_hours=3.0,
        batch_size=1,
        accumulation=8,
    )

    assert captured["allow_data_profile_change"] is True
    assert captured["reset_data_sampler"] is True
    assert captured["expected_tokens"] == 220_000_000
    assert captured["token_window"] == {
        "start_token": 175_000_000,
        "end_token": 220_000_000,
        "pack_sha256": hashlib.sha256(
            (tmp_path / "pack_manifest.json").read_bytes()
        ).hexdigest(),
    }
    assert captured["allow_data_profile_change"] is True
    assert captured["reset_data_sampler"] is True


def test_cloud_launch_accepts_verified_pack_from_historical_builder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    metadata = tmp_path / "tokenizer.json.meta.json"
    train = tmp_path / "train-manifest.json"
    validation = tmp_path / "validation-manifest.json"
    for path in (tokenizer, metadata, train, validation):
        path.write_text("{}", encoding="utf-8")
    parent = tmp_path / "parent.pt"
    torch.save({"continuation_token_counts": {"A": 10}}, parent)
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (tokenizer, metadata, train, validation)
    ]
    pack = {
        "builder_commit": "historical-builder",
        "files": files,
        "tokenizer_path": tokenizer.name,
        "tokenizer_metadata_path": metadata.name,
        "train_manifest": train.name,
        "validation_manifest": validation.name,
        "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        "tokenizer_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "training_tokens_requested": 100,
        "cumulative_phase_tokens": 100,
        "seed": 1301,
    }
    (tmp_path / "pack_manifest.json").write_text(json.dumps(pack), encoding="utf-8")

    monkeypatch.setattr(
        "scripts.create_cloud_launch.subprocess.check_output",
        lambda *_args, **_kwargs: "training-commit\n",
    )
    monkeypatch.setattr(
        "scripts.create_cloud_launch.build_launch_manifest",
        lambda **kwargs: dict(kwargs),
    )
    monkeypatch.setattr(
        "scripts.create_cloud_launch.sign_manifest",
        lambda manifest, _output: manifest,
    )

    result = create_cloud_launch(
        pack_root=tmp_path,
        output=tmp_path / "launch.json",
        artifact_path=str(tmp_path / "child.pt"),
        checkpoint_source=str(parent),
        worker_id="worker-1",
        runtime_estimate_hours=3.0,
        batch_size=1,
        accumulation=8,
    )

    assert result["data_pack_provenance"]["builder_commit"] == "historical-builder"
    assert result["data_pack_provenance"]["training_commit"] == "training-commit"
    assert result["token_window"]["start_token"] == 10


def test_signed_checkout_rejects_local_modifications(monkeypatch) -> None:
    clean_hash = hashlib.sha256(b"").hexdigest()

    def git_output(command, **_kwargs):
        return "commit-1\n" if command[1:3] == ["rev-parse", "HEAD"] else " M trainer.py\n"

    monkeypatch.setattr(
        "training.train_unified.subprocess.check_output",
        git_output,
    )

    with pytest.raises(RuntimeError, match="clean"):
        assert_launch_checkout(
            {"git_commit": "commit-1", "dirty_state_hash": clean_hash}
        )
