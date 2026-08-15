from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anra.anra_paths import V4_TOKENIZER_FILE
from training.launch_manifest import build_launch_manifest


def _tokenizer_copy(tmp_path: Path) -> tuple[Path, str]:
    tokenizer = tmp_path / V4_TOKENIZER_FILE.name
    tokenizer.write_bytes(V4_TOKENIZER_FILE.read_bytes())
    source_metadata = V4_TOKENIZER_FILE.with_suffix(
        V4_TOKENIZER_FILE.suffix + ".meta.json"
    )
    tokenizer.with_suffix(tokenizer.suffix + ".meta.json").write_bytes(
        source_metadata.read_bytes()
    )
    return tokenizer, hashlib.sha256(tokenizer.read_bytes()).hexdigest()


def _base(tmp_path: Path) -> dict[str, object]:
    tokenizer, tokenizer_hash = _tokenizer_copy(tmp_path)
    return {
        "model_profile": "anra-v4-180m",
        "extension_profile": "none",
        "tokenizer_hash": tokenizer_hash,
        "tokenizer_path": str(tokenizer),
        "data_manifests": [],
        "data_manifest_roles": {},
        "stage": "foundation",
        "optimizer": "adamw",
        "batch_size": 1,
        "accumulation": 8,
        "schedule": {
            "kind": "cosine_with_warmup",
            "warmup_fraction": 0.02,
            "min_lr": 1e-5,
        },
        "seeds": [1301],
        "checkpoint_source": "scratch",
        "expected_tokens": 50_000_000,
        "runtime_estimate_hours": 3.0,
        "owner_authorized": True,
        "artifact_path": str(tmp_path / "full-resume.pt"),
    }


def test_v4_contract_rejects_noncanonical_seed(tmp_path: Path) -> None:
    values = _base(tmp_path)
    values["seeds"] = [42]
    with pytest.raises(ValueError, match="seed 1301"):
        build_launch_manifest(**values)


def test_v4_contract_rejects_old_tokenizer_schema(tmp_path: Path) -> None:
    values = _base(tmp_path)
    metadata_path = Path(str(values["tokenizer_path"]) + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="schema V4"):
        build_launch_manifest(**values)


def test_v4_contract_requires_a_full_resume_destination(tmp_path: Path) -> None:
    values = _base(tmp_path)
    values["artifact_destinations"] = [
        {
            "kind": "fp16_inference",
            "uri": str(tmp_path / "model-only.pt"),
            "required": True,
        }
    ]
    with pytest.raises(ValueError, match="mandatory full_resume"):
        build_launch_manifest(**values)


def test_v4_contract_binds_window_to_manifests_and_shards(tmp_path: Path) -> None:
    values = _base(tmp_path)
    values["shard_assignment"] = [3]
    values["token_window"] = {
        "start_token": 0,
        "end_token": 50_000_000,
        "data_manifest_hashes": {"unbound.json": "0" * 64},
        "shards": [3],
    }
    with pytest.raises(ValueError, match="data hashes"):
        build_launch_manifest(**values)


def test_v4_contract_caps_checkpoint_cadence(tmp_path: Path) -> None:
    values = _base(tmp_path)
    values["resource_limits"] = {"checkpoint_steps": 201}
    with pytest.raises(ValueError, match="at most 200"):
        build_launch_manifest(**values)


def test_v4_contract_rejects_legacy_stage_aliases(tmp_path: Path) -> None:
    values = _base(tmp_path)
    values["stage"] = "stage_a"
    with pytest.raises(ValueError, match="legacy Stage A-E labels are retired"):
        build_launch_manifest(**values)
