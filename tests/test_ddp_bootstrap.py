from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from training.checkpoint_durability import build_checkpoint_lineage
from training.ddp_bootstrap import (
    RESTART_CONTRACT,
    create_bootstrap_manifest,
    file_bindings,
    load_and_verify_bootstrap_manifest,
    load_parent_model_and_progress,
    validate_runtime_contract,
)
from training.distributed import canonical_training_ddp_contract


def _parent_checkpoint(path: Path, model: torch.nn.Module) -> dict[str, object]:
    model_config = {"architecture_version": "test-v4", "width": 4}
    tokenizer = {"sha256": "a" * 64, "vocab_size": 8}
    payload: dict[str, object] = {
        "checkpoint_schema_version": 9,
        "checkpoint_artifact_class": "full_resume",
        "completed_optimizer_boundary": True,
        "accum_micro_steps": 0,
        "lineage_id_exact": "single-gpu-parent",
        "source_commit": "1" * 40,
        "model": model.state_dict(),
        "optimizer": {"secret_parent_optimizer": torch.tensor([91])},
        "scheduler": {"last_epoch": 17},
        "scaler": {"scale": 1.0},
        "rng_states": {"torch": torch.tensor([7])},
        "model_config": model_config,
        "tokenizer_contract": tokenizer,
        "dataset_manifest_hashes": {"train.json": "b" * 64},
        "data_profile": "v4-test",
        "training_data_layout": "raw_causal_shards_v1",
        "training_recipe": {"model_profile": "anra-v4-180m"},
        "seed_contract": {"seed": 1301},
        "global_step": 10_400,
        "tokens_seen": 181_000_000,
        "sessions_completed": 12,
        "continuation_token_counts": {"A": 181_000_000},
        "raw_window_consumption": {"unique_windows": 9, "repeated_windows": 0},
        "data_sampler_state": {
            "schema_version": 1,
            "algorithm": "deterministic_global_permutation_v1",
            "seed": 1301,
            "position": 9,
            "num_samples": 100,
            "curriculum": "none",
            "dataset_size": 100,
        },
        "unique_token_ids_seen": [1, 2, 3],
        "best_loss": 2.5,
        "best_validation_loss": 2.7,
        "best_answer_validation_loss": 2.8,
        "validation_history": [{"step": 10_000, "loss": 2.7}],
    }
    payload["checkpoint_lineage"] = build_checkpoint_lineage(payload)
    torch.save(payload, path)
    return payload


def _create(tmp_path: Path) -> tuple[dict[str, object], Path, Path, dict[str, object]]:
    parent_model = torch.nn.Linear(4, 4)
    parent = tmp_path / "parent.pt"
    parent_payload = _parent_checkpoint(parent, parent_model)
    train_manifest = tmp_path / "train.json"
    validation_manifest = tmp_path / "validation.json"
    train_manifest.write_text('{"train": true}', encoding="utf-8")
    validation_manifest.write_text('{"validation": true}', encoding="utf-8")
    bindings = file_bindings({"training": train_manifest, "validation": validation_manifest})
    contract = canonical_training_ddp_contract(
        backend="nccl",
        world_size=2,
        micro_batch_size_per_rank=1,
        gradient_accumulation=8,
        visible_device_order="0,1",
    )
    manifest_path = tmp_path / "bootstrap.json"
    child = tmp_path / "child.pt"
    create_bootstrap_manifest(
        parent_checkpoint=parent,
        child_checkpoint=child,
        output_manifest=manifest_path,
        child_lineage_id="ddp-child-10400",
        target_source_commit="2" * 40,
        target_ddp_contract=contract,
        target_data_bindings=bindings,
        seed=1301,
        signing_key="test-owner-key",
    )
    manifest = load_and_verify_bootstrap_manifest(
        manifest_path, signing_key="test-owner-key"
    )
    validate_runtime_contract(
        manifest,
        parent_checkpoint=parent,
        child_checkpoint=child,
        source_commit="2" * 40,
        ddp_contract=contract,
        model_config=parent_payload["model_config"],
        tokenizer_contract=parent_payload["tokenizer_contract"],
        data_bindings=bindings,
        seed=1301,
    )
    return manifest, parent, child, parent_payload


def test_signed_bootstrap_loads_only_model_and_whitelisted_progress(tmp_path: Path) -> None:
    manifest, parent, _child, parent_payload = _create(tmp_path)
    target = torch.nn.Linear(4, 4)
    result = load_parent_model_and_progress(manifest, parent, target)

    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, parent_payload["model"][name])
    assert result["progress"]["global_step"] == 10_400
    assert result["progress"]["tokens_seen"] == 181_000_000
    assert "optimizer" not in result
    assert "rng_states" not in result
    assert result["provenance"]["restart"] == RESTART_CONTRACT
    assert result["provenance"]["child_lineage_id"] == "ddp-child-10400"


def test_bootstrap_fails_closed_on_signature_parent_and_runtime_mismatch(
    tmp_path: Path,
) -> None:
    manifest, parent, child, parent_payload = _create(tmp_path)
    manifest_path = tmp_path / "bootstrap.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["child"]["seed"] = 9
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PermissionError, match="body hash"):
        load_and_verify_bootstrap_manifest(manifest_path, signing_key="test-owner-key")

    with parent.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_runtime_contract(
            manifest,
            parent_checkpoint=parent,
            child_checkpoint=child,
            source_commit="2" * 40,
            ddp_contract=dict(manifest["child"]["ddp_contract"]),
            model_config=parent_payload["model_config"],
            tokenizer_contract=parent_payload["tokenizer_contract"],
            data_bindings=dict(manifest["child"]["data_bindings"]),
            seed=1301,
        )


def test_bootstrap_refuses_overwrite_and_topology_drift(tmp_path: Path) -> None:
    manifest, parent, child, parent_payload = _create(tmp_path)
    bindings = dict(manifest["child"]["data_bindings"])
    wrong_contract = dict(manifest["child"]["ddp_contract"])
    wrong_contract["world_size"] = 4
    with pytest.raises(RuntimeError, match="DDP contract"):
        validate_runtime_contract(
            manifest,
            parent_checkpoint=parent,
            child_checkpoint=child,
            source_commit="2" * 40,
            ddp_contract=wrong_contract,
            model_config=parent_payload["model_config"],
            tokenizer_contract=parent_payload["tokenizer_contract"],
            data_bindings=bindings,
            seed=1301,
        )
    child.write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="new and non-overwriting"):
        validate_runtime_contract(
            manifest,
            parent_checkpoint=parent,
            child_checkpoint=child,
            source_commit="2" * 40,
            ddp_contract=dict(manifest["child"]["ddp_contract"]),
            model_config=parent_payload["model_config"],
            tokenizer_contract=parent_payload["tokenizer_contract"],
            data_bindings=bindings,
            seed=1301,
        )
