import json
from pathlib import Path

import pytest
import torch

from anra_core.checkpoint import (
    _historical_dormant_shape,
    _parameter_sha256,
    _validate_historical_dormant_tensors,
    _verify_tokenizer_contract,
    load_core_checkpoint,
)
from anra_core.config import CANONICAL_CONFIG, CoreConfig
from anra_core.contracts import CheckpointIdentity
from anra_core.errors import CheckpointIncompatibleError, RepresentationIncompatibleError
from anra_core.model import AnRaCore
from anra_core.tokenizer import V4Tokenizer

TOKENIZER_PATH = Path(__file__).parents[1] / "anra_core" / "assets" / "tokenizer_v4_32k.json"


def _tiny_v4_config() -> CoreConfig:
    return CoreConfig(
        d_model=8,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=8,
        d_ff=64,
        block_size=16,
        base_seq_len=16,
        target_seq_len=16,
        sliding_window=8,
        full_attention_every=1,
    )


def test_parameter_identity_is_container_independent_and_value_sensitive() -> None:
    tensors_a = {
        "second": torch.tensor([3.0], dtype=torch.float32),
        "first": torch.tensor([[1, 2]], dtype=torch.int64),
    }
    tensors_b = {"first": tensors_a["first"].clone(), "second": tensors_a["second"].clone()}
    assert _parameter_sha256(tensors_a) == _parameter_sha256(tensors_b)

    tensors_b["second"][0] = 4.0
    assert _parameter_sha256(tensors_a) != _parameter_sha256(tensors_b)


def test_exact_historical_dormant_tensor_inventory() -> None:
    state = {
        "mod_routers.4.gate.weight": torch.zeros((1, 896)),
        "mod_routers.4.capacity_control": torch.zeros(()),
        "mod_routers.4.context_weights": torch.zeros(3),
        "esv_module.predictor.0.weight": torch.zeros((3, 64)),
        "esv_module.predictor.0.bias": torch.zeros(3),
        "esv_module.state": torch.zeros(3),
        "rim_modules.17.raw_alpha": torch.zeros(()),
        "rim_modules.17.projection.parametrizations.weight.original": torch.zeros((896, 64)),
        "rim_modules.17.projection.parametrizations.weight.0._u": torch.zeros(896),
        "rim_modules.17.projection.parametrizations.weight.0._v": torch.zeros(64),
        "residual_depth_logits": torch.zeros(18),
        "dstp_temperature_log": torch.zeros(18),
        "layer_temperature_bias_log": torch.zeros(18),
    }
    assert _validate_historical_dormant_tensors(state, CANONICAL_CONFIG) == set(state)

    assert _historical_dormant_shape("mod_routers.3.gate.weight", CANONICAL_CONFIG) is None
    assert _historical_dormant_shape("rim_modules.18.raw_alpha", CANONICAL_CONFIG) is None
    assert _historical_dormant_shape("esv_module.unrecognized", CANONICAL_CONFIG) is None


def test_historical_dormant_tensor_shape_drift_is_rejected() -> None:
    with pytest.raises(CheckpointIncompatibleError, match="shape mismatch"):
        _validate_historical_dormant_tensors(
            {"rim_modules.0.projection.parametrizations.weight.original": torch.zeros((896, 63))},
            CANONICAL_CONFIG,
        )


def test_missing_tokenizer_contract_requires_explicit_legacy_mode() -> None:
    with pytest.raises(RepresentationIncompatibleError, match="missing its tokenizer contract"):
        _verify_tokenizer_contract({}, legacy_unverified=False)
    assert _verify_tokenizer_contract({}, legacy_unverified=True) == (False, False)


def test_canonical_tokenizer_contract_is_verified_and_mismatch_is_typed() -> None:
    tokenizer = V4Tokenizer.load_canonical()
    contract = {"available": True, **tokenizer.identity(probe_count=4)}
    assert _verify_tokenizer_contract(
        {"tokenizer_contract": contract}, legacy_unverified=False
    ) == (True, True)

    contract["vocabulary_sha256"] = "0" * 64
    with pytest.raises(RepresentationIncompatibleError, match="vocabulary_sha256"):
        _verify_tokenizer_contract(
            {"tokenizer_contract": contract}, legacy_unverified=False
        )


def test_inconsistent_token_mapping_uses_typed_representation_error() -> None:
    payload = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    meta = json.loads(
        TOKENIZER_PATH.with_suffix(".json.meta.json").read_text(encoding="utf-8")
    )
    first_token = payload["id_to_token"][0]
    payload["token_to_id"][first_token] = 123
    with pytest.raises(RepresentationIncompatibleError, match="mapping is inconsistent"):
        V4Tokenizer(payload, meta)


def test_checkpoint_identity_reports_verification_and_ignored_inventory() -> None:
    identity = CheckpointIdentity(
        checkpoint_sha256="file-hash",
        parameter_sha256="parameter-hash",
        source_path="checkpoint.pt",
        global_step=42,
        training_stage="pretrain",
        source_commit="commit",
        tokenizer_contract_valid=True,
        tokenizer_contract_present=True,
        tokenizer_contract_verified=True,
        ignored_tensor_names=("esv_module.state",),
    )
    payload = identity.to_dict()
    assert payload["checkpoint_sha256"] == "file-hash"
    assert payload["parameter_sha256"] == "parameter-hash"
    assert payload["tokenizer_contract_present"] is True
    assert payload["tokenizer_contract_verified"] is True
    assert payload["ignored_tensor_names"] == ("esv_module.state",)


def test_strict_and_legacy_loading_and_parameter_identity(tmp_path: Path) -> None:
    config = _tiny_v4_config()
    model_state = AnRaCore(config).state_dict()
    packaged_state = dict(model_state)
    packaged_state["esv_module.state"] = torch.zeros(3)
    contract = {"available": True, **V4Tokenizer.load_canonical().identity(probe_count=2)}

    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    raw_path = tmp_path / "raw.pt"
    unknown_path = tmp_path / "unknown.pt"
    torch.save(
        {"model_state_dict": packaged_state, "global_step": 1, "tokenizer_contract": contract,
         "checkpoint_artifact_class": "model_only", "checkpoint_schema_version": 1},
        first_path,
    )
    torch.save(
        {"model_state_dict": packaged_state, "global_step": 2, "tokenizer_contract": contract,
         "checkpoint_artifact_class": "model_only", "checkpoint_schema_version": 1},
        second_path,
    )
    torch.save(model_state, raw_path)
    unknown_state = dict(model_state)
    unknown_state["esv_module.unrecognized"] = torch.zeros(3)
    torch.save(
        {"model_state_dict": unknown_state, "tokenizer_contract": contract,
         "checkpoint_artifact_class": "model_only", "checkpoint_schema_version": 1},
        unknown_path,
    )

    _, _, first_identity = load_core_checkpoint(first_path, config=config)
    _, _, second_identity = load_core_checkpoint(second_path, config=config)
    assert first_identity.checkpoint_sha256 != second_identity.checkpoint_sha256
    assert first_identity.parameter_sha256 == second_identity.parameter_sha256
    assert first_identity.tokenizer_contract_present is True
    assert first_identity.tokenizer_contract_verified is True
    assert first_identity.ignored_tensor_names == ("esv_module.state",)
    assert first_identity.legacy_unverified is False

    with pytest.raises(CheckpointIncompatibleError, match="unknown checkpoint tensor"):
        load_core_checkpoint(unknown_path, config=config)

    with pytest.raises(CheckpointIncompatibleError, match="artifact class"):
        load_core_checkpoint(raw_path, config=config)
    _, _, raw_identity = load_core_checkpoint(
        raw_path, config=config, legacy_unverified=True
    )
    assert raw_identity.parameter_sha256 == first_identity.parameter_sha256
    assert raw_identity.tokenizer_contract_present is False
    assert raw_identity.tokenizer_contract_verified is False
    assert raw_identity.legacy_unverified is True
