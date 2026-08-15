from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from anra.extensions import (
    adapter_state_dict,
    attach_candidate_adapters,
    detach_candidate_adapters,
    load_capability_adapter,
    save_capability_adapter,
)
from inference.adapters import AdapterRegistry


def _model() -> nn.Sequential:
    torch.manual_seed(1301)
    return nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))


def test_capability_adapter_is_zero_init_reversible_and_strict(tmp_path: Path) -> None:
    model = _model()
    inputs = torch.randn(3, 4)
    baseline = model(inputs).detach().clone()
    attached = attach_candidate_adapters(
        model,
        rank=2,
        alpha=4.0,
        dora=False,
        target_modules=("0", "2"),
    )
    assert attached == ["0", "2"]
    assert torch.equal(model(inputs), baseline)
    assert all(
        "lora_" in name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    with torch.no_grad():
        for name, value in adapter_state_dict(model).items():
            if name.endswith("lora_b"):
                value.fill_(0.01)
                dict(model.named_parameters())[name].copy_(value)
    adapted = model(inputs).detach().clone()
    assert not torch.equal(adapted, baseline)

    artifact = tmp_path / "math-capability.pt"
    checkpoint_hash = "a" * 64
    tokenizer_hash = "b" * 64
    save_capability_adapter(
        model,
        artifact,
        capability_id="math-v1",
        base_model_profile="anra-v4-180m",
        base_checkpoint_sha256=checkpoint_hash,
        tokenizer_sha256=tokenizer_hash,
        source_commit="test",
    )
    assert detach_candidate_adapters(model) == ("0", "2")
    assert torch.equal(model(inputs), baseline)

    spec = load_capability_adapter(
        model,
        artifact,
        expected_base_model_profile="anra-v4-180m",
        expected_base_checkpoint_sha256=checkpoint_hash,
        expected_tokenizer_sha256=tokenizer_hash,
    )
    assert spec.capability_id == "math-v1"
    assert torch.equal(model(inputs), adapted)
    with pytest.raises(ValueError, match="tokenizer"):
        load_capability_adapter(
            model,
            artifact,
            expected_base_model_profile="anra-v4-180m",
            expected_base_checkpoint_sha256=checkpoint_hash,
            expected_tokenizer_sha256="c" * 64,
        )


def test_registry_activates_typed_capability_and_can_remove_it(tmp_path: Path) -> None:
    model = _model()
    attach_candidate_adapters(model, rank=2, target_modules=("0",))
    artifact = tmp_path / "adapter.pt"
    checkpoint_hash = "d" * 64
    tokenizer_hash = "e" * 64
    save_capability_adapter(
        model,
        artifact,
        capability_id="owner-style-v1",
        base_model_profile="anra-v4-180m",
        base_checkpoint_sha256=checkpoint_hash,
        tokenizer_sha256=tokenizer_hash,
        source_commit="test",
    )
    detach_candidate_adapters(model)
    registry = AdapterRegistry()
    registry.register(
        adapter_id="owner-style-v1",
        path=artifact,
        base_checkpoint_hash=checkpoint_hash,
        tokenizer_hash=tokenizer_hash,
    )
    spec = registry.activate_on_model(
        "owner-style-v1",
        model,
        base_model_profile="anra-v4-180m",
        base_checkpoint_hash=checkpoint_hash,
        tokenizer_hash=tokenizer_hash,
    )
    assert spec is not None and spec.target_modules == ("0",)
    assert registry.activate_on_model(
        None,
        model,
        base_model_profile="anra-v4-180m",
        base_checkpoint_hash=checkpoint_hash,
        tokenizer_hash=tokenizer_hash,
    ) is None
    assert not adapter_state_dict(model)


def test_dora_uses_normalized_weight_direction_and_trains_magnitude() -> None:
    model = _model()
    inputs = torch.randn(3, 4)
    baseline = model(inputs).detach().clone()
    attach_candidate_adapters(
        model,
        rank=2,
        alpha=4.0,
        dora=True,
        target_modules=("0",),
    )
    torch.testing.assert_close(model(inputs), baseline, rtol=1e-5, atol=1e-6)
    model(inputs).square().mean().backward()
    magnitude = dict(model.named_parameters())["0.magnitude"]
    assert magnitude.grad is not None
    assert torch.isfinite(magnitude.grad).all()
    assert torch.count_nonzero(magnitude.grad).item() > 0
    assert all(parameter.grad is None for parameter in model[0].base.parameters())


def test_failed_registry_load_does_not_claim_active_adapter(tmp_path: Path) -> None:
    artifact = tmp_path / "not-a-capability.bin"
    artifact.write_bytes(b"registered but not a typed capability")
    model = _model()
    checkpoint_hash = "f" * 64
    tokenizer_hash = "0" * 64
    registry = AdapterRegistry()
    registry.register(
        adapter_id="invalid",
        path=artifact,
        base_checkpoint_hash=checkpoint_hash,
        tokenizer_hash=tokenizer_hash,
    )
    with pytest.raises(FileNotFoundError):
        registry.activate_on_model(
            "invalid",
            model,
            base_model_profile="anra-v4-180m",
            base_checkpoint_hash=checkpoint_hash,
            tokenizer_hash=tokenizer_hash,
        )
    assert registry.provenance()["active_adapter_id"] is None


def test_adapter_parameters_inherit_base_device_and_dtype() -> None:
    model = _model().to(dtype=torch.float64)
    attach_candidate_adapters(model, rank=2, dora=True, target_modules=("0",))
    for name, parameter in model[0].named_parameters():
        if name.startswith(("lora_", "magnitude")):
            assert parameter.device == model[0].base.weight.device
            assert parameter.dtype == model[0].base.weight.dtype


def test_invalid_adapter_recipe_does_not_freeze_or_modify_model() -> None:
    model = _model()
    before = tuple(model.modules())
    before_trainability = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    with pytest.raises(ValueError, match="rank and alpha"):
        attach_candidate_adapters(model, rank=0, target_modules=("0",))
    assert tuple(model.modules()) == before
    assert {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    } == before_trainability
