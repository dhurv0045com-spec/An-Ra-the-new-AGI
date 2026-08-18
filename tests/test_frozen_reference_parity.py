"""Conformance to the frozen f72 V4 mathematical reference, not just ourselves."""

from __future__ import annotations

import subprocess
import types
from pathlib import Path

import torch

from anra_core.config import CoreConfig
from anra_core.model import AnRaCore

ROOT = Path(__file__).parents[1]
FROZEN_REF = "f72f193:anra_core/model.py"


def _small_config() -> CoreConfig:
    return CoreConfig(
        vocab_size=128,
        d_model=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_ff=64,
        block_size=16,
        base_seq_len=16,
        target_seq_len=16,
        sliding_window=4,
        full_attention_every=2,
    )


def _frozen_model_module() -> types.ModuleType:
    source = subprocess.check_output(
        ["git", "show", FROZEN_REF], cwd=ROOT, text=True
    )
    module = types.ModuleType("anra_core._frozen_v4_reference")
    module.__package__ = "anra_core"
    exec(compile(source, FROZEN_REF, "exec"), module.__dict__)  # isolated reference source
    return module


def test_current_full_forward_matches_frozen_v4_reference_and_gradients() -> None:
    config = _small_config()
    torch.manual_seed(41)
    current = AnRaCore(config).train()
    reference_module = _frozen_model_module()
    reference = reference_module.AnRaCore(config).train()
    reference.load_state_dict(current.state_dict(), strict=True)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)

    current_logits = current(ids)
    reference_logits = reference(ids)
    assert torch.equal(current_logits, reference_logits)

    current_logits.square().mean().backward()
    reference_logits.square().mean().backward()
    for (name, current_parameter), (_, reference_parameter) in zip(
        current.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert current_parameter.grad is not None, name
        assert reference_parameter.grad is not None, name
        assert torch.equal(current_parameter.grad, reference_parameter.grad), name

