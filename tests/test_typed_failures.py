import pytest
import torch

from anra_core.checkpoint import load_core_checkpoint
from anra_core.config import CANONICAL_CONFIG
from anra_core.errors import (
    CheckpointIncompatibleError,
    StateIncompatibleError,
)
from anra_core.executor import CoreExecutor
from anra_core.model import AnRaCore


def test_corrupted_checkpoint_rejection(tmp_path) -> None:
    bad_ckpt = tmp_path / "bad.pt"
    torch.save({"invalid_key": "not_a_state_dict"}, bad_ckpt)

    with pytest.raises(CheckpointIncompatibleError):
        load_core_checkpoint(bad_ckpt)


def test_missing_dense_tensors_rejection(tmp_path) -> None:
    partial_ckpt = tmp_path / "partial.pt"
    torch.save({"model_state_dict": {"token_embedding_table.weight": torch.randn(32768, 896)}}, partial_ckpt)

    with pytest.raises(CheckpointIncompatibleError):
        load_core_checkpoint(partial_ckpt)


def test_state_architecture_mismatch() -> None:
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    other_executor = CoreExecutor(AnRaCore(CANONICAL_CONFIG).eval())
    incompatible_state = other_executor.create_state()

    with pytest.raises(StateIncompatibleError):
        executor.forward_step(10, state=incompatible_state)


def test_double_prefill_rejection() -> None:
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    state = executor.create_state()
    tokens = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 5))
    _ = executor.prefill(tokens, state=state)

    with pytest.raises(StateIncompatibleError):
        executor.prefill(tokens, state=state)
