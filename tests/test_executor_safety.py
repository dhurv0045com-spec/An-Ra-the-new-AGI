"""High-value conformance tests for executor-owned state boundaries."""

from __future__ import annotations

import pytest
import torch

from anra_core.config import CoreConfig
from anra_core.errors import StateIncompatibleError, UnsupportedProfileError
from anra_core.executor import CoreExecutor
from anra_core.model import AnRaCore


def _config() -> CoreConfig:
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


def _executor(seed: int = 7) -> CoreExecutor:
    torch.manual_seed(seed)
    return CoreExecutor(AnRaCore(_config()).eval())


def test_state_is_bound_to_its_exact_executor_and_homogeneous_batch() -> None:
    first = _executor(7)
    second = _executor(8)
    state = first.create_state(batch_size=1)
    token = torch.tensor([[3]], dtype=torch.long)
    first.prefill(token, state=state)

    with pytest.raises(StateIncompatibleError, match="does not belong"):
        second.forward_step(token, state=state)
    with pytest.raises(StateIncompatibleError, match="batch"):
        first.forward(torch.tensor([[3], [4]], dtype=torch.long), state=state)


def test_invalid_profile_and_capacity_fail_without_fallback() -> None:
    model = AnRaCore(_config()).eval()
    with pytest.raises(UnsupportedProfileError):
        CoreExecutor(model, dtype="not-a-dtype")
    with pytest.raises(UnsupportedProfileError):
        CoreExecutor(model, dtype="float16", device="cpu")

    executor = _executor()
    with pytest.raises(StateIncompatibleError, match="capacity"):
        executor.create_state(capacity=17)


def test_failed_incremental_execution_does_not_commit_partial_state() -> None:
    executor = _executor()
    state = executor.create_state()
    prefix = torch.tensor([[2, 3, 4]], dtype=torch.long)
    executor.prefill(prefix, state=state)
    before = state.current_length
    original = executor.model.blocks[2].forward

    def fail(*args, **kwargs):
        raise RuntimeError("injected layer fault")

    executor.model.blocks[2].forward = fail  # type: ignore[method-assign]
    with pytest.raises(Exception):
        executor.forward_step(torch.tensor([[5]], dtype=torch.long), state=state)
    assert state.current_length == before

    executor.model.blocks[2].forward = original  # type: ignore[method-assign]
    retried = executor.forward_step(torch.tensor([[5]], dtype=torch.long), state=state)
    clean = executor.create_state()
    executor.prefill(prefix, state=clean)
    expected = executor.forward_step(torch.tensor([[5]], dtype=torch.long), state=clean)
    assert torch.allclose(retried.logits, expected.logits, atol=1e-5, rtol=0)


def test_chunked_prefill_preserves_all_logits_and_sliding_boundary() -> None:
    executor = _executor()
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    direct_state = executor.create_state()
    chunked_state = executor.create_state()
    direct = executor.prefill(tokens, state=direct_state)
    chunked = executor.prefill(tokens, state=chunked_state, chunk_size=3)
    assert chunked.logits.shape == direct.logits.shape == (1, 8, 128)
    assert torch.allclose(chunked.logits, direct.logits, atol=1e-5, rtol=0)
    assert torch.equal(chunked.logits.argmax(-1), direct.logits.argmax(-1))
