import pytest
import torch

from anra_core.config import CANONICAL_CONFIG
from anra_core.errors import UnsupportedCapabilityError
from anra_core.executor import CoreExecutor
from anra_core.model import AnRaCore


def test_batched_stateful_decode() -> None:
    torch.manual_seed(888)
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    batch_size = 3
    seq_len = 8
    prompt_ids = torch.randint(0, CANONICAL_CONFIG.vocab_size, (batch_size, seq_len))

    # Prefill batched prompt
    state = executor.create_state(batch_size=batch_size)
    res_prefill = executor.prefill(prompt_ids, state=state)
    assert res_prefill.logits.shape == (batch_size, seq_len, 32_768)
    assert state.current_length == seq_len

    # Step batched tokens
    next_tokens = torch.randint(0, CANONICAL_CONFIG.vocab_size, (batch_size, 1))
    res_step = executor.forward_step(next_tokens, state=state)
    assert res_step.logits.shape == (batch_size, 1, 32_768)
    assert state.current_length == seq_len + 1


def test_chunked_prefill_equivalence() -> None:
    torch.manual_seed(777)
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    prompt = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 16))

    # 1. Unchunked prefill
    state_a = executor.create_state()
    res_a = executor.prefill(prompt, state=state_a)

    # 2. Chunked prefill with chunk_size = 4
    state_b = executor.create_state()
    res_b = executor.prefill(prompt, state=state_b, chunk_size=4)

    assert state_a.current_length == state_b.current_length == 16
    diff = (res_a.logits[:, -1, :] - res_b.logits[:, -1, :]).abs().max().item()
    assert diff < 5e-4


def test_state_serialization_is_fail_closed_until_a_portable_schema_exists() -> None:
    torch.manual_seed(666)
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    prompt = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 6))
    state = executor.create_state()
    _ = executor.prefill(prompt, state=state)

    with pytest.raises(UnsupportedCapabilityError):
        executor.serialize_state(state)
    assert not executor.capabilities.supports_state_serialization


def test_state_rollback_truncation() -> None:
    torch.manual_seed(555)
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    prefix_a = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 6))
    suffix_b = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 4))
    suffix_c = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 4))

    # Prefill prefix A + suffix B (length 10)
    state = executor.create_state()
    _ = executor.prefill(torch.cat([prefix_a, suffix_b], dim=1), state=state)
    assert state.current_length == 10

    # Roll back state to prefix length 6
    executor.rollback_state(state, target_length=6)
    assert state.current_length == 6

    # Step with alternative suffix C
    res_branched = executor.forward_step(suffix_c[:, :1], state=state)

    # Compare with clean prefix A stepped on suffix C
    clean_state = executor.create_state()
    _ = executor.prefill(prefix_a, state=clean_state)
    res_clean = executor.forward_step(suffix_c[:, :1], state=clean_state)

    diff = (res_branched.logits - res_clean.logits).abs().max().item()
    assert diff < 5e-4
    assert torch.equal(res_branched.logits.argmax(dim=-1), res_clean.logits.argmax(dim=-1))


def test_diagnostic_telemetry_probes() -> None:
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model, enable_telemetry=True)

    tokens = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 5))
    res = executor.forward(tokens)

    assert "telemetry" in res.metadata
    telem = res.metadata["telemetry"]
    assert "logit_entropy" in telem
    assert "peak_logit" in telem
    assert "top2_margin" in telem
    assert "min_logit" in telem
    assert telem["logit_entropy"] >= 0.0


def test_state_memory_bytes_calculation() -> None:
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    state = executor.create_state()
    assert state.memory_bytes() == 0

    tokens = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 8))
    _ = executor.prefill(tokens, state=state)

    mem_bytes = state.memory_bytes()
    # 18 layers * 2 tensors (k, v) * 1 batch * 2 kv_heads * 8 tokens * 64 head_dim * 4 bytes (fp32)
    # = 18 * 2 * 1 * 2 * 8 * 64 * 4 = 147,456 bytes
    expected_bytes = 18 * 2 * 1 * 2 * 8 * 64 * 4
    assert mem_bytes == expected_bytes
    descriptor = state.descriptor()
    assert descriptor["logical_memory_bytes"] == expected_bytes
    assert descriptor["reserved_memory_bytes"] >= expected_bytes
