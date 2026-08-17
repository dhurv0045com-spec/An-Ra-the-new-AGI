import pytest
import torch

from anra_core.config import CANONICAL_CONFIG
from anra_core.errors import ContextOverflowError, StateReleasedError
from anra_core.executor import CoreExecutor
from anra_core.model import AnRaCore


def test_alternating_state_isolation() -> None:
    torch.manual_seed(303)
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    state_a = executor.create_state()
    state_b = executor.create_state()

    prompt_a = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 6))
    prompt_b = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 10))

    # Prefill both states
    res_a_prefill = executor.prefill(prompt_a, state=state_a)
    res_b_prefill = executor.prefill(prompt_b, state=state_b)

    # Step on A
    step_a1 = torch.tensor([[42]])
    res_a_step1 = executor.forward_step(step_a1, state=state_a)

    # Step on B multiple times
    step_b1 = torch.tensor([[100]])
    step_b2 = torch.tensor([[101]])
    _ = executor.forward_step(step_b1, state=state_b)
    res_b_step2 = executor.forward_step(step_b2, state=state_b)

    # Step on A again
    step_a2 = torch.tensor([[43]])
    res_a_step2 = executor.forward_step(step_a2, state=state_a)

    # Compare with clean single-stream execution of A
    clean_state_a = executor.create_state()
    _ = executor.prefill(prompt_a, state=clean_state_a)
    clean_res_a1 = executor.forward_step(step_a1, state=clean_state_a)
    clean_res_a2 = executor.forward_step(step_a2, state=clean_state_a)

    diff1 = (res_a_step1.logits - clean_res_a1.logits).abs().max().item()
    diff2 = (res_a_step2.logits - clean_res_a2.logits).abs().max().item()

    assert diff1 < 1e-5
    assert diff2 < 1e-5


def test_reset_and_release_semantics() -> None:
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    state_a = executor.create_state()
    state_b = executor.create_state()

    tokens = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 4))
    _ = executor.prefill(tokens, state=state_a)
    _ = executor.prefill(tokens, state=state_b)

    # Reset state A
    executor.reset_state(state_a)
    assert state_a.current_length == 0
    assert state_b.current_length == 4

    # Release state A
    executor.release_state(state_a)
    assert state_a.is_released
    assert not state_b.is_released

    # Operations on released state must fail with StateReleasedError
    with pytest.raises(StateReleasedError):
        executor.forward_step(torch.tensor([[10]]), state=state_a)

    # State B continues unaffected
    res_b = executor.forward_step(torch.tensor([[10]]), state=state_b)
    assert res_b.logits.shape == (1, 1, 32_768)


def test_state_forking() -> None:
    torch.manual_seed(404)
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    state_parent = executor.create_state()
    prompt = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 8))
    _ = executor.prefill(prompt, state=state_parent)

    # Fork state
    state_child = executor.fork_state(state_parent)
    assert state_child.state_id != state_parent.state_id
    assert state_child.current_length == state_parent.current_length

    # Step parent on token X, step child on token Y
    res_parent = executor.forward_step(torch.tensor([[111]]), state=state_parent)
    res_child = executor.forward_step(torch.tensor([[222]]), state=state_child)

    assert res_parent.logits.shape == (1, 1, 32_768)
    assert res_child.logits.shape == (1, 1, 32_768)


def test_context_overflow_rejection() -> None:
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    state = executor.create_state(capacity=10)
    tokens = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 11))

    with pytest.raises(ContextOverflowError):
        executor.prefill(tokens, state=state)
