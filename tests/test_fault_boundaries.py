"""Regression tests for typed Core fault boundaries.

1. A torch.OutOfMemoryError raised while lazily allocating incremental state
   buffers must surface as ResourceExhaustionError, not a raw OOM.
2. An invalid rollback target length must surface as StateIncompatibleError,
   not a raw ValueError.
"""

import pytest
import torch

from anra_core.config import CANONICAL_CONFIG
from anra_core.errors import ResourceExhaustionError, StateIncompatibleError
from anra_core.executor import CoreExecutor
from anra_core.model import AnRaCore
from anra_core.state import CoreState


def _tiny_executor() -> CoreExecutor:
    return CoreExecutor(AnRaCore(CANONICAL_CONFIG).eval())


def test_state_allocation_oom_translates_to_resource_exhaustion(monkeypatch) -> None:
    executor = _tiny_executor()
    state = executor.create_state(capacity=8)

    def boom(*_args, **_kwargs) -> None:
        raise torch.OutOfMemoryError("CUDA out of memory during buffer alloc")

    monkeypatch.setattr(CoreState, "_ensure_buffers", boom)
    ids = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 2))
    with pytest.raises(ResourceExhaustionError):
        executor.prefill(ids, state=state)


def test_invalid_rollback_target_is_typed() -> None:
    executor = _tiny_executor()
    state = executor.create_state(capacity=8)
    ids = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 4))
    executor.prefill(ids, state=state)

    with pytest.raises(StateIncompatibleError):
        executor.rollback_state(state, state.current_length + 1)
    with pytest.raises(StateIncompatibleError):
        executor.rollback_state(state, -1)
    # Valid rollback still works and stays within contract.
    executor.rollback_state(state, 2)
    assert state.current_length == 2
