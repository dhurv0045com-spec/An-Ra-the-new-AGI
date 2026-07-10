from __future__ import annotations

import numpy as np
import pytest

from phase3.ghost_memory_45p.ghost_memory.memory_store import _build_default_embedder


def test_default_ghost_embedder_is_offline_and_deterministic(monkeypatch) -> None:
    monkeypatch.delenv("ANRA_GHOST_EMBEDDER", raising=False)
    embed = _build_default_embedder("unused-external-model", 16)
    first = embed("same memory")
    second = embed("same memory")
    assert first.shape == (16,)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)


def test_unknown_ghost_embedder_provider_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("ANRA_GHOST_EMBEDDER", "mystery-provider")
    with pytest.raises(ValueError, match="unsupported"):
        _build_default_embedder("unused", 8)
