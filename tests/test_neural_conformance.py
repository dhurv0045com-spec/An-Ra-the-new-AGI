from pathlib import Path
import pytest
import torch

from anra_core.config import CANONICAL_CONFIG
from anra_core.model import AnRaCore


def test_parameter_count_and_ties() -> None:
    model = AnRaCore(CANONICAL_CONFIG)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 180_093_312
    assert model.lm_head.weight.data_ptr() == model.token_embedding_table.weight.data_ptr()


def test_attention_schedule() -> None:
    model = AnRaCore(CANONICAL_CONFIG)
    full_attn = [i for i, b in enumerate(model.blocks) if b.attn.full_attention]
    sliding_attn = [i for i, b in enumerate(model.blocks) if not b.attn.full_attention]
    assert full_attn == [3, 7, 11, 15]
    assert len(sliding_attn) == 14


def test_full_forward_execution() -> None:
    model = AnRaCore(CANONICAL_CONFIG).eval()
    tokens = torch.randint(0, CANONICAL_CONFIG.vocab_size, (2, 32))
    with torch.no_grad():
        logits = model(tokens)
    assert logits.shape == (2, 32, 32_768)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
