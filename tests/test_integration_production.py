"""Production integration tests - real modules, real PyTorch, no archived NumPy."""

from __future__ import annotations

import pathlib
import subprocess

import pytest
import torch

from anra.core.config import AnRaConfig
from anra.core.registry import MODEL_REGISTRY
from anra_brain import CausalTransformerV2


@pytest.fixture(scope="module")
def small_model():
    return CausalTransformerV2(
        vocab_size=256, n_embd=64, n_head=4, n_kv_head=2, n_layer=2, block_size=64
    )


def test_registry_builds_model():
    m = MODEL_REGISTRY.build(
        "causal_transformer_v2",
        vocab_size=256,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=64,
    )
    assert isinstance(m, CausalTransformerV2)


def test_forward_and_loss(small_model):
    idx = torch.randint(0, 256, (2, 32))
    tgt = torch.randint(0, 256, (2, 32))
    logits, loss = small_model(idx, targets=tgt)
    assert logits.shape == (2, 32, 256)
    assert loss is not None and not torch.isnan(loss)


def test_save_reload_produces_identical_output(small_model, tmp_path):
    ckpt = tmp_path / "m.pt"
    torch.save(small_model.state_dict(), ckpt)
    m2 = CausalTransformerV2(
        vocab_size=256, n_embd=64, n_head=4, n_kv_head=2, n_layer=2, block_size=64
    )
    m2.load_state_dict(torch.load(ckpt, weights_only=True))
    small_model.eval()
    m2.eval()
    idx = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        a, _ = small_model(idx)
        b, _ = m2(idx)
    torch.testing.assert_close(a, b)
    small_model.train()


def test_three_steps_reduce_loss():
    m = CausalTransformerV2(vocab_size=256, n_embd=64, n_head=4, n_layer=2, block_size=64)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    idx = torch.randint(0, 256, (2, 32))
    tgt = torch.randint(0, 256, (2, 32))
    prev = float("inf")
    for _ in range(3):
        opt.zero_grad()
        _, loss = m(idx, targets=tgt)
        loss.backward()
        opt.step()
        assert loss.item() < prev
        prev = loss.item()


def test_config_validates_model_fields(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "experiment_name: ci_test\nmodel:\n  type: causal_transformer_v2\n"
        "  vocab_size: 256\n  n_embd: 64\n  n_head: 4\n  n_layer: 2\n  block_size: 64\n"
        "training:\n  seq_len: 64\n"
    )
    cfg = AnRaConfig.from_yaml(cfg_path)
    assert cfg.model.vocab_size == 256
    assert cfg.model.n_head == 4


def test_no_archived_imports_in_live_tests(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    archived_imports = (
        "from decoder import",
        "from encoder import",
        "from model import LanguageModel",
    )
    for path in (root / "tests").rglob("*.py"):
        if path == pathlib.Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern in text for pattern in archived_imports):
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f"Archived imports in live tests:\n{chr(10).join(offenders)}"
