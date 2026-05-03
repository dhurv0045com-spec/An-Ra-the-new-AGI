from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from identity.civ import CIVGuard


class Tok:
    def encode(self, text: str) -> list[int]:
        return [(ord(ch) % 20) + 1 for ch in text] or [1]


class Model(torch.nn.Module):
    block_size = 16
    n_embd = 8

    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(32, 8)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(2)])
        self.head = torch.nn.Linear(8, 32)

    def forward(self, x):
        h = self.emb(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h), None


def test_civ_similarity_drops_after_weight_drift(tmp_path: Path) -> None:
    identity = tmp_path / "identity.txt"
    identity.write_text("H: Who are you?\nH: What is your purpose?\n", encoding="utf-8")
    model = Model()
    guard = CIVGuard(model, Tok(), identity, layer_idx=0, threshold=0.99)
    baseline = guard.compute_baseline()
    assert baseline.shape == (8,)
    assert guard.verify()[1]
    with torch.no_grad():
        model.blocks[0].weight.mul_(-1)
    score, ok = guard.verify()
    assert score < 0.99
    assert not ok
