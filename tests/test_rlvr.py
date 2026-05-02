from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from training.rlvr import RLVRTask, RLVRTrainer


class TinyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [(ord(ch) % 15) + 1 for ch in text] or [1]

    def decode(self, ids: list[int]) -> str:
        return "x" * max(1, len(ids))


class TinyModel(torch.nn.Module):
    block_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)

    def forward(self, x):
        h = self.emb(x)
        return self.head(h), None


@dataclass
class Score:
    score: float


class Verifier:
    def __init__(self) -> None:
        self.values = [1.0, 0.3, 0.0, 0.0]
        self.i = 0

    def score(self, *args, **kwargs):
        value = self.values[self.i]
        self.i += 1
        return Score(value)


def test_rlvr_population_normalized_advantages_and_optimizer_step() -> None:
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = RLVRTrainer(model, TinyTokenizer(), optimizer, Verifier(), G=4)
    before = model.emb.weight.detach().clone()

    step = trainer.train_step(RLVRTask(prompt="abcd", task_type="unit"))

    assert step.advantages[0] > 1.67
    assert -0.07 < step.advantages[1] < -0.05
    assert -0.82 < step.advantages[2] < -0.79
    assert -0.82 < step.advantages[3] < -0.79
    assert not torch.equal(before, model.emb.weight.detach())
