from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from training.star import STaRLoop


class Tok:
    def encode(self, text: str) -> list[int]:
        return [(ord(ch) % 20) + 1 for ch in text] or [1]

    def decode(self, ids: list[int]) -> str:
        return "<think>\nreason\n</think>\nAnswer: 42"


class Model(torch.nn.Module):
    block_size = 32

    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)

    def forward(self, x):
        return self.head(self.emb(x)), None


@dataclass
class Score:
    score: float


class Verifier:
    def score(self, *args, **kwargs):
        return Score(1.0 if kwargs.get("expected") == kwargs.get("response") else 0.0)


def test_star_extracts_answer_and_finetunes_chains() -> None:
    loop = STaRLoop(Model(), Tok(), Verifier())
    examples = loop.step("What is 40+2?", correct_answer="42", n_attempts=1)
    assert examples[0].answer == "42"
    assert loop.accepted
    losses = loop.finetune_on_chains(torch.optim.SGD(loop.model.parameters(), lr=0.01), n_steps=1)
    assert len(losses) == 1


def test_star_has_finetune():
    from training.star import STaRLoop

    assert hasattr(STaRLoop, "finetune_on_chains")


def test_star_extract_answer():
    from training.star import STaRLoop

    s = STaRLoop(model=None, tokenizer=None, verifier=None)
    assert s._extract_answer("<think>reasoning</think>42") == "42"
    assert s._extract_answer("no think tags\nanswer here") == "answer here"


def test_star_rationalization_fires(monkeypatch: pytest.MonkeyPatch):
    """When all attempts fail and correct_answer is given, rationalization runs."""
    from training.star import STaRLoop

    class _Tok:
        special_ids = {}

        def encode(self, text):
            return [1, 2, 3]

        def decode(self, ids):
            return "because the known answer can be explained</think>42"

    class _Verifier:
        def score(self, *args, **kwargs):
            return Score(0.0)

    loop = STaRLoop(model=None, tokenizer=_Tok(), verifier=_Verifier())
    monkeypatch.setattr(loop, "_generate_chain", lambda prompt: "<think>bad</think>wrong")
    monkeypatch.setattr(loop, "_generate_tokens", lambda ids: [4, 5])

    examples = loop.step("What is 40+2?", correct_answer="42", n_attempts=1)

    assert examples[-1].source == "rationalization"
    assert examples[-1].weight == 0.5
    assert loop.accepted[-1].answer == "42"
