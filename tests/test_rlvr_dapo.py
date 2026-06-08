from __future__ import annotations

from dataclasses import dataclass

import torch

from training.replay_pipeline import ReplayPipeline
from training.rlvr import RLVRDapoConfig, RLVRTask, RLVRTrainer


class TinyTokenizer:
    special_ids = {"<eos>": 0}

    def __init__(self) -> None:
        self.vocab = {"<eos>": 0}

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in text.lower().split():
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab)
            ids.append(self.vocab[piece])
        return ids or [0]

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, "<unk>") for i in ids)


class TinyModel(torch.nn.Module):
    block_size = 32

    def __init__(self, vocab_size: int = 64) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 8)
        self.head = torch.nn.Linear(8, vocab_size)

    def forward(self, idx):
        return self.head(self.embed(idx)), None


@dataclass
class Score:
    score: float


class Verifier:
    def score(self, _task_type, **kwargs):
        response = kwargs.get("response", "")
        return Score(1.0 if "ok" in response else 0.1)


def _trainer(replay_pipeline=None, dapo_config=None) -> RLVRTrainer:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return RLVRTrainer(
        model,
        TinyTokenizer(),
        optimizer,
        Verifier(),
        G=2,
        entropy_bonus=0.0,
        replay_pipeline=replay_pipeline,
        replay_min_reward=0.8,
        dapo_config=dapo_config,
    )


def test_rlvr_dapo_step_logs_kl_lengths_and_pass_rate(tmp_path) -> None:
    trainer = _trainer(
        dapo_config=RLVRDapoConfig(
            overlong_penalty=0.2,
            overlong_token_limit=2,
            token_level_policy_loss=True,
        )
    )

    step = trainer.train_step(
        RLVRTask(prompt="solve", task_type="instruction", task_id="t1"),
        completions=["ok", "bad bad bad bad"],
    )

    assert step.output_lengths == [1, 4]
    assert step.verifier_pass_rate == 0.5
    assert step.rewards[1] < 0.1
    assert step.policy_loss != 0.0
    assert "kl_mean" in step.reward_stats

    out = tmp_path / "rlvr_report.json"
    report = trainer.write_last_step_report(out)

    assert out.exists()
    assert report["task_id"] == "t1"
    assert report["dapo_config"]["token_level_policy_loss"] is True


def test_rlvr_replay_metadata_records_provenance() -> None:
    replay = ReplayPipeline()
    trainer = _trainer(replay_pipeline=replay)

    step = trainer.train_step(
        RLVRTask(prompt="solve", task_type="math", task_id="rlvr-1"),
        completions=["ok", "bad"],
    )

    assert step.replay_additions >= 1
    assert len(replay.records) >= 1
    metadata = replay.records[0].metadata
    assert metadata["task_id"] == "rlvr-1"
    assert metadata["task_type"] == "math"
    assert metadata["source_detail"] == "rlvr_grpo_dapo"
    assert "advantage" in metadata
