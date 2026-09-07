"""Bounded CPU/CUDA training with full optimizer and sampler continuation."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

from .curriculum import Demonstrations, demonstration_digest
from .learner import Investigator, LearnerConfig
from .worlds import inputs


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 800
    batch_size: int = 64
    learning_rate: float = 0.001
    policy_weight: float = 0.25
    seed: int = 701

    def __post_init__(self):
        if self.steps <= 0 or self.batch_size <= 0 or not 0 < self.learning_rate < 1:
            raise ValueError("invalid training budget or learning rate")
        if not 0 <= self.policy_weight <= 100:
            raise ValueError("invalid policy weight")


def fingerprint(model: Investigator) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Trainer:
    def __init__(self, model: Investigator, data: Demonstrations, config: TrainConfig):
        actual_digest = demonstration_digest(
            (data.observations, data.lengths, data.targets, data.labels, data.legal, data.gains)
        )
        if data.manifest.get("tensor_sha256") != actual_digest:
            raise ValueError("demonstration tensor/manifest mismatch")
        self.model, self.data, self.config = model, data, config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.sampler = torch.Generator().manual_seed(config.seed + 1000)
        self.step = 0
        self.candidates = torch.tensor(inputs(model.config.bits), dtype=torch.float32)

    def update(self) -> dict:
        self.model.train()
        ix = torch.randint(len(self.data), (self.config.batch_size,), generator=self.sampler)
        device = next(self.model.parameters()).device
        def batch(name):
            return getattr(self.data, name)[ix].to(device)
        self.optimizer.zero_grad(set_to_none=True)
        state = self.model.encode(batch("observations"), batch("lengths"))
        prediction = self.model.predict(state, batch("targets"))
        prediction_loss = F.binary_cross_entropy_with_logits(prediction, batch("labels"))
        scores = self.model.select(state, batch("targets"), self.candidates.to(device), batch("legal"))
        gains = batch("gains")
        informative = gains.max(dim=1).values > 1e-7
        # Tie-aware imitation: uniform mass over maximally informative queries.
        best = ((gains - gains.max(dim=1, keepdim=True).values).abs() < 1e-7) & batch("legal")
        distribution = best.float() / best.sum(dim=1, keepdim=True)
        per_row = -(distribution * F.log_softmax(scores, dim=-1)).sum(dim=-1)
        policy_loss = (per_row * informative).sum() / informative.sum().clamp_min(1)
        loss = prediction_loss + self.config.policy_weight * policy_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        return {"step": self.step, "loss": float(loss.detach()),
                "prediction_loss": float(prediction_loss.detach()), "policy_loss": float(policy_loss.detach()),
                "informative_fraction": float(informative.float().mean()), "grad_norm": float(norm)}

    def save(self, path: Path) -> None:
        payload = {"schema": "bramastra-discovery-checkpoint/v1", "model_config": asdict(self.model.config),
                   "train_config": asdict(self.config), "model": self.model.state_dict(),
                   "optimizer": self.optimizer.state_dict(), "sampler": self.sampler.get_state(),
                   "torch_rng": torch.get_rng_state(), "step": self.step,
                   "data_manifest": self.data.manifest}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def restore(cls, path: Path, data: Demonstrations) -> "Trainer":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["schema"] != "bramastra-discovery-checkpoint/v1" or payload["data_manifest"] != data.manifest:
            raise ValueError("checkpoint schema/data mismatch")
        model = Investigator(LearnerConfig(**payload["model_config"]))
        model.load_state_dict(payload["model"])
        trainer = cls(model, data, TrainConfig(**payload["train_config"]))
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.sampler.set_state(payload["sampler"])
        trainer.step = payload["step"]
        torch.set_rng_state(payload["torch_rng"])
        return trainer


def continuation_probe(trainer: Trainer, path: Path) -> dict:
    """Check next-update equality without changing the reported trained model."""
    if next(trainer.model.parameters()).device.type != "cpu":
        return {"status": "NOT_RUN", "reason": "probe currently supports CPU only"}
    trainer.save(path)
    left = Trainer.restore(path, trainer.data)
    right = Trainer.restore(path, trainer.data)
    # One continuation uses live pre-save state, the other deserialized state.
    left.model.load_state_dict(copy.deepcopy(trainer.model.state_dict()))
    left.optimizer.load_state_dict(copy.deepcopy(trainer.optimizer.state_dict()))
    left.sampler.set_state(trainer.sampler.get_state())
    a, b = left.update(), right.update()
    equal = fingerprint(left.model) == fingerprint(right.model) and a == b
    for p, q in zip(left.optimizer.state.values(), right.optimizer.state.values()):
        equal = equal and all(torch.equal(p[k], q[k]) if torch.is_tensor(p[k]) else p[k] == q[k] for k in p)
    if not equal:
        raise AssertionError("checkpoint continuation diverged")
    return {"status": "PASS", "scope": "local CPU, same process; optimizer and sampler restored",
            "next_step": left.step, "parameter_sha256": fingerprint(left.model)}
