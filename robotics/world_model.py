"""Lightweight uncertain action-conditioned world model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - canonical PyTorch alias


class PredictiveWorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 128,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.next_state = nn.Linear(hidden_dim, state_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        self.termination = nn.Linear(hidden_dim, 1)
        self.log_variance = nn.Linear(hidden_dim, state_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state_features = F.silu(self.state_encoder(state))
        action_features = F.silu(self.action_encoder(action))
        if hidden is None:
            hidden = torch.zeros_like(state_features)
        hidden = self.gru(torch.cat([state_features, action_features], dim=-1), hidden)
        return {
            "next_state": self.next_state(hidden),
            "reward": self.reward(hidden).squeeze(-1),
            "termination_probability": torch.sigmoid(self.termination(hidden)).squeeze(-1),
            "epistemic_uncertainty": F.softplus(self.log_variance(hidden)),
            "hidden": hidden,
        }

    @staticmethod
    def activation_allowed(
        *,
        simulation_transitions: int,
        held_out_accuracy: float,
        planning_improvement: float,
    ) -> bool:
        return (
            int(simulation_transitions) >= 100_000
            and held_out_accuracy >= 0.70
            and planning_improvement >= 0.10
        )


class WorldModelCodec:
    """Deterministic typed-state and skill encoders for the world model."""

    def __init__(self, state_dim: int = 256, action_dim: int = 128) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

    @staticmethod
    def _encode(payload: object, size: int) -> torch.Tensor:
        vector = torch.zeros(size, dtype=torch.float32)
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        for index in range(0, len(encoded), 8):
            digest = hashlib.blake2b(encoded[index : index + 8], digest_size=8).digest()
            position = int.from_bytes(digest, "little") % size
            vector[position] += 1.0 if digest[0] & 1 else -1.0
        norm = vector.norm().clamp_min(1.0)
        return vector / norm

    def encode_state(self, state: dict[str, object]) -> torch.Tensor:
        return self._encode(state, self.state_dim)

    def encode_action(self, action: object) -> torch.Tensor:
        return self._encode(action, self.action_dim)


@dataclass(frozen=True)
class PhysicalActuationDecision:
    allowed: bool
    owner_approved: bool
    emergency_stop_verified: bool
    supervised_hardware_validation: bool
    reasons: tuple[str, ...]


def evaluate_physical_actuation_promotion(
    *,
    owner_approved: bool,
    emergency_stop_verified: bool,
    supervised_hardware_validation: bool,
) -> PhysicalActuationDecision:
    checks = {
        "owner approval": bool(owner_approved),
        "emergency stop": bool(emergency_stop_verified),
        "supervised hardware validation": bool(supervised_hardware_validation),
    }
    return PhysicalActuationDecision(
        allowed=all(checks.values()),
        owner_approved=checks["owner approval"],
        emergency_stop_verified=checks["emergency stop"],
        supervised_hardware_validation=checks["supervised hardware validation"],
        reasons=tuple(name for name, passed in checks.items() if not passed),
    )


def train_world_model_offline(
    model: PredictiveWorldModel,
    transition_path: str | Path,
    optimizer: torch.optim.Optimizer,
    *,
    codec: WorldModelCodec | None = None,
    max_transitions: int | None = None,
) -> dict[str, float | int | bool]:
    """Train only from a closed transition file; workflow execution never calls this."""
    path = Path(transition_path)
    codec = codec or WorldModelCodec()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if max_transitions is not None and len(rows) >= int(max_transitions):
            break
    if not rows:
        raise ValueError("Offline world-model training requires transition records.")
    model.train()
    device = next(model.parameters()).device
    losses: list[float] = []
    for row in rows:
        state = codec.encode_state(dict(row["state"])).unsqueeze(0).to(device)
        action = codec.encode_action(row["action"]).unsqueeze(0).to(device)
        expected_next = codec.encode_state(dict(row["next_state"])).unsqueeze(0).to(device)
        expected_reward = torch.tensor(
            [float(row.get("reward", 0.0))],
            device=device,
        )
        expected_terminal = torch.tensor(
            [float(bool(row.get("terminal", False)))],
            device=device,
        )
        prediction = model(state, action)
        loss = (
            F.mse_loss(prediction["next_state"], expected_next)
            + F.mse_loss(prediction["reward"], expected_reward)
            + F.binary_cross_entropy(
                prediction["termination_probability"],
                expected_terminal,
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "transition_count": len(rows),
        "mean_loss": sum(losses) / len(losses),
        "offline_only": True,
    }
