"""M4 imagination-before-action rollout API with an explicit calibration gate."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from robotics.world_model import PredictiveWorldModel, WorldModelCodec


def rollout_actions(
    model: PredictiveWorldModel,
    initial_state: dict[str, object],
    actions: Sequence[object],
    *,
    codec: WorldModelCodec | None = None,
    max_uncertainty: float = 1.0,
) -> dict[str, object]:
    """Predict offline action outcomes; never sends actions to an external system."""
    codec = codec or WorldModelCodec(
        state_dim=model.state_encoder.in_features,
        action_dim=model.action_encoder.in_features,
    )
    state = codec.encode_state(initial_state).unsqueeze(0)
    hidden = None
    steps = []
    with torch.no_grad():
        for action in actions:
            prediction = model(state, codec.encode_action(action).unsqueeze(0), hidden)
            uncertainty = float(prediction["epistemic_uncertainty"].mean())
            steps.append(
                {
                    "reward": float(prediction["reward"][0]),
                    "termination_probability": float(prediction["termination_probability"][0]),
                    "uncertainty": uncertainty,
                }
            )
            state, hidden = prediction["next_state"], prediction["hidden"]
    return {
        "offline_only": True,
        "steps": steps,
        "calibrated": bool(steps) and max(item["uncertainty"] for item in steps) <= max_uncertainty,
    }
