"""Cross-Scale Identity Inheritance for frontier-to-3B transfer."""

from __future__ import annotations

import torch
from torch.nn import functional as F


class CrossScaleIdentityInheritance:
    @staticmethod
    def transfer(source, target) -> dict[str, object]:
        copied: list[str] = []
        target.esv_module.predictor.load_state_dict(source.esv_module.predictor.state_dict())
        copied.append("esv_predictor")
        if hasattr(source, "dstp_logits") and hasattr(target, "dstp_logits"):
            values = source.dstp_logits.detach().view(1, 1, -1)
            interpolated = F.interpolate(
                values,
                size=target.dstp_logits.numel(),
                mode="linear",
                align_corners=True,
            ).view_as(target.dstp_logits)
            target.dstp_logits.data.copy_(interpolated)
            copied.append("dstp")
        return {"copied": copied, "source_layers": source.n_layer, "target_layers": target.n_layer}

    @staticmethod
    def alignment_loss(
        target_state: torch.Tensor,
        reference_state: torch.Tensor,
        *,
        step: int,
        warmup_steps: int = 5000,
    ) -> torch.Tensor:
        weight = max(0.0, 1.0 - float(step) / max(1, warmup_steps))
        return weight * F.mse_loss(target_state, reference_state.detach())
