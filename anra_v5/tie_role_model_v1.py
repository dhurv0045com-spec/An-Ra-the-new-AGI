"""Forward-equivalent tied input/output gradient-role treatments.

Every tied arm has byte-identical parameters and numerically identical forward
values at initialization for a matched seed. Only autograd's contribution from
the input-embedding path and output-projection path is rescaled.
"""
from __future__ import annotations

import types
from typing import Any

from anra_v5.formation_mux_model_v2 import PHYSICAL_VOCAB, SHARED_VOCAB
from v5_contracts.model_spec import ModelSpec
from v5_experiments import tie_role_protocol_v1 as proto

ACTIVE_ROWS = tuple(range(SHARED_VOCAB))
EXTRA_ROWS = tuple(range(SHARED_VOCAB, PHYSICAL_VOCAB))
ARMS = (*proto.ARMS_A, *proto.ARMS_B)


def spec() -> ModelSpec:
    return ModelSpec(
        schema="anra-v5-model-spec/v1",
        family="dense-decoder-transformer",
        vocabulary_size=PHYSICAL_VOCAB,
        width=256,
        layers=8,
        query_heads=4,
        kv_heads=2,
        head_dimension=64,
        ffn_width=1024,
        context_length=1024,
        rope_base=10_000.0,
        norm_epsilon=1e-5,
        tied_embeddings=True,
        qk_norm=True,
        qk_norm_affine=True,
        linear_bias=False,
        dropout=0.0,
    )


def gradient_scaled_view(weight: Any, scale: float) -> Any:
    """Return the same forward value while multiplying d(output)/d(weight)."""
    scale = float(scale)
    return weight.detach() + scale * (weight - weight.detach())


def set_gradient_scales(model: Any, *, input_scale: float, output_scale: float) -> None:
    model._tie_role_input_scale = float(input_scale)
    model._tie_role_output_scale = float(output_scale)


def get_gradient_scales(model: Any) -> tuple[float, float]:
    return float(model._tie_role_input_scale), float(model._tie_role_output_scale)


def _patch_forward(model: Any, *, torch: Any) -> None:
    functional = torch.nn.functional

    def forward(self, token_ids, positions, mask, use_activation_checkpointing: bool = False):
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.config.context_length:
            raise ValueError("token ids must be [batch, length] within native context")
        input_weight = gradient_scaled_view(self.embedding.weight, self._tie_role_input_scale)
        hidden = functional.embedding(token_ids, input_weight)
        for block in self.blocks:
            if use_activation_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    block, hidden, positions, mask, use_reentrant=False
                )
            else:
                hidden = block(hidden, positions, mask)
        output_weight = gradient_scaled_view(self.embedding.weight, self._tie_role_output_scale)
        return functional.linear(self.final_norm(hidden), output_weight)

    model.forward = types.MethodType(forward, model)


def build_model(seed: int, arm: str, *, torch: Any, device: Any) -> Any:
    from v5_model.core import initialize

    if arm not in ARMS:
        raise ValueError(f"unknown TIE-ROLE arm {arm}")
    model = initialize(spec(), int(seed), torch_module=torch).to(device)
    _patch_forward(model, torch=torch)
    input_scale, output_scale = proto.gradient_scales(arm)
    set_gradient_scales(model, input_scale=input_scale, output_scale=output_scale)
    model._tie_role_arm = arm
    return model


def make_optimizers(model: Any, arm: str, *, torch: Any, lr: float) -> dict[str, Any]:
    from v5_training.optimizer import build_adamw_optimizer

    if arm not in ARMS:
        raise ValueError(f"unknown TIE-ROLE arm {arm}")
    main = build_adamw_optimizer(model, torch_module=torch, lr=float(lr))
    return {"main": main, "rows": None, "view": main}


def mask_frozen_gradients_before_clip(model: Any, arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown TIE-ROLE arm {arm}")
    return None


def assert_latent_ids_are_shared(ids: list[int] | tuple[int, ...]) -> None:
    if ids and (min(ids) < 0 or max(ids) >= SHARED_VOCAB):
        raise RuntimeError("latent scientific token escaped shared 0..4095 region")
