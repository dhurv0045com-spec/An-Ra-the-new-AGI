"""CPU smoke: frozen counterfactual pair, tiny packed V5-style core (HORM-001).

Control: hormonal projection inert (raw_alpha=0) -> scale exactly 1.0.
Treatment: same seed/data; nonzero hormonal state -> bounded scale != 1.0.
This is a wiring smoke, NOT capability evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import torch

from v5_identity import (
    HORMONES,
    HormonalProjection,
    HormonalState,
    V5A_250M_HORMONAL_V1,
)
from v5_model.attention import build_attention
from v5_model.block import build_block, build_rmsnorm
from v5_model.initialize import initialize_module


def build_tiny(spec_like, seed: int):
    torch.manual_seed(seed)
    config = SimpleNamespace(
        width=spec_like["width"], query_heads=spec_like["query_heads"],
        kv_heads=spec_like["kv_heads"], head_dimension=spec_like["head_dimension"],
        qk_norm=True, qk_norm_affine=True, qk_norm_epsilon=1e-6,
        rope_base=spec_like["rope_base"], norm_epsilon=1e-5,
        ffn_width=spec_like["ffn_width"], layers=spec_like["layers"],
    )
    embedding = torch.nn.Embedding(spec_like["vocabulary_size"], spec_like["width"])
    blocks = torch.nn.ModuleList(
        build_block(config, torch_module=torch) for _ in range(config.layers))
    final_norm = build_rmsnorm(
        config.width, epsilon=config.norm_epsilon, torch_module=torch)
    model = SimpleNamespace(
        embedding=embedding, blocks=blocks, final_norm=final_norm, config=config)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = embedding
            self.blocks = blocks
            self.final_norm = final_norm

        def forward(self, tokens, positions, mask):
            hidden = self.embedding(tokens)
            for block in self.blocks:
                hidden = block(hidden, positions, mask)
            return torch.nn.functional.linear(self.final_norm(hidden),
                                              self.embedding.weight)

    core = Tiny()
    initialize_module(core, layers=config.layers, torch_module=torch)
    return core


def packed(segments: torch.Tensor):
    length = segments.shape[1]
    indices = torch.arange(length)
    valid = segments >= 0
    starts = torch.ones_like(valid)
    starts[:, 1:] = segments[:, 1:] != segments[:, :-1]
    offsets = torch.where(starts, indices[None, :], 0).cummax(dim=1).values
    positions = torch.where(valid, indices[None, :] - offsets, 0)
    causal = indices[None, :] <= indices[:, None]
    same = segments[:, :, None] == segments[:, None, :]
    mask = causal & same & valid[:, :, None] & valid[:, None, :]
    mask = mask | ((~valid)[:, :, None] & torch.eye(length, dtype=torch.bool))
    return positions, mask[:, None, :, :]


def run_arm(scale_factor):
    torch.set_num_threads(2)
    spec_like = {"width": 32, "layers": 2, "query_heads": 4, "kv_heads": 2,
                 "head_dimension": 8, "ffn_width": 64, "vocabulary_size": 64,
                 "rope_base": 10000.0}
    torch.manual_seed(707)
    core = build_tiny(spec_like, seed=707).eval()
    tokens = torch.randint(0, spec_like["vocabulary_size"], (2, 10))
    segment_ids = torch.tensor(
        [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, -1, -1, -1, -1, -1, -1]])
    positions, mask = packed(segment_ids)

    state = HormonalState.baseline()
    if scale_factor != 0:
        for _ in range(3):
            state.appraise("failure")
            state.decay()
    projection = HormonalProjection(
        weights=tuple(0.1 if name in ("cortisol", "adrenaline") else 0.05
                      for name in HORMONES),
        bound=V5A_250M_HORMONAL_V1.bound,
        raw_alpha=0.0 if scale_factor == 0 else 0.35,
    )
    scale = projection.scale(state.vector())

    outputs = []
    with torch.no_grad():
        for arm_scale in (1.0, scale):
            saved = []
            for block in core.blocks:
                module = block.attention
                original = module.forward

                def patched(hidden, positions_, mask_, m=module, s=arm_scale, o=original):
                    return o(hidden, positions_, mask_) * s

                saved.append((module, original))
                module.forward = patched
            outputs.append(core(tokens, positions, mask))
            for module, original in saved:
                module.forward = original
    control, treatment = outputs
    difference = (control - treatment).abs().max().item()
    return {
        "scale": scale,
        "max_abs_difference": difference,
        "control_finite": bool(torch.isfinite(control).all()),
        "treatment_finite": bool(torch.isfinite(treatment).all()),
    }


def main() -> None:
    control = run_arm(scale_factor=0)
    treatment = run_arm(scale_factor=1)
    assert control["scale"] == 1.0
    assert control["max_abs_difference"] == 0.0, control
    assert treatment["scale"] != 1.0
    assert 0.8 <= treatment["scale"] <= 1.2
    assert treatment["max_abs_difference"] > 0.0
    assert treatment["max_abs_difference"] < 1.0
    assert control["control_finite"] and treatment["treatment_finite"]
    result = {
        "schema": "anra-horm-001-smoke/v1",
        "control": control,
        "treatment": treatment,
        "verdict": "WIRING_VERIFIED_PENDING_TRAINING_EVIDENCE",
        "claim_level": "wiring-only",
        "honesty_note": "tiny CPU smoke; no 250M training, no capability evidence",
    }
    payload = json.dumps(result, sort_keys=True, indent=2).encode("utf-8")
    result["sha256"] = hashlib.sha256(payload).hexdigest()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
