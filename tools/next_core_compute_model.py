#!/usr/bin/env python3
"""Deterministic parameter/compute/memory calculator for next-Core candidates.

The parameter accounting reproduces the live V5 contract formula
(``v5_contracts.model_spec.ModelSpec.parameter_receipt``) exactly, then
extends it with untied-output variants, FLOPs and memory estimates.

All estimates are analytic approximations documented in docs/cymek/next_core/
SCALING_PLAN.md; only the parameter counts are exact.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    vocabulary_size: int
    width: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dimension: int
    ffn_width: int
    context_length: int
    rope_base: float = 10_000.0
    tied_embeddings: bool = True
    qk_norm_affine: bool = True

    def __post_init__(self) -> None:
        if self.width != self.query_heads * self.head_dimension:
            raise ValueError("width must equal query_heads * head_dimension")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        if self.head_dimension % 2:
            raise ValueError("head dimension must be even for pairwise RoPE")


def parameter_receipt(g: Geometry) -> dict[str, int]:
    """Exact parameter accounting; identical to the live V5 contract when tied."""
    q_width = g.query_heads * g.head_dimension
    kv_width = g.kv_heads * g.head_dimension
    embedding = g.vocabulary_size * g.width
    attention = g.width * q_width + 2 * g.width * kv_width + q_width * g.width
    ffn = 3 * g.width * g.ffn_width
    block_norms = 2 * g.width
    qk_norms = (q_width + kv_width) if g.qk_norm_affine else 0
    block = attention + ffn + block_norms + qk_norms
    all_blocks = g.layers * block
    final_norm = g.width
    output_head = 0 if g.tied_embeddings else g.vocabulary_size * g.width
    total = embedding + all_blocks + final_norm + output_head
    return {
        "embedding": embedding,
        "attention_per_layer": attention,
        "ffn_per_layer": ffn,
        "block_norms_per_layer": block_norms,
        "qk_norms_per_layer": qk_norms,
        "block_total": block,
        "all_blocks": all_blocks,
        "final_norm": final_norm,
        "output_head": output_head,
        "total": total,
    }


def training_flops_per_token(g: Geometry) -> dict[str, int]:
    """Approximate FLOPs/token. Forward ~= 2 * non-embedding params + attention
    score terms (2 * L * ctx * w, causal halves ignored as a documented margin).
    Training ~= 3x forward (backward ~2x forward)."""
    r = parameter_receipt(g)
    non_embedding = r["all_blocks"] + r["final_norm"] + r["output_head"]
    embedding_lookup = 0  # lookup is a gather, not a matmul; projection counted via output_head
    attention_scores = 2 * g.layers * g.context_length * g.width
    forward = 2 * (non_embedding + embedding_lookup) + attention_scores + 2 * r["embedding"] * (
        1 if not g.tied_embeddings else 0
    )
    if g.tied_embeddings:
        # tied output projection: width x vocab matmul per token
        forward += 2 * g.vocabulary_size * g.width
    return {
        "forward_flops_per_token": forward,
        "backward_flops_per_token": 2 * forward,
        "training_flops_per_token": 3 * forward,
    }


def memory_bytes(g: Geometry) -> dict[str, int]:
    """Memory for the adopted precision layout: BF16 compute copy + FP32 master
    params + FP32 AdamW moments (m, v) + FP32 gradients transient."""
    r = parameter_receipt(g)
    p = r["total"]
    return {
        "bf16_compute_params": 2 * p,
        "fp32_master_params": 4 * p,
        "fp32_adam_moments": 8 * p,
        "fp32_gradients_transient": 4 * p,
        "optimizer_state_bytes": 8 * p,
        "checkpoint_bytes_params_plus_moments": 12 * p,
        "peak_training_bytes_approx": 18 * p,
    }


def scale_ladder() -> list[dict[str, object]]:
    """Evidence-driven ladder. Each rung: geometry, exact params, tokens at the
    Chinchilla-like ~20 tokens/param planning prior, training FLOPs, memory.
    An-Ra-specific modifiers are documented in SCALING_PLAN.md and do NOT come
    from these numbers."""
    rungs = []
    ladder = [
        # byte-codec rungs (V=260) mirror BRAMASTRA's measured profiles: a small
        # vocabulary is the only way to reach these scales at all
        ("byte-6M", dict(vocabulary_size=260, width=256, layers=6, query_heads=4, kv_heads=2, head_dimension=64, ffn_width=1_024, context_length=1_024)),
        ("byte-38M", dict(vocabulary_size=260, width=512, layers=13, query_heads=8, kv_heads=4, head_dimension=64, ffn_width=1_408, context_length=2_048)),
        # production-vocabulary rungs (V=24,576): embedding alone is ~22M params
        ("~40M", dict(vocabulary_size=24_576, width=512, layers=10, query_heads=8, kv_heads=4, head_dimension=64, ffn_width=1_408, context_length=2_048)),
        ("~97M", dict(vocabulary_size=24_576, width=768, layers=12, query_heads=12, kv_heads=6, head_dimension=64, ffn_width=2_048, context_length=4_096)),
        ("~250M (V5-A)", dict(vocabulary_size=24_576, width=896, layers=26, query_heads=14, kv_heads=7, head_dimension=64, ffn_width=2_368, context_length=4_096)),
        ("~500M", dict(vocabulary_size=24_576, width=1_024, layers=40, query_heads=16, kv_heads=8, head_dimension=64, ffn_width=2_816, context_length=4_096)),
    ]
    for name, dims in ladder:
        g = Geometry(**dims)
        r = parameter_receipt(g)
        f = training_flops_per_token(g)
        m = memory_bytes(g)
        tokens_at_20x = 20 * r["total"]
        rungs.append({
            "name": name,
            "geometry": dims,
            "parameters": r["total"],
            "tokens_at_chinchilla_20x": tokens_at_20x,
            "training_flops": tokens_at_20x * f["training_flops_per_token"],
            "peak_training_gb": round(m["peak_training_bytes_approx"] / 1e9, 2),
            "checkpoint_gb": round(m["checkpoint_bytes_params_plus_moments"] / 1e9, 2),
        })
    return rungs


V5A = Geometry(
    vocabulary_size=24_576, width=896, layers=26, query_heads=14, kv_heads=7,
    head_dimension=64, ffn_width=2_368, context_length=4_096,
)

# Task-3 canary rungs (Task-2 geometry family, exact counts):
#   RUNG_A: micro/integration canary — canonical 24,576 class space, head_dim 64,
#           GQA 2:1, V5 norms/SwiGLU/RoPE/tied output; smallest honest geometry (~10.2M).
#   RUNG_B: development GPU canary — ~42M, fits a single T4 with checkpoint/eval headroom
#           (peak training bytes ~18 B/param ~ 0.76 GiB + activations).
RUNG_A = Geometry(
    vocabulary_size=24_576, width=256, layers=4, query_heads=4, kv_heads=2,
    head_dimension=64, ffn_width=1_024, context_length=1_024,
)
RUNG_B = Geometry(
    vocabulary_size=24_576, width=512, layers=10, query_heads=8, kv_heads=4,
    head_dimension=64, ffn_width=1_408, context_length=2_048,
)


def validate_against_live_v5() -> dict[str, object]:
    """Reproduce the live V5-A receipt exactly (250,216,960). If v5_contracts is
    importable, cross-check against the contract itself."""
    ours = parameter_receipt(V5A)["total"]
    result: dict[str, object] = {"calculator_total": ours, "documented_total": 250_216_960}
    result["match_documented"] = ours == 250_216_960
    try:
        from v5_contracts.model_spec import V5A_250M  # type: ignore

        contract_total = V5A_250M.parameter_receipt().total
        result["contract_total"] = contract_total
        result["match_contract"] = contract_total == ours
    except Exception as exc:  # pragma: no cover - environment without repo path
        result["contract_check"] = f"skipped: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", action="store_true", help="print the scale ladder")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    payload = {
        "v5a_validation": validate_against_live_v5(),
        "ladder": scale_ladder() if args.ladder else None,
    }
    if args.json:
        print(json.dumps(payload, indent=1))
    else:
        print(json.dumps(payload["v5a_validation"], indent=1))
        if args.ladder:
            for rung in payload["ladder"]:
                print(f"{rung['name']:>14} params={rung['parameters']:>12,} "
                      f"tokens@20x={rung['tokens_at_chinchilla_20x']:.2e} "
                      f"peak={rung['peak_training_gb']}GB ckpt={rung['checkpoint_gb']}GB")
    return 0 if payload["v5a_validation"]["match_documented"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
