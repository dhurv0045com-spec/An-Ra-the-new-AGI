"""100M-class configuration for the existing auditable V5 core.

This is a scale rung, not a new neural mechanism. Behavior, data, and target
hardware evidence remain separate launch gates.
"""

from __future__ import annotations

from dataclasses import asdict

from v5_contracts.model_spec import ModelSpec


MODEL_SPEC = ModelSpec(
    schema="anra-v5-model-spec/v1",
    family="dense-decoder-transformer",
    vocabulary_size=24_576,
    width=640,
    layers=20,
    query_heads=10,
    kv_heads=5,
    head_dimension=64,
    ffn_width=1_600,
    context_length=4_096,
    rope_base=10_000.0,
    norm_epsilon=1e-5,
    tied_embeddings=True,
    qk_norm=True,
    qk_norm_affine=True,
    linear_bias=False,
    dropout=0.0,
)

# A TPU-tiling challenger, not a promoted replacement. All large GEMM
# dimensions (768, 384, 2,176) align to 128-wide tiles, but this is shallower
# and has different attention cost; promotion requires matched target and
# semantic measurements against MODEL_SPEC.
TPU_TILED_CHALLENGER = ModelSpec(
    schema="anra-v5-model-spec/v1",
    family="dense-decoder-transformer",
    vocabulary_size=24_576,
    width=768,
    layers=12,
    query_heads=12,
    kv_heads=6,
    head_dimension=64,
    ffn_width=2_176,
    context_length=4_096,
    rope_base=10_000.0,
    norm_epsilon=1e-5,
    tied_embeddings=True,
    qk_norm=True,
    qk_norm_affine=True,
    linear_bias=False,
    dropout=0.0,
)

# Depth-preserving TPU efficiency challenger: only the SwiGLU width changes
# from M102, retaining the tokenizer, width, depth, heads, context and residual
# pathway. This is a candidate, not a promotion, until target and semantic
# comparisons are measured.
TPU_DEPTH_PRESERVING_CHALLENGER = ModelSpec(
    schema="anra-v5-model-spec/v1",
    family="dense-decoder-transformer",
    vocabulary_size=24_576,
    width=640,
    layers=20,
    query_heads=10,
    kv_heads=5,
    head_dimension=64,
    ffn_width=1_536,
    context_length=4_096,
    rope_base=10_000.0,
    norm_epsilon=1e-5,
    tied_embeddings=True,
    qk_norm=True,
    qk_norm_affine=True,
    linear_bias=False,
    dropout=0.0,
)


def parameter_receipt(spec: ModelSpec = MODEL_SPEC) -> dict[str, int]:
    """Return the V5 contract's exact receipt for this geometry."""

    return spec.parameter_receipt().as_dict()


def resource_estimate(spec: ModelSpec = MODEL_SPEC) -> dict[str, object]:
    """Return transparent analytic estimates; only parameters are exact."""

    receipt = parameter_receipt(spec)
    parameters = receipt["total"]
    return {
        "schema": "anra-signac-100m-resource-estimate/v1",
        "parameters_exact": parameters,
        "precision_assumption": "BF16 compute + FP32 master, Adam moments, transient gradients",
        "peak_training_bytes_approx": 18 * parameters,
        "checkpoint_bytes_params_plus_moments": 12 * parameters,
        "planning_tokens_at_20x_generic_prior": 20 * parameters,
        "attention_score_tensor_bytes_bf16_per_replica_per_active_layer": (
            spec.query_heads * spec.context_length * spec.context_length * 2
        ),
        "estimate_scope": "18 bytes/parameter assumes BF16 weights plus FP32 master weights, two Adam moments, and transient gradients.",
        "caveat": "Analytic component estimate, not total peak memory; excludes activations, XLA buffers, compiler padding, fragmentation, and data staging.",
        "geometry": asdict(spec),
        "receipt": receipt,
    }


def candidate_receipts() -> dict[str, dict[str, object]]:
    """Exact inventory and content identity for each candidate geometry."""

    return {
        "m102_primary": {
            "model_spec_sha256": MODEL_SPEC.sha256(),
            "resources": resource_estimate(MODEL_SPEC),
        },
        "tpu_tiled_challenger": {
            "model_spec_sha256": TPU_TILED_CHALLENGER.sha256(),
            "resources": resource_estimate(TPU_TILED_CHALLENGER),
        },
        "tpu_depth_preserving_challenger": {
            "model_spec_sha256": TPU_DEPTH_PRESERVING_CHALLENGER.sha256(),
            "resources": resource_estimate(TPU_DEPTH_PRESERVING_CHALLENGER),
        },
    }
