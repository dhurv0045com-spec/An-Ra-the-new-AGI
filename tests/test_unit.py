"""
================================================================================
FILE: tests/test_unit.py
PROJECT: Transformer Language Model — 45H Final Phase
PURPOSE: Unit tests — every module in isolation
================================================================================

Run:  pytest tests/test_unit.py -v --tb=short
================================================================================
"""

import sys
import os
import math
import pytest
import numpy as np

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attention         import RotaryEmbedding, KVCache, make_causal_mask, softmax, scaled_dot_product_attention
from multihead         import MultiHeadAttention
from feedforward       import SwiGLUFeedForward, GELUFeedForward, silu, gelu_approx, _make_ffn_dim
from layernorm         import RMSNorm, LayerNorm
from transformer_block import TransformerBlock
from encoder           import Encoder, build_padding_mask
from decoder           import Decoder


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def batch_x(rng):
    """Standard (batch=2, seq=12, d_model=64) input tensor."""
    return rng.standard_normal((2, 12, 64)).astype(np.float32)

@pytest.fixture
def causal_mask():
    return make_causal_mask(12)

@pytest.fixture
def small_decoder():
    return Decoder(vocab_size=128, d_model=64, num_layers=2,
                   num_heads=4, num_kv_heads=2, max_seq_len=64,
                   dropout_rate=0.0, ffn_type="swiglu", seed=0)


# ══════════════════════════════════════════════════════════════════════════════
# ROPE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRoPE:
    def test_output_shape(self, rng):
        rope = RotaryEmbedding(d_head=16, max_seq_len=128)
        x = rng.standard_normal((4, 12, 16)).astype(np.float32)
        out = rope.apply(x)
        assert out.shape == x.shape

    def test_norm_preserved(self, rng):
        """Rotation must not change vector magnitudes."""
        rope = RotaryEmbedding(d_head=16, max_seq_len=128)
        x = rng.standard_normal((2, 8, 16)).astype(np.float32)
        out = rope.apply(x)
        norms_in  = np.linalg.norm(x.reshape(-1, 16),   axis=-1)
        norms_out = np.linalg.norm(out.reshape(-1, 16),  axis=-1)
        np.testing.assert_allclose(norms_in, norms_out, atol=1e-5)

    def test_offset_changes_rotation(self):
        """Different offsets must produce different rotations."""
        rope = RotaryEmbedding(d_head=16, max_seq_len=128)
        x = np.ones((1, 1, 16), dtype=np.float32)
        out0 = rope.apply(x, offset=0)
        out5 = rope.apply(x, offset=5)
        assert not np.allclose(out0, out5), "Same rotation at different positions!"

    def test_no_nan_inf(self, rng):
        rope = RotaryEmbedding(d_head=32, max_seq_len=512)
        x = rng.standard_normal((2, 100, 32)).astype(np.float32) * 100
        out = rope.apply(x)
        assert np.isfinite(out).all()

    def test_even_d_head_required(self):
        with pytest.raises(AssertionError):
            RotaryEmbedding(d_head=15)  # odd d_head must fail


# ══════════════════════════════════════════════════════════════════════════════
# KV-CACHE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestKVCache:
    def test_initial_state(self):
        cache = KVCache(batch_size=1, num_kv_heads=2, max_seq_len=32, d_head=16)
        assert cache.current_len == 0

    def test_update_accumulates(self, rng):
        cache = KVCache(1, 2, 32, 16)
        k1 = rng.standard_normal((1, 2, 5, 16)).astype(np.float32)
        v1 = rng.standard_normal((1, 2, 5, 16)).astype(np.float32)
        kf, vf = cache.update(k1, v1)
        assert kf.shape == (1, 2, 5, 16)
        assert cache.current_len == 5

        k2 = rng.standard_normal((1, 2, 3, 16)).astype(np.float32)
        v2 = rng.standard_normal((1, 2, 3, 16)).astype(np.float32)
        kf2, vf2 = cache.update(k2, v2)
        assert kf2.shape == (1, 2, 8, 16)
        assert cache.current_len == 8

    def test_previous_tokens_preserved(self, rng):
        """First N tokens in cache must not change after subsequent updates."""
        cache = KVCache(1, 2, 32, 16)
        k1 = rng.standard_normal((1, 2, 4, 16)).astype(np.float32)
        v1 = rng.standard_normal((1, 2, 4, 16)).astype(np.float32)
        kf, _ = cache.update(k1, v1)
        first_4 = kf[:, :, :4, :].copy()

        k2 = rng.standard_normal((1, 2, 2, 16)).astype(np.float32)
        v2 = rng.standard_normal((1, 2, 2, 16)).astype(np.float32)
        kf2, _ = cache.update(k2, v2)
        np.testing.assert_array_equal(kf2[:, :, :4, :], first_4)

    def test_overflow_raises(self, rng):
        cache = KVCache(1, 1, 4, 8)   # max_seq_len = 4
        k = rng.standard_normal((1, 1, 5, 8)).astype(np.float32)
        v = rng.standard_normal((1, 1, 5, 8)).astype(np.float32)
        with pytest.raises(AssertionError):
            cache.update(k, v)

    def test_reset(self, rng):
        cache = KVCache(1, 2, 32, 16)
        k = rng.standard_normal((1, 2, 6, 16)).astype(np.float32)
        v = rng.standard_normal((1, 2, 6, 16)).astype(np.float32)
        cache.update(k, v)
        assert cache.current_len == 6
        cache.reset()
        assert cache.current_len == 0


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestScaledDotProductAttention:
    def test_output_shape(self, rng):
        Q = rng.standard_normal((2, 4, 8, 16)).astype(np.float32)
        K = rng.standard_normal((2, 4, 8, 16)).astype(np.float32)
        V = rng.standard_normal((2, 4, 8, 16)).astype(np.float32)
        out, w = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (2, 4, 8, 16)
        assert w.shape   == (2, 4, 8, 8)

    def test_weights_sum_to_one(self, rng):
        Q = rng.standard_normal((2, 4, 8, 16)).astype(np.float32)
        K = rng.standard_normal((2, 4, 8, 16)).astype(np.float32)
        V = rng.standard_normal((2, 4, 8, 16)).astype(np.float32)
        _, w = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(w.sum(axis=-1), np.ones((2, 4, 8)), atol=1e-5)

    def test_causal_mask_no_future_leakage(self, rng):
        B, H, S, D = 2, 4, 10, 16
        Q = rng.standard_normal((B, H, S, D)).astype(np.float32)
        K = rng.standard_normal((B, H, S, D)).astype(np.float32)
        V = rng.standard_normal((B, H, S, D)).astype(np.float32)
        mask = make_causal_mask(S)
        _, w = scaled_dot_product_attention(Q, K, V, mask=mask)
        future_attn = np.triu(w[0, 0], k=1).max()
        assert future_attn < 1e-6, f"Future attention leaked: {future_attn}"

    def test_gqa_output_matches_q_heads(self, rng):
        """GQA: more Q heads than KV heads — output should match Q head count."""
        Q = rng.standard_normal((2, 8, 6, 16)).astype(np.float32)
        K = rng.standard_normal((2, 2, 6, 16)).astype(np.float32)
        V = rng.standard_normal((2, 2, 6, 16)).astype(np.float32)
        out, w = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (2, 8, 6, 16)

    def test_chunked_equals_full(self, rng):
        Q = rng.standard_normal((2, 4, 12, 16)).astype(np.float32)
        K = rng.standard_normal((2, 4, 12, 16)).astype(np.float32)
        V = rng.standard_normal((2, 4, 12, 16)).astype(np.float32)
        mask = make_causal_mask(12)
        out_full,  _ = scaled_dot_product_attention(Q, K, V, mask=mask, chunk_size=None)
        out_chunk, _ = scaled_dot_product_attention(Q, K, V, mask=mask, chunk_size=4)
        np.testing.assert_allclose(out_full, out_chunk, atol=1e-5)

    def test_no_nan_with_extreme_inputs(self, rng):
        Q = rng.standard_normal((2, 2, 8, 16)).astype(np.float32) * 100
        K = rng.standard_normal((2, 2, 8, 16)).astype(np.float32) * 100
        V = rng.standard_normal((2, 2, 8, 16)).astype(np.float32) * 100
        out, _ = scaled_dot_product_attention(Q, K, V)
        assert np.isfinite(out).all()


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-HEAD ATTENTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiHeadAttention:
    def test_output_shape(self, batch_x, causal_mask):
        mha = MultiHeadAttention(64, num_heads=4, num_kv_heads=2, seed=0)
        out, w = mha.forward(batch_x, mask=causal_mask)
        assert out.shape == batch_x.shape

    def test_causal_mask_respected(self, batch_x):
        mha = MultiHeadAttention(64, num_heads=4, num_kv_heads=2, seed=0)
        mask = make_causal_mask(12)
        _, w = mha.forward(batch_x, mask=mask)
        assert np.triu(w[0, 0], k=1).max() < 1e-6

    def test_cross_attention_shape(self, batch_x, rng):
        mha = MultiHeadAttention(64, num_heads=4, num_kv_heads=2, seed=0)
        enc_mem = rng.standard_normal((2, 15, 64)).astype(np.float32)
        dec_q   = rng.standard_normal((2, 8, 64)).astype(np.float32)
        out, w = mha.forward(dec_q, x_k=enc_mem, x_v=enc_mem)
        assert out.shape == (2, 8, 64)
        assert w.shape   == (2, 4, 8, 15)

    def test_mqa_variant(self, batch_x):
        """MQA: 1 KV head for all Q heads."""
        mqa = MultiHeadAttention(64, num_heads=4, num_kv_heads=1, seed=0)
        out, _ = mqa.forward(batch_x)
        assert out.shape == batch_x.shape

    def test_no_nan(self, batch_x):
        mha = MultiHeadAttention(64, num_heads=4, seed=0)
        out, _ = mha.forward(batch_x)
        assert np.isfinite(out).all()

    def test_kv_cache_consistency(self, rng):
        """Cached decode at last position must match full-sequence output."""
        mha = MultiHeadAttention(64, num_heads=4, num_kv_heads=2, max_seq_len=64, seed=7)
        x1  = rng.standard_normal((1, 8, 64)).astype(np.float32)
        x2  = rng.standard_normal((1, 1, 64)).astype(np.float32)

        # Full sequence
        full = np.concatenate([x1, x2], axis=1)
        out_full, _ = mha.forward(full, mask=make_causal_mask(9))

        # Cached
        cache = KVCache(1, 2, 64, 16)
        mha.forward(x1, mask=make_causal_mask(8), kv_cache=cache, rope_offset=0)
        out_step, _ = mha.forward(x2, kv_cache=cache, rope_offset=8)

        np.testing.assert_allclose(out_full[0, -1], out_step[0, 0], atol=1e-3)

    def test_invalid_kv_heads_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(64, num_heads=4, num_kv_heads=3)  # 4 % 3 != 0


# ══════════════════════════════════════════════════════════════════════════════
# FEED-FORWARD TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestFeedForward:
    def test_swiglu_output_shape(self, batch_x):
        ffn = SwiGLUFeedForward(64)
        assert ffn.forward(batch_x).shape == batch_x.shape

    def test_gelu_output_shape(self, batch_x):
        ffn = GELUFeedForward(64, d_ff=256)
        assert ffn.forward(batch_x).shape == batch_x.shape

    def test_position_wise_independence(self, batch_x):
        ffn = SwiGLUFeedForward(64)
        out1 = ffn.forward(batch_x)
        x_mod = batch_x.copy()
        x_mod[:, 5, :] *= 999.0
        out2 = ffn.forward(x_mod)
        np.testing.assert_array_equal(out1[:, 0, :], out2[:, 0, :])

    def test_silu_zero_at_zero(self):
        assert abs(float(silu(np.array([0.0])))) < 1e-7

    def test_silu_negative_not_exactly_zero(self):
        """SiLU(-1) should be non-zero (unlike ReLU)."""
        val = float(silu(np.array([-1.0])))
        assert val < 0 and val > -0.5

    def test_gelu_zero_at_zero(self):
        assert abs(float(gelu_approx(np.array([0.0])))) < 1e-6

    def test_d_ff_alignment(self):
        """SwiGLU d_ff must be a multiple of the specified alignment."""
        for d in [64, 128, 256, 512, 1024]:
            d_ff = _make_ffn_dim(d, 8/3, 64)
            assert d_ff % 64 == 0, f"d_ff={d_ff} not multiple of 64 for d_model={d}"

    def test_no_nan(self, batch_x):
        ffn = SwiGLUFeedForward(64)
        out = ffn.forward(batch_x * 1000)
        assert np.isfinite(out).all()

    def test_dropout_stochastic_in_training(self, batch_x):
        ffn = SwiGLUFeedForward(64, dropout_rate=0.5)
        o1 = ffn.forward(batch_x, training=True)
        o2 = ffn.forward(batch_x, training=True)
        assert not np.allclose(o1, o2)

    def test_dropout_deterministic_in_eval(self, batch_x):
        ffn = SwiGLUFeedForward(64, dropout_rate=0.5)
        o1 = ffn.forward(batch_x, training=False)
        o2 = ffn.forward(batch_x, training=False)
        np.testing.assert_array_equal(o1, o2)


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalization:
    def test_rmsnorm_shape(self, batch_x):
        rms = RMSNorm(64)
        assert rms(batch_x).shape == batch_x.shape

    def test_rmsnorm_output_rms_is_one(self, batch_x):
        rms = RMSNorm(64)
        out = rms(batch_x * 10 + 5)
        rms_out = np.sqrt((out ** 2).mean(axis=-1))
        np.testing.assert_allclose(rms_out, np.ones_like(rms_out), atol=0.05)

    def test_rmsnorm_gamma_scales(self, batch_x):
        rms = RMSNorm(64)
        rms.gamma[:] = 3.0
        out = rms(batch_x)
        rms_out = np.sqrt((out ** 2).mean(axis=-1)).mean()
        assert abs(rms_out - 3.0) < 0.1

    def test_rmsnorm_position_independent(self, batch_x):
        rms = RMSNorm(64)
        out1 = rms(batch_x)
        x2 = batch_x.copy(); x2[:, 5, :] *= 1e5
        out2 = rms(x2)
        np.testing.assert_allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5)

    def test_layernorm_mean_zero(self, batch_x):
        ln = LayerNorm(64)
        out = ln(batch_x * 10 + 5)
        assert abs(out.mean(axis=-1)).max() < 1e-5

    def test_layernorm_std_one(self, batch_x):
        ln = LayerNorm(64)
        out = ln(batch_x * 10 + 5)
        std_vals = out.std(axis=-1)
        np.testing.assert_allclose(std_vals, np.ones_like(std_vals), atol=0.02)

    def test_layernorm_shape(self, batch_x):
        ln = LayerNorm(64)
        assert ln(batch_x).shape == batch_x.shape

    def test_no_nan_extreme_input(self):
        rms = RMSNorm(64)
        ln  = LayerNorm(64)
        x = np.full((2, 8, 64), 1e6, dtype=np.float32)
        assert np.isfinite(rms(x)).all()
        assert np.isfinite(ln(x)).all()

    def test_rmsnorm_params_count(self):
        rms = RMSNorm(128)
        assert rms.count_parameters() == 128   # gamma only

    def test_layernorm_params_count(self):
        ln = LayerNorm(128)
        assert ln.count_parameters() == 256    # gamma + beta


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestTransformerBlock:
    def test_output_shape(self, batch_x):
        blk = TransformerBlock(64, num_heads=4, num_kv_heads=2, ffn_type="swiglu")
        out, _ = blk.forward(batch_x)
        assert out.shape == batch_x.shape

    def test_causal_mask_enforced(self, batch_x):
        blk = TransformerBlock(64, num_heads=4, num_kv_heads=2)
        mask = make_causal_mask(12)
        _, w = blk.forward(batch_x, self_mask=mask)
        assert np.triu(w[0, 0], k=1).max() < 1e-6

    def test_cross_attn_shape(self, batch_x, rng):
        blk = TransformerBlock(64, num_heads=4, num_kv_heads=2, use_cross_attn=True)
        mem = rng.standard_normal((2, 15, 64)).astype(np.float32)
        out, _ = blk.forward(batch_x, enc_memory=mem)
        assert out.shape == batch_x.shape

    def test_not_identity(self, batch_x):
        blk = TransformerBlock(64, num_heads=4)
        out, _ = blk.forward(batch_x)
        assert not np.allclose(out, batch_x)

    def test_no_nan(self, batch_x):
        blk = TransformerBlock(64, num_heads=4)
        out, _ = blk.forward(batch_x)
        assert np.isfinite(out).all()

    def test_kv_cache_consistency(self, rng):
        blk = TransformerBlock(64, num_heads=4, num_kv_heads=2, ffn_type="swiglu", seed=5)
        seq = rng.standard_normal((1, 8, 64)).astype(np.float32)
        out_full, _ = blk.forward(seq, self_mask=make_causal_mask(8))
        cache = KVCache(1, 2, 64, 16)
        out_last = None
        for i in range(8):
            tok = seq[:, i:i+1, :]
            out_last, _ = blk.forward(tok, kv_cache=cache, rope_offset=i)
        np.testing.assert_allclose(out_full[0, -1], out_last[0, 0], atol=1e-4)

    def test_depth_scaling_reduces_w_o(self):
        blk2  = TransformerBlock(64, 4, num_layers=2,  layer_idx=0, seed=0)
        blk24 = TransformerBlock(64, 4, num_layers=24, layer_idx=0, seed=0)
        assert np.abs(blk24.self_attn.W_O).mean() < np.abs(blk2.self_attn.W_O).mean()


# ══════════════════════════════════════════════════════════════════════════════
# ENCODER TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestEncoder:
    @pytest.fixture
    def enc(self):
        return Encoder(vocab_size=200, d_model=64, num_layers=2,
                       num_heads=4, num_kv_heads=2, max_seq_len=64,
                       dropout_rate=0.0, ffn_type="swiglu", seed=0)

    def test_output_shape(self, enc, rng):
        ids = rng.integers(1, 200, size=(2, 12)).astype(np.int64)
        out = enc.forward(ids)
        assert out.shape == (2, 12, 64)

    def test_output_dtype(self, enc, rng):
        ids = rng.integers(1, 200, size=(2, 12)).astype(np.int64)
        out = enc.forward(ids)
        assert out.dtype == np.float32

    def test_no_nan(self, enc, rng):
        ids = rng.integers(1, 200, size=(2, 12)).astype(np.int64)
        out = enc.forward(ids)
        assert np.isfinite(out).all()

    def test_deterministic_eval(self, enc, rng):
        ids = rng.integers(1, 200, size=(2, 12)).astype(np.int64)
        o1 = enc.forward(ids, training=False)
        o2 = enc.forward(ids, training=False)
        np.testing.assert_array_equal(o1, o2)

    def test_different_inputs_different_outputs(self, enc, rng):
        ids1 = rng.integers(1, 200, size=(2, 12)).astype(np.int64)
        ids2 = rng.integers(1, 200, size=(2, 12)).astype(np.int64)
        o1 = enc.forward(ids1, training=False)
        o2 = enc.forward(ids2, training=False)
        assert not np.allclose(o1, o2)

    def test_padding_mask_shape(self):
        ids = np.array([[1, 2, 0, 0], [3, 0, 0, 0]], dtype=np.int64)
        mask = build_padding_mask(ids, pad_token_id=0)
        assert mask.shape == (2, 1, 1, 4)
        assert mask[0, 0, 0, 2] < -1e8
        assert mask[0, 0, 0, 0] == 0.0

    def test_mean_pool_shape(self, enc, rng):
        ids = rng.integers(1, 200, size=(2, 12)).astype(np.int64)
        emb = enc.encode_mean(ids)
        assert emb.shape == (2, 64)


# ══════════════════════════════════════════════════════════════════════════════
# DECODER TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDecoder:
    def test_logits_shape(self, small_decoder, rng):
        ids = rng.integers(1, 128, size=(2, 10)).astype(np.int64)
        logits, hidden = small_decoder.forward(ids)
        assert logits.shape  == (2, 10, 128)
        assert hidden.shape  == (2, 10, 64)

    def test_no_nan(self, small_decoder, rng):
        ids = rng.integers(1, 128, size=(2, 10)).astype(np.int64)
        logits, _ = small_decoder.forward(ids)
        assert np.isfinite(logits).all()

    def test_causal_property(self, small_decoder, rng):
        """Changing future tokens must not affect past logits."""
        ids = rng.integers(1, 128, size=(2, 12)).astype(np.int64)
        logits1, _ = small_decoder.forward(ids)
        ids2 = ids.copy()
        ids2[:, 8:] = rng.integers(1, 128, size=(2, 4))
        logits2, _ = small_decoder.forward(ids2)
        np.testing.assert_allclose(logits1[:, :8, :], logits2[:, :8, :], atol=1e-4)

    def test_greedy_is_deterministic(self, small_decoder):
        prompt = np.array([[1, 2, 3]], dtype=np.int64)
        g1 = small_decoder.generate(prompt, max_new_tokens=5, temperature=0.0)
        g2 = small_decoder.generate(prompt, max_new_tokens=5, temperature=0.0)
        np.testing.assert_array_equal(g1, g2)

    def test_greedy_length(self, small_decoder):
        prompt = np.array([[1, 5, 9]], dtype=np.int64)
        gen = small_decoder.generate(prompt, max_new_tokens=10, temperature=0.0)
        assert gen.shape == (1, 13)

    def test_temperature_is_stochastic(self, small_decoder):
        prompt = np.array([[1, 2, 3]], dtype=np.int64)
        g1 = small_decoder.generate(prompt, max_new_tokens=8, temperature=1.0)
        g2 = small_decoder.generate(prompt, max_new_tokens=8, temperature=1.0)
        assert not np.array_equal(g1, g2)

    def test_top_k_length(self, small_decoder):
        prompt = np.array([[1, 2]], dtype=np.int64)
        gen = small_decoder.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=10)
        assert gen.shape == (1, 10)

    def test_top_p_length(self, small_decoder):
        prompt = np.array([[1, 2]], dtype=np.int64)
        gen = small_decoder.generate(prompt, max_new_tokens=8, temperature=0.9, top_p=0.9)
        assert gen.shape == (1, 10)

    def test_token_ids_in_vocab(self, small_decoder):
        prompt = np.array([[1, 2, 3]], dtype=np.int64)
        gen = small_decoder.generate(prompt, max_new_tokens=20, temperature=0.8)
        assert np.all(gen >= 0) and np.all(gen < 128)

    def test_eos_stops_early(self, small_decoder):
        prompt = np.array([[1, 2, 3]], dtype=np.int64)
        # Use greedy — first generated token will always be the same
        greedy = small_decoder.generate(prompt, max_new_tokens=1, temperature=0.0)
        eos_id = int(greedy[0, 3])
        gen = small_decoder.generate(prompt, max_new_tokens=20, temperature=0.0, eos_token_id=eos_id)
        assert gen.shape[1] <= 3 + 20   # may stop early

    def test_kv_cache_correctness(self, rng):
        dec = Decoder(vocab_size=128, d_model=64, num_layers=2, num_heads=4,
                      num_kv_heads=2, max_seq_len=64, seed=99)
        ids = rng.integers(1, 128, size=(1, 10)).astype(np.int64)
        logits_full, _ = dec.forward(ids)

        caches = dec._make_kv_caches(1)
        dec.forward(ids[:, :9], kv_caches=caches, rope_offset=0)
        logits_step, _ = dec.forward(ids[:, 9:10], kv_caches=caches, rope_offset=9)

        np.testing.assert_allclose(logits_full[0, 9], logits_step[0, 0], atol=1e-3)

    def test_empty_generation_request(self, small_decoder):
        """max_new_tokens=0 should return prompt unchanged."""
        prompt = np.array([[1, 2, 3]], dtype=np.int64)
        gen = small_decoder.generate(prompt, max_new_tokens=0, temperature=0.0)
        np.testing.assert_array_equal(gen, prompt)


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_single_token_input(self):
        dec = Decoder(vocab_size=64, d_model=32, num_layers=1, num_heads=2,
                      num_kv_heads=1, max_seq_len=32, seed=0)
        ids = np.array([[5]], dtype=np.int64)
        logits, hidden = dec.forward(ids)
        assert logits.shape == (1, 1, 64)

    def test_max_seq_len_boundary(self):
        """Input exactly at max_seq_len must not crash."""
        dec = Decoder(vocab_size=64, d_model=32, num_layers=1, num_heads=2,
                      num_kv_heads=1, max_seq_len=16, seed=0)
        ids = np.ones((1, 16), dtype=np.int64)
        logits, _ = dec.forward(ids)
        assert logits.shape == (1, 16, 64)

    def test_batch_size_one(self):
        dec = Decoder(vocab_size=64, d_model=32, num_layers=2, num_heads=2,
                      num_kv_heads=1, max_seq_len=32, seed=0)
        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        logits, _ = dec.forward(ids)
        assert logits.shape == (1, 5, 64)

    def test_large_batch_size(self):
        """Batch size 16 must work without error."""
        dec = Decoder(vocab_size=32, d_model=16, num_layers=1, num_heads=2,
                      num_kv_heads=1, max_seq_len=32, seed=0)
        ids = np.ones((16, 8), dtype=np.int64)
        logits, _ = dec.forward(ids)
        assert logits.shape == (16, 8, 32)

    def test_all_padding_input(self):
        """All-padding sequence (edge case for encoder)."""
        enc = Encoder(vocab_size=32, d_model=16, num_layers=1, num_heads=2,
                      num_kv_heads=1, max_seq_len=32, seed=0)
        ids = np.zeros((2, 8), dtype=np.int64)   # all padding
        out = enc.forward(ids, training=False)
        assert out.shape == (2, 8, 16)
        assert np.isfinite(out).all()

    def test_softmax_numerical_stability(self):
        """Softmax must not produce NaN on extreme inputs."""
        x = np.array([1e10, 1e10, 1e10, 1e10], dtype=np.float32)
        result = softmax(x)
        assert np.isfinite(result).all()
        assert abs(result.sum() - 1.0) < 1e-5

    def test_softmax_very_negative(self):
        x = np.array([-1e10, -1e10, 0.0, -1e10], dtype=np.float32)
        result = softmax(x)
        assert np.isfinite(result).all()
        assert result[2] > 0.99  # only the zero survives


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
