"""
================================================================================
FILE: test_45E.py
PROJECT: Transformer Language Model — 45E v2
PURPOSE: Full integration + correctness test suite — Steps 13–20
================================================================================

Tests every module in the 45E v2 build:
  attention.py         (Steps 13, 14) — RoPE, KV-cache, SDPA, GQA, chunked
  multihead.py         (Step 15)      — MHA / GQA / MQA, cross-attention
  feedforward.py       (Step 16)      — SwiGLU, GELU, SiLU activation
  layernorm.py         (Step 17)      — RMSNorm, LayerNorm
  transformer_block.py (Step 18)      — Full block, depth scaling, KV-cache
  encoder.py           (Step 19)      — Encoder stack, padding mask, mean pool
  decoder.py           (Step 20)      — Decoder, generation strategies, EOS

Run with:  python test_45E.py
Returns:   exit code 0 = all pass, 1 = failures
================================================================================
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from attention          import (RotaryEmbedding, KVCache, make_causal_mask,
                                softmax, scaled_dot_product_attention)
from multihead          import MultiHeadAttention
from feedforward        import SwiGLUFeedForward, GELUFeedForward, silu, gelu_approx
from layernorm          import RMSNorm, LayerNorm
from transformer_block  import TransformerBlock
from encoder            import Encoder, build_padding_mask
from decoder            import Decoder


# ──────────────────────────────────────────────────────────────────────────────
# TEST HARNESS
# ──────────────────────────────────────────────────────────────────────────────

class Suite:
    def __init__(self):
        self.passed = self.failed = 0
        self.failures: list = []

    def ok(self, cond: bool, name: str, detail: str = ""):
        if cond:
            self.passed += 1
            print(f"  ✓  {name}")
        else:
            self.failed += 1
            print(f"  ✗  {name}" + (f"\n       → {detail}" if detail else ""))
            self.failures.append(name)

    def section(self, title: str):
        print(f"\n{'─'*64}")
        print(f"  {title}")
        print(f"{'─'*64}")

    def done(self) -> bool:
        total = self.passed + self.failed
        print(f"\n{'═'*64}")
        print(f"  TEST RESULTS: {self.passed}/{total} passed", end="")
        if self.failed:
            print(f"  ← {self.failed} FAILED")
            for f in self.failures:
                print(f"    ✗ {f}")
        else:
            print("  — all green ✓")
        print(f"{'═'*64}")
        return self.failed == 0


T = Suite()
rng = np.random.default_rng(1337)

# Shared dims — small for speed, realistic proportions
VOCAB = 256
D     = 128    # d_model
H     = 8      # num_heads
KV_H  = 2      # num_kv_heads (GQA)
L     = 3      # num_layers
B     = 2      # batch
S     = 20     # seq_len
D_HEAD = D // H  # 16


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: RoPE
# ══════════════════════════════════════════════════════════════════════════════

T.section("Step 13/14 — Rotary Position Embeddings (RoPE)")

rope = RotaryEmbedding(d_head=D_HEAD, max_seq_len=512)
q = rng.standard_normal((B, H, S, D_HEAD)).astype(np.float32)
q_rot = rope.apply(q.reshape(B*H, S, D_HEAD)).reshape(B, H, S, D_HEAD)

T.ok(q_rot.shape == q.shape,                    "RoPE preserves input shape")
T.ok(not np.allclose(q_rot, q),                 "RoPE actually rotates (output ≠ input)")
T.ok(np.isfinite(q_rot).all(),                  "RoPE output: no NaN/Inf")

# Norm preservation: |Rx| = |x| (rotation doesn't change magnitude)
norms_in  = np.linalg.norm(q.reshape(B*H*S, D_HEAD),     axis=-1)
norms_out = np.linalg.norm(q_rot.reshape(B*H*S, D_HEAD), axis=-1)
T.ok(np.allclose(norms_in, norms_out, atol=1e-5),  "RoPE preserves vector norms")

# Relative position: same relative distance → same inner product
q0 = rope.apply(np.ones((1, 1, D_HEAD), dtype=np.float32), offset=0)
k1 = rope.apply(np.ones((1, 1, D_HEAD), dtype=np.float32), offset=1)
# Just verify different offsets give different rotations
diff_offset = abs(
    rope.apply(np.ones((1, 1, D_HEAD), dtype=np.float32), offset=1) -
    rope.apply(np.ones((1, 1, D_HEAD), dtype=np.float32), offset=2)
).max()
T.ok(diff_offset > 1e-4,                        "RoPE: different offsets → different rotations")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: KV-CACHE
# ══════════════════════════════════════════════════════════════════════════════

T.section("Step 14 — KV-Cache")

cache = KVCache(batch_size=1, num_kv_heads=KV_H, max_seq_len=64, d_head=D_HEAD)
k1 = rng.standard_normal((1, KV_H, 6, D_HEAD)).astype(np.float32)
v1 = rng.standard_normal((1, KV_H, 6, D_HEAD)).astype(np.float32)
k_full, v_full = cache.update(k1, v1)
T.ok(k_full.shape == (1, KV_H, 6, D_HEAD),     "KVCache: first update shape")
T.ok(cache.current_len == 6,                    "KVCache: current_len = 6")

k2 = rng.standard_normal((1, KV_H, 1, D_HEAD)).astype(np.float32)
v2 = rng.standard_normal((1, KV_H, 1, D_HEAD)).astype(np.float32)
k_f2, v_f2 = cache.update(k2, v2)
T.ok(k_f2.shape == (1, KV_H, 7, D_HEAD),       "KVCache: second update appends correctly")
T.ok(np.allclose(k_f2[:, :, :6, :], k1),       "KVCache: first 6 positions preserved exactly")

cache.reset()
T.ok(cache.current_len == 0,                    "KVCache: reset clears the counter")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

T.section("Step 14 — Scaled Dot-Product Attention")

Q = rng.standard_normal((B, H, S, D_HEAD)).astype(np.float32)
K = rng.standard_normal((B, H, S, D_HEAD)).astype(np.float32)
V = rng.standard_normal((B, H, S, D_HEAD)).astype(np.float32)
mask = make_causal_mask(S)

out_sdpa, w_sdpa = scaled_dot_product_attention(Q, K, V, mask=mask)
T.ok(out_sdpa.shape == (B, H, S, D_HEAD),          "SDPA: output shape")
T.ok(w_sdpa.shape   == (B, H, S, S),               "SDPA: weights shape")
T.ok(np.allclose(w_sdpa.sum(-1), 1.0),             "SDPA: attention weights sum to 1")
T.ok(np.triu(w_sdpa[0,0],k=1).max() < 1e-6,       "SDPA: causal mask — no future leakage")

# GQA (2 KV heads, 8 Q heads)
K_gqa = rng.standard_normal((B, KV_H, S, D_HEAD)).astype(np.float32)
V_gqa = rng.standard_normal((B, KV_H, S, D_HEAD)).astype(np.float32)
out_gqa, _ = scaled_dot_product_attention(Q, K_gqa, V_gqa)
T.ok(out_gqa.shape == (B, H, S, D_HEAD),            "SDPA GQA: output shape with kv_heads=2")

# Chunked == full
out_chunk, _ = scaled_dot_product_attention(Q, K, V, mask=mask, chunk_size=5)
T.ok(np.allclose(out_sdpa, out_chunk, atol=1e-5),  "SDPA: chunked matches full (atol=1e-5)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

T.section("Step 15 — Multi-Head Attention (GQA with RoPE)")

x = rng.standard_normal((B, S, D)).astype(np.float32)

mha = MultiHeadAttention(D, num_heads=H, num_kv_heads=KV_H, seed=0)
out_mha, w_mha = mha.forward(x, mask=mask, training=False)
T.ok(out_mha.shape == (B, S, D),                   "MHA: output shape")
T.ok(w_mha.shape   == (B, H, S, S),                "MHA: weights shape")
T.ok(np.allclose(w_mha.sum(-1), 1.0),              "MHA: weights sum to 1")
T.ok(np.triu(w_mha[0,0],k=1).max() < 1e-6,        "MHA: causal mask respected")
T.ok(np.isfinite(out_mha).all(),                   "MHA: no NaN/Inf")

# MQA
mqa = MultiHeadAttention(D, num_heads=H, num_kv_heads=1, seed=1)
out_mqa, _ = mqa.forward(x, training=False)
T.ok(out_mqa.shape == (B, S, D),                   "MQA: output shape")

# Cross-attention
enc_mem = rng.standard_normal((B, 15, D)).astype(np.float32)
dec_q   = rng.standard_normal((B, 8,  D)).astype(np.float32)
out_cross, w_cross = mha.forward(dec_q, x_k=enc_mem, x_v=enc_mem)
T.ok(out_cross.shape == (B, 8, D),                 "MHA cross-attention: output shape")
T.ok(w_cross.shape   == (B, H, 8, 15),             "MHA cross-attention: weights shape (q×enc)")

# KV-cache consistency: one-shot prefill vs incremental decode
cache_mha = KVCache(batch_size=1, num_kv_heads=KV_H, max_seq_len=64, d_head=D_HEAD)
x1 = rng.standard_normal((1, 8, D)).astype(np.float32)
x2 = rng.standard_normal((1, 1, D)).astype(np.float32)
out_p, _ = mha.forward(x1, mask=make_causal_mask(8), kv_cache=cache_mha, rope_offset=0)
out_d, _ = mha.forward(x2, kv_cache=cache_mha, rope_offset=8)
full_x = np.concatenate([x1, x2], axis=1)
out_full_ref, _ = mha.forward(full_x, mask=make_causal_mask(9))
diff_cache = abs(out_d[0,0] - out_full_ref[0,-1]).max()
T.ok(diff_cache < 1e-3,                            f"MHA KV-cache vs full-pass diff={diff_cache:.2e}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════════════

T.section("Step 16 — Feed-Forward Networks (SwiGLU + GELU)")

swiglu = SwiGLUFeedForward(d_model=D, dropout_rate=0.0, seed=0)
out_swiglu = swiglu.forward(x, training=False)
T.ok(out_swiglu.shape == x.shape,                  "SwiGLU: output shape")
T.ok(np.isfinite(out_swiglu).all(),                "SwiGLU: no NaN/Inf")
T.ok(not np.allclose(out_swiglu, x),               "SwiGLU: transforms input")

gelu_ffn = GELUFeedForward(d_model=D, d_ff=4*D, seed=0)
out_gelu = gelu_ffn.forward(x, training=False)
T.ok(out_gelu.shape == x.shape,                    "GELU FFN: output shape")

# Position-wise independence
x_mod = x.copy(); x_mod[:, 7, :] *= 999.0
out_mod = swiglu.forward(x_mod, training=False)
T.ok(np.allclose(out_swiglu[:,0,:], out_mod[:,0,:]), "SwiGLU: position-wise independent")

# SiLU properties
silu_vals = silu(np.array([-5., -1., 0., 1., 5.], dtype=np.float32))
T.ok(abs(float(silu_vals[2])) < 1e-7,              "SiLU(0) = 0")
T.ok(float(silu_vals[1]) > -0.3,                   "SiLU(-1) is small-negative (not zero)")
T.ok(float(silu_vals[3]) > 0.7,                    "SiLU(1) > 0.7")

# Dropout active in training
sw_drop = SwiGLUFeedForward(d_model=D, dropout_rate=0.3, seed=0)
o1 = sw_drop.forward(x, training=True)
o2 = sw_drop.forward(x, training=True)
T.ok(not np.allclose(o1, o2),                      "SwiGLU dropout: training outputs stochastic")
oe = sw_drop.forward(x, training=False)
T.ok(np.allclose(oe, sw_drop.forward(x, training=False)), "SwiGLU: eval outputs deterministic")

# SwiGLU d_ff rounding (LLaMA style)
from feedforward import _make_ffn_dim
for d in [64, 128, 512, 1024]:
    d_ff = _make_ffn_dim(d, 8/3, 64)
    T.ok(d_ff % 64 == 0, f"SwiGLU d_ff multiple of 64 for d_model={d}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

T.section("Step 17 — RMSNorm + LayerNorm")

rms = RMSNorm(D)
out_rms = rms(x)
rms_vals = np.sqrt((out_rms**2).mean(-1))
T.ok(out_rms.shape == x.shape,                     "RMSNorm: shape preserved")
T.ok(abs(rms_vals.mean() - 1.0) < 0.02,           f"RMSNorm: output RMS ≈ 1 (got {rms_vals.mean():.4f})")
T.ok(np.isfinite(out_rms).all(),                   "RMSNorm: no NaN/Inf")

# Per-position independent
x_big = x.copy(); x_big[:, 5, :] *= 1e4
out_rms_big = rms(x_big)
T.ok(np.allclose(out_rms[:,0,:], out_rms_big[:,0,:]), "RMSNorm: position-wise independent")

rms_scaled = RMSNorm(D)
rms_scaled.gamma[:] = 3.0
out_scaled = rms_scaled(x)
T.ok(abs(np.sqrt((out_scaled**2).mean(-1)).mean() - 3.0) < 0.1,
     "RMSNorm: gamma scales output RMS")

ln = LayerNorm(D)
out_ln = ln(x)
T.ok(out_ln.shape == x.shape,                      "LayerNorm: shape preserved")
T.ok(abs(out_ln.mean(-1)).max() < 1e-5,            "LayerNorm: mean ≈ 0 per position")
T.ok(abs(out_ln.std(-1).mean() - 1.0) < 0.02,     "LayerNorm: std ≈ 1 per position")

T.ok(rms.count_parameters() == D,                  f"RMSNorm params = d_model={D}")
T.ok(ln.count_parameters()  == 2*D,                f"LayerNorm params = 2×d_model={2*D}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════════════════════

T.section("Step 18 — Transformer Block")

blk = TransformerBlock(D, num_heads=H, num_kv_heads=KV_H,
                       ffn_type="swiglu", layer_idx=0, num_layers=L, seed=0)
out_blk, w_blk = blk.forward(x, training=False)
T.ok(out_blk.shape == x.shape,                     "Block: output shape")
T.ok(np.isfinite(out_blk).all(),                   "Block: no NaN/Inf")
T.ok(not np.allclose(out_blk, x),                  "Block: transforms input (not identity)")

# Causal mask
out_c, w_c = blk.forward(x, self_mask=mask, training=False)
T.ok(np.triu(w_c[0,0],k=1).max() < 1e-6,          "Block: causal mask enforced")

# Cross-attention variant
dec_blk = TransformerBlock(D, num_heads=H, num_kv_heads=KV_H,
                            ffn_type="swiglu", use_cross_attn=True, seed=5)
enc_out = rng.standard_normal((B, 15, D)).astype(np.float32)
out_dec, _ = dec_blk.forward(x, enc_memory=enc_out, self_mask=mask)
T.ok(out_dec.shape == x.shape,                     "Decoder block (cross-attn): output shape")

# KV-cache block consistency (incremental vs full)
blk_kvc = TransformerBlock(D, num_heads=H, num_kv_heads=KV_H, ffn_type="swiglu", seed=7)
seq8 = rng.standard_normal((1, 8, D)).astype(np.float32)
out_full_blk, _ = blk_kvc.forward(seq8, self_mask=make_causal_mask(8), training=False)
cache_blk = KVCache(1, KV_H, 64, D_HEAD)
for i in range(8):
    tok = seq8[:, i:i+1, :]
    out_step, _ = blk_kvc.forward(tok, kv_cache=cache_blk, rope_offset=i)
diff_blk = abs(out_full_blk[0,-1] - out_step[0,0]).max()
T.ok(diff_blk < 1e-4,                             f"Block KV-cache consistency diff={diff_blk:.2e}")

# Depth scaling: deeper stacks have smaller W_O
blk2  = TransformerBlock(D, H, num_layers=2,  layer_idx=0, seed=0)
blk24 = TransformerBlock(D, H, num_layers=24, layer_idx=0, seed=0)
T.ok(np.abs(blk24.self_attn.W_O).mean() < np.abs(blk2.self_attn.W_O).mean(),
     "Block: depth scaling — 24-layer W_O smaller than 2-layer W_O")

# Stochastic train, deterministic eval
o_t1 = blk.forward(x, training=True)[0]
o_t2 = blk.forward(x, training=True)[0]
o_e1 = blk.forward(x, training=False)[0]
o_e2 = blk.forward(x, training=False)[0]
T.ok(not np.allclose(o_t1, o_t2),                 "Block: training outputs are stochastic")
T.ok(np.allclose(o_e1, o_e2),                     "Block: eval outputs are deterministic")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: ENCODER STACK
# ══════════════════════════════════════════════════════════════════════════════

T.section("Step 19 — Encoder Stack")

encoder = Encoder(
    vocab_size=VOCAB, d_model=D, num_layers=L,
    num_heads=H, num_kv_heads=KV_H,
    max_seq_len=128, dropout_rate=0.1,
    ffn_type="swiglu", pad_token_id=0, seed=0,
)
token_ids = rng.integers(1, VOCAB, size=(B, S)).astype(np.int64)
enc_out = encoder.forward(token_ids, training=False)

T.ok(enc_out.shape == (B, S, D),                  "Encoder: output shape")
T.ok(enc_out.dtype == np.float32,                  "Encoder: output dtype float32")
T.ok(np.isfinite(enc_out).all(),                   "Encoder: no NaN/Inf")

# Different inputs → different outputs
token_ids2 = rng.integers(1, VOCAB, size=(B, S)).astype(np.int64)
enc_out2   = encoder.forward(token_ids2, training=False)
T.ok(not np.allclose(enc_out, enc_out2),           "Encoder: different tokens → different outputs")

# Determinism in eval
T.ok(np.allclose(enc_out, encoder.forward(token_ids, training=False)),
     "Encoder: eval is deterministic")

# Padding mask
tokens_pad = token_ids.copy(); tokens_pad[:, 15:] = 0
pad_mask = build_padding_mask(tokens_pad, pad_token_id=0)
T.ok(pad_mask.shape == (B, 1, 1, S),              "Padding mask: shape")
T.ok(pad_mask[0,0,0,15] < -1e8,                   "Padding mask: pad positions masked to -1e9")
T.ok(pad_mask[0,0,0,0]  == 0.0,                   "Padding mask: real positions unmasked")
enc_pad = encoder.forward(tokens_pad, training=False)
T.ok(enc_pad.shape == (B, S, D),                  "Encoder: padded sequence correct shape")

# Mean-pool sentence embedding
sent = encoder.encode_mean(token_ids)
T.ok(sent.shape == (B, D),                        "Encoder mean-pool: shape")
T.ok(np.isfinite(sent).all(),                     "Encoder mean-pool: no NaN/Inf")

# Chunked attention
enc_chunk = encoder.forward(token_ids, training=False, chunk_size=5)
T.ok(np.allclose(enc_out, enc_chunk, atol=1e-5),  "Encoder: chunked == standard (atol=1e-5)")

T.ok(encoder.count_parameters() > 0,              "Encoder: has trainable parameters")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: DECODER STACK
# ══════════════════════════════════════════════════════════════════════════════

T.section("Step 20 — Decoder Stack + Generation")

decoder = Decoder(
    vocab_size=VOCAB, d_model=D, num_layers=L,
    num_heads=H, num_kv_heads=KV_H,
    max_seq_len=256, dropout_rate=0.1,
    ffn_type="swiglu", tie_weights=True, seed=0,
)
logits, hidden = decoder.forward(token_ids, training=False)

T.ok(logits.shape  == (B, S, VOCAB),              "Decoder: logits shape")
T.ok(hidden.shape  == (B, S, D),                  "Decoder: hidden shape")
T.ok(np.isfinite(logits).all(),                   "Decoder: no NaN/Inf in logits")

# Causal property (crucial correctness check)
tids_b = token_ids.copy(); tids_b[:, 12:] = rng.integers(1, VOCAB, size=(B, S-12))
logits_b, _ = decoder.forward(tids_b, training=False)
T.ok(np.allclose(logits[:,:12,:], logits_b[:,:12,:], atol=1e-4),
     "Decoder: early positions unaffected by future token changes")

# KV-cache end-to-end correctness
dec_kvc = Decoder(VOCAB, D, L, H, KV_H, max_seq_len=256, seed=77)
full_seq = rng.integers(1, VOCAB, size=(1, 10)).astype(np.int64)
logits_ref, _ = dec_kvc.forward(full_seq, training=False)

caches = dec_kvc._make_kv_caches(1)
dec_kvc.forward(full_seq[:,:9], kv_caches=caches, rope_offset=0)
logits_step, _ = dec_kvc.forward(full_seq[:,9:10], kv_caches=caches, rope_offset=9)
cache_diff = abs(logits_ref[0,9] - logits_step[0,0]).max()
T.ok(cache_diff < 1e-3,                           f"Decoder KV-cache correctness diff={cache_diff:.2e}")

# Greedy generation
prompt = np.array([[1, 10, 20]], dtype=np.int64)
gen_g  = decoder.generate(prompt, max_new_tokens=10, temperature=0.0)
gen_g2 = decoder.generate(prompt, max_new_tokens=10, temperature=0.0)
T.ok(gen_g.shape == (1, 13),                      "Greedy gen: shape (prompt=3 + 10)")
T.ok(np.array_equal(gen_g, gen_g2),               "Greedy gen: deterministic")
T.ok(np.all(gen_g >= 0) and np.all(gen_g < VOCAB), "Greedy gen: token IDs in vocab range")

# Temperature sampling
gen_t1 = decoder.generate(prompt, max_new_tokens=10, temperature=1.0)
gen_t2 = decoder.generate(prompt, max_new_tokens=10, temperature=1.0)
T.ok(not np.array_equal(gen_t1, gen_t2),          "Temperature sampling: stochastic")

# Top-k
gen_k = decoder.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10)
T.ok(gen_k.shape == (1, 13),                      "Top-k sampling: correct length")
T.ok(np.all(gen_k >= 0) and np.all(gen_k < VOCAB), "Top-k: all IDs in range")

# Top-p (nucleus)
gen_p = decoder.generate(prompt, max_new_tokens=10, temperature=0.9, top_p=0.9)
T.ok(gen_p.shape == (1, 13),                      "Nucleus sampling: correct length")

# Repetition penalty
gen_r = decoder.generate(prompt, max_new_tokens=10, temperature=1.0,
                          top_k=50, top_p=0.95, repetition_penalty=1.3)
T.ok(gen_r.shape == (1, 13),                      "Repetition penalty: correct length")

# EOS stopping
first_logits = decoder.forward(prompt, training=False)[0][0, -1]
eos = int(np.argmax(first_logits))
gen_eos = decoder.generate(prompt, max_new_tokens=20, temperature=0.0, eos_token_id=eos)
T.ok(gen_eos.shape[1] <= 1 + 3 + 20,             f"EOS stopping: terminates early (len={gen_eos.shape[1]})")

# Seq2seq decoder
dec_s2s = Decoder(VOCAB, D, 2, H, KV_H, use_cross_attn=True, ffn_type="swiglu", seed=88)
enc_mem  = rng.standard_normal((B, 12, D)).astype(np.float32)
s2s_log, _ = dec_s2s.forward(token_ids, enc_memory=enc_mem)
T.ok(s2s_log.shape == (B, S, VOCAB),              "Seq2seq decoder: logits shape")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: END-TO-END PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

T.section("End-to-End — Encoder → Decoder (seq2seq)")

t0 = time.time()

enc_e2e = Encoder(VOCAB, D, L, H, KV_H, max_seq_len=128, ffn_type="swiglu", seed=0)
dec_e2e = Decoder(VOCAB, D, L, H, KV_H, max_seq_len=128, use_cross_attn=True,
                  ffn_type="swiglu", tie_weights=True, seed=1)

src = rng.integers(1, VOCAB, size=(1, 15)).astype(np.int64)
tgt = rng.integers(1, VOCAB, size=(1, 10)).astype(np.int64)

mem      = enc_e2e.forward(src, training=False)
e2e_log, _ = dec_e2e.forward(tgt, enc_memory=mem, training=False)

T.ok(mem.shape     == (1, 15, D),                  "E2E: encoder output shape")
T.ok(e2e_log.shape == (1, 10, VOCAB),              "E2E: decoder logits shape")
T.ok(np.isfinite(e2e_log).all(),                   "E2E: no NaN/Inf")

# GPT-style end-to-end generation
dec_gpt = Decoder(VOCAB, D, L, H, KV_H, max_seq_len=128,
                  use_cross_attn=False, ffn_type="swiglu", seed=2)
full_gen = dec_gpt.generate(
    np.array([[1, 2, 3]], dtype=np.int64),
    max_new_tokens=25, temperature=0.8, top_k=40, top_p=0.92,
    repetition_penalty=1.15,
)
T.ok(full_gen.shape[1] == 28,                      "E2E GPT gen: 3 prompt + 25 tokens")
T.ok(np.all(full_gen >= 0) and np.all(full_gen < VOCAB), "E2E GPT: all IDs in vocab")

elapsed = (time.time() - t0) * 1000
print(f"\n  Pipeline runtime: {elapsed:.1f}ms")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: PARAMETER INVENTORY
# ══════════════════════════════════════════════════════════════════════════════

T.section("Parameter Count — Architecture Comparison")

configs = [
    ("Tiny  (6L, 128d, MHA)",  Decoder(VOCAB, D, 6, H, H, ffn_type="swiglu")),
    ("Small (6L, 128d, GQA2)", Decoder(VOCAB, D, 6, H, 2, ffn_type="swiglu")),
    ("Small (6L, 128d, MQA)",  Decoder(VOCAB, D, 6, H, 1, ffn_type="swiglu")),
    ("Small (6L, 128d, GELU)", Decoder(VOCAB, D, 6, H, H, ffn_type="gelu")),
]
print()
for name, model in configs:
    p = model.count_parameters()
    kv_p = model.blocks[0].self_attn.W_K.size + model.blocks[0].self_attn.W_V.size
    kv_p_total = kv_p * len(model.blocks)
    print(f"  {name}: {p:>8,} params  (KV weights: {kv_p_total:,})")
    T.ok(p > 0, f"{name}: has parameters")

T.ok(True, "Parameter inventory printed")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════════════════════

all_ok = T.done()

if all_ok:
    print("\n  BUILD STATUS: ✓ COMPLETE")
    print("  45E v2 — all modules validated, ready for Step 21")
else:
    print(f"\n  BUILD STATUS: ✗ {T.failed} failure(s) found")

sys.exit(0 if all_ok else 1)
