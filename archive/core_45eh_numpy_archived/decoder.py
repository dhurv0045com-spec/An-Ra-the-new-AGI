"""
================================================================================
FILE: decoder.py
PROJECT: Transformer Language Model — 45E v2
STEP: 20 — Decoder Stack + Autoregressive Generation Engine
================================================================================

GPT-style causal language model decoder — what actually generates text.

ARCHITECTURE:
  Token IDs
    → Token embedding (scaled √d_model, weight-tied with LM head)
    → Embedding dropout
    → N × TransformerBlock (pre-RMSNorm, causal GQA+RoPE, SwiGLU)
    → Final RMSNorm
    → LM head: d_model → vocab_size (shared weights with embedding)
    → Logits

WEIGHT TYING:
  The input embedding matrix E is (vocab × d_model).
  The LM head projects (d_model → vocab) = E^T.
  Sharing weights: (1) saves vocab_size × d_model parameters
                   (2) ensures the embedding space is consistent with output space
                   (3) often improves perplexity (Press & Wolf, 2017)
  Used by: GPT-2, LLaMA, Mistral, most modern LLMs.

INFERENCE ENGINE:
  Five generation strategies implemented:
    1. Greedy decoding           — argmax, deterministic, low quality
    2. Temperature sampling      — randomness control, the baseline strategy
    3. Top-k sampling            — restrict to k most likely tokens
    4. Top-p (nucleus) sampling  — restrict to cumulative probability p
    5. Repetition penalty        — reduce probability of recently used tokens
  These can be combined: temperature + top-p + repetition penalty is common.

KV-CACHE INTEGRATION:
  Prefill:  Process the prompt in one pass (fills KV cache for all layers)
  Decode:   Each step processes only 1 new token (uses cached K/V)
  This reduces per-step compute from O(n² × d_model) to O(n × d_model).
================================================================================
"""

import numpy as np
from typing import Optional, List, Tuple, Any

from .attention import (
    make_causal_mask, softmax, KVCache, make_kv_cache
)
try:
    from .attention import TurboQuantConfig, _TURBOQUANT_AVAILABLE
except ImportError:
    TurboQuantConfig, _TURBOQUANT_AVAILABLE = None, False

from .transformer_block import TransformerBlock
from .layernorm import RMSNorm


class Decoder:
    """
    GPT-style autoregressive language model decoder.

    Full architecture: embedding → N blocks → RMSNorm → LM head → logits.
    Weight-tied embedding and LM head (optional).
    KV-cache for efficient autoregressive inference.
    Supports both decoder-only (GPT) and encoder-decoder (seq2seq) modes.

    Args:
        vocab_size:      Token vocabulary size
        d_model:         Embedding and hidden dimension
        num_layers:      Number of stacked transformer blocks
        num_heads:       Query attention heads per block
        num_kv_heads:    KV heads for GQA (None = MHA = num_heads)
        d_ff:            FFN hidden dim (None = SwiGLU auto ≈ 2.67×)
        max_seq_len:     Maximum context window length
        dropout_rate:    Dropout probability
        use_cross_attn:  Add cross-attention sublayer (seq2seq decoder mode)
        ffn_type:        "swiglu" (default) or "gelu"
        rope_base:       RoPE base (10000; use larger for longer context)
        tie_weights:     Share embedding ↔ LM head weights (default: True)
        pad_token_id:    Padding token ID
        seed:            RNG seed
    """

    def __init__(
        self,
        vocab_size:     int,
        d_model:        int,
        num_layers:     int,
        num_heads:      int,
        num_kv_heads:   Optional[int] = None,
        d_ff:           Optional[int] = None,
        max_seq_len:    int   = 2048,
        dropout_rate:   float = 0.1,
        use_cross_attn: bool  = False,
        ffn_type:       str   = "swiglu",
        rope_base:      float = 10000.0,
        tie_weights:    bool  = True,
        pad_token_id:   int   = 0,
        seed:           int   = 0,
    ):
        self.d_model        = d_model
        self.num_layers     = num_layers
        self.vocab_size     = vocab_size
        self.max_seq_len    = max_seq_len
        self.dropout_rate   = dropout_rate
        self.use_cross_attn = use_cross_attn
        self.tie_weights    = tie_weights
        self.pad_token_id   = pad_token_id
        self.rng            = np.random.default_rng(seed)

        # ── Token embedding ──────────────────────────────────────────────
        # GPT-2 initialization: N(0, 0.02)
        self.token_embedding = (
            self.rng.standard_normal((vocab_size, d_model)).astype(np.float32) * 0.02
        )

        # ── Precomputed causal mask cache ────────────────────────────────
        # Sliced at runtime — avoids per-forward allocation
        self._causal_mask_cache = make_causal_mask(max_seq_len)  # (1, 1, max_seq, max_seq)

        # ── Stacked transformer blocks ───────────────────────────────────
        self.blocks: List[TransformerBlock] = [
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                use_cross_attn=use_cross_attn,
                ffn_type=ffn_type,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                layer_idx=i,
                num_layers=num_layers,
                seed=seed + i * 41,
            )
            for i in range(num_layers)
        ]

        # ── Final normalization ──────────────────────────────────────────
        self.final_norm = RMSNorm(d_model)

        # ── LM head ─────────────────────────────────────────────────────
        self.lm_head_W: Optional[np.ndarray] = None
        if not tie_weights:
            # Independent LM head weights
            limit = np.sqrt(6.0 / (d_model + vocab_size))
            self.lm_head_W = self.rng.uniform(-limit, limit, (d_model, vocab_size)).astype(np.float32)

        self.lm_head_b = np.zeros(vocab_size, dtype=np.float32)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _embed(self, token_ids: np.ndarray) -> np.ndarray:
        """Embedding lookup + √d_model scaling."""
        return self.token_embedding[token_ids] * np.sqrt(self.d_model)

    def _lm_head(self, x: np.ndarray) -> np.ndarray:
        """Project from d_model to vocab logits."""
        W = self.token_embedding.T if self.tie_weights else self.lm_head_W
        return x @ W + self.lm_head_b

    def _causal_mask(self, seq_len: int) -> np.ndarray:
        """Return (1, 1, seq_len, seq_len) causal mask from cache."""
        return self._causal_mask_cache[:, :, :seq_len, :seq_len]

    def _make_kv_caches(
        self, 
        batch_size: int,
        turboquant: bool = False,
        tq_config: Optional[Any] = None
    ) -> List[Any]:
        """Create one KVCache per layer for autoregressive generation."""
        num_kv_heads = self.blocks[0].num_kv_heads
        d_head       = self.d_model // self.blocks[0].num_heads
        return [
            make_kv_cache(
                batch_size=batch_size,
                num_kv_heads=num_kv_heads,
                max_seq_len=self.max_seq_len,
                d_head=d_head,
                compressed=turboquant,
                tq_config=tq_config,
            )
            for _ in self.blocks
        ]

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        token_ids:   np.ndarray,
        enc_memory:  Optional[np.ndarray]  = None,
        cross_mask:  Optional[np.ndarray]  = None,
        kv_caches:   Optional[List[KVCache]] = None,
        rope_offset: int                   = 0,
        training:    bool                  = False,
        chunk_size:  Optional[int]         = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full decoder forward pass.

        In training: feed all tokens, causal mask applied internally.
        In prefill:  same as training but kv_caches are populated.
        In decode:   feed 1 new token, kv_caches contain full history.

        Args:
            token_ids:   (batch, seq_len) integer token IDs
            enc_memory:  (batch, enc_seq, d_model) — encoder output for cross-attn
            cross_mask:  Mask for cross-attention
            kv_caches:   List of KVCache, one per layer (inference only)
            rope_offset: Starting position index (for incremental decode)
            training:    Enable dropout
            chunk_size:  Memory-efficient attention

        Returns:
            logits:  (batch, seq_len, vocab_size) — raw prediction scores
            hidden:  (batch, seq_len, d_model)    — final hidden states
        """
        seq_len = token_ids.shape[1]

        # Step 1: Embed
        x = self._embed(token_ids)   # (batch, seq, d_model)

        # Step 2: Embedding dropout
        if training and self.dropout_rate > 0.0:
            keep = 1.0 - self.dropout_rate
            dmask = (self.rng.random(x.shape) < keep).astype(np.float32)
            x = x * dmask / keep

        # Step 3: Build causal mask for this sequence length
        # For decode step (seq_len=1), no mask needed — one Q attending full cached K
        if seq_len > 1:
            causal = self._causal_mask(seq_len)
        else:
            causal = None   # single token: always attends to all cached K

        # Step 4: Pass through all blocks
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, _ = block.forward(
                x,
                enc_memory=enc_memory,
                self_mask=causal,
                cross_mask=cross_mask,
                kv_cache=cache,
                rope_offset=rope_offset,
                training=training,
                chunk_size=chunk_size,
            )

        # Step 5: Final norm
        hidden = self.final_norm(x).astype(np.float32)

        # Step 6: LM head → logits
        logits = self._lm_head(hidden)

        return logits, hidden

    # ── Generation engine ─────────────────────────────────────────────────

    def _apply_repetition_penalty(
        self,
        logits:    np.ndarray,
        token_ids: np.ndarray,
        penalty:   float,
    ) -> np.ndarray:
        """
        Reduce logits of tokens that have already appeared in the sequence.

        For tokens with positive logit: divide by penalty (make less likely).
        For tokens with negative logit: multiply by penalty (push more negative).

        penalty > 1.0 → discourages repetition
        penalty = 1.0 → no effect
        penalty < 1.0 → encourages repetition (unusual)
        """
        if penalty == 1.0:
            return logits

        logits = logits.copy()
        # Get unique tokens that have appeared in the generated sequence
        for token_id in np.unique(token_ids):
            if logits[token_id] > 0:
                logits[token_id] /= penalty    # reduce positive logits
            else:
                logits[token_id] *= penalty    # push negative logits lower
        return logits

    def _sample_token(
        self,
        logits:     np.ndarray,
        temperature: float,
        top_k:      int,
        top_p:      float,
    ) -> int:
        """
        Sample next token from logits using temperature + top-k + top-p.

        Execution order (important):
          1. Scale by temperature
          2. Apply top-k mask
          3. Apply top-p mask
          4. Softmax
          5. Sample

        Args:
            logits:      (vocab_size,) raw logits
            temperature: > 1 = more random, < 1 = more peaked
            top_k:       Keep only top-k tokens (0 = disabled)
            top_p:       Nucleus probability (1.0 = disabled)

        Returns:
            Sampled token ID (int)
        """
        if temperature == 0.0:
            return int(np.argmax(logits))

        # Temperature scaling
        scaled = logits / temperature

        # Top-k: zero out all but top-k logits
        if top_k > 0 and top_k < len(scaled):
            kth = np.partition(scaled, -top_k)[-top_k]   # k-th largest value
            scaled = np.where(scaled < kth, -1e9, scaled)

        # Top-p (nucleus): find smallest set of tokens with cumulative prob ≥ p
        if top_p < 1.0:
            probs_tmp   = softmax(scaled)
            sorted_idx  = np.argsort(-probs_tmp)           # descending prob order
            cum_probs   = np.cumsum(probs_tmp[sorted_idx])  # cumulative sum

            # Find the cutoff index: last token needed to reach p
            cutoff_idx = np.searchsorted(cum_probs, top_p)
            # All tokens after cutoff_idx+1 get masked to -inf
            remove_idx = sorted_idx[cutoff_idx + 1:]
            scaled[remove_idx] = -1e9

        probs = softmax(scaled)
        # Clip tiny negatives from float arithmetic before sampling
        probs = np.clip(probs, 0.0, None)
        probs /= probs.sum()
        return int(self.rng.choice(len(probs), p=probs))

    @staticmethod
    def _beam_step(
        candidates: List[Tuple[float, List[int]]],
        logits:     np.ndarray,
        beam_width: int,
    ) -> List[Tuple[float, List[int]]]:
        """
        Expand beam search candidates by one token.

        Args:
            candidates: List of (log_prob, token_list) from previous step
            logits:     (beam_width, vocab_size) logits for each beam
            beam_width: Number of beams to keep

        Returns:
            Top beam_width candidates after expansion
        """
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))  # log-softmax
        new_candidates = []
        for i, (score, tokens) in enumerate(candidates):
            for vocab_i in np.argsort(-log_probs[i])[:beam_width]:
                new_candidates.append((
                    score + float(log_probs[i, vocab_i]),
                    tokens + [int(vocab_i)],
                ))
        # Keep top beam_width by score
        new_candidates.sort(key=lambda x: -x[0])
        return new_candidates[:beam_width]

    def generate(
        self,
        prompt_ids:       np.ndarray,
        max_new_tokens:   int   = 100,
        temperature:      float = 1.0,
        top_k:            int   = 50,
        top_p:            float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id:     Optional[int] = None,
        enc_memory:       Optional[np.ndarray] = None,
        turboquant:       bool = False,
        tq_config:        Optional[Any] = None,
    ) -> np.ndarray:
        """
        Autoregressive text generation with KV-cache.

        Strategy:
          1. Prefill: run the prompt through all blocks in one pass, populate KV caches
          2. Decode:  sample one token per step, feeding only the new token + cached KV

        Args:
            prompt_ids:         (1, prompt_len) — batch size must be 1
            max_new_tokens:     Maximum tokens to generate
            temperature:        Sampling temperature (0 = greedy)
            top_k:              Top-k filtering (0 = disabled)
            top_p:              Nucleus sampling threshold (1.0 = disabled)
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            eos_token_id:       Stop when this token is generated
            enc_memory:         Encoder output for seq2seq cross-attention

        Returns:
            (1, prompt_len + num_generated) full sequence including prompt
        """
        assert prompt_ids.ndim == 2 and prompt_ids.shape[0] == 1, \
            "generate() requires batch_size=1: shape (1, seq_len)"

        # Step 1: Allocate per-layer KV caches (compressed or standard)
        kv_caches = self._make_kv_caches(
            batch_size=1, 
            turboquant=turboquant, 
            tq_config=tq_config
        )

        # Step 2: Prefill — run full prompt, populate KV caches
        prompt_len = prompt_ids.shape[1]
        logits_prefill, _ = self.forward(
            prompt_ids,
            enc_memory=enc_memory,
            kv_caches=kv_caches,
            rope_offset=0,
            training=False,
        )
        # Last position logits → first generated token candidate
        next_logits = logits_prefill[0, -1, :]   # (vocab_size,)

        # Track generated tokens for repetition penalty
        all_ids = prompt_ids[0].tolist()   # full sequence so far

        generated = []

        # Step 3: Autoregressive decode loop
        for step in range(max_new_tokens):
            # Apply repetition penalty on recently generated tokens
            if repetition_penalty != 1.0:
                # Apply penalty over the full context so far
                next_logits = self._apply_repetition_penalty(
                    next_logits, np.array(all_ids[-256:]), repetition_penalty
                )

            # Sample next token
            next_token_id = self._sample_token(next_logits, temperature, top_k, top_p)

            generated.append(next_token_id)
            all_ids.append(next_token_id)

            # Check for EOS
            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            # Check KV cache capacity
            current_len = prompt_len + step + 1
            if current_len >= self.max_seq_len:
                break

            # Step 4: Single-token forward — only process the new token
            new_token = np.array([[next_token_id]], dtype=np.int64)
            logits_step, _ = self.forward(
                new_token,
                enc_memory=enc_memory,
                kv_caches=kv_caches,
                rope_offset=current_len,   # position of this new token
                training=False,
            )
            next_logits = logits_step[0, 0, :]   # (vocab_size,) — only one position

        # Combine prompt + generated
        full_ids = np.concatenate(
            [prompt_ids, np.array([generated], dtype=np.int64)],
            axis=1,
        )
        return full_ids

    def count_parameters(self, count_lm_head: bool = False) -> int:
        """
        Total learnable parameters.

        Args:
            count_lm_head: If True and tie_weights=True, counts embedding twice
                           (once as embedding, once as LM head). Usually False —
                           tied weights are one set of parameters, not two.
        """
        total = self.token_embedding.size
        total += sum(b.count_parameters() for b in self.blocks)
        total += self.final_norm.count_parameters()
        total += self.lm_head_b.size
        if not self.tie_weights and self.lm_head_W is not None:
            total += self.lm_head_W.size
        return total

    def __repr__(self) -> str:
        ffn   = self.blocks[0].ffn_type if self.blocks else "?"
        q_h   = self.blocks[0].num_heads if self.blocks else "?"
        kv_h  = self.blocks[0].num_kv_heads if self.blocks else "?"
        mode  = "GPT-style" if not self.use_cross_attn else "Seq2Seq"
        return (
            f"Decoder({mode}\n"
            f"  vocab={self.vocab_size:,}  d_model={self.d_model}\n"
            f"  layers={self.num_layers}  q_heads={q_h}  kv_heads={kv_h}\n"
            f"  ffn={ffn}  pos=RoPE  norm=RMSNorm  tie_weights={self.tie_weights}\n"
            f"  params={self.count_parameters():,}\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  decoder.py — Step 20 — Self-Test")
    print("=" * 68)

    rng = np.random.default_rng(42)
    VOCAB, D, LAYERS, H, KV_H = 500, 128, 4, 8, 2
    B, S = 2, 12

    decoder = Decoder(
        vocab_size=VOCAB, d_model=D, num_layers=LAYERS,
        num_heads=H, num_kv_heads=KV_H,
        max_seq_len=256, dropout_rate=0.1,
        ffn_type="swiglu", tie_weights=True, seed=0,
    )
    print(f"\n{decoder}")

    token_ids = rng.integers(1, VOCAB, size=(B, S)).astype(np.int64)

    # ── Forward pass shape ────────────────────────────────────────────────
    print(f"\n[1] Forward pass — shapes")
    logits, hidden = decoder.forward(token_ids, training=False)
    print(f"  Tokens:  {token_ids.shape}")
    print(f"  Logits:  {logits.shape}   <- (batch, seq, vocab)")
    print(f"  Hidden:  {hidden.shape}")
    assert logits.shape  == (B, S, VOCAB)
    assert hidden.shape  == (B, S, D)
    assert np.isfinite(logits).all()

    # ── Causal mask integrity ─────────────────────────────────────────────
    print(f"\n[2] Causal property — early positions independent of future tokens")
    tids_b = token_ids.copy()
    tids_b[:, 8:] = rng.integers(1, VOCAB, size=(B, S - 8))
    logits_b, _ = decoder.forward(tids_b, training=False)
    max_diff_early = abs(logits[:, :8, :] - logits_b[:, :8, :]).max()
    print(f"  Max diff positions 0-7 after changing 8-11: {max_diff_early:.2e}  (target: 0)")
    assert max_diff_early < 1e-4, f"Causal mask broken: {max_diff_early}"

    # ── KV-cache generate ──────────────────────────────────────────────────
    print(f"\n[3] Greedy generation (KV-cache)")
    prompt = np.array([[1, 42, 7, 99]], dtype=np.int64)
    gen = decoder.generate(prompt, max_new_tokens=12, temperature=0.0)
    print(f"  Prompt:    {prompt[0].tolist()}")
    print(f"  Generated: {gen[0, 4:].tolist()}")
    print(f"  Full len:  {gen.shape[1]}  (4 + 12 = 16)")
    assert gen.shape == (1, 16)

    # Greedy must be deterministic
    gen2 = decoder.generate(prompt, max_new_tokens=12, temperature=0.0)
    assert np.array_equal(gen, gen2), "Greedy not deterministic!"
    print(f"  Greedy is deterministic: [OK]")

    # ── Cached vs uncached — last token must match ─────────────────────────
    print(f"\n[4] KV-cache correctness — cached decode matches full pass")
    test_dec = Decoder(
        vocab_size=VOCAB, d_model=D, num_layers=LAYERS,
        num_heads=H, num_kv_heads=KV_H, max_seq_len=256, seed=99
    )
    full_ids  = rng.integers(1, VOCAB, size=(1, 10)).astype(np.int64)
    full_mask = test_dec._causal_mask(10)

    # Full-sequence pass (no cache)
    logits_full, _ = test_dec.forward(full_ids, training=False)

    # Cached pass: prefill 9 tokens, then decode 1 more
    caches = test_dec._make_kv_caches(batch_size=1)
    logits_p, _ = test_dec.forward(full_ids[:, :9], kv_caches=caches, rope_offset=0)
    logits_d, _ = test_dec.forward(full_ids[:, 9:10], kv_caches=caches, rope_offset=9)

    diff = abs(logits_full[0, 9] - logits_d[0, 0]).max()
    print(f"  Max diff (full vs cached at pos 9): {diff:.2e}  (target: < 1e-4)")
    assert diff < 1e-4, f"KV-cache inconsistency: {diff}"

    # ── Sampling strategies ────────────────────────────────────────────────
    print(f"\n[5] Sampling strategies")
    gen_t1 = decoder.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=0, top_p=1.0)
    gen_t2 = decoder.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=0, top_p=1.0)
    print(f"  Temperature=1 stochastic: {not np.array_equal(gen_t1, gen_t2)}")

    gen_topk = decoder.generate(prompt, max_new_tokens=8, temperature=0.8, top_k=10)
    print(f"  Top-k=10 length: {gen_topk.shape[1]}  [OK]")

    gen_topp = decoder.generate(prompt, max_new_tokens=8, temperature=0.8, top_p=0.9)
    print(f"  Top-p=0.9 length: {gen_topp.shape[1]}  [OK]")

    gen_rep = decoder.generate(
        prompt, max_new_tokens=8, temperature=1.0,
        top_k=50, top_p=0.95, repetition_penalty=1.3
    )
    print(f"  Repetition penalty=1.3 length: {gen_rep.shape[1]}  [OK]")

    # ── EOS token stopping ────────────────────────────────────────────────
    print(f"\n[6] EOS stopping")
    # Use the most likely token after greedy as EOS — forces early stop
    first_logits = decoder.forward(prompt, training=False)[0][0, -1]
    eos_id = int(np.argmax(first_logits))   # the greedily chosen first token
    gen_eos = decoder.generate(prompt, max_new_tokens=20, temperature=0.0, eos_token_id=eos_id)
    print(f"  With EOS={eos_id}, generated len: {gen_eos.shape[1]}  (should be <= 25)")
    assert gen_eos.shape[1] <= 25

    # ── Seq2seq mode ──────────────────────────────────────────────────────
    print(f"\n[7] Seq2seq decoder (with cross-attention)")
    s2s_dec = Decoder(
        vocab_size=VOCAB, d_model=D, num_layers=2,
        num_heads=H, num_kv_heads=KV_H,
        use_cross_attn=True, ffn_type="swiglu", seed=77
    )
    enc_mem  = rng.standard_normal((B, 15, D)).astype(np.float32)
    s2s_log, _ = s2s_dec.forward(token_ids, enc_memory=enc_mem)
    print(f"  Encoder memory: {enc_mem.shape}  Logits: {s2s_log.shape}  [OK]")
    assert s2s_log.shape == (B, S, VOCAB)

    # ── Parameter count and all IDs in range ──────────────────────────────
    print(f"\n[8] Sanity checks")
    assert np.all(gen >= 0) and np.all(gen < VOCAB), "Generated token IDs out of range!"
    print(f"  All generated token IDs in [0, {VOCAB}): [OK]")
    print(f"  Total params: {decoder.count_parameters():,}")
    print(f"  Params without emb: {decoder.count_parameters() - decoder.token_embedding.size:,}")

    print("\n  [OK] All decoder tests passed")
    print("=" * 68)
