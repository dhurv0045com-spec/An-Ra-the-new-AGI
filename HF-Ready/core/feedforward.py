"""
================================================================================
FILE: feedforward.py
PROJECT: Transformer Language Model — 45E v2
STEP: 16 — Position-wise Feed-Forward Network
================================================================================

Two implementations, both production-grade:

  SWIGLU (default — LLaMA, PaLM, Mistral, Gemma, GPT-4 rumored)
  ────────────────────────────────────────────────────────────────
    FFN_SwiGLU(x) = ( SiLU(x @ W_gate) ⊙ (x @ W_up) ) @ W_down

    Three weight matrices: W_gate, W_up, W_down.
    Two parallel projections (gate + up) multiplied elementwise (GLU).
    SiLU activation: x * sigmoid(x) — smooth, non-zero gradient everywhere.

    WHY: Empirically outperforms GELU FFN by ~0.5-1 PPL on language modeling.
    The gating mechanism lets the network learn which features to "pass through"
    and which to suppress, giving more expressive power per parameter.

    DIMENSION CHOICE: Standard LLaMA uses d_ff = (8/3) × d_model, rounded to
    a multiple of 256 for hardware efficiency. With SwiGLU the parameter count
    is: 3 × d_model × d_ff vs 2 × d_model × d_ff for standard FFN.
    To keep total params equal, use d_ff = (2/3) × 4 × d_model ≈ 2.67 × d_model.

  STANDARD GELU FFN (included as fallback / for comparison)
  ────────────────────────────────────────────────────────────────
    FFN_GELU(x) = GELU( x @ W1 + b1 ) @ W2 + b2

    The original transformer FFN. Two matrices, one activation.
    Used by GPT-2, BERT, original Transformer.
================================================================================
"""

import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# ACTIVATION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU / Swish activation: x * sigmoid(x).

    Smooth, non-monotonic, non-zero for negative inputs (unlike ReLU).
    Self-gating: output magnitude is input-modulated.
    Used in SwiGLU — the gate branch passes through SiLU.

    Properties:
      silu(0)  = 0
      silu(x) → x as x → +∞
      silu(x) → 0 as x → -∞
      Has a slight bump below 0 (minimum at x ≈ -1.28)
    """
    return x / (1.0 + np.exp(-x))   # x * σ(x)


def gelu_approx(x: np.ndarray) -> np.ndarray:
    """
    GELU activation — GPT-2 tanh approximation.

    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    sqrt_2_over_pi = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x ** 3)))


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """
    Exact GELU via the error function: x * Φ(x) where Φ is the CDF of N(0,1).
    Uses scipy.special.ndtr if available, falls back to approximation.
    """
    try:
        from scipy.special import ndtr
        return x * ndtr(x)
    except ImportError:
        return gelu_approx(x)


# ──────────────────────────────────────────────────────────────────────────────
# SWIGLU FEED-FORWARD NETWORK
# ──────────────────────────────────────────────────────────────────────────────

def _make_ffn_dim(d_model: int, multiplier: float = 8/3, multiple_of: int = 64) -> int:
    """
    Compute SwiGLU d_ff: scale d_model by multiplier, round up to multiple_of.

    LLaMA uses 8/3 × d_model rounded to nearest multiple of 256.
    We use multiple_of=64 for smaller test models.

    Args:
        d_model:     Model dimension
        multiplier:  Target expansion ratio (8/3 ≈ 2.667 for LLaMA)
        multiple_of: Round up to this multiple (hardware alignment)
    """
    d_ff = int(d_model * multiplier)
    # Round UP to nearest multiple_of
    d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)
    return d_ff


class SwiGLUFeedForward:
    """
    SwiGLU position-wise feed-forward network.

    Architecture:
      gate = SiLU( x @ W_gate )        ← gating branch
      up   = x @ W_up                  ← value branch
      hidden = gate ⊙ up               ← elementwise gate (GLU)
      output = hidden @ W_down         ← project back to d_model

    Three weight matrices: W_gate, W_up (both: d_model → d_ff),
                           W_down (d_ff → d_model).

    No bias terms — following LLaMA, which omits bias everywhere.
    Biases add parameters without improving quality in large models.

    Args:
        d_model:      Input/output dimension
        d_ff:         Hidden dimension. None → auto-compute (8/3 × d_model, rounded)
        dropout_rate: Applied after hidden, before W_down
        multiple_of:  Round d_ff to this multiple for hardware alignment
        seed:         RNG seed
    """

    def __init__(
        self,
        d_model:      int,
        d_ff:         Optional[int] = None,
        dropout_rate: float = 0.0,
        multiple_of:  int   = 64,
        seed:         int   = 0,
    ):
        self.d_model      = d_model
        self.d_ff         = d_ff if d_ff is not None else _make_ffn_dim(d_model, 8/3, multiple_of)
        self.dropout_rate = dropout_rate
        self.rng          = np.random.default_rng(seed)

        # Xavier-uniform initialization for all three projections
        self.W_gate = self._xavier(d_model, self.d_ff)  # gate branch
        self.W_up   = self._xavier(d_model, self.d_ff)  # value branch
        self.W_down = self._xavier(self.d_ff, d_model)  # output projection

    def _xavier(self, fan_in: int, fan_out: int) -> np.ndarray:
        """Xavier uniform weight initialization."""
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Apply SwiGLU FFN.

        Gating computation:
          gate_pre = x @ W_gate               (batch, seq, d_ff)
          up_pre   = x @ W_up                 (batch, seq, d_ff)
          hidden   = SiLU(gate_pre) * up_pre  (elementwise — the GLU gate)
          output   = hidden @ W_down          (batch, seq, d_model)

        Args:
            x:        (batch, seq, d_model)
            training: Apply dropout if True

        Returns:
            (batch, seq, d_model)
        """
        # Gate branch: will be activated by SiLU
        gate = x @ self.W_gate          # (batch, seq, d_ff)

        # Value branch: passes through raw
        up   = x @ self.W_up            # (batch, seq, d_ff)

        # GLU: elementwise product of activated gate and raw up
        hidden = silu(gate) * up        # (batch, seq, d_ff) — gated, activated

        # Dropout on the hidden representation
        if training and self.dropout_rate > 0.0:
            keep = 1.0 - self.dropout_rate
            dmask = (self.rng.random(hidden.shape) < keep).astype(np.float32)
            hidden = hidden * dmask / keep

        # Project back to d_model
        return hidden @ self.W_down     # (batch, seq, d_model)

    def count_parameters(self) -> int:
        """Total parameters: W_gate + W_up + W_down."""
        return self.W_gate.size + self.W_up.size + self.W_down.size

    def __repr__(self) -> str:
        return (
            f"SwiGLUFeedForward(d_model={self.d_model}, d_ff={self.d_ff}, "
            f"ratio={self.d_ff/self.d_model:.2f}×, params={self.count_parameters():,})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# STANDARD GELU FEED-FORWARD (original transformer / GPT-2 style)
# ──────────────────────────────────────────────────────────────────────────────

class GELUFeedForward:
    """
    Standard two-layer position-wise FFN with GELU activation.

    FFN(x) = GELU( x @ W1 + b1 ) @ W2 + b2

    With d_ff = 4 × d_model (standard) or custom.

    Included for:
      - Compatibility with GPT-2/BERT-style models
      - Ablation comparison against SwiGLU
      - Architectures where bias is required

    Args:
        d_model:      Input/output dimension
        d_ff:         Hidden dimension (default: 4 × d_model)
        dropout_rate: Applied after GELU, before W2
        use_bias:     Whether to include bias terms (default: True)
        seed:         RNG seed
    """

    def __init__(
        self,
        d_model:      int,
        d_ff:         Optional[int] = None,
        dropout_rate: float = 0.1,
        use_bias:     bool  = True,
        seed:         int   = 0,
    ):
        self.d_model      = d_model
        self.d_ff         = d_ff if d_ff is not None else 4 * d_model
        self.dropout_rate = dropout_rate
        self.rng          = np.random.default_rng(seed)

        limit1 = np.sqrt(6.0 / (d_model + self.d_ff))
        limit2 = np.sqrt(6.0 / (self.d_ff  + d_model))

        self.W1 = self.rng.uniform(-limit1, limit1, (d_model, self.d_ff)).astype(np.float32)
        self.W2 = self.rng.uniform(-limit2, limit2, (self.d_ff, d_model)).astype(np.float32)
        self.b1 = np.zeros(self.d_ff,   dtype=np.float32) if use_bias else None
        self.b2 = np.zeros(d_model, dtype=np.float32) if use_bias else None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Apply GELU FFN."""
        hidden = x @ self.W1
        if self.b1 is not None:
            hidden = hidden + self.b1
        hidden = gelu_approx(hidden)

        if training and self.dropout_rate > 0.0:
            keep = 1.0 - self.dropout_rate
            dmask = (self.rng.random(hidden.shape) < keep).astype(np.float32)
            hidden = hidden * dmask / keep

        out = hidden @ self.W2
        if self.b2 is not None:
            out = out + self.b2
        return out

    def count_parameters(self) -> int:
        total = self.W1.size + self.W2.size
        if self.b1 is not None:
            total += self.b1.size + self.b2.size
        return total

    def __repr__(self) -> str:
        return (
            f"GELUFeedForward(d_model={self.d_model}, d_ff={self.d_ff}, "
            f"ratio={self.d_ff/self.d_model:.1f}×, params={self.count_parameters():,})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  feedforward.py — Step 16 — Self-Test")
    print("=" * 68)

    rng = np.random.default_rng(42)
    B, S, D = 2, 12, 128

    x = rng.standard_normal((B, S, D)).astype(np.float32)

    # ── SwiGLU ────────────────────────────────────────────────────────────
    print(f"\n[1] SwiGLU FFN")
    swiglu = SwiGLUFeedForward(d_model=D, dropout_rate=0.0)
    out_s = swiglu.forward(x, training=False)
    print(f"  {swiglu}")
    print(f"  Input:  {x.shape}  Output: {out_s.shape}")
    assert out_s.shape == x.shape

    # ── SwiGLU auto-dim matches LLaMA formula ─────────────────────────────
    print(f"\n[2] SwiGLU d_ff auto-computation (8/3 × d_model, rounded to 64)")
    for d in [64, 128, 256, 512, 1024, 4096]:
        d_ff_auto = _make_ffn_dim(d, 8/3, 256)
        print(f"  d_model={d:>5} → d_ff={d_ff_auto:>5} ({d_ff_auto/d:.3f}×)")
        assert d_ff_auto % 256 == 0, "Not a multiple of 256!"

    # ── GELU FFN ──────────────────────────────────────────────────────────
    print(f"\n[3] GELU FFN (GPT-2 style)")
    gelu_ffn = GELUFeedForward(d_model=D, d_ff=4*D, dropout_rate=0.1)
    out_g = gelu_ffn.forward(x, training=False)
    print(f"  {gelu_ffn}")
    print(f"  Output: {out_g.shape}")
    assert out_g.shape == x.shape

    # ── Position-wise independence ─────────────────────────────────────────
    print(f"\n[4] Position-wise independence")
    x2 = x.copy()
    x2[:, 5, :] *= 500.0
    out_s2 = swiglu.forward(x2, training=False)
    pos0_same = np.allclose(out_s[:, 0, :], out_s2[:, 0, :])
    print(f"  Changing position 5 does not affect position 0: {pos0_same}")
    assert pos0_same

    # ── SiLU vs GELU comparison ────────────────────────────────────────────
    print(f"\n[5] Activation function comparison")
    vals = np.array([-3., -1., -0.5, 0., 0.5, 1., 3.], dtype=np.float32)
    print(f"  {'x':>6} | {'SiLU':>8} | {'GELU':>8}")
    print(f"  {'─'*6}-+-{'─'*8}-+-{'─'*8}")
    for v, s, g in zip(vals, silu(vals), gelu_approx(vals)):
        print(f"  {v:>6.2f} | {s:>8.4f} | {g:>8.4f}")

    # ── Parameter comparison ───────────────────────────────────────────────
    print(f"\n[6] Parameter count comparison at d_model=128")
    print(f"  SwiGLU (d_ff auto):  {swiglu.count_parameters():>8,}")
    print(f"  GELU   (d_ff=4×D):   {gelu_ffn.count_parameters():>8,}")
    swiglu_4x = SwiGLUFeedForward(d_model=D, d_ff=4*D)
    print(f"  SwiGLU (d_ff=4×D):   {swiglu_4x.count_parameters():>8,}  ← +50% for 3 matrices")

    # ── Dropout stochasticity ─────────────────────────────────────────────
    print(f"\n[7] Dropout — active in training, absent in eval")
    swiglu_d = SwiGLUFeedForward(d_model=D, dropout_rate=0.2)
    out_t1 = swiglu_d.forward(x, training=True)
    out_t2 = swiglu_d.forward(x, training=True)
    out_e  = swiglu_d.forward(x, training=False)
    out_e2 = swiglu_d.forward(x, training=False)
    print(f"  Two training passes differ: {not np.allclose(out_t1, out_t2)}")
    print(f"  Two eval passes equal:      {np.allclose(out_e, out_e2)}")

    print("\n  ✓ All feed-forward tests passed")
    print("=" * 68)
