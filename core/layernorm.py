"""
================================================================================
FILE: layernorm.py
PROJECT: Transformer Language Model — 45E v2
STEP: 17 — Normalization Layers
================================================================================

PRIMARY: RMSNorm (Root Mean Square Normalization)
  Used by: LLaMA, Mistral, Falcon, Gemma, GPT-NeoX
  
  RMSNorm(x) = (x / RMS(x)) * γ
  RMS(x)     = sqrt( mean(x²) + ε )

  Differences from LayerNorm:
    - No mean subtraction (no "re-centering")
    - No learned bias β (only scale γ)
    - ~10% faster than LayerNorm (one less operation)
    - Empirically equivalent or better quality

  Why RMSNorm wins:
    The mean-subtraction in LayerNorm is expensive and empirically unnecessary.
    The scale γ is sufficient. LLaMA used RMSNorm and achieved better
    throughput with no quality loss.

SECONDARY: LayerNorm
  Used by: GPT-2, BERT, original Transformer, T5
  Included for compatibility and ablation.

PLACEMENT STRATEGY: Pre-norm (always)
  pre-norm:  x = x + sublayer(norm(x))   ← what we use (GPT-2, LLaMA, Mistral)
  post-norm: x = norm(x + sublayer(x))   ← original 2017 paper
  
  Pre-norm is more stable at depth. Deep post-norm models need careful
  learning rate warmup and often explode without it.
================================================================================
"""

import numpy as np


class RMSNorm:
    """
    Root Mean Square Layer Normalization — Zhang & Sennrich, 2019.

    Normalizes by RMS only (no mean subtraction), then scales by γ.

    Formula:
        RMS(x)     = sqrt( mean(x², axis=-1, keepdims=True) + ε )
        RMSNorm(x) = (x / RMS(x)) * γ

    Args:
        d_model: Feature dimension to normalize over
        eps:     Stability constant (prevents /0; 1e-6 matches LLaMA default)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps     = eps
        # γ (gain/scale): learned, initialized to 1 — starts as identity norm
        self.gamma   = np.ones(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply RMS normalization to last dimension.

        Computation:
          ms   = mean(x², axis=-1, keepdims=True)   ← mean of squares
          rms  = sqrt(ms + ε)                         ← root mean square
          x_n  = x / rms                              ← normalize
          out  = gamma * x_n                          ← learned scale

        Args:
            x: (..., d_model) — any number of leading dimensions

        Returns:
            Same shape as x, normalized and scaled
        """
        # Compute mean of squares across the feature dimension
        ms = (x * x).mean(axis=-1, keepdims=True)     # (..., 1)

        # RMS: square root of mean squares + epsilon for stability
        rms = np.sqrt(ms + self.eps)                   # (..., 1)

        # Normalize and scale
        return self.gamma * (x / rms)                  # (..., d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def count_parameters(self) -> int:
        return self.gamma.size   # only γ, no β

    def __repr__(self) -> str:
        return f"RMSNorm(d_model={self.d_model}, eps={self.eps}, params={self.count_parameters()})"


class LayerNorm:
    """
    Layer Normalization — Ba et al., 2016.

    Normalizes across the feature dimension within each example.
    Zero-centers (subtracts mean), unit-variance (divides by std),
    then applies learned affine transform (γ, β).

    Formula:
        μ = mean(x, axis=-1, keepdims=True)
        σ² = var(x, axis=-1, keepdims=True)
        x̂ = (x - μ) / sqrt(σ² + ε)
        out = γ * x̂ + β

    Args:
        d_model: Feature dimension
        eps:     Stability constant (PyTorch default: 1e-5)
        bias:    Whether to include the β shift parameter (default: True)
    """

    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True):
        self.d_model = d_model
        self.eps     = eps
        self.bias    = bias
        self.gamma   = np.ones(d_model, dtype=np.float32)   # scale
        self.beta    = np.zeros(d_model, dtype=np.float32) if bias else None  # shift

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = x.mean(axis=-1, keepdims=True)                 # (..., 1)
        # Use mean of squared deviations for variance (biased, matching PyTorch)
        var  = ((x - mean) ** 2).mean(axis=-1, keepdims=True) # (..., 1)
        x_hat = (x - mean) / np.sqrt(var + self.eps)          # (..., d_model)
        out = self.gamma * x_hat
        if self.beta is not None:
            out = out + self.beta
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def count_parameters(self) -> int:
        total = self.gamma.size
        if self.beta is not None:
            total += self.beta.size
        return total

    def __repr__(self) -> str:
        return (
            f"LayerNorm(d_model={self.d_model}, eps={self.eps}, "
            f"bias={self.bias}, params={self.count_parameters()})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  layernorm.py — Step 17 — Self-Test")
    print("=" * 68)

    rng = np.random.default_rng(42)
    B, S, D = 3, 8, 64

    x = rng.standard_normal((B, S, D)).astype(np.float32) * 10 + 3

    # ── RMSNorm basic ─────────────────────────────────────────────────────
    print(f"\n[1] RMSNorm")
    rms = RMSNorm(d_model=D)
    out_rms = rms(x)
    print(f"  {rms}")
    print(f"  Input  RMS: {np.sqrt((x**2).mean(-1)).mean():.3f}")
    out_rms_per = np.sqrt((out_rms**2).mean(-1))
    print(f"  Output RMS: {out_rms_per.mean():.4f}  (target: ≈ 1.0)")
    assert abs(out_rms_per.mean() - 1.0) < 0.02, f"RMSNorm RMS = {out_rms_per.mean()}"
    assert out_rms.shape == x.shape

    # ── RMSNorm gamma scaling ─────────────────────────────────────────────
    print(f"\n[2] RMSNorm gamma scaling")
    rms2 = RMSNorm(D)
    rms2.gamma[:] = 5.0   # scale by 5
    out_rms2 = rms2(x)
    print(f"  With gamma=5: output RMS ≈ {np.sqrt((out_rms2**2).mean(-1)).mean():.3f}  (target: ≈ 5)")
    assert abs(np.sqrt((out_rms2**2).mean(-1)).mean() - 5.0) < 0.1

    # ── LayerNorm basic ───────────────────────────────────────────────────
    print(f"\n[3] LayerNorm")
    ln = LayerNorm(d_model=D)
    out_ln = ln(x)
    print(f"  {ln}")
    mean_out = out_ln.mean(-1)
    std_out  = out_ln.std(-1)
    print(f"  Output mean per position: {abs(mean_out).mean():.2e}  (target: ≈ 0)")
    print(f"  Output std  per position: {std_out.mean():.4f}  (target: ≈ 1)")
    assert abs(mean_out).max() < 1e-5, f"LayerNorm mean not zero: {abs(mean_out).max()}"
    assert abs(std_out.mean() - 1.0) < 0.02

    # ── LayerNorm vs RMSNorm difference ───────────────────────────────────
    print(f"\n[4] LayerNorm vs RMSNorm — they differ (by design)")
    max_diff = abs(out_ln - out_rms).max()
    print(f"  Max diff LN vs RMS: {max_diff:.4f}  (expected > 0)")
    assert max_diff > 0.01, "LN and RMSNorm should differ"

    # ── Per-position independence ─────────────────────────────────────────
    print(f"\n[5] Per-position independence")
    x_mod = x.copy()
    x_mod[:, 3, :] *= 1000.0
    out_rms_mod = rms(x_mod)
    diff_pos0 = abs(out_rms(x[:, :1, :]) - out_rms(x_mod[:, :1, :])).max()
    # Use different approach for position independence
    out1 = rms(x)
    out2 = rms(x_mod)
    diff_p0 = abs(out1[:, 0, :] - out2[:, 0, :]).max()
    print(f"  Position 0 unchanged after scaling position 3: {diff_p0 < 1e-6}  (diff: {diff_p0:.2e})")
    assert diff_p0 < 1e-6

    # ── Speed comparison ──────────────────────────────────────────────────
    print(f"\n[6] Numerical properties — large input stability")
    x_large = np.full((1, 1, D), 1e4, dtype=np.float32)
    out_rms_large = rms(x_large)
    out_ln_large  = ln(x_large)
    print(f"  RMSNorm on 1e4 input: max={out_rms_large.max():.4f}  (no inf/nan: {np.isfinite(out_rms_large).all()})")
    print(f"  LayerNorm on 1e4 input: max={out_ln_large.max():.4f}  (no inf/nan: {np.isfinite(out_ln_large).all()})")

    # ── Parameter count ───────────────────────────────────────────────────
    print(f"\n[7] Parameter counts")
    print(f"  RMSNorm(D={D}):           {rms.count_parameters()} params  (γ only)")
    print(f"  LayerNorm(D={D}):         {ln.count_parameters()} params  (γ + β)")
    print(f"  LayerNorm(D={D}, no bias):{LayerNorm(D, bias=False).count_parameters()} params")

    print("\n  ✓ All normalization tests passed")
    print("=" * 68)
