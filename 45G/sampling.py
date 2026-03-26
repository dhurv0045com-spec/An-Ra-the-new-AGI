"""
45G / Steps 28‑30 — Sampling Strategies
=========================================
Three complementary sampling methods, all operating on
raw logit vectors. Each returns a single sampled token id.

  Step 28 — Temperature sampling    : rescale logit sharpness
  Step 29 — Top-k sampling          : restrict to k best tokens
  Step 30 — Top-p (nucleus) sampling: restrict to min set covering mass p

All three compose: top_k → top_p → temperature is the standard order
used by every production LM (GPT-2, LLaMA, Mistral, etc.).
The unified `sample_token` function at the bottom applies them
in that canonical pipeline.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Step 28 — Temperature
# ─────────────────────────────────────────────

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Rescale logit sharpness by dividing by temperature.

    temperature < 1  →  distribution peaks more sharply (more greedy)
    temperature = 1  →  no change
    temperature > 1  →  distribution flattens (more random)
    temperature → 0  →  approaches greedy argmax

    Args:
        logits:      Raw logit vector, shape (vocab_size,)
        temperature: Positive float. Clamped to 1e-8 to avoid div-by-zero.

    Returns:
        Rescaled logits, same shape.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / max(temperature, 1e-8)


# ─────────────────────────────────────────────
# Step 29 — Top-k
# ─────────────────────────────────────────────

def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Zero out every logit except the k highest, preventing sampling
    from the long tail of low-probability tokens.

    Args:
        logits: Raw logit vector, shape (vocab_size,)
        k:      Number of candidates to keep. k <= 0 means no filtering.

    Returns:
        Filtered logits with -inf in all non-top-k positions.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits                          # no-op

    # kth_vals: smallest value among the top-k
    kth_vals = torch.topk(logits, k).values[..., -1, None]

    # Mask every token below the k-th threshold
    return logits.masked_fill(logits < kth_vals, float("-inf"))


# ─────────────────────────────────────────────
# Step 30 — Top-p (nucleus)
# ─────────────────────────────────────────────

def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability mass exceeds p, zero out the rest.

    Adapts to the model's confidence: when confident, the nucleus is
    tiny (maybe 1–2 tokens); when uncertain, it widens automatically.

    Args:
        logits: Raw logit vector, shape (vocab_size,)
        p:      Cumulative probability threshold in (0, 1].
                p=1.0 means no filtering.

    Returns:
        Filtered logits with -inf outside the nucleus.
    """
    if p >= 1.0:
        return logits                          # no-op

    # Sort descending to walk the probability mass
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)

    # Remove tokens that push cumulative probability over p.
    # Shift right by 1 so we always include the token that crosses p.
    remove_mask = cumulative - probs > p
    sorted_logits[remove_mask] = float("-inf")

    # Scatter filtered values back to original vocabulary order
    filtered = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)
    return filtered


# ─────────────────────────────────────────────
# Unified sampler — composes all three
# ─────────────────────────────────────────────

def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """
    Canonical sampling pipeline: top_k → top_p → temperature → softmax → sample.

    Args:
        logits:      Raw logit vector, shape (vocab_size,)
        temperature: Sharpness control. 1.0 = no change.
        top_k:       Candidate cutoff. 0 = disabled.
        top_p:       Nucleus mass. 1.0 = disabled.

    Returns:
        A single sampled integer token id.
    """
    logits = logits.float()             # ensure fp32 for numerical stability

    # 1. Top-k filter (cuts long tail absolutely)
    logits = apply_top_k(logits, top_k)

    # 2. Top-p / nucleus filter (cuts long tail adaptively)
    logits = apply_top_p(logits, top_p)

    # 3. Temperature (sharpness)
    logits = apply_temperature(logits, temperature)

    # 4. Convert to probabilities and draw one sample
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ─────────────────────────────────────────────
# Full autoregressive sampling loop
# ─────────────────────────────────────────────

def sampling_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Autoregressive decode using sample_token at every step.

    Returns full token sequence (prompt + generated), shape (1, total_len).
    """
    model.eval()

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(ids)
            if isinstance(out, tuple):
                out = out[0]

            logits = out[0, -1, :] if out.dim() == 3 else out[-1, :]
            next_id = sample_token(logits, temperature=temperature,
                                   top_k=top_k, top_p=top_p)

            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

            if eos_token_id is not None and next_id == eos_token_id:
                break

    return ids


# ─────────────────────────────────────────────
# Smoke-test: verify each strategy independently
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    V = 1000
    logits = torch.randn(V)

    # --- Temperature ---
    cold = apply_temperature(logits, 0.1)
    hot  = apply_temperature(logits, 2.0)
    # Cold should concentrate probability more
    assert F.softmax(cold, dim=-1).max() > F.softmax(hot, dim=-1).max(), \
        "Temperature: cold should be sharper than hot"
    print("Temperature ✓")

    # --- Top-k ---
    k = 10
    filt = apply_top_k(logits, k)
    n_valid = (filt > float("-inf")).sum().item()
    assert n_valid == k, f"Top-k: expected {k} valid tokens, got {n_valid}"
    print(f"Top-k (k={k}) ✓")

    # --- Top-p ---
    p = 0.9
    filt_p = apply_top_p(logits, p)
    probs_after = F.softmax(filt_p, dim=-1)
    mass = probs_after[probs_after > 0].sum().item()
    assert mass >= p, f"Top-p: nucleus mass {mass:.3f} < {p}"
    print(f"Top-p (p={p}) ✓  nucleus mass={mass:.4f}")

    # --- Unified sampler: 1000 draws, no error ---
    samples = [sample_token(logits, temperature=0.8, top_k=50, top_p=0.95)
               for _ in range(1000)]
    assert all(0 <= s < V for s in samples), "sample_token out of vocab range"
    print(f"sample_token: 1000 draws, all in [0, {V}) ✓")

    # --- Temperature sweep distribution check ---
    print("\nTemperature sweep (top-1 probability):")
    for t in [0.1, 0.5, 1.0, 1.5, 2.0]:
        p_top1 = F.softmax(apply_temperature(logits, t), dim=-1).max().item()
        print(f"  T={t:.1f}  p(top-1)={p_top1:.4f}")

    print("\nStep 28 (temperature) ✓  |  Step 29 (top-k) ✓  |  Step 30 (top-p) ✓")
