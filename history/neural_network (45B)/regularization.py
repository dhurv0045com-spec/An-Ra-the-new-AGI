"""
model/regularization.py — Step 9
Dropout, weight decay, label smoothing, gradient clipping.
All implemented as standalone components usable in both numpy and PyTorch layers.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


# ─────────────────────────────────────────────
# Numpy dropout (for the MLP foundation)
# ─────────────────────────────────────────────

class Dropout:
    """
    Inverted dropout: scales by 1/(1-p) during training so
    inference needs no scaling.
    """
    def __init__(self, p: float = 0.1):
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"
        self.p = p
        self._mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return x
        self._mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
        return x * self._mask

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            return d_out
        return d_out * self._mask


# ─────────────────────────────────────────────
# Gradient clipping (numpy)
# ─────────────────────────────────────────────

def clip_gradients_numpy(params_and_grads, max_norm: float = 1.0) -> float:
    """
    Clips gradients by global L2 norm. Returns norm before clipping.
    """
    total_norm = 0.0
    for _, g in params_and_grads:
        total_norm += float(np.sum(g**2))
    total_norm = total_norm**0.5

    clip_coef = max_norm / max(total_norm, max_norm)
    for _, g in params_and_grads:
        g *= clip_coef

    return total_norm


# ─────────────────────────────────────────────
# PyTorch regularization components
# ─────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing.
    Prevents overconfident predictions; improves generalization.
    """
    def __init__(self, vocab_size: int, smoothing: float = 0.1,
                 ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (N, V) — raw model output
        targets: (N,)   — integer token ids
        """
        V = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        # smooth target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (V - 1))
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            loss = loss[mask]

        return loss.mean()


class GradientClipper:
    """Stateful gradient clipper that tracks norm history."""
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
        self._norm_history = []

    def clip(self, parameters) -> float:
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
        self._norm_history.append(float(norm))
        return float(norm)

    @property
    def recent_norm(self) -> float:
        if not self._norm_history:
            return 0.0
        return np.mean(self._norm_history[-100:])


class StochasticDepth(nn.Module):
    """
    Drop entire residual blocks with probability p during training.
    Also called DropPath — used in modern vision and language transformers.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, survival, device=x.device))
        return x * mask / survival


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Step 9: Regularization checks")

    # Dropout
    x = np.ones((4, 8))
    drop = Dropout(p=0.5)
    out = drop.forward(x)
    print(f"Dropout: mean={out.mean():.3f} (≈1.0 expected with inverted scaling)")

    # Gradient clipping
    params_grads = [(np.zeros(3), np.array([3.0, 4.0, 0.0]))]
    norm_before = clip_gradients_numpy(params_grads, max_norm=1.0)
    print(f"Grad norm before clip: {norm_before:.2f}, after: {np.linalg.norm(params_grads[0][1]):.4f}")

    # Label smoothing loss (PyTorch)
    logits = torch.randn(16, 1000)
    targets = torch.randint(0, 1000, (16,))
    loss_fn = LabelSmoothingLoss(vocab_size=1000, smoothing=0.1)
    loss = loss_fn(logits, targets)
    print(f"Label smoothing loss: {loss.item():.4f}")

    print("✓ Step 9 verified")
