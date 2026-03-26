"""
scheduler.py — Step 25: Learning Rate Scheduling
=================================================
Implements warmup + cosine annealing decay — the de-facto standard for
transformer language model training (GPT-2, GPT-3, LLaMA all use this).

Design:
  Phase 1 — Linear warmup: LR ramps from 0 → peak over `warmup_steps`.
             Avoids large gradient updates on randomly-initialized weights.
  Phase 2 — Cosine decay: LR decays from peak → min_lr following a cosine
             curve over the remaining `total_steps - warmup_steps`.

Built as a PyTorch LambdaLR wrapper so it integrates with any optimizer
and checkpoints correctly via scheduler.state_dict().
"""

import logging
import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core schedule function — used by get_cosine_schedule_with_warmup
# ---------------------------------------------------------------------------

def _cosine_warmup_lr_lambda(
    current_step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """
    Lambda function passed to LambdaLR.
    Returns the lr multiplier (relative to optimizer's base lr) at `current_step`.

    Args:
        current_step:  Current optimizer step (starts at 0).
        warmup_steps:  Steps to linearly ramp from 0 to 1.0.
        total_steps:   Total training steps.
        min_lr_ratio:  Final lr = peak_lr * min_lr_ratio (typically 0.1).
    """
    if current_step < warmup_steps:
        # Linear warmup: 0/warmup → warmup/warmup
        return float(current_step) / max(1, warmup_steps)

    # Cosine decay from peak (1.0) to min_lr_ratio
    progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)  # clamp — scheduler can be called past total_steps
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    # Scale cosine output into [min_lr_ratio, 1.0]
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Build a LambdaLR scheduler with linear warmup + cosine decay.

    Args:
        optimizer:     The optimizer whose LR this scheduler controls.
        warmup_steps:  Number of warmup steps.
        total_steps:   Total number of training steps.
        min_lr_ratio:  Final LR as a fraction of peak LR. Default 0.1
                       (i.e. final LR = peak_lr * 0.1). GPT-3 uses 0.1.
        last_epoch:    Used for resuming; pass global_step - 1.

    Returns:
        LambdaLR scheduler instance.
    """
    fn = lambda step: _cosine_warmup_lr_lambda(step, warmup_steps, total_steps, min_lr_ratio)
    return LambdaLR(optimizer, lr_lambda=fn, last_epoch=last_epoch)


# ---------------------------------------------------------------------------
# Convenience builder — creates optimizer + scheduler together
# ---------------------------------------------------------------------------

class TransformerScheduler:
    """
    Wraps optimizer and scheduler setup for transformer LM training.

    Reproduces the recipe from the GPT-3 / Chinchilla papers:
      - AdamW optimizer with β1=0.9, β2=0.95, ε=1e-8
      - Weight decay 0.1 (applied only to non-bias, non-norm parameters)
      - Peak LR typically 6e-4 for small models, 3e-4 for medium
      - Warmup: ~1% of total steps (Chinchilla) or 2000 steps (GPT-3)
      - Cosine decay to 10% of peak LR

    Args:
        model:          The transformer model.
        peak_lr:        Maximum learning rate.
        warmup_steps:   Steps for linear warmup.
        total_steps:    Total training steps (used for cosine decay).
        weight_decay:   L2 regularization coefficient.
        betas:          AdamW betas.
        eps:            AdamW epsilon.
        min_lr_ratio:   Final LR fraction.
        resume_step:    Global step to resume from (adjusts scheduler).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        peak_lr: float = 3e-4,
        warmup_steps: int = 2000,
        total_steps: int = 100_000,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        min_lr_ratio: float = 0.1,
        resume_step: int = 0,
    ):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # Separate weight-decayed vs non-decayed params
        # Biases and layernorm parameters should not be weight-decayed
        decay_params = [
            p for name, p in model.named_parameters()
            if p.requires_grad and p.dim() >= 2
        ]
        no_decay_params = [
            p for name, p in model.named_parameters()
            if p.requires_grad and p.dim() < 2
        ]

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups, lr=peak_lr, betas=betas, eps=eps
        )

        # If resuming, fast-forward the scheduler by passing last_epoch
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
            last_epoch=resume_step - 1,  # -1 means "not started"
        )

        n_decay = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in no_decay_params)
        logger.info(
            f"AdamW: {n_decay:,} params with weight decay, "
            f"{n_nodecay:,} without. Peak LR={peak_lr:.2e}, "
            f"warmup={warmup_steps}, total={total_steps}"
        )

    def step(self):
        """Advance optimizer and scheduler by one step."""
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def current_lr(self) -> float:
        """Return the current learning rate (for logging)."""
        return self.scheduler.get_last_lr()[0]

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tempfile
    from pathlib import Path

    print("=" * 60)
    print("Step 25: LR Scheduler — self test")
    print("=" * 60)

    model = nn.Linear(64, 64)
    peak_lr = 3e-4
    warmup = 200
    total = 2000

    ts = TransformerScheduler(
        model, peak_lr=peak_lr, warmup_steps=warmup, total_steps=total
    )

    lrs = []
    for step in range(total + 100):  # go slightly past total to test clamping
        lrs.append(ts.current_lr())
        ts.zero_grad()
        ts.step()

    print(f"LR at step 0:    {lrs[0]:.6f}  (expected ~0)")
    print(f"LR at step {warmup}: {lrs[warmup]:.6f}  (expected ~{peak_lr:.6f})")
    print(f"LR at step {total}: {lrs[total]:.6f}  (expected ~{peak_lr*0.1:.6f})")

    assert lrs[0] < 1e-7, "Warmup start should be near 0"
    assert abs(lrs[warmup] - peak_lr) < 1e-6, "LR at warmup end should be peak"
    assert lrs[total] < peak_lr * 0.15, "LR should have decayed significantly"

    # Save plot
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(lrs, color="#3a86ff", linewidth=2)
        ax.axvline(warmup, color="#ffd166", linestyle="--", label=f"End warmup ({warmup})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Cosine Schedule with Linear Warmup")
        ax.legend()
        plt.tight_layout()
        plot_path = Path(tmpdir) / "lr_schedule.png"
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"\nLR curve plot: {plot_path.stat().st_size:,} bytes")

    # Test state_dict / load_state_dict round-trip
    sd = ts.state_dict()
    ts2 = TransformerScheduler(model, peak_lr=peak_lr, warmup_steps=warmup, total_steps=total)
    ts2.load_state_dict(sd)
    assert abs(ts2.current_lr() - ts.current_lr()) < 1e-10, "LR mismatch after restore"
    print("State dict round-trip: OK")

    print("\n✓ scheduler.py OK")
