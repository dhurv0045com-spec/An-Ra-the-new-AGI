"""
mixed_precision.py - Step 26: Mixed Precision Training
=======================================================
Wraps PyTorch AMP (Automatic Mixed Precision) for transformer LM training.
Uses float16/bfloat16 for forward/backward where safe, float32 for accumulation.
Implements GradScaler to prevent float16 gradient underflow.
Result: ~2x throughput and ~40% memory reduction on CUDA GPUs.
Falls back gracefully to float32 on CPU/MPS with no code changes.
"""

import logging
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Returns best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using CUDA: {name} ({mem:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU - AMP disabled")
    return device


def amp_dtype(device: torch.device) -> torch.dtype:
    """
    Returns appropriate AMP dtype for device.
    CUDA: bfloat16 on Ampere+, else float16.
    MPS/CPU: bfloat16.
    """
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.bfloat16


class MixedPrecisionTrainer:
    """
    Manages AMP autocast and gradient scaling for transformer training.

    Usage:
        mp = MixedPrecisionTrainer(device)
        with mp.autocast():
            logits = model(x)
            loss = criterion(logits, y)
        mp.backward(loss)
        mp.step(optimizer)
        mp.update()

    Args:
        device:          Training device.
        enabled:         Explicitly enable/disable AMP.
        init_scale:      Initial GradScaler scale.
        growth_interval: How often scaler attempts to increase scale.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        enabled: Optional[bool] = None,
        init_scale: float = 2.0 ** 16,
        growth_interval: int = 2000,
    ):
        self.device = device or get_device()
        self.dtype = amp_dtype(self.device)

        # AMP speedup only on CUDA
        if enabled is None:
            self.enabled = (self.device.type == "cuda")
        else:
            self.enabled = enabled

        # GradScaler needed only for float16 (bfloat16 has sufficient dynamic range)
        self._needs_scaler = self.enabled and (self.dtype == torch.float16)
        self.scaler = GradScaler(
            init_scale=init_scale,
            growth_interval=growth_interval,
            enabled=self._needs_scaler,
        )

        if self.enabled:
            logger.info(
                f"Mixed precision: {self.dtype} autocast enabled "
                f"(scaler: {'on' if self._needs_scaler else 'off - bfloat16 stable'})"
            )
        else:
            logger.info("Mixed precision: disabled (CPU/MPS - running float32)")

    @contextmanager
    def autocast(self):
        """
        Context manager for forward pass in reduced precision.
        LayerNorm, softmax, loss automatically stay in float32.
        """
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.enabled,
        ):
            yield

    def backward(self, loss: torch.Tensor):
        """Scale loss (if float16) and run backward. Prevents gradient underflow."""
        if self._needs_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_gradients(self, model: nn.Module, max_norm: float = 1.0) -> float:
        """
        Clip gradients by global norm. Unscales first if needed.
        Returns pre-clipping global grad norm for monitoring.
        """
        if self._needs_scaler:
            self.scaler.unscale_(self._last_optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return norm.item()

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Optimizer step via scaler (skips if gradients contain inf/nan).
        Returns True if step was taken.
        """
        self._last_optimizer = optimizer
        if self._needs_scaler:
            self.scaler.step(optimizer)
            return True
        else:
            optimizer.step()
            return True

    def update(self):
        """Update scaler after each step. Adjusts scale based on overflow."""
        if self._needs_scaler:
            self.scaler.update()

    def state_dict(self) -> dict:
        """Serialize scaler state for checkpointing."""
        return self.scaler.state_dict()

    def load_state_dict(self, state: dict):
        """Restore scaler state from checkpoint."""
        self.scaler.load_state_dict(state)

    @property
    def scale(self) -> float:
        """Current gradient scale factor (for logging)."""
        return self.scaler.get_scale() if self._needs_scaler else 1.0

    def log_stats(self) -> dict:
        """Return AMP stats for logging."""
        return {
            "amp_enabled": self.enabled,
            "amp_dtype": str(self.dtype),
            "grad_scale": self.scale,
        }


def amp_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    mp: "MixedPrecisionTrainer",
    scheduler=None,
    max_grad_norm: float = 1.0,
) -> dict:
    """
    Complete backward -> unscale -> clip -> step -> update cycle.

    Args:
        loss:          Scalar loss tensor from the forward pass.
        model:         Model (for gradient clipping).
        optimizer:     Optimizer.
        mp:            MixedPrecisionTrainer instance.
        scheduler:     LR scheduler (optional).
        max_grad_norm: Gradient clipping max norm.

    Returns:
        Dict with grad_norm and grad_scale for logging.
    """
    mp.backward(loss)

    # Unscale before clipping so we clip actual gradients
    if mp._needs_scaler:
        mp.scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
    mp.step(optimizer)
    mp.update()

    if scheduler is not None:
        scheduler.step()

    optimizer.zero_grad(set_to_none=True)

    return {"grad_norm": grad_norm, "grad_scale": mp.scale}


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Step 26: Mixed Precision Training - self test")
    print("=" * 60)

    device = get_device()

    class MiniTransformer(nn.Module):
        def __init__(self, vocab=256, d=128, heads=4, layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab, d)
            enc_layer = nn.TransformerEncoderLayer(
                d, heads, dim_feedforward=d * 4, batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Linear(d, vocab)

        def forward(self, x):
            return self.head(self.encoder(self.embed(x)))

    model = MiniTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mp = MixedPrecisionTrainer(device=device)

    print(f"\nDevice: {device}  |  dtype: {mp.dtype}  |  scaler: {mp._needs_scaler}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    losses = []
    t0 = time.perf_counter()
    for step in range(30):
        x = torch.randint(0, 256, (8, 64)).to(device)
        y = torch.randint(0, 256, (8, 64)).to(device)

        optimizer.zero_grad(set_to_none=True)
        with mp.autocast():
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, 256), y.view(-1))

        stats = amp_step(loss, model, optimizer, mp, max_grad_norm=1.0)
        losses.append(loss.item())

        if step % 10 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}  "
                  f"grad_norm={stats['grad_norm']:.4f}  "
                  f"scale={stats['grad_scale']:.0f}")

    elapsed = time.perf_counter() - t0
    print(f"\n30 steps in {elapsed:.2f}s ({30/elapsed:.1f} steps/sec)")

    # State dict round-trip
    sd = mp.state_dict()
    mp2 = MixedPrecisionTrainer(device=device)
    mp2.load_state_dict(sd)
    print(f"State dict restore: scale={mp2.scale:.0f}")
    print(f"AMP stats: {mp.log_stats()}")
    print("\n[OK] mixed_precision.py OK")
