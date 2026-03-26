"""
loss_tracker.py — Step 23: Loss Curve Tracking
===============================================
Tracks training and validation loss across steps and epochs.
Detects overfitting via val/train divergence.
Saves loss history to JSON and plots curves with matplotlib.
Designed to be injected into the trainer with zero coupling.
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = logging.getLogger(__name__)


class LossTracker:
    """
    Records, persists, and visualizes training and validation loss.

    Features:
    - Per-step and per-epoch loss history
    - Overfitting detection via val/train ratio
    - JSON persistence (resume-safe)
    - Matplotlib loss curve plots saved to disk
    - Running average smoothing for noisy train loss

    Args:
        log_dir:        Directory to write history JSON and plot PNGs.
        smoothing:      Exponential moving average factor for train loss display.
        overfit_ratio:  val_loss/train_loss threshold that triggers a warning.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        smoothing: float = 0.95,
        overfit_ratio: float = 1.3,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.smoothing = smoothing
        self.overfit_ratio = overfit_ratio

        # Step-level records: every optimizer step
        self.train_steps: list[int] = []
        self.train_losses: list[float] = []
        self.train_smooth: list[float] = []  # EMA-smoothed

        # Epoch-level records: end of each epoch
        self.val_epochs: list[int] = []
        self.val_losses: list[float] = []
        self.train_epoch_losses: list[float] = []  # mean train loss per epoch

        # Internal EMA state
        self._ema: Optional[float] = None

        # Best validation loss seen
        self.best_val_loss: float = float("inf")
        self.best_val_epoch: int = -1

        self._history_path = self.log_dir / "loss_history.json"
        self._plot_path = self.log_dir / "loss_curves.png"

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_train_step(self, step: int, loss: float) -> float:
        """
        Record a single training step loss.
        Returns the smoothed EMA loss for logging.
        """
        self.train_steps.append(step)
        self.train_losses.append(loss)

        # Exponential moving average — surfaces trends through noisy batches
        if self._ema is None:
            self._ema = loss
        else:
            self._ema = self.smoothing * self._ema + (1 - self.smoothing) * loss
        self.train_smooth.append(self._ema)

        return self._ema

    def record_epoch(self, epoch: int, val_loss: float, train_loss: float):
        """
        Record end-of-epoch validation loss.
        Detects overfitting and tracks the best checkpoint epoch.
        """
        self.val_epochs.append(epoch)
        self.val_losses.append(val_loss)
        self.train_epoch_losses.append(train_loss)

        # Overfitting detection
        if train_loss > 0 and (val_loss / train_loss) > self.overfit_ratio:
            logger.warning(
                f"[Epoch {epoch}] Overfitting detected: "
                f"val={val_loss:.4f} train={train_loss:.4f} "
                f"ratio={val_loss/train_loss:.2f}x"
            )

        # Track best val
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_epoch = epoch
            logger.info(f"[Epoch {epoch}] New best val loss: {val_loss:.4f}")

        self.save()
        self.plot()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Serialize full history to JSON — safe for resume."""
        history = {
            "train_steps": self.train_steps,
            "train_losses": self.train_losses,
            "train_smooth": self.train_smooth,
            "val_epochs": self.val_epochs,
            "val_losses": self.val_losses,
            "train_epoch_losses": self.train_epoch_losses,
            "best_val_loss": self.best_val_loss,
            "best_val_epoch": self.best_val_epoch,
        }
        with open(self._history_path, "w") as f:
            json.dump(history, f, indent=2)

    def load(self) -> bool:
        """
        Restore history from disk. Returns True if successful.
        Call this when resuming a training run.
        """
        if not self._history_path.exists():
            return False
        with open(self._history_path) as f:
            h = json.load(f)
        self.train_steps = h["train_steps"]
        self.train_losses = h["train_losses"]
        self.train_smooth = h["train_smooth"]
        self.val_epochs = h["val_epochs"]
        self.val_losses = h["val_losses"]
        self.train_epoch_losses = h["train_epoch_losses"]
        self.best_val_loss = h["best_val_loss"]
        self.best_val_epoch = h["best_val_epoch"]
        # Restore EMA state from last recorded smooth value
        if self.train_smooth:
            self._ema = self.train_smooth[-1]
        logger.info(f"Loss history restored: {len(self.train_steps)} train steps, "
                    f"{len(self.val_epochs)} val epochs")
        return True

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, save_path: Optional[str] = None):
        """
        Generate and save a two-panel loss curve figure:
        - Top: step-level train loss (raw + smoothed EMA)
        - Bottom: epoch-level train vs val loss
        """
        save_path = save_path or str(self._plot_path)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.patch.set_facecolor("#0f1117")

        for ax in axes:
            ax.set_facecolor("#1a1d27")
            ax.tick_params(colors="#aaaaaa")
            ax.spines[:].set_color("#333344")
            ax.grid(True, color="#222233", linewidth=0.5, linestyle="--")
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        # --- Panel 1: step-level train loss ---
        ax0 = axes[0]
        if self.train_steps:
            ax0.plot(
                self.train_steps, self.train_losses,
                color="#3a86ff", alpha=0.25, linewidth=0.8, label="Train loss (raw)"
            )
            ax0.plot(
                self.train_steps, self.train_smooth,
                color="#3a86ff", linewidth=1.8, label="Train loss (EMA)"
            )
        ax0.set_xlabel("Step", color="#aaaaaa")
        ax0.set_ylabel("Loss", color="#aaaaaa")
        ax0.set_title("Training Loss (Step Level)", color="#ffffff", fontsize=13)
        ax0.legend(facecolor="#1a1d27", edgecolor="#333344", labelcolor="#cccccc")

        # --- Panel 2: epoch-level train vs val ---
        ax1 = axes[1]
        if self.val_epochs:
            ax1.plot(
                self.val_epochs, self.train_epoch_losses,
                color="#3a86ff", linewidth=2, marker="o", markersize=4, label="Train loss"
            )
            ax1.plot(
                self.val_epochs, self.val_losses,
                color="#ff6b6b", linewidth=2, marker="s", markersize=4, label="Val loss"
            )
            # Mark best val epoch
            if self.best_val_epoch >= 0 and self.best_val_epoch in self.val_epochs:
                bi = self.val_epochs.index(self.best_val_epoch)
                ax1.axvline(
                    self.best_val_epoch, color="#ffd166", linewidth=1,
                    linestyle=":", alpha=0.8
                )
                ax1.scatter(
                    [self.best_val_epoch], [self.val_losses[bi]],
                    color="#ffd166", zorder=5, s=80, label=f"Best val ({self.best_val_loss:.4f})"
                )
        ax1.set_xlabel("Epoch", color="#aaaaaa")
        ax1.set_ylabel("Loss", color="#aaaaaa")
        ax1.set_title("Train vs Validation Loss (Epoch Level)", color="#ffffff", fontsize=13)
        ax1.legend(facecolor="#1a1d27", edgecolor="#333344", labelcolor="#cccccc")

        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Loss curve saved → {save_path}")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of training progress."""
        lines = ["─" * 50, "Loss Tracker Summary"]
        lines.append(f"  Train steps logged : {len(self.train_steps):,}")
        if self.train_smooth:
            lines.append(f"  Current train loss : {self.train_smooth[-1]:.4f} (EMA)")
        if self.val_losses:
            lines.append(f"  Last val loss      : {self.val_losses[-1]:.4f}")
            lines.append(f"  Best val loss      : {self.best_val_loss:.4f} @ epoch {self.best_val_epoch}")
        lines.append("─" * 50)
        return "\n".join(lines)

    def perplexity(self, loss: float) -> float:
        """Convert cross-entropy loss to perplexity."""
        return math.exp(min(loss, 20))  # cap at e^20 to avoid overflow


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, random, math

    print("=" * 60)
    print("Step 23: Loss Tracker — self test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = LossTracker(log_dir=tmpdir, smoothing=0.95)

        # Simulate a training run: loss decays with noise
        step = 0
        for epoch in range(1, 6):
            epoch_losses = []
            for _ in range(50):
                # Simulated decaying noisy loss
                base = 4.0 * math.exp(-0.015 * step) + 0.5
                loss = base + random.gauss(0, 0.15)
                ema = tracker.record_train_step(step, loss)
                epoch_losses.append(loss)
                step += 1

            train_mean = sum(epoch_losses) / len(epoch_losses)
            val_loss = train_mean + random.uniform(0.05, 0.15)  # val slightly worse
            tracker.record_epoch(epoch, val_loss=val_loss, train_loss=train_mean)
            print(f"  Epoch {epoch}: train={train_mean:.4f}  val={val_loss:.4f}  ppl={tracker.perplexity(val_loss):.1f}")

        print()
        print(tracker.summary())

        # Test persistence
        tracker2 = LossTracker(log_dir=tmpdir)
        ok = tracker2.load()
        assert ok, "Load failed"
        assert len(tracker2.train_steps) == len(tracker.train_steps)
        assert tracker2.best_val_epoch == tracker.best_val_epoch

        plot_path = Path(tmpdir) / "loss_curves.png"
        assert plot_path.exists(), "Plot not generated"
        print(f"\nPlot saved: {plot_path.stat().st_size:,} bytes")
        print("\n✓ loss_tracker.py OK")
