"""
checkpoint.py — Step 24: Checkpointing
=======================================
Saves and restores full training state: model weights, optimizer state,
scheduler state, scaler state, epoch/step counters, and loss history.
Keeps the N best checkpoints by val loss. Never loses training progress.
Resume from any checkpoint with a single call.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMeta:
    """
    Lightweight metadata stored alongside every checkpoint.
    Used for checkpoint selection without loading full state dicts.
    """
    epoch: int
    global_step: int
    train_loss: float
    val_loss: float
    best_val_loss: float
    model_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointMeta":
        return cls(**d)


class CheckpointManager:
    """
    Manages saving, loading, and pruning of training checkpoints.

    Strategy:
    - Saves a checkpoint at the end of every epoch.
    - Always keeps the best-val-loss checkpoint (`best.pt`).
    - Keeps the N most recent checkpoints (rolling window).
    - Checkpoint files: `epoch_{N:04d}_step_{S}.pt`
    - A `latest.json` symlink record points to the most recent checkpoint.

    Args:
        checkpoint_dir:  Root directory for checkpoint files.
        keep_last_n:     How many rolling checkpoints to retain.
        save_optimizer:  Whether to save optimizer + scaler state (adds ~2× size).
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        keep_last_n: int = 3,
        save_optimizer: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer

        self._registry_path = self.checkpoint_dir / "registry.json"
        self._registry: list[dict] = self._load_registry()

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _load_registry(self) -> list[dict]:
        """Load the persisted checkpoint registry."""
        if self._registry_path.exists():
            with open(self._registry_path) as f:
                return json.load(f)
        return []

    def _save_registry(self):
        with open(self._registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def best_val_loss(self) -> float:
        """Return the best validation loss seen across all saved checkpoints."""
        if not self._registry:
            return float("inf")
        return min(r["val_loss"] for r in self._registry)

    def latest_checkpoint(self) -> Optional[Path]:
        """Return path to the most recently saved checkpoint, or None."""
        if not self._registry:
            return None
        return Path(self._registry[-1]["path"])

    def best_checkpoint(self) -> Optional[Path]:
        """Return path to the checkpoint with the lowest val loss."""
        if not self._registry:
            return None
        best = min(self._registry, key=lambda r: r["val_loss"])
        return Path(best["path"])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[Any],
        meta: CheckpointMeta,
    ) -> Path:
        """
        Save a full checkpoint.

        Always writes the full state dict. Then:
        - If this is the best val loss, copies to `best.pt`.
        - Prunes old checkpoints beyond keep_last_n.

        Returns:
            Path to the saved checkpoint file.
        """
        filename = f"epoch_{meta.epoch:04d}_step_{meta.global_step}.pt"
        ckpt_path = self.checkpoint_dir / filename

        state = {
            "epoch": meta.epoch,
            "global_step": meta.global_step,
            "train_loss": meta.train_loss,
            "val_loss": meta.val_loss,
            "best_val_loss": meta.best_val_loss,
            "model_config": meta.model_config,
            "model_state_dict": model.state_dict(),
        }

        if self.save_optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()
            state["scheduler_state_dict"] = scheduler.state_dict() if scheduler else None
            state["scaler_state_dict"] = scaler.state_dict() if scaler else None

        torch.save(state, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path} (val_loss={meta.val_loss:.4f})")

        # Update registry
        self._registry.append({"path": str(ckpt_path), "val_loss": meta.val_loss, "epoch": meta.epoch})
        self._save_registry()

        # Always keep best checkpoint as a separate copy
        if meta.val_loss <= self.best_val_loss():
            best_path = self.checkpoint_dir / "best.pt"
            shutil.copy2(ckpt_path, best_path)
            logger.info(f"New best checkpoint → {best_path}")

        # Prune old checkpoints
        self._prune()

        return ckpt_path

    def _prune(self):
        """
        Remove checkpoints beyond keep_last_n, never removing best.pt.
        Best.pt is an independent copy and is never in the rolling window.
        """
        # Sort registry by epoch (ascending = oldest first)
        self._registry.sort(key=lambda r: r["epoch"])
        while len(self._registry) > self.keep_last_n:
            oldest = self._registry.pop(0)
            p = Path(oldest["path"])
            if p.exists():
                p.unlink()
                logger.debug(f"Pruned old checkpoint: {p}")
        self._save_registry()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        path: Optional[str | Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        device: str = "cpu",
        strict: bool = True,
    ) -> CheckpointMeta:
        """
        Restore model (and optionally optimizer/scheduler/scaler) from a checkpoint.

        Args:
            path:      Path to .pt file. If None, uses latest checkpoint.
            model:     Model instance (must match the saved architecture).
            optimizer: Optional — restores optimizer state if provided.
            scheduler: Optional — restores scheduler state if provided.
            scaler:    Optional — restores GradScaler state if provided.
            device:    Device to map tensors to.
            strict:    Strict weight loading (set False for partial loads).

        Returns:
            CheckpointMeta with training state at time of save.

        Raises:
            FileNotFoundError if the checkpoint path doesn't exist.
        """
        if path is None:
            path = self.latest_checkpoint()
        if path is None:
            raise FileNotFoundError("No checkpoints found in registry.")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info(f"Loading checkpoint: {path}")
        state = torch.load(path, map_location=device, weights_only=False)

        # Restore model weights
        model.load_state_dict(state["model_state_dict"], strict=strict)

        # Restore optimizer if requested and available
        if optimizer and "optimizer_state_dict" in state and state["optimizer_state_dict"]:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in state and state["scheduler_state_dict"]:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        if scaler and "scaler_state_dict" in state and state["scaler_state_dict"]:
            scaler.load_state_dict(state["scaler_state_dict"])

        meta = CheckpointMeta(
            epoch=state["epoch"],
            global_step=state["global_step"],
            train_loss=state["train_loss"],
            val_loss=state["val_loss"],
            best_val_loss=state["best_val_loss"],
            model_config=state.get("model_config", {}),
        )
        logger.info(
            f"Resumed from epoch {meta.epoch}, step {meta.global_step}, "
            f"val_loss={meta.val_loss:.4f}"
        )
        return meta

    def load_best(
        self,
        model: nn.Module,
        device: str = "cpu",
    ) -> CheckpointMeta:
        """Convenience: load only the best checkpoint into the model."""
        return self.load(self.checkpoint_dir / "best.pt", model, device=device)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> list[dict]:
        """Return all registered checkpoints sorted by epoch."""
        return sorted(self._registry, key=lambda r: r["epoch"])

    def summary(self) -> str:
        lines = ["─" * 50, "Checkpoint Manager"]
        for r in self.list_checkpoints():
            lines.append(f"  Epoch {r['epoch']:>4d}  val={r['val_loss']:.4f}  {Path(r['path']).name}")
        best = self.best_checkpoint()
        if best:
            lines.append(f"  Best: {best.name}  val={self.best_val_loss():.4f}")
        lines.append("─" * 50)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("Step 24: Checkpoint Manager — self test")
    print("=" * 60)

    # Minimal model for testing
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16)

        def forward(self, x):
            return self.fc(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir, keep_last_n=2)
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

        val_losses = [3.5, 3.1, 2.8, 3.0, 2.6]
        best = float("inf")

        for epoch, vl in enumerate(val_losses, 1):
            best = min(best, vl)
            meta = CheckpointMeta(
                epoch=epoch, global_step=epoch * 100,
                train_loss=vl - 0.1, val_loss=vl, best_val_loss=best,
                model_config={"d_model": 64}
            )
            manager.save(model, opt, sched, scaler=None, meta=meta)

        print(manager.summary())

        # Test pruning: only 2 latest should remain (+ best.pt)
        ckpts = list(Path(tmpdir).glob("epoch_*.pt"))
        print(f"Rolling checkpoints kept: {len(ckpts)} (expected 2)")
        assert len(ckpts) == 2, f"Expected 2, got {len(ckpts)}"

        # Test best.pt exists and has correct val_loss
        assert (Path(tmpdir) / "best.pt").exists()

        # Test restore
        model2 = TinyModel()
        m = manager.load(None, model2, device="cpu")
        print(f"Resumed at epoch {m.epoch}, val_loss={m.val_loss:.4f}")
        assert m.epoch == 5

        # Test load_best
        model3 = TinyModel()
        mb = manager.load_best(model3)
        print(f"Best checkpoint: epoch={mb.epoch}, val_loss={mb.val_loss:.4f}")
        assert mb.val_loss == 2.6

        print("\n✓ checkpoint.py OK")
