"""
training/trainer.py — Steps 21–25
Dataset loader, full LM training loop, loss tracking,
checkpointing, and learning rate scheduling.
"""

import os
import math
import time
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from ..config import ModelConfig
from ..model.regularization import LabelSmoothingLoss, GradientClipper


# ─────────────────────────────────────────────
# STEP 21 — Dataset loader
# ─────────────────────────────────────────────

class TextDataset(Dataset):
    """
    Token-level dataset for language modeling.
    Pre-tokenizes text and chunks into fixed-length sequences.
    """
    def __init__(self, token_ids: List[int], seq_len: int,
                 stride: Optional[int] = None):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        # Build sliding-window chunks
        self.chunks = []
        for start in range(0, len(token_ids) - seq_len, self.stride):
            chunk = token_ids[start: start + seq_len + 1]  # +1 for target
            self.chunks.append(chunk)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return {"input_ids": x, "labels": y}


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large corpora that don't fit in RAM.
    Yields packed sequences from a token stream.
    """
    def __init__(self, token_stream: Iterator[int], seq_len: int):
        self.token_stream = token_stream
        self.seq_len = seq_len

    def __iter__(self):
        buf = []
        for tok in self.token_stream:
            buf.append(tok)
            if len(buf) == self.seq_len + 1:
                x = torch.tensor(buf[:-1], dtype=torch.long)
                y = torch.tensor(buf[1:],  dtype=torch.long)
                yield {"input_ids": x, "labels": y}
                buf = []


def build_dataloader(dataset, batch_size: int, shuffle: bool = True,
                     num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if not isinstance(dataset, IterableDataset) else False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


# ─────────────────────────────────────────────
# STEP 25 — Learning rate scheduling
# ─────────────────────────────────────────────

class CosineWarmupScheduler:
    """
    Linear warmup → cosine decay → optional floor.
    The de-facto LR schedule for transformer training.
    """
    def __init__(self, optimizer, warmup_steps: int, max_steps: int,
                 min_lr: float = 1e-5, base_lr: Optional[float] = None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]["lr"]
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _get_lr(self) -> float:
        step = self._step
        if step < self.warmup_steps:
            return self.base_lr * step / max(1, self.warmup_steps)
        if step > self.max_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + cosine * (self.base_lr - self.min_lr)

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class LinearWarmupConstantScheduler:
    """Linear warmup then constant LR. Simpler, useful for short runs."""
    def __init__(self, optimizer, warmup_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = optimizer.param_groups[0]["lr"]
        self._step = 0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            lr = self.base_lr * self._step / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    @property
    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ─────────────────────────────────────────────
# STEP 23 — Loss tracking
# ─────────────────────────────────────────────

class MetricsTracker:
    def __init__(self, window: int = 100):
        self.window = window
        self._recent: List[float] = []
        self._all:    List[float] = []
        self._steps:  List[int]   = []
        self._step = 0

    def update(self, loss: float):
        self._step += 1
        self._recent.append(loss)
        self._all.append(loss)
        self._steps.append(self._step)
        if len(self._recent) > self.window:
            self._recent.pop(0)

    @property
    def smoothed(self) -> float:
        if not self._recent:
            return float("inf")
        return sum(self._recent) / len(self._recent)

    @property
    def perplexity(self) -> float:
        return math.exp(min(self.smoothed, 100))

    def to_dict(self) -> dict:
        return {
            "step": self._step,
            "loss": self.smoothed,
            "perplexity": self.perplexity,
            "all_losses": self._all,
            "steps": self._steps,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ─────────────────────────────────────────────
# STEP 24 — Checkpointing
# ─────────────────────────────────────────────

class Checkpointer:
    """Saves and loads training state. Keeps best + last N checkpoints."""

    def __init__(self, checkpoint_dir: str, keep_last: int = 3):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._saved: List[Path] = []
        self.best_loss = float("inf")

    def save(self, step: int, model: nn.Module, optimizer,
             scheduler, metrics: MetricsTracker,
             config: ModelConfig) -> Path:
        """Save complete checkpoint."""
        path = self.dir / f"step_{step:07d}.pt"
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_step": scheduler._step if hasattr(scheduler, "_step") else 0,
            "metrics": metrics.to_dict(),
            "config": config.__dict__,
        }, path)

        self._saved.append(path)

        # Save best checkpoint separately
        current_loss = metrics.smoothed
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            best_path = self.dir / "best.pt"
            torch.save(torch.load(path), best_path)
            logging.info(f"  ★ New best checkpoint: loss={current_loss:.4f}")

        # Remove old checkpoints beyond keep_last
        while len(self._saved) > self.keep_last:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()

        # Always keep a "latest" symlink
        latest = self.dir / "latest.pt"
        if latest.exists():
            latest.unlink()
        latest.symlink_to(path.name)

        return path

    def load(self, path: str, model: nn.Module, optimizer=None,
             scheduler=None, map_location: str = "cpu") -> dict:
        """Load checkpoint. Returns metadata dict."""
        ckpt = torch.load(path, map_location=map_location)
        model.load_state_dict(ckpt["model_state"])
        if optimizer and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and "scheduler_step" in ckpt:
            scheduler._step = ckpt["scheduler_step"]
        logging.info(f"Loaded checkpoint from {path} (step {ckpt.get('step', '?')})")
        return ckpt

    def load_latest(self, model: nn.Module, optimizer=None,
                    scheduler=None, map_location: str = "cpu") -> Optional[dict]:
        latest = self.dir / "latest.pt"
        if not latest.exists():
            return None
        return self.load(str(latest), model, optimizer, scheduler, map_location)


# ─────────────────────────────────────────────
# STEP 22 — Language model training loop
# ─────────────────────────────────────────────

class LanguageModelTrainer:
    """
    Full training loop with:
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Gradient clipping
    - LR scheduling
    - Periodic evaluation
    - Checkpointing
    - Loss tracking and logging
    """

    def __init__(self, model: nn.Module, config: ModelConfig,
                 train_dataset: Dataset,
                 val_dataset: Optional[Dataset] = None,
                 tokenizer=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

        # Device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.model = self.model.to(self.device)

        # Mixed precision
        self.use_amp = self.device.type == "cuda"
        self.dtype = torch.bfloat16 if (
            self.use_amp and torch.cuda.is_bf16_supported()
        ) else torch.float16
        self.scaler = GradScaler(enabled=self.use_amp and self.dtype == torch.float16)

        # Data
        self.train_loader = build_dataloader(
            train_dataset, config.batch_size, shuffle=True
        )
        self.val_loader = (
            build_dataloader(val_dataset, config.batch_size, shuffle=False)
            if val_dataset else None
        )

        # Optimizer — separate weight decay for non-norm, non-bias params
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["norm", "bias", "embedding"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = AdamW([
            {"params": decay_params,    "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.learning_rate, betas=(config.beta1, config.beta2))

        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )

        # Loss function
        self.loss_fn = LabelSmoothingLoss(
            vocab_size=config.vocab_size,
            smoothing=config.label_smoothing,
            ignore_index=config.pad_token_id,
        )

        # Supporting systems
        self.grad_clipper = GradientClipper(max_norm=config.grad_clip)
        self.metrics = MetricsTracker(window=100)
        self.checkpointer = Checkpointer(config.checkpoint_dir)

        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    os.path.join(config.log_dir, "train.log"),
                    mode="a"
                ) if os.path.isdir(config.log_dir) else logging.StreamHandler(),
            ]
        )
        self.log = logging.getLogger(__name__)

        self.global_step = 0
        self.best_val_loss = float("inf")

    def _forward_loss(self, batch: Dict) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        labels    = batch["labels"].to(self.device)

        with autocast(enabled=self.use_amp, dtype=self.dtype):
            logits, _ = self.model(input_ids)
            # Flatten: (B*T, V) and (B*T,)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = self.loss_fn(logits_flat, labels_flat)

        return loss

    @torch.no_grad()
    def evaluate(self) -> float:
        if not self.val_loader:
            return float("inf")
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for i, batch in enumerate(self.val_loader):
            if i >= self.config.eval_steps:
                break
            loss = self._forward_loss(batch)
            total_loss += loss.item()
            n_batches += 1
        self.model.train()
        return total_loss / max(n_batches, 1)

    def train(self, resume: bool = True) -> MetricsTracker:
        """Main training loop."""
        # Resume from checkpoint if available
        if resume:
            ckpt = self.checkpointer.load_latest(
                self.model, self.optimizer, self.scheduler,
                map_location=str(self.device)
            )
            if ckpt:
                self.global_step = ckpt.get("step", 0)
                self.log.info(f"Resumed from step {self.global_step}")

        self.model.train()
        self.optimizer.zero_grad()
        accum_loss = 0.0
        t0 = time.time()

        self.log.info(
            f"Training on {self.device} | "
            f"AMP={'bf16' if self.dtype==torch.bfloat16 else 'fp16' if self.use_amp else 'off'} | "
            f"steps={self.config.max_steps:,} | "
            f"params={sum(p.numel() for p in self.model.parameters()):,}"
        )

        data_iter = iter(self.train_loader)
        accum_steps = self.config.grad_accum_steps

        while self.global_step < self.config.max_steps:
            # ── Gradient accumulation micro-steps ──
            for micro_step in range(accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                loss = self._forward_loss(batch) / accum_steps
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            # ── Optimizer step ──
            self.scaler.unscale_(self.optimizer)
            grad_norm = self.grad_clipper.clip(self.model.parameters())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            lr = self.scheduler.step()
            self.metrics.update(accum_loss)
            self.global_step += 1
            accum_loss = 0.0

            # ── Logging ──
            if self.global_step % 50 == 0:
                dt = time.time() - t0
                tok_per_sec = (
                    50 * accum_steps * self.config.batch_size
                    * self.config.max_seq_len / max(dt, 1e-8)
                )
                self.log.info(
                    f"step {self.global_step:6d} | "
                    f"loss {self.metrics.smoothed:.4f} | "
                    f"ppl {self.metrics.perplexity:.2f} | "
                    f"lr {lr:.2e} | "
                    f"gnorm {grad_norm:.3f} | "
                    f"tok/s {tok_per_sec:.0f}"
                )
                t0 = time.time()

            # ── Evaluation ──
            if self.global_step % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                val_ppl = math.exp(min(val_loss, 100))
                self.log.info(
                    f"  [EVAL] step {self.global_step} | "
                    f"val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}"
                )

            # ── Checkpoint ──
            if self.global_step % self.config.save_interval == 0:
                path = self.checkpointer.save(
                    self.global_step, self.model, self.optimizer,
                    self.scheduler, self.metrics, self.config
                )
                self.log.info(f"  [CKPT] saved → {path}")

        # Final checkpoint
        self.checkpointer.save(
            self.global_step, self.model, self.optimizer,
            self.scheduler, self.metrics, self.config
        )
        self.log.info("Training complete.")
        return self.metrics
