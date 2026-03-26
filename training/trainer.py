"""
trainer.py — Step 22: Language Model Training Loop
====================================================
Full end-to-end training loop for a decoder-only transformer language model.
Connects every piece built in steps 1-26:
  dataset.py      → data batches
  loss_tracker.py → loss logging and plotting
  checkpoint.py   → save/resume
  scheduler.py    → optimizer + LR schedule
  mixed_precision.py → float16/bfloat16 AMP

Training objective: predict next token at every position (causal LM).
Validates on held-out data. Logs everything. Resumes from any crash.
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LMDataModule
from loss_tracker import LossTracker
from checkpoint import CheckpointManager, CheckpointMeta
from scheduler import TransformerScheduler
from mixed_precision import MixedPrecisionTrainer, amp_step, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """
    All hyperparameters and paths for a training run.
    Designed to be serializable for logging and checkpointing.
    """
    # Model
    vocab_size: int = 50257            # GPT-2 BPE vocabulary
    d_model: int = 256                 # embedding dimension
    n_heads: int = 8                   # attention heads
    n_layers: int = 4                  # transformer blocks
    d_ff: int = 1024                   # feedforward hidden dim (4x d_model)
    seq_len: int = 256                 # context window (tokens)
    dropout: float = 0.1

    # Data
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    batch_size: int = 8
    num_workers: int = 2

    # Optimization
    peak_lr: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    total_steps: int = 50_000

    # Training control
    n_epochs: int = 10
    eval_every: int = 500              # steps between validation passes
    eval_batches: int = 50             # batches to use for validation
    log_every: int = 50                # steps between console logs
    save_every_epoch: bool = True

    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    data_cache: str = "./data_cache"

    # Resume
    resume_from: Optional[str] = None  # path to checkpoint, or "latest", or "best"

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Minimal decoder-only LM — wires into the transformer blocks from steps 1-20
# ---------------------------------------------------------------------------

class TransformerLM(nn.Module):
    """
    Decoder-only transformer language model (GPT-style).
    Uses causal masking so each position only attends to prior positions.
    This wires into the architecture built in steps 1-20.

    For the full pipeline, this can be replaced with the actual model
    built in steps 13-20 (MultiHeadAttention, TransformerBlock, DecoderStack).
    Here we use nn.TransformerDecoder for robustness without circular imports.

    Args:
        config: TrainerConfig with model hyperparameters.
    """

    def __init__(self, config: TrainerConfig):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len

        # Token + positional embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        # Causal transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,      # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with lm_head
        # Reduces parameters and improves generalization
        self.lm_head.weight = self.token_emb.weight

        # Causal mask (upper-triangular, registered as buffer so it moves with .to(device))
        mask = torch.triu(torch.ones(config.seq_len, config.seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (B, T) token indices.

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.seq_len, f"Sequence length {T} > context window {self.seq_len}"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(positions))

        # Causal mask for this sequence length
        causal = self.causal_mask[:T, :T]
        x = self.transformer(x, mask=causal, is_causal=True)
        x = self.ln_f(x)
        return self.lm_head(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates the full training pipeline.

    Responsibilities:
    - Setup: model, data, optimizer, scheduler, AMP, checkpointing, loss tracking
    - Training loop: forward, loss, backward, step
    - Validation: held-out perplexity
    - Checkpointing: periodic save, best-model tracking
    - Resuming: transparent restart from any saved checkpoint
    - Logging: console + file, loss curves

    Args:
        config: TrainerConfig.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = get_device()
        self.global_step = 0
        self.start_epoch = 0

        # --- Data ---
        logger.info("Setting up data pipeline...")
        self.data = LMDataModule(
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            cache_dir=config.data_cache,
        )
        self.data.setup()
        self.train_loader = self.data.train_loader()
        self.val_loader = self.data.val_loader()

        # --- Model ---
        logger.info("Building model...")
        self.model = TransformerLM(config).to(self.device)
        logger.info(f"Model parameters: {self.model.n_params():,}")

        # --- Optimizer + Scheduler ---
        self.ts = TransformerScheduler(
            model=self.model,
            peak_lr=config.peak_lr,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
        )

        # --- Mixed Precision ---
        self.mp = MixedPrecisionTrainer(device=self.device)

        # --- Loss Tracker ---
        self.tracker = LossTracker(log_dir=config.log_dir)

        # --- Checkpoint Manager ---
        self.ckpt = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            keep_last_n=3,
            save_optimizer=True,
        )

        # --- Resume ---
        if config.resume_from:
            self._resume(config.resume_from)

    def _resume(self, from_spec: str):
        """
        Resume training from a checkpoint.
        from_spec: "latest", "best", or a file path.
        """
        if from_spec == "latest":
            path = self.ckpt.latest_checkpoint()
        elif from_spec == "best":
            path = self.ckpt.best_checkpoint()
        else:
            path = Path(from_spec)

        if path is None or not path.exists():
            logger.warning(f"Resume checkpoint not found: {from_spec}. Starting fresh.")
            return

        meta = self.ckpt.load(
            path=path,
            model=self.model,
            optimizer=self.ts.optimizer,
            scheduler=self.ts.scheduler,
            scaler=self.mp.scaler,
            device=str(self.device),
        )

        self.global_step = meta.global_step
        self.start_epoch = meta.epoch  # resume from next epoch
        self.tracker.load()  # restore loss history

        logger.info(
            f"Resumed: epoch={meta.epoch} step={meta.global_step} "
            f"best_val={meta.best_val_loss:.4f}"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, n_batches: Optional[int] = None) -> float:
        """
        Run validation loop. Returns mean cross-entropy loss.

        Args:
            n_batches: If set, only evaluate on this many batches (faster).
        """
        self.model.eval()
        total_loss = 0.0
        n = 0

        for i, batch in enumerate(self.val_loader):
            if n_batches and i >= n_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with self.mp.autocast():
                logits = self.model(input_ids)
                # Flatten: (B, T, V) -> (B*T, V), labels: (B, T) -> (B*T,)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )

            total_loss += loss.item()
            n += 1

        self.model.train()
        return total_loss / max(n, 1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """
        Full training run across all epochs.
        Each epoch: iterate train_loader, periodic eval, checkpoint.
        """
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info(f"  Epochs:    {self.config.n_epochs}")
        logger.info(f"  Device:    {self.device}")
        logger.info(f"  AMP:       {self.mp.enabled} ({self.mp.dtype})")
        logger.info(f"  Batch:     {self.config.batch_size}")
        logger.info(f"  Seq len:   {self.config.seq_len}")
        logger.info(f"  Peak LR:   {self.config.peak_lr:.2e}")
        logger.info("=" * 60)

        self.model.train()
        best_val_loss = self.ckpt.best_val_loss()

        for epoch in range(self.start_epoch, self.config.n_epochs):
            epoch_losses = []
            t_epoch = time.perf_counter()

            for batch in self.train_loader:
                t_step = time.perf_counter()
                self.global_step += 1

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                # Forward
                self.ts.zero_grad()
                with self.mp.autocast():
                    logits = self.model(input_ids)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, self.config.vocab_size),
                        labels.view(-1),
                        ignore_index=-100,
                    )

                # Backward + step (AMP-aware)
                stats = amp_step(
                    loss=loss,
                    model=self.model,
                    optimizer=self.ts.optimizer,
                    mp=self.mp,
                    scheduler=self.ts.scheduler,
                    max_grad_norm=self.config.max_grad_norm,
                )

                loss_val = loss.item()
                epoch_losses.append(loss_val)
                ema = self.tracker.record_train_step(self.global_step, loss_val)

                # Console log
                if self.global_step % self.config.log_every == 0:
                    step_ms = (time.perf_counter() - t_step) * 1000
                    ppl = math.exp(min(ema, 20))
                    logger.info(
                        f"Epoch {epoch+1:>3d} | Step {self.global_step:>7,d} | "
                        f"loss={loss_val:.4f} (ema={ema:.4f}) | ppl={ppl:.1f} | "
                        f"lr={self.ts.current_lr():.2e} | "
                        f"gnorm={stats['grad_norm']:.3f} | "
                        f"{step_ms:.0f}ms/step"
                    )

                # Mid-epoch validation
                if self.global_step % self.config.eval_every == 0:
                    val_loss = self.evaluate(n_batches=self.config.eval_batches)
                    train_mean = sum(epoch_losses[-self.config.eval_batches:]) / max(
                        len(epoch_losses[-self.config.eval_batches:]), 1
                    )
                    logger.info(
                        f"[EVAL step {self.global_step:,}] "
                        f"val_loss={val_loss:.4f} | "
                        f"val_ppl={math.exp(min(val_loss,20)):.1f}"
                    )

                # Stop if we hit total_steps
                if self.global_step >= self.config.total_steps:
                    logger.info("Reached total_steps — stopping training.")
                    break

            # --- End of epoch ---
            epoch_time = time.perf_counter() - t_epoch
            train_mean = sum(epoch_losses) / max(len(epoch_losses), 1)
            val_loss = self.evaluate(n_batches=self.config.eval_batches)

            logger.info(
                f"\n{'='*60}\n"
                f"Epoch {epoch+1} complete ({epoch_time:.0f}s)\n"
                f"  Train loss: {train_mean:.4f}  ppl={math.exp(min(train_mean,20)):.1f}\n"
                f"  Val loss:   {val_loss:.4f}  ppl={math.exp(min(val_loss,20)):.1f}\n"
                f"{'='*60}"
            )

            self.tracker.record_epoch(epoch + 1, val_loss=val_loss, train_loss=train_mean)
            best_val_loss = min(best_val_loss, val_loss)

            if self.config.save_every_epoch:
                meta = CheckpointMeta(
                    epoch=epoch + 1,
                    global_step=self.global_step,
                    train_loss=train_mean,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    model_config=self.config.to_dict(),
                )
                self.ckpt.save(
                    model=self.model,
                    optimizer=self.ts.optimizer,
                    scheduler=self.ts.scheduler,
                    scaler=self.mp.scaler,
                    meta=meta,
                )

            if self.global_step >= self.config.total_steps:
                break

        logger.info("Training complete.")
        logger.info(self.tracker.summary())
        logger.info(self.ckpt.summary())


# ---------------------------------------------------------------------------
# Standalone test (smoke test — tiny config, runs fast)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Step 22: Trainer — smoke test (tiny config)")
    print("=" * 60)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainerConfig(
            d_model=64,
            n_heads=2,
            n_layers=1,
            d_ff=128,
            seq_len=64,
            batch_size=4,
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            peak_lr=1e-3,
            warmup_steps=10,
            total_steps=50,
            n_epochs=2,
            eval_every=25,
            eval_batches=5,
            log_every=10,
            checkpoint_dir=f"{tmpdir}/checkpoints",
            log_dir=f"{tmpdir}/logs",
            data_cache=f"{tmpdir}/cache",
        )

        trainer = Trainer(cfg)
        trainer.train()

        # Verify checkpoints were saved
        from pathlib import Path
        ckpts = list(Path(f"{tmpdir}/checkpoints").glob("*.pt"))
        print(f"\nCheckpoints saved: {len(ckpts)}")
        assert len(ckpts) >= 1, "No checkpoints were saved!"

        # Verify loss history was written
        history_path = Path(f"{tmpdir}/logs/loss_history.json")
        assert history_path.exists(), "Loss history not saved!"
        print(f"Loss history: {history_path.stat().st_size:,} bytes")

        print("\n[OK] trainer.py OK")
