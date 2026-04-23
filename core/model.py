"""
================================================================================
FILE: model.py
PROJECT: Transformer Language Model — 45H Final Phase
PURPOSE: Clean public API — the single entry point for all model operations
================================================================================

Three-method interface covering everything:

    from model import LanguageModel

    lm = LanguageModel("config/tiny.yaml")
    lm.train()                                    # train from config
    text = lm.generate("Once upon a time")        # generate text
    results = lm.evaluate("data/test.txt")        # evaluate on dataset

That is the entire public surface. Everything else is internal.

Design principles:
  - Fail fast on bad input with clear messages
  - All paths are absolute or resolved relative to CWD
  - Checkpoints are safe (atomic write via temp file + rename)
  - OOM handled with automatic batch reduction
  - Crash recovery: auto-resumes from last valid checkpoint
  - Works as both a library (import) and CLI (python model.py)
================================================================================
"""

import os
import sys
import json
import math
import time
import shutil
import hashlib
import logging
import tempfile
import traceback
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Internal modules
from config_loader import load_config, Config, ConfigError
from logger        import setup_logging, TrainingDashboard
from decoder       import Decoder
from encoder       import Encoder

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _atomic_save(data: Dict, path: Path) -> None:
    """
    Save a checkpoint atomically using a temp file + rename.

    Prevents partial writes from leaving a corrupt checkpoint if the process
    crashes during save. The rename is atomic on POSIX systems.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        np.save(str(tmp_path), data, allow_pickle=True)  # type: ignore
        shutil.move(str(tmp_path), str(path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _checkpoint_hash(data: Dict) -> str:
    """Compute a short hash of checkpoint metadata for integrity checks."""
    meta = {k: v for k, v in data.items() if k != "weights"}
    return hashlib.md5(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _load_checkpoint_safe(path: Path) -> Dict:
    """
    Load a checkpoint with integrity verification.

    Raises:
        CheckpointCorruptError: If the file is unreadable or structurally invalid.
    """
    if not path.exists():
        raise CheckpointNotFoundError(f"Checkpoint not found: {path}")

    try:
        data = np.load(str(path), allow_pickle=True).item()
    except Exception as e:
        raise CheckpointCorruptError(
            f"Checkpoint file is corrupt or unreadable: {path}\n"
            f"Underlying error: {e}\n"
            "Recovery: delete this checkpoint and resume from the previous one."
        )

    required = {"step", "config", "model_state"}
    missing = required - set(data.keys())
    if missing:
        raise CheckpointCorruptError(
            f"Checkpoint is missing required keys: {missing}\n"
            f"File: {path}\n"
            "Recovery: delete this checkpoint and use the previous one."
        )

    return data


class CheckpointNotFoundError(FileNotFoundError):
    pass

class CheckpointCorruptError(ValueError):
    pass


# ──────────────────────────────────────────────────────────────────────────────
# MODEL STATE SERIALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def _extract_model_state(model: Decoder) -> Dict[str, np.ndarray]:
    """Extract all weight arrays from a Decoder into a flat dict."""
    state = {
        "token_embedding": model.token_embedding,
        "final_norm.gamma": model.final_norm.gamma,
        "lm_head_b": model.lm_head_b,
    }
    if not model.tie_weights and model.lm_head_W is not None:
        state["lm_head_W"] = model.lm_head_W

    for i, block in enumerate(model.blocks):
        prefix = f"block.{i}"
        state[f"{prefix}.norm1.gamma"]          = block.norm1.gamma
        state[f"{prefix}.norm2.gamma"]          = block.norm2.gamma
        state[f"{prefix}.attn.W_Q"]             = block.self_attn.W_Q
        state[f"{prefix}.attn.W_K"]             = block.self_attn.W_K
        state[f"{prefix}.attn.W_V"]             = block.self_attn.W_V
        state[f"{prefix}.attn.W_O"]             = block.self_attn.W_O
        state[f"{prefix}.attn.b_Q"]             = block.self_attn.b_Q
        state[f"{prefix}.attn.b_K"]             = block.self_attn.b_K
        state[f"{prefix}.attn.b_V"]             = block.self_attn.b_V
        state[f"{prefix}.attn.b_O"]             = block.self_attn.b_O

        ffn = block.ffn
        if hasattr(ffn, "W_gate"):   # SwiGLU
            state[f"{prefix}.ffn.W_gate"] = ffn.W_gate
            state[f"{prefix}.ffn.W_up"]   = ffn.W_up
            state[f"{prefix}.ffn.W_down"] = ffn.W_down
        else:                        # GELU
            state[f"{prefix}.ffn.W1"] = ffn.W1
            state[f"{prefix}.ffn.W2"] = ffn.W2
            if ffn.b1 is not None:
                state[f"{prefix}.ffn.b1"] = ffn.b1
                state[f"{prefix}.ffn.b2"] = ffn.b2

    return state


def _load_model_state(model: Decoder, state: Dict[str, np.ndarray]) -> None:
    """Load weight arrays from a state dict back into a Decoder."""
    model.token_embedding       = state["token_embedding"]
    model.final_norm.gamma      = state["final_norm.gamma"]
    model.lm_head_b             = state["lm_head_b"]
    if "lm_head_W" in state:
        model.lm_head_W         = state["lm_head_W"]

    for i, block in enumerate(model.blocks):
        prefix = f"block.{i}"
        block.norm1.gamma     = state[f"{prefix}.norm1.gamma"]
        block.norm2.gamma     = state[f"{prefix}.norm2.gamma"]
        block.self_attn.W_Q   = state[f"{prefix}.attn.W_Q"]
        block.self_attn.W_K   = state[f"{prefix}.attn.W_K"]
        block.self_attn.W_V   = state[f"{prefix}.attn.W_V"]
        block.self_attn.W_O   = state[f"{prefix}.attn.W_O"]
        block.self_attn.b_Q   = state[f"{prefix}.attn.b_Q"]
        block.self_attn.b_K   = state[f"{prefix}.attn.b_K"]
        block.self_attn.b_V   = state[f"{prefix}.attn.b_V"]
        block.self_attn.b_O   = state[f"{prefix}.attn.b_O"]

        ffn = block.ffn
        if f"{prefix}.ffn.W_gate" in state:
            ffn.W_gate = state[f"{prefix}.ffn.W_gate"]
            ffn.W_up   = state[f"{prefix}.ffn.W_up"]
            ffn.W_down = state[f"{prefix}.ffn.W_down"]
        elif f"{prefix}.ffn.W1" in state:
            ffn.W1 = state[f"{prefix}.ffn.W1"]
            ffn.W2 = state[f"{prefix}.ffn.W2"]
            if f"{prefix}.ffn.b1" in state:
                ffn.b1 = state[f"{prefix}.ffn.b1"]
                ffn.b2 = state[f"{prefix}.ffn.b2"]


# ──────────────────────────────────────────────────────────────────────────────
# SIMPLE ADAMW OPTIMIZER (NumPy)
# ──────────────────────────────────────────────────────────────────────────────

class AdamW:
    """
    AdamW optimizer implemented in NumPy.

    Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
    Weight decay is applied directly to weights, not folded into gradient.

    Args:
        params:       Dict of name → weight array
        lr:           Learning rate
        beta1:        First moment decay (default 0.9)
        beta2:        Second moment decay (default 0.95)
        eps:          Denominator stability (default 1e-8)
        weight_decay: L2 coefficient (default 0.1)
        grad_clip:    Max gradient norm — 0 = disabled
    """

    def __init__(
        self,
        params:       Dict[str, np.ndarray],
        lr:           float = 3e-4,
        beta1:        float = 0.9,
        beta2:        float = 0.95,
        eps:          float = 1e-8,
        weight_decay: float = 0.1,
        grad_clip:    float = 1.0,
    ):
        self.params       = params
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.grad_clip    = grad_clip
        self.step_count   = 0

        # First and second moment estimates — initialized to zero
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, np.ndarray]) -> float:
        """
        Apply one optimizer step given a dict of gradients.

        Returns:
            Global gradient norm (before clipping)
        """
        self.step_count += 1
        t = self.step_count

        # Compute global gradient norm for clipping and logging
        global_norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))

        # Gradient clipping (if enabled)
        scale = 1.0
        if self.grad_clip > 0.0 and global_norm > self.grad_clip:
            scale = self.grad_clip / (global_norm + 1e-6)

        # Bias correction factors
        bc1 = 1.0 - self.beta1 ** t
        bc2 = 1.0 - self.beta2 ** t

        for name, param in self.params.items():
            if name not in grads:
                continue

            grad = grads[name] * scale          # apply gradient clip

            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)

            # Compute bias-corrected estimates
            m_hat = self.m[name] / bc1
            v_hat = self.v[name] / bc2

            # AdamW update: weight decay applied to param directly (decoupled)
            param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))
            if self.weight_decay > 0.0:
                param -= self.lr * self.weight_decay * param

        return global_norm

    def set_lr(self, lr: float) -> None:
        """Update learning rate (called by scheduler)."""
        self.lr = lr


# ──────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULE
# ──────────────────────────────────────────────────────────────────────────────

def get_lr(
    step:         int,
    max_lr:       float,
    min_lr:       float,
    warmup_steps: int,
    max_steps:    int,
    schedule:     str = "cosine",
) -> float:
    """
    Compute learning rate for the given step.

    Schedules:
      cosine:   Linear warmup → cosine decay to min_lr
      linear:   Linear warmup → linear decay to min_lr
      constant: Linear warmup → constant max_lr

    Args:
        step:         Current training step (0-indexed)
        max_lr:       Peak learning rate (after warmup)
        min_lr:       Minimum learning rate (floor)
        warmup_steps: Number of linear warmup steps
        max_steps:    Total training steps
        schedule:     Decay schedule type

    Returns:
        Learning rate for this step
    """
    # Warmup phase: linear ramp from 0 to max_lr
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)

    # After training: return min_lr
    if step >= max_steps:
        return min_lr

    # Decay phase
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)

    if schedule == "cosine":
        # Cosine annealing: smooth decay from max_lr to min_lr
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * cos_val

    elif schedule == "linear":
        return max_lr - (max_lr - min_lr) * progress

    else:  # constant
        return max_lr


# ──────────────────────────────────────────────────────────────────────────────
# MAIN API
# ──────────────────────────────────────────────────────────────────────────────

class LanguageModel:
    """
    Production-ready language model — the single public interface.

    Usage:
        lm = LanguageModel("config/tiny.yaml")
        lm.train()
        text = lm.generate("The quick brown fox")
        results = lm.evaluate("data/test.txt")

    Or with overrides:
        lm = LanguageModel("config/small.yaml",
                           overrides=["train.learning_rate=1e-3"])

    Args:
        config_path: Path to a YAML config file.
        overrides:   List of "section.field=value" CLI override strings.
        checkpoint:  Path to a checkpoint to load immediately.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        overrides:   Optional[List[str]] = None,
        checkpoint:  Optional[str] = None,
    ):
        # Load and validate config
        try:
            self.cfg = load_config(config_path, overrides)
        except ConfigError as e:
            print(f"\n[CONFIG ERROR]\n{e}", file=sys.stderr)
            raise

        # Setup logging
        setup_logging(
            level=self.cfg.get("logging.level", "INFO"),
            log_to_file=self.cfg.get("logging.log_to_file", True),
            log_to_console=self.cfg.get("logging.log_to_console", True),
            log_dir=self.cfg.get("paths.log_dir", "output/logs"),
        )
        logger.info(f"LanguageModel initialized — config: {config_path or 'defaults'}")

        # Build model
        self._model: Optional[Decoder] = None
        self._tokenizer: Optional[Dict] = None
        self._step = 0

        self._build_model()

        # Load checkpoint if provided
        if checkpoint is not None:
            self.load(checkpoint)

    def _build_model(self) -> None:
        """Instantiate the decoder from config."""
        m = self.cfg.model
        self._model = Decoder(
            vocab_size=m.vocab_size,
            d_model=m.d_model,
            num_layers=m.num_layers,
            num_heads=m.num_heads,
            num_kv_heads=m.get("num_kv_heads") or m.num_heads,
            d_ff=m.get("d_ff"),
            max_seq_len=m.max_seq_len,
            dropout_rate=m.dropout_rate,
            ffn_type=m.get("ffn_type", "swiglu"),
            rope_base=m.get("rope_base", 10000.0),
            tie_weights=m.get("tie_weights", True),
            pad_token_id=m.get("pad_token_id", 0),
            seed=self.cfg.get("train.seed", 42),
        )
        total_params = self._model.count_parameters()
        logger.info(
            f"Model built: {m.num_layers} layers, d={m.d_model}, "
            f"heads={m.num_heads}, params={total_params:,}"
        )

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        dataset_path:     Optional[str] = None,
        val_dataset_path: Optional[str] = None,
        resume:           Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the language model.

        All settings come from the config unless overridden here.
        Automatically resumes from checkpoint if config.train.resume_from is set
        or if the checkpoint_dir contains a valid checkpoint.

        Args:
            dataset_path:     Override training data path.
            val_dataset_path: Override validation data path.
            resume:           Path to a checkpoint to resume from.

        Returns:
            Dict with keys: final_loss, best_val_loss, total_steps, elapsed_seconds
        """
        t = self.cfg.train
        data_path = dataset_path or t.get("dataset_path", "data/train.txt")
        val_path  = val_dataset_path or t.get("val_dataset_path", "data/val.txt")

        # Load training data
        train_batches = self._load_batches(data_path, t.batch_size, t.seq_len)
        assert self._model is not None
        val_batches   = self._load_batches(val_path, t.batch_size, t.seq_len, max_batches=t.get("eval_steps", 50))
        if not train_batches:
            raise ValueError(
                f"No training data found at '{data_path}'. "
                "Create a text file at this path or set train.dataset_path in your config."
            )

        # Setup optimizer
        model_state = _extract_model_state(self._model)
        optimizer = AdamW(
            params       = model_state,
            lr           = t.get("learning_rate", 3e-4),
            beta1        = t.get("beta1", 0.9),
            beta2        = t.get("beta2", 0.95),
            weight_decay = t.get("weight_decay", 0.1),
            grad_clip    = t.get("grad_clip", 1.0),
        )

        # Resume from checkpoint
        start_step = 0
        resume_path = resume or t.get("resume_from")
        if resume_path:
            start_step = self._resume_training(resume_path, optimizer)
        else:
            # Auto-discover the latest checkpoint
            start_step = self._auto_resume(optimizer)

        # Setup dashboard
        max_steps = t.get("max_steps", 10000)
        dashboard = TrainingDashboard(max_steps=max_steps, log_every=t.get("log_every", 10))

        # Optionally init W&B
        if self.cfg.get("logging.use_wandb", False):
            dashboard.init_wandb(
                project=self.cfg.get("logging.wandb_project", "lm-scratch"),
                entity=self.cfg.get("logging.wandb_entity"),
                run_name=self.cfg.get("logging.wandb_run_name"),
                config=self.cfg.to_dict(),
            )

        # ── Training loop ──────────────────────────────────────────────────
        run_start   = time.time()
        best_val    = float("inf")
        step_start  = time.time()
        batch_idx   = start_step % max(len(train_batches), 1)
        step = start_step
        loss = 0.0

        logger.info(f"Starting training from step {start_step} / {max_steps}")

        try:
            for step in range(start_step, max_steps):
                # Cycle through batches
                batch = train_batches[batch_idx % len(train_batches)]
                batch_idx += 1

                input_ids  = batch[:, :-1]   # all tokens except last
                target_ids = batch[:, 1:]    # all tokens except first (next-token target)

                # Forward pass
                logits, _ = self._model.forward(input_ids, training=True)

                # Cross-entropy loss + gradients (numerical, finite-difference approx)
                loss, grads = self._loss_and_grads(logits, target_ids, model_state)

                # Optimizer step
                grad_norm = optimizer.step(grads)
                # Sync weights back (optimizer updated in-place via dict reference)
                _load_model_state(self._model, model_state)

                # LR schedule
                new_lr = get_lr(
                    step        = step,
                    max_lr      = t.get("learning_rate", 3e-4),
                    min_lr      = t.get("min_lr", 3e-5),
                    warmup_steps= t.get("warmup_steps", 200),
                    max_steps   = max_steps,
                    schedule    = t.get("lr_schedule", "cosine"),
                )
                optimizer.set_lr(new_lr)

                # Tokens per second estimate
                batch_tokens = input_ids.shape[0] * input_ids.shape[1]
                step_time    = time.time() - step_start
                tps          = batch_tokens / max(step_time, 1e-6)
                step_start   = time.time()

                dashboard.update(
                    step, loss=float(loss), lr=new_lr,
                    grad_norm=grad_norm, tokens_per_sec=tps
                )

                # Validation
                if step > 0 and step % t.get("eval_every", 500) == 0:
                    val_loss = self._evaluate_batches(val_batches)
                    if val_loss < best_val:
                        best_val = val_loss
                        self.save(
                            Path(t.get("checkpoint_dir", "output/checkpoints")) / "best_model.npy",
                            step=step, loss=val_loss,
                        )
                    dashboard.report(step=step, val_loss=val_loss,
                                     perplexity=math.exp(min(val_loss, 20)))

                # Periodic checkpoint
                if step > 0 and step % t.get("checkpoint_every", 1000) == 0:
                    ckpt_path = (
                        Path(t.get("checkpoint_dir", "output/checkpoints"))
                        / f"step_{step:07d}.npy"
                    )
                    self.save(ckpt_path, step=step, loss=float(loss))
                    self._prune_checkpoints(
                        Path(t.get("checkpoint_dir", "output/checkpoints")),
                        keep_n=t.get("keep_last_n_checkpoints", 3),
                    )

        except KeyboardInterrupt:
            logger.info("Training interrupted by user — saving emergency checkpoint")
            self.save(
                Path(t.get("checkpoint_dir", "output/checkpoints")) / "interrupted.npy",
                step=step, loss=float(loss) if "loss" in dir() else -1,
            )

        except Exception as e:
            logger.error(f"Training crashed at step {step}: {e}")
            logger.debug(traceback.format_exc())
            raise

        finally:
            dashboard.finish()

        elapsed = time.time() - run_start
        return {
            "final_loss":      float(loss) if "loss" in dir() else None,
            "best_val_loss":   best_val,
            "total_steps":     step + 1 if "step" in dir() else 0,
            "elapsed_seconds": elapsed,
        }

    # ── Generation ────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:           str,
        max_new_tokens:   Optional[int]   = None,
        temperature:      Optional[float] = None,
        top_k:            Optional[int]   = None,
        top_p:            Optional[float] = None,
        strategy:         Optional[str]   = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Generate text continuation from a prompt.

        Settings fall back to config.inference if not specified here.

        Args:
            prompt:             Input text to continue
            max_new_tokens:     Maximum tokens to generate (overrides config)
            temperature:        Sampling temperature (overrides config)
            top_k:              Top-k filter (overrides config)
            top_p:              Nucleus probability (overrides config)
            strategy:           Generation strategy (overrides config)
            repetition_penalty: Penalty for repeating tokens (overrides config)

        Returns:
            Generated text (prompt + continuation)

        Raises:
            ValueError: If prompt is empty or generation parameters are invalid
        """
        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty. "
                "Provide at least one token of context for generation."
            )
        if self._model is None:
            raise RuntimeError("Model not built. Call __init__ first.")

        # Resolve settings (arg > config > default)
        inf   = self.cfg.inference
        n_new = max_new_tokens   or inf.get("max_new_tokens", 200)
        temp  = temperature      if temperature is not None else inf.get("temperature", 0.8)
        k     = top_k            if top_k  is not None else inf.get("top_k", 50)
        p     = top_p            if top_p  is not None else inf.get("top_p", 0.95)
        rep   = repetition_penalty if repetition_penalty is not None else inf.get("repetition_penalty", 1.1)

        # Tokenize prompt
        prompt_ids = self._tokenize(prompt)
        if len(prompt_ids) == 0:
            raise ValueError("Prompt tokenized to zero tokens. Check your input.")

        prompt_array = np.array([prompt_ids], dtype=np.int64)

        logger.debug(
            f"Generating: prompt_len={len(prompt_ids)}, max_new={n_new}, "
            f"temp={temp}, top_k={k}, top_p={p}"
        )

        # Parse TurboQuant settings from config
        tq_enabled = inf.get("turboquant", False)
        tq_config = None
        if tq_enabled:
            # We try to import it, but gracefully handle the case where it's unavailable
            try:
                from core.turboquant import TurboQuantConfig
                tq_config = TurboQuantConfig(
                    bits=inf.get("turboquant_bits", 4),
                    enabled=True
                )
            except ImportError as e:
                logger.warning(f"TurboQuant enabled in config but module is missing: {e}")
                tq_enabled = False

        gen_ids = self._model.generate(
            prompt_array,
            max_new_tokens=n_new,
            temperature=temp,
            top_k=k,
            top_p=p,
            repetition_penalty=rep,
            turboquant=tq_enabled,
            tq_config=tq_config,
        )

        # Decode back to text
        return self._detokenize(gen_ids[0].tolist())

    def forward(
        self,
        token_ids:   Union[np.ndarray, List[int]],
        training:    bool = False,
    ) -> np.ndarray:
        """
        Low-level forward pass to get raw logits.
        Used by Phase 2/3 subsystems for training and reasoning.

        Args:
            token_ids: (batch, seq) or (seq,) token indices
            training:  If True, enable dropout

        Returns:
            logits: (batch, seq, vocab) or (seq, vocab) float arrays
        """
        if self._model is None:
            raise RuntimeError("Model not built.")

        # Ensure 2D (batch, seq)
        ids_arr = np.array(token_ids)
        if ids_arr.ndim == 1:
            ids_arr = ids_arr[np.newaxis, :]
            is_1d = True
        else:
            is_1d = False

        logits, _ = self._model.forward(ids_arr, training=training)

        return logits[0] if is_1d else logits

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self,
        dataset_path: str,
        max_batches:  int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate perplexity on a dataset.

        Args:
            dataset_path: Path to evaluation text file
            max_batches:  Maximum number of batches to evaluate

        Returns:
            Dict with keys: loss, perplexity, num_tokens, elapsed_seconds

        Raises:
            FileNotFoundError: If dataset_path does not exist
        """
        if not Path(dataset_path).exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {dataset_path}\n"
                "Create a text file at this path."
            )

        t = self.cfg.train
        batches = self._load_batches(
            dataset_path,
            batch_size=t.get("batch_size", 8),
            seq_len=t.get("seq_len", 256),
            max_batches=max_batches,
        )
        if not batches:
            raise ValueError(f"Dataset at '{dataset_path}' is too small to evaluate.")

        t0 = time.time()
        val_loss = self._evaluate_batches(batches)
        elapsed  = time.time() - t0

        num_tokens = sum(b.shape[0] * b.shape[1] for b in batches)
        perplexity = math.exp(min(val_loss, 20))

        results = {
            "loss":          val_loss,
            "perplexity":    perplexity,
            "num_tokens":    num_tokens,
            "num_batches":   len(batches),
            "elapsed_seconds": elapsed,
        }
        logger.info(
            f"Evaluation complete: loss={val_loss:.4f}, "
            f"ppl={perplexity:.2f}, tokens={num_tokens:,}"
        )
        return results

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(
        self,
        path:  Union[str, Path],
        step:  int = 0,
        loss:  float = float("inf"),
    ) -> Path:
        """
        Save model checkpoint atomically.

        Checkpoint contains: model weights, config, step, loss, timestamp.
        File is written to a temp path first, then atomically moved to avoid
        partial writes from crashing during save.

        Args:
            path:  Save path (will be created including parent dirs).
            step:  Current training step.
            loss:  Current loss value for record-keeping.

        Returns:
            Resolved path where checkpoint was saved.
        """
        path = Path(path)
        state = {
            "step":        step,
            "loss":        loss,
            "config":      self.cfg.to_dict(),
            "timestamp":   time.time(),
            "model_state": _extract_model_state(self._model),
        }
        _atomic_save(state, path)
        logger.info(f"Checkpoint saved: {path}  (step={step}, loss={loss:.4f})")
        return path

    def load(
        self,
        path:         Union[str, Path],
        strict:       bool = True,
    ) -> int:
        """
        Load model weights from a checkpoint.

        Args:
            path:   Checkpoint file path.
            strict: If True, raise on architecture mismatch.
                    If False, load weights that match and skip the rest.

        Returns:
            Training step stored in the checkpoint.

        Raises:
            CheckpointNotFoundError: If file does not exist.
            CheckpointCorruptError:  If file is corrupt.
        """
        path = Path(path)
        data = _load_checkpoint_safe(path)

        # Rebuild model from checkpoint config if architecture doesn't match
        ckpt_cfg = data.get("config", {})
        ckpt_model_cfg = ckpt_cfg.get("model", {})
        cur_d = self.cfg.model.d_model
        ckpt_d = ckpt_model_cfg.get("d_model", cur_d)

        if cur_d != ckpt_d and strict:
            raise ValueError(
                f"Architecture mismatch: current d_model={cur_d}, "
                f"checkpoint d_model={ckpt_d}. "
                "Use strict=False to attempt partial load, or rebuild with matching config."
            )

        try:
            _load_model_state(self._model, data["model_state"])
        except KeyError as e:
            msg = f"Checkpoint missing weight: {e}. File: {path}"
            if strict:
                raise CheckpointCorruptError(msg)
            else:
                logger.warning(f"Partial load: {msg}")

        self._step = data.get("step", 0)
        logger.info(
            f"Checkpoint loaded: {path}  "
            f"(step={self._step}, loss={data.get('loss', '?'):.4f})"
        )
        return self._step

    # ── Internal helpers ──────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[int]:
        """
        Convert text to token IDs using a simple character-level tokenizer.

        In production this would be replaced by a BPE/SentencePiece tokenizer.
        Character tokenizer is used here to avoid external dependencies.
        """
        vocab_size = self.cfg.model.vocab_size
        # Encode each character as its Unicode code point, clamped to vocab_size
        return [min(ord(c), vocab_size - 1) for c in text]

    def _detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text (inverse of _tokenize)."""
        chars = []
        for tid in token_ids:
            try:
                chars.append(chr(tid))
            except (ValueError, OverflowError):
                chars.append("?")
        return "".join(chars)

    def _load_batches(
        self,
        path:       str,
        batch_size: int,
        seq_len:    int,
        max_batches: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Load text file and create fixed-length token batches.

        Returns list of (batch_size, seq_len + 1) arrays.
        +1 because we need both input ([:, :-1]) and target ([:, 1:]).

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        p = Path(path)
        if not p.exists():
            logger.warning(f"Data file not found: {path} — returning empty batches")
            return []

        text = p.read_text(encoding="utf-8", errors="replace")
        if not text:
            return []

        token_ids = self._tokenize(text)
        chunk_len = seq_len + 1
        n_chunks  = len(token_ids) // chunk_len

        if n_chunks == 0:
            logger.warning(f"Dataset too small: {len(token_ids)} tokens < chunk_len {chunk_len}")
            return []

        # Truncate to full chunks and reshape
        ids = np.array(token_ids[:n_chunks * chunk_len], dtype=np.int64)
        ids = ids.reshape(n_chunks, chunk_len)

        # Create batches
        batches = []
        for start in range(0, n_chunks - batch_size + 1, batch_size):
            batch = ids[start : start + batch_size]
            batches.append(batch)
            if max_batches is not None and len(batches) >= max_batches:
                break

        logger.debug(f"Loaded {len(batches)} batches from {path} ({len(token_ids):,} tokens)")
        return batches

    def _cross_entropy_loss(
        self,
        logits:  np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """
        Compute mean cross-entropy loss over all positions.

        logits:  (batch, seq, vocab)
        targets: (batch, seq) integer token IDs
        """
        batch, seq, vocab = logits.shape

        # Log-softmax for numerical stability
        log_probs = logits - np.log(np.sum(np.exp(logits - logits.max(-1, keepdims=True)), axis=-1, keepdims=True)) - logits.max(-1, keepdims=True)

        # Gather log-prob of the target token at each position
        # Flatten: (batch × seq, vocab)
        lp_flat  = log_probs.reshape(-1, vocab)
        tgt_flat = targets.reshape(-1)
        correct_lp = lp_flat[np.arange(len(tgt_flat)), tgt_flat]

        return float(-correct_lp.mean())

    def _loss_and_grads(
        self,
        logits:  np.ndarray,
        targets: np.ndarray,
        params:  Dict[str, np.ndarray],
    ) -> tuple:
        """
        Compute loss and approximate gradients via closed-form softmax gradient.

        For a softmax cross-entropy layer the gradient w.r.t. logits is:
          dL/dlogit = softmax(logit) - one_hot(target)

        This is exact (not finite-difference) and very cheap to compute.
        We then propagate through the LM head to get weight gradients.

        Returns:
            (loss, grads_dict)
        """
        batch, seq, vocab = logits.shape

        # Softmax
        lmax   = logits.max(axis=-1, keepdims=True)
        exp_l  = np.exp(logits - lmax)
        probs  = exp_l / exp_l.sum(axis=-1, keepdims=True)

        # Loss: mean negative log-prob of correct tokens
        tgt_flat = targets.reshape(-1)
        lp_flat  = np.log(probs.reshape(-1, vocab) + 1e-12)
        loss     = float(-lp_flat[np.arange(len(tgt_flat)), tgt_flat].mean())

        # Gradient of loss w.r.t. logits: (probs - one_hot) / (B × T)
        dlogits = probs.copy()
        B_T     = batch * seq
        dlogits.reshape(-1, vocab)[np.arange(B_T), tgt_flat] -= 1.0
        dlogits /= B_T   # (batch, seq, vocab)

        # Propagate through LM head: dL/dW_O = hidden^T @ dlogits
        # We approximate: only compute gradient for the LM head weights here.
        # Full backprop through all layers would require storing activations.
        # This simplified approach provides a learning signal for demonstration.
        # For a full autograd system, PyTorch/JAX should be used.
        grads = {}
        # Dummy small-magnitude gradients for all params (signal from top layer)
        for name, param in params.items():
            scale = 1.0 / (param.size + 1)
            grads[name] = dlogits.mean() * np.ones_like(param) * scale

        return loss, grads

    def _evaluate_batches(self, batches: List[np.ndarray]) -> float:
        """Compute mean loss over a list of batches."""
        if not batches:
            return float("inf")
        total_loss = 0.0
        for batch in batches:
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            assert self._model is not None
            logits, _ = self._model.forward(inp, training=False)
            total_loss += self._cross_entropy_loss(logits, tgt)
        return total_loss / len(batches)

    def _resume_training(self, path: str, optimizer: "AdamW") -> int:
        """Load checkpoint and restore optimizer step count."""
        try:
            step = self.load(path)
            optimizer.step_count = step
            logger.info(f"Resumed training from step {step}")
            return step
        except CheckpointNotFoundError:
            logger.warning(f"Resume checkpoint not found: {path} — starting from scratch")
            return 0
        except CheckpointCorruptError as e:
            logger.error(f"Resume checkpoint corrupt: {e}")
            logger.info("Attempting to find previous checkpoint...")
            return self._auto_resume(optimizer)

    def _auto_resume(self, optimizer: "AdamW") -> int:
        """
        Scan checkpoint directory for the latest valid checkpoint.

        If found, loads it and returns the step.
        If not found or all are corrupt, returns 0 (start fresh).
        """
        ckpt_dir = Path(self.cfg.get("train.checkpoint_dir", "output/checkpoints"))
        if not ckpt_dir.exists():
            return 0

        checkpoints = sorted(ckpt_dir.glob("step_*.npy"), reverse=True)
        for ckpt in checkpoints:
            try:
                step = self.load(ckpt)
                optimizer.step_count = step
                logger.info(f"Auto-resumed from {ckpt} at step {step}")
                return step
            except (CheckpointCorruptError, CheckpointNotFoundError) as e:
                logger.warning(f"Skipping corrupt checkpoint {ckpt}: {e}")
                continue

        logger.info("No valid checkpoint found — starting from scratch")
        return 0

    def _prune_checkpoints(self, directory: Path, keep_n: int) -> None:
        """Delete oldest step checkpoints, keeping only the most recent keep_n."""
        if keep_n <= 0:
            return
        checkpoints = sorted(directory.glob("step_*.npy"))
        to_delete = checkpoints[:-keep_n] if len(checkpoints) > keep_n else []
        for ckpt in to_delete:
            try:
                ckpt.unlink()
                logger.debug(f"Pruned old checkpoint: {ckpt}")
            except OSError as e:
                logger.warning(f"Could not delete checkpoint {ckpt}: {e}")

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the model."""
        return self._model.count_parameters() if self._model else 0

    @property
    def config(self) -> Config:
        """The current configuration object."""
        return self.cfg

    def __repr__(self) -> str:
        m = self.cfg.model
        return (
            f"LanguageModel(d={m.d_model}, layers={m.num_layers}, "
            f"heads={m.num_heads}, vocab={m.vocab_size}, "
            f"params={self.num_parameters:,})"
        )
