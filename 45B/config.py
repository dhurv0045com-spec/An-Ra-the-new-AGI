"""
config.py — Master configuration for the language model.
All hyperparameters live here. Override per experiment.
"""

from dataclasses import dataclass, field
from typing import Optional
import json, os


@dataclass
class ModelConfig:
    # Vocabulary / tokenizer
    vocab_size: int = 32000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Architecture
    d_model: int = 512          # embedding / hidden dimension
    n_heads: int = 8            # attention heads
    n_layers: int = 6           # transformer blocks
    d_ff: int = 2048            # feed-forward inner dimension
    max_seq_len: int = 1024     # maximum context window
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    tie_embeddings: bool = True  # tie input/output embeddings

    # Training
    batch_size: int = 32
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100_000
    eval_interval: int = 500
    eval_steps: int = 100
    save_interval: int = 2000

    # Regularization
    label_smoothing: float = 0.1

    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Inference defaults
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self):
        lines = ["ModelConfig("]
        for k, v in self.__dict__.items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)


# Preset configs for quick experiments
CONFIGS = {
    "nano": ModelConfig(
        d_model=128, n_heads=4, n_layers=3, d_ff=512,
        max_seq_len=256, vocab_size=8000,
        batch_size=64, max_steps=5_000,
    ),
    "small": ModelConfig(
        d_model=256, n_heads=4, n_layers=4, d_ff=1024,
        max_seq_len=512, vocab_size=16000,
        batch_size=32, max_steps=20_000,
    ),
    "base": ModelConfig(),   # defaults above
    "medium": ModelConfig(
        d_model=768, n_heads=12, n_layers=12, d_ff=3072,
        max_seq_len=1024, vocab_size=32000,
        batch_size=16, max_steps=200_000,
    ),
    "large": ModelConfig(
        d_model=1024, n_heads=16, n_layers=24, d_ff=4096,
        max_seq_len=2048, vocab_size=32000,
        batch_size=8, grad_accum_steps=16,
        max_steps=500_000,
    ),
}
