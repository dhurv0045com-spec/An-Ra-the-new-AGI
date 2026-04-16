#!/usr/bin/env python3
"""
Build and train the first An-Ra brain checkpoint (anra_brain.pt).

This script:
1) Loads all identity text files from phase3/identity (45N)/
2) Parses H:/ANRA: dialog format into training text
3) Builds a character-level vocabulary
4) Trains the repository's custom transformer (CausalTransformer)
5) Saves a complete checkpoint dict to anra_brain.pt

Designed to run in Google Colab or local Python with PyTorch installed.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import re
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
IDENTITY_DIR = REPO_ROOT / "phase3/identity (45N)"
CUSTOM_MODEL_DIR = REPO_ROOT / "history/neural_network (45B)"


# ---------------------------------------------------------------------------
# Dynamic import of repo custom transformer modules
# ---------------------------------------------------------------------------

def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_custom_causal_transformer():
    """
    Load CausalTransformer from history/neural_network (45B)/transformer.py
    while preserving its relative imports (.attention, .embeddings).
    """
    if not CUSTOM_MODEL_DIR.exists():
        raise FileNotFoundError(f"Custom model directory not found: {CUSTOM_MODEL_DIR}")

    package_name = "anra_nn45b"
    pkg = types.ModuleType(package_name)
    pkg.__path__ = [str(CUSTOM_MODEL_DIR)]
    sys.modules[package_name] = pkg

    _load_module(f"{package_name}.attention", CUSTOM_MODEL_DIR / "attention.py")
    _load_module(f"{package_name}.embeddings", CUSTOM_MODEL_DIR / "embeddings.py")
    transformer_mod = _load_module(f"{package_name}.transformer", CUSTOM_MODEL_DIR / "transformer.py")

    return transformer_mod.CausalTransformer


# ---------------------------------------------------------------------------
# Identity data loading/parsing
# ---------------------------------------------------------------------------

SPEAKER_RE = re.compile(r"^\s*(H|ANRA)\s*:\s*(.*)$")


def load_identity_files(identity_dir: Path) -> List[Path]:
    txt_files = sorted(identity_dir.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {identity_dir}")
    return txt_files


def parse_identity_text(raw_text: str) -> List[Tuple[str, str]]:
    """
    Parse H:/ANRA: style text into (speaker, content) segments.
    Continuation lines are attached to the previous speaker segment.
    """
    segments: List[Tuple[str, str]] = []

    current_speaker = None
    current_lines: List[str] = []

    def flush_current():
        nonlocal current_speaker, current_lines
        if current_speaker is not None and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                segments.append((current_speaker, content))
        current_speaker = None
        current_lines = []

    for line in raw_text.splitlines():
        striped = line.strip()
        if not striped:
            if current_speaker is not None and current_lines:
                current_lines.append("")
            continue
        if striped.startswith("#"):
            continue

        match = SPEAKER_RE.match(line)
        if match:
            flush_current()
            current_speaker = match.group(1)
            current_lines = [match.group(2).strip()]
        else:
            if current_speaker is not None:
                current_lines.append(striped)

    flush_current()
    return segments


def normalize_raw_text(raw_text: str) -> str:
    """
    Fallback normalizer for files that are not fully H:/ANRA: structured.
    Keeps non-comment lines so every character contributes to vocabulary/training.
    """
    lines: List[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def build_training_corpus(identity_dir: Path) -> str:
    files = load_identity_files(identity_dir)

    all_segments: List[Tuple[str, str]] = []
    fallback_text_blocks: List[str] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_identity_text(text)
        if parsed:
            all_segments.extend(parsed)
        else:
            normalized = normalize_raw_text(text)
            if normalized:
                fallback_text_blocks.append(normalized)

    if not all_segments and not fallback_text_blocks:
        raise ValueError("No usable identity content parsed from identity files")

    lines = []
    for speaker, content in all_segments:
        lines.append(f"{speaker}: {content}")
    lines.extend(fallback_text_blocks)

    corpus = "\n".join(lines)
    if len(corpus) < 256:
        raise ValueError("Identity corpus is too small to train.")

    return corpus


# ---------------------------------------------------------------------------
# Char-level tokenization
# ---------------------------------------------------------------------------

def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab_chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(vocab_chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode_text(text: str, char_to_idx: Dict[str, int]) -> torch.Tensor:
    try:
        encoded = [char_to_idx[ch] for ch in text]
    except KeyError as exc:
        raise ValueError(f"Missing character in vocabulary: {exc}") from exc
    return torch.tensor(encoded, dtype=torch.long)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    seq_len: int = 128
    batch_size: int = 4
    epochs: int = 12
    steps_per_epoch: int = 120
    lr: float = 1e-3
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    n_kv_heads: int = 4
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    if data.numel() <= seq_len + 1:
        raise ValueError(
            f"Dataset too short ({data.numel()} tokens) for seq_len={seq_len}."
        )
    max_start = data.numel() - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s : s + seq_len] for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


def train_model(encoded_data: torch.Tensor, vocab_size: int, cfg: TrainConfig, device: torch.device):
    CausalTransformer = load_custom_causal_transformer()

    model = CausalTransformer(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        n_kv_heads=cfg.n_kv_heads,
        norm_type="rmsnorm",
        ff_activation="swiglu",
        tie_embeddings=True,
        pad_idx=0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    loss_history: List[float] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        for _ in range(cfg.steps_per_epoch):
            x, y = sample_batch(encoded_data, cfg.batch_size, cfg.seq_len, device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / cfg.steps_per_epoch
        loss_history.append(avg_loss)
        print(f"Epoch {epoch:02d}/{cfg.epochs} | loss = {avg_loss:.6f}")

    return model, loss_history


# ---------------------------------------------------------------------------
# Checkpoint save/validation
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    vocab_size: int,
    current_epoch: int,
    final_loss: float,
    loss_history: List[float],
    output_path: Path,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "vocab_size": vocab_size,
        "epoch": current_epoch,
        "loss": final_loss,
        "training_history": loss_history,
        "training": "initial_brain_build",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    torch.save(checkpoint, output_path)


def validate_checkpoint(output_path: Path, vocab_size: int, cfg: TrainConfig, device: torch.device) -> None:
    checkpoint = torch.load(output_path, map_location=device)

    required_keys = {
        "model",
        "char_to_idx",
        "idx_to_char",
        "vocab_size",
        "epoch",
        "loss",
        "training",
        "created_at",
    }
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise KeyError(f"Checkpoint missing required keys: {missing}")

    if checkpoint["vocab_size"] != vocab_size:
        raise ValueError(
            f"Checkpoint vocab_size mismatch: {checkpoint['vocab_size']} != {vocab_size}"
        )

    CausalTransformer = load_custom_causal_transformer()
    model = CausalTransformer(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.seq_len,
        dropout=cfg.dropout,
        n_kv_heads=cfg.n_kv_heads,
        norm_type="rmsnorm",
        ff_activation="swiglu",
        tie_embeddings=True,
        pad_idx=0,
    ).to(device)

    model.load_state_dict(checkpoint["model"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build and save anra_brain.pt")
    parser.add_argument("--output", type=str, default="anra_brain.pt")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps_per_epoch", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not (128 <= args.seq_len <= 256):
        raise ValueError("seq_len must be between 128 and 256.")
    if not (2 <= args.batch_size <= 8):
        raise ValueError("batch_size must be between 2 and 8.")

    cfg = TrainConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        lr=args.lr,
    )

    set_seed(cfg.seed)
    device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Loading identity data from: {IDENTITY_DIR}")
    identity_files = load_identity_files(IDENTITY_DIR)
    for path in identity_files:
        print(f" - {path.relative_to(REPO_ROOT)}")
    corpus = build_training_corpus(IDENTITY_DIR)
    print(f"Corpus size (chars): {len(corpus):,}")

    char_to_idx, idx_to_char = build_vocab(corpus)
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size (char-level): {vocab_size}")

    encoded_data = encode_text(corpus, char_to_idx)

    model, loss_history = train_model(encoded_data, vocab_size, cfg, device)

    if loss_history[-1] >= loss_history[0]:
        print(
            "Warning: final epoch loss did not drop below first epoch loss. "
            "Consider increasing epochs/steps_per_epoch."
        )

    output_path = Path(args.output).resolve()
    save_checkpoint(
        model=model,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        vocab_size=vocab_size,
        current_epoch=cfg.epochs,
        final_loss=float(loss_history[-1]),
        loss_history=[float(v) for v in loss_history],
        output_path=output_path,
    )
    print(f"Saved checkpoint: {output_path}")

    validate_checkpoint(output_path, vocab_size, cfg, device)
    print("Checkpoint validation passed: load_state_dict(checkpoint['model']) works.")


if __name__ == "__main__":
    main()
