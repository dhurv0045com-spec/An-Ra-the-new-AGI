#!/usr/bin/env python3
"""
scripts/train.py — Standalone AN-RA training script.

Works on any machine. No Colab required. No notebooks.

Usage:
    python scripts/train.py --config config/tiny.yaml --max_steps 1000
    python scripts/train.py --config config/base.yaml --max_steps 10000 --device cuda

Checkpoints saved to output/checkpoints/{experiment_name}/
Metrics logged to output/metrics/{experiment_name}.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from anra.core.config import AnRaConfig
from anra.core.registry import MODEL_REGISTRY
from anra_brain import CausalTransformerV2  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    return torch.device(requested)


def load_text_data(path: Path, block_size: int, device: torch.device) -> torch.Tensor:
    """Load raw text, encode as token IDs when possible, otherwise as character IDs."""
    del block_size
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tokenizer_path = ROOT / "tokenizer_v3.json"
        if tokenizer_path.exists():
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tokenizer_path))
            ids = tok.encode(text).ids
            data = torch.tensor(ids, dtype=torch.long, device=device)
        else:
            raise FileNotFoundError
    except (ImportError, FileNotFoundError):
        chars = sorted(set(text))
        c2i = {c: i for i, c in enumerate(chars)}
        ids = [c2i[c] for c in text]
        data = torch.tensor(ids, dtype=torch.long, device=device)
    return data


def get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,), device=data.device)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def train(cfg: AnRaConfig, args: argparse.Namespace) -> None:
    device = get_device(args.device)
    print(f"Training on: {device}")
    print(f"Experiment: {cfg.experiment_name}")

    ckpt_dir = ROOT / "output" / "checkpoints" / cfg.experiment_name
    metrics_dir = ROOT / "output" / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / f"{cfg.experiment_name}.jsonl"

    model = MODEL_REGISTRY.build(
        cfg.model.type,
        vocab_size=cfg.model.vocab_size,
        n_embd=cfg.model.n_embd,
        n_head=cfg.model.n_head,
        n_kv_head=cfg.model.n_kv_head,
        n_layer=cfg.model.n_layer,
        block_size=cfg.model.block_size,
        dropout=cfg.model.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    data_path = ROOT / "training_data" / "anra_training.txt"
    if not data_path.exists():
        print("WARNING: No training data found. Using synthetic data for smoke test.")
        synthetic = "The quick brown fox jumps over the lazy dog. " * 1000
        data_path = ROOT / "output" / "synthetic_train.txt"
        data_path.write_text(synthetic)

    data = load_text_data(data_path, cfg.model.block_size, device)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"Train tokens: {len(train_data):,}  Val tokens: {len(val_data):,}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )
    max_steps = args.max_steps or cfg.training.max_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)

    model.train()
    start_time = time.time()
    val_loss = None

    for step in range(1, max_steps + 1):
        if step <= cfg.training.warmup_steps:
            lr_scale = step / max(1, cfg.training.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.training.learning_rate * lr_scale

        x, y = get_batch(train_data, cfg.model.block_size, cfg.training.batch_size)
        _, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()
        if step > cfg.training.warmup_steps:
            scheduler.step()

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            print(f"step {step:6d}/{max_steps} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s")
            record = {
                "step": step,
                "train_loss": round(loss.item(), 6),
                "lr": round(lr, 8),
                "elapsed_s": round(elapsed, 1),
            }
            with metrics_file.open("a") as f:
                f.write(json.dumps(record) + "\n")

        if step % cfg.training.eval_every == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(
                    val_data,
                    cfg.model.block_size,
                    cfg.training.batch_size * 4,
                )
                _, val_loss = model(x_val, targets=y_val)
            print(f"  VAL loss: {val_loss.item():.4f}")
            model.train()

        if step % cfg.training.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg.model_dump(),
                    "val_loss": val_loss.item() if val_loss is not None else None,
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved: {ckpt_path}")

    print(f"Training complete. Metrics: {metrics_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AN-RA from any machine")
    parser.add_argument("--config", default="config/tiny.yaml", help="Path to AnRaConfig YAML file")
    parser.add_argument("--max_steps", type=int, default=None, help="Override config max_steps")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device",
    )
    args = parser.parse_args()

    cfg = AnRaConfig.from_yaml(Path(args.config))
    train(cfg, args)


if __name__ == "__main__":
    main()
