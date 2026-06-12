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
from torch.optim.lr_scheduler import CosineAnnealingLR

from anra.core.config import AnRaConfig
from anra.core.registry import MODEL_REGISTRY
from anra_brain import CausalTransformerV2  # noqa: F401
from training.anra_optimizer import build_optimizer_with_report

ROOT = Path(__file__).resolve().parent.parent

REJECT_WORDS = [
    "ChatGPT",
    "GPT-4",
    "GPT4",
    "Claude",
    "Anthropic",
    "OpenAI",
    "As an AI",
    "I am an AI",
]


def filter_dfc_text(text: str) -> str:
    """Remove DFC rows containing teacher-identity leakage."""
    rejected = tuple(word.lower() for word in REJECT_WORDS)
    return "\n".join(
        line for line in text.splitlines() if not any(word in line.lower() for word in rejected)
    )


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    return torch.device(requested)


def load_text_data(
    path: Path,
    block_size: int,
    device: torch.device,
    *,
    filter_dfc: bool = False,
) -> torch.Tensor:
    """Load raw text, encode as token IDs when possible, otherwise as character IDs."""
    del block_size
    text = path.read_text(encoding="utf-8", errors="replace")
    if filter_dfc:
        text = filter_dfc_text(text)
    try:
        tokenizer_path = ROOT / "tokenizer" / "tokenizer_v3.json"
        if tokenizer_path.exists():
            from tokenizer.subword_tokenizer import SubwordTokenizer

            tokenizer = SubwordTokenizer.load(tokenizer_path)
            ids = tokenizer.encode(text)
            data = torch.tensor(ids, dtype=torch.long, device=device)
        else:
            raise FileNotFoundError
    except (ImportError, FileNotFoundError, ValueError, json.JSONDecodeError):
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
        mod_layers=cfg.model.mod_layers,
        use_hal=cfg.model.use_hal,
        use_rim=cfg.model.use_rim,
        use_dstp=cfg.model.use_dstp,
        base_seq_len=cfg.model.base_seq_len,
        target_seq_len=cfg.model.target_seq_len,
    ).to(device)
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    data_path = ROOT / "training_data" / "anra_training.txt"
    if not data_path.exists():
        print("WARNING: No training data found. Using synthetic data for smoke test.")
        synthetic = "The quick brown fox jumps over the lazy dog. " * 1000
        data_path = ROOT / "output" / "synthetic_train.txt"
        data_path.write_text(synthetic)

    data = load_text_data(
        data_path,
        cfg.model.block_size,
        device,
        filter_dfc=args.filter_dfc,
    )
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"Train tokens: {len(train_data):,}  Val tokens: {len(val_data):,}")

    optimizer, optimizer_report = build_optimizer_with_report(
        model,
        optimizer_name=args.optimizer,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    (metrics_dir / f"{cfg.experiment_name}_optimizer.json").write_text(
        json.dumps(optimizer_report, indent=2, default=str), encoding="utf-8"
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

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(cfg.training.gradient_accumulation):
            x, y = get_batch(train_data, cfg.model.block_size, cfg.training.batch_size)
            _, micro_loss = model(x, targets=y)
            (micro_loss / cfg.training.gradient_accumulation).backward()
            accumulated_loss += float(micro_loss.detach().item())
        loss_value = accumulated_loss / cfg.training.gradient_accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()
        if step > cfg.training.warmup_steps:
            scheduler.step()

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            print(f"step {step:6d}/{max_steps} | loss {loss_value:.4f} | lr {lr:.2e} | {elapsed:.0f}s")
            record = {
                "step": step,
                "train_loss": round(loss_value, 6),
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
        "--optimizer",
        default="auto",
        choices=["auto", "adamw", "adam8bit", "adafactor", "muon", "scale", "galore", "qgalore"],
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device",
    )
    parser.add_argument(
        "--filter_dfc",
        action="store_true",
        help="Remove rows containing teacher-identity leakage before tokenization",
    )
    args = parser.parse_args()

    cfg = AnRaConfig.from_yaml(Path(args.config))
    train(cfg, args)


if __name__ == "__main__":
    main()
