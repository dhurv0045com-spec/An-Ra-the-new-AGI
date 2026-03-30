"""
train.py — Master Training Script
===================================
Single entry point to run the full training pipeline.
Connects: dataset -> trainer -> loss tracking -> checkpointing -> scheduling -> AMP.

Usage:
    python train.py                          # train with defaults
    python train.py --resume latest          # resume from last checkpoint
    python train.py --resume best            # resume from best checkpoint
    python train.py --resume path/to/ck.pt  # resume from specific file
    python train.py --config custom.json     # load config from JSON
    python train.py --smoke-test             # fast smoke test (2 min)

All hyperparameters live in TrainerConfig in trainer.py.
Modify config_overrides below or pass a JSON config file.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure 45F directory is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

from trainer import Trainer, TrainerConfig
from mixed_precision import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default production config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = TrainerConfig(
    # Model — ~12M params: big enough to learn, small enough to train on CPU/1x GPU
    vocab_size=50257,
    d_model=256,
    n_heads=8,
    n_layers=4,
    d_ff=1024,
    seq_len=256,
    dropout=0.1,

    # Data — WikiText-103: 103M tokens, clean Wikipedia, standard LM benchmark
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    batch_size=16,
    num_workers=4,

    # Optimization — Chinchilla-style: ~1% warmup, cosine to 10% of peak LR
    peak_lr=3e-4,
    weight_decay=0.1,
    max_grad_norm=1.0,
    warmup_steps=2000,
    total_steps=100_000,

    # Training control
    n_epochs=20,
    eval_every=500,
    eval_batches=100,
    log_every=50,
    save_every_epoch=True,

    # Paths
    checkpoint_dir="./checkpoints",
    log_dir="./logs",
    data_cache="./data_cache",
)

# Smoke test config — completes in ~2 minutes on any hardware
SMOKE_CONFIG = TrainerConfig(
    vocab_size=50257,
    d_model=64,
    n_heads=2,
    n_layers=2,
    d_ff=256,
    seq_len=64,
    dropout=0.1,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    batch_size=4,
    num_workers=0,
    peak_lr=1e-3,
    weight_decay=0.1,
    max_grad_norm=1.0,
    warmup_steps=20,
    total_steps=100,
    n_epochs=3,
    eval_every=50,
    eval_batches=10,
    log_every=10,
    save_every_epoch=True,
    checkpoint_dir="./checkpoints_smoke",
    log_dir="./logs_smoke",
    data_cache="./data_cache",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from: 'latest', 'best', or path to a .pt checkpoint file"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON file overriding TrainerConfig fields"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a fast smoke test with tiny model and minimal data"
    )
    parser.add_argument(
        "--print-config", action="store_true",
        help="Print the active config and exit"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.smoke_test:
        config = SMOKE_CONFIG
        logger.info("Running smoke test config")
    else:
        config = DEFAULT_CONFIG

    # Apply JSON overrides
    if args.config:
        path = Path(args.config)
        if not path.exists():
            logger.error(f"Config file not found: {path}")
            sys.exit(1)
        with open(path) as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)
                logger.info(f"Config override: {k} = {v!r}")
            else:
                logger.warning(f"Unknown config key: {k}")

    # Apply resume
    if args.resume:
        config.resume_from = args.resume

    if args.print_config:
        print(json.dumps(asdict(config), indent=2))
        sys.exit(0)

    # Log environment
    device = get_device()
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"PyTorch: {__import__('torch').__version__}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(asdict(config), indent=2)}")

    # Run
    trainer = Trainer(config)
    trainer.train()

    logger.info("Done. Check ./logs/loss_curves.png for training curves.")


if __name__ == "__main__":
    main()
