"""
train_ouroboros.py — Phase 3 | Component 45O
Training script for the Ouroboros recursive architecture.

Standard LM training: one loss, one backward pass, done.
Ouroboros training: three losses, one backward pass, careful scheduling.

The training schedule has three phases:
  Phase A (steps 0–999)    : LM loss only. Stabilise the base behaviour.
  Phase B (steps 1000–2999): + pass consistency loss. Teach passes to cooperate.
  Phase C (steps 3000+)    : + verification reward.  Teach pass 3 to self-correct.

This phased introduction prevents the auxiliary losses from destabilising
early training when the base model is still finding its footing.

Usage:
  python train_ouroboros.py \
    --base_model_path anra_brain.pt \
    --output_path     anra_ouroboros.pt \
    --data_path       data/train.bin \
    --steps           5000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import argparse
import os
import time
from typing import Optional

from ouroboros   import OuroborosDecoder
from pass_gates  import SemanticAnchorLoss, LogicIntegrationLoss, VerificationRewardLoss, PassConsistencyLoss
from weight_sharing import parameter_budget_report


# ─────────────────────────────────────────────────────────────────────────────
# LOSS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

LAMBDA_LM           = 1.0   # Language modelling loss weight (never changes)
LAMBDA_CONSISTENCY  = 0.1   # Pass consistency loss weight
LAMBDA_VERIFICATION = 0.05  # Verification reward loss weight
LAMBDA_ANCHOR       = 0.05  # Semantic anchor loss weight (pass 1)
LAMBDA_LOGIC        = 0.05  # Logic integration loss weight (pass 2)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING STEP
# ─────────────────────────────────────────────────────────────────────────────

def training_step(
    model: OuroborosDecoder,
    batch: dict,
    step: int,
    losses: dict,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """
    Single training step. Returns a dict of scalar loss values for logging.

    Args:
        model     : OuroborosDecoder instance
        batch     : dict with keys 'tokens' (B, T) and optionally 'domain_labels' (B,)
        step      : current global step (controls which losses are active)
        losses    : dict of loss module instances
        optimizer : optimizer

    Returns:
        log_dict  : dict of {loss_name: float}
    """
    x       = batch["tokens"][:, :-1]   # input:  all but last token
    targets = batch["tokens"][:, 1:]    # target: all but first token

    # ── Forward pass ──────────────────────────────────────────────────────────
    logits, lm_loss = model(x, targets=targets)
    pass_hiddens    = model._last_pass_hiddens   # list of (B, T, d_model)

    # ── Always active: LM loss ────────────────────────────────────────────────
    total_loss = LAMBDA_LM * lm_loss
    log        = {"lm_loss": lm_loss.item()}

    # ── Phase B+ (step ≥ 1000): pass consistency loss ─────────────────────────
    if step >= 1000:
        consistency_loss = losses["consistency"](pass_hiddens)
        total_loss       = total_loss + LAMBDA_CONSISTENCY * consistency_loss
        log["consistency_loss"] = consistency_loss.item()

        # Semantic anchor on pass 1
        if "domain_labels" in batch and batch["domain_labels"] is not None:
            anchor_loss = losses["anchor"](pass_hiddens[0], batch["domain_labels"])
            total_loss  = total_loss + LAMBDA_ANCHOR * anchor_loss
            log["anchor_loss"] = anchor_loss.item()

        # Logic integration on pass 2
        if len(pass_hiddens) >= 2:
            logic_loss = losses["logic"](pass_hiddens[0], pass_hiddens[1])
            total_loss = total_loss + LAMBDA_LOGIC * logic_loss
            log["logic_loss"] = logic_loss.item()

    # ── Phase C (step ≥ 3000): verification reward ────────────────────────────
    if step >= 3000 and len(pass_hiddens) >= 3:
        verification_loss = losses["verification"](
            pass2_hidden=pass_hiddens[1],
            pass3_hidden=pass_hiddens[2],
            targets=targets,
        )
        total_loss = total_loss + LAMBDA_VERIFICATION * verification_loss
        log["verification_loss"] = verification_loss.item()

    log["total_loss"] = total_loss.item()

    # ── Backward ──────────────────────────────────────────────────────────────
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return log


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (minimal — replace with real DataLoader for production)
# ─────────────────────────────────────────────────────────────────────────────

def load_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict:
    """
    Sample a random batch from a flat token tensor.
    """
    max_start = data.size(0) - seq_len - 1
    starts    = torch.randint(0, max_start, (batch_size,))
    tokens    = torch.stack([data[s : s + seq_len + 1] for s in starts])
    return {"tokens": tokens.to(device), "domain_labels": None}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(
    base_model_path: str,
    output_path: str,
    data_path: Optional[str],
    n_steps: int,
    batch_size: int = 8,
    seq_len: int    = 128,
    lr: float       = 3e-4,
    n_passes: int   = 3,
    device_str: str = "cpu",
):
    device = torch.device(device_str)

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"Loading base model from {base_model_path}")
    base_model = torch.load(base_model_path, map_location=device)
    base_model.train()

    # ── Wrap in Ouroboros ─────────────────────────────────────────────────────
    model = OuroborosDecoder(base_model, n_passes=n_passes).to(device)

    print(parameter_budget_report(model))

    # ── Loss modules ──────────────────────────────────────────────────────────
    loss_modules = {
        "consistency"  : PassConsistencyLoss().to(device),
        "anchor"       : SemanticAnchorLoss(base_model.d_model).to(device),
        "logic"        : LogicIntegrationLoss().to(device),
        "verification" : VerificationRewardLoss(base_model.lm_head).to(device),
    }

    # ── Optimizer (only Ouroboros new params + base model fine-tune) ──────────
    param_groups = [
        {"params": model.model.parameters(),       "lr": lr * 0.1},   # low LR for base
        {"params": model.pass_gates.unsqueeze(0),  "lr": lr},          # full LR for new
        {"params": model.blend_weights.unsqueeze(0), "lr": lr},
        *[{"params": lm.parameters(), "lr": lr} for lm in loss_modules.values()],
    ]
    # Flatten properly
    optimizer = AdamW([
        {"params": model.model.parameters(),     "lr": lr * 0.1},
        {"params": [model.pass_gates],           "lr": lr},
        {"params": [model.blend_weights],        "lr": lr},
        *[{"params": list(lm.parameters()), "lr": lr} for lm in loss_modules.values()],
    ], weight_decay=1e-2)

    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.01)

    # ── Data ──────────────────────────────────────────────────────────────────
    if data_path and os.path.exists(data_path):
        data = torch.load(data_path).long().to(device)
    else:
        print("No data file found — using random tokens for smoke test.")
        data = torch.randint(0, base_model.vocab_size, (100_000,)).to(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining Ouroboros for {n_steps} steps...")
    print(f"Phase A (steps 0–999):    LM loss only")
    print(f"Phase B (steps 1000–2999): + consistency + anchor + logic")
    print(f"Phase C (steps 3000+):     + verification reward\n")

    best_loss = float("inf")
    t0        = time.time()

    for step in range(n_steps):
        batch  = load_batch(data, batch_size, seq_len, device)
        log    = training_step(model, batch, step, loss_modules, optimizer)
        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % 100 == 0 or step < 10:
            elapsed = time.time() - t0
            phase   = "A" if step < 1000 else ("B" if step < 3000 else "C")
            loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in log.items())
            print(f"Step {step:5d} [Phase {phase}] | {loss_str} | {elapsed:.1f}s elapsed")
            t0 = time.time()

        # ── Save checkpoint on improvement ────────────────────────────────────
        if log["total_loss"] < best_loss:
            best_loss = log["total_loss"]
            if step % 500 == 0 and step > 0:
                torch.save(model, output_path + ".ckpt")

    # ── Final save ────────────────────────────────────────────────────────────
    print(f"\nSaving final model to {output_path}")
    torch.save(model, output_path)
    print("Training complete.")
    print(f"Best total loss: {best_loss:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ouroboros recursive architecture")
    parser.add_argument("--base_model_path", default="anra_brain.pt")
    parser.add_argument("--output_path",     default="anra_ouroboros.pt")
    parser.add_argument("--data_path",       default="data/train.bin", type=str)
    parser.add_argument("--steps",           default=5000,   type=int)
    parser.add_argument("--batch_size",      default=8,      type=int)
    parser.add_argument("--seq_len",         default=128,    type=int)
    parser.add_argument("--lr",              default=3e-4,   type=float)
    parser.add_argument("--n_passes",        default=3,      type=int)
    parser.add_argument("--device",          default="cpu",  type=str)
    args = parser.parse_args()

    train(
        base_model_path=args.base_model_path,
        output_path=args.output_path,
        data_path=args.data_path,
        n_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        n_passes=args.n_passes,
        device_str=args.device,
    )
