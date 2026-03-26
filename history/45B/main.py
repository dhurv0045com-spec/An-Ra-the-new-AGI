"""
main.py — Entry point and integration test.
Trains a nano language model end-to-end.
Validates every major component is wired correctly.
"""

import sys
import os
import math
import torch
import torch.nn as nn
from pathlib import Path

# ─── Project root on path ──────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import ModelConfig, CONFIGS
from model.transformer import CausalTransformer
from model.tokenizer import BPETokenizer
from model.regularization import LabelSmoothingLoss
from training.trainer import (
    TextDataset, build_dataloader,
    CosineWarmupScheduler, MetricsTracker, Checkpointer,
)
from inference.sampler import Generator, GenerationConfig
from utils.model_utils import count_parameters, WeightInspector, save_model, load_model


# ─────────────────────────────────────────────
# Sample corpus for demonstration
# ─────────────────────────────────────────────

CORPUS = """
the transformer is a model architecture that eschews recurrence and instead relies entirely on
an attention mechanism to draw global dependencies between input and output attention is all you need
language models are probabilistic models of text that can be used for generation translation
summarization and question answering the key insight of the transformer is that attention allows
each position to attend to all positions in the previous layer of the encoder this gives the model
a global receptive field from the first layer unlike recurrent networks which require many steps
to propagate information from one end of a sequence to the other self attention is the core mechanism
it relates different positions of a single sequence in order to compute a representation of the sequence
multi head attention allows the model to jointly attend to information from different representation subspaces
at different positions with a single attention head this would be inhibited we use the scaled dot product
attention which divides the dot products by the square root of the dimensionality of the keys
the feed forward network is applied to each position separately and identically it consists of two linear
transformations with a relu activation in between the model also makes use of positional encodings
to give the model information about the relative or absolute position of the tokens in the sequence
layer normalization is applied before each sub layer and residual connections are used throughout
""" * 30   # repeat to give the tiny tokenizer enough data


def train_demo(config_name: str = "nano", steps: int = 300):
    """
    End-to-end demo: tokenize → build model → train → generate.
    Uses the "nano" config so it runs fast on any hardware.
    """
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    cfg = CONFIGS[config_name]
    cfg.max_steps = steps
    cfg.eval_interval = 100
    cfg.save_interval = 100
    cfg.warmup_steps = 50
    cfg.batch_size = 8
    cfg.grad_accum_steps = 1

    print(f"\n{'='*60}")
    print(f" Language Model — End-to-End Demo")
    print(f" Config: {config_name}  |  Steps: {steps}")
    print(f"{'='*60}\n")

    # ── Step 10: Train tokenizer ──────────────────────────────
    print("[1/6] Training tokenizer...")
    texts = [s.strip() for s in CORPUS.split("\n") if s.strip()]
    tok = BPETokenizer(vocab_size=cfg.vocab_size)
    tok.train(texts, verbose=False)
    cfg.vocab_size = len(tok)
    print(f"  Vocabulary size: {len(tok)}")

    # ── Tokenize corpus ───────────────────────────────────────
    print("[2/6] Tokenizing corpus...")
    all_ids = []
    for text in texts:
        all_ids.extend(tok.encode(text, add_bos=False, add_eos=False))
    print(f"  Total tokens: {len(all_ids):,}")

    split = int(0.9 * len(all_ids))
    train_ids, val_ids = all_ids[:split], all_ids[split:]

    train_ds = TextDataset(train_ids, seq_len=cfg.max_seq_len, stride=cfg.max_seq_len // 2)
    val_ds   = TextDataset(val_ids,   seq_len=cfg.max_seq_len, stride=cfg.max_seq_len)

    train_loader = build_dataloader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = build_dataloader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Steps 16–20: Build model ──────────────────────────────
    print("[3/6] Building model...")
    model = CausalTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        n_kv_heads=cfg.n_heads // 2,   # GQA
        norm_type="rmsnorm",
        ff_activation="swiglu",
        tie_embeddings=cfg.tie_embeddings,
    )
    stats = count_parameters(model, verbose=False)
    print(f"  Parameters: {stats['total']:,}")

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = model.to(device)
    print(f"  Device: {device}")

    # ── Steps 22–25: Train ────────────────────────────────────
    print(f"[4/6] Training for {steps} steps...")
    loss_fn = LabelSmoothingLoss(
        vocab_size=cfg.vocab_size,
        smoothing=cfg.label_smoothing,
        ignore_index=cfg.pad_token_id,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=cfg.warmup_steps, max_steps=cfg.max_steps
    )
    metrics = MetricsTracker(window=50)
    checkpointer = Checkpointer("checkpoints", keep_last=2)

    model.train()
    data_iter = iter(train_loader)
    best_val_loss = float("inf")

    for step in range(1, steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids)
        loss = loss_fn(logits.view(-1, cfg.vocab_size), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        metrics.update(loss.item())

        if step % 50 == 0:
            # Eval
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i, vb in enumerate(val_loader):
                    if i >= 20:
                        break
                    vi = vb["input_ids"].to(device)
                    vl = vb["labels"].to(device)
                    vlogits, _ = model(vi)
                    vloss = loss_fn(vlogits.view(-1, cfg.vocab_size), vl.view(-1))
                    val_losses.append(vloss.item())
            val_loss = sum(val_losses) / len(val_losses)
            val_ppl = math.exp(min(val_loss, 100))

            print(
                f"  step {step:4d} | "
                f"train_loss {metrics.smoothed:.4f} | "
                f"val_loss {val_loss:.4f} | "
                f"ppl {val_ppl:.2f} | "
                f"lr {scheduler.current_lr:.2e}"
            )
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, "checkpoints/best_model.pt",
                           config=cfg.__dict__)

    # ── Step 30: Save final model ─────────────────────────────
    print("[5/6] Saving model...")
    save_model(model, "checkpoints/final_model.pt", config=cfg.__dict__)

    # ── Steps 26–29: Generate ─────────────────────────────────
    print("[6/6] Generating text...")
    model.eval()
    gen = Generator(model, tokenizer=tok, device=device)

    prompts = [
        "the transformer",
        "attention mechanism",
        "language models are",
    ]
    gencfg = GenerationConfig(
        max_new_tokens=40,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        do_sample=True,
    )

    print("\n" + "─" * 60)
    for prompt in prompts:
        ids = tok.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long)
        output_ids = gen.generate(input_ids, gencfg)
        full_text = tok.decode(output_ids[0].tolist())
        print(f"PROMPT: {prompt!r}")
        print(f"OUTPUT: {full_text!r}")
        print()

    # ── Step 32–33: Inspect ───────────────────────────────────
    print("─" * 60)
    print("Weight inspection:")
    inspector = WeightInspector(model)
    for name, param in list(model.named_parameters())[:5]:
        print(f"  {name:<50} mean={param.mean().item():.4f}  std={param.std().item():.4f}")

    print(f"\n{'='*60}")
    print(f" Demo complete. Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")

    return model, tok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="nano", choices=list(CONFIGS.keys()))
    p.add_argument("--steps", type=int, default=300)
    args = p.parse_args()
    train_demo(config_name=args.config, steps=args.steps)
