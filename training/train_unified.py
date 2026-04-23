from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from anra_paths import ROOT, ensure_dirs, get_checkpoint, get_tokenizer_file, get_dataset_file, inject_all_paths
from anra_brain import CausalTransformer
from optimizations import AdaptiveScheduler, GradientCheckpointedOuroboros, MultiScaleHardSampleDetector
from training.curriculum import get_phase

inject_all_paths()
ensure_dirs()


def _load_tokenizer():
    with open(get_tokenizer_file(), "rb") as f:
        return pickle.load(f)



def _load_model(tok, device):
    model = CausalTransformer(tok.vocab_size, 256, 4, 4, 128).to(device)
    ckpt = get_checkpoint()
    if ckpt:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    return model


def run_identity(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = _load_tokenizer()
    base_model = _load_model(tok, device)
    model = GradientCheckpointedOuroboros(base_model, passes=1).to(device)
    detector = MultiScaleHardSampleDetector()
    text = get_dataset_file().read_text(encoding="utf-8", errors="replace")
    ids = tok.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    opt = AdamW(base_model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = AdaptiveScheduler(args.lr, 100, max(args.steps, 1))

    metrics = []
    for step in range(args.steps):
        phase = get_phase(step // max(args.steps // max(args.epochs, 1), 1))
        model.passes = phase.ouroboros_passes
        idx = torch.randint(0, len(data) - args.seq_len - 1, (args.batch_size,))
        batch = torch.stack([data[i:i + args.seq_len + 1] for i in idx]).to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        sample_text = tok.decode(x[0].tolist())
        is_hard, difficulty = detector.detect(sample_text)
        if is_hard:
            model.passes = min(3, max(phase.ouroboros_passes, difficulty))
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
        lr = scheduler.get_lr(step, loss.item())
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"step={step+1} loss={loss.item():.4f} lr={lr:.2e} passes={model.passes}")
        metrics.append({"step": step + 1, "loss": float(loss.item()), "lr": lr, "phase": phase.name})

    out = ROOT / "anra_brain_identity.pt"
    torch.save({"model_state_dict": base_model.state_dict(), "metrics": metrics}, out)
    Path("finetune_report.json").write_text(json.dumps({"mode": "identity", "metrics": metrics[-20:], "timestamp": time.time()}, indent=2))
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="identity", choices=["identity", "ouroboros", "full", "eval"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()
    if args.mode in {"identity", "full", "ouroboros"}:
        run_identity(args)
    else:
        print("eval mode: use test_suite.py for full runtime verification")


if __name__ == "__main__":
    main()
