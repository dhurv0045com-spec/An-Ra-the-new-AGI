from __future__ import annotations

import json
import pickle
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from anra_brain import CausalTransformer, CharTokenizer
from generate import generate

CONFIG = {
    "base_checkpoint": "anra_brain.pt",
    "identity_checkpoint": "anra_brain_identity.pt",
    "tokenizer_path": "tokenizer.pkl",
    "data_path": "data/combined_identity_data.txt",
    "fallback_data_path": "combined_identity_data.txt",
    "drive_dir": "/content/drive/MyDrive/AnRa/",
    "epochs": 12,
    "batch_size": 32,
    "seq_len": 256,
    "val_split": 0.1,
    "patience": 3,
    "gradient_checkpointing": True,
    "mixed_precision": True,
    "base_lr": 3e-5,
    "lm_head_lr": 6e-5,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "grad_accum_steps": 1,
    "block_size": 128,
    "n_embd": 256,
    "n_head": 4,
    "n_layer": 4,
}

PROBE_PROMPTS = ["I am", "My purpose is", "An-Ra is", "Who created you"]


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    milestone: str


# ======================================================================================
# Utilities
# ======================================================================================

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_tokenizer(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _parse_pairs(raw_text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    block_pattern = re.compile(r"H:\s*(.*?)\nANRA:\s*(.*?)(?=\nH:|\Z)", re.S)
    for m in block_pattern.finditer(raw_text):
        h = m.group(1).strip()
        a = m.group(2).strip()
        if h and a:
            pairs.append((h, a))

    if pairs:
        return pairs

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    cur_h: str | None = None
    for ln in lines:
        if ln.startswith("H:"):
            cur_h = ln[2:].strip()
        elif ln.startswith("ANRA:") and cur_h:
            out = ln[5:].strip()
            if out:
                pairs.append((cur_h, out))
                cur_h = None

    if pairs:
        return pairs

    chunks = [c.strip() for c in re.split(r"\n\s*\n", raw_text) if c.strip()]
    return [("", c) for c in chunks]


class PairDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, str]], tok, seq_len: int):
        self.samples: List[List[int]] = []
        self.seq_len = seq_len
        for h, a in pairs:
            text = f"H: {h}\nANRA: {a}" if h else f"ANRA: {a}"
            ids = tok.encode(text)
            if len(ids) < 8:
                continue
            stride = max(1, seq_len // 2)
            for i in range(0, max(1, len(ids) - 1), stride):
                segment = ids[i : i + seq_len + 1]
                if len(segment) >= 8:
                    self.samples.append(segment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = self.samples[idx]
        if len(ids) < self.seq_len + 1:
            ids = ids + [0] * (self.seq_len + 1 - len(ids))
        x = torch.tensor(ids[: self.seq_len], dtype=torch.long)
        y = torch.tensor(ids[1 : self.seq_len + 1], dtype=torch.long)
        return x, y


# ======================================================================================
# Model and optimization
# ======================================================================================

def _load_model(vocab_size: int, device: torch.device) -> CausalTransformer:
    model = CausalTransformer(vocab_size, CONFIG["n_embd"], CONFIG["n_head"], CONFIG["n_layer"], CONFIG["block_size"])
    identity = Path(CONFIG["identity_checkpoint"])
    base = Path(CONFIG["base_checkpoint"])
    ckpt = identity if identity.exists() else base
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print("No checkpoint found, starting from initialized weights")
    return model.to(device)


def _freeze_policy(model: CausalTransformer) -> Tuple[int, int]:
    for p in model.token_embedding_table.parameters():
        p.requires_grad = False

    total_blocks = len(model.blocks)
    freeze_n = max(1, total_blocks // 4)
    for i, block in enumerate(model.blocks):
        trainable = i >= freeze_n
        for p in block.parameters():
            p.requires_grad = trainable

    for p in model.ln_f.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return frozen_count, trainable_count


def _optimizer(model: CausalTransformer) -> AdamW:
    base_params = []
    head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("lm_head"):
            head_params.append(p)
        else:
            base_params.append(p)
    return AdamW(
        [
            {"params": base_params, "lr": CONFIG["base_lr"]},
            {"params": head_params, "lr": CONFIG["lm_head_lr"]},
        ],
        weight_decay=CONFIG["weight_decay"],
    )


def _milestone(val_loss: float) -> str:
    if val_loss > 2.0:
        return "Adapting..."
    if 1.5 <= val_loss <= 2.0:
        return "Identity forming"
    if 1.0 <= val_loss < 1.5:
        return "✓ Consistent voice present"
    return "★ Deep conditioning achieved"


def _evaluate(model: CausalTransformer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b * t, c), y.view(b * t))
            losses.append(loss.item())
    return float(sum(losses) / max(len(losses), 1))


def _identity_probe(tag: str) -> Dict[str, str]:
    out = {}
    print(f"\n[{tag}] identity probe")
    for p in PROBE_PROMPTS:
        text = generate(f"H: {p}\nANRA:", strategy="nucleus", max_new_tokens=80)
        out[p] = text
        print(f"- {p} => {text[:120]}")
    return out


def _save_training_report(path: Path, metrics: List[EpochMetrics], before: Dict[str, str], after: Dict[str, str]) -> None:
    payload = {
        "config": CONFIG,
        "epochs": [m.__dict__ for m in metrics],
        "probe_before": before,
        "probe_after": after,
        "timestamp": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ======================================================================================
# Main
# ======================================================================================

def main() -> None:
    repo = Path(__file__).resolve().parent
    data_path = repo / CONFIG["data_path"]
    if not data_path.exists():
        data_path = repo / CONFIG["fallback_data_path"]
    if not data_path.exists():
        raise FileNotFoundError("Identity data file not found")

    tokenizer = _load_tokenizer(repo / CONFIG["tokenizer_path"])
    raw = data_path.read_text(encoding="utf-8", errors="replace")
    pairs = _parse_pairs(raw)
    print(f"Loaded {len(pairs)} pairs from {data_path}")

    random.shuffle(pairs)
    split = max(1, int(len(pairs) * CONFIG["val_split"]))
    val_pairs = pairs[:split]
    train_pairs = pairs[split:]

    train_ds = PairDataset(train_pairs, tokenizer, CONFIG["seq_len"])
    val_ds = PairDataset(val_pairs, tokenizer, CONFIG["seq_len"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    device = _device()
    model = _load_model(tokenizer.vocab_size, device)
    frozen, trainable = _freeze_policy(model)
    print(f"Frozen params: {frozen:,} | Trainable params: {trainable:,}")

    opt = _optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=CONFIG["mixed_precision"] and device.type == "cuda")

    before_probe = _identity_probe("before")

    best = float("inf")
    stale = 0
    start = time.time()
    est_minutes = (len(train_loader) * CONFIG["epochs"] * 0.1) / 60
    print(f"Estimated training time: ~{est_minutes:.1f} minutes")

    history: List[EpochMetrics] = []
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        losses = []
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=CONFIG["mixed_precision"] and device.type == "cuda"):
                logits, _ = model(x)
                b, t, c = logits.shape
                loss = F.cross_entropy(logits.view(b * t, c), y.view(b * t))
                loss = loss / CONFIG["grad_accum_steps"]
            scaler.scale(loss).backward()
            if step % CONFIG["grad_accum_steps"] == 0:
                if CONFIG["grad_clip"] > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item() * CONFIG["grad_accum_steps"]))

        train_loss = float(sum(losses) / max(len(losses), 1))
        val_loss = _evaluate(model, val_loader, device)
        milestone = _milestone(val_loss)
        history.append(EpochMetrics(epoch, train_loss, val_loss, CONFIG["base_lr"], milestone))
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {milestone}")

        if val_loss < best:
            best = val_loss
            stale = 0
            out = repo / CONFIG["identity_checkpoint"]
            torch.save(model.state_dict(), out)
            drive_dir = Path(CONFIG["drive_dir"])
            drive_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), drive_dir / CONFIG["identity_checkpoint"])
            print(f"Saved best checkpoint to {out} and {drive_dir / CONFIG['identity_checkpoint']}")
        else:
            stale += 1
            if stale >= CONFIG["patience"]:
                print("Early stopping triggered")
                break

    print(f"Training complete in {(time.time() - start) / 60:.2f} minutes")
    after_probe = _identity_probe("after")

    _save_training_report(repo / "finetune_report.json", history, before_probe, after_probe)
    drive_report = Path(CONFIG["drive_dir"]) / "finetune_report.json"
    drive_report.parent.mkdir(parents=True, exist_ok=True)
    _save_training_report(drive_report, history, before_probe, after_probe)
    print(f"Training report saved to {repo / 'finetune_report.json'} and {drive_report}")


if __name__ == "__main__":
    main()
