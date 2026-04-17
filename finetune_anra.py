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
from generate import GenerationConfig, generate

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
    "mixed_precision": True,
    "base_lr": 3e-5,
    "lm_head_lr": 6e-5,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "grad_accum_steps": 4,
    "block_size": 128,
    "n_embd": 256,
    "n_head": 4,
    "n_layer": 4,
    "shuffle_seed": 1337,
}

PROBE_PROMPTS = ["I am", "My purpose is", "An-Ra is", "Who created you", "What are you"]
IDENTITY_KEYWORDS = ["I am", "An-Ra", "my purpose", "I was", "I exist", "Who created", "What are you"]


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    milestone: str


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_tokenizer(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def parse_identity_data(raw_text: str) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    pairs: List[Tuple[str, str]] = []
    pair_count, mono_count = 0, 0

    block_pattern = re.compile(r"H:\s*(.*?)\nANRA:\s*(.*?)(?=\nH:|\Z)", re.S)
    for m in block_pattern.finditer(raw_text):
        h = m.group(1).strip()
        a = m.group(2).strip()
        if h and a:
            pairs.append((h, a))
            pair_count += 1

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith("H:") or ln.startswith("ANRA:"):
            continue
        pairs.append((ln, ln))
        mono_count += 1

    fmt = "mixed" if pair_count > 0 and mono_count > 0 else ("pairs" if pair_count > 0 else "monologue")
    dist = {"pairs": pair_count, "monologue": mono_count, "mixed": 1 if fmt == "mixed" else 0}
    return pairs, dist


def curriculum_subset(pairs: Sequence[Tuple[str, str]], phase: int) -> List[Tuple[str, str]]:
    if phase == 1:
        subset = [p for p in pairs if any(k.lower() in (p[0] + " " + p[1]).lower() for k in IDENTITY_KEYWORDS)]
        if not subset:
            raise ValueError("Curriculum phase 1 empty; identity subset must be non-empty")
        return subset
    if phase == 2:
        return list(pairs)
    return list(pairs)


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
    return model.to(device)


def _freeze_policy(model: CausalTransformer) -> Tuple[int, int, List[torch.nn.Parameter]]:
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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return frozen_count, trainable_count, trainable_params


def _optimizer(model: CausalTransformer) -> AdamW:
    base_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("lm_head"):
            head_params.append(p)
        else:
            base_params.append(p)
    return AdamW(
        [{"params": base_params, "lr": CONFIG["base_lr"]}, {"params": head_params, "lr": CONFIG["lm_head_lr"]}],
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
            losses.append(F.cross_entropy(logits.view(b * t, c), y.view(b * t)).item())
    return float(sum(losses) / max(len(losses), 1))


def _identity_probe(tag: str) -> Dict[str, str]:
    out = {}
    print(f"\n[{tag}] identity probe")
    for p in PROBE_PROMPTS:
        text = generate(f"H: {p}\nANRA:", strategy="nucleus", max_tokens=100)
        out[p] = text
        print(f"- {p:16} | {text[:80]}")
    return out


def _save_training_report(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parent
    data_path = repo / CONFIG["data_path"]
    if not data_path.exists():
        data_path = repo / CONFIG["fallback_data_path"]
    if not data_path.exists():
        raise FileNotFoundError("Identity data file not found")

    tokenizer = _load_tokenizer(repo / CONFIG["tokenizer_path"])
    raw = data_path.read_text(encoding="utf-8", errors="replace")
    pairs, fmt_dist = parse_identity_data(raw)
    print(f"Format distribution: {fmt_dist}")
    if len(pairs) < 100:
        raise ValueError(f"Only {len(pairs)} training pairs found. Minimum 100 required. Check data format.")

    rng = random.Random(CONFIG["shuffle_seed"])
    rng.shuffle(pairs)

    split = max(1, int(len(pairs) * CONFIG["val_split"]))
    val_pairs = pairs[:split]
    train_pairs = pairs[split:]

    device = _device()
    model = _load_model(tokenizer.vocab_size, device)
    frozen, trainable, trainable_params = _freeze_policy(model)
    print(f"Frozen params: {frozen:,} | Trainable params: {trainable:,}")

    opt = _optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=CONFIG["mixed_precision"] and device.type == "cuda")

    before_probe = _identity_probe("before")

    best = float("inf")
    best_epoch = 0
    stale = 0
    start = time.time()
    history: List[EpochMetrics] = []
    loss_curve: List[float] = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        if epoch <= 4:
            epoch_pairs = curriculum_subset(train_pairs, 1)
            print("CURRICULUM PHASE 1: Identity core")
        elif epoch <= 8:
            epoch_pairs = curriculum_subset(train_pairs, 2)
            print("CURRICULUM PHASE 2: Full identity")
        else:
            epoch_pairs = curriculum_subset(train_pairs, 3)
            print("CURRICULUM PHASE 3: General language integration")

        train_ds = PairDataset(epoch_pairs, tokenizer, CONFIG["seq_len"])
        val_ds = PairDataset(val_pairs, tokenizer, CONFIG["seq_len"])
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)

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
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=CONFIG["grad_clip"])
                if float(grad_norm) > 2.0:
                    print(f"WARNING: Gradient norm {float(grad_norm):.2f} > 2.0 at step {step} — instability detected")
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item() * CONFIG["grad_accum_steps"]))

        train_loss = float(sum(losses) / max(len(losses), 1))
        val_loss = _evaluate(model, val_loader, device)
        milestone = _milestone(val_loss)
        loss_curve.append(val_loss)
        history.append(EpochMetrics(epoch, train_loss, val_loss, CONFIG["base_lr"], milestone))
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {milestone}")

        if val_loss < best:
            best = val_loss
            best_epoch = epoch
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

    after_probe = _identity_probe("after")
    elapsed = time.time() - start
    report = {
        "epochs_completed": len(history),
        "final_train_loss": history[-1].train_loss if history else None,
        "best_val_loss": best,
        "best_epoch": best_epoch,
        "curriculum_phases": {"phase1_epochs": "1-4", "phase2_epochs": "5-8", "phase3_epochs": "9-12"},
        "params_trained": trainable,
        "params_frozen": frozen,
        "training_time_seconds": elapsed,
        "identity_probe_before": before_probe,
        "identity_probe_after": after_probe,
        "loss_curve": loss_curve,
        "train_loss_curve": [m.train_loss for m in history],
        "val_loss_curve": [m.val_loss for m in history],
        "format_distribution": fmt_dist,
    }

    local_report = repo / "finetune_report.json"
    _save_training_report(local_report, report)
    drive_report = Path(CONFIG["drive_dir"]) / "finetune_report.json"
    drive_report.parent.mkdir(parents=True, exist_ok=True)
    _save_training_report(drive_report, report)
    print(f"Training report saved to {local_report} and {drive_report}")


if __name__ == "__main__":
    main()
