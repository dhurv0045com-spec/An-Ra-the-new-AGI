"""
finetune_anra.py
================
Fine-tuning pipeline for An-Ra — sovereign AI identity conditioning.

Strategy: Selective Layer Unfreezing + Differential Learning Rates
  - Embedding layers: FROZEN (preserve character-level encoding)
  - First 25% of transformer blocks: FROZEN (low-level syntax, spacing, structure)
  - Remaining blocks: TRAINED at base LR (identity, reasoning, voice)
  - Final LayerNorm + lm_head: TRAINED at 2× base LR (output calibration)

Why not LoRA: At 3.24M params, LoRA adapter overhead is not worth it. The model
is small enough to directly fine-tune the target layers. LoRA shines at 7B+.
Why not full fine-tune: Catastrophic forgetting risk on a small dataset (2000 lines).
Freezing early layers acts as a structural regularizer.
"""

import os
import re
import sys
import time
import math
import pickle
import random
import struct
import collections
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# ─────────────────────────────────────────────
#  CONFIG — all hyperparams in one place
# ─────────────────────────────────────────────
CONFIG = {
    # Paths
    "base_checkpoint":    "anra_brain.pt",
    "finetune_checkpoint":"anra_brain_identity.pt",
    "tokenizer_path":     "tokenizer.pkl",
    "data_path":          "anra_identity_data.txt",   # your ~2000 line file
    "drive_dir":          "/content/drive/MyDrive/AnRa/",

    # Training
    "epochs":             12,          # 8–15 is the sweet spot for 2000-line identity data
    "batch_size":         16,          # reduce to 8 if OOM on Colab
    "seq_len":            256,         # context window per sample
    "base_lr":            2e-4,        # lower than base training (typically 3e-3 → 2e-4)
    "head_lr_multiplier": 2.0,         # lm_head + final norm get 2× base_lr
    "weight_decay":       1e-2,
    "grad_clip":          1.0,
    "warmup_steps":       100,

    # Freezing strategy (fraction of transformer blocks to freeze from bottom)
    "freeze_fraction":    0.25,        # freeze bottom 25% of blocks
    "freeze_embeddings":  True,

    # Memory optimisation
    "gradient_checkpointing": True,
    "mixed_precision":         True,   # fp16 — Colab T4/A100 both support this
    "accumulation_steps":      2,      # effective batch = batch_size × accumulation_steps

    # Logging
    "log_every":          25,          # print loss every N steps
    "save_every_epoch":   True,

    # Curriculum learning — sort samples by length (short→long) for first N epochs
    "curriculum_epochs":  3,

    # Data
    "val_split":          0.05,        # 5% held out for validation loss
    "min_seq_len":        8,           # discard samples shorter than this
}


# ─────────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[device] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("[device] CPU — training will be slow")
    return dev


# ─────────────────────────────────────────────
#  TOKENIZER  (loads CharTokenizer from .pkl)
# ─────────────────────────────────────────────
def load_tokenizer(path: str):
    with open(path, "rb") as f:
        tok = pickle.load(f)
    print(f"[tokenizer] vocab size = {tok.vocab_size}")
    return tok


def encode(tok, text: str) -> List[int]:
    """Encode text → token ids. Handles both .encode() and direct char-map."""
    if hasattr(tok, "encode"):
        return tok.encode(text)
    if hasattr(tok, "stoi"):
        return [tok.stoi.get(c, 0) for c in text]
    if hasattr(tok, "char_to_idx"):
        return [tok.char_to_idx.get(c, 0) for c in text]
    raise AttributeError("Tokenizer has no encode/stoi/char_to_idx attribute")


def decode(tok, ids: List[int]) -> str:
    if hasattr(tok, "decode"):
        return tok.decode(ids)
    if hasattr(tok, "itos"):
        return "".join(tok.itos.get(i, "") for i in ids)
    if hasattr(tok, "idx_to_char"):
        return "".join(tok.idx_to_char.get(i, "") for i in ids)
    raise AttributeError("Tokenizer has no decode/itos/idx_to_char attribute")


# ─────────────────────────────────────────────
#  DATA FORMAT DETECTION & PARSING
# ─────────────────────────────────────────────
QA_PATTERNS = [
    # Q: ... / A: ...
    (re.compile(r"^Q\s*:\s*(.+)", re.IGNORECASE),
     re.compile(r"^A\s*:\s*(.+)", re.IGNORECASE)),
    # Human: ... / An-Ra: ...
    (re.compile(r"^(Human|User|Person)\s*:\s*(.+)", re.IGNORECASE),
     re.compile(r"^(An-Ra|AnRa|AI|Assistant)\s*:\s*(.+)", re.IGNORECASE)),
    # ### Human / ### Assistant markdown style
    (re.compile(r"^#+\s*(Human|User)\s*[:\-]?\s*(.+)", re.IGNORECASE),
     re.compile(r"^#+\s*(An-Ra|AnRa|Assistant)\s*[:\-]?\s*(.+)", re.IGNORECASE)),
]

def detect_format(lines: List[str]) -> str:
    """Returns 'qa', 'monologue', or 'mixed'."""
    qa_score = 0
    mono_score = 0
    sample = lines[:min(200, len(lines))]
    for line in sample:
        stripped = line.strip()
        if not stripped:
            continue
        for q_pat, a_pat in QA_PATTERNS:
            if q_pat.match(stripped) or a_pat.match(stripped):
                qa_score += 1
                break
        else:
            if len(stripped) > 20:
                mono_score += 1

    total = qa_score + mono_score
    if total == 0:
        return "monologue"
    qa_ratio = qa_score / total
    if qa_ratio > 0.4:
        return "qa"
    if qa_ratio > 0.15:
        return "mixed"
    return "monologue"


def parse_qa_to_text(lines: List[str]) -> List[str]:
    """
    Convert Q&A pairs into continuations An-Ra can learn from.
    Format: '<question>\n<answer>' — the model learns to complete the answer.
    """
    segments = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # Try to match a question line
        matched_q = None
        for q_pat, _ in QA_PATTERNS:
            m = q_pat.match(line)
            if m:
                matched_q = m.group(m.lastindex) if m.lastindex else m.group(0)
                break
        if matched_q is not None and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            for _, a_pat in QA_PATTERNS:
                m = a_pat.match(next_line)
                if m:
                    ans = m.group(m.lastindex) if m.lastindex else m.group(0)
                    segments.append(f"{matched_q}\n{ans}")
                    i += 2
                    break
            else:
                segments.append(line)
                i += 1
        else:
            if len(line) > 10:
                segments.append(line)
            i += 1
    return segments


def parse_monologue(lines: List[str]) -> List[str]:
    """Group blank-line-separated paragraphs into segments."""
    segments = []
    current = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            current.append(stripped)
        else:
            if current:
                segments.append(" ".join(current))
                current = []
    if current:
        segments.append(" ".join(current))
    return segments


def load_and_parse_data(path: str) -> Tuple[List[str], str]:
    """Load data file, detect format, return (segments, format_name)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    fmt = detect_format(lines)
    print(f"[data] Detected format: '{fmt}' ({len(lines)} raw lines)")

    if fmt == "qa":
        segments = parse_qa_to_text(lines)
    elif fmt == "mixed":
        # Handle both styles; non-QA lines go in as monologue
        qa_segs = parse_qa_to_text(lines)
        mono_segs = parse_monologue(lines)
        # Deduplicate roughly
        seen = set()
        segments = []
        for s in qa_segs + mono_segs:
            key = s[:40]
            if key not in seen and len(s) > 10:
                seen.add(key)
                segments.append(s)
    else:
        segments = parse_monologue(lines)

    print(f"[data] Parsed {len(segments)} segments in '{fmt}' mode")
    return segments, fmt


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class IdentityDataset(Dataset):
    """
    Sliding window over token sequences.
    Each item: (input_ids[seq_len], target_ids[seq_len])
    """
    def __init__(self, segments: List[str], tok, seq_len: int,
                 min_len: int = 8, curriculum: bool = False):
        self.seq_len = seq_len
        self.samples: List[List[int]] = []

        for seg in segments:
            ids = encode(tok, seg)
            if len(ids) < min_len:
                continue
            # Stride = seq_len // 2 for overlap → richer training signal
            stride = max(1, seq_len // 2)
            for start in range(0, max(1, len(ids) - seq_len), stride):
                chunk = ids[start: start + seq_len + 1]
                if len(chunk) > min_len:
                    self.samples.append(chunk)

        if curriculum:
            self.samples.sort(key=len)

        print(f"[dataset] {len(self.samples)} samples (seq_len={seq_len})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        # Pad to seq_len + 1 if needed
        if len(ids) < self.seq_len + 1:
            ids = ids + [0] * (self.seq_len + 1 - len(ids))
        x = torch.tensor(ids[:self.seq_len], dtype=torch.long)
        y = torch.tensor(ids[1:self.seq_len + 1], dtype=torch.long)
        return x, y


def make_loaders(segments: List[str], tok, cfg: dict, curriculum_epoch: bool
                 ) -> Tuple[DataLoader, DataLoader]:
    random.shuffle(segments)
    split = max(1, int(len(segments) * cfg["val_split"]))
    train_segs = segments[split:]
    val_segs   = segments[:split]

    train_ds = IdentityDataset(train_segs, tok, cfg["seq_len"],
                               cfg["min_seq_len"], curriculum=curriculum_epoch)
    val_ds   = IdentityDataset(val_segs,   tok, cfg["seq_len"],
                               cfg["min_seq_len"], curriculum=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=not curriculum_epoch, num_workers=0,
                              pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


# ─────────────────────────────────────────────
#  CHECKPOINT LOADING
# ─────────────────────────────────────────────
def load_checkpoint(path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """
    Load anra_brain.pt. Handles multiple checkpoint formats:
      - Raw state_dict
      - {'model': state_dict, 'config': {...}}
      - {'model_state_dict': ..., 'model_config': ...}
      - Full model object (torch.save(model, ...))
    """
    print(f"[checkpoint] Loading {path} ...")
    ckpt = torch.load(path, map_location=device)

    # Case 1: raw nn.Module
    if isinstance(ckpt, nn.Module):
        model = ckpt
        model_cfg = {}
        print("[checkpoint] Format: raw nn.Module")
        return model, model_cfg

    # Extract state dict and config
    state_dict = None
    model_cfg  = {}

    if isinstance(ckpt, dict):
        # Try common keys
        for sd_key in ("model", "model_state_dict", "state_dict", "weights"):
            if sd_key in ckpt:
                state_dict = ckpt[sd_key]
                break
        for cfg_key in ("config", "model_config", "cfg", "hparams"):
            if cfg_key in ckpt:
                model_cfg = ckpt[cfg_key]
                break
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            # The dict IS the state_dict
            state_dict = ckpt

    if state_dict is None:
        raise ValueError(f"Cannot find state_dict in checkpoint. Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    # Build model from inferred config
    model = build_model_from_state_dict(state_dict, model_cfg, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[checkpoint] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[checkpoint] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    print(f"[checkpoint] Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, model_cfg


def infer_arch(state_dict: dict) -> dict:
    """Infer model architecture from state_dict key shapes."""
    cfg = {}

    # vocab size from embedding or lm_head
    for key in state_dict:
        if "embed" in key and "weight" in key:
            shape = state_dict[key].shape
            cfg["vocab_size"] = shape[0]
            cfg["n_embd"]     = shape[1]
            break
    if "vocab_size" not in cfg:
        for key in state_dict:
            if ("lm_head" in key or "head" in key) and "weight" in key:
                shape = state_dict[key].shape
                cfg["vocab_size"] = shape[0]
                cfg["n_embd"]     = shape[-1]
                break

    # n_layer from counting attention blocks
    block_indices = set()
    for key in state_dict:
        m = re.search(r"blocks?[.\[](\d+)", key)
        if m:
            block_indices.add(int(m.group(1)))
        m = re.search(r"layers?[.\[](\d+)", key)
        if m:
            block_indices.add(int(m.group(1)))
        m = re.search(r"h[.\[](\d+)", key)   # GPT-2 style
        if m:
            block_indices.add(int(m.group(1)))
    cfg["n_layer"] = max(block_indices) + 1 if block_indices else 4

    # n_head from attention weight shape
    for key in state_dict:
        if ("attn" in key or "attention" in key) and "weight" in key:
            shape = state_dict[key].shape
            if len(shape) == 2 and "n_embd" in cfg:
                embd = cfg["n_embd"]
                # qkv projection: shape[0] == 3*embd
                if shape[0] == 3 * embd:
                    # default head count
                    cfg["n_head"] = max(1, embd // 64)
                    break
    if "n_head" not in cfg and "n_embd" in cfg:
        cfg["n_head"] = max(1, cfg["n_embd"] // 64)

    # n_embd fallback
    if "n_embd" not in cfg:
        cfg["n_embd"] = 128

    print(f"[arch] Inferred: vocab={cfg.get('vocab_size','?')} "
          f"embd={cfg.get('n_embd','?')} "
          f"layers={cfg.get('n_layer','?')} "
          f"heads={cfg.get('n_head','?')}")
    return cfg


def build_model_from_state_dict(state_dict: dict, model_cfg: dict,
                                device: torch.device) -> nn.Module:
    """Reconstruct the GPT-style model that matches the state_dict."""
    inferred = infer_arch(state_dict)
    # model_cfg (from checkpoint) overrides inferred where present
    merged = {**inferred, **model_cfg}

    vocab_size = merged.get("vocab_size", 93)
    n_embd     = merged.get("n_embd", 128)
    n_layer    = merged.get("n_layer", 4)
    n_head     = merged.get("n_head", max(1, n_embd // 64))
    block_size = merged.get("block_size", merged.get("seq_len", 256))
    dropout    = merged.get("dropout", 0.1)

    model = AnRaGPT(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer,
                    n_head=n_head, block_size=block_size, dropout=dropout)
    model = model.to(device)
    return model


# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
#  (mirrored from typical AnRa GPT build)
# ─────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head  = n_head
        self.n_embd  = n_embd
        self.head_dim = n_embd // n_head

        self.c_attn  = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj  = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer("bias",
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = MLP(n_embd, dropout)
        self.use_checkpoint = False

    def _forward_impl(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward(self, x):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)
        return self._forward_impl(x)


class AnRaGPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_layer: int,
                 n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(vocab_size, n_embd),
            "wpe": nn.Embedding(block_size, n_embd),
            "drop": nn.Dropout(dropout),
            "h": nn.ModuleList([
                Block(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]),
            "ln_f": nn.LayerNorm(n_embd),
        })
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.transformer["wte"].weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence {T} > block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)

        x = self.transformer["drop"](
            self.transformer["wte"](idx) +
            self.transformer["wpe"](pos)
        )
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int,
                 temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        self.eval()
        for _ in range(max_new):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ─────────────────────────────────────────────
#  FREEZING STRATEGY
# ─────────────────────────────────────────────
def apply_freezing(model: nn.Module, cfg: dict) -> Dict[str, Any]:
    """
    Freeze embeddings and bottom fraction of transformer blocks.
    Returns param groups with differential learning rates.
    """
    base_lr = cfg["base_lr"]
    head_lr = base_lr * cfg["head_lr_multiplier"]

    # 1. Freeze embeddings
    if cfg["freeze_embeddings"]:
        for name in ("wte", "wpe", "drop"):
            if name in model.transformer:
                for p in model.transformer[name].parameters():
                    p.requires_grad = False

    # 2. Freeze bottom fraction of blocks
    blocks = model.transformer["h"]
    n_freeze = max(0, int(len(blocks) * cfg["freeze_fraction"]))
    for i, block in enumerate(blocks):
        if i < n_freeze:
            for p in block.parameters():
                p.requires_grad = False

    # 3. Enable gradient checkpointing on trainable blocks
    if cfg["gradient_checkpointing"]:
        for i, block in enumerate(blocks):
            if i >= n_freeze:
                block.use_checkpoint = True

    # Summary
    total   = sum(p.numel() for p in model.parameters())
    frozen  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = total - frozen
    print(f"[freeze] Frozen {n_freeze}/{len(blocks)} blocks + "
          f"{'embeddings' if cfg['freeze_embeddings'] else 'no embeddings'}")
    print(f"[freeze] Trainable: {trainable:,} / {total:,} params "
          f"({100*trainable/total:.1f}%)")

    # 4. Build param groups for differential LR
    head_params   = list(model.lm_head.parameters()) + \
                    list(model.transformer["ln_f"].parameters())
    head_ids      = {id(p) for p in head_params}
    other_params  = [p for p in model.parameters()
                     if p.requires_grad and id(p) not in head_ids]

    head_params_req = [p for p in head_params if p.requires_grad]

    param_groups = [
        {"params": other_params,     "lr": base_lr,  "name": "body"},
        {"params": head_params_req,  "lr": head_lr,  "name": "head"},
    ]
    return param_groups


# ─────────────────────────────────────────────
#  LR SCHEDULE  (cosine with linear warmup)
# ─────────────────────────────────────────────
def get_lr(step: int, warmup: int, total_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    # Cosine decay to 10% of base_lr
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def update_lr(optimizer: torch.optim.Optimizer, step: int, warmup: int,
              total_steps: int, base_lr: float, head_multiplier: float):
    lr = get_lr(step, warmup, total_steps, base_lr)
    for group in optimizer.param_groups:
        mult = head_multiplier if group.get("name") == "head" else 1.0
        group["lr"] = lr * mult


# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device, amp: bool) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(enabled=amp):
            _, loss = model(x, y)
        if loss is not None:
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


# ─────────────────────────────────────────────
#  IDENTITY PROBE  (quick qualitative check)
# ─────────────────────────────────────────────
PROBE_PROMPTS = [
    "I am",
    "My purpose",
    "What am I",
    "An-Ra is",
]

def identity_probe(model: nn.Module, tok, device: torch.device, n_tokens: int = 80):
    print("\n" + "═" * 60)
    print("  IDENTITY PROBE")
    print("═" * 60)
    model.eval()
    for prompt in PROBE_PROMPTS:
        ids = encode(tok, prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new=n_tokens, temperature=0.7, top_k=30)
        text = decode(tok, out[0].tolist())
        print(f"\n  Prompt: '{prompt}'")
        print(f"  → {text[:160]}")
    print("═" * 60 + "\n")
    model.train()


# ─────────────────────────────────────────────
#  GOOGLE DRIVE SAVE
# ─────────────────────────────────────────────
def save_to_drive(src: str, drive_dir: str):
    dst = Path(drive_dir)
    if not dst.exists():
        print(f"[drive] Directory {drive_dir} not found — skipping Drive save")
        return
    import shutil
    dest_path = dst / Path(src).name
    shutil.copy2(src, dest_path)
    print(f"[drive] Saved → {dest_path}")


# ─────────────────────────────────────────────
#  SAVE CHECKPOINT
# ─────────────────────────────────────────────
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, step: int, val_loss: float,
                    cfg: dict, path: str):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step":  step,
        "val_loss": val_loss,
        "config": {
            "n_embd":      model.transformer["wte"].weight.shape[1],
            "vocab_size":  model.transformer["wte"].weight.shape[0],
            "n_layer":     len(model.transformer["h"]),
            "n_head":      model.transformer["h"][0].attn.n_head,
            "block_size":  model.block_size,
        },
        "finetune_config": cfg,
    }
    torch.save(ckpt, path)
    print(f"[save] {path}  (epoch={epoch}, val_loss={val_loss:.4f})")
    save_to_drive(path, cfg["drive_dir"])


# ─────────────────────────────────────────────
#  RESUME
# ─────────────────────────────────────────────
def try_resume(model: nn.Module, optimizer: torch.optim.Optimizer,
               cfg: dict, device: torch.device) -> Tuple[int, int]:
    """If fine-tuned checkpoint exists, resume from it. Returns (start_epoch, step)."""
    path = cfg["finetune_checkpoint"]
    if not os.path.exists(path):
        return 0, 0
    print(f"[resume] Found {path} — resuming fine-tuning")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as e:
        print(f"[resume] Could not restore optimizer state: {e}")
    epoch = ckpt.get("epoch", 0)
    step  = ckpt.get("step",  0)
    print(f"[resume] Resuming from epoch {epoch}, step {step}, "
          f"val_loss={ckpt.get('val_loss', '?'):.4f}")
    return epoch, step


# ─────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────
def train(cfg: dict):
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  An-Ra Identity Fine-Tuning Pipeline".center(58) + "║")
    print("╚" + "═"*58 + "╝\n")

    device = get_device()
    amp    = cfg["mixed_precision"] and torch.cuda.is_available()
    scaler = GradScaler(enabled=amp)

    # ── Tokenizer ──────────────────────────────
    tok = load_tokenizer(cfg["tokenizer_path"])

    # ── Load base model ────────────────────────
    model, _ = load_checkpoint(cfg["base_checkpoint"], device)
    model.train()

    # ── Data ───────────────────────────────────
    segments, fmt = load_and_parse_data(cfg["data_path"])
    if len(segments) < 10:
        raise ValueError(f"Only {len(segments)} segments parsed — check data file path/format")

    # ── Freezing + param groups ────────────────
    param_groups = apply_freezing(model, cfg)
    optimizer = torch.optim.AdamW(param_groups,
                                  weight_decay=cfg["weight_decay"],
                                  betas=(0.9, 0.95))

    # ── Resume ─────────────────────────────────
    start_epoch, global_step = try_resume(model, optimizer, cfg, device)

    # ── Training plan ──────────────────────────
    # Estimate steps
    sample_ds = IdentityDataset(segments, tok, cfg["seq_len"], cfg["min_seq_len"])
    n_train   = int(len(sample_ds) * (1 - cfg["val_split"]))
    steps_per_epoch = max(1, n_train // cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    print(f"\n[plan] {cfg['epochs']} epochs × ~{steps_per_epoch} steps "
          f"= ~{total_steps} total steps")
    print(f"[plan] LR schedule: warmup {cfg['warmup_steps']} → cosine decay")
    print(f"[plan] Mixed precision: {amp} | Grad accum: {cfg['accumulation_steps']}×")
    del sample_ds

    # ── Initial probe ──────────────────────────
    print("\n[probe] Before fine-tuning:")
    identity_probe(model, tok, device)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, cfg["epochs"]):
        curriculum = epoch < cfg["curriculum_epochs"]
        train_loader, val_loader = make_loaders(segments, tok, cfg, curriculum)

        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Update LR
            update_lr(optimizer, global_step, cfg["warmup_steps"],
                      total_steps, cfg["base_lr"], cfg["head_lr_multiplier"])

            with autocast(enabled=amp):
                _, loss = model(x, y)
                loss = loss / cfg["accumulation_steps"]

            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg["accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg["grad_clip"]
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            raw_loss = loss.item() * cfg["accumulation_steps"]
            epoch_loss  += raw_loss
            epoch_steps += 1

            if global_step % cfg["log_every"] == 0 and global_step > 0:
                elapsed  = time.time() - t0
                lr_now   = optimizer.param_groups[0]["lr"]
                avg_loss = epoch_loss / epoch_steps
                print(f"  epoch {epoch+1:>2}/{cfg['epochs']}  "
                      f"step {global_step:>6}  "
                      f"loss {raw_loss:.4f}  "
                      f"avg {avg_loss:.4f}  "
                      f"lr {lr_now:.2e}  "
                      f"{elapsed:.0f}s")
                t0 = time.time()

        # ── End of epoch ──────────────────────
        val_loss  = evaluate(model, val_loader, device, amp)
        avg_train = epoch_loss / max(epoch_steps, 1)
        elapsed   = time.time() - t0

        print(f"\n{'─'*60}")
        print(f"  EPOCH {epoch+1}/{cfg['epochs']}  "
              f"train_loss={avg_train:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"elapsed={elapsed:.1f}s")

        # Loss interpretation
        if val_loss < 1.0:
            print(f"  ★ EXCELLENT — identity deeply conditioned (val_loss < 1.0)")
        elif val_loss < 1.5:
            print(f"  ✓ GOOD — identity forming (val_loss < 1.5)")
        elif val_loss < 2.0:
            print(f"  ~ PROGRESSING — continue training")
        else:
            print(f"  △ EARLY — still adapting")

        print(f"{'─'*60}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch + 1, global_step,
                            val_loss, cfg, cfg["finetune_checkpoint"])

        elif cfg["save_every_epoch"]:
            save_checkpoint(model, optimizer, epoch + 1, global_step,
                            val_loss, cfg, cfg["finetune_checkpoint"])

    # ── Final probe ───────────────────────────
    print("\n[probe] After fine-tuning:")
    identity_probe(model, tok, device)

    print(f"\n[done] Best val_loss: {best_val_loss:.4f}")
    print(f"[done] Fine-tuned weights: {cfg['finetune_checkpoint']}")
    print(f"[done] Original weights:   {cfg['base_checkpoint']}  (untouched)")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="An-Ra Identity Fine-Tuning")
    parser.add_argument("--data",     default=CONFIG["data_path"],
                        help="Path to identity training data file")
    parser.add_argument("--epochs",   type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr",       type=float, default=CONFIG["base_lr"])
    parser.add_argument("--batch",    type=int, default=CONFIG["batch_size"])
    parser.add_argument("--seq",      type=int, default=CONFIG["seq_len"])
    parser.add_argument("--no-amp",   action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--probe-only", action="store_true",
                        help="Run identity probe on existing checkpoint and exit")
    args = parser.parse_args()

    CONFIG["data_path"]       = args.data
    CONFIG["epochs"]          = args.epochs
    CONFIG["base_lr"]         = args.lr
    CONFIG["batch_size"]      = args.batch
    CONFIG["seq_len"]         = args.seq
    if args.no_amp:
        CONFIG["mixed_precision"] = False

    if args.probe_only:
        device = get_device()
        tok    = load_tokenizer(CONFIG["tokenizer_path"])
        ckpt_path = (CONFIG["finetune_checkpoint"]
                     if os.path.exists(CONFIG["finetune_checkpoint"])
                     else CONFIG["base_checkpoint"])
        model, _ = load_checkpoint(ckpt_path, device)
        identity_probe(model, tok, device, n_tokens=120)
        sys.exit(0)

    train(CONFIG)
