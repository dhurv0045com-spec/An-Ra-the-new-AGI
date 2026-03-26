"""
45G / run.py — Master Entry Point
===================================
Single command to load a model and generate text.

Usage:
    python run.py --prompt "Your text here" \\
                  --checkpoint best_model.pt \\
                  --strategy top_p \\
                  --temperature 0.8 \\
                  --max_tokens 200

All flags:
  --prompt TEXT           Input prompt (required, or use --prompt_file)
  --prompt_file FILE      Read prompt from file (overrides --prompt)
  --checkpoint PATH       .pt checkpoint to load  [required]
  --strategy STR          greedy | temperature | top_k | top_p  [top_p]
  --temperature FLOAT     Sampling temperature  [0.8]
  --top_k INT             Top-k cutoff  [50]
  --top_p FLOAT           Nucleus mass  [0.95]
  --max_tokens INT        Maximum new tokens  [200]
  --repetition_penalty F  Repetition penalty  [1.0]
  --stop STR              Stop token string (repeatable)
  --stream                Stream output token by token
  --batch_file FILE       JSON file with list of prompts — run all and save
  --device STR            cpu | cuda | cuda:0  [auto]
  --eval                  Run full evaluation suite instead of generating
  --eval_corpus FILE      Text file for perplexity evaluation
  --list_checkpoints DIR  Show available checkpoints in a directory and exit
  --no_header             Suppress the [inference] log lines
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Optional

import torch

# 45G modules
from inference import InferencePipeline, GenerationConfig
from model_io import load_checkpoint, list_checkpoints, print_checkpoint_table
from evaluate import run_eval_suite

# ─────────────────────────────────────────────
# Built-in small transformer for standalone use
# (used when no external model class is provided)
# ─────────────────────────────────────────────

import torch.nn as nn
import torch.nn.functional as F
import math


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerLM(nn.Module):
    """
    Compact decoder-only transformer language model.
    Compatible with the full 45A-45F build pipeline.

    Config keys (all passed as kwargs or via a dict):
      vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        self.emb    = nn.Embedding(vocab_size, d_model)
        self.pos    = _PositionalEncoding(d_model, max_seq_len)
        self.drop   = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,  # pre-norm = GPT2 style
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm   = nn.LayerNorm(d_model)
        self.proj   = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embedding and output projection share weights
        self.proj.weight = self.emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb.weight, std=0.02)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask so each token only attends to past tokens."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) integer token ids

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        _, seq_len = input_ids.shape
        mask = self._causal_mask(seq_len, input_ids.device)

        x = self.drop(self.pos(self.emb(input_ids)))
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.norm(x)
        return self.proj(x)


# ─────────────────────────────────────────────
# Tokenizer — bundled char-level (no HuggingFace needed)
# ─────────────────────────────────────────────

class CharTokenizer:
    """
    Character-level tokenizer.
    Covers full printable ASCII (vocab_size = 128).
    Connects to the tokenizer built in Step 10.
    """
    def __init__(self, vocab: Optional[List[str]] = None):
        if vocab is None:
            vocab = [chr(i) for i in range(128)]
        self.vocab    = vocab
        self.ch2id    = {c: i for i, c in enumerate(vocab)}
        self.id2ch    = {i: c for i, c in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> List[int]:
        return [self.ch2id.get(c, 0) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id2ch.get(i, "?") for i in ids)

    def to_dict(self) -> dict:
        return {"vocab": self.vocab}

    @staticmethod
    def from_dict(d: dict) -> "CharTokenizer":
        return CharTokenizer(vocab=d["vocab"])


# ─────────────────────────────────────────────
# Model loader — handles external .pt checkpoints
# ─────────────────────────────────────────────

def _build_model_from_checkpoint(
    ckpt_path: str,
    device: str,
) -> tuple:
    """
    Load a .pt checkpoint and reconstruct the model + tokenizer.

    Reads architecture config from checkpoint metadata when available;
    falls back to sensible defaults so the pipeline still works with
    checkpoints that predate the metadata schema.

    Returns: (model, tokenizer, metadata)
    """
    import torch
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Reconstruct tokenizer
    tok_dict = payload.get("tokenizer")
    if tok_dict and "vocab" in tok_dict:
        tokenizer = CharTokenizer.from_dict(tok_dict)
    else:
        tokenizer = CharTokenizer()

    # Reconstruct model architecture from saved metadata
    meta_dict = payload.get("metadata", {})
    cfg = payload.get("config", {})

    model = TransformerLM(
        vocab_size  = meta_dict.get("vocab_size",  tokenizer.vocab_size),
        d_model     = meta_dict.get("d_model",     128),
        n_heads     = meta_dict.get("n_heads",     4),
        n_layers    = meta_dict.get("n_layers",    2),
        d_ff        = meta_dict.get("d_ff",        512),
        max_seq_len = meta_dict.get("max_seq_len", 512),
    )

    from model_io import load_checkpoint, CheckpointMetadata
    meta = load_checkpoint(model, ckpt_path, device=device, strict=False)
    model.to(device)

    return model, tokenizer, meta


# ─────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="45G Inference Engine — generate text from a trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    p.add_argument("--prompt",        type=str, default="Once upon a time")
    p.add_argument("--prompt_file",   type=str, default=None,
                   help="Read prompt from this text file")
    p.add_argument("--checkpoint",    type=str, default=None,
                   help=".pt checkpoint file. If omitted, uses untrained model.")

    # Sampling
    p.add_argument("--strategy",    type=str,   default="top_p",
                   choices=["greedy", "temperature", "top_k", "top_p"])
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k",       type=int,   default=50)
    p.add_argument("--top_p",       type=float, default=0.95)
    p.add_argument("--max_tokens",  type=int,   default=200)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--stop",        action="append", default=[],
                   metavar="TOKEN", help="Stop generation when this string appears")
    p.add_argument("--stream",      action="store_true",
                   help="Stream output token by token")

    # Batch
    p.add_argument("--batch_file", type=str, default=None,
                   help="JSON list of prompts; run all and save to batch_output.json")

    # System
    p.add_argument("--device", type=str, default=None,
                   help="cpu | cuda | cuda:0 (auto-detected if omitted)")
    p.add_argument("--no_header", action="store_true",
                   help="Suppress [inference] log lines")

    # Utility modes
    p.add_argument("--eval",         action="store_true",
                   help="Run full eval suite instead of generating")
    p.add_argument("--eval_corpus",  type=str, default=None,
                   help="Text file for perplexity eval")
    p.add_argument("--list_checkpoints", type=str, default=None, metavar="DIR",
                   help="List checkpoints in DIR and exit")

    return p


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # ── --list_checkpoints ─────────────────────
    if args.list_checkpoints:
        print_checkpoint_table(args.list_checkpoints)
        return 0

    # ── device ─────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] device={device}")

    # ── suppress headers ───────────────────────
    if args.no_header:
        import builtins
        _orig_print = builtins.print
        def _filtered(*a, **kw):
            if a and isinstance(a[0], str) and a[0].startswith("["):
                return
            _orig_print(*a, **kw)
        builtins.print = _filtered

    # ── load model ─────────────────────────────
    if args.checkpoint:
        print(f"[run] Loading checkpoint: {args.checkpoint}")
        model, tokenizer, meta = _build_model_from_checkpoint(args.checkpoint, device)
    else:
        print("[run] No checkpoint — using untrained model with random weights")
        tokenizer = CharTokenizer()
        model = TransformerLM(vocab_size=tokenizer.vocab_size)
        model.to(device)

    # ── eval mode ──────────────────────────────
    if args.eval:
        corpus = ""
        if args.eval_corpus:
            corpus = Path(args.eval_corpus).read_text(encoding="utf-8")
        else:
            corpus = "the quick brown fox jumps over the lazy dog " * 100

        pipe = InferencePipeline(model, tokenizer, device=device)
        run_eval_suite(model, tokenizer, corpus,
                       inference_pipeline=pipe,
                       label=Path(args.checkpoint).stem if args.checkpoint else "untrained",
                       device=device)
        return 0

    # ── build pipeline ─────────────────────────
    pipe = InferencePipeline(model, tokenizer, device=device)
    cfg  = GenerationConfig(
        strategy           = args.strategy,
        temperature        = args.temperature,
        top_k              = args.top_k,
        top_p              = args.top_p,
        max_new_tokens     = args.max_tokens,
        stop_tokens        = args.stop,
        stream             = args.stream,
        repetition_penalty = args.repetition_penalty,
    )

    # ── batch mode ─────────────────────────────
    if args.batch_file:
        prompts = json.loads(Path(args.batch_file).read_text())
        print(f"[run] Batch mode: {len(prompts)} prompts")
        results = pipe.batch_generate(prompts, config=cfg)
        out_path = "batch_output.json"
        with open(out_path, "w") as f:
            json.dump([{"prompt": p, "output": o}
                       for p, o in zip(prompts, results)], f, indent=2)
        print(f"[run] Saved to {out_path}")
        return 0

    # ── single prompt ──────────────────────────
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    print(f"\nPrompt: {prompt!r}\n")
    print("─" * 56)

    if args.stream:
        for piece in pipe.stream(prompt, config=cfg):
            print(piece, end="", flush=True)
        print("\n" + "─" * 56)
    else:
        output = pipe.generate(prompt, config=cfg)
        print(output)
        print("─" * 56)

    return 0


if __name__ == "__main__":
    sys.exit(main())
