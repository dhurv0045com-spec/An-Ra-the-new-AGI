"""
45G / Step 33 — Model Evaluation & Benchmarking
=================================================
Full evaluation suite for language models.

  perplexity()        — perplexity on any token sequence
  speed_benchmark()   — tokens/second across batch sizes and lengths
  memory_profile()    — peak GPU/CPU memory during forward pass
  generation_quality()— coherence, diversity, repetition scores
  compare_checkpoints()— before-vs-after diff table
  run_eval_suite()    — one call runs everything, writes report to disk
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import platform
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Perplexity
# ─────────────────────────────────────────────

@torch.no_grad()
def perplexity(
    model: nn.Module,
    token_ids: List[int],
    context_len: int = 128,
    stride: int = 64,
    device: str = "cpu",
) -> float:
    """
    Compute perplexity on a token sequence using a sliding-window approach.

    Sliding window avoids giving the model extra context it wouldn't have
    during real autoregressive inference, while still evaluating on full
    corpora longer than the model's context window.

    Args:
        model:       Language model; forward(ids) → logits (B, seq, vocab).
        token_ids:   Flat list of integer token ids (the test corpus).
        context_len: Maximum context window passed to the model.
        stride:      How many tokens to advance the window each step.
                     stride == context_len → non-overlapping (faster, noisier).
                     stride < context_len  → overlapping (slower, more accurate).
        device:      Compute device.

    Returns:
        Perplexity as a float. Lower is better.
        A random model on vocab V gives perplexity ≈ V.
    """
    model.eval()
    model.to(device)

    ids     = torch.tensor(token_ids, dtype=torch.long, device=device)
    n       = len(ids)
    total_nll = 0.0        # accumulated negative log-likelihood
    total_tok = 0          # number of evaluated tokens

    for begin in range(0, n - 1, stride):
        end = min(begin + context_len, n)
        chunk = ids[begin:end].unsqueeze(0)          # (1, chunk_len)

        out = model(chunk)
        if isinstance(out, tuple):
            out = out[0]
        logits = out[0] if out.dim() == 3 else out   # (seq, vocab)

        # Predict tokens at positions 1..end, using context at 0..end-1
        # When the window is the first chunk, evaluate all tokens.
        # For subsequent overlapping windows, only score the strided portion.
        eval_start = 0 if begin == 0 else max(0, context_len - stride)
        target_ids = chunk[0, eval_start + 1:]
        pred_logits = logits[eval_start : eval_start + len(target_ids)]

        if len(target_ids) == 0:
            continue

        nll = F.cross_entropy(pred_logits, target_ids, reduction="sum")
        total_nll += nll.item()
        total_tok += len(target_ids)

        if end == n:
            break

    if total_tok == 0:
        return float("inf")

    avg_nll = total_nll / total_tok
    return math.exp(avg_nll)


# ─────────────────────────────────────────────
# Speed benchmark
# ─────────────────────────────────────────────

@dataclass
class SpeedResult:
    """Results from a single speed benchmark configuration."""
    batch_size: int
    seq_len: int
    tokens_per_second: float
    latency_ms: float          # per-forward-pass latency
    throughput_tokens: int     # total tokens processed
    device: str


@torch.no_grad()
def speed_benchmark(
    model: nn.Module,
    vocab_size: int = 256,
    batch_sizes: List[int] = (1, 4, 8),
    seq_lens: List[int] = (32, 128, 256),
    n_warmup: int = 3,
    n_trials: int = 10,
    device: str = "cpu",
) -> List[SpeedResult]:
    """
    Benchmark forward-pass throughput across batch sizes and sequence lengths.

    Args:
        model:       The model to benchmark.
        vocab_size:  Upper bound for dummy token id generation.
        batch_sizes: List of batch sizes to test.
        seq_lens:    List of sequence lengths to test.
        n_warmup:    Warm-up passes (not timed).
        n_trials:    Timed passes per configuration.
        device:      "cuda" or "cpu".

    Returns:
        List of SpeedResult, one per (batch_size, seq_len) combination.
    """
    model.eval()
    model.to(device)
    results = []

    for bs in batch_sizes:
        for sl in seq_lens:
            dummy = torch.randint(0, min(vocab_size, 64), (bs, sl), device=device)

            # Warm-up
            for _ in range(n_warmup):
                _ = model(dummy)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

            # Timed trials
            t0 = time.perf_counter()
            for _ in range(n_trials):
                _ = model(dummy)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            total_tokens = bs * sl * n_trials
            tps = total_tokens / elapsed
            lat = elapsed / n_trials * 1000  # ms per forward

            results.append(SpeedResult(
                batch_size=bs, seq_len=sl,
                tokens_per_second=round(tps, 1),
                latency_ms=round(lat, 2),
                throughput_tokens=total_tokens,
                device=device,
            ))
            print(f"  bs={bs:>3}  seq={sl:>4}  {tps:>10,.0f} tok/s  "
                  f"{lat:>7.2f} ms/fwd")

    return results


# ─────────────────────────────────────────────
# Memory profiling
# ─────────────────────────────────────────────

@dataclass
class MemoryResult:
    """Peak memory usage for a single forward pass."""
    device: str
    peak_mb: float
    allocated_mb: float
    param_mb: float
    n_params: int


def memory_profile(
    model: nn.Module,
    seq_len: int = 128,
    vocab_size: int = 256,
    device: str = "cpu",
) -> MemoryResult:
    """
    Measure peak memory during a single forward pass.

    For CUDA: uses torch.cuda.max_memory_allocated.
    For CPU:  estimates from parameter count (exact RSS profiling
              requires psutil, treated as optional).
    """
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    param_mb  = sum(p.numel() * p.element_size() for p in model.parameters()) / 1_048_576

    dummy = torch.randint(0, min(vocab_size, 64), (1, seq_len), device=device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        _ = model(dummy)
        torch.cuda.synchronize()
        peak_mb  = torch.cuda.max_memory_allocated(device) / 1_048_576
        alloc_mb = torch.cuda.memory_allocated(device) / 1_048_576
    else:
        # CPU: track parameter bytes as a proxy; full RSS needs psutil
        gc.collect()
        _ = model(dummy)
        peak_mb  = param_mb          # conservative lower bound
        alloc_mb = param_mb

    return MemoryResult(
        device=device,
        peak_mb=round(peak_mb, 2),
        allocated_mb=round(alloc_mb, 2),
        param_mb=round(param_mb, 2),
        n_params=n_params,
    )


# ─────────────────────────────────────────────
# Generation quality metrics
# ─────────────────────────────────────────────

@dataclass
class GenerationQuality:
    """Lightweight automatic quality metrics for generated text."""
    prompt: str
    output: str
    n_tokens: int
    unique_tokens: int
    type_token_ratio: float    # TTR: unique/total — diversity proxy
    repetition_score: float    # fraction of tokens that repeat a prior token
    avg_token_len: float


def score_generation(prompt: str, output: str, tokenizer) -> GenerationQuality:
    """
    Compute automatic quality metrics for one generated string.
    No external model needed — purely statistical.

    type_token_ratio close to 1.0 → high diversity (good)
    repetition_score close to 0.0 → low repetition (good)
    """
    ids = tokenizer.encode(output)
    n   = len(ids)
    if n == 0:
        return GenerationQuality(prompt, output, 0, 0, 0.0, 0.0, 0.0)

    unique  = len(set(ids))
    ttr     = unique / n

    # Repetition: what fraction of tokens appeared in the previous 20 tokens?
    rep_count = sum(1 for i, t in enumerate(ids) if t in ids[max(0,i-20):i])
    rep_score = rep_count / n

    avg_len = sum(len(tokenizer.decode([t])) for t in ids) / n

    return GenerationQuality(
        prompt=prompt, output=output, n_tokens=n,
        unique_tokens=unique, type_token_ratio=round(ttr, 4),
        repetition_score=round(rep_score, 4),
        avg_token_len=round(avg_len, 2),
    )


# ─────────────────────────────────────────────
# Checkpoint comparison
# ─────────────────────────────────────────────

def compare_checkpoints(
    results_a: Dict,
    results_b: Dict,
    label_a: str = "baseline",
    label_b: str = "new",
) -> str:
    """
    Produce a human-readable before-vs-after comparison table.

    Args:
        results_a / results_b: Dicts returned by run_eval_suite().

    Returns:
        Formatted comparison string.
    """
    lines = [
        f"{'Metric':<28} {label_a:>14} {label_b:>14} {'Δ':>10}",
        "─" * 70,
    ]

    def _row(label, key, fmt=".4f", lower_better=True):
        va = results_a.get(key, float("nan"))
        vb = results_b.get(key, float("nan"))
        delta = vb - va if isinstance(va, float) else "n/a"
        arrow = ""
        if isinstance(delta, float):
            better = delta < 0 if lower_better else delta > 0
            arrow = " ↑" if better else (" ↓" if not better else "")
        lines.append(
            f"{label:<28} {va:>14{fmt}} {vb:>14{fmt}} "
            f"{(str(round(delta, 4)) + arrow) if isinstance(delta, float) else delta:>10}"
        )

    _row("Perplexity",          "perplexity",          ".2f", lower_better=True)
    _row("Tokens / second",     "tokens_per_second",   ".1f", lower_better=False)
    _row("Peak memory (MB)",    "peak_mb",             ".1f", lower_better=True)
    _row("Type-token ratio",    "type_token_ratio",    ".4f", lower_better=False)
    _row("Repetition score",    "repetition_score",    ".4f", lower_better=True)

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Master evaluation suite
# ─────────────────────────────────────────────

def run_eval_suite(
    model: nn.Module,
    tokenizer,
    test_corpus: str,
    inference_pipeline=None,
    output_dir: str = "eval_reports",
    label: str = "model",
    device: str = "cpu",
    prompts: Optional[List[str]] = None,
) -> Dict:
    """
    Run the full evaluation suite and write a timestamped report to disk.

    Steps:
      1. Perplexity on test_corpus
      2. Speed benchmark (forward pass throughput)
      3. Memory profiling
      4. Generation quality (if inference_pipeline provided)

    Args:
        model:              Model to evaluate.
        tokenizer:          Tokenizer with encode()/decode().
        test_corpus:        Raw text used for perplexity calculation.
        inference_pipeline: Optional InferencePipeline for quality tests.
        output_dir:         Directory to write the report file.
        label:              Label embedded in the report filename.
        device:             Compute device.
        prompts:            Prompts to generate from for quality scoring.

    Returns:
        Dict with all metric values (also written to JSON report).
    """
    import datetime
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*56}")
    print(f"  Evaluation Suite — {label}  [{ts}]")
    print(f"{'='*56}\n")

    results: Dict = {"label": label, "timestamp": ts, "device": device}

    # ── 1. Perplexity ──────────────────────────────
    print("[1/4] Perplexity...")
    token_ids = tokenizer.encode(test_corpus)
    ppl = perplexity(model, token_ids, context_len=128, stride=64, device=device)
    results["perplexity"]   = round(ppl, 4)
    results["test_tokens"]  = len(token_ids)
    print(f"      Perplexity = {ppl:.4f}  ({len(token_ids)} tokens)")

    # ── 2. Speed ───────────────────────────────────
    print("\n[2/4] Speed benchmark...")
    vocab = getattr(tokenizer, "vocab_size", 256)
    speed_results = speed_benchmark(
        model, vocab_size=vocab,
        batch_sizes=[1, 4], seq_lens=[32, 128],
        n_trials=5, device=device
    )
    best_tps = max(r.tokens_per_second for r in speed_results)
    results["tokens_per_second"] = best_tps
    results["speed_details"]     = [asdict(r) for r in speed_results]

    # ── 3. Memory ──────────────────────────────────
    print("\n[3/4] Memory profile...")
    mem = memory_profile(model, seq_len=128, vocab_size=vocab, device=device)
    results["peak_mb"]    = mem.peak_mb
    results["param_mb"]   = mem.param_mb
    results["n_params"]   = mem.n_params
    print(f"      Params: {mem.n_params:,}  ({mem.param_mb:.1f} MB params, "
          f"{mem.peak_mb:.1f} MB peak)")

    # ── 4. Generation quality ──────────────────────
    print("\n[4/4] Generation quality...")
    if inference_pipeline is not None:
        test_prompts = prompts or [
            "The most important thing about language is",
            "Once upon a time in a world where",
            "The key insight is that",
        ]
        quality_rows = []
        for prompt in test_prompts:
            out = inference_pipeline.generate(
                prompt, max_new_tokens=60, strategy="top_p",
                temperature=0.8, top_p=0.95
            )
            q = score_generation(prompt, out, tokenizer)
            quality_rows.append(asdict(q))
            print(f"      TTR={q.type_token_ratio:.3f}  "
                  f"Rep={q.repetition_score:.3f}  "
                  f'  "{prompt[:30]}..."')

        avg_ttr = sum(r["type_token_ratio"] for r in quality_rows) / len(quality_rows)
        avg_rep = sum(r["repetition_score"] for r in quality_rows) / len(quality_rows)
        results["type_token_ratio"]  = round(avg_ttr, 4)
        results["repetition_score"]  = round(avg_rep, 4)
        results["generation_samples"] = quality_rows
    else:
        print("      (skipped — no inference_pipeline provided)")
        results["type_token_ratio"] = float("nan")
        results["repetition_score"] = float("nan")

    # ── Write report ───────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(output_dir) / f"eval_{label}_{ts}.json"

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*56}")
    print(f"  Perplexity       : {results['perplexity']:.4f}")
    print(f"  Tokens/second    : {results['tokens_per_second']:,.0f}")
    print(f"  Peak memory      : {results['peak_mb']:.1f} MB")
    print(f"  Parameters       : {results['n_params']:,}")
    if "type_token_ratio" in results and not math.isnan(results["type_token_ratio"]):
        print(f"  Type-token ratio : {results['type_token_ratio']:.4f}")
        print(f"  Repetition score : {results['repetition_score']:.4f}")
    print(f"{'='*56}")
    print(f"  Report → {report_path}")
    print(f"{'='*56}\n")

    return results


# ─────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch.nn as nn
    from typing import List

    class _TinyLM(nn.Module):
        VOCAB = 64
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(self.VOCAB, 32)
            self.rnn  = nn.GRU(32, 64, batch_first=True)
            self.proj = nn.Linear(64, self.VOCAB)
        def forward(self, x):
            h = self.emb(x); h, _ = self.rnn(h); return self.proj(h)

    class _TinyTok:
        vocab_size = 64
        def encode(self, t):  return [b % 64 for b in t.encode()]
        def decode(self, ids): return bytes([i+32 for i in ids]).decode(errors="replace")

    torch.manual_seed(7)
    model = _TinyLM()
    tok   = _TinyTok()

    corpus = "the quick brown fox jumps over the lazy dog " * 40

    ppl = perplexity(model, tok.encode(corpus), context_len=32, stride=16, device="cpu")
    print(f"Perplexity = {ppl:.2f}  (random ≈ {_TinyLM.VOCAB})")
    assert ppl > 1, "Perplexity sanity check"

    sr = speed_benchmark(model, vocab_size=64, batch_sizes=[1], seq_lens=[32],
                         n_trials=5, device="cpu")
    assert sr[0].tokens_per_second > 0
    print(f"Speed: {sr[0].tokens_per_second:,.0f} tok/s ✓")

    mem = memory_profile(model, seq_len=32, vocab_size=64, device="cpu")
    print(f"Memory: {mem.peak_mb:.2f} MB  ({mem.n_params:,} params) ✓")

    q = score_generation("hello", "world is a place where", tok)
    assert 0 <= q.type_token_ratio <= 1
    print(f"Generation quality: TTR={q.type_token_ratio:.3f} ✓")

    results = run_eval_suite(model, tok, corpus, output_dir="/tmp/eval_test",
                             label="smoke", device="cpu")
    assert "perplexity" in results
    print("\nStep 33 — evaluation & benchmarking ✓")
