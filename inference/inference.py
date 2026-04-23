"""
45G / Step 31 — Full Inference Pipeline
=========================================
Single clean entry point: prompt → output.

Features:
  - All sampling strategies: greedy, temperature, top_k, top_p, nucleus
  - Streaming token-by-token output (generator interface)
  - Automatic tokenisation / detokenisation
  - Batch inference for multiple prompts
  - Stop-token lists, max-length hard cap
  - Graceful OOM recovery (retry on CPU if CUDA OOM)
  - Token/second benchmarking baked in
  - Works with any model whose forward() returns logits
"""

from __future__ import annotations

import time
import sys
from dataclasses import dataclass, field
from typing import Generator, Iterable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from greedy import greedy_step
from sampling import sample_token, sampling_decode
from model_io import build_kv_cache


# ─────────────────────────────────────────────
# Generation config — one dataclass controls all
# ─────────────────────────────────────────────

@dataclass
class GenerationConfig:
    """
    All knobs for a single inference call.

    strategy:         "greedy" | "temperature" | "top_k" | "top_p"
    temperature:      Logit sharpness scale (1.0 = unchanged).
    top_k:            Top-k filter width. 0 = disabled.
    top_p:            Nucleus probability mass. 1.0 = disabled.
    max_new_tokens:   Hard cap on generated tokens.
    stop_tokens:      List of strings; generation halts when any appears.
    stream:           If True, print tokens as they are generated.
    repetition_penalty: > 1.0 penalises previously seen tokens.
    """
    strategy: str = "top_p"
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 200
    stop_tokens: List[str] = field(default_factory=list)
    stream: bool = False
    repetition_penalty: float = 1.0
    use_turboquant: bool = True


# ─────────────────────────────────────────────
# Repetition penalty
# ─────────────────────────────────────────────

def _apply_rep_penalty(
    logits: torch.Tensor,
    prev_ids: List[int],
    penalty: float,
) -> torch.Tensor:
    """
    Divide logits for previously generated tokens by `penalty`.
    penalty=1.0 is a no-op; >1.0 discourages repetition.
    """
    if penalty == 1.0 or not prev_ids:
        return logits
    for tid in set(prev_ids):
        if 0 <= tid < logits.size(-1):
            logits[tid] /= penalty
    return logits


# ─────────────────────────────────────────────
# Core single-sequence generator (streaming)
# ─────────────────────────────────────────────

def _generate_tokens(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    device: str,
) -> Generator[int, None, None]:
    """
    Inner generator: yields one integer token id per step.
    Caller is responsible for decoding and stop-condition checks.
    """
    model.eval()
    ids = input_ids.to(device)
    generated: List[int] = []

    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            out = model(ids)
            if isinstance(out, tuple):
                out = out[0]

            # Accept (batch, seq, vocab) or (seq, vocab)
            logits = (out[0, -1, :] if out.dim() == 3 else out[-1, :]).float()

            # Repetition penalty
            logits = _apply_rep_penalty(logits, generated, config.repetition_penalty)

            # Decode strategy
            if config.strategy == "greedy":
                next_id = greedy_step(logits)
            else:
                # All non-greedy paths share the unified sampler
                tk = config.top_k if config.strategy in ("top_k", "top_p") else 0
                tp = config.top_p if config.strategy in ("top_p",)          else 1.0
                t  = config.temperature if config.strategy != "greedy"      else 1.0
                next_id = sample_token(logits, temperature=t, top_k=tk, top_p=tp)

            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
            yield next_id


# ─────────────────────────────────────────────
# Public inference pipeline
# ─────────────────────────────────────────────

class InferencePipeline:
    """
    Production inference pipeline — wraps model + tokenizer into a single
    generate() call that handles everything.

    Example:
        pipe = InferencePipeline(model, tokenizer, device="cuda")
        text = pipe.generate("The future of AI is", max_new_tokens=150)

    Streaming example:
        for chunk in pipe.stream("Tell me a story"):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[str] = None,
    ):
        self.model     = model
        self.tok       = tokenizer
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.kv_cache = build_kv_cache(use_turboquant=True)

    # ── single prompt ────────────────────────

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """
        Encode prompt, decode to max_new_tokens, return text.

        Keyword args are forwarded to GenerationConfig so you can do:
            pipe.generate("Hi", temperature=0.9, top_p=0.95, max_new_tokens=100)
        """
        cfg = config or GenerationConfig(**{k: v for k, v in kwargs.items()
                                            if k in GenerationConfig.__dataclass_fields__})

        try:
            return self._run_generate(prompt, cfg)
        except torch.cuda.OutOfMemoryError:
            print("[inference] CUDA OOM — retrying on CPU", file=sys.stderr)
            self.model.to("cpu")
            self.device = "cpu"
            return self._run_generate(prompt, cfg)

    def _run_generate(self, prompt: str, cfg: GenerationConfig) -> str:
        encoded = self.tok.encode(prompt)
        if len(encoded) == 0:
            encoded = [0]          # pad empty prompt with a single BOS/null token
        input_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)

        t0 = time.perf_counter()
        output_pieces: List[str] = []
        full_generated: List[int] = []

        for token_id in _generate_tokens(self.model, input_ids, cfg, self.device):
            full_generated.append(token_id)
            piece = self.tok.decode([token_id])
            output_pieces.append(piece)

            if cfg.stream:
                print(piece, end="", flush=True)

            # Check stop tokens against the running output
            current = "".join(output_pieces)
            if any(s in current for s in cfg.stop_tokens):
                break

        if cfg.stream:
            print()  # newline after streamed output

        elapsed = time.perf_counter() - t0
        n_tok = len(full_generated)
        tps = n_tok / elapsed if elapsed > 0 else 0.0
        print(f"[inference] {n_tok} tokens | {elapsed:.2f}s | {tps:.1f} tok/s | strategy={cfg.strategy}")

        return "".join(output_pieces)

    # ── streaming interface ───────────────────

    def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Token-by-token generator; yields decoded string pieces.

        for piece in pipe.stream("Once"):
            print(piece, end="", flush=True)
        """
        cfg = config or GenerationConfig(**{k: v for k, v in kwargs.items()
                                            if k in GenerationConfig.__dataclass_fields__})
        cfg.stream = False  # prevent double-print inside _generate_tokens

        input_ids = torch.tensor(
            [self.tok.encode(prompt)], dtype=torch.long, device=self.device
        )
        collected: List[str] = []

        for token_id in _generate_tokens(self.model, input_ids, cfg, self.device):
            piece = self.tok.decode([token_id])
            collected.append(piece)
            yield piece
            if any(s in "".join(collected) for s in cfg.stop_tokens):
                break

    # ── batch inference ───────────────────────

    def batch_generate(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> List[str]:
        """
        Run inference on multiple prompts sequentially.
        Returns one output string per prompt.

        For true parallel batching the caller should pad/pack ids and
        forward a (B, seq) tensor — left as a model-specific concern since
        padding token is tokenizer-dependent.
        """
        cfg = config or GenerationConfig(**{k: v for k, v in kwargs.items()
                                            if k in GenerationConfig.__dataclass_fields__})
        results = []
        t0 = time.perf_counter()

        for i, prompt in enumerate(prompts):
            print(f"[inference] batch {i+1}/{len(prompts)}")
            results.append(self._run_generate(prompt, cfg))

        total = time.perf_counter() - t0
        print(f"[inference] batch complete — {len(prompts)} prompts in {total:.2f}s")
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
            h = self.emb(x)
            h, _ = self.rnn(h)
            return self.proj(h)

    class _TinyTok:
        def encode(self, t: str) -> List[int]:  return [b % 64 for b in t.encode()]
        def decode(self, ids: List[int]) -> str: return bytes([i+32 for i in ids]).decode(errors="replace")

    torch.manual_seed(0)
    pipe = InferencePipeline(_TinyLM(), _TinyTok(), device="cpu")

    # Greedy
    cfg_greedy = GenerationConfig(strategy="greedy", max_new_tokens=20)
    out = pipe.generate("Hello", config=cfg_greedy)
    print(f"Greedy:      {repr(out)}")

    # Temperature
    cfg_temp = GenerationConfig(strategy="temperature", temperature=0.7, max_new_tokens=20)
    out = pipe.generate("Hello", config=cfg_temp)
    print(f"Temperature: {repr(out)}")

    # Top-k
    cfg_topk = GenerationConfig(strategy="top_k", top_k=10, temperature=0.8, max_new_tokens=20)
    out = pipe.generate("Hello", config=cfg_topk)
    print(f"Top-k:       {repr(out)}")

    # Top-p
    cfg_topp = GenerationConfig(strategy="top_p", top_p=0.9, temperature=0.8, max_new_tokens=20)
    out = pipe.generate("Hello", config=cfg_topp)
    print(f"Top-p:       {repr(out)}")

    # Streaming
    print("Stream: ", end="")
    for piece in pipe.stream("Once", temperature=1.0, max_new_tokens=15):
        print(piece, end="", flush=True)
    print()

    # Batch
    results = pipe.batch_generate(["A", "B", "C"], max_new_tokens=10)
    assert len(results) == 3
    print(f"Batch: {len(results)} results ✓")

    print("\nStep 31 — full inference pipeline ✓")
