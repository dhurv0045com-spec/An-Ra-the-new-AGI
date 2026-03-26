"""
45G / Step 27 — Greedy Decoding
================================
Always picks the single highest-probability next token.
Deterministic, fast, zero hyperparameters.
Serves as the baseline against which every other
sampling strategy is measured.

Connect point: wraps any model that exposes
    logits = model(token_ids)          # (seq,) or (1, seq)
and a tokenizer with encode() / decode().
"""

from __future__ import annotations

import time
from typing import List, Optional

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Core greedy step
# ─────────────────────────────────────────────

def greedy_step(logits: torch.Tensor) -> int:
    """
    Select the single most probable token from a logit vector.

    Args:
        logits: raw (un-normalised) scores, shape (vocab_size,)

    Returns:
        Integer token id with the highest logit.
    """
    return int(torch.argmax(logits, dim=-1).item())


# ─────────────────────────────────────────────
# Full greedy decode loop
# ─────────────────────────────────────────────

def greedy_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 200,
    eos_token_id: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Autoregressive greedy decode: extend input_ids one token at a time
    by always picking argmax over the vocabulary.

    Args:
        model:          Any nn.Module whose forward returns logits
                        of shape (batch, seq, vocab) or (seq, vocab).
        input_ids:      Prompt token ids, shape (1, seq) or (seq,).
        max_new_tokens: Hard cap on tokens to generate.
        eos_token_id:   Stop early when this token is produced.
        device:         "cuda" or "cpu".

    Returns:
        Full token id tensor including the original prompt,
        shape (1, prompt_len + generated_len).
    """
    model.eval()

    # Normalise to (1, seq)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward — accept (batch, seq, vocab) or (seq, vocab)
            out = model(ids)
            if isinstance(out, tuple):
                out = out[0]                        # some models return (logits, cache, ...)

            logits = out[0, -1, :] if out.dim() == 3 else out[-1, :]
            next_id = greedy_step(logits)

            next_tensor = torch.tensor([[next_id]], device=device)
            ids = torch.cat([ids, next_tensor], dim=1)

            if eos_token_id is not None and next_id == eos_token_id:
                break

    return ids


# ─────────────────────────────────────────────
# High-level convenience wrapper
# ─────────────────────────────────────────────

class GreedyDecoder:
    """
    Stateless greedy decoder. One object, reusable across prompts.

    Usage:
        decoder = GreedyDecoder(model, tokenizer, device="cuda")
        text = decoder.generate("Once upon a time", max_new_tokens=100)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: str = "cpu",
        eos_token: Optional[str] = None,
    ):
        self.model = model
        self.tok = tokenizer
        self.device = device
        # Resolve eos token id once at construction time
        self.eos_id: Optional[int] = None
        if eos_token and hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(eos_token)
            self.eos_id = ids[-1] if ids else None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
    ) -> str:
        """
        Encode prompt → greedy decode → decode back to string.

        Returns the generated continuation only (prompt stripped).
        """
        t0 = time.perf_counter()

        input_ids = torch.tensor(
            [self.tok.encode(prompt)], dtype=torch.long, device=self.device
        )
        prompt_len = input_ids.shape[1]

        output_ids = greedy_decode(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.eos_id,
            device=self.device,
        )

        # Decode only the new tokens
        new_ids: List[int] = output_ids[0, prompt_len:].tolist()
        elapsed = time.perf_counter() - t0
        tps = len(new_ids) / elapsed if elapsed > 0 else 0.0

        result = self.tok.decode(new_ids)
        print(f"[greedy] {len(new_ids)} tokens in {elapsed:.2f}s  ({tps:.1f} tok/s)")
        return result


# ─────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch.nn as nn

    class _TinyLM(nn.Module):
        """Minimal LM for offline testing — random weights, fixed vocab 64."""
        VOCAB = 64

        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(self.VOCAB, 32)
            self.rnn = nn.GRU(32, 64, batch_first=True)
            self.proj = nn.Linear(64, self.VOCAB)

        def forward(self, x):
            h = self.emb(x)
            h, _ = self.rnn(h)
            return self.proj(h)   # (batch, seq, vocab)

    class _TinyTok:
        """Dead-simple byte-level tokenizer for the smoke-test."""
        def encode(self, text: str) -> List[int]:
            return [b % 64 for b in text.encode()]

        def decode(self, ids: List[int]) -> str:
            return bytes([i + 32 for i in ids]).decode(errors="replace")

    model = _TinyLM()
    tok   = _TinyTok()

    decoder = GreedyDecoder(model, tok, device="cpu")
    out = decoder.generate("Hello", max_new_tokens=40)
    print(f"Output: {repr(out)}")

    # Verify greedy_step always picks argmax
    logits = torch.tensor([0.1, 5.0, 0.3, 2.1])
    assert greedy_step(logits) == 1, "greedy_step failed"
    print("greedy_step ✓  |  GreedyDecoder ✓  |  Step 27 complete")
