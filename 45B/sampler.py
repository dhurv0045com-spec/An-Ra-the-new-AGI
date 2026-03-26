"""
inference/sampler.py — Steps 26–29
Greedy decoding, temperature sampling, top-k, top-p (nucleus) sampling.
Full streaming inference loop with KV cache.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Iterator, Tuple
import time


# ─────────────────────────────────────────────
# STEP 26 — Greedy decoding
# ─────────────────────────────────────────────

def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Take the argmax — most likely next token. (B, V) → (B,)"""
    return logits.argmax(dim=-1)


# ─────────────────────────────────────────────
# STEP 27 — Temperature sampling
# ─────────────────────────────────────────────

def temperature_sample(logits: torch.Tensor,
                        temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from softmax(logits / T).
    T → 0: greedy (sharp)
    T = 1: standard sampling
    T > 1: more uniform / creative
    """
    if temperature <= 0:
        return greedy_sample(logits)
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ─────────────────────────────────────────────
# STEP 28 — Top-k and top-p (nucleus) sampling
# ─────────────────────────────────────────────

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits below the k-th largest value."""
    if k <= 0:
        return logits
    top_k_vals = torch.topk(logits, k, dim=-1).values
    threshold = top_k_vals[..., -1, None]  # k-th largest
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: keep only the smallest set of tokens whose
    cumulative probability exceeds p.
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Shift right so the token that pushes cumprob over p is kept
    cumulative_probs = torch.cat([
        torch.zeros_like(cumulative_probs[..., :1]),
        cumulative_probs[..., :-1]
    ], dim=-1)

    remove = cumulative_probs >= p
    sorted_logits[remove] = float("-inf")

    # Restore original ordering
    return sorted_logits.scatter(-1, sorted_idx, sorted_logits)


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generated_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Unified sampling function.
    Applies: repetition penalty → top_k → top_p → temperature → sample.
    (B, V) logits → (B,) next token ids.
    """
    # Repetition penalty — discourage repeating recent tokens
    if repetition_penalty != 1.0 and generated_ids is not None:
        for i in range(logits.size(0)):
            for tok_id in generated_ids[i].tolist():
                if logits[i, tok_id] > 0:
                    logits[i, tok_id] /= repetition_penalty
                else:
                    logits[i, tok_id] *= repetition_penalty

    # Pure greedy — skip all filtering
    if temperature <= 0:
        return greedy_sample(logits)

    if top_k > 0:
        logits = top_k_filter(logits, top_k)

    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)

    return temperature_sample(logits, temperature)


# ─────────────────────────────────────────────
# STEP 29 — Full inference loop
# ─────────────────────────────────────────────

class GenerationConfig:
    def __init__(self,
                 max_new_tokens: int = 256,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.1,
                 eos_token_id: Optional[int] = 2,
                 pad_token_id: int = 0,
                 do_sample: bool = True):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.do_sample = do_sample


class Generator:
    """
    Full autoregressive generation with KV cache.
    Supports streaming (yield token by token) and batch generation.
    """

    def __init__(self, model: torch.nn.Module,
                 tokenizer=None,
                 device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        config: Optional[GenerationConfig] = None,
    ) -> torch.Tensor:
        """
        Full generation — returns complete output token ids.
        input_ids: (B, T_prompt)
        Returns:   (B, T_prompt + max_new_tokens)
        """
        cfg = config or GenerationConfig()
        input_ids = input_ids.to(self.device)
        B, T_prompt = input_ids.shape

        generated = input_ids.clone()
        n_layers = self.model.n_layers
        kv_caches = [None] * n_layers

        # Process prompt (can be batched efficiently)
        _, kv_caches = self.model(input_ids, kv_caches=[None]*n_layers)

        # Autoregressive loop
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for step in range(cfg.max_new_tokens):
            # Only feed the last generated token (KV cache has the rest)
            last_tok = generated[:, -1:]
            offset = generated.shape[1] - 1

            logits, kv_caches = self.model(
                last_tok,
                kv_caches=kv_caches,
                cache_offset=offset,
            )

            next_logits = logits[:, -1, :]  # (B, V)

            if cfg.do_sample:
                next_ids = sample_token(
                    next_logits.clone(),
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                    repetition_penalty=cfg.repetition_penalty,
                    generated_ids=generated[:, -50:],  # check last 50 toks
                )
            else:
                next_ids = greedy_sample(next_logits)

            # Force EOS for finished sequences
            if cfg.eos_token_id is not None:
                next_ids[finished] = cfg.eos_token_id

            generated = torch.cat([generated, next_ids.unsqueeze(-1)], dim=-1)

            # Update finished flags
            if cfg.eos_token_id is not None:
                finished |= (next_ids == cfg.eos_token_id)
            if finished.all():
                break

        return generated

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        decode: bool = True,
    ) -> Iterator[str]:
        """
        Streaming generation — yields one token (or decoded text) at a time.
        For interactive / API use.
        """
        assert self.tokenizer is not None, "Need tokenizer for text streaming"
        cfg = config or GenerationConfig()

        ids = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        _, kv_caches = self.model(input_ids, kv_caches=[None]*self.model.n_layers)
        generated_ids = list(ids)

        for _ in range(cfg.max_new_tokens):
            last = torch.tensor([[generated_ids[-1]]], dtype=torch.long,
                                  device=self.device)
            logits, kv_caches = self.model(
                last, kv_caches=kv_caches,
                cache_offset=len(generated_ids) - 1
            )
            next_logits = logits[:, -1, :]

            if cfg.do_sample:
                recent = torch.tensor([generated_ids[-50:]], device=self.device)
                next_id = sample_token(
                    next_logits.clone(), cfg.temperature,
                    cfg.top_k, cfg.top_p,
                    cfg.repetition_penalty, recent
                ).item()
            else:
                next_id = greedy_sample(next_logits).item()

            generated_ids.append(next_id)

            if decode:
                yield self.tokenizer.decode([next_id], skip_special_tokens=False)
            else:
                yield next_id

            if cfg.eos_token_id is not None and next_id == cfg.eos_token_id:
                break

    def generate_text(self, prompt: str,
                      config: Optional[GenerationConfig] = None) -> str:
        """Convenience wrapper: text in → text out."""
        assert self.tokenizer is not None
        cfg = config or GenerationConfig()
        ids = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long)
        output_ids = self.generate(input_ids, cfg)
        new_ids = output_ids[0, len(ids):].tolist()
        return self.tokenizer.decode(new_ids)

    def benchmark(self, seq_len: int = 128, n_tokens: int = 100) -> dict:
        """Measure tokens/sec for the current model."""
        cfg = GenerationConfig(max_new_tokens=n_tokens, do_sample=False)
        dummy = torch.randint(0, self.model.vocab_size, (1, seq_len),
                               device=self.device)
        # Warmup
        self.generate(dummy, GenerationConfig(max_new_tokens=10, do_sample=False))

        t0 = time.time()
        self.generate(dummy, cfg)
        dt = time.time() - t0

        return {
            "tokens_generated": n_tokens,
            "time_seconds": dt,
            "tokens_per_second": n_tokens / dt,
            "ms_per_token": dt / n_tokens * 1000,
        }


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Steps 26–29: Sampling functions check")
    V = 1000
    B = 2
    logits = torch.randn(B, V)

    # Greedy
    g = greedy_sample(logits)
    assert g.shape == (B,), f"Expected ({B},), got {g.shape}"
    print(f"  Greedy:     {g.tolist()}")

    # Temperature
    t = temperature_sample(logits, temperature=0.7)
    assert t.shape == (B,)
    print(f"  Temp=0.7:   {t.tolist()}")

    # Top-k
    logits_k = top_k_filter(logits, k=10)
    n_valid = (logits_k[0] != float("-inf")).sum().item()
    assert n_valid == 10, f"Expected 10 valid logits, got {n_valid}"
    print(f"  Top-k=10:   {n_valid} valid logits  ✓")

    # Top-p
    logits_p = top_p_filter(logits, p=0.9)
    n_valid_p = (logits_p[0] != float("-inf")).sum().item()
    print(f"  Top-p=0.9:  {n_valid_p} valid logits  ✓")

    # Combined
    s = sample_token(logits.clone(), temperature=0.8, top_k=50, top_p=0.9)
    assert s.shape == (B,)
    print(f"  Combined:   {s.tolist()}")

    print("✓ Steps 26–29 verified")
