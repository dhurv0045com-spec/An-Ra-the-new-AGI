"""Reference autoregressive generation utility using the Core Executor.

Decouples token sampling and stopping conditions from neural execution.
Uses stateful prefill and incremental decode to avoid O(N^2) prefix recomputations.
"""

from __future__ import annotations

import torch

from .errors import ContextOverflowError
from .executor import CoreExecutor
from .model import AnRaCore
from .tokenizer import V4Tokenizer


@torch.inference_mode()
def generate(
    model_or_executor: AnRaCore | CoreExecutor,
    tokenizer: V4Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 0.92,
    seed: int = 0,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 4,
) -> str:
    """Generate next tokens from prompt using stateful incremental decoding.

    Degeneration controls (applied to greedy and sampled paths alike):
      - ``repetition_penalty``: logits of already-generated tokens are divided
        by the penalty (>1 discourages repeats; CTRL-style, count-free).
      - ``no_repeat_ngram_size``: a token that would complete an n-gram already
        present in the output is banned outright.
    Both can be disabled (penalty 1.0 / ngram 0) for exact legacy behavior.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be >= 1.0")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be >= 0")

    if isinstance(model_or_executor, CoreExecutor):
        executor = model_or_executor
        if executor.tokenizer is not None and executor.tokenizer is not tokenizer:
            raise ValueError(
                "generate() tokenizer must be the executor's bound tokenizer; "
                "a foreign tokenizer breaks the representation contract"
            )
    else:
        executor = CoreExecutor(model_or_executor, tokenizer=tokenizer)

    device = executor.device
    prompt_ids = tokenizer.encode(prompt)
    required = 1 + len(prompt_ids) + max_new_tokens
    if required > executor.model.config.block_size:
        raise ContextOverflowError(
            "Prompt plus requested generation exceeds the Core context capacity",
            details={
                "prompt_tokens_with_bos": 1 + len(prompt_ids),
                "max_new_tokens": max_new_tokens,
                "capacity": executor.model.config.block_size,
            },
        )
    ids = torch.tensor([[tokenizer.bos_token_id, *prompt_ids]], dtype=torch.long, device=device)

    # The Executor owns storage; generation knows this bounded request's exact
    # need.  Avoid reserving the full V4 context for a short one-shot reply.
    state = executor.create_state(capacity=required)
    try:
        pred = executor.prefill(ids, state=state)
        logits = pred.logits[:, -1, :]
        generated: list[int] = []
        generator = torch.Generator(device=device).manual_seed(seed)

        def _ban_repeated_ngram(raw: torch.Tensor) -> torch.Tensor:
            """Set -inf on tokens that would complete a seen n-gram."""
            if no_repeat_ngram_size <= 0 or len(generated) < no_repeat_ngram_size - 1:
                return raw
            prefix = tuple(generated[-(no_repeat_ngram_size - 1) :])
            banned = set()
            for i in range(len(generated) - no_repeat_ngram_size + 1):
                if tuple(generated[i : i + no_repeat_ngram_size - 1]) == prefix:
                    banned.add(generated[i + no_repeat_ngram_size - 1])
            if banned:
                raw = raw.clone()
                for token_id in banned:
                    raw[:, token_id] = float("-inf")
            return raw

        for _ in range(max_new_tokens):
            working = logits
            if repetition_penalty > 1.0 and generated:
                penalty_indices = torch.tensor(
                    [generated], dtype=torch.long, device=device
                )
                working = working.clone()
                # Positive logits divide; negative logits multiply (CTRL rule).
                gathered = working.gather(1, penalty_indices)
                penalized = torch.where(
                    gathered > 0,
                    gathered / repetition_penalty,
                    gathered * repetition_penalty,
                )
                working.scatter_(1, penalty_indices, penalized)
            working = _ban_repeated_ngram(working)

            if temperature <= 0:
                next_id = int(working.argmax(dim=-1).item())
            else:
                probabilities = torch.softmax(working / temperature, dim=-1)
                sorted_probs, sorted_ids = probabilities.sort(descending=True)
                cumulative = sorted_probs.cumsum(dim=-1)
                remove = cumulative - sorted_probs > top_p
                sorted_probs = sorted_probs.masked_fill(remove, 0)
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                choice = torch.multinomial(sorted_probs, 1, generator=generator)
                next_id = int(sorted_ids.gather(-1, choice).item())

            if next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)
            if len(generated) < max_new_tokens:
                # Only advance the cache when another token will be sampled;
                # the final token's forward pass would be discarded work.
                pred = executor.forward_step(next_id, state=state)
                logits = pred.logits[:, -1, :]
        return tokenizer.decode(generated)
    finally:
        if not state.is_released:
            executor.release_state(state)
