from __future__ import annotations

import torch

from .model import AnRaCore
from .tokenizer import V4Tokenizer


@torch.inference_mode()
def generate(
    model: AnRaCore,
    tokenizer: V4Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 0.92,
    seed: int = 0,
) -> str:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt)
    prompt_ids = prompt_ids[-(model.config.block_size - 1):]
    ids = torch.tensor([[tokenizer.bos_token_id, *prompt_ids]], dtype=torch.long, device=device)
    generated: list[int] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(max_new_tokens):
        logits = model(ids[:, -model.config.block_size:])[:, -1, :]
        if temperature <= 0:
            next_id = int(logits.argmax(dim=-1).item())
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
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
        ids = torch.cat((ids, torch.tensor([[next_id]], device=device)), dim=1)
    return tokenizer.decode(generated)
