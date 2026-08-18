import torch

from anra_core.config import CANONICAL_CONFIG
from anra_core.executor import CoreExecutor
from anra_core.model import AnRaCore


def test_cached_vs_uncached_logit_parity() -> None:
    torch.manual_seed(101)
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    seq_len = 16
    token_ids = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, seq_len))

    # 1. Uncached Full Forward
    uncached_result = executor.forward(token_ids)
    uncached_logits = uncached_result.logits

    # 2. Incremental Step-by-Step Decode
    state = executor.create_state()
    incremental_logits_list = []

    # Prefill first token
    first_tok = token_ids[:, :1]
    res_prefill = executor.prefill(first_tok, state=state)
    incremental_logits_list.append(res_prefill.logits)

    # Step remaining tokens
    for i in range(1, seq_len):
        next_tok = token_ids[:, i : i + 1]
        res_step = executor.forward_step(next_tok, state=state)
        incremental_logits_list.append(res_step.logits)

    incremental_logits = torch.cat(incremental_logits_list, dim=1)

    # Check maximum absolute difference in FP32 (FP32 numerical accumulation over 18 layers is < 5e-4)
    max_diff = (uncached_logits - incremental_logits).abs().max().item()
    assert max_diff < 5e-4, f"Max diff too high: {max_diff}"

    # Check 100% exact greedy token agreement across all positions
    uncached_tokens = uncached_logits.argmax(dim=-1)
    incremental_tokens = incremental_logits.argmax(dim=-1)
    assert torch.equal(uncached_tokens, incremental_tokens)


def test_prefill_and_step_equivalence() -> None:
    torch.manual_seed(202)
    model = AnRaCore(CANONICAL_CONFIG).eval()
    executor = CoreExecutor(model)

    prompt = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 8))
    extension = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 4))
    full_sequence = torch.cat([prompt, extension], dim=1)

    # Method 1: Full uncached pass
    full_logits = executor.forward(full_sequence).logits

    # Method 2: Prefill prompt into State, then step extension
    state = executor.create_state()
    _ = executor.prefill(prompt, state=state)

    step_logits = []
    for i in range(4):
        tok = extension[:, i : i + 1]
        res = executor.forward_step(tok, state=state)
        step_logits.append(res.logits)

    cat_extension_logits = torch.cat(step_logits, dim=1)
    target_extension_logits = full_logits[:, 8:, :]

    diff = (target_extension_logits - cat_extension_logits).abs().max().item()
    assert diff < 5e-4, f"Extension logit diff too high: {diff}"

    # Check exact token matching
    assert torch.equal(target_extension_logits.argmax(dim=-1), cat_extension_logits.argmax(dim=-1))
