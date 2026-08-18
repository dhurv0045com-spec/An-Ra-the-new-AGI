import torch
import torch.nn.functional as F

from anra_core.config import CANONICAL_CONFIG
from anra_core.model import AnRaCore


def test_differentiable_forward_and_autograd() -> None:
    torch.manual_seed(505)
    model = AnRaCore(CANONICAL_CONFIG).train()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, CANONICAL_CONFIG.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, CANONICAL_CONFIG.vocab_size, (batch_size, seq_len))

    # 1. Forward Pass
    logits = model(input_ids)
    assert logits.requires_grad

    # 2. External Objective Formulation
    loss = F.cross_entropy(logits.view(-1, CANONICAL_CONFIG.vocab_size), target_ids.view(-1))
    assert not torch.isnan(loss)

    # 3. Backward Pass
    loss.backward()

    # 4. Verify Gradients on Parameters
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
            grad_count += 1

    assert grad_count == 164

    # 5. Verify Tied LM Head / Token Embedding Gradient Coherence
    assert model.lm_head.weight.grad is not None
    assert model.token_embedding_table.weight.grad is not None

    # 6. Execute One Optimizer Step Externally
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    optimizer.zero_grad()
