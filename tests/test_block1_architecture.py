import torch

from anra_brain import CausalTransformerV2
from training.v2_config import V2_MODEL


def test_block1_architecture(tmp_path):
    assert V2_MODEL.vocab_size == 4096
    assert V2_MODEL.pad_token_id == 0
    assert V2_MODEL.n_embd == 512
    assert V2_MODEL.n_head == 8
    assert V2_MODEL.n_kv_head == 2
    assert V2_MODEL.n_layer == 8
    assert V2_MODEL.block_size == 512

    model = CausalTransformerV2(vocab_size=4096, n_embd=512, n_head=8, n_kv_head=2, n_layer=8, block_size=512, mod_layers={2, 4, 6})
    x = torch.randint(0, 4096, (2, 32))
    logits, loss = model(x, x)
    assert logits.shape == (2, 32, 4096)
    assert not torch.isnan(logits).any()
    assert model.blocks[0].attn.k_proj.weight.shape == (128, 512)
    loss.backward()
    assert model.blocks[2].attn.q_proj.weight.grad is not None
    assert set(model.mod_routers.keys()) == {"2", "4", "6"}
    assert model.blocks[0].attn.rope._attn_scale > 1.0

    params = sum(p.numel() for p in model.parameters())
    print(f"params={params}")
    assert 28_000_000 <= params <= 56_000_000
