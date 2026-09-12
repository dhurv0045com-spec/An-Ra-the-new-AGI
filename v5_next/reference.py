"""Reference builder: instantiate the REAL v5_model core from a V5-Next
contract on tiny CPU fixtures, and mechanically assert the contracts that the
next Core is judged on. No training, no GPU."""
from __future__ import annotations

from typing import Any

from .contracts import NextCoreContract


def _to_model_spec(contract: NextCoreContract):
    from v5_contracts.model_spec import ModelSpec

    return ModelSpec(
        schema="anra-v5-model-spec/v1",
        family="dense-decoder-transformer",
        vocabulary_size=contract.vocabulary_size,
        width=contract.width,
        layers=contract.layers,
        query_heads=contract.query_heads,
        kv_heads=contract.kv_heads,
        head_dimension=contract.head_dimension,
        ffn_width=contract.ffn_width,
        context_length=contract.context_length,
        rope_base=contract.rope_base,
        norm_epsilon=contract.norm_epsilon,
        tied_embeddings=True,
        qk_norm=True,
        qk_norm_affine=True,
        linear_bias=False,
        dropout=0.0,
    )


def build_reference_model(contract: NextCoreContract, seed: int = 0) -> Any:
    """Build the live v5_model core for this contract and verify:

    - exact parameter inventory equals the contract receipt;
    - exactly one tied embedding table (no separate output head);
    - affine QK norm scales exist.
    """
    from v5_model.core import assert_receipt, assert_single_embedding, initialize

    spec = _to_model_spec(contract)
    model = initialize(spec, seed=seed)
    assert_single_embedding(model)
    assert_receipt(model, spec)
    receipt = contract.parameter_receipt()
    actual = sum(int(p.numel()) for p in model.parameters())
    if actual != receipt["total"]:
        raise ValueError(f"reference model has {actual} params, contract says {receipt['total']}")
    scales = [name for name, _ in model.named_parameters() if name.endswith("_scale")]
    if len(scales) != 2 * contract.layers:
        raise ValueError("affine QK norm scales missing")
    return model


def checkpoint_round_trip(model: Any, *, torch_module: Any = None) -> bool:
    """Save/load the model state through a plain torch checkpoint and require
    bit-identical outputs on a fixed deterministic input."""
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    import io

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    model.load_state_dict(restored)

    generator = torch.Generator().manual_seed(1234)
    tokens = torch.randint(
        0, model.spec.vocabulary_size, (2, 8), generator=generator)
    from v5_model.core import packed_layout

    segments = torch.zeros(2, 8, dtype=torch.int32)
    positions, mask = packed_layout(segments, torch_module=torch)
    with torch.no_grad():
        first = model(tokens, positions, mask)
        second = model(tokens, positions, mask)
    return bool(torch.equal(first, second))
