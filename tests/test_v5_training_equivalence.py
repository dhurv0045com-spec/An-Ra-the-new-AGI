"""Activation-checkpointing and replica-sharded accumulation oracle (CPU torch).

Proves: checkpointed blocks preserve logits/loss/gradients, and sharding one
logical microstep into replicas with the replica-global denominator reproduces
the single-shot global update (loss, grads, params) within justified
tolerance (different kernel tiling changes fp summation order).
Run: python tests/test_v5_training_equivalence.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _tiny_spec():
    from v5_contracts.model_spec import ModelSpec
    return ModelSpec(
        schema="anra-v5-model-spec/v1", family="dense-decoder-transformer",
        vocabulary_size=64, width=32, layers=2, query_heads=2, kv_heads=1,
        head_dimension=16, ffn_width=64, context_length=128,
        rope_base=10_000.0, norm_epsilon=1e-5, tied_embeddings=True,
        qk_norm=True, qk_norm_affine=True, linear_bias=False, dropout=0.0)


def test_activation_checkpointing_equivalence():
    import torch
    from v5_model.core import initialize, packed_layout

    torch.manual_seed(11)
    model = initialize(_tiny_spec(), 11, torch_module=torch)
    model.train()
    tokens = torch.tensor([[2, 5, 6, 7, 3, 2, 8, 9, 3, 0, 0, 0],
                           [2, 4, 4, 5, 6, 7, 8, 9, 10, 11, 3, 0]])
    segments = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]])
    positions, mask = packed_layout(segments, torch_module=torch)
    plain = model(tokens, positions, mask)
    checked = model(tokens, positions, mask, use_activation_checkpointing=True)
    assert torch.equal(plain, checked), "checkpointing changed logits"
    model.zero_grad(set_to_none=True)
    plain.sum().backward()
    grads_plain = [p.grad.detach().clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    checked.sum().backward()
    for before, after in zip(grads_plain, (p.grad for p in model.parameters())):
        assert torch.equal(before, after), "checkpointing changed gradients"
    model.eval()
    with torch.no_grad():
        again = model(tokens, positions, mask, use_activation_checkpointing=True)
    assert torch.equal(plain, again), "inference path diverged"


def test_replica_sharded_accumulation_oracle():
    import torch
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    from v5_training.production_backend import (
        ProductionTrainingBackend,
        bounded_warmup_schedule,
    )
    from v5_training.topology_map import replica_shards

    schedule = bounded_warmup_schedule(peak_learning_rate=3e-4)
    # One logical microstep, three UNEQUAL replicas (hides denominator bugs).
    rows = [
        ([2, 5, 6, 7, 3, 0, 0, 0], [0, 0, 0, 0, 0, -1, -1, -1]),
        ([2, 8, 9, 3, 0, 0, 0, 0], [1, 1, 1, 1, -1, -1, -1, -1]),
        ([2, 4, 5, 6, 7, 8, 9, 3], [2, 2, 2, 2, 2, 2, 2, 2]),
    ]
    full_tokens = torch.tensor([row for row, _ in rows], dtype=torch.long)
    full_segments = torch.tensor([seg for _, seg in rows], dtype=torch.long)
    shards = replica_shards([(list(t), list(s)) for t, s in
                             zip(full_tokens.tolist(), full_segments.tolist())],
                            replicas=3)
    assert [len(shard) for shard in shards] == [1, 1, 1]

    from v5_objectives.causal_lm import causal_lm_loss

    def fresh():
        torch.manual_seed(21)
        model = initialize(_tiny_spec(), 21, torch_module=torch)
        optimizer = build_adamw_optimizer(model, torch_module=torch)
        backend = ProductionTrainingBackend(
            model=model, optimizer=optimizer, bos_id=2, pad_id=0,
            device="cpu", schedule=schedule, torch_module=torch,
            activation_checkpointing=False)
        return model, backend

    def eligible_of(tokens, segments):
        logits = torch.zeros(tokens.shape[0], tokens.shape[1], 64)
        _, count = causal_lm_loss(logits, tokens, segments, bos_id=2,
                                  pad_id=0, torch_module=torch)
        return count

    global_total = eligible_of(full_tokens, full_segments)
    shard_totals = [eligible_of(torch.tensor([t for t, _ in shard], dtype=torch.long),
                                torch.tensor([s for _, s in shard], dtype=torch.long))
                    for shard in shards]
    assert global_total == sum(shard_totals) and global_total > 0
    assert len(set(shard_totals)) > 1, "fixture replicas must be unequal"
    # Finish both with identical denominators and compare.
    from v5_training.state import (CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState,
                                   IdentityBindings, TrainingState)
    import hashlib
    pack_sha = hashlib.sha256(b"oracle").hexdigest()
    identities = IdentityBindings(
        IDENTITY_SCHEMA, "a" * 40, pack_sha, pack_sha, pack_sha, pack_sha,
        pack_sha, pack_sha, pack_sha, pack_sha)
    mkcursor = lambda: CursorState(CURSOR_SCHEMA, pack_sha, 0, 0, 0)
    state_a = TrainingState.initial(
        lineage_id="oa", token_budget=1000, tokens_per_update=1000,
        cursor=mkcursor(), rng_state_sha256="b" * 64,
        curriculum_phase="u", identities=identities)
    # Re-run with real states so finish_update certifies honestly.
    torch.manual_seed(21)
    model_a, backend_a = fresh()
    ctx_a = backend_a.begin_update(state_a)
    ctx_a = backend_a.accumulate_microstep(
        ctx_a, tokens=full_tokens, segment_ids=full_segments,
        eligible=torch.ones_like(full_tokens, dtype=torch.bool),
        tokens_by_source={"t": int((full_segments >= 0).sum())},
        planned_total=global_total)
    report_a = backend_a.finish_update(
        state_a, ctx_a, planned_total=global_total, cursor=mkcursor())
    params_a = {n: p.detach().clone() for n, p in model_a.named_parameters()}
    grads_a = {n: p.grad.detach().clone() for n, p in model_a.named_parameters()}
    torch.manual_seed(21)
    model_b, backend_b = fresh()
    ctx_b = backend_b.begin_update(state_a)
    for shard in shards:
        tokens = torch.tensor([t for t, _ in shard], dtype=torch.long)
        segments = torch.tensor([s for _, s in shard], dtype=torch.long)
        ctx_b = backend_b.accumulate_microstep(
            ctx_b, tokens=tokens, segment_ids=segments,
            eligible=torch.ones_like(tokens, dtype=torch.bool),
            tokens_by_source={"t": int((segments >= 0).sum())},
            planned_total=global_total)
    report_b = backend_b.finish_update(
        state_a, ctx_b, planned_total=global_total, cursor=mkcursor())
    assert report_a.tokens_by_source == report_b.tokens_by_source
    assert abs(backend_a.last_receipt["loss"] - backend_b.last_receipt["loss"]) < 1e-5
    params_b = {n: p.detach().clone() for n, p in model_b.named_parameters()}
    grads_b = {n: p.grad.detach().clone() for n, p in model_b.named_parameters()}
    for name in params_a:
        assert torch.allclose(params_a[name], params_b[name],
                              atol=1e-5), f"param diverged: {name}"
        assert torch.allclose(grads_a[name], grads_b[name],
                              atol=1e-5), f"grad diverged: {name}"


_TESTS = [test_activation_checkpointing_equivalence,
          test_replica_sharded_accumulation_oracle]


def main() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {exc}", flush=True)
    print(f"{len(_TESTS) - failed}/{len(_TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
