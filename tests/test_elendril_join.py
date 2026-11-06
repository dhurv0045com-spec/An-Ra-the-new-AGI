import torch

from anra_v5.elendril.associative import AssociativeDecoder
from anra_v5.elendril.codec import BOS, encode
from anra_v5.elendril.join import controls, join_fixture, parse, solve
from anra_v5.elendril.training import MODEL, logits_for
from v5_model.core import initialize


def test_oracles_shortcuts_and_disjoint_graphs():
    train = join_fixture("train")
    dev = join_fixture("development")
    assert len(train) == 256 and len(dev) == 128
    assert {r.visible.cluster_id for r in train}.isdisjoint(r.visible.cluster_id for r in dev)
    assert {r.visible.prompt for r in train}.isdisjoint(r.visible.prompt for r in dev)
    scores = controls(dev)
    assert scores["oracle"]["query_swap_both_correct"] == [64, 64]
    assert scores["one_step"]["exact_eos"] == [0, 128]
    for policy in ("first", "last", "frequency"):
        assert scores[policy]["query_swap_both_correct"] == [0, 64]
    for row in dev:
        edges, root = parse(row.visible.prompt)
        endpoints = {target for source, middle in edges if source == root
                     for next_source, target in edges if middle == next_source}
        assert endpoints == {row.truth.gold}
    for base, rewire in zip(dev, join_fixture("development", "rewire")):
        edges, root = parse(base.visible.prompt)
        changed, other = parse(rewire.visible.prompt)
        assert root == other and solve(edges, root, 1) == solve(changed, root, 1)
        assert sorted(base.visible.prompt) == sorted(rewire.visible.prompt)
        assert base.truth.gold != rewire.truth.gold
    assert solve((("A", "B"),), "A", 2) == "!"
    assert solve((("A", "B"), ("A", "C")), "A", 1) == "!"


def test_two_reads_are_causal_and_differentiable():
    model = AssociativeDecoder(initialize(MODEL, 7), 7, mode="two-read")
    ids = torch.tensor([(BOS, *encode("B:A:B;B:C?A>C"))])
    segments = torch.zeros_like(ids)
    logits = logits_for(model, ids, segments)
    changed = ids.clone()
    changed[:, -1] = encode("D")[0]
    assert torch.equal(logits[:, :-1], logits_for(model, changed, segments)[:, :-1])
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits.exp().sum(-1), torch.ones_like(ids, dtype=torch.float32))
    dropped = logits_for(model, ids, segments, remove_second_read=True)
    assert not torch.equal(dropped, logits)
    (-logits[0, -2, encode("C")[0]]).backward()
    assert model.join_strength.grad.abs().item() > 0
    assert torch.count_nonzero(model.join_offsets.grad) > 0
    assert torch.isfinite(model.address_query.weight.grad).all()


def test_two_read_resume_matches_uninterrupted_tensors(tmp_path):
    from anra_v5.elendril.training import Trainer, configure_cpu

    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    configure_cpu()
    try:
        config = dict(rows=join_fixture("train"), seed=7, steps=32, root=tmp_path,
                      source_commit="a"*40, method="two-read")
        reference = Trainer(**config, lineage="reference")
        interrupted = Trainer(**config, lineage="interrupted")
        assert reference.step() == interrupted.step()
        interrupted.save()
        resumed = Trainer(**config, lineage="interrupted", resume=True)
        assert reference.step() == resumed.step()
        assert all(torch.equal(a, b) for a, b in zip(reference.model.parameters(), resumed.model.parameters()))
        for a, b in zip(reference.optimizer.state.values(), resumed.optimizer.state.values()):
            assert a.keys() == b.keys()
            assert all(torch.equal(a[key], b[key]) for key in a)
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
