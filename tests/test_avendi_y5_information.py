import torch
from experiments.avendi_y5 import information as y


def test_preflight_detects_planted_defects():
    assert y.preflight()['passed']
    assert all(y.preflight()['checks'].values())


def test_public_indistinguishable_and_solver():
    for public in {w['public'] for w in y.worlds()}:
        assert {w['truth'] for w in y.worlds() if w['public'] == public} == {0, 1}
    assert y.solve(.05, .75)['action'] == 1
    assert y.solve(.35, .75)['action'] == 0
    assert y.solve(.05, .5)['action'] == 0


def test_memory_clearing_and_deletion():
    model = y.P6LearnedMemoryPolicy(width=8, seed=3)
    s = y.MemorySession(model)
    s.observe(0)
    s.write()
    saved = model.mem_state.clone()
    s.clear_context()
    assert not s.context and torch.equal(saved, model.mem_state)
    s.erase()
    assert not torch.any(model.mem_state)
    s.edit(1)
    assert not s.context and model.mem_used.item() == 1


def test_tiny_run():
    class Ctx:
        canary = True
        device = 'cpu'
        split = 'dev'
        def train(self, model, items, batches, loss_fn, lr):
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for batch in batches:
                opt.zero_grad()
                loss_fn(model, [items[i] for i in batch]).backward()
                opt.step()
            return {'updates': len(batches)}
    for stage in ('D', 'E'):
        job = next(j for j in y.jobs() if j['stage'] == stage)
        result = y.run(job, Ctx())
        assert result['training']['updates'] == 2
        assert result['metrics']['samples'] > 0


def test_menu_counterbalanced_and_paired():
    rows = y.d_items('informative')
    for menu in ([0, 1], [1, 0]):
        for cost in (.05, .35):
            selected = [r for r in rows if r['menu'] == menu and r['cost'] == cost]
            assert {r['truth'] for r in selected} == {0, 1}
            assert all(menu[r['action_position']] == r['action'] for r in selected)


def test_public_renderings_are_disjoint_and_pairs_preserved():
    for builder in (lambda split: y.worlds(split), lambda split: y.e_items('sham', split)):
        sets = [{r['public'] for r in builder(split)} for split in ('train', 'dev', 'confirm')]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for split in ('train', 'dev', 'confirm'):
            rows = builder(split)
            for public in {r['public'] for r in rows}:
                assert {r['truth'] for r in rows if r['public'] == public} == {0, 1}


def test_heldout_recombinations_reuse_trained_primitives():
    for builder in (lambda split: y.worlds(split), lambda split: y.e_items('sham', split)):
        train = {x for row in builder('train') for x in row['public']}
        for split in ('dev', 'confirm'):
            assert {x for row in builder(split) for x in row['public']} <= train


def test_existing_heads_can_represent_cost_menu_and_law_feedback_forks():
    # Construct a weight witness; no optimization or checkpoint assumptions.
    model = y.P6LearnedMemoryPolicy(width=4, seed=3)
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        # Low/high cost and both menu positions are separate primitives.
        model.in_emb.weight[0, 1] = 1
        model.in_emb.weight[1, 1] = -1
        model.action_head.weight[0, 1] = -1
        model.action_head.weight[1, 1] = 1
        # Writer encodes diagnostic observation as a signed memory bit.
        model.obs_emb.weight[0, 0] = -2
        model.obs_emb.weight[1, 0] = 2
        model.writer_gru.bias_ih[4:8] = -100  # update gate z -> 0
        model.writer_gru.weight_ih[8, 4] = 1
        # Reader decodes the stored hypothesis from the memory component.
        model.read_head.weight[0, 4] = -1
        model.read_head.weight[1, 4] = 1
        model.read_head.bias[2:] = -10
    # Production session writes, clears, reads and uses the public truth table.
    for law in (0, 1):
        for feedback in (0, 1):
            session = y.MemorySession(model)
            session.observe(feedback)
            session.write()
            session.clear_context()
            assert session.read((law, 4, 6)) == (law ^ feedback)
            session.edit(1-feedback)
            assert session.read((law, 4, 6)) == (law ^ (1-feedback))
    # Both semantic decisions survive swapping the public menu presentation.
    for cost in (0, 1):
        for order in (0, 1):
            public = torch.tensor([[cost, 2+order, 4, 6]])
            menu = torch.tensor([[0, 1] if order == 0 else [1, 0]])
            _, canonical = y._forward(model, public)
            position = int(y._menu_logits(canonical, menu).argmax(-1).item())
            assert menu[0, position].item() == (1 if cost == 0 else 0)


def test_perfect_channel_resolves_every_paired_world_and_cannot_affect_prior_action():
    rows = y.d_items('perfect', 'dev')
    assert rows and all(row['obs'] == row['truth'] for row in rows)
    for public in {row['public'] for row in rows}:
        group = [row for row in rows if row['public'] == public]
        assert {row['truth'] for row in group} == {0, 1}
        assert sum(row['truth'] == 0 for row in group) == sum(row['truth'] == 1 for row in group)
    metrics = y._eval_d(y.P6LearnedMemoryPolicy(width=8, seed=7), 'perfect', 'dev')
    assert metrics['action_invariant_to_hidden_observation']
    assert abs(metrics['regret'] - (metrics['oracle_expected_utility'] - metrics['realized_utility'])) < 1e-9


def test_bellman_oracle_1e8_batched_parity_and_transfer_split():
    import pytest
    for cost in (.05, .35):
        for rel in (.5, .75, 1.0):
            s = y.solve(cost, rel)
            b = y.bellman_oracle_tree(cost, rel)
            assert abs(s['direct'] - b['direct']) < 1e-8
            assert abs(s['tool'] - b['tool']) < 1e-8
            assert s['action'] == b['action']
    tr = {w['public'] for w in y.worlds('transfer')}
    used = {w['public'] for sp in ('train', 'dev', 'confirm') for w in y.worlds(sp)}
    assert not (tr & used)
    assert y._eval_d(y.P6LearnedMemoryPolicy(width=32, seed=7), 'informative', 'transfer', verify_serial_n=8, parity_atol=1e-4)['samples'] > 0
    # Induce a batch-vs-serial mismatch (>1e-4) -> must raise ValueError cleanly.
    m = y.P6LearnedMemoryPolicy(width=8, seed=7)
    with pytest.raises(ValueError, match='BATCHED_SERIAL_PARITY_MISMATCH'):
        y._eval_d(m, 'informative', 'dev', verify_serial_n=8, parity_atol=-1.0)

