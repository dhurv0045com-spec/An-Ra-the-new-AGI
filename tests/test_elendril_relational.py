import pytest
import torch

from anra_v5.elendril.associative import AssociativeDecoder, DenseCapacityControl
from anra_v5.elendril.codec import BOS, Prediction, answer_tokens, encode
from anra_v5.elendril.counterfactual import CounterfactualBatches
from anra_v5.elendril.relational import KNOWN_KEYS, NOVEL_KEYS, positional_control, relational_fixture
from anra_v5.elendril.scoring import score
from anra_v5.elendril.training import MODEL, logits_for
from v5_model.core import initialize, packed_layout


def test_relational_split_stress_identity_and_query_blind_controls():
    train = relational_fixture("train")
    dev = relational_fixture("development")
    assert len(train) == 352 and len(dev) == 96
    assert {row.visible.cluster_id for row in train}.isdisjoint(
        row.visible.cluster_id for row in dev)
    assert {row.visible.prompt for row in train}.isdisjoint(row.visible.prompt for row in dev)
    assert CounterfactualBatches(train, 16).block_steps == 44
    with pytest.raises(ValueError, match="positive and even"):
        CounterfactualBatches(train, 0)
    for panel, fact_count in (("base", 3), ("alias", 3), ("two-facts", 2), ("five-facts", 5)):
        rows = relational_fixture("development", panel)
        assert [(r.visible.cluster_id, r.truth.gold) for r in rows] == [
            (r.visible.cluster_id, r.truth.gold) for r in dev]
        assert all(r.visible.prompt.count(":") == fact_count + 1 for r in rows)
        keys = NOVEL_KEYS if panel == "alias" else KNOWN_KEYS
        assert all(r.visible.prompt[-2] in keys for r in rows)
        assert score(rows, positional_control(rows, last=False))["query_swap_both_correct"] == [0, 48]
        assert score(rows, positional_control(rows, last=True))["query_swap_both_correct"] == [0, 48]
        oracle = [Prediction(r.visible.task_id, answer_tokens(r.truth.gold)) for r in rows]
        assert score(rows, oracle)["query_swap_both_correct"] == [48, 48]


def test_address_ablations_and_capacity_control_keep_core_and_causality():
    hybrid = AssociativeDecoder(initialize(MODEL, 3), 3)
    lexical = AssociativeDecoder(initialize(MODEL, 3), 3, mode="lexical")
    contextual = AssociativeDecoder(initialize(MODEL, 3), 3, mode="context")
    control = DenseCapacityControl(initialize(MODEL, 3), 3)
    assert sum(p.numel() for p in control.parameters()) - sum(p.numel() for p in hybrid.parameters()) == 6
    assert not hasattr(lexical, "address_query")
    assert not hasattr(contextual, "lexical_weights")
    tokens = torch.tensor([(BOS, *encode("B:J:A;K:B?J>A"))])
    changed = tokens.clone()
    changed[0, -1] = encode("H")[0]
    segments = torch.zeros_like(tokens)
    for model in (hybrid, lexical, contextual, control):
        assert torch.equal(model.core.embedding.weight, hybrid.core.embedding.weight)
        scores = logits_for(model, tokens, segments)
        assert torch.isfinite(scores).all()
        torch.testing.assert_close(scores[:, :-1], logits_for(model, changed, segments)[:, :-1],
                                   atol=0, rtol=0)


def test_lexical_address_scores_are_invariant_under_symbol_bijection():
    model = AssociativeDecoder(initialize(MODEL, 7), 7, mode="lexical")
    with torch.no_grad():
        model.lexical_weights.copy_(torch.arange(9).reshape(3, 3).float())
    a = torch.tensor([(BOS, *encode("B:J:A;K:B?J>"))])
    b = torch.tensor([(BOS, *encode("B:U:A;V:B?U>"))])
    positions, _ = packed_layout(torch.zeros_like(a), torch_module=torch)
    assert torch.equal(model._lexical_scores(a, positions), model._lexical_scores(b, positions))
