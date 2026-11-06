from collections import Counter
from dataclasses import replace
import random

import pytest
import torch

from anra_v5.elendril.__main__ import benchmark_audit
from anra_v5.elendril.benchmark import corrupt_binding, fixture
from anra_v5.elendril.codec import ALPHABET, BOS, EOS, PAD, Prediction, answer_tokens, decode, encode
from anra_v5.elendril.scoring import score
from anra_v5.elendril.training import (
    CLIP_NORM, MODEL, Trainer, batch_tensors, generate, logits_for, serialize,
)
from v5_model.core import initialize
from v5_objectives.causal_lm import causal_lm_loss


SOURCE = "a" * 40


@pytest.fixture(autouse=True)
def cpu_reference():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def make_trainer(root, **changes):
    config = {"rows": fixture("binding", "train"), "root": root, "lineage": "test",
              "steps": 5, "seed": 7, "source_commit": SOURCE}
    config.update(changes)
    return Trainer(**config)


def assert_nested_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_symbol_roundtrip_and_control_rejection():
    assert decode(encode(ALPHABET)) == ALPHABET
    assert decode(encode("")) == ""
    with pytest.raises(ValueError, match="unsupported symbol"):
        encode("é")
    with pytest.raises(ValueError, match="content tokens"):
        decode((EOS,))


def test_answer_only_mask_shifts_exactly_once_including_eos():
    copy = fixture("copy", "train")[0]
    binding = fixture("binding", "train")[0]
    tokens, segments, eligible = batch_tensors([serialize(copy), serialize(binding)])
    assert tokens[0, -1] == PAD and not eligible[0, -1]
    logits = torch.zeros((*tokens.shape, MODEL.vocabulary_size), requires_grad=True)
    expected = []
    for index, row in enumerate((copy, binding)):
        start = len(row.visible.prompt)
        gold = answer_tokens(row.truth.gold)
        assert tokens[index, start + 1: start + 1 + len(gold)].tolist() == list(gold)
        expected.extend((index, start + position, token) for position, token in enumerate(gold))
    with torch.no_grad():
        for row, position, gold in expected:
            logits[row, position, gold] = 20
    loss, count = causal_lm_loss(logits, tokens, segments, bos_id=BOS, pad_id=PAD, eligible=eligible)
    assert count == 6 and loss.item() < 1e-5
    loss.backward()
    nonzero = logits.grad.abs().sum(-1).nonzero().tolist()
    assert nonzero == [[row, position] for row, position, _ in expected]


def test_causal_mask_and_exact_parameter_inventory():
    model = initialize(MODEL, 2)
    assert sum(tensor.numel() for tensor in model.parameters()) == MODEL.parameter_receipt().total
    assert list(name for name, _ in model.named_parameters() if name.endswith("embedding.weight")) == [
        "embedding.weight"]
    tokens = torch.tensor([(BOS, *encode("C:ABC>ABC"), EOS)])
    changed = tokens.clone()
    changed[:, 7:] = torch.tensor([*encode("HHH"), EOS])
    with torch.no_grad():
        reference = logits_for(model, tokens, torch.zeros_like(tokens))
        perturbed = logits_for(model, changed, torch.zeros_like(changed))
    assert torch.isfinite(reference).all()
    assert torch.equal(reference[:, :7], perturbed[:, :7])
    assert not torch.equal(reference[:, 7:], perturbed[:, 7:])


def test_generation_keeps_complete_answer_and_first_eos():
    class ScriptedModel(torch.nn.Module):
        def forward(self, tokens, positions, mask):
            generated = answer_tokens("ABC")
            logits = torch.zeros((*tokens.shape, MODEL.vocabulary_size))
            logits[:, -1, generated[tokens.shape[1] - 7]] = 1
            return logits

    predictions, cost = generate(ScriptedModel(), [("id", "C:ABC>")], max_new_tokens=5)
    assert predictions == (Prediction("id", answer_tokens("ABC")),)
    assert cost == {"forward_calls": 4, "evaluated_positions": 34}


def test_checkpoint_resume_matches_uninterrupted_oracle(tmp_path):
    oracle = make_trainer(tmp_path, lineage="oracle")
    subject = make_trainer(tmp_path)
    original = subject.model.embedding.weight.detach().clone()
    assert subject.sample_indices() == oracle.sample_indices()
    first = subject.step()
    assert first == oracle.step()
    assert not torch.equal(original, subject.model.embedding.weight)
    assert first["supervised_tokens"] == 32 and first["post_clip"] <= CLIP_NORM + 1e-6
    assert first["pre_clip"] >= first["post_clip"] - 1e-6
    optimizer = subject.optimizer.state[subject.model.embedding.weight]
    assert optimizer["step"] == 1
    assert torch.count_nonzero(optimizer["exp_avg"])
    assert torch.count_nonzero(optimizer["exp_avg_sq"])
    assert subject.step() == oracle.step()
    checkpoint = subject.save()
    resumed = make_trainer(tmp_path, resume=True)
    assert resumed.parent == checkpoint
    assert torch.equal(subject.rng.get_state(), resumed.rng.get_state())
    for _ in range(3):
        assert resumed.step() == oracle.step()
    assert_nested_equal(oracle.model.state_dict(), resumed.model.state_dict())
    assert_nested_equal(oracle.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert resumed.state.cumulative_tokens == oracle.state.cumulative_tokens
    assert resumed.state.cursor == oracle.state.cursor
    assert resumed.state.complete
    with pytest.raises(ValueError, match="complete"):
        resumed.step()


def test_checkpoint_fails_closed_on_missing_corruption_and_identity(tmp_path):
    with pytest.raises(ValueError, match="no committed"):
        make_trainer(tmp_path, resume=True)
    trainer = make_trainer(tmp_path)
    trainer.step()
    checkpoint = trainer.save()
    for changes in ({"seed": 8}, {"steps": 4}, {"source_commit": "b" * 40},
                    {"rows": fixture("binding", "train")[::-1]}):
        with pytest.raises(ValueError, match="identity mismatch"):
            make_trainer(tmp_path, resume=True, **changes)
    with pytest.raises(ValueError, match="explicit resume"):
        make_trainer(tmp_path)
    path = trainer.store.objects / checkpoint / "model.bin"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        make_trainer(tmp_path, resume=True)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        make_trainer(tmp_path, resume=True)


def test_checkpoint_fences_stale_writer_and_retries_interrupted_generation(tmp_path):
    first, stale = make_trainer(tmp_path), make_trainer(tmp_path)
    first.step()
    orphan = first.store.lineage_root / ".staging-1"
    orphan.mkdir()
    (orphan / "evidence").write_text("preserve")
    first.save()
    assert not orphan.exists()
    assert len(list(first.store.lineage_root.glob(".interrupted-1-*/evidence"))) == 1
    stale.step()
    stale.step()
    with pytest.raises(ValueError, match="stale parent"):
        stale.save()
    assert first.store.restore()[0].global_update == 1
    with pytest.raises(ValueError, match="same update"):
        first.save()


def test_evaluation_firewall_and_fail_closed_scoring(tmp_path):
    rows = fixture("binding", "development")
    with pytest.raises(ValueError, match="training rows"):
        make_trainer(tmp_path, rows=rows)
    assert not hasattr(rows[0].visible, "gold")
    assert not hasattr(rows[0].visible, "truth")
    predictions = [Prediction(row.visible.task_id, answer_tokens(row.truth.gold)) for row in rows]
    assert score(rows, predictions)["exact_eos"] == [56, 56]
    bad = replace(predictions[0], tokens=(BOS, EOS))
    assert score(rows, [bad, *predictions[1:]])["valid_eos"] == [55, 56]
    for tokens in (predictions[0].tokens[:-1], (*predictions[0].tokens, EOS), (PAD, EOS)):
        bad = replace(predictions[0], tokens=tokens)
        assert score(rows, [bad, *predictions[1:]])["exact_eos"] == [55, 56]
    with pytest.raises(ValueError, match="duplicate prediction"):
        score(rows, [*predictions, predictions[0]])
    with pytest.raises(ValueError, match="complete scoring fixture"):
        score(rows, predictions[1:])
    with pytest.raises(ValueError, match="invalid token"):
        score(rows, [replace(predictions[0], tokens=(-1,)), *predictions[1:]])


def test_fixture_partition_oracles_and_shortcuts():
    audit = benchmark_audit()
    assert audit == benchmark_audit()
    for family, result in audit.items():
        assert result["oracle"]["exact_eos"] == [result["dev_rows"]] * 2, family
        assert result["controls"]["deliberate_failure"]["exact_eos"][0] == 0
    binding = audit["binding"]["controls"]
    for policy in ("query_blind", "recency"):
        assert binding[policy]["exact_eos"] == [28, 56]
        assert binding[policy]["query_swap_both_correct"] == [0, 28]
    assert audit["composition"]["dev_clusters"] == 1
    assert audit["copy"]["controls"]["format_copy"]["exact_eos"] == [128, 128]


def test_label_permutation_preserves_visible_rows_and_marginals():
    rows = fixture("binding", "train")
    altered = corrupt_binding(rows, 100_007)
    assert altered == corrupt_binding(rows, 100_007)
    assert tuple(row.visible for row in altered) == tuple(row.visible for row in rows)
    assert any(a.truth != b.truth for a, b in zip(rows, altered))
    assert Counter((row.visible.cluster_id, row.truth.gold) for row in rows) == Counter(
        (row.visible.cluster_id, row.truth.gold) for row in altered)


def test_trainer_rejects_unsafe_checkpoint_directory(tmp_path):
    for lineage in ("..", ".", "A/B", "A:B"):
        with pytest.raises(ValueError, match="safe checkpoint directory name"):
            make_trainer(tmp_path, lineage=lineage)
    assert list(tmp_path.iterdir()) == []



def test_counterfactual_exposure_and_manipulation():
    from anra_v5.elendril.counterfactual import CounterfactualBatches, METHODS

    rows = fixture("binding", "train")
    batches = CounterfactualBatches(rows, 16)
    assert batches.block_steps == 21
    for method in ("shuffled-ce", "paired-ce"):
        stream = [batches.indices(7, update, METHODS[method]) for update in range(21)]
        assert Counter(i for batch in stream for i in batch) == dict.fromkeys(range(len(rows)), 2)
        if method == "paired-ce":
            assert all(batches.colocated(batch) == 16 for batch in stream)
        else:
            assert sum(batches.colocated(batch) for batch in stream) < 336
    assert batches.targets[0] == batches.targets[batches.partners[0]][::-1]
    with pytest.raises(ValueError, match="distinct queries"):
        CounterfactualBatches((rows[0], rows[0]), 2)


def test_contrast_targets_pre_answer_and_ignores_other_positions():
    from anra_v5.elendril.counterfactual import answer_contrast

    logits = torch.zeros((2, 5, 6), requires_grad=True)
    eligible = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]], dtype=torch.bool)
    loss = answer_contrast(logits, eligible, torch.tensor([[3, 4], [4, 3]]))
    loss.backward()
    assert logits.grad[0, 2, 3] < 0 < logits.grad[0, 2, 4]
    assert logits.grad[1, 1, 4] < 0 < logits.grad[1, 1, 3]
    assert logits.grad.count_nonzero().item() == 4


def test_associative_read_is_normalized_packed_causal_and_finite():
    from anra_v5.elendril.associative import AssociativeDecoder

    core = initialize(MODEL, 2)
    model = AssociativeDecoder(initialize(MODEL, 2), 2)
    assert_nested_equal(model.core.state_dict(), core.state_dict())
    tokens = torch.tensor([(BOS, *encode("ABC"), BOS, *encode("DEF"), PAD)])
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1]])
    original = logits_for(model, tokens, segments)
    assert torch.isfinite(original).all()
    torch.testing.assert_close(original.exp().sum(-1), torch.ones_like(tokens, dtype=torch.float32))
    changed = tokens.clone()
    changed[0, [2, 3, 7]] = torch.tensor(encode("HHH"))
    alternative = logits_for(model, changed, segments)
    assert torch.equal(original[:, :2], alternative[:, :2])
    assert torch.equal(original[:, 4:7], alternative[:, 4:7])
    original.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert model.lexical_weights.grad.count_nonzero() > 0


def test_associative_method_resume_and_recipe_identity(tmp_path):
    config = {"steps": 21, "method": "associative"}
    trainer = make_trainer(tmp_path, **config)
    oracle = make_trainer(tmp_path, lineage="oracle", **config)
    step = trainer.step()
    assert step == oracle.step() and step["colocated_rows"] == 16 and step["contrast"] > 0
    trainer.save()
    with pytest.raises(ValueError, match="identity mismatch"):
        make_trainer(tmp_path, steps=21, method="paired-contrast", resume=True)
    resumed = make_trainer(tmp_path, resume=True, **config)
    assert resumed.step() == oracle.step()
    assert_nested_equal(resumed.model.state_dict(), oracle.model.state_dict())
    assert_nested_equal(resumed.optimizer.state_dict(), oracle.optimizer.state_dict())


def test_stress_panels_preserve_truth_and_pairing():
    from anra_v5.elendril.benchmark import binding_panel, binding_panel_prompts

    dev = fixture("binding", "development")
    for name in ("alias", "distractor", "query-neutral"):
        rows = binding_panel(name)
        assert [r.truth.gold for r in rows] == [r.truth.gold for r in dev]
        assert [r.visible.cluster_id for r in rows] == [r.visible.cluster_id for r in dev]
        oracle = [Prediction(r.visible.task_id, answer_tokens(r.truth.gold)) for r in rows]
        assert score(rows, oracle)["query_swap_both_correct"] == [28, 28]
        if name != "query-neutral":
            assert {r.visible.prompt for r in rows}.isdisjoint(r.visible.prompt for r in dev)
    neutral = binding_panel_prompts("query-neutral", dev)
    assert len({prompt for _, prompt in neutral}) == 28
