from dataclasses import replace
import random
import unittest

try:
    import pytest
    import torch
except ImportError:
    raise unittest.SkipTest("BRAMASTRA experiment checks require the bramastra extra")

from bramastra_lab.data import BOS, EOS, PAD, assert_disjoint, build_worlds, decode, encode, make_batch
from bramastra_lab.experiment import cpu_tree, evaluate, objective, optimizer_for, resume_check, tensor_hash, trees_equal, update
from bramastra_lab.model import BramastraModel, ModelConfig
from bramastra_lab.analyze import readouts


def test_query_marker_does_not_conflict_with_an_entity_named_q():
    records = [
        {"world_id": "same", "prompt": "A=1;Q=2;Q=A;V=", "answer": "1", "prediction": "2", "stop_reason": "EOS", "correct": False},
        {"world_id": "same", "prompt": "A=1;Q=2;Q=Q;V=", "answer": "2", "prediction": "2", "stop_reason": "EOS", "correct": True},
    ]
    result = readouts(records)
    assert result["same_answer_despite_query_swap_worlds"] == 1
    assert result["both_correct_worlds"] == 0
    assert result["copy_first_value_correct"] == 1


def test_bytes_roundtrip_and_special_ids_are_not_text():
    text = "नमस्ते λ 😀\n"
    assert decode(encode(text)) == text
    with pytest.raises(ValueError):
        decode([EOS])
    with pytest.raises(UnicodeError):
        decode([255 + 4])


def test_all_queries_stay_in_world_and_rendering_does_not_change_identity():
    rows = build_worlds(seed=4, count=8)
    alternate = build_worlds(seed=4, count=8, style="alternate")
    assert len(rows) == 16
    assert {row.world_id for row in rows} == {row.world_id for row in alternate}
    assert [row.answer for row in rows] == [row.answer for row in alternate]
    for start in range(0, len(rows), 2):
        assert rows[start].world_id == rows[start + 1].world_id
        assert rows[start].answer != rows[start + 1].answer
    with pytest.raises(ValueError, match="crosses"):
        assert_disjoint(rows, alternate)
    fresh = build_worlds(seed=4, count=8, exclude_ids={row.world_id for row in rows})
    assert_disjoint(rows, fresh)


def test_terminal_is_supervised_and_prompt_padding_are_not():
    rows = build_worlds(seed=1, count=1)
    batch = make_batch(rows, max_length=32)
    for row, tokens, active in zip(rows, batch["tokens"], batch["target_mask"]):
        assert tokens[0] == BOS
        assert tokens[active].tolist() == encode(row.answer) + [EOS]
        assert not active[tokens == PAD].any()
    with pytest.raises(ValueError, match="truncation"):
        make_batch(rows, max_length=4)


def test_objective_shifts_once_and_terminal_weight_changes_only_terminal_gradient():
    rows = build_worlds(seed=1, count=1)
    batch = make_batch(rows, max_length=32)
    logits = torch.zeros(2, 32, 260, requires_grad=True)
    objective(logits, **batch, terminal_weight=0).backward()
    without = logits.grad.clone()
    logits.grad.zero_()
    objective(logits, **batch, terminal_weight=1).backward()
    with_terminal = logits.grad
    answer = batch["target_mask"][:, 1:] & (batch["tokens"][:, 1:] != EOS)
    terminal = batch["target_mask"][:, 1:] & (batch["tokens"][:, 1:] == EOS)
    assert torch.equal(without[:, :-1][answer], with_terminal[:, :-1][answer])
    assert without[:, :-1][terminal].abs().sum() == 0
    assert with_terminal[:, :-1][terminal].abs().sum() > 0
    assert without[:, -1].abs().sum() == 0


class ScriptedModel:
    config = ModelConfig(max_seq=64)

    def __init__(self, prompt_length, suffix):
        self.prompt_length = prompt_length
        self.suffix = suffix

    def eval(self):
        return self

    def __call__(self, tokens):
        logits = torch.full((*tokens.shape, 260), -100.0)
        token = self.suffix[min(tokens.shape[1] - self.prompt_length, len(self.suffix) - 1)]
        logits[:, -1, token] = 100.0
        return logits


def test_generation_requires_correct_complete_answer_and_eos():
    row = build_worlds(seed=1, count=1)[0]
    prefix = 1 + len(encode(row.prompt))
    correct = evaluate(ScriptedModel(prefix, encode(row.answer) + [EOS]), [row], device="cpu")
    assert correct["exact_accuracy"] == 1
    garbage = evaluate(ScriptedModel(prefix, encode(row.answer + "9") + [EOS]), [row], device="cpu")
    assert garbage["exact_accuracy"] == 0
    assert garbage["answer_prefix_accuracy_diagnostic"] == 1
    nonstop = evaluate(ScriptedModel(prefix, encode(row.answer)), [row], device="cpu")
    assert nonstop["exact_accuracy"] == 0
    assert nonstop["stop_histogram"] == {"MAX_TOKENS": 1}
    # Changing evaluator truth cannot influence the generated sequence.
    other = evaluate(ScriptedModel(prefix, encode(row.answer) + [EOS]), [replace(row, answer="x")], device="cpu")
    assert correct["records"][0]["prediction"] == other["records"][0]["prediction"]


def test_checkpoint_restores_optimizer_sampler_and_registered_final_weights(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(11)
    config = ModelConfig(width=16, layers=1, heads=2, ffn=32, max_seq=32)
    model = BramastraModel(config)
    optimizer = optimizer_for(model)
    rows = build_worlds(seed=2, count=2)
    sampler = random.Random(31)
    for _ in range(2):
        indices = sampler.sample(range(len(rows)), 2)
        update(model, optimizer, [rows[i] for i in indices], config=config, terminal_weight=1, device="cpu")
    final_hash = tensor_hash(model.state_dict())
    final_optimizer = cpu_tree(optimizer.state_dict())
    final_sampler = sampler.getstate()
    result = resume_check(model, optimizer, sampler, rows, completed=2, batch_size=2, config=config,
                          terminal_weight=1, device="cpu", checkpoint_path=tmp_path / "state.pt", manifest_hash="a" * 64)
    assert result["sampler_equal"] and result["parameters_exact"] and result["optimizer_exact"]
    assert final_hash == tensor_hash(model.state_dict())
    assert trees_equal(final_optimizer, optimizer.state_dict())
    assert final_sampler == sampler.getstate()
