import numpy as np
import pytest
import torch

from bramastra_lab.discovery.training import TrainConfig, Trainer
from bramastra_lab.discovery.worlds import RuleWorld, make_worlds
from bramastra_lab.research.learning.inquiry import (
    build_paired_teaching_data,
    depth_two_scores,
    stratified_semantic_split,
)
from bramastra_lab.research.learning.inquiry import run_d02 as d02
from bramastra_lab.discovery.learner import Investigator, LearnerConfig


def parity_fixture():
    return [
        RuleWorld("00", "parity-diagnostic", 2, (0, 0, 0, 0)),
        RuleWorld("10", "parity-diagnostic", 2, (0, 1, 0, 1)),
        RuleWorld("01", "parity-diagnostic", 2, (0, 0, 1, 1)),
        RuleWorld("11", "parity-diagnostic", 2, (0, 1, 1, 0)),
    ]


def test_depth_two_values_preparatory_parity_queries_at_one_bit():
    table = np.asarray([world.table for world in parity_fixture()], dtype=np.uint8)
    legal = np.asarray([False, True, True, False])
    scores, branches = depth_two_scores(table, target=3, legal=legal)
    assert scores.tolist() == pytest.approx([0.0, 1.0, 1.0, 0.0])
    assert branches == 4

    costly, _ = depth_two_scores(table, target=3, legal=legal, query_cost=0.1)
    assert costly.tolist() == pytest.approx([0.0, 0.8, 0.8, 0.0])


def test_depth_two_never_labels_target_or_repeat_as_legal():
    table = np.asarray([world.table for world in parity_fixture()], dtype=np.uint8)
    legal = np.asarray([True, False, True, False])  # action 1 used, target 3 excluded
    scores, _ = depth_two_scores(table, target=3, legal=legal)
    assert scores[1] == scores[3] == 0.0
    # With the unconditioned four-world posterior, querying action 2 alone
    # leaves the XOR target unresolved, so its gain is correctly zero.
    assert scores[2] == 0.0

    # Once the posterior is conditioned on action 0, action 2 determines the
    # target. This is the positive one-bit case the original fixture intended.
    conditioned = table[table[:, 0] == 0]
    conditioned_legal = np.asarray([False, True, True, False])
    conditioned_scores, _ = depth_two_scores(conditioned, target=3, legal=conditioned_legal)
    assert conditioned_scores[2] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="target.*masked"):
        depth_two_scores(table, target=3, legal=np.asarray([True, False, True, True]))
    with pytest.raises(ValueError, match="out of range"):
        depth_two_scores(table, target=4, legal=legal)
    with pytest.raises(ValueError, match="legal mask"):
        depth_two_scores(table, target=3, legal=legal.astype(np.float32))


def test_paired_adapter_changes_only_teacher_tensor_and_trainer_accepts_hashes():
    worlds = stratified_semantic_split(make_worlds(3))["train"]
    paired = build_paired_teaching_data(worlds, episodes=10, budget=2, seed=801)
    for name in ("observations", "lengths", "targets", "labels", "legal"):
        assert torch.equal(getattr(paired.one_step, name), getattr(paired.depth_two, name))
    assert not torch.equal(paired.one_step.gains, paired.depth_two.gains)
    assert paired.one_step.manifest["tensor_sha256"] != paired.depth_two.manifest["tensor_sha256"]
    stride = paired.one_step.manifest["budget"] + 1
    assert torch.equal(paired.one_step.gains[1::stride], paired.depth_two.gains[1::stride])
    assert not paired.one_step.gains[2::stride].any()
    assert not paired.depth_two.gains[2::stride].any()
    assert paired.diagnostics["one_remaining_labels_equal"]
    assert paired.diagnostics["zero_remaining_both_zero"]


def test_query_cost_is_applied_to_both_teachers_at_one_remaining_query():
    worlds = stratified_semantic_split(make_worlds(3))["train"]
    paired = build_paired_teaching_data(worlds, episodes=8, budget=2, seed=17, query_cost=0.1)
    stride = 3
    assert torch.equal(paired.one_step.gains[1::stride], paired.depth_two.gains[1::stride])
    assert (paired.one_step.gains >= 0).all()

    torch.manual_seed(9)
    model = Investigator(LearnerConfig(bits=3, width=8))
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    left = Trainer(model, paired.one_step, TrainConfig(steps=1, batch_size=4, seed=11))
    right_model = Investigator(LearnerConfig(bits=3, width=8))
    right_model.load_state_dict(initial)
    right = Trainer(right_model, paired.depth_two, TrainConfig(steps=1, batch_size=4, seed=11))
    assert all(torch.equal(left.model.state_dict()[key], right.model.state_dict()[key]) for key in initial)
    assert torch.equal(left.sampler.get_state(), right.sampler.get_state())


def test_stratified_split_is_disjoint_stable_and_populates_every_family():
    for bits in (3, 4):
        worlds = make_worlds(bits)
        split = stratified_semantic_split(worlds)
        assert split == stratified_semantic_split(list(reversed(worlds)))
        assert sum(map(len, split.values())) == len(worlds)
        assert len({world.table for values in split.values() for world in values}) == len(worlds)
        for values in split.values():
            assert {world.family for world in values} == {"parity", "conjunction", "threshold"}


def test_d02_pairing_rejects_duplicate_and_mismatched_rows():
    row = {"world_id": "w", "target": 0, "policy": "learned", "budget": 2,
           "label": 1, "family": "parity", "correct": True, "probability": 0.9}
    with pytest.raises(ValueError, match="duplicate"):
        d02._paired_comparison([row, row], [row, row], bootstrap=10, seed=1)
    mismatch = {**row, "label": 0}
    with pytest.raises(ValueError, match="labels"):
        d02._paired_comparison([row], [mismatch], bootstrap=10, seed=1)


def _mock_d02_pipeline(monkeypatch):
    model = Investigator(LearnerConfig(bits=4, width=32))
    monkeypatch.setattr(d02, "_check_paired_data", lambda paired, bits: None)
    monkeypatch.setattr(d02, "_train", lambda data, seed, output, config, initial_state, deadline:
                        (model, {"status": "COMPLETE", "initial_sha256": "init",
                                 "sampler_initial_sha256": "sampler", "parameter_count": 1,
                                 "final_sha256": "final", "seconds": 0.0, "history": [],
                                 "data": data.manifest}))
    def fake_evaluate(model, worlds, **kwargs):
        return [{"world_id": "dev", "family": "parity", "target": 0, "policy": policy,
                 "budget": budget, "correct": True, "probability": 0.9, "label": 1, "queries": budget}
                for policy in ("learned", "random", "coverage", "no_memory") for budget in (0, 1, 2)]
    monkeypatch.setattr(d02, "evaluate", fake_evaluate)
    monkeypatch.setattr(d02, "summarize", lambda rows, **kwargs: {})
    return model


def test_d02_runner_persists_timeout_and_failure_statuses(tmp_path, monkeypatch):
    timeout = d02.run(tmp_path / "timeout", deadline_seconds=0.0)
    assert timeout["status"] == "TIMEOUT"
    _mock_d02_pipeline(monkeypatch)
    monkeypatch.setattr(d02, "_train", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected")))
    failed = d02.run(tmp_path / "failed", config={"episodes": 1, "steps": 1}, deadline_seconds=30.0)
    assert failed["status"] == "FAILED" and failed["failure"]["type"] == "RuntimeError"


def test_d02_runner_source_mutation_invalidates_completion(tmp_path, monkeypatch):
    _mock_d02_pipeline(monkeypatch)
    original = d02._sha256
    calls = {"count": 0}
    def mutated(path):
        calls["count"] += 1
        value = original(path)
        return "0" * 64 if calls["count"] == 11 else value
    monkeypatch.setattr(d02, "_sha256", mutated)
    result = d02.run(tmp_path / "mutated", config={"episodes": 1, "steps": 1}, deadline_seconds=30.0)
    assert result["status"] == "FAILED" and result["source_integrity"] is False
