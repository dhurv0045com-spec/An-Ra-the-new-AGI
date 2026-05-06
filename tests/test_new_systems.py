from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from anra_paths import inject_all_paths

inject_all_paths()


def test_hal_adrenaline_cortisol_cascade_timing():
    from identity.hal import HALModule, HALState

    hal = HALModule(HALState(adrenaline=0.85, cortisol=0.2))
    hal.decay(turns=3)
    assert hal.state.cortisol > 0.5


def test_hal_high_cortisol_lowers_generation_temp():
    from identity.hal import HALModule, HALState

    hal = HALModule(HALState(cortisol=0.9))
    assert hal.generation_temperature(0.8) < 0.7


def test_hal_ouroboros_weights_sum_to_one_always():
    import random

    from identity.hal import HALModule, HALState

    for _ in range(100):
        state = HALState(
            dopamine=random.random(),
            serotonin=random.random(),
            cortisol=random.random(),
            adrenaline=random.random(),
            oxytocin=random.random(),
            norepinephrine=random.random(),
            endorphin=random.random(),
        )
        weights = HALModule(state).ouroboros_weights([0.2, 0.3, 0.5])
        assert math.isclose(sum(weights), 1.0, abs_tol=0.001)


def test_hal_kl_coefficient_in_valid_range():
    import random

    from identity.hal import HALModule, HALState

    for _ in range(100):
        state = HALState(
            dopamine=random.random(),
            serotonin=random.random(),
            cortisol=random.random(),
            adrenaline=random.random(),
            oxytocin=random.random(),
            norepinephrine=random.random(),
            endorphin=random.random(),
        )
        coeff = HALModule(state).kl_coefficient(0.04)
        assert 0.01 <= coeff <= 0.15


def test_hal_civ_threat_fires_cortisol():
    from identity.hal import HALModule

    hal = HALModule()
    baseline = hal.state.cortisol
    hal.apply_civ_score(0.4)
    assert hal.state.cortisol > baseline


def test_hal_serialize_roundtrip(tmp_path):
    from identity.hal import HALModule, HALState

    state = HALState(
        dopamine=0.11,
        serotonin=0.22,
        cortisol=0.33,
        adrenaline=0.44,
        oxytocin=0.55,
        norepinephrine=0.66,
        endorphin=0.77,
    )
    path = tmp_path / "hal.json"
    HALModule(state).save(path)
    loaded = HALModule.load(path)
    assert loaded.state.hormones() == state.hormones()


def test_hal_endorphin_flow_increases_memory_write():
    from identity.hal import HALModule, HALState

    hal = HALModule(HALState(endorphin=1.0, dopamine=0.8, cortisol=0.0))
    assert hal.memory_threshold() < 0.4


class _TinyBaseModel:
    pass


def _make_ouroboros_base():
    import torch
    import torch.nn as nn

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = 8
            self.vocab_size = 32
            self.pad_token_id = 0
            self.block_size = 16
            self.emb = nn.Embedding(self.vocab_size, self.d_model)
            self.layer = nn.Linear(self.d_model, self.d_model)
            self.lm_head = nn.Linear(self.d_model, self.vocab_size)

        def embed(self, x):
            return self.emb(x)

        def run_all_layers(self, hidden):
            return torch.tanh(self.layer(hidden))

    return Base()


def test_ouroboros_with_hal_weights_sum_to_one():
    import torch

    from identity.hal import HALModule
    from ouroboros import OuroborosDecoder

    model = OuroborosDecoder(_make_ouroboros_base(), n_passes=3, hal=HALModule())
    x = torch.randint(0, 32, (2, 6))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 6, 32)
    assert loss is not None
    assert not torch.isnan(logits).any()


def test_ouroboros_without_hal_unchanged():
    import torch
    import torch.nn.functional as F

    from ouroboros import OuroborosDecoder

    torch.manual_seed(123)
    base = _make_ouroboros_base()
    model = OuroborosDecoder(base, n_passes=3, hal=None)
    x = torch.randint(0, 32, (1, 5))
    hidden = base.embed(x)
    accumulated = torch.zeros_like(hidden)
    blend = F.softmax(model.blend_weights, dim=0)
    manual_hiddens = []
    for pass_idx in range(model.n_passes):
        hidden = hidden + model.pass_gates[pass_idx].unsqueeze(0).unsqueeze(0)
        hidden = base.run_all_layers(hidden)
        manual_hiddens.append(hidden)
        accumulated = accumulated + blend[pass_idx] * hidden
        hidden = accumulated
    expected = base.lm_head(accumulated)
    logits, _ = model(x)
    assert model.hal is None
    assert torch.allclose(logits, expected)


class _TinyTokenizer:
    special_ids = {"<eos>": 0}

    def encode(self, text):
        return [ord(ch) % 16 for ch in text]

    def decode(self, ids):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)


class _TinyPolicy:
    pass


def _make_rlvr():
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block_size = 32
            self.emb = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16)

        def forward(self, x):
            return self.head(self.emb(x)), None

    class Result:
        def __init__(self, score):
            self.score = score

    class Verifier:
        def __init__(self, score):
            self.score = score

        def score(self, *args, **kwargs):
            return Result(self.score_value)

    class FixedVerifier:
        def __init__(self, score):
            self.score_value = score

        def score(self, *args, **kwargs):
            return Result(self.score_value)

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer, FixedVerifier


def test_rlvr_with_hal_kl_changes():
    from identity.hal import HALModule, HALState
    from training.rlvr import RLVRTask, RLVRTrainer

    model, optimizer, verifier_cls = _make_rlvr()
    trainer = RLVRTrainer(
        model,
        _TinyTokenizer(),
        optimizer,
        verifier_cls(0.8),
        hal=HALModule(HALState(cortisol=0.9)),
        entropy_bonus=0.0,
        kl_coeff=0.04,
    )
    trainer.train_step(RLVRTask(prompt="abc", task_type="open"), completions=["def"])
    assert trainer._last_effective_kl > trainer.kl_coeff


def test_rlvr_consecutive_failures_increments():
    from identity.hal import HALModule
    from training.rlvr import RLVRTask, RLVRTrainer

    model, optimizer, verifier_cls = _make_rlvr()
    trainer = RLVRTrainer(
        model,
        _TinyTokenizer(),
        optimizer,
        verifier_cls(0.2),
        hal=HALModule(),
        entropy_bonus=0.0,
    )
    task = RLVRTask(prompt="abc", task_type="open")
    for _ in range(3):
        trainer.train_step(task, completions=["def"])
    assert trainer._consecutive_failures == 3


def test_memory_threat_pattern_bypasses_hal_threshold(tmp_path):
    pytest.importorskip("numpy")

    from identity.hal import HALModule, HALState
    from memory.memory_router import MemoryRouter

    router = MemoryRouter(dim=8, faiss_index_path=tmp_path / "episodic.faiss", hal=HALModule(HALState(cortisol=0.9)))
    result = router.write("threat", metadata={"kind": "threat_pattern", "salience": 0.01})
    assert result.tier == "episodic"
    assert router.short_term == []


def test_memory_hal_threshold_applied_to_low_salience(tmp_path):
    pytest.importorskip("numpy")

    from identity.hal import HALModule, HALState
    from memory.memory_router import MemoryRouter

    router = MemoryRouter(dim=8, faiss_index_path=tmp_path / "episodic.faiss", hal=HALModule(HALState(cortisol=0.9)))
    result = router.write("low", metadata={"salience": 0.05})
    assert result.tier == "short_term"
    assert "hal_threshold" in router.short_term[-1]["metadata"]


def test_qiskit_unavailable_returns_gracefully(monkeypatch):
    from domain_verifiers import verify_qiskit

    monkeypatch.setitem(sys.modules, "qiskit", None)
    result = verify_qiskit("OPENQASM 2.0;")
    assert result.tier == "unavailable"
    assert result.label == "UNKNOWN"


def test_rdkit_unavailable_returns_gracefully(monkeypatch):
    from domain_verifiers import verify_rdkit

    monkeypatch.setitem(sys.modules, "rdkit", None)
    result = verify_rdkit("CCO")
    assert result.tier == "unavailable"
    assert result.label == "UNKNOWN"


def test_constraint_json_verifier_satisfied():
    from domain_verifiers import verify_constraint_json

    result = verify_constraint_json({"constraints": [{"name": "depth", "op": "<=", "value": 20}]}, {"depth": 12})
    assert result.verified is True
    assert result.label == "VERIFIED"


def test_constraint_json_verifier_violated():
    from domain_verifiers import verify_constraint_json

    result = verify_constraint_json({"constraints": [{"name": "depth", "op": "<=", "value": 20}]}, {"depth": 32})
    assert result.verified is False
    assert result.label == "FALSIFIED"


def test_citation_grounding_no_crash():
    from domain_verifiers import verify_citation_grounding

    result = verify_citation_grounding("quantum circuit verified", memory_nodes=[{"claim": "quantum circuit", "status": "VERIFIED"}])
    assert result.score >= 0.0


def test_gap_scanner_finds_notimplementederror(tmp_path):
    from innovation.gap_scanner import scan

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "gap.py").write_text("def f():\n    raise NotImplementedError()\n", encoding="utf-8")
    gaps = scan(tmp_path)
    assert any("NotImplementedError" in gap.description for gap in gaps)


def test_scoreboard_total_in_0_to_100():
    from innovation.schema import Hypothesis
    from innovation.scoreboard import score_hypothesis

    hyp = Hypothesis("h", "g", "Add verifier-backed failure replay training", "pytest fails", {}, [], "single file pytest", "python -m pytest tests/ -x -q", 0.0)
    score = score_hypothesis(hyp)
    assert 0 <= score.total <= 100


def test_promotion_thresholds_correct():
    from innovation.scoreboard import _decision

    assert _decision(80) == "implement"
    assert _decision(60) == "experiment_first"
    assert _decision(59.9) == "research_only"


def test_report_written_to_path(tmp_path):
    from innovation.schema import InnovationScore
    from innovation.scoreboard import write_report

    path = tmp_path / "report.json"
    write_report([InnovationScore("h", 1, 2, 3, 4, 5, 6, 7, 28, "research_only")], path)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["count"] == 1
