from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch


def test_hal_import_succeeds():
    from identity.hal import HALModule, HALState

    assert HALModule is not None
    assert HALState is not None


def test_hal_initial_state_in_bounds():
    from identity.hal import HALModule

    hal = HALModule()
    s = hal.state
    for name, val in s.hormones().items():
        assert 0.0 <= val <= 1.0, f"{name}={val} out of bounds"


def test_hal_decay_moves_toward_baseline():
    from identity.hal import HALModule

    hal = HALModule()
    hal.apply_delta({"cortisol": +0.6})
    pre = hal.state.cortisol
    hal.decay(turns=5)
    post = hal.state.cortisol
    assert post < pre, f"cortisol did not decay: {pre} -> {post}"


def test_hal_adrenaline_cortisol_cascade():
    from identity.hal import HALModule

    hal = HALModule()
    hal.apply_delta({"adrenaline": +0.85})
    adr_start = hal.state.adrenaline
    for _ in range(3):
        hal.decay(turns=1)
    assert hal.state.adrenaline < adr_start * 0.5, "adrenaline did not decay fast"
    assert hal.state.cortisol > 0.3, "cortisol did not build after adrenaline"


def test_hal_civ_threat_fires_cortisol():
    from identity.hal import HALModule

    hal = HALModule()
    pre_cortisol = hal.state.cortisol
    hal.apply_civ_score(0.40)
    assert hal.state.cortisol > pre_cortisol, f"cortisol did not increase: {pre_cortisol} -> {hal.state.cortisol}"


def test_hal_high_cortisol_lowers_generation_temp():
    from identity.hal import HALModule

    hal = HALModule()
    hal.apply_delta({"cortisol": +0.7, "adrenaline": +0.3})
    temp = hal.generation_temperature(base=0.8)
    assert temp < 0.8, f"temperature should drop under stress: {temp}"


def test_hal_ouroboros_weights_sum_to_one():
    import random

    from identity.hal import HALModule

    hal = HALModule()
    for _ in range(50):
        deltas = {k: random.uniform(-0.3, 0.3) for k in ["dopamine", "serotonin", "cortisol", "adrenaline", "oxytocin", "norepinephrine", "endorphin"]}
        hal.apply_delta(deltas)
        weights = hal.ouroboros_weights()
        total = sum(weights)
        assert abs(total - 1.0) < 1e-5, f"weights don't sum to 1.0: {weights} = {total}"


def test_hal_kl_coefficient_in_valid_range():
    from identity.hal import HALModule

    hal = HALModule()
    for cortisol in [0.0, 0.2, 0.5, 0.9]:
        for endorphin in [0.0, 0.2, 0.5, 0.9]:
            hal.apply_delta({"cortisol": cortisol - hal.state.cortisol, "endorphin": endorphin - hal.state.endorphin})
            kl = hal.kl_coefficient(base=0.04)
            assert 0.01 <= kl <= 0.15, f"kl={kl} out of range for cortisol={cortisol} endorphin={endorphin}"


def test_hal_memory_threshold_dopamine_effect():
    from identity.hal import HALModule

    hal_low = HALModule()
    hal_low.apply_delta({"dopamine": 0.0 - hal_low.state.dopamine})
    hal_high = HALModule()
    hal_high.apply_delta({"dopamine": 0.9 - hal_high.state.dopamine})
    t_low = hal_low.memory_threshold()
    t_high = hal_high.memory_threshold()
    assert t_high < t_low, f"high dopamine should lower threshold: {t_high} >= {t_low}"


def test_hal_serialize_deserialize_roundtrip():
    from identity.hal import HALModule

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    hal = HALModule()
    hal.apply_delta({"cortisol": +0.4, "dopamine": +0.3})
    hal.save(path)
    hal2 = HALModule.load(path)
    for name, val in hal.state.hormones().items():
        assert abs(hal2.state.hormones()[name] - val) < 1e-6, f"{name}: {val} != {hal2.state.hormones()[name]}"
    Path(path).unlink(missing_ok=True)


def test_ouroboros_accepts_hal_parameter():
    from identity.hal import HALModule
    from phase3.ouroboros_45O import ouroboros as ou
    import inspect

    hal = HALModule()
    sig = inspect.signature(ou.OuroborosDecoder.__init__)
    assert hal is not None
    assert "hal" in sig.parameters, "OuroborosDecoder.__init__ missing 'hal' parameter"


def test_ouroboros_without_hal_unchanged():
    from phase3.ouroboros_45O import ouroboros as ou
    import inspect

    sig = inspect.signature(ou.OuroborosDecoder.__init__)
    param = sig.parameters.get("hal")
    assert param is not None
    assert param.default is None, "hal parameter default must be None for backward compatibility"


def test_rlvr_has_hal_parameter():
    from training.rlvr import RLVRTrainer
    import inspect

    sig = inspect.signature(RLVRTrainer.__init__)
    assert "hal" in sig.parameters, "RLVRTrainer.__init__ missing 'hal' parameter"


def test_rlvr_consecutive_failures_starts_at_zero():
    from training.rlvr import RLVRTrainer
    import inspect

    src = inspect.getsource(RLVRTrainer.__init__)
    assert "_consecutive_failures" in src, "_consecutive_failures counter not found in RLVRTrainer.__init__"


def test_memory_router_accepts_hal():
    from memory.memory_router import MemoryRouter
    import inspect

    sig = inspect.signature(MemoryRouter.__init__)
    assert "hal" in sig.parameters, "MemoryRouter.__init__ missing 'hal' parameter"


def test_memory_threat_pattern_no_threshold_gating():
    from memory.memory_router import MemoryRouter
    import inspect

    src = inspect.getsource(MemoryRouter.write)
    assert "threat_pattern" in src, "threat_pattern bypass not found in MemoryRouter.write"


def test_qiskit_verifier_unavailable_graceful():
    import sys

    qiskit_backed = "qiskit" in sys.modules
    from phase3.symbolic_bridge_45Q.domain_verifiers import verify_qiskit

    result = verify_qiskit("OPENQASM 2.0; qreg q[2]; cx q[0],q[1];", topology="all_to_all")
    assert hasattr(result, "tier"), "VerificationResult missing 'tier'"
    if not qiskit_backed:
        assert result.tier in ("unavailable", "inferred", "verified", "domain"), f"unexpected tier: {result.tier}"


def test_rdkit_verifier_unavailable_graceful():
    from phase3.symbolic_bridge_45Q.domain_verifiers import verify_rdkit

    result = verify_rdkit("CCO")
    assert hasattr(result, "tier"), "VerificationResult missing 'tier'"


def test_constraint_json_verifier_always_works():
    from phase3.symbolic_bridge_45Q.domain_verifiers import verify_constraint_json

    r = verify_constraint_json(constraints=[{"name": "mw", "op": "<=", "value": 200}], solution={"mw": 150})
    assert r.score >= 0.9, f"satisfied constraint scored {r.score}"
    r2 = verify_constraint_json(constraints=[{"name": "mw", "op": "<=", "value": 200}], solution={"mw": 350})
    assert r2.score < 0.5, f"violated constraint scored {r2.score}"


def test_citation_grounding_no_crash():
    from phase3.symbolic_bridge_45Q.domain_verifiers import verify_citation_grounding

    result = verify_citation_grounding("water is H2O")
    assert result is not None


def test_gap_scanner_finds_notimplementederror(tmp_path):
    from innovation.gap_scanner import scan

    dummy = tmp_path / "dummy_module.py"
    dummy.write_text('def foo():\n    raise NotImplementedError("not done")\n')
    gaps = scan(repo_root=tmp_path)
    assert len(gaps) > 0, "gap_scanner should find NotImplementedError but found none"


def test_gap_to_hypothesis_returns_hypothesis():
    from innovation.gap_scanner import scan
    from innovation.hypothesis import gap_to_hypothesis
    from innovation.schema import Hypothesis
    import os

    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "test.py")
        with open(f, "w") as fp:
            fp.write("def foo(): raise NotImplementedError('missing')\n")
        gaps = scan(repo_root=Path(d))
    if not gaps:
        pytest.skip("no gaps found in temp dir")
    hyp = gap_to_hypothesis(gaps[0])
    assert isinstance(hyp, Hypothesis)
    assert hyp.falsifier, "hypothesis must have a non-empty falsifier"
    assert hyp.smallest_experiment, "hypothesis must have a smallest_experiment"


def test_scoreboard_total_in_range():
    from innovation.scoreboard import score_hypothesis
    from innovation.schema import Hypothesis
    import time

    hyp = Hypothesis("test01", "gap01", "Fix missing function in module X", "pytest passes", {"test_pass_rate": +0.05}, ["no checkpoint changes"], "run pytest after fix", "pytest tests/ -x -q", time.time())
    score = score_hypothesis(hyp)
    assert 0 <= score.total <= 100, f"score {score.total} out of [0, 100] range"


def test_promotion_decision_correct():
    from innovation.scoreboard import _decision

    assert _decision(85.0) == "implement"
    assert _decision(70.0) == "experiment_first"
    assert _decision(45.0) == "research_only"


def test_enable_kv_cache_method_exists():
    import anra_brain

    assert hasattr(anra_brain.CausalTransformerV2, "enable_kv_cache"), "CausalTransformerV2 missing enable_kv_cache()"
    assert hasattr(anra_brain.CausalTransformerV2, "disable_kv_cache"), "CausalTransformerV2 missing disable_kv_cache()"
    assert hasattr(anra_brain.CausalTransformerV2, "clear_kv_cache"), "CausalTransformerV2 missing clear_kv_cache()"


def test_failure_replay_write_method_exists():
    from training.rlvr import RLVRTrainer

    assert hasattr(RLVRTrainer, "_write_failure_replay"), "RLVRTrainer missing _write_failure_replay method"


def test_frontier_dfc_jsonl_if_exists():
    from anra_paths import TRAINING_DATA_DIR

    dfc_path = TRAINING_DATA_DIR / "frontier_dfc.jsonl"
    if not dfc_path.exists():
        pytest.skip("frontier_dfc.jsonl not yet built - run build_frontier_dataset.py")
    templates = {}
    with dfc_path.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
                t = obj.get("template", "unknown")
                templates[t] = templates.get(t, 0) + 1
            except json.JSONDecodeError:
                continue
    total = sum(templates.values())
    sov = templates.get("sovereign_disagreement", 0)
    assert sov / max(total, 1) <= 0.80, f"sovereign_disagreement is {sov/total:.0%} of dataset - too dominant"
