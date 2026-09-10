from __future__ import annotations

import math

import pytest

from v5_experiments import cyr_gpu014_r1c as core


def _acq(value: float, *, updates: int = core.UPDATES) -> dict:
    trace = []
    for u in range(core.EVAL_EVERY, core.UPDATES + 1, core.EVAL_EVERY):
        trace.append({"update": u, "dev_measurement": {"complete_exact_with_valid_stop": value}})
    return {
        "updates": updates,
        "trace": trace,
        "reasoning_battery_final": {"STANDARD": {"complete_exact_with_valid_stop": value}},
    }


def _arm(value: float) -> dict:
    return {"acquisition": _acq(value)}


def test_treatment_specs_are_frozen_and_offset_matches_formula():
    assert core.treatment_spec("FULL_24576")["kind"] == "full"
    assert core.treatment_spec("MASK_19")["candidate_count"] == 19
    assert core.treatment_spec("MASK_4096")["candidate_count"] == 4096
    got = core.treatment_spec("OFFSET_EQ4096")["inactive_logit_offset"]
    want = math.log((24576 - 19) / (4096 - 19))
    assert got == pytest.approx(want)


def test_training_mask_preserves_shape_and_zeroes_excluded_gradient():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((2, 3, core.PHYSICAL_VOCAB), dtype=torch.float32, requires_grad=True)
    target = torch.zeros((2, 3), dtype=torch.long)
    treated = core.apply_training_logits(logits, "MASK_4096", torch_module=torch)
    assert treated.shape == logits.shape
    loss = torch.nn.functional.cross_entropy(treated.reshape(-1, treated.shape[-1]), target.reshape(-1))
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[..., 4096:].abs().max().item()) == 0.0
    assert float(logits.grad[..., :4096].abs().sum().item()) > 0.0


def test_offset_keeps_all_inactive_rows_in_gradient():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 1, core.PHYSICAL_VOCAB), dtype=torch.float32, requires_grad=True)
    treated = core.apply_training_logits(logits, "OFFSET_EQ4096", torch_module=torch)
    loss = torch.nn.functional.cross_entropy(treated.reshape(-1, treated.shape[-1]), torch.tensor([0]))
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[..., 19:].abs().sum().item()) > 0.0


def test_formation_metrics_require_fixed_endpoint_and_detect_sustained_g50():
    a = _acq(0.55)
    m = core.formation_metrics(a)
    assert m["complete"] is True
    assert m["formation_auc"] == pytest.approx(0.55)
    assert m["sustained_g50_update"] == 600
    assert m["endpoint_standard"] == pytest.approx(0.55)
    assert core.formation_metrics(_acq(0.55, updates=2999))["complete"] is False


def test_primary_decision_cannot_be_rescued_by_secondary_arms():
    arms = {}
    for i in range(4):
        for arm in core.ARMS:
            value = 0.90 if arm in {"MASK_8192", "MASK_16384"} else 0.05
            arms[core.arm_label(i, arm)] = _arm(value)
    d = core.decision(arms)
    assert d["verdict"] == "SOFTMAX_COMPETITION_NOT_SUFFICIENT"


def test_primary_supported_requires_replicated_mask4096_advantage():
    arms = {}
    for i in range(4):
        for arm in core.ARMS:
            if arm == "MASK_4096":
                value = 0.75
            elif arm == "FULL_24576":
                value = 0.05
            elif arm == "OFFSET_EQ4096":
                value = 0.70
            else:
                value = 0.30
            arms[core.arm_label(i, arm)] = _arm(value)
    d = core.decision(arms)
    assert d["verdict"] == "SOFTMAX_COMPETITION_CAUSALLY_SUPPORTED"
    assert d["inactive_partition_mass_rescue_supported"] is True


def _cal(ups: float = 100.0, eps: float = 1000.0, diag: float = 0.01):
    return {
        "status": "PASS",
        "batch_rows": core.BATCH_ROWS,
        "training_updates_per_sec": ups,
        "generation_examples_per_sec": eps,
        "diagnostic_seconds": diag,
    }


def test_runtime_resolver_requires_every_arm_and_never_drops_seeds():
    c = {arm: _cal() for arm in core.ARMS}
    r = core.resolve_from_calibrations(c)
    assert r["model_seeds"] == list(core.MODEL_SEEDS)
    assert r["arms"] == list(core.ARMS)
    broken = dict(c)
    broken.pop("MASK_4096")
    with pytest.raises(RuntimeError):
        core.resolve_from_calibrations(broken)


def test_runtime_resolver_fails_closed_when_full_campaign_does_not_fit():
    c = {arm: _cal(ups=0.5, eps=1.0, diag=100.0) for arm in core.ARMS}
    with pytest.raises(RuntimeError):
        core.resolve_from_calibrations(c)
