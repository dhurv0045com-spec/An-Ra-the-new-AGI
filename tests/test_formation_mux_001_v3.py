"""Science-S3 regression tests for FORMATION-MUX-001 clip isolation."""

import pytest

torch = pytest.importorskip("torch")

from anra_v5 import formation_mux_model_v3 as fxm
from v5_experiments import formation_mux_protocol_v3 as proto


def test_v3_causal_matrix_isolated():
    proto.assert_contrast_isolation()
    rows = {r["arm"]: r for r in proto.causal_variable_matrix()}
    assert rows["M1_EXTRA_NO_DECAY"]["training_denominator_classes"] == 24576
    assert rows["M2_EXTRA_FROZEN"]["training_denominator_classes"] == 24576
    assert rows["M1_EXTRA_NO_DECAY"]["extra_row_weight_decay"] == 0.0
    assert rows["M2_EXTRA_FROZEN"]["extra_row_weight_decay"] == 0.0
    assert rows["M1_EXTRA_NO_DECAY"]["extra_row_parameter_updates"] is True
    assert rows["M2_EXTRA_FROZEN"]["extra_row_parameter_updates"] is False


def test_frozen_extra_gradient_is_not_zeroed_before_clip():
    model = fxm.build_model(73011, "M2_EXTRA_FROZEN", torch=torch, device=torch.device("cpu"))
    model.embedding.weight.grad = torch.ones_like(model.embedding.weight)
    before = model.embedding.weight.grad.clone()
    fxm.mask_frozen_gradients_before_clip(model, "M2_EXTRA_FROZEN")
    assert torch.equal(model.embedding.weight.grad, before)
    assert float(model.embedding.weight.grad[4096:].norm()) > 0.0


def test_frozen_extra_rows_still_do_not_update():
    p = torch.randn((24576, 2), dtype=torch.float32)
    initial_extra = p[4096:].clone()
    p.grad = torch.randn_like(p)
    opt = fxm.EmbeddingRowOptimizer(
        p,
        trainable_rows={i: 0.1 for i in range(4096)},
        lr=1e-3,
        torch=torch,
    )
    # Mimic the production order: whole-parameter clip first, row update second.
    torch.nn.utils.clip_grad_norm_([p], 1.0)
    opt.step()
    assert torch.equal(p[4096:], initial_extra)
    assert not torch.equal(p[:4096], torch.zeros_like(p[:4096]))


def test_protocol_hash_binds_clip_semantics():
    payload = proto.protocol_payload(proto.EXPERIMENT_A)
    assert "frozen extra-row gradients retained" in payload["clip_semantics"]
    assert len(proto.protocol_sha(proto.EXPERIMENT_A)) == 64
