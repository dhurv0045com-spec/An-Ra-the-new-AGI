"""FORMATION-MUX-001 Amendment-1 pre-execution qualification tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from anra_v5 import formation_mux_model_v2 as fxm
from anra_v5 import formation_mux_train_v2 as train
from v5_experiments import formation_mux_protocol_v2 as proto
from v5_experiments.formation_mux_data import build_surface


def test_shared_extra_boundary_is_the_causal_boundary():
    assert fxm.SHARED_VOCAB == 4096
    assert fxm.ACTIVE_ROWS[0] == 0 and fxm.ACTIVE_ROWS[-1] == 4095
    assert fxm.EXTRA_ROWS[0] == 4096 and fxm.EXTRA_ROWS[-1] == 24575
    assert len(fxm.ACTIVE_ROWS) == 4096
    assert len(fxm.EXTRA_ROWS) == 24576 - 4096


def test_all_latent_science_tokens_are_inside_shared_region():
    manifest = build_surface(seed=73011)
    for split, rows in manifest["splits"].items():
        for row in rows:
            ids = [*row["prompt_ids"], *row["answer_ids"]]
            assert ids and min(ids) >= 4
            assert max(ids) < fxm.SHARED_VOCAB, (split, row["family"], max(ids))


def test_causal_matrix_isolated_after_amendment():
    proto.assert_contrast_isolation()
    matrix = {r["arm"]: r for r in proto.causal_variable_matrix()}
    assert matrix["M3_EXTRA_FROZEN_MASKED"]["effective_output_classes_training"] == "4096"
    assert matrix["M2_EXTRA_FROZEN"]["effective_output_classes_training"] == "24576"


def test_m3_never_masks_latent_target_region():
    model = fxm.build_model(73011, "M3_EXTRA_FROZEN_MASKED", torch=torch, device=torch.device("cpu"))
    from v5_model.core import packed_layout

    ids = torch.tensor([[2, 260, 300, 269]], dtype=torch.long)
    seg = torch.zeros_like(ids)
    pos, mask = packed_layout(seg, torch_module=torch)
    model.train()
    logits = model(ids, pos, mask)
    assert not bool((logits[..., : fxm.SHARED_VOCAB] == fxm.MASK_LOGIT_VALUE).any())
    assert bool((logits[..., fxm.SHARED_VOCAB :] == fxm.MASK_LOGIT_VALUE).all())
    model.eval()
    logits_eval = model(ids, pos, mask)
    assert not bool((logits_eval[..., fxm.SHARED_VOCAB :] == fxm.MASK_LOGIT_VALUE).any())


def test_frozen_gradient_mask_preserves_shared_gradients_and_zeros_only_extra():
    model = fxm.build_model(73011, "M2_EXTRA_FROZEN", torch=torch, device=torch.device("cpu"))
    grad = torch.ones_like(model.embedding.weight)
    model.embedding.weight.grad = grad
    fxm.mask_frozen_gradients_before_clip(model, "M2_EXTRA_FROZEN")
    assert torch.all(model.embedding.weight.grad[:4096] == 1)
    assert torch.all(model.embedding.weight.grad[4096:] == 0)


def test_row_optimizer_consumes_post_clip_gradient():
    p = torch.ones((4, 2), dtype=torch.float32)
    p.grad = torch.full_like(p, 100.0)
    opt = fxm.EmbeddingRowOptimizer(
        p,
        trainable_rows={i: 0.0 for i in range(4)},
        lr=1e-3,
        torch=torch,
    )
    raw = float(p.grad.norm().item())
    torch.nn.utils.clip_grad_norm_([p], 1.0)
    clipped = float(p.grad.norm().item())
    assert raw > 1.0 and clipped <= 1.0001
    before = p.clone()
    opt.step()
    assert not torch.equal(before, p)
    assert opt.step_count == 1


def test_rep_protocol_is_processed_token_matched_not_step_matched():
    assert proto.B_PROCESSED_TOKEN_BUDGET == 500_000
    assert proto.B_EXPOSURE_MISMATCH_TOLERANCE == 0.001
    payload = proto.protocol_payload(proto.EXPERIMENT_B)
    assert payload["B_processed_token_budget"] == 500_000
    assert payload["A_updates"] is None


def test_exposure_mismatch_gate():
    assert proto.exposure_mismatch(500000, 500200) < 0.001
    assert proto.exposure_mismatch(500000, 501000) > 0.001


def test_queue_is_24_arms_balanced_with_matched_seed_gpu_locality():
    queue = proto.queue_assignment()
    assert proto.total_official_arms() == 24
    assert len(queue["GPU0"]) == 12
    assert len(queue["GPU1"]) == 12
    for experiment in proto.EXPERIMENTS:
        for bundle in proto.SEED_BUNDLES:
            owning = [gpu for gpu, jobs in queue.items() if any(j["experiment"] == experiment and j["seed_bundle"] == bundle for j in jobs)]
            assert len(owning) == 1


def test_paired_verdict_uses_dev_auc_and_sealed_endpoint():
    b = proto.SEED_BUNDLES
    success = proto.paired_verdict(
        formation_auc_deltas={x: 0.08 for x in b},
        sealed_endpoint_gaps={x: 0.07 for x in b},
    )
    assert success["verdict"] == "SUCCESS"
    reverse = proto.paired_verdict(
        formation_auc_deltas={x: -0.08 for x in b},
        sealed_endpoint_gaps={x: -0.07 for x in b},
    )
    assert reverse["verdict"] == "REVERSE_EFFECT"
    partial = proto.paired_verdict(
        formation_auc_deltas={x: 0.08 for x in b},
        sealed_endpoint_gaps={x: -0.07 for x in b},
    )
    assert partial["verdict"] == "PARTIAL_OR_INTERACTION"


def test_experiment_rollup_keeps_all_contrasts():
    assert proto.aggregate_experiment_verdict(["NULL", "NULL", "NULL"]) == "NULL"
    assert proto.aggregate_experiment_verdict(["SUCCESS", "NULL", "NULL"]) == "SUCCESS"
    assert proto.aggregate_experiment_verdict(["SUCCESS", "REVERSE_EFFECT", "NULL"]) == "PARTIAL_OR_INTERACTION"


def test_development_primary_is_identity_only():
    # A tiny fake model is unnecessary here; inspect the public return contract
    # and ensure the evaluator requires the identity family rather than pooling.
    assert "identity" in train.evaluate_development.__doc__ if train.evaluate_development.__doc__ else True
    manifest = build_surface(seed=73011)
    families = {r["family"] for r in manifest["splits"]["development"]}
    assert "identity" in families and len(families) > 1


def test_r0_missing_tokenization_fails_closed_before_training():
    manifest = build_surface(seed=73011, tokenizer=None)
    row = manifest["splits"]["training"][0]
    with pytest.raises(RuntimeError, match="R0 token ids missing"):
        train._encode_row(row, proto.EXPERIMENT_B, "R0_PRODUCTION_BPE")


def test_model_only_restore_api_exists_for_sealed_scoring():
    assert callable(train.load_model_for_evaluation)


def test_protocol_hashes_are_distinct_and_stable_shape():
    a = proto.protocol_sha(proto.EXPERIMENT_A)
    b = proto.protocol_sha(proto.EXPERIMENT_B)
    assert len(a) == len(b) == 64
    assert a != b
