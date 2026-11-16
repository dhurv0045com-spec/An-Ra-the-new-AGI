"""Priority 1 verification: versioned PlanthirBackend contract, X1 R/D2 grouped loss, and provenance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from anra_v5.elendril.foundation_training import Recipe
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.contracts import build_v8_model_spec
from anra_v5.elendril.v8.data import D1_TRAIN_SEED, build_d1_training_dataset
from anra_v5.elendril.v8.training import prepare_v8_training_data
from anra_v5.planthir import (
    CartridgeArmSpec,
    CartridgeRunner,
    CausalContrastSpec,
    ExperimentCartridge,
    V8EnvironmentKernel,
)
from anra_v5.planthir.adapters.x1_rd2_adapter import (
    X1RD2Backend,
    X1RD2EnvironmentKernel,
    X1RD2Recipe,
    X1RD2Spec,
    X1RD2Trainer,
    build_x1_model,
    compute_x1_d2_grouped_loss,
    hand_computed_x1_d2_reference,
    prepare_x1_rd2_training_data,
)
from anra_v5.planthir.backend import (
    BACKEND_CONTRACT_VERSION,
    V8PlanthirBackend,
    build_provenance_identity,
    get_backend,
    list_backends,
    register_backend,
    verify_provenance_identity,
)
from anra_v5.planthir.causal import _probe_split_keys

COMMIT = "00b2128b34755d332414ffae8eb78dd3f430e04f"


def _disjoint_x1_eval_worlds(train_worlds, *, seed_offset: int = 23):
    _, forbidden = _probe_split_keys(train_worlds)
    pool = build_d1_training_dataset(seed=D1_TRAIN_SEED + seed_offset, quota_per_family=16).worlds
    eval_w = [w for w in pool if not (_probe_split_keys([w])[1] & forbidden)][:4]
    _, f_eval = _probe_split_keys(eval_w)
    probe_w = [w for w in build_d1_training_dataset(seed=D1_TRAIN_SEED + 501, quota_per_family=16).worlds
               if not (_probe_split_keys([w])[1] & (forbidden | f_eval))][:4]
    mh_w = [w for w in eval_w if int(w.get("hops", 1)) >= 2][:2]
    assert len(eval_w) == 4 and len(probe_w) == 4
    return tuple(eval_w), tuple(mh_w), tuple(probe_w)


@pytest.mark.parametrize("backend_id", ["v8", "x1_rd2"])
def test_backend_contract_registry_and_negative_validation(backend_id):
    backends = list_backends()
    assert backend_id in backends
    backend = get_backend(backend_id)
    assert backend.contract_version == BACKEND_CONTRACT_VERSION
    with pytest.raises(ValueError, match="version mismatch"):
        get_backend(backend_id, expected_version="incompatible-v999")

    class _WrongVersion:
        backend_id = "bad"
        backend_version = "0.0"
        contract_version = "anra-planthir-backend-v999"

    with pytest.raises(ValueError, match="incompatible backend contract_version"):
        register_backend(_WrongVersion())  # type: ignore[arg-type]


def test_cross_backend_kernel_and_objective_rejection():
    v8 = V8PlanthirBackend()
    x1 = X1RD2Backend()
    with pytest.raises(ValueError, match="V8 backend requires V8EnvironmentKernel"):
        v8.validate_kernel(X1RD2EnvironmentKernel())

    base = replace(MODEL_PROFILES["S20"], width=32, layers=1, query_heads=2, kv_heads=1, head_dimension=16, ffn_width=64, context_length=512)
    v8_data = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=12)
    bad_obj_arm = CartridgeArmSpec(
        "J", "J", True, build_v8_model_spec("J", base_spec=base),
        Recipe(total_updates=2, batch_size=4, micro_batch_size=2, lr=1e-4, objective="action-weighted"),
        v8_data.data_identity,
    )
    with pytest.raises(ValueError, match="uniform CE only"):
        v8.validate_arm(bad_obj_arm, v8_data, device="cpu", cpu_canary=True)

    x1_data, _ = prepare_x1_rd2_training_data(num_groups=4)
    x1_spec = X1RD2Spec(architecture="X1-R", hidden_dim=32, num_layers=1, num_heads=2, pointer_dim=16, num_slots=4, slot_dim=16, slot_edge_hidden=16)
    bad_arch_arm = CartridgeArmSpec(
        "bad", "X1-P", True, None,
        X1RD2Recipe(total_updates=2, batch_size=4, micro_batch_size=2, lr=1e-4, objective="pair_eq_cf"),
        x1_data.data_identity, backend_id="x1_rd2", model_spec=x1_spec,
    )
    with pytest.raises(ValueError, match="architecture_family differs"):
        x1.validate_arm(bad_arch_arm, x1_data, device="cpu", cpu_canary=True)


def test_x1_d2_grouped_loss_and_gradients_match_hand_computed_reference():
    torch.manual_seed(17)
    n_states = 8
    logits_op = torch.randn(n_states, 4, requires_grad=True)
    logits_val = torch.randn(n_states, 16, requires_grad=True)
    log_p_op = torch.log_softmax(logits_op, dim=-1)
    log_p_val = torch.log_softmax(logits_val, dim=-1)
    log_probs_19 = torch.cat([log_p_op[:, 0:1] + log_p_val, log_p_op[:, 1:]], dim=-1)

    gold_ops = [0, 0, 0, 0, 1, 0, 2, 0]
    gold_args = [2, 3, 2, 5, 0, 4, 0, 6]
    perm = tuple((i + 1) % 16 for i in range(16))

    # Case 1: mixed masked states and eligible/ineligible EQ + CF pairs
    state_mask = [True, True, False, True, True, True, True, False]
    eq_specs = [(0, 1, perm, True), (0, 2, tuple(range(16)), False), (4, 5, perm, True)]
    cf_specs = [(0, 3, 2, 5, True), (5, 7, 4, 6, False)]

    ref_loss = hand_computed_x1_d2_reference(
        log_probs_19, log_p_op, log_p_val,
        gold_ops=gold_ops, gold_args=gold_args, state_mask=state_mask,
        eq_specs=eq_specs, cf_specs=cf_specs, eq_weight=0.1, cf_weight=0.1,
    )
    ref_grads = torch.autograd.grad(ref_loss, (logits_op, logits_val), retain_graph=True)

    act_loss, metrics = compute_x1_d2_grouped_loss(
        log_probs_19, log_p_op, log_p_val,
        gold_ops=torch.tensor(gold_ops), gold_args=torch.tensor(gold_args),
        state_mask=torch.tensor(state_mask, dtype=torch.bool),
        eq_a=torch.tensor([e[0] for e in eq_specs]),
        eq_b=torch.tensor([e[1] for e in eq_specs]),
        eq_perm_16=torch.tensor([list(e[2]) for e in eq_specs]),
        eq_mask=torch.tensor([e[3] for e in eq_specs], dtype=torch.bool),
        cf_a=torch.tensor([c[0] for c in cf_specs]),
        cf_b=torch.tensor([c[1] for c in cf_specs]),
        cf_target_a=torch.tensor([c[2] for c in cf_specs]),
        cf_target_b=torch.tensor([c[3] for c in cf_specs]),
        cf_mask=torch.tensor([c[4] for c in cf_specs], dtype=torch.bool),
        eq_weight=0.1, cf_weight=0.1,
    )
    act_grads = torch.autograd.grad(act_loss, (logits_op, logits_val), retain_graph=True)
    assert metrics["active_states"] == 6
    assert metrics["eligible_eq_pairs"] == 2
    assert metrics["eligible_cf_pairs"] == 1
    assert torch.allclose(act_loss, ref_loss, rtol=1e-5, atol=1e-6)
    for g_act, g_ref in zip(act_grads, ref_grads):
        assert torch.allclose(g_act, g_ref, rtol=1e-5, atol=1e-6)

    # Case 2: empty/all-masked groups still stay differentiable with zero gradient
    zero_loss, zero_metrics = compute_x1_d2_grouped_loss(
        log_probs_19, log_p_op, log_p_val,
        gold_ops=torch.tensor(gold_ops), gold_args=torch.tensor(gold_args),
        state_mask=torch.zeros(n_states, dtype=torch.bool),
        eq_a=torch.zeros(0, dtype=torch.int64), eq_b=torch.zeros(0, dtype=torch.int64),
        eq_perm_16=torch.zeros((0, 16), dtype=torch.int64), eq_mask=torch.zeros(0, dtype=torch.bool),
        cf_a=torch.zeros(0, dtype=torch.int64), cf_b=torch.zeros(0, dtype=torch.int64),
        cf_target_a=torch.zeros(0, dtype=torch.int64), cf_target_b=torch.zeros(0, dtype=torch.int64),
        cf_mask=torch.zeros(0, dtype=torch.bool), eq_weight=0.1, cf_weight=0.1,
    )
    g0 = torch.autograd.grad(zero_loss, logits_op)[0]
    assert zero_loss.item() == pytest.approx(0.0) and zero_metrics["active_states"] == 0
    assert torch.all(g0 == 0.0)


def test_x1_rd2_trainer_resume_preserves_group_ordering_and_weights(tmp_path):
    data, _ = prepare_x1_rd2_training_data(num_groups=6)
    spec = X1RD2Spec(architecture="X1-R", hidden_dim=16, num_layers=1, num_heads=2, pointer_dim=8, num_slots=2, slot_dim=8, slot_edge_hidden=8, max_sequence_length=512)
    recipe = X1RD2Recipe(seed=7, total_updates=3, batch_size=8, micro_batch_size=4, lr=1e-3, objective="pair_eq_cf")

    ref = X1RD2Trainer(
        data.groups, x1_spec=spec, recipe=recipe, order_seed=19,
        source_commit=COMMIT, root=tmp_path / "ref", lineage="x1-ref",
        data_provenance=data.manifest, device="cpu", cpu_canary=True,
    )
    planned_orders = []
    for _ in range(3):
        planned_orders.append(ref.next_group_ids())
        ref.step()
    ref_sha = ref.save()

    part1 = X1RD2Trainer(
        data.groups, x1_spec=spec, recipe=recipe, order_seed=19,
        source_commit=COMMIT, root=tmp_path / "res", lineage="x1-res",
        data_provenance=data.manifest, device="cpu", cpu_canary=True,
    )
    part1.step()
    part1.save()

    part2 = X1RD2Trainer(
        data.groups, x1_spec=spec, recipe=recipe, order_seed=19,
        source_commit=COMMIT, root=tmp_path / "res", lineage="x1-res",
        data_provenance=data.manifest, device="cpu", cpu_canary=True, resume=True,
    )
    assert part2.next_group_ids() == planned_orders[1]
    part2.step()
    assert part2.next_group_ids() == planned_orders[2]
    part2.step()
    res_sha = part2.save()
    assert part2.state.global_update == 3
    for k, v in ref.model.state_dict().items():
        assert torch.allclose(v, part2.model.state_dict()[k])
    assert res_sha != "" and part2.state.cumulative_tokens == ref.state.cumulative_tokens


def test_x1_rd2_end_to_end_cartridge_runner_with_non_v8_kernel(tmp_path):
    data, train_worlds = prepare_x1_rd2_training_data(num_groups=6)
    eval_w, mh_w, probe_w = _disjoint_x1_eval_worlds(train_worlds)
    kernel = X1RD2EnvironmentKernel()
    spec_p = X1RD2Spec(architecture="X1-P", hidden_dim=16, num_layers=1, num_heads=2, pointer_dim=8, max_sequence_length=512)
    spec_r = X1RD2Spec(architecture="X1-R", hidden_dim=16, num_layers=1, num_heads=2, pointer_dim=8, num_slots=2, slot_dim=8, slot_edge_hidden=8, max_sequence_length=512)
    recipe = X1RD2Recipe(seed=7, total_updates=2, batch_size=8, micro_batch_size=4, lr=1e-3, objective="pair_eq_cf")

    ctrl = CartridgeArmSpec(
        arm_id="X1P", architecture_family="X1-P", is_control=True,
        v8_spec=None, recipe=recipe, dataset_identity=data.data_identity,
        milestone_updates=(1, 2), backend_id="x1_rd2", model_spec=spec_p,
    )
    trt = CartridgeArmSpec(
        arm_id="X1R", architecture_family="X1-R", is_control=False,
        v8_spec=None, recipe=recipe, dataset_identity=data.data_identity,
        milestone_updates=(1, 2), backend_id="x1_rd2", model_spec=spec_r,
        contrasts=(CausalContrastSpec("C-R-VS-P", "architecture", "X1P", "X1R", (), "x1_relational_slot_readout"),),
    )
    cartridge = ExperimentCartridge(
        campaign_id="X1-MECH-TEST", program="X1-RD2", source_commit=COMMIT,
        kernel=kernel, control_arm=ctrl, treatment_arms=(trt,),
        train_data=data, eval_worlds=eval_w, multihop_worlds=mh_w,
        probe_train_worlds=probe_w, parameter_tolerance_ratio=2.0,
        train_worlds=train_worlds, backend_id="x1_rd2",
    )
    out = CartridgeRunner().run(cartridge, work_root=tmp_path / "x1_run")
    assert out["gates"]["all_5_gates_passed"] is True
    assert out["arms"]["X1R"]["causal_diagnostics"]["surgical_ablation"]["mechanism_type"] == "x1_relational_slot_readout"
    assert (tmp_path / "x1_run" / "PROVENANCE_IDENTITY.json").is_file()


def test_provenance_identity_changes_on_order_or_data_mutation():
    data, train_worlds = prepare_x1_rd2_training_data(num_groups=4)
    eval_w, mh_w, probe_w = _disjoint_x1_eval_worlds(train_worlds)
    spec_r = X1RD2Spec(architecture="X1-R", hidden_dim=16, num_layers=1, num_heads=2, pointer_dim=8, num_slots=2, slot_dim=8, slot_edge_hidden=8, max_sequence_length=512)
    recipe = X1RD2Recipe(seed=7, total_updates=2, batch_size=8, micro_batch_size=4, lr=1e-3, objective="pair_eq")
    arm = CartridgeArmSpec(
        "X1R", "X1-R", True, None, recipe, data.data_identity,
        backend_id="x1_rd2", model_spec=spec_r,
    )
    cart = ExperimentCartridge(
        "PROV", "X1", COMMIT, X1RD2EnvironmentKernel(), arm, (),
        data, eval_w, mh_w, probe_w, train_worlds=train_worlds, backend_id="x1_rd2",
    )
    backend = X1RD2Backend()
    m1 = build_provenance_identity(
        cartridge=cart, backend=backend, train_worlds=train_worlds,
        resolved_dataset_digest={"data_identity": data.data_identity},
        device="cpu", checkpoint_interval=4, max_result_zip_bytes=1024,
    )
    cart_reordered = ExperimentCartridge(
        "PROV", "X1", COMMIT, X1RD2EnvironmentKernel(), arm, (),
        data, tuple(reversed(eval_w)), mh_w, probe_w, train_worlds=train_worlds, backend_id="x1_rd2",
    )
    m2 = build_provenance_identity(
        cartridge=cart_reordered, backend=backend, train_worlds=train_worlds,
        resolved_dataset_digest={"data_identity": data.data_identity},
        device="cpu", checkpoint_interval=4, max_result_zip_bytes=1024,
    )
    diff = verify_provenance_identity(m1, m2)
    assert diff["matched"] is False
    assert "ORDERED_SPLITS_OR_MEMBERSHIP_CHANGED" in diff["mismatches"]
