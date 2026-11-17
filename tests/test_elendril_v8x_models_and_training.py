"""CPU verification of X1 models, capacity matching, gradient parity, crash/resume, and Planthir x1_s20."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from anra_v5.elendril.v8x.analysis import (
    cluster_bootstrap_paired_world_diff,
    summarize_x1_episodes,
)
from anra_v5.elendril.v8x.contracts import (
    X1ModelSpec,
    X1_ARCHITECTURES,
    build_x1_model_spec,
    build_x1_s20_base,
    solve_x1_capacity_controls,
)
from anra_v5.elendril.v8x.data import (
    build_x1_d1h_training_dataset,
    build_x1_d2_paired_dataset,
    build_x1_eval_splits,
)
from anra_v5.elendril.v8x.evaluation import (
    X1S20EnvironmentKernel,
    X1S20ObservablePolicy,
    evaluate_x1_worlds_batched,
    run_h01_frozen_intervention,
    run_x1_lesion_and_intervention_battery,
)
from anra_v5.elendril.v8x.models import (
    build_x1_s20_model,
    expected_x1_active_parameter_names,
)
from anra_v5.elendril.v8x.objectives import compute_x1_objective_batch
from anra_v5.elendril.v8x.training import X1S20Trainer, X1TrainRecipe
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.planthir import __main__ as planthir_cli
from dataclasses import replace
from v5_model.core import ModelSpec


def _tiny_base() -> ModelSpec:
    return replace(
        MODEL_PROFILES["S20"],
        family="v5-s20-tiny-test",
        layers=1,
        width=16,
        query_heads=2,
        kv_heads=1,
        head_dimension=8,
        ffn_width=32,
        context_length=640,
    )


def _tiny_x1_spec(arch: str) -> X1ModelSpec:
    return build_x1_model_spec(
        arch,
        base_spec=_tiny_base(),
        pointer_dim=8,
        slot_dim=8,
        num_slots=4,
        slot_edge_hidden=8,
        agg_iterations=2,
        msg_iterations=2,
        recurrent_dim=8,
    )


def _assert_equal_state_subset(m1: Any, m2: Any, prefix: str) -> None:
    sd1 = {k: v for k, v in m1.state_dict().items() if k.startswith(prefix)}
    sd2 = {k: v for k, v in m2.state_dict().items() if k.startswith(prefix)}
    assert sd1 and sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), f"shared parameter {k} differed across arms"


def test_s20_parameter_budget_and_capacity_matching_contracts() -> None:
    s20_base = build_x1_s20_base()
    assert s20_base.parameter_receipt().total == 19_871_616
    cap = solve_x1_capacity_controls(s20_base)
    assert float(cap["r_noedge_rel_diff"]) <= 0.01
    assert float(cap["p_cap_r_rel_diff"]) <= 0.01

    r_spec = build_x1_model_spec("R")
    rn_spec = build_x1_model_spec("R-noedge")
    pc_spec = build_x1_model_spec("P-capR")
    assert r_spec.added_parameters() == int(cap["r_added_parameters"])
    assert abs(rn_spec.added_parameters() - r_spec.added_parameters()) / r_spec.added_parameters() <= 0.01
    assert abs(pc_spec.added_parameters() - r_spec.added_parameters()) / r_spec.added_parameters() <= 0.01


def test_shared_component_initialization_and_active_gradients() -> None:
    # B0, J, F, F-cap, P, P-capR, R, R-noedge share identical backbone under the same init_seed
    tiny_models = {arch: build_x1_s20_model(_tiny_x1_spec(arch), init_seed=17) for arch in X1_ARCHITECTURES}
    b0 = tiny_models["B0"]
    for arch, mod in tiny_models.items():
        if arch == "B0":
            continue
        sd_core = {k: v for k, v in mod.core.state_dict().items()}
        sd_b0 = {k: v for k, v in b0.state_dict().items()}
        assert sd_core.keys() == sd_b0.keys()
        for k in sd_core:
            assert torch.equal(sd_core[k], sd_b0[k]), f"{arch} core mismatch on {k}"

    # F-head and pointer parameters are component-wise identical across P, P-capR, R, R-noedge
    ref_p = tiny_models["P"]
    for arch in ("P-capR", "R", "R-noedge"):
        _assert_equal_state_subset(ref_p, tiny_models[arch], "op_head.")
        _assert_equal_state_subset(ref_p, tiny_models[arch], "val_head.")
        _assert_equal_state_subset(ref_p, tiny_models[arch], "q_proj.")
        _assert_equal_state_subset(ref_p, tiny_models[arch], "k_proj.")
        _assert_equal_state_subset(ref_p, tiny_models[arch], "gate.")

    # Prepare one tiny batch containing ANSWER targets so every head receives active gradients
    d1h = build_x1_d1h_training_dataset(quota_per_family=1)
    ans_states = [s for s in d1h.states if s.gold_op == 0][:2]
    non_ans_states = [s for s in d1h.states if s.gold_op != 0][:2]
    states = tuple([*ans_states, *non_ans_states])

    for arch, mod in tiny_models.items():
        mod.zero_grad(set_to_none=True)
        if arch == "B0":
            from v5_model.core import packed_layout

            tokens = torch.tensor([list(s.tokens[:64]) for s in states], dtype=torch.int64)
            segments = torch.ones_like(tokens, dtype=torch.int64)
            positions, mask_attn = packed_layout(segments, torch_module=torch)
            logits = mod(tokens, positions, mask_attn)
            loss = logits[:, -1, :].sum()
            loss.backward()
        else:
            compute_x1_objective_batch(mod, objective="standard", states=states, device=torch.device("cpu"), backward=True)
        expected_active = expected_x1_active_parameter_names(mod, _tiny_x1_spec(arch), has_answer_targets=True)
        nonzero_grad = {
            name
            for name, param in mod.named_parameters()
            if param.grad is not None and param.grad.abs().max().item() > 0.0
        }
        missing = expected_active - nonzero_grad
        assert not missing, f"{arch} missing non-zero gradients on: {sorted(missing)}"


def test_grouped_d2_accumulation_parity_and_empty_mask_differentiability() -> None:
    d1h = build_x1_d1h_training_dataset(quota_per_family=1)
    # Mix canonical/ANSWER-bearing groups and non-ANSWER groups
    d2 = build_x1_d2_paired_dataset(base_worlds=d1h.worlds[:4])
    groups = d2.groups[:4]
    spec = _tiny_x1_spec("R")

    for objective in ("PAIR-CE", "PAIR-EQ", "PAIR-EQ-CF"):
        m_full = build_x1_s20_model(spec, init_seed=5)
        m_micro = build_x1_s20_model(spec, init_seed=5)

        total_states = sum(sum(1 for s in g.samples if s.state_mask) for g in groups)
        total_eq = sum(sum(1 for ep in g.eq_pairs if ep.eligible) for g in groups)
        total_cf = sum(sum(1 for cp in g.cf_pairs if cp.eligible) for g in groups)

        m_full.zero_grad(set_to_none=True)
        res_full = compute_x1_objective_batch(
            m_full,
            objective=objective,
            groups=groups,
            device=torch.device("cpu"),
            eq_weight=0.1,
            cf_weight=0.1,
            cf_margin=0.5,
            backward=True,
        )

        m_micro.zero_grad(set_to_none=True)
        for g in groups:
            compute_x1_objective_batch(
                m_micro,
                objective=objective,
                groups=(g,),
                device=torch.device("cpu"),
                eq_weight=0.1,
                cf_weight=0.1,
                cf_margin=0.5,
                ce_denom_override=float(max(total_states, 1)),
                eq_denom_override=float(max(total_eq, 1)),
                cf_denom_override=float(max(total_cf, 1)),
                backward=True,
            )

        assert res_full["loss_value"] == res_full["loss_value"]
        for (n1, p1), (n2, p2) in zip(m_full.named_parameters(), m_micro.named_parameters()):
            assert n1 == n2
            g1 = p1.grad if p1.grad is not None else torch.zeros_like(p1)
            g2 = p2.grad if p2.grad is not None else torch.zeros_like(p2)
            assert torch.allclose(g1, g2, atol=1e-5, rtol=1e-4), f"grad mismatch for {objective} param {n1}"

    # Zero eligible eq/cf pairs must remain cleanly finite and differentiable
    g0 = d2.groups[0]
    empty_pair_group = replace(
        g0,
        eq_pairs=tuple(replace(ep, eligible=False) for ep in g0.eq_pairs),
        cf_pairs=tuple(replace(cp, eligible=False) for cp in g0.cf_pairs),
    )
    m_z = build_x1_s20_model(spec, init_seed=5)
    res_z = compute_x1_objective_batch(
        m_z,
        objective="PAIR-EQ-CF",
        groups=(empty_pair_group,),
        device=torch.device("cpu"),
        backward=True,
    )
    assert res_z["eligible_eq_pairs"] == 0 and res_z["eligible_cf_pairs"] == 0
    assert torch.isfinite(torch.tensor(res_z["loss_value"]))


def test_x1_trainer_crash_resume_and_inference_milestones(tmp_path: Path) -> None:
    d1h = build_x1_d1h_training_dataset(quota_per_family=1)
    d2 = build_x1_d2_paired_dataset(base_worlds=d1h.worlds[:2])
    spec = _tiny_x1_spec("R")
    recipe = X1TrainRecipe(seed=3, total_updates=4, batch_size=2, micro_batch_size=1, objective="PAIR-EQ", sampling="uniform")
    commit = "5933d97e" + "0" * 32

    shared_bb_dir = tmp_path / "shared_bb"
    clean_ms_dir = tmp_path / "clean" / "milestones"
    clean = X1S20Trainer(
        x1_spec=spec,
        recipe=recipe,
        order_seed=11,
        source_commit=commit,
        root=tmp_path / "clean",
        lineage="clean_run",
        data_provenance=d2.manifest,
        groups=d2.groups[:4],
        device="cpu",
        cpu_canary=True,
    )
    clean.save_inference_milestone_weights(clean_ms_dir, update=0, shared_backbone_dir=shared_bb_dir)
    assert (clean_ms_dir / "u0000" / "milestone_weights_manifest.json").is_file()
    assert (clean_ms_dir / "u0000" / "head_weights.bin").is_file()
    assert any(shared_bb_dir.glob("init_backbone_*.bin"))
    for step_idx in range(4):
        clean.step()
        if step_idx + 1 in (2, 4):
            clean.save()
            clean.save_inference_milestone_weights(clean_ms_dir, update=step_idx + 1, shared_backbone_dir=shared_bb_dir)
    assert (clean_ms_dir / "u0002" / "model_inference.bin").is_file()

    # Interrupted + resumed run matches weights and receipts identically
    part = X1S20Trainer(
        x1_spec=spec,
        recipe=recipe,
        order_seed=11,
        source_commit=commit,
        root=tmp_path / "resumed",
        lineage="resumed_run",
        data_provenance=d2.manifest,
        groups=d2.groups[:4],
        device="cpu",
        cpu_canary=True,
    )
    part.step()
    part.step()
    part.save()
    resumed = X1S20Trainer(
        x1_spec=spec,
        recipe=recipe,
        order_seed=11,
        source_commit=commit,
        root=tmp_path / "resumed",
        lineage="resumed_run",
        data_provenance=d2.manifest,
        groups=d2.groups[:4],
        device="cpu",
        cpu_canary=True,
        resume=True,
    )
    resumed.step()
    resumed.step()
    resumed.save()

    for (n1, p1), (n2, p2) in zip(clean.model.named_parameters(), resumed.model.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1.detach(), p2.detach()), f"resume mismatch on {n1}"
    assert clean.state.global_update == resumed.state.global_update == 4


def test_batched_evaluation_lesions_u_sweep_and_cluster_bootstrap() -> None:
    d1h = build_x1_d1h_training_dataset(quota_per_family=1)
    splits = build_x1_eval_splits(train_worlds=d1h.worlds, dev_per_family=1, conf_per_family=1, depth_per_hop=1, milestone_per_family=1)
    kernel = X1S20EnvironmentKernel()

    r_spec = _tiny_x1_spec("R")
    r_mod = build_x1_s20_model(r_spec, init_seed=7)
    recs_r = evaluate_x1_worlds_batched(X1S20ObservablePolicy(r_mod, r_spec, model_identity="r0", mode="raw"), kernel, splits["dev_views"][:24])
    summary_r = summarize_x1_episodes(recs_r)
    assert summary_r["episodes"] == 24
    assert "macro_12_family_accuracy" in summary_r and "first_divergence_histogram" in summary_r

    lesion_rep = run_x1_lesion_and_intervention_battery(r_mod, r_spec, splits["dev_views"][:12], model_identity="r0")
    assert {"zero_r", "permute_edges", "sham", "disable_copy", "permute_val_bins"} <= set(lesion_rep["interventions"])

    u2_spec = _tiny_x1_spec("U-2")
    u2_mod = build_x1_s20_model(u2_spec, init_seed=7)
    u_rep = run_x1_lesion_and_intervention_battery(u2_mod, u2_spec, splits["dev_views"][:6], model_identity="u2", u_sweep_worlds=splits["depth_stress_views"][:4])
    assert set(u_rep["u_depth_sweep"]) == {"1", "2", "4", "8"}

    # Paired bootstrap compares R vs sham P-capR over base_world_id clusters
    p_spec = _tiny_x1_spec("P-capR")
    p_mod = build_x1_s20_model(p_spec, init_seed=7)
    recs_p = evaluate_x1_worlds_batched(X1S20ObservablePolicy(p_mod, p_spec, model_identity="p0", mode="raw"), kernel, splits["dev_views"][:24])
    boot = cluster_bootstrap_paired_world_diff(recs_r, recs_p, replicates=200)
    assert boot["base_world_clusters"] >= 2 and len(boot["ci95"]) == 2
    assert run_h01_frozen_intervention()["status"] == "HISTORICAL_WEIGHTS_NOT_AVAILABLE"


def test_planthir_x1_s20_end_to_end_cli_workflow(tmp_path: Path, capsys) -> None:
    assert planthir_cli.main(["doctor"]) == 0
    doc_out = json.loads(capsys.readouterr().out)
    assert doc_out["status"] == "GREEN_SIGNAL"
    assert "x1_s20" in doc_out["checks"]["backends_and_contracts"]["registered"]

    spec_path = tmp_path / "x1_s20_spec.json"
    assert planthir_cli.main(["init", "--backend", "x1_s20", "--out", str(spec_path)]) == 0
    _ = json.loads(capsys.readouterr().out)
    assert planthir_cli.main(["validate", "--spec", str(spec_path)]) == 0
    val_out = json.loads(capsys.readouterr().out)
    assert val_out["valid"] is True and val_out["backend_id"] == "x1_s20"

    work_root = tmp_path / "run_root"
    assert planthir_cli.main(["run", "--spec", str(spec_path), "--work-root", str(work_root)]) == 0
    run_out = json.loads(capsys.readouterr().out)
    assert run_out["status"] == "COMPLETE"
    bundle_dir = Path(run_out["bundle_dir"])
    assert (bundle_dir / "result.zip").is_file()

    assert planthir_cli.main(["verify-bundle", "--bundle-dir", str(bundle_dir)]) == 0
    ver_out = json.loads(capsys.readouterr().out)
    assert ver_out["verified"] is True

    replay_json = tmp_path / "replay_receipt.json"
    assert planthir_cli.main(["replay-bundle", "--bundle-dir", str(bundle_dir), "--output", str(replay_json)]) == 0
    rep_out = json.loads(capsys.readouterr().out)
    assert rep_out.get("verified", rep_out.get("passed", True)) is True and replay_json.is_file()
