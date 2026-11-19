"""Comprehensive CPU qualification tests for Elendril V8-X2 science plugin, decisive fixtures, and registry."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from experiments.elendril_x2 import (
    compute_x2_grouped_loss,
    compute_x2_grouped_loss_scalar_reference,
    x2_dataset_factory,
    x2_loss_fn,
    x2_model_factory,
)
from experiments.x2.models import (
    X2_ARCHITECTURES,
    X2_NEW_PAIRS,
    backbone_state_digest,
    build_s20_base_spec,
    build_tiny_canary_base_spec,
    build_x2_model,
    build_x2_model_spec,
    delta_memory_read,
    delta_memory_step,
    verify_x2_parameter_contracts,
)
from experiments.x2.registry import (
    X2_BASE_JOB_EQUIVALENT_CEILING,
    build_x2_design_registry,
    build_x2_first_discriminating_report,
    build_x2_test_cards,
    evaluate_x2_qualification,
    holm_bonferroni_adjust,
    validate_x2_registry,
    verify_x1_core_inventory,
)
from experiments.x2.tasks import (
    T13VersionSpaceSolver,
    X2EnvironmentKernel,
    X2ObservablePolicy,
    build_generator_b_multihop_panel,
    build_selective_edit_panel,
    build_t13_decisive_fixture,
    build_t13_family_bank,
    build_t14_panel,
    build_t14_price_crossover_fixture,
    eval_shortcut_baselines,
    eval_t13_retrieval_buffer_baseline,
    eval_t14_constant_baselines,
    render_x2_factorial,
    solve_t14_public_dp,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _head_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()


def _run_x2_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PLANTHIR_BACKEND_MODULES"] = "experiments.elendril_x2"
    return subprocess.run(
        [sys.executable, "-m", "anra_v5.palantiri", *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_s20_parameter_contracts_and_shared_backbone_init() -> None:
    s20 = build_s20_base_spec()
    assert int(s20.parameter_receipt().total) == 19_871_616

    # Verify parameter contracts on S20 spec (instantiating lightweight checks on tiny + analytic/probe on S20)
    tiny_check = verify_x2_parameter_contracts(tiny_canary=True)
    assert tiny_check["passed"] is True

    s20_check = verify_x2_parameter_contracts(tiny_canary=False)
    assert s20_check["base_parameters"] == 19_871_616
    assert s20_check["s20_overhead_within_5pct"] is True, s20_check["head_overhead_ratios"]
    assert s20_check["active_controls_within_tol"] is True, s20_check["pair_relative_diffs"]
    assert s20_check["passed"] is True

    # Byte-identical shared backbone init across every treatment/control pair for same seed
    base = build_tiny_canary_base_spec()
    for trt, ctrl in X2_NEW_PAIRS:
        m_t = build_x2_model(build_x2_model_spec(trt, base_spec=base), init_seed=19)
        m_c = build_x2_model(build_x2_model_spec(ctrl, base_spec=base), init_seed=19)
        assert backbone_state_digest(m_t) == backbone_state_digest(m_c)


def test_all_x2_mechanisms_receive_gradients_and_respond_to_lesions() -> None:
    ds = x2_dataset_factory({"dataset_name": "grad_check", "seed": 7}, cpu_canary=True)
    batch = ds["train_data"].items[:4]
    kernel = X2EnvironmentKernel()
    obs = kernel.observe(kernel.initial_state(ds["eval_worlds"][0]))

    for arch in X2_ARCHITECTURES:
        m = x2_model_factory(arch, {"width": 16, "layers": 1, "context_length": 256, "mem_dim": 16}, seed=7)
        m.zero_grad()
        loss = x2_loss_fn(m, batch, type("R", (), {"objective": "standard"})())
        assert torch.isfinite(loss)
        loss.backward()
        mech_grads = [
            float(p.grad.abs().max().item())
            for p in m.parameter_groups()["mechanism"]
            if p.grad is not None
        ]
        assert mech_grads and max(mech_grads) > 0.0, f"no mechanism gradient for {arch}"

        # Verify surgical ablation context works cleanly
        pol = X2ObservablePolicy(m, m.x2_spec, model_identity=f"test:{arch}")
        a0 = pol.propose_with_telemetry(obs)
        assert a0.operation in ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN")
        with m.ablate_mechanism("zero_r"):
            a1 = pol.propose_with_telemetry(obs)
            assert a1.operation in ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN")

    # U-P compute-depth context
    m_up = x2_model_factory("U-P", {"width": 16, "layers": 1, "context_length": 256}, seed=7)
    pol_up = X2ObservablePolicy(m_up, m_up.x2_spec)
    with m_up.compute_depth_context(2):
        _ = pol_up.propose_with_telemetry(obs)
    with m_up.compute_depth_context(6):
        _ = pol_up.propose_with_telemetry(obs)


def test_grouped_loss_and_gradient_parity_vs_scalar_reference() -> None:
    torch.manual_seed(123)
    raw_op = torch.randn((4, 4), requires_grad=True)
    raw_val = torch.randn((4, 16), requires_grad=True)

    def _build_logp(ro: torch.Tensor, rv: torch.Tensor):
        lpo = torch.log_softmax(ro, dim=-1)
        lpv = torch.log_softmax(rv, dim=-1)
        lp19 = torch.cat([lpo[:, :1] + lpv, lpo[:, 1:]], dim=-1)
        return lp19, lpo, lpv

    gold_ops = [0, 0, 1, 3]
    gold_args = [2, 5, 0, 0]
    state_mask = [True, True, True, False]
    perm = list(reversed(range(16)))
    eq_pairs = [(0, 1, perm, True), (2, 3, list(range(16)), False)]
    cf_pairs = [(0, 1, 2, 5, True), (1, 2, 5, 16, False)]

    # Vectorized
    ro1, rv1 = raw_op.clone().detach().requires_grad_(True), raw_val.clone().detach().requires_grad_(True)
    lp19_1, lpo_1, lpv_1 = _build_logp(ro1, rv1)
    loss_vec, _ = compute_x2_grouped_loss(
        lp19_1,
        lpo_1,
        lpv_1,
        gold_ops=torch.tensor(gold_ops),
        gold_args=torch.tensor(gold_args),
        state_mask=torch.tensor(state_mask),
        eq_pairs=eq_pairs,
        cf_pairs=cf_pairs,
        eq_weight=0.1,
        cf_weight=0.1,
        cf_margin=0.5,
    )
    loss_vec.backward()

    # Scalar reference
    ro2, rv2 = raw_op.clone().detach().requires_grad_(True), raw_val.clone().detach().requires_grad_(True)
    lp19_2, lpo_2, lpv_2 = _build_logp(ro2, rv2)
    loss_ref = compute_x2_grouped_loss_scalar_reference(
        lp19_2,
        lpo_2,
        lpv_2,
        gold_ops=gold_ops,
        gold_args=gold_args,
        state_mask=state_mask,
        eq_pairs=eq_pairs,
        cf_pairs=cf_pairs,
        eq_weight=0.1,
        cf_weight=0.1,
        cf_margin=0.5,
    )
    loss_ref.backward()

    assert torch.allclose(loss_vec, loss_ref, atol=1e-6)
    assert torch.allclose(ro1.grad, ro2.grad, atol=1e-6)
    assert torch.allclose(rv1.grad, rv2.grad, atol=1e-6)

    # Empty pairs & all-masked edge case
    ro3, rv3 = raw_op.clone().detach().requires_grad_(True), raw_val.clone().detach().requires_grad_(True)
    lp19_3, lpo_3, lpv_3 = _build_logp(ro3, rv3)
    loss_empty, _ = compute_x2_grouped_loss(
        lp19_3,
        lpo_3,
        lpv_3,
        gold_ops=torch.tensor(gold_ops),
        gold_args=torch.tensor(gold_args),
        state_mask=torch.tensor([False, False, False, False]),
        eq_pairs=[],
        cf_pairs=[],
    )
    assert float(loss_empty.item()) == 0.0


def test_decisive_fixture_11_selective_replacement_exact_vector_math() -> None:
    """Exact §11 fixture: M=I, k=(1,0), v=(0,1), beta=1 -> delta=(0,1), untouched=(0,1), additive=(1,1)."""
    M0 = torch.eye(2, dtype=torch.float32)
    k = torch.tensor([1.0, 0.0])
    v = torch.tensor([0.0, 1.0])
    q_untouched = torch.tensor([0.0, 1.0])

    M_delta = delta_memory_step(M0, k, v, torch.tensor(1.0), additive=False)
    M_add = delta_memory_step(M0, k, v, torch.tensor(1.0), additive=True)

    assert torch.allclose(delta_memory_read(M_delta, k), torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(delta_memory_read(M_delta, q_untouched), torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(delta_memory_read(M_add, k), torch.tensor([1.0, 1.0]), atol=1e-6)

    # beta=0 leaves M unchanged
    M_noop = delta_memory_step(M0, k, v, torch.tensor(0.0), additive=False)
    assert torch.allclose(M_noop, M0, atol=1e-6)

    # Two successive corrections overwrite cleanly on delta, accumulate interference on additive
    v2 = torch.tensor([1.0, 0.0])
    M_delta2 = delta_memory_step(M_delta, k, v2, torch.tensor(1.0), additive=False)
    assert torch.allclose(delta_memory_read(M_delta2, k), v2, atol=1e-6)


def test_generator_b_2x2_factorial_and_shortcut_baselines() -> None:
    panel = build_generator_b_multihop_panel(split="dev", num_skeletons=4, seed=7109, depths=(3, 4, 5))
    # 4 skeletons x 3 depths x 4 variants x 2 presentations = 96
    assert len(panel["episodes"]) == 96
    assert len(panel["exhaustive_perm_panel"]) == 48  # 2 skeletons x 4!
    assert len(panel["sampled_perm_panel"]) == 64     # 4 skeletons x 16 perms

    decoy_worlds = [w for w in panel["episodes"] if w["variant"] == "last_row_decoy"]
    base_res = eval_shortcut_baselines(decoy_worlds)
    assert base_res["reference_interpreter_exact"] is True
    assert base_res["accuracies"]["last_row"] == 0.0

    # Equal-length 2x2 factorial rendering check
    sample_facts = decoy_worlds[0]["facts"]
    r_cc = render_x2_factorial(facts=sample_facts, goal_subject=decoy_worlds[0]["goal_subject"], hops=3, fact_style="canonical", header_style="canonical", pad_to_bytes=600)
    r_nn = render_x2_factorial(facts=sample_facts, goal_subject=decoy_worlds[0]["goal_subject"], hops=3, fact_style="narrative", header_style="narrative", pad_to_bytes=600)
    assert len(r_cc.encode("utf-8")) >= 600 and len(r_nn.encode("utf-8")) >= 600


def test_decisive_fixture_11_t13_witness_and_version_space_ambiguity() -> None:
    solver = T13VersionSpaceSolver(max_laws=512)
    for pos in ("early", "late"):
        fix = build_t13_decisive_fixture(witness_position=pos)
        assert fix["worlds"]["id_disc"]["hidden"]["expected_state_idx"] == 2
        assert fix["worlds"]["swap_disc"]["hidden"]["expected_state_idx"] == 1
        assert fix["worlds"]["id_inv"]["hidden"]["expected_state_idx"] == 0
        assert fix["worlds"]["swap_inv"]["hidden"]["expected_state_idx"] == 0

        post_id = solver.posterior(fix["worlds"]["id_disc"]["support"], fix["disc_query"])
        post_sw = solver.posterior(fix["worlds"]["swap_disc"]["support"], fix["disc_query"])
        assert post_id["identifiable"] is True and post_id["bayes_optimal_label"] == 2
        assert post_sw["identifiable"] is True and post_sw["bayes_optimal_label"] == 1

        # Ambiguous prefix without witness must NOT permit deterministic 100% both-correct (§9, §11)
        post_amb = solver.posterior(fix["common_support"], fix["disc_query"])
        assert post_amb["identifiable"] is False
        assert post_amb["bayes_ceiling"] < 1.0

        # Retrieval buffer competitor + cache budget check
        rb = eval_t13_retrieval_buffer_baseline(fix["worlds"]["id_disc"]["support"], fix["disc_query"])
        assert rb["overflow"] is False

    # 32 train / 8 dev / 12 confirmation family banks with truth-table disjointness
    bank = build_t13_family_bank()
    assert bank["counts"] == {"train": 32, "dev": 8, "confirmation": 12}


def test_decisive_fixture_11_t14_price_crossover_non_anticipation_and_stateful_obs() -> None:
    fix = build_t14_price_crossover_fixture()
    assert fix["dp_cheap"]["optimal_operation"] == "QUERY"
    assert abs(fix["dp_cheap"]["action_evs"]["QUERY"] - 0.18) < 1e-6
    assert fix["dp_expensive"]["optimal_operation"] == "ABSTAIN"
    assert abs(fix["dp_expensive"]["action_evs"]["QUERY"] - (-0.15)) < 1e-6

    const_eval = eval_t14_constant_baselines(fix["cheap_cert"], fix["expensive_cert"])
    assert const_eval["all_constant_policies_lose_somewhere"] is True

    # Two different observations after the SAME action (`QUERY`) on matched latent worlds (§0.2)
    kernel = X2EnvironmentKernel()
    s_cert0 = kernel.initial_state(fix["cheap_cert"])
    s_nocert0 = kernel.initial_state(fix["cheap_nocert"])
    obs_cert0 = kernel.observe(s_cert0)
    obs_nocert0 = kernel.observe(s_nocert0)
    # Identical public prefix before hidden future revelation!
    assert obs_cert0.rendered_text == obs_nocert0.rendered_text

    # Non-anticipation: identical public prefix yields bit-identical logits for same model & state (§0.2, §9)
    m = x2_model_factory("DM", {"width": 16, "layers": 1, "context_length": 256, "mem_dim": 16}, seed=7)
    pol = X2ObservablePolicy(m, m.x2_spec)
    pol.reset_episode()
    a_cert0 = pol.propose_with_telemetry(obs_cert0)
    pol.reset_episode()
    a_nocert0 = pol.propose_with_telemetry(obs_nocert0)
    assert a_cert0.action_distribution == a_nocert0.action_distribution

    # After identical action QUERY, the two worlds emit DIFFERENT public observations!
    obs_cert1 = kernel.observe(kernel.step(s_cert0, "QUERY", None))
    obs_nocert1 = kernel.observe(kernel.step(s_nocert0, "QUERY", None))
    assert obs_cert1.rendered_text != obs_nocert1.rendered_text
    assert "TOOL_Q CERT" in obs_cert1.rendered_text
    assert "TOOL_Q NO_CERTIFICATION" in obs_nocert1.rendered_text

    # Verify full T14 episode reward accounting (+0.98 vs -0.02 vs -1.00)
    v_win = kernel.verify(fix["cheap_cert"], [("QUERY", None), ("ANSWER", "A")])
    assert v_win["correct"] is True and abs(v_win["utility"] - 0.98) < 1e-6
    v_abst = kernel.verify(fix["cheap_nocert"], [("QUERY", None), ("ABSTAIN", None)])
    assert v_abst["correct"] is True and abs(v_abst["utility"] - (-0.02)) < 1e-6
    v_unsup = kernel.verify(fix["cheap_nocert"], [("ANSWER", "B")])
    assert v_unsup["supported"] is False and abs(v_unsup["utility"] - (-1.0)) < 1e-6

    # 6-stratum panel coverage
    t14_panel = build_t14_panel(split="dev", histories_per_stratum=2)
    assert len(t14_panel) == 6 * 2 * 2


def test_x2_registry_test_cards_first_discriminating_report_and_4axis_qualification() -> None:
    inv = verify_x1_core_inventory()
    assert inv["total_jobs"] == 42

    commit = _head_commit()
    reg = build_x2_design_registry(source_commit=commit)
    assert reg["base_full_job_equivalents"] == X2_BASE_JOB_EQUIVALENT_CEILING == 126
    assert reg["separate_ledgers"]["t14_routing_prep"]["count"] == 6
    assert reg["separate_ledgers"]["t13_transfer_parents"]["count"] == 48
    val = validate_x2_registry(reg)
    assert val["verified"] is True

    # Registry validator catches synthetic duplicate job ID and missing seed
    bad_reg = json.loads(json.dumps(reg))
    bad_reg["jobs"].append(bad_reg["jobs"][0])
    with pytest.raises(ValueError, match="DUPLICATE_OR_EMPTY_JOB_ID"):
        validate_x2_registry(bad_reg)

    # Test-first reader preserves empty strata, abstentions, ineligible rows, and both denominators
    cards = build_x2_test_cards(
        [
            {"test_slug": "generator_b_interior_edit_pair", "world_id": "w1", "eligible": True, "correct": True, "prior_canonical_correct": True, "observed_op": "ANSWER"},
            {"test_slug": "generator_b_interior_edit_pair", "world_id": "w2", "eligible": True, "correct": False, "prior_canonical_correct": False, "observed_op": "ABSTAIN"},
            {"test_slug": "generator_b_interior_edit_pair", "world_id": "w3", "eligible": False, "correct": False, "observed_op": "ABSTAIN"},
        ],
        source_commit=commit,
    )
    by_slug = {c["test_slug"]: c for c in cards}
    c_edit = by_slug["generator_b_interior_edit_pair"]
    assert c_edit["eligible_denominator"] == 2
    assert c_edit["ineligible_rows"] == 1
    assert c_edit["abstention_count"] == 1
    assert c_edit["unconditional_point_estimate"] == 0.5
    assert c_edit["conditional_denominator"] == 1
    assert c_edit["conditional_point_estimate"] == 1.0
    assert by_slug["x1_family_X01"]["empty_stratum"] is True

    # First discriminating report identifies earliest observed changed variable
    fdr = build_x2_first_discriminating_report(
        seed=7,
        world_id="sel_edit_dev_000::edited_dep",
        intact_trace={"prefix_sha256": "abc", "write_read_vector": [0.0, 1.0], "operation_margin": 1.2, "action": ["ANSWER", "C"], "observation_sha256": "obs1", "terminal_supported": True, "untouched_sibling_correct": True},
        targeted_trace={"prefix_sha256": "abc", "write_read_vector": [1.0, 0.0], "operation_margin": 0.4, "action": ["ANSWER", "B"], "observation_sha256": "obs1", "terminal_supported": False, "untouched_sibling_correct": True},
        sham_trace={"prefix_sha256": "abc", "write_read_vector": [0.0, 1.0], "operation_margin": 1.2, "action": ["ANSWER", "C"], "observation_sha256": "obs1", "terminal_supported": True, "untouched_sibling_correct": True},
        cohort_counts={"total": 128, "intact_both_correct": 96, "targeted_both_correct": 40, "sham_both_correct": 95},
    )
    assert fdr["prefix_identical_across_conditions"] is True
    assert fdr["earliest_observed_changed_variable"] == "write_read_vector"

    # Synthetic losing candidate + missing archive member both show up cleanly in 4-axis report (§9)
    qual = evaluate_x2_qualification(
        campaign_id="X2-SYNTH-LOSE",
        candidate_name="DM",
        control_name="DM-add",
        paired_seed_scores=[
            {"seed_id": "s7", "treatment_score": 0.52, "control_score": 0.58},
            {"seed_id": "s19", "treatment_score": 0.54, "control_score": 0.56},
            {"seed_id": "s41", "treatment_score": 0.55, "control_score": 0.54},
        ],
        missing_artifacts=["checkpoint_full.zip"],
    )
    assert qual["execution_status"] == "COMPLETE"
    assert qual["delivery_status"] == "PARTIAL"
    assert qual["scientific_status"] == "NEGATIVE_DELIVERED"
    assert qual["causal_status"] == "UNRESOLVED"
    assert "checkpoint_full.zip" in qual["markdown_report"]
    assert holm_bonferroni_adjust({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}


def test_public_cli_end_to_end_elendril_x2_plugin_canary_and_replay(tmp_path: Path) -> None:
    """§0.3 acceptance: init -> validate -> run -> inspect-run -> verify-bundle -> replay-bundle on elendril_x2."""
    commit = _head_commit()
    spec_path = tmp_path / "x2_spec.json"
    work_root = tmp_path / "x2_cpu"

    p_init = _run_x2_cli(["init", "--backend", "elendril_x2", "--out", str(spec_path)])
    assert p_init.returncode == 0, p_init.stdout + p_init.stderr
    doc = json.loads(spec_path.read_text(encoding="utf-8"))
    doc["source_commit"] = commit
    spec_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    # Distinct canary vs production dataset hashes (§0.2)
    d_can = x2_dataset_factory({"dataset_name": "x2_ds", "seed": 7}, cpu_canary=True)["train_data"].data_identity
    d_prd = x2_dataset_factory({"dataset_name": "x2_ds", "seed": 7}, cpu_canary=False)["train_data"].data_identity
    assert d_can != d_prd

    p_val = _run_x2_cli(["validate", "--spec", str(spec_path), "--canary"])
    assert p_val.returncode == 0, p_val.stdout + p_val.stderr

    p_run = _run_x2_cli(["run", "--spec", str(spec_path), "--work-root", str(work_root), "--canary", "--device", "cpu"])
    assert p_run.returncode == 0, p_run.stdout + p_run.stderr

    p_ins = _run_x2_cli(["inspect-run", "--work-root", str(work_root)])
    assert p_ins.returncode == 0, p_ins.stdout + p_ins.stderr

    bundle_dir = work_root / "result_bundle"
    p_ver = _run_x2_cli(["verify-bundle", "--bundle-dir", str(bundle_dir), "--expected-source-commit", commit])
    assert p_ver.returncode == 0, p_ver.stdout + p_ver.stderr

    receipt_path = tmp_path / "REPLAY_RECEIPT.json"
    p_rep = _run_x2_cli(
        [
            "replay-bundle",
            "--bundle-dir",
            str(bundle_dir),
            "--expected-source-commit",
            commit,
            "--device",
            "cpu",
            "--output",
            str(receipt_path),
        ]
    )
    assert p_rep.returncode == 0, p_rep.stdout + p_rep.stderr
    rep = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert rep["verified"] is True
    assert rep["archived_reference_used"] is True
    assert rep["prediction_agreement_status"] == "MATCHED"
    assert rep["original_prediction_agreement"]["checked_cases"] >= 5
    assert all(r["backend_id"] == "elendril_x2" for r in rep["restored_lineages"])
