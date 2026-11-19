from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from anra_v5.elendril.representation import UTF8
from anra_v5.planthir import (
    BytePromptTrainer,
    CampaignOrchestrator,
    GenericTrainRecipe,
    SessionJob,
    export_session_archives,
    frame_byte_prompt_sample,
)
from experiments.avendi_y3.models import (
    AvendiByteModel,
    P6AdaptivePolicy,
    _P4Base,
    apply_bounded_weight_update,
    get_byte_model_spec,
    p4_transition_loss,
    run_p4_causal_suite,
    run_p7_donor_audit_and_interventions,
    select_p4_matched_widths,
    verify_u19_gradient_reversal,
)
from experiments.avendi_y3.module import (
    evaluate_with_confirmation_lock,
    run_avendi_y3_campaign,
    write_confirmation_lock,
)
from experiments.avendi_y3.plan import build_plan
from experiments.avendi_y3.tasks import (
    P5_CONDITIONS,
    audit_case_evidence_corruptions,
    audit_split_custody,
    build_interchange_pairs,
    build_one_hop_families,
    build_p5_executive_worlds,
    build_p6_blocks,
    evaluate_p5_reference_policies,
    evaluate_shortcuts_and_destructive_masks,
    law_bank,
    lookup_cube,
    oracle_traverse_tokens,
    public_action_values,
    update_rule_belief,
)


def test_campaign_denominators():
    plan = build_plan()
    jobs = plan["training_jobs"]
    ids = {j["job_id"] for j in jobs}
    assert len(jobs) == len(ids) == 54
    assert len(plan["short_fork_jobs"]) == 24 and plan["update_totals"]["short_forks"] == 5376
    assert len(plan["frozen_lineages"]) == 6
    assert set(plan["reused_scale_controls"]) <= ids
    assert [s["study_id"] for s in plan["studies"]] == [f"U{i:02}" for i in range(1, 51)]
    assert len(plan["platform_uses"]) == 44 and len(plan["y1_discovery_links"]) == 25 and len(plan["defect_register"]) == 4
    assert all(j["status"] == "PLANNED" for j in jobs)


def test_lookup_and_positional_shortcut_on_the_same_cube():
    trials = list(lookup_cube())
    assert len({(rows, key) for rows, key, _ in trials}) == 8
    assert sum(dict(rows)[key] == gold for rows, key, gold in trials) == 8
    assert sum(rows[-1][1] == gold for rows, _, gold in trials) == 4
    assert sum(rows[0][1] == gold for rows, _, gold in trials) == 4
    with pytest.raises(ValueError):
        list(lookup_cube(("A", "A")))


def test_exact_version_space_and_unseen_transfer():
    laws, tensor = law_bank()
    assert set(map(tuple, laws)) == set(product(range(4), repeat=4))
    np.testing.assert_array_equal(tensor.argmax(axis=2), laws)
    assert tensor.sum(axis=2).min() == tensor.sum(axis=2).max() == 1
    pair = laws[[0, 5]]  # 0000 and 0011 agree at 0, differ at both 2 and 3
    prior = np.array([0.5, 0.5])
    np.testing.assert_allclose(update_rule_belief(prior, pair, 0, 0), prior)
    for true_law in (0, 1):
        posterior = update_rule_belief(prior, pair, 2, pair[true_law, 2])
        assert posterior[pair[:, 3] == pair[true_law, 3]].sum() == 1.0
    with pytest.raises(ValueError, match="impossible"):
        update_rule_belief(prior, pair, 0, 3)


def test_public_bellman_against_enumerated_one_tool_policy_trees():
    rewards = np.array([[1, -2], [-2, 1], [0, 0]])
    prior = np.array([0.5, 0.5])
    observations = np.array([np.eye(2), [[0.5, 0.5], [0.5, 0.5]]])
    costs = [0.1, 0.05]
    values = public_action_values(prior, rewards, observations, costs)
    direct = rewards @ prior
    for t, cost in enumerate(costs):
        tree_values = [
            sum(prior[s] * observations[t, s, o] * rewards[policy[o], s] for s, o in product(range(2), repeat=2)) - cost
            for policy in product(range(3), repeat=2)
        ]
        assert values[len(direct) + t] == pytest.approx(max(tree_values))
    np.testing.assert_allclose(values, [-0.5, -0.5, 0, 0.9, -0.05])


def test_two_tool_horizon_and_consumption():
    states = list(product(range(2), repeat=2))
    truth = np.array([a ^ b for a, b in states])
    rewards = np.stack((np.where(truth == 0, 1, -2), np.where(truth == 1, 1, -2), np.zeros(4)))
    signals = np.eye(2)[np.array(states).T]
    for horizon, expected in [(0, -np.inf), (1, -0.1), (2, 0.8), (3, 0.8)]:
        values = public_action_values([0.25] * 4, rewards, signals, [0.1, 0.1], horizon)
        np.testing.assert_allclose(values[-2:], [expected, expected])


def test_p0_oracle_custody_shortcuts_corruptions_and_grad_sign():
    tr = build_one_hop_families("train", 4, 3301, max_cells_per_family=4)
    dv = build_one_hop_families("dev", 4, 3302, max_cells_per_family=8)
    cf = build_one_hop_families("confirm", 4, 3303, max_cells_per_family=4)
    assert all(oracle_traverse_tokens(UTF8.frame(c["public_prompt"], c["gold"]).ids) == c["gold"] for f in dv for c in f["cells"])
    cust = audit_split_custody({"train": tr, "dev": dv, "confirm": cf})
    assert cust["passed"] and cust["planted_leak_caught"]
    sc = evaluate_shortcuts_and_destructive_masks(dv)
    assert sc["oracle_all_families_ok"] and sc["all_shortcuts_zero_families"] and sc["opposing_truth_masked_views"] > 0
    recs = [{"world_id": f["cells"][0]["world_id"], "tokens": list(UTF8.frame(f["cells"][0]["public_prompt"], f["cells"][0]["gold"]).ids), "pred": f["cells"][0]["gold"]} for f in dv]
    assert audit_case_evidence_corruptions(recs)["passed"]
    u19 = verify_u19_gradient_reversal(AvendiByteModel(get_byte_model_spec("S3", cpu_canary=True), 7), dv[0]["cells"][:2])
    assert u19["passed"] and u19["model_grad_cosine"] <= -0.9


def test_p4_transition_interchange_suffix_rewire_and_x1_parity():
    widths = select_p4_matched_widths(32)
    assert widths["categorical"]["width"] == 32 and widths["dense"]["width"] > 0
    m_cat = _P4Base("categorical", widths["categorical"]["width"], depth=3, seed=7)
    pairs = build_interchange_pairs(6, 3302, depth=3)
    for p in pairs:
        assert p["donor_step1"] != p["recipient_step1"]
        assert p["donor_on_recipient_answer"] not in (p["recipient_answer"], p["donor_own_answer"])
        assert p["rewired_donor_on_recipient_answer"] != p["donor_on_recipient_answer"]
    loss_inter = p4_transition_loss(m_cat, [{"rows": pairs[0]["recipient_rows"], "query": pairs[0]["query"], "gold": pairs[0]["recipient_answer"], "intermediate": pairs[0]["recipient_step1"], "depth": 3}], intermediate_weight=0.1)
    loss_end = p4_transition_loss(m_cat, [{"rows": pairs[0]["recipient_rows"], "query": pairs[0]["query"], "gold": pairs[0]["recipient_answer"], "intermediate": pairs[0]["recipient_step1"], "depth": 3}], intermediate_weight=0.0)
    assert float(loss_inter.detach().item()) >= 0.0 and float(loss_end.detach().item()) >= 0.0
    causal = run_p4_causal_suite(m_cat, pairs)
    assert causal["identity_preserved"] == pytest.approx(1.0) and causal["distinguishable_pairs"] == 6
    p7 = run_p7_donor_audit_and_interventions(cpu_canary=True)
    assert p7["cymek"]["same_shape_mismatch_rejected"] and p7["arkenstone"]["cap_enforced"]
    x1 = p7["elendril"]["x1_parity"]
    assert x1["overhead_check"]["passed"] and x1["overhead_check"]["backbone_digest_match"]


def test_p5_executive_dp_and_p6_stateful_reset_and_bounded_adaptation():
    worlds = build_p5_executive_worlds("dev", 14, 3302)
    assert {w["condition"] for w in worlds} == set(P5_CONDITIONS)
    assert all(w["irrelevant_posterior_shift"] == pytest.approx(0.0) for w in worlds)
    refs = evaluate_p5_reference_policies(worlds)
    assert refs["dp_optimal"]["max_regret"] <= 1e-8 and refs["dp_optimal"]["macro_optimal_rate"] == pytest.approx(1.0)
    assert refs["always_abstain"]["sufficient_false_abstention"] == pytest.approx(1.0)
    b_tr, b_dv, b_cf = build_p6_blocks("train", 8, 3301), build_p6_blocks("dev", 8, 3302), build_p6_blocks("confirm", 8, 3303)
    assert not ({tuple(b["law_indices"]) for b in b_tr} & {tuple(b["law_indices"]) for b in b_dv} & {tuple(b["law_indices"]) for b in b_cf})
    pol = P6AdaptivePolicy(32, 7)
    pol.write_feedback(0, 2)
    pol.clear_context()
    assert int(pol.mem_used.sum()) == 1
    pol.reset_episode()
    assert int(pol.mem_used.sum()) == 0
    pol.write_feedback(0, 2)
    pol.ablate_memory("erase")
    assert int(pol.mem_used.sum()) == 0
    snap = {k: v.clone() for k, v in pol.state_dict().items()}
    _, stats = apply_bounded_weight_update(pol, 0, 2, lr=0.5, max_rel_norm=0.001)
    assert stats["applied_norm"] <= 0.001 * stats["parent_norm"] + 1e-8 and stats["rollback_verified"]
    assert all(torch.equal(pol.state_dict()[k], snap[k]) for k in snap)


def test_confirmation_lock_and_bounded_campaign_receipts(tmp_path: Path):
    lock_p = tmp_path / "LOCK.json"
    sha = write_confirmation_lock(lock_p, {"arms": ["ce", "margin"], "seeds": [7, 19, 41]})
    doc = json.loads(lock_p.read_text())
    doc["payload"]["arms"].append("tampered")
    lock_p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="CONFIRMATION_LOCK_MISMATCH"):
        evaluate_with_confirmation_lock(lock_p, sha, lambda p: p)
    res = run_avendi_y3_campaign(mode="CANARY", work_root=tmp_path / "camp")
    assert res["studies_executed"] == 50 and res["platform_uses_executed"] == 44 and res["rescore_verified"]
    assert res["summary_bytes"] <= 5 * 1024 * 1024
    cov = json.loads((tmp_path / "camp" / "PLATFORM_COVERAGE.json").read_text())
    assert len(cov["uses"]) == 44 and all(u["status"] != "READY" for u in cov["uses"])


def test_recovery_1_mid_update_interrupt_parity(tmp_path: Path):
    from v5_training.checkpoint import InjectedCrash
    spec = get_byte_model_spec("S3", cpu_canary=True)
    samp = [frame_byte_prompt_sample("ROWS: t=01 A->C ; t=02 B->D || Q:A", "C", rival_answer="D", task_id=f"t{i}", world_id=f"w{i}") for i in range(4)]
    recipe = GenericTrainRecipe(seed=7, total_updates=4, batch_size=2, micro_batch_size=2)
    t_full = BytePromptTrainer(samp, spec=spec, recipe=recipe, source_commit="0" * 40, root=tmp_path / "full", lineage="rec1", data_provenance={"s": "t"}, device="cpu", cpu_canary=True)
    losses_full = [t_full.step()["loss"] for _ in range(4)]
    t_int = BytePromptTrainer(samp, spec=spec, recipe=recipe, source_commit="0" * 40, root=tmp_path / "int", lineage="rec1", data_provenance={"s": "t"}, device="cpu", cpu_canary=True)
    t_int.step(); t_int.step(); t_int.save(); t_int.step()
    with pytest.raises(InjectedCrash):
        t_int.save(inject_crash_at="after_publish_before_pointer")
    t_res = BytePromptTrainer(samp, spec=spec, recipe=recipe, source_commit="0" * 40, root=tmp_path / "int", lineage="rec1", data_provenance={"s": "t"}, device="cpu", cpu_canary=True, resume=True)
    assert t_res.state.global_update == 2
    assert [t_res.step()["loss"], t_res.step()["loss"]] == pytest.approx(losses_full[2:])
    assert t_res.state.global_update == 4 and t_res.rng_bytes() == t_full.rng_bytes()
    assert all(torch.equal(t_res.model.state_dict()[k], t_full.model.state_dict()[k]) for k in t_full.model.state_dict())


def test_recovery_2_strict_resume_rejects_swapped_spec_and_missing_member(tmp_path: Path):
    orch = CampaignOrchestrator(campaign_id="Y3-REC2", source_commit="a" * 40, work_dir=tmp_path / "rec2")
    zp = tmp_path / "ok.zip"
    with zipfile.ZipFile(zp, "w") as zf: zf.writestr("FINAL_STATUS.json", '{"status":"COMPLETE"}\n')
    zsha = sha256(zp.read_bytes()).hexdigest()
    reg = [{"job_id": "j0", "source_commit": "a" * 40, "spec_sha256": "s0", "updates": 100}, {"job_id": "j1", "source_commit": "a" * 40, "spec_sha256": "s1", "updates": 100}]
    with pytest.raises(ValueError, match="SWAPPED_SPEC_SHA256|MISSING_ARCHIVE_MEMBER"):
        orch.reconcile_verified_resume(
            [{"job_id": "j0", "source_commit": "a" * 40, "spec_sha256": "WRONG", "completed_updates": 100, "eval_rows": 8, "archive_path": str(zp), "archive_sha256": zsha},
             {"job_id": "j1", "source_commit": "a" * 40, "spec_sha256": "s1", "completed_updates": 100, "eval_rows": 8, "archive_path": str(zp), "archive_sha256": zsha, "required_archive_members": ["MISSING.json"]}],
            reg, required_archive_members=("FINAL_STATUS.json",), fail_on_rejection=True,
        )


def test_recovery_3_failed_worker_preserves_peer_and_partial_export(tmp_path: Path):
    peer_out = tmp_path / "peer_ok.json"
    def _runner(job, env, cmd):
        if job.worker_id == "j_fail": return {"exit_code": 1, "stdout": "", "stderr": "boom"}
        peer_out.write_text('{"peer":"ok"}\n', encoding="utf-8")
        return {"exit_code": 0, "stdout": "ok", "stderr": ""}
    orch = CampaignOrchestrator(campaign_id="Y3-REC3", source_commit="a" * 40, work_dir=tmp_path / "rec3", in_process_runner=_runner)
    jobs = [SessionJob("j_ok", "ce", 7, 7, 2), SessionJob("j_fail", "margin", 7, 7, 2)]
    wres = orch.run_waves(jobs, command_builder=lambda j, wr: ("python", "-m", "w", "--work-root", str(wr), "--device", j.device), devices=("cuda:0", "cuda:1"))
    assert peer_out.is_file() and wres["completed_count"] == 1 and wres["status"] != "COMPLETE"
    part = export_session_archives(tmp_path, tmp_path / "partial_arch", prefix="Y3P", summary_files={"peer_ok.json": peer_out}, analysis_files={"peer_ok.json": peer_out}, status="PARTIAL", source_commit="a" * 40)
    assert part["verified"] and part["final_status"]["status"] == "PARTIAL"
