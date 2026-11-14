"""CPU tests for V8 training, crash recovery, evaluation, pointer interventions, and conditional modules."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import signal
import pytest
import torch

from anra_v5.elendril.codec import digest
from anra_v5.elendril.foundation_training import Recipe, StageTransition
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.analysis import (
    evaluate_advancement_gate,
    pair_condition_runs,
    paired_world_comparison,
)
from anra_v5.elendril.v8.contracts import PublicInput, build_v8_model_spec
from anra_v5.elendril.v8.data import D1_FAMILIES, build_d1_dev_suite, build_d1_training_dataset
from anra_v5.elendril.v8.evaluation import (
    diagnostic_probe,
    evaluate_a20_recovery_extension,
    evaluate_f03_reverse_extension,
    evaluate_worlds_by_mode,
    run_pointer_interventions,
    summarize_suite_episodes,
)
from anra_v5.elendril.v8.memory import evaluate_e16_memory_regimes
from anra_v5.elendril.v8.models import build_v8_model
from anra_v5.elendril.v8.protocol import jobs
from anra_v5.elendril.v8.replay import (
    GuardianMonitorState,
    ReplayQualification,
    build_stratified_support_index,
    sample_fixed_replay_batch,
    verify_retention_prerequisites,
)
from anra_v5.elendril.v8.teacher import TeacherStep
from anra_v5.elendril.v8.training import V8TypedTrainer, create_v8_trainer, prepare_v8_training_data
from anra_v5.elendril.v8.transfer import (
    evaluate_transfer_gain,
    right_censored_acquisition,
    verify_transfer_prerequisites,
)
from tools.elendril_v8_campaign import (
    _resolve_condition_comparison_pairs,
    validate_job_completion_artifacts,
)
import tools.elendril_v8_worker as v8_worker
from v5_training.checkpoint import InjectedCrash


def _tiny_base():
    return replace(
        MODEL_PROFILES["S20"],
        width=32,
        layers=1,
        query_heads=2,
        kv_heads=1,
        head_dimension=16,
        ffn_width=64,
        context_length=512,
    )


def test_v8_cpu_rejects_oversized_s20_and_supports_d0_d1_accum_and_migration(tmp_path: Path) -> None:
    d1 = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=16)
    s20_p = build_v8_model_spec("P")
    with pytest.raises(ValueError, match="tiny fixture bounds"):
        create_v8_trainer(
            d1,
            jobs("reader-screen")[0],
            source="a" * 40,
            root=tmp_path / "oversized",
            device="cpu",
            cpu_canary=True,
            v8_spec_override=s20_p,
            updates_override=2,
        )

    tiny = _tiny_base()
    # B0 D0 & D1
    d0 = prepare_v8_training_data("D0", sample_limit_for_cpu=16)
    for j, d in [(jobs("data-control")[0], d0), (jobs("data-control")[3], d1)]:
        tr = create_v8_trainer(
            d,
            j,
            source="a" * 40,
            root=tmp_path / f"b0_{j['dataset']}",
            device="cpu",
            cpu_canary=True,
            v8_spec_override=build_v8_model_spec("B0", base_spec=tiny),
            updates_override=2,
        )
        tr.step()
        tr.step()
        tr.save()

    # Gradient accumulation parity: micro=1 vs micro=8 on P
    spec_p = build_v8_model_spec("P", base_spec=tiny, pointer_dim=16)
    r1 = Recipe(seed=7, total_updates=2, batch_size=8, micro_batch_size=1, lr=1e-4, schedule="cosine", precision="fp32")
    r8 = Recipe(seed=7, total_updates=2, batch_size=8, micro_batch_size=8, lr=1e-4, schedule="cosine", precision="fp32")
    t1 = V8TypedTrainer(
        d1.typed_samples, v8_spec=spec_p, recipe=r1, order_seed=7, source_commit="a" * 40,
        root=tmp_path / "acc1", lineage="acc1", data_provenance=d1.manifest, device="cpu", cpu_canary=True,
    )
    t8 = V8TypedTrainer(
        d1.typed_samples, v8_spec=spec_p, recipe=r8, order_seed=7, source_commit="a" * 40,
        root=tmp_path / "acc8", lineage="acc8", data_provenance=d1.manifest, device="cpu", cpu_canary=True,
    )
    t1.step()
    t8.step()
    for k in t1.model.state_dict():
        assert torch.allclose(t1.model.state_dict()[k], t8.model.state_dict()[k], atol=1e-6)


@pytest.mark.parametrize(
    "crash_point",
    [
        "after_component:model.bin",
        "after_manifest",
        "after_stage",
        "after_publish_before_pointer",
        "after_pointer_stage",
    ],
)
def test_v8_interrupted_restore_parity_at_every_crash_point(tmp_path: Path, crash_point: str) -> None:
    d1 = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=16)
    spec = build_v8_model_spec("P", base_spec=_tiny_base(), pointer_dim=16)
    job = jobs("reader-screen")[0]
    src = "a" * 40

    t_cont = create_v8_trainer(
        d1, job, source=src, root=tmp_path / "cont", device="cpu", cpu_canary=True,
        v8_spec_override=spec, updates_override=4,
    )
    for _ in range(4):
        t_cont.step()

    res_root = tmp_path / f"res_{crash_point.replace(':', '_')}"
    t_a = create_v8_trainer(
        d1, job, source=src, root=res_root, device="cpu", cpu_canary=True,
        v8_spec_override=spec, updates_override=4,
    )
    t_a.step()
    t_a.step()
    sha2 = t_a.save()

    with pytest.raises(InjectedCrash):
        t_a.step()
        t_a.save(inject_crash_at=crash_point)

    t_b = create_v8_trainer(
        d1, job, source=src, root=res_root, resume=True, device="cpu", cpu_canary=True,
        v8_spec_override=spec, updates_override=4,
    )
    assert t_b.state.global_update == 2
    assert t_b.parent == sha2
    t_b.step()
    t_b.step()
    for k in t_cont.model.state_dict():
        assert torch.equal(t_cont.model.state_dict()[k], t_b.model.state_dict()[k])


def test_v8_stage_transition_and_mutated_checkpoint_rejection(tmp_path: Path) -> None:
    d1 = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=16)
    spec = build_v8_model_spec("F", base_spec=_tiny_base())
    job = jobs("head-screen")[3]
    src = "a" * 40
    root = tmp_path / "parent"
    tr = create_v8_trainer(
        d1, job, source=src, root=root, device="cpu", cpu_canary=True,
        v8_spec_override=spec, updates_override=2,
    )
    tr.step()
    c1 = tr.save()

    child_rec = Recipe(seed=7, total_updates=2, batch_size=8, micro_batch_size=2, lr=1e-4, schedule="cosine", precision="fp32")
    probe_tr = V8TypedTrainer(
        d1.typed_samples, v8_spec=spec, recipe=child_rec, order_seed=7, source_commit=src,
        root=tmp_path / "probe", lineage="child", data_provenance=d1.manifest, device="cpu", cpu_canary=True,
    )
    parent_rec = json.loads(tr.store.restore(c1)[1]["scheduler.json"])["metadata"]["recipe"]
    trans = StageTransition(
        parent_checkpoint=c1,
        parent_recipe=digest(parent_rec),
        child_recipe=digest(asdict(child_rec)),
        child_data=probe_tr.data_sha,
    )
    child = V8TypedTrainer(
        d1.typed_samples, v8_spec=spec, recipe=child_rec, order_seed=7, source_commit=src,
        root=tmp_path / "child_real", lineage="child", data_provenance=d1.manifest,
        device="cpu", cpu_canary=True, transition=trans, parent_store=tr.store,
    )
    assert len(child.optimizer.state) == 0
    for k in child.model.state_dict():
        assert torch.equal(child.model.state_dict()[k], tr.model.state_dict()[k])

    # Mutate archived object byte and ensure restore rejects it
    obj_model = tr.store.objects / c1 / "model.bin"
    raw = bytearray(obj_model.read_bytes())
    raw[-1] ^= 0xFF
    obj_model.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        tr.store.restore(c1)


def test_v8_eval_pointer_interventions_and_conditional_w8_gates(tmp_path: Path) -> None:
    d1 = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=16)
    spec_p = build_v8_model_spec("P", base_spec=_tiny_base(), pointer_dim=16)
    tr = create_v8_trainer(
        d1, jobs("reader-screen")[0], source="a" * 40, root=tmp_path / "p_eval",
        device="cpu", cpu_canary=True, v8_spec_override=spec_p, updates_override=2,
    )
    tr.step()

    cohorts = build_d1_dev_suite(worlds_per_cohort=2)
    worlds = [w for rows in cohorts.values() for w in rows][:15]
    eps = evaluate_worlds_by_mode(tr.model, spec_p, worlds, model_identity="P")
    summary = summarize_suite_episodes(eps, worlds)
    assert set(summary) == {"raw", "grammar", "availability-mask", "direct-answer-only"}
    d1_raw = build_d1_training_dataset(quota_per_family=16)
    diag = diagnostic_probe(tr.model, spec_p, d1_raw.steps, max_probe_steps=16)
    assert diag["architecture"] == "P"
    interv = run_pointer_interventions(tr.model, spec_p, d1_raw.steps[:32])
    assert set(["baseline_pointer", "no_copy_control", "cyclic_value_permutation", "sham_value_permutation"]).issubset(interv)
    assert evaluate_f03_reverse_extension(tr.model, spec_p, model_identity="P")["scenario"] == "F03"
    assert evaluate_a20_recovery_extension(tr.model, spec_p, worlds, model_identity="P")["scenario"] == "A20"

    comp = paired_world_comparison(eps["raw"], eps["grammar"], arm_a="P-raw", arm_b="P-gram", bootstrap_samples=50)
    assert comp["worlds"] == len(worlds)
    gate_fail = evaluate_advancement_gate(
        [{"paired_gain": 0.02, "candidate_raw_r09": 0.5, "candidate_relevant_cf_both_correct": 0.4, "candidate_irrelevant_cf_stability": 0.8}]
    )
    assert gate_fail["qualified"] is False

    # W8 gates & guardian state
    unqual = verify_retention_prerequisites(
        ReplayQualification("A", 0.70, 0.80, "a" * 40, "b" * 64),
        ReplayQualification("B", 0.85, 0.80, "a" * 40, "c" * 64),
    )
    assert unqual["status"] == "PREREQUISITE_NOT_QUALIFIED"

    g = GuardianMonitorState()
    g.observe_stratum_errors(64, {0: 0.9, 1: 0.1})
    g2 = GuardianMonitorState.from_dict(g.to_dict())
    assert g2.ema_errors == g.ema_errors
    sup = build_stratified_support_index(d1.typed_samples)
    batch = sample_fixed_replay_batch(d1.typed_samples, sup, d1.typed_samples, regime="G", b_indices=(0, 1, 2, 3), step_seed=5, guardian=g2)
    assert len(batch) == 8

    # Frozen-weight E16 check
    assert evaluate_e16_memory_regimes(tr.model, spec_p, worlds[:4], model_identity="P", prerequisite_qualified=False)["status"] == "PREREQUISITE_NOT_QUALIFIED"
    e16 = evaluate_e16_memory_regimes(tr.model, spec_p, worlds[:4], model_identity="P", prerequisite_qualified=True)
    assert e16["frozen_weights_verified"] is True

    # Right-censored C24 transfer check
    assert verify_transfer_prerequisites(retention_qualified=False, c_achievable_accuracy=0.85)["status"] == "PREREQUISITE_NOT_QUALIFIED"
    rc = right_censored_acquisition([{"updates": 64, "accuracy": 0.45}, {"updates": 128, "accuracy": 0.62}], threshold=0.80)
    assert rc["right_censored"] is True and rc["right_censored_at_updates"] == 128
    tg = evaluate_transfer_gain({
        "fresh": [{"updates": 64, "examples": 100, "accuracy": 0.60}, {"updates": 128, "examples": 200, "accuracy": 0.82}],
        "ab_qualified_replay": [{"updates": 64, "examples": 100, "accuracy": 0.84}],
    })
    assert tg["meets_transfer_rule"] is True


def test_v8_r3_sigterm_interrupt_and_resume_preserves_steps_and_initial_milestone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = "a" * 40
    job = jobs("head-screen")[3]  # V8-F-D1-i7-o7
    job_id = str(job["job_id"])
    work_dir = tmp_path / "sigterm_resume"

    orig_save_milestone = v8_worker._save_milestone_file

    def _save_and_sigterm(*args, **kwargs):
        res = orig_save_milestone(*args, **kwargs)
        if kwargs.get("milestone_name") == "epoch-2":
            signal.raise_signal(signal.SIGTERM)
        return res

    monkeypatch.setattr(v8_worker, "_save_milestone_file", _save_and_sigterm)
    partial = v8_worker.run_job_worker(
        job_id=job_id,
        source_commit=src,
        work_dir=work_dir,
        device="cpu",
        cpu_canary=True,
        updates_override=4,
        dev_worlds_per_cohort=1,
        save_every=1,
    )
    assert partial["status"] == "INTERRUPTED_CHECKPOINTED"
    assert partial["completed_updates"] == 2
    assert partial["signal"] == signal.SIGTERM

    initial_path = work_dir / "milestones" / job_id / "initial.json"
    assert initial_path.exists()
    initial_bytes_before = initial_path.read_bytes()

    steps_path = work_dir / "steps" / f"{job_id}.jsonl"
    updates_before = [json.loads(line)["global_update"] for line in steps_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert updates_before == [1, 2]

    monkeypatch.setattr(v8_worker, "_save_milestone_file", orig_save_milestone)
    completed = v8_worker.run_job_worker(
        job_id=job_id,
        source_commit=src,
        work_dir=work_dir,
        device="cpu",
        resume=True,
        cpu_canary=True,
        updates_override=4,
        dev_worlds_per_cohort=1,
        save_every=1,
    )
    assert completed["completed_updates"] == 4
    updates_after = [json.loads(line)["global_update"] for line in steps_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert updates_after == [1, 2, 3, 4]
    assert initial_path.read_bytes() == initial_bytes_before
    violations = validate_job_completion_artifacts(
        job,
        completed,
        work_dir,
        source_commit=src,
        expected_updates=4,
        expected_split_audit_sha256=str(completed["split_audit_sha256"]),
    )
    assert violations == []


def test_v8_r4_diagnostic_probe_exact_joint_action_fixtures() -> None:
    spec_f = build_v8_model_spec("F", base_spec=_tiny_base())
    model = build_v8_model(spec_f, 7, torch_module=torch)
    with torch.no_grad():
        model.op_head.weight.zero_()
        model.val_head.weight.zero_()
        model.op_head.bias.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))  # ANSWER
        model.val_head.bias.copy_(torch.tensor([10.0] + [0.0] * 15))  # A

    def _make_step(idx: int, *, gold_op: str, gold_arg: str | None, target_text: str) -> TeacherStep:
        pub = PublicInput(
            text_utf8=f"Current value of A? Query=1 Memory=0\nA=B@{idx}/s0\nAction:",
            query_available=True,
            memory_available=False,
            action_budget=2,
        )
        return TeacherStep(
            task_id=f"t-{idx}",
            world_id=f"w-{idx}",
            family="direct_one_hop",
            stage="s0",
            public_input=pub,
            gold_op=gold_op,
            gold_arg=gold_arg,
            target_text=target_text,
            prompt_bytes=len(pub.text_utf8.encode("utf-8")),
            is_answer_step=(gold_op == "ANSWER"),
        )

    # 1. Correct ops / wrong answer values
    ans_steps = [
        _make_step(1, gold_op="ANSWER", gold_arg="B", target_text="B"),
        _make_step(2, gold_op="ANSWER", gold_arg="C", target_text="C"),
    ]
    diag_wrong_val = diagnostic_probe(model, spec_f, ans_steps)
    assert diag_wrong_val["op_accuracy"] == pytest.approx(1.0)
    assert diag_wrong_val["val_accuracy_on_answer_steps"] == pytest.approx(0.0)
    assert diag_wrong_val["val_accuracy_given_op_correct"] == pytest.approx(0.0)
    assert diag_wrong_val["exact_action_accuracy"] < 1.0

    # 2. Non-answer steps only (no answer steps)
    with torch.no_grad():
        model.op_head.bias.copy_(torch.tensor([0.0, 10.0, 0.0, 0.0]))  # QUERY
    non_ans_steps = [
        _make_step(3, gold_op="QUERY", gold_arg=None, target_text="Q"),
        _make_step(4, gold_op="QUERY", gold_arg=None, target_text="Q"),
    ]
    diag_non_ans = diagnostic_probe(model, spec_f, non_ans_steps)
    assert diag_non_ans["val_accuracy_on_answer_steps"] is None
    assert diag_non_ans["val_accuracy_given_op_correct"] is None
    assert diag_non_ans["non_answer_op_accuracy"] == pytest.approx(1.0)
    assert diag_non_ans["exact_action_accuracy"] == pytest.approx(1.0)


def test_v8_r5_seed_pairing_data_control_and_advancement_gates() -> None:
    src = "c" * 40
    split_sha = "1" * 64
    seeds = ((7, 7), (19, 19), (41, 41))
    runs: list[dict[str, object]] = []
    for idx, (i_seed, o_seed) in enumerate(seeds):
        runs.append({
            "job_id": f"V8-F-D1-i{i_seed}-o{o_seed}",
            "condition_id": "F-D1",
            "arm": "F",
            "dataset": "D1",
            "init_seed": i_seed,
            "order_seed": o_seed,
            "source_commit": src,
            "split_audit_sha256": split_sha,
            "data_identity": "d1-id",
            "raw_overall_accuracy": 0.80 + 0.02 * idx,
            "raw_r09_accuracy": 0.82 + 0.02 * idx,
            "canonical_raw_accuracy": 0.85,
            "relevant_cf_both_correct": 0.75,
            "irrelevant_cf_stability": 0.94,
        })
        runs.append({
            "job_id": f"V8-J-D1-i{i_seed}-o{o_seed}",
            "condition_id": "J-D1",
            "arm": "J",
            "dataset": "D1",
            "init_seed": i_seed,
            "order_seed": o_seed,
            "source_commit": src,
            "split_audit_sha256": split_sha,
            "data_identity": "d1-id",
            "raw_overall_accuracy": 0.65 + 0.01 * idx,
            "raw_r09_accuracy": 0.68 + 0.01 * idx,
            "canonical_raw_accuracy": 0.86,
            "relevant_cf_both_correct": 0.55,
            "irrelevant_cf_stability": 0.88,
        })

    paired = pair_condition_runs(runs, candidate_condition="F-D1", comparator_condition="J-D1", expected_seed_pairs=seeds)
    shuffled = list(runs)
    random.Random(99).shuffle(shuffled)
    paired_shuffled = pair_condition_runs(shuffled, candidate_condition="F-D1", comparator_condition="J-D1", expected_seed_pairs=seeds)
    assert paired == paired_shuffled

    # Missing, duplicate, and mismatched runs must fail closed with ValueError
    with pytest.raises(ValueError, match="seed parity mismatch"):
        pair_condition_runs(runs[:-1], candidate_condition="F-D1", comparator_condition="J-D1")
    with pytest.raises(ValueError, match="duplicate run"):
        pair_condition_runs(runs + [runs[0]], candidate_condition="F-D1", comparator_condition="J-D1")
    bad_commit_runs = [dict(r) for r in runs]
    bad_commit_runs[0]["source_commit"] = "d" * 40
    with pytest.raises(ValueError, match="source_commit mismatch"):
        pair_condition_runs(bad_commit_runs, candidate_condition="F-D1", comparator_condition="J-D1")
    bad_audit_runs = [dict(r) for r in runs]
    bad_audit_runs[0]["split_audit_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="split_audit_sha256 mismatch"):
        pair_condition_runs(bad_audit_runs, candidate_condition="F-D1", comparator_condition="J-D1")

    # data-control resolves all 3 pairs (B0-D1 vs B0-D0, P-D1 vs P-D0, A-D1 vs A-D0)
    dc_jobs = [
        {"arm": arm, "dataset": ds}
        for arm in ("A", "P", "B0")
        for ds in ("D0", "D1")
    ]
    assert _resolve_condition_comparison_pairs("data-control", dc_jobs) == [
        ("B0-D1", "B0-D0"),
        ("P-D1", "P-D0"),
        ("A-D1", "A-D0"),
    ]

    # Advancement gate uses MEAN (not min) R09 >= 0.80 and rejects canonical_raw_regression > 0.05
    mean_r09_pairs = [
        {
            "paired_gain": 0.14,
            "candidate_raw_r09": r09,
            "candidate_relevant_cf_both_correct": 0.75,
            "candidate_irrelevant_cf_stability": 0.94,
            "canonical_raw_delta": 0.0,
        }
        for r09 in (0.75, 0.82, 0.85)  # min=0.75 < 0.80, mean=0.8067 >= 0.80
    ]
    gate_mean_ok = evaluate_advancement_gate(mean_r09_pairs, confirmation=False)
    assert gate_mean_ok["qualified"] is True
    assert gate_mean_ok["summary"]["min_raw_r09"] == pytest.approx(0.75)
    assert gate_mean_ok["summary"]["mean_raw_r09"] >= 0.80

    regressed_pairs = [{**p, "canonical_raw_delta": -0.06} for p in mean_r09_pairs]
    gate_reg_fail = evaluate_advancement_gate(regressed_pairs, confirmation=False)
    assert gate_reg_fail["qualified"] is False
    assert any("canonical_raw_regression" in r for r in gate_reg_fail["reasons"])

    # Replication confirmation isolation: changing ONLY confirmation_summary flips confirmation gate
    dev_unchanged = {
        "phase": "replication",
        "split_name": "confirmation",
        "is_confirmation": True,
        "seed_count": 3,
        "positive_seed_differences": 3,
        "paired_gains": [0.15, 0.16, 0.17],
        "r09_by_seed": [0.85, 0.86, 0.87],
        "relevant_cf_by_seed": [0.75, 0.78, 0.80],
        "irrelevant_cf_by_seed": [0.92, 0.94, 0.95],
    }
    conf_pass = {
        "pairs": [
            {
                "split_name": "confirmation",
                "paired_gain": 0.15,
                "candidate_raw_r09": 0.85,
                "candidate_relevant_cf_both_correct": 0.76,
                "candidate_irrelevant_cf_stability": 0.93,
            }
            for _ in range(3)
        ]
    }
    conf_fail = {
        "pairs": [
            {
                "split_name": "confirmation",
                "paired_gain": 0.02,
                "candidate_raw_r09": 0.65,
                "candidate_relevant_cf_both_correct": 0.50,
                "candidate_irrelevant_cf_stability": 0.80,
            }
            for _ in range(3)
        ]
    }
    assert evaluate_advancement_gate({**dev_unchanged, "confirmation_summary": conf_pass}, confirmation=True)["qualified"] is True
    assert evaluate_advancement_gate({**dev_unchanged, "confirmation_summary": conf_fail}, confirmation=True)["qualified"] is False
    assert evaluate_advancement_gate({**dev_unchanged, "confirmation_summary": conf_fail}, confirmation=False)["qualified"] is True


def test_v8_r7_initial_milestone_stratified_d1_and_d0_probes(tmp_path: Path) -> None:
    src = "b" * 40
    # D1 job initial milestone: all 7 D1 families + non-null suite_summary
    d1_dir = tmp_path / "d1_job"
    v8_worker.run_job_worker(
        job_id="V8-F-D1-i7-o7",
        source_commit=src,
        work_dir=d1_dir,
        device="cpu",
        cpu_canary=True,
        updates_override=2,
        dev_worlds_per_cohort=1,
    )
    d1_init = json.loads((d1_dir / "milestones" / "V8-F-D1-i7-o7" / "initial.json").read_text(encoding="utf-8"))
    assert set(d1_init["probe"]["families_covered"]) == set(D1_FAMILIES)
    assert d1_init["probe"]["probe_scope"]["dataset"] == "D1"
    assert d1_init["suite_summary"] is not None
    assert d1_init["suite_summary"]["episodes"] > 0
    assert d1_init["suite_summary"]["exact_episode_accuracy"] is not None

    # D0 job initial milestone: D0 corpus covering all 7 D0 families + non-null suite_summary
    d0_dir = tmp_path / "d0_job"
    v8_worker.run_job_worker(
        job_id="V8-B0-D0-i7-o7",
        source_commit=src,
        work_dir=d0_dir,
        device="cpu",
        cpu_canary=True,
        updates_override=2,
        dev_worlds_per_cohort=1,
    )
    d0_init = json.loads((d0_dir / "milestones" / "V8-B0-D0-i7-o7" / "initial.json").read_text(encoding="utf-8"))
    assert set(d0_init["probe"]["families_covered"]) == {
        "conflict", "known", "memory", "missing", "null", "route", "unavailable",
    }
    assert d0_init["probe"]["probe_scope"]["dataset"] == "D0"
    assert d0_init["suite_summary"] is not None
    assert d0_init["suite_summary"]["episodes"] > 0
    assert d0_init["suite_summary"]["exact_episode_accuracy"] is not None
