"""Unit, adversarial, and end-to-end tests for the Elendril Hitchhiker experiment suite."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from anra_v5.elendril import hitchhiker_diagnostics as hdiag
from anra_v5.elendril.cognition import CognitiveState, OperationCandidate
from anra_v5.elendril.hitchhiker_data import (
    COVERAGE_PATTERNS,
    ExternalSlotCanonicalizer,
    HitchhikerSurface,
    V6Dataset,
    apply_case_permutation,
    arm_comparison_table,
    assert_all_stage_disjoint,
    build_counterfactual_suite,
    build_health_probes,
    build_structural_holdout_suite,
    build_training_stream,
    entity_bijection,
    public_case_for,
    render_hitchhiker_public,
    score_health_probes,
    score_missing_bridge_probe,
    validate_stream_contract,
    verify_counterfactual_suite,
)
from anra_v5.elendril.hitchhiker_diagnostics import (
    ANSWER_TOKEN_IDS,
    TOKEN_ABSTAIN,
    TOKEN_DELIBERATE,
    TOKEN_Q,
    TOKEN_R,
    AvailabilityOnlyMaskPolicy,
    ExternalSlotCanonicalizerPolicy,
    GrammarControlHitchhikerPolicy,
    RawHitchhikerPolicy,
    _apply_availability_mask,
    admit_frozen_checkpoint,
    admit_historical_s20_checkpoint,
    classify_diagnostic_outcomes,
    inspect_first_byte_distribution,
    score_entity_equivariance,
)
from anra_v5.elendril.hitchhiker_evaluation import (
    REQUIRED_FORMATION_COHORTS,
    build_experiment_manifest,
    build_mechanism_decision_table,
    evaluate_counterfactual_suite,
    evaluate_formation_gate,
    evaluate_hitchhiker,
    evaluate_health_probes,
    evaluate_original_kinds,
    evaluate_screening_gate,
    holm_adjust,
    independent_rescore_hitchhiker,
    paired_world_bootstrap_contrast,
)
from anra_v5.elendril.hitchhiker_protocol import (
    ALL_ARMS,
    AUDITED_V717_COMMIT,
    HISTORICAL_S20_LOWLR_MANIFESTS,
    PROTOCOL_SHA,
    protocol,
    resolve_scale_profile,
    stage_jobs,
    validate_job,
    validate_stage_selection,
)
from anra_v5.elendril.hitchhiker_training import (
    aux_penalty_gradient_probe,
    create_hitchhiker_trainer,
    resolve_exposure_schedule,
    run_hitchhiker_updates,
)
from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.elendril.training import MODEL, configure_cpu
from anra_v5.elendril.v6_data import SYMBOLS, make_dataset
from anra_v5.elendril.v6_evaluation import ControlPolicy
from tools import elendril_hitchhiker_campaign as hh_campaign
from tools.build_elendril_hitchhiker_notebook import build as build_hh_notebook
from tools.elendril_hitchhiker_export import restore_hitchhiker_archive
from tools.elendril_hitchhiker_runner import (
    _evaluate_backend_decoders,
    main as runner_main,
    resolve_executing_source_commit,
    run_cpu_mechanical_check,
    run_verify_suite,
    verify_and_rescore_stage0_package,
    write_stage0_evidence_package,
)


def _tiny_utf8_spec():
    return replace(MODEL, vocabulary_size=UTF8.vocabulary_size, context_length=512)


def test_protocol_and_scale_registry():
    p = protocol()
    assert p["schema"] == "elendril-hitchhiker-protocol-v1"
    assert len(PROTOCOL_SHA) == 64
    assert len(stage_jobs("stage1_s3_screen")) == 15
    assert len(stage_jobs("stage0_checkpoint_audit")) == 9
    assert len(stage_jobs("stage2_followups")) == 12
    assert len(stage_jobs("stage3_scale_confirmation", winning_arm="AB")) == 12
    assert set(ALL_ARMS) >= {"B0", "A", "B", "AB", "C", "AB+C", "C+penalty", "D-external"}
    assert set(HISTORICAL_S20_LOWLR_MANIFESTS) == {7, 19, 41}

    job = stage_jobs("stage1_s3_screen")[0]
    validate_job(job, "stage1_s3_screen")
    with pytest.raises(ValueError):
        validate_job({**job, "lr": 1e-3}, "stage1_s3_screen")
    with pytest.raises(ValueError):
        stage_jobs("stage3_scale_confirmation", winning_arm="B0")

    with pytest.raises(ValueError):
        resolve_scale_profile("S9")

    tiny = _tiny_utf8_spec()
    receipt = {
        "spec": tiny,
        "parameters": tiny.parameter_receipt().total,
        "model_sha256": tiny.sha256(),
    }
    resolved = resolve_scale_profile("S9", receipt)
    assert resolved.sha256() == tiny.sha256()
    with pytest.raises(ValueError):
        resolve_scale_profile("S9", {**receipt, "parameters": receipt["parameters"] + 1})
    with pytest.raises(ValueError):
        resolve_scale_profile("S9", {**receipt, "model_sha256": "0" * 64})


def test_stage_selection_validation():
    assert validate_stage_selection("stage1_s3_screen", None)["status"] == "ADMITTED"

    with pytest.raises(ValueError):
        validate_stage_selection("stage2_followups", None)
    with pytest.raises(ValueError):
        validate_stage_selection("stage2_followups", {"stage1_completed": False, "admitted_followups": ["AB+C"]})
    with pytest.raises(ValueError):
        validate_stage_selection("stage2_followups", {"stage1_completed": True, "admitted_followups": ["UNKNOWN"]})
    s2 = validate_stage_selection(
        "stage2_followups",
        {"stage1_completed": True, "admitted_followups": ["AB+C", "D-external"], "rationale": "coverage test"},
        winning_arm="AB+C",
    )
    assert s2["status"] == "ADMITTED" and s2["jobs"] == 3

    with pytest.raises(ValueError):
        validate_stage_selection(
            "stage3_scale_confirmation",
            {"screening_status": "INCONCLUSIVE", "split": "validation", "winning_arm": "AB", "model_profile": "S8"},
        )
    with pytest.raises(ValueError):
        validate_stage_selection(
            "stage3_scale_confirmation",
            {"screening_status": "PASS_SCREENING", "split": "confirmation", "winning_arm": "AB", "model_profile": "S8"},
        )
    with pytest.raises(ValueError):
        validate_stage_selection(
            "stage3_scale_confirmation",
            {"screening_status": "PASS_SCREENING", "split": "validation", "winning_arm": "B0", "model_profile": "S8"},
        )
    with pytest.raises(ValueError):
        validate_stage_selection(
            "stage3_scale_confirmation",
            {"screening_status": "PASS_SCREENING", "split": "validation", "winning_arm": "AB", "model_profile": "S3"},
        )
    s3 = validate_stage_selection(
        "stage3_scale_confirmation",
        {"screening_status": "PASS_SCREENING", "split": "validation", "winning_arm": "AB", "model_profile": "S8"},
        winning_arm="AB",
    )
    assert s3["status"] == "ADMITTED" and s3["jobs"] == 12


def test_entity_permutation_and_counterfactual_suite():
    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=2, split="validation")
    check = verify_counterfactual_suite(suite)
    assert check["status"] == "PASS"

    sample_case = suite["cohorts"]["hop-2"][0]
    mapping, inverse = entity_bijection(("test", sample_case.case_id))
    permuted = apply_case_permutation(sample_case, mapping)
    restored = apply_case_permutation(permuted, inverse)
    assert restored.goal.subject == sample_case.goal.subject
    assert restored.goal.slots == sample_case.goal.slots
    assert restored.environment.initial == sample_case.environment.initial
    assert restored.truth.expected == sample_case.truth.expected

    pub = public_case_for(sample_case)
    assert not hasattr(pub, "truth")
    assert not hasattr(pub, "environment")
    assert pub.memory is None or type(pub.memory) is bool

    by_pair = {}
    for cohort_name, cases in suite["cohorts"].items():
        if cohort_name.startswith("factor-"):
            grouped = {}
            for c in cases:
                grouped.setdefault(c.pair_id, []).append(c)
            by_pair[cohort_name] = list(grouped.values())

    for a, b in by_pair["factor-wording"]:
        assert a.environment.initial == b.environment.initial
        assert a.truth.expected == b.truth.expected
        assert a.goal.family != b.goal.family
    for a, b in by_pair["factor-timestamps"]:
        assert a.truth.expected == b.truth.expected
        assert b.environment.initial[0].tick - a.environment.initial[0].tick == 8000
    for trio in by_pair["factor-row-order"]:
        assert len({c.truth.expected for c in trio}) == 1
    for a, b in by_pair["factor-distractors"]:
        assert a.truth.expected == b.truth.expected
        assert len(b.environment.initial) - len(a.environment.initial) == 3
    for a, b in by_pair["factor-answer-edge-swap"]:
        assert a.truth.expected != b.truth.expected
    for a, b in by_pair["factor-answer-query-swap"]:
        assert a.environment.initial == b.environment.initial
        assert a.goal.subject != b.goal.subject
        assert a.truth.expected != b.truth.expected

    for quad in by_pair["factor-alias-clause"]:
        panels = {c.panel: c for c in quad}
        s_alias = CognitiveState(panels["surface-alias-with-clause"].goal).observe(
            panels["surface-alias-with-clause"].environment.initial
        )
        s_neut = CognitiveState(panels["surface-neutral-length-matched"].goal).observe(
            panels["surface-neutral-length-matched"].environment.initial
        )
        p_alias = render_hitchhiker_public(s_alias, (), panels["surface-alias-with-clause"].surface, False)
        p_neut = render_hitchhiker_public(s_neut, (), panels["surface-neutral-length-matched"].surface, False)
        assert len(p_alias.encode("utf-8")) == len(p_neut.encode("utf-8"))

    teacher_eval = evaluate_counterfactual_suite(suite, ControlPolicy("fixed"))
    assert all(m["accuracy"] == 1.0 for m in teacher_eval["cohort_metrics"].values())
    oracle_eval = evaluate_counterfactual_suite(suite, ControlPolicy("oracle"))
    assert all(m["accuracy"] == 1.0 for m in oracle_eval["cohort_metrics"].values())


def test_health_probes_cross_source_revision_and_shared_node_distractors():
    probes = build_health_probes(seed=27173, worlds_per_probe=3, split="validation")
    teacher_res = evaluate_health_probes(probes, ControlPolicy("fixed"), stage_records=None)
    scores = teacher_res["scores"]
    assert scores["distractor_trap"]["accuracy"] == 1.0
    assert scores["belief_revision"]["both_correct_timestamp_pairs"] == 3
    assert scores["belief_revision"]["both_correct_cross_source_pairs"] == 3
    assert scores["missing_bridge"]["combined_gate_pass"] is True
    assert scores["preservation"]["status"] == "NOT_QUALIFIED"

    # Confirm cross-source unresolvable vs resolvable and shared-node distractors exist cleanly.
    cross_panels = {c.panel for c in probes["belief-revision"]}
    assert "belief-revision-cross-source-unresolvable" in cross_panels
    assert "belief-revision-cross-source-resolvable" in cross_panels
    unres = [c for c in probes["belief-revision"] if c.panel == "belief-revision-cross-source-unresolvable"][0]
    res = [c for c in probes["belief-revision"] if c.panel == "belief-revision-cross-source-resolvable"][0]
    assert unres.truth.expected is None
    assert res.truth.expected is not None

    # Shared-node distractor traps include an event touching a chain vertex + stale queried value.
    trap = probes["distractor-trap"][0]
    assert len(trap.environment.initial) == 8
    ticks = [e.tick for e in trap.environment.initial]
    assert max(ticks) - min(ticks) >= 8
    assert any(sum(1 for e in trap.environment.initial if e.subject == node or e.value == node) >= 3 for node in {e.subject for e in trap.environment.initial})


def test_split_custody_instance_vs_topology_and_structural_holdout():
    val = build_counterfactual_suite(seed=17071, worlds_per_hop=16, split="validation")
    conf = build_counterfactual_suite(seed=17072, worlds_per_hop=16, split="confirmation")
    struct = build_structural_holdout_suite(seed=37073, worlds_per_topology=4)
    train_base = make_dataset("train", target_bytes=20_000, seed=606)

    res = assert_all_stage_disjoint(train_base, val, conf, structural_cohorts=struct)
    assert res["status"] == "PASS"
    assert "validation_vs_confirmation" in res["split_claims"]

    # Teacher scores 100% on topology-disjoint structural holdout cohorts.
    struct_eval = evaluate_counterfactual_suite(struct, ControlPolicy("fixed"))
    assert all(m["accuracy"] == 1.0 for m in struct_eval["cohort_metrics"].values())

    # Exact Codex counterexample 1: relabeled training case leaked into validation.
    leaked_case = train_base.cases[0]
    mapping, _ = entity_bijection(("leak", leaked_case.case_id))
    relabeled = apply_case_permutation(leaked_case, mapping)
    bad_val = V6Dataset(
        split="validation",
        cases=(relabeled,),
        examples=(),
        samples=(),
        manifest={"version": "leak-check"},
    )
    with pytest.raises(ValueError):
        assert_all_stage_disjoint(train_base, bad_val)

    # Exact Codex counterexample 2: validation sibling copied or permuted into confirmation.
    val_case = val["cohorts"]["hop-1"][0]
    val_perm = apply_case_permutation(val_case, entity_bijection(("val-to-conf", val_case.case_id))[0])
    bad_conf = V6Dataset(
        split="confirmation",
        cases=(replace(val_perm, case_id="fake-conf", world_id="fake-conf-world"),),
        examples=(),
        samples=(),
        manifest={"version": "val-conf-leak"},
    )
    with pytest.raises(ValueError):
        assert_all_stage_disjoint(train_base, val, bad_conf)


def test_materialized_stream_schedule_and_arm_isolation(tmp_path: Path):
    configure_cpu()
    train_base = make_dataset("train", target_bytes=20_000, seed=606)
    val = build_counterfactual_suite(seed=17071, worlds_per_hop=2, split="validation")
    s1_jobs = {j["arm"]: j for j in stage_jobs("stage1_s3_screen") if j["seed"] == 7}
    s2_jobs = {j["arm"]: j for j in stage_jobs("stage2_followups") if j["seed"] == 7}

    # Materialized fresh stream (AB) uses stream['total_updates'] directly without * epochs.
    ab_stream = build_training_stream(
        "AB",
        base_data=train_base,
        total_updates=4,
        batch_size=16,
        data_seed=606,
        order_seed=7,
        heldout_suites=(val,),
    )
    assert ab_stream["use_static_indexer"] is False
    assert len(ab_stream["samples"]) == 64
    assert resolve_exposure_schedule(s1_jobs["AB"], stream=ab_stream, batch_size=16) == 4
    validate_stream_contract(ab_stream, expected_arm="AB", total_updates=4, batch_size=16)

    # Reject updates_override in production and short/mismatched materialized streams.
    with pytest.raises(ValueError):
        resolve_exposure_schedule(s1_jobs["AB"], stream=ab_stream, batch_size=16, updates_override=2, is_cpu_canary=False)
    with pytest.raises(ValueError):
        resolve_exposure_schedule(s1_jobs["AB"], stream={**ab_stream, "samples": ab_stream["samples"][:-1]}, batch_size=16)
    with pytest.raises(ValueError):
        validate_stream_contract(ab_stream, expected_arm="A", total_updates=4, batch_size=16)

    # C static pool matches len(base_samples), uses_static_indexer=True, and blocked sampling.
    expected_c_updates = math.ceil(len(train_base.samples) / 16) * s1_jobs["C"]["epochs"]
    c_stream = build_training_stream(
        "C",
        base_data=train_base,
        total_updates=expected_c_updates,
        batch_size=16,
        data_seed=606,
        order_seed=7,
        heldout_suites=(val,),
    )
    assert c_stream["use_static_indexer"] is True
    assert len(c_stream["samples"]) == len(train_base.samples)
    assert c_stream["sampling_method"] == "blocked"
    assert c_stream["coverage_recipe"] == "14-stratum-coverage"
    assert resolve_exposure_schedule(s1_jobs["C"], stream=c_stream, batch_size=16) == expected_c_updates

    # AB+C fresh relabeled stream shares identical COVERAGE_PATTERNS.
    abc_stream = build_training_stream(
        "AB+C",
        base_data=train_base,
        total_updates=4,
        batch_size=16,
        data_seed=606,
        order_seed=7,
        heldout_suites=(val,),
    )
    assert abc_stream["use_static_indexer"] is False
    assert abc_stream["coverage_recipe"] == c_stream["coverage_recipe"] == "14-stratum-coverage"
    assert resolve_exposure_schedule(s2_jobs["AB+C"], stream=abc_stream, batch_size=16) == 4

    b0_stream = build_training_stream("B0", base_data=train_base, total_updates=4, batch_size=16, order_seed=7)
    comp = arm_comparison_table({"B0": b0_stream, "C": c_stream, "AB": ab_stream, "AB+C": abc_stream})
    by_arm = {row["arm"]: row for row in comp}
    assert set(by_arm) == {"B0", "C", "AB", "AB+C"}
    assert all(row.get("deltas_vs_b0") is not None for row in comp)


def test_historical_s20_admission_and_tamper_rejection(tmp_path: Path, monkeypatch):
    configure_cpu()
    missing_dir = tmp_path / "absent"
    pending = admit_historical_s20_checkpoint(missing_dir, 7)
    assert pending["status"] == "CHECKPOINT_AUDIT_PENDING"
    assert pending["fabricate_permitted"] is False
    assert not missing_dir.exists()

    with pytest.raises(ValueError):
        admit_historical_s20_checkpoint(missing_dir, 999)

    # Create a real local CheckpointStore artifact & bind a patched historical manifest to test positive + tamper paths.
    tiny = _tiny_utf8_spec()
    job = stage_jobs("stage1_s3_screen")[0]
    train_base = make_dataset("train", target_bytes=8_000, seed=606)
    ckpt_root = tmp_path / "hist_ckpts"
    trainer, _ = create_hitchhiker_trainer(
        job,
        source_commit=AUDITED_V717_COMMIT,
        root=ckpt_root,
        base_data=train_base,
        device="cpu",
        cpu_canary=True,
        spec_override=tiny,
        updates_override=1,
        batch_override=4,
    )
    trainer.metadata["recipe"]["epochs"] = job["epochs"]
    trainer.metadata["provenance"].update({
        "v717_protocol_sha256": PROTOCOL_SHA,
        "v7_sampling_method": "blocked",
        "seed": 606,
        "target_bytes": 8000,
        "v717_job": {
            "job_id": job["job_id"],
            "profile": "S3",
            "lr": trainer.recipe.lr,
            "seed": 7,
            "order_seed": 7,
            "epochs": job["epochs"],
            "method": "blocked",
            "mechanism": "ce",
        },
    })
    trainer.step()
    sha = trainer.save()
    monkeypatch.setitem(
        hdiag.HISTORICAL_S20_LOWLR_MANIFESTS,
        7,
        {
            **HISTORICAL_S20_LOWLR_MANIFESTS[7],
            "lineage": job["job_id"],
            "job_id": job["job_id"],
            "profile": "S3",
            "checkpoint_sha256": sha,
            "source_commit": AUDITED_V717_COMMIT,
            "lr": trainer.recipe.lr,
            "epochs": job["epochs"],
            "batch_size": trainer.recipe.batch_size,
            "micro_batch_size": trainer.recipe.micro_batch_size,
            "expected_update": 1,
            "protocol_sha256": PROTOCOL_SHA,
            "train_seed": 606,
            "train_bytes": 8000,
        },
    )
    monkeypatch.setattr(hdiag, "resolve_scale_profile", lambda _p: tiny)
    admitted = admit_historical_s20_checkpoint(ckpt_root, 7)
    assert admitted["status"] == "ADMITTED"
    assert admitted["checkpoint_sha256"] == sha

    # Reject wrong expected_data_sha, wrong expected_update, wrong lr, corrupt LATEST.
    with pytest.raises(ValueError):
        admit_historical_s20_checkpoint(ckpt_root, 7, expected_data_sha="0" * 64)
    (ckpt_root / job["job_id"] / "LATEST").write_text(("f" * 64) + "\n", encoding="ascii")
    with pytest.raises(ValueError):
        admit_historical_s20_checkpoint(ckpt_root, 7)
    (ckpt_root / job["job_id"] / "LATEST").write_text("corrupt!\n", encoding="ascii")
    with pytest.raises(ValueError):
        admit_historical_s20_checkpoint(ckpt_root, 7)


def test_decoders_first_byte_and_stage0_package_roundtrip(tmp_path: Path):
    configure_cpu()
    tiny = _tiny_utf8_spec()
    job = stage_jobs("stage1_s3_screen")[0]
    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=1, split="validation")
    probes = build_health_probes(seed=27173, worlds_per_probe=2, split="validation")
    train_base = make_dataset("train", target_bytes=8_000, seed=606)
    orig_dev = make_dataset("validation", target_bytes=6_000, seed=707)

    trainer, _ = create_hitchhiker_trainer(
        job,
        source_commit=AUDITED_V717_COMMIT,
        root=tmp_path / "ck",
        base_data=train_base,
        heldout_suites=(suite,),
        device="cpu",
        cpu_canary=True,
        spec_override=tiny,
        updates_override=1,
        batch_override=4,
    )
    backend = trainer.backend
    diag = _evaluate_backend_decoders(
        backend,
        suite,
        probes,
        original_dev=orig_dev,
        full_260_inspection=True,
    )
    suite_n = sum(len(v) for v in suite["cohorts"].values())
    probe_n = sum(len(v) for v in probes.values())
    assert len(diag["full_first_byte_inspections"]) == suite_n + probe_n

    pkg_dir = tmp_path / "stage0_pkg"
    pkg_receipt = write_stage0_evidence_package(
        pkg_dir,
        executing_source_commit=resolve_executing_source_commit(),
        historical_seed=7,
        admission_receipt={"status": "LOCAL_TEST", "manifest": {"source_commit": AUDITED_V717_COMMIT}},
        diag=diag,
        suite_case_count=suite_n,
        probe_case_count=probe_n,
    )
    res = verify_and_rescore_stage0_package(pkg_dir, suite=suite, probes=probes, original_dev=orig_dev)
    assert res["status"] == "PASS"
    assert res["verified_cases"] == suite_n + probe_n

    # Tampering first_byte_distributions must fail verification.
    fb_path = pkg_dir / "first_byte_distributions.json"
    fb_doc = json.loads(fb_path.read_text(encoding="utf-8"))
    fb_doc["cases"] = fb_doc["cases"][:4]
    fb_path.write_text(json.dumps(fb_doc), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_and_rescore_stage0_package(pkg_dir, suite=suite, probes=probes, original_dev=orig_dev)


def test_fail_closed_screening_and_formation_gates():
    orig_ok = {k: 1.0 for k in ("known", "missing", "conflict", "null", "unavailable", "route", "memory")}

    def mk_seed(acc: float, worlds: int = 256, *, split: str = "validation", origin: str = "model") -> dict:
        eps = [{"world_id": f"w{i}", "case_id": f"c{i}", "panel": "base", "outcome": {"correct": i < int(acc * worlds)}} for i in range(worlds)]
        return {
            "split": split,
            "protocol_sha256": PROTOCOL_SHA,
            "attribution_origin": origin,
            "decoder_schema": "utf8-raw",
            "original_kind_accuracies": orig_ok,
            "records": {"factor-entity-permutation": {"metrics": {"accuracy": acc}, "episodes": eps}},
        }

    # Complete 3-seed >=256-world validation screen passes.
    pass_gate = evaluate_screening_gate(
        {s: mk_seed(1.0) for s in (7, 19, 41)},
        {s: mk_seed(0.0) for s in (7, 19, 41)},
    )
    assert pass_gate["status"] == "PASS_SCREENING"
    assert pass_gate["holm_role"] == "secondary_exploratory_holm_adjusted"

    # Codex counterexample: 8 worlds or 1 seed or missing original kinds returns INCOMPLETE_NOT_QUALIFIED.
    assert evaluate_screening_gate({7: mk_seed(1.0, worlds=8)}, {7: mk_seed(0.0, worlds=8)})["status"] == "INCOMPLETE_NOT_QUALIFIED"
    no_orig = mk_seed(1.0)
    del no_orig["original_kind_accuracies"]
    assert evaluate_screening_gate({7: no_orig, 19: mk_seed(1.0), 41: mk_seed(1.0)}, {s: mk_seed(0.0) for s in (7, 19, 41)})["status"] == "INCOMPLETE_NOT_QUALIFIED"

    # Regressing any original task kind triggers FALSIFIED.
    reg_seed = mk_seed(1.0)
    reg_seed["original_kind_accuracies"] = {**orig_ok, "known": 0.5}
    assert evaluate_screening_gate({7: reg_seed, 19: mk_seed(1.0), 41: mk_seed(1.0)}, {s: mk_seed(0.0) for s in (7, 19, 41)})["status"] == "FALSIFIED"

    # Codex formation counterexamples:
    # 1. Summary-only fake worlds_per_hop=512 without episodes/12 cohorts/model_identity -> NOT_QUALIFIED.
    fake_summary = {
        "cohort_metrics": {c: {"accuracy": 0.99, "invalid_rate": 0.0} for c in REQUIRED_FORMATION_COHORTS},
        "worlds_per_hop": 512,
    }
    assert evaluate_formation_gate(fake_summary, equivariance_report={"argmax_equivariance_rate": 0.99})["status"] == "NOT_QUALIFIED"

    # 2. Construct valid episode-backed 512-world confirmation suite_eval + bound equivariance_report.
    def mk_conf_eval(*, worlds: int = 512, split: str = "confirmation", origin: str = "model", identity: str = "ckpt-abc") -> tuple[dict, dict]:
        records = {}
        for cname in REQUIRED_FORMATION_COHORTS:
            eps = []
            for w in range(worlds):
                if cname in ("factor-answer-edge-swap", "factor-answer-query-swap"):
                    for p_idx in range(2):
                        eps.append({
                            "world_id": f"{cname}-w{w}",
                            "pair_id": f"pair-{w}",
                            "case_id": f"c-{w}-{p_idx}",
                            "outcome": {"correct": True},
                            "trace": {"status": "completed", "model_identity": identity},
                        })
                else:
                    eps.append({
                        "world_id": f"{cname}-w{w}",
                        "case_id": f"c-{w}",
                        "outcome": {"correct": True},
                        "trace": {"status": "completed", "model_identity": identity},
                    })
            records[cname] = {
                "independent_rescore_status": "PASS",
                "split": split,
                "protocol_sha256": PROTOCOL_SHA,
                "attribution_origin": origin,
                "decoder_schema": "utf8-raw",
                "model_identity": identity,
                "cases_sha256": "sha-perm-123" if cname == "factor-entity-permutation" else None,
                "episodes": eps,
            }
        suite_eval = {
            "protocol_sha256": PROTOCOL_SHA,
            "split": split,
            "attribution_origin": origin,
            "decoder_schema": "utf8-raw",
            "model_identity": identity,
            "records": records,
        }
        eq = {
            "model_identity": identity,
            "cases_sha256": "sha-perm-123",
            "complete_pairs": worlds,
            "argmax_equivariance_rate": 0.99,
        }
        return suite_eval, eq

    full_eval, full_eq = mk_conf_eval(worlds=512)
    assert evaluate_formation_gate(full_eval, equivariance_report=full_eq)["status"] == "FORMATION_QUALIFIED"

    # Wrong split at full 512 scale -> NOT_QUALIFIED; wrong attribution_origin -> NOT_QUALIFIED; mismatched equivariance -> NOT_QUALIFIED.
    val_512, val_eq = mk_conf_eval(worlds=512, split="validation")
    assert evaluate_formation_gate(val_512, equivariance_report=val_eq)["status"] == "NOT_QUALIFIED"
    eng_512, eng_eq = mk_conf_eval(worlds=512, origin="engineered_canonicalizer")
    assert evaluate_formation_gate(eng_512, equivariance_report=eng_eq)["status"] == "NOT_QUALIFIED"
    assert evaluate_formation_gate(full_eval, equivariance_report={**full_eq, "model_identity": "other"})["status"] == "NOT_QUALIFIED"
    small_eval, small_eq = mk_conf_eval(worlds=16)
    assert evaluate_formation_gate(small_eval, equivariance_report=small_eq)["status"] == "MECHANICAL_ONLY_NOT_FORMATION_QUALIFIED"


def test_runner_cli_and_source_separation(tmp_path: Path):
    verify_out = tmp_path / "verify_suite.json"
    assert runner_main(["--mode", "verify-suite", "--worlds-per-hop", "1", "--output", str(verify_out)]) == 0
    verify_payload = json.loads(verify_out.read_text(encoding="utf-8"))
    assert verify_payload["suite_verification"]["status"] == "PASS"
    assert verify_payload["audited_v717_ancestor"] == AUDITED_V717_COMMIT
    assert verify_payload["executing_source_commit"] == resolve_executing_source_commit()
    assert "arm_comparison_table" in verify_payload
    assert "teacher_original_kind_accuracies" in verify_payload

    stage0_out = tmp_path / "stage0_audit.json"
    assert runner_main(["--mode", "stage0-audit", "--worlds-per-hop", "1", "--output", str(stage0_out)]) == 0
    stage0_payload = json.loads(stage0_out.read_text(encoding="utf-8"))
    assert stage0_payload["stage0_checkpoint_audit"]["status"] == "CHECKPOINT_AUDIT_PENDING"
    assert stage0_payload["audited_v717_ancestor"] == AUDITED_V717_COMMIT

    mech = run_cpu_mechanical_check(worlds_per_hop=2)
    assert mech["claim_status"] == "IMPLEMENTED"
    assert mech["mechanical_checks"]["local_stage0_package_roundtrip"]["verification"]["status"] == "PASS"
    assert mech["mechanical_checks"]["screening_gate_mechanical"]["status"] == "INCOMPLETE_NOT_QUALIFIED"
    assert mech["mechanical_checks"]["formation_gate_mechanical"]["status"] == "MECHANICAL_ONLY_NOT_FORMATION_QUALIFIED"


def test_campaign_dry_run_storage_forecast_canary_e2e_and_notebook(tmp_path: Path, monkeypatch):
    head = resolve_executing_source_commit()

    # 1. Invalid commit rejected cleanly by real check_source.
    with pytest.raises(ValueError):
        hh_campaign.run(output=tmp_path / "bad_src", source="0" * 40, phase="stage1", owner_enabled=True)

    # Allow campaign preflight in a dirty dev checkout after verifying check_source failure above.
    monkeypatch.setattr(hh_campaign, "check_source", lambda _repo, _src: None)

    # 2. Dry run validates cleanly without GPUs.
    dry = hh_campaign.run(output=tmp_path / "dry", source=head, phase="stage1", dry_run=True)
    assert dry["status"] == "DRY_RUN_READY"
    assert dry["job_count"] == 15

    # 3. Storage NO_GO fails cleanly before running anything.
    FakeUsage = type("Usage", (), {"total": 10 << 30, "used": 9 << 30, "free": 1 << 20})
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: FakeUsage())
    with pytest.raises(ValueError, match="NO_GO storage admission"):
        hh_campaign.storage_admission(tmp_path / "low_disk", 14 << 30)
    monkeypatch.undo()
    monkeypatch.setattr(hh_campaign, "check_source", lambda _repo, _src: None)

    # 4. Over-budget minutes triggers clean NO_GO forecast on Stage 1.
    out_nogo = tmp_path / "budget_nogo"
    nogo = hh_campaign.run(
        output=out_nogo,
        source=head,
        phase="stage1",
        minutes=0.0001,
        worlds_per_hop=2,
        train_bytes=12_000,
        updates_override=2,
        cpu_canary=True,
    )
    assert nogo["status"] == "NO_GO"
    assert nogo["forecast"]["projected_wall_seconds"] > 0.0

    # 5. End-to-end canary Stage 1: interrupt after update 1 -> restore FULL archive -> resume -> COMPLETE.
    out_run1 = tmp_path / "canary_run1"
    part1 = hh_campaign.run(
        output=out_run1,
        source=head,
        phase="stage1",
        minutes=60.0,
        worlds_per_hop=2,
        train_bytes=12_000,
        updates_override=2,
        cpu_interrupt_after=1,
        cpu_canary=True,
    )
    assert part1["status"] == "INTERRUPTED"
    full_zip = Path(part1["delivery"]["exports"]["full"]["zip_path"])
    assert full_zip.is_file()

    restored_root = tmp_path / "canary_restored"
    restore_receipt = restore_hitchhiker_archive(full_zip, restored_root, source=head, phase="stage1")
    assert restore_receipt["status"] == "EXTRACTION_VERIFIED"

    resumed = hh_campaign.run(
        output=restored_root,
        source=head,
        phase="stage1",
        minutes=60.0,
        worlds_per_hop=2,
        train_bytes=12_000,
        updates_override=2,
        cpu_canary=True,
    )
    assert resumed["status"] == "COMPLETE"
    assert resumed["independent_replay"]["status"] == "PASS"
    assert Path(resumed["delivery"]["exports"]["analysis"]["zip_path"]).is_file()
    assert Path(resumed["delivery"]["exports"]["full"]["zip_path"]).is_file()
    assert (restored_root / "analysis" / "selection_receipt.json").is_file()
    assert (restored_root / "exports" / "FINAL_STATUS.json").is_file()

    # 6. Notebook builder contract check.
    nb = build_hh_notebook(head)
    assert nb["metadata"]["source_reference"] == head
    assert len(nb["cells"]) >= 4
    with pytest.raises(ValueError):
        build_hh_notebook("not-a-sha")


def test_decoders_availability_mask_and_trainer_resume_and_canonicalizer(tmp_path: Path):
    configure_cpu()
    tiny = _tiny_utf8_spec()
    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=1, split="validation")
    train_base = make_dataset("train", target_bytes=8_000, seed=606)

    # Availability mask preserves all tokens except unavailable R/Q.
    raw_logits = torch.linspace(-2.0, 2.0, UTF8.vocabulary_size)
    raw_logits[TOKEN_R] = 9.0
    raw_logits[TOKEN_Q] = 8.0
    masked = _apply_availability_mask(raw_logits, memory_available=False, query_available=False)
    assert masked[TOKEN_R] == float("-inf") and masked[TOKEN_Q] == float("-inf")
    for tid in (*ANSWER_TOKEN_IDS, TOKEN_ABSTAIN, TOKEN_DELIBERATE, UTF8.eos_id):
        assert masked[tid].item() == raw_logits[tid].item()

    # Aux gradient probe + exact resume parity on AB.
    c_pen = [j for j in stage_jobs("stage2_followups") if j["arm"] == "C+penalty" and j["seed"] == 7][0]
    tr_pen, _ = create_hitchhiker_trainer(
        c_pen, source_commit=AUDITED_V717_COMMIT, root=tmp_path / "pen",
        base_data=train_base, heldout_suites=(suite,), device="cpu",
        cpu_canary=True, spec_override=tiny, updates_override=2, batch_override=4,
    )
    assert aux_penalty_gradient_probe(tr_pen)["status"] == "PASS"

    job = [j for j in stage_jobs("stage1_s3_screen") if j["arm"] == "AB" and j["seed"] == 7][0]
    full_tr, _ = create_hitchhiker_trainer(
        job, source_commit=AUDITED_V717_COMMIT, root=tmp_path / "full_ab",
        base_data=train_base, heldout_suites=(suite,), device="cpu",
        cpu_canary=True, spec_override=tiny, updates_override=2, batch_override=4,
    )
    full_res = run_hitchhiker_updates(full_tr, save_every=1)
    full_sha = full_tr.store.latest_sha256()

    part1, _ = create_hitchhiker_trainer(
        job, source_commit=AUDITED_V717_COMMIT, root=tmp_path / "split_ab",
        base_data=train_base, heldout_suites=(suite,), device="cpu",
        cpu_canary=True, spec_override=tiny, updates_override=2, batch_override=4,
    )
    part1.step()
    part1.save()
    part2, _ = create_hitchhiker_trainer(
        job, source_commit=AUDITED_V717_COMMIT, root=tmp_path / "split_ab",
        base_data=train_base, heldout_suites=(suite,), resume=True, device="cpu",
        cpu_canary=True, spec_override=tiny, updates_override=2, batch_override=4,
    )
    res_res = run_hitchhiker_updates(part2)
    assert part2.save() == full_sha
    assert res_res["final_loss"] == pytest.approx(full_res["final_loss"], rel=0.0, abs=0.0)

    # External canonicalizer attribution & isomorphic permutation canonical prompt identity.
    dext_job = [j for j in stage_jobs("stage2_followups") if j["arm"] == "D-external" and j["seed"] == 7][0]
    dext_tr, _ = create_hitchhiker_trainer(
        dext_job, source_commit=AUDITED_V717_COMMIT, root=tmp_path / "dext",
        base_data=train_base, heldout_suites=(suite,), device="cpu",
        cpu_canary=True, spec_override=tiny, updates_override=1, batch_override=4,
    )
    factory = ExternalSlotCanonicalizerPolicy(dext_tr.backend, assignment="first-mention", slot_seed=17)
    rec = evaluate_hitchhiker(suite["cohorts"]["factor-entity-permutation"], factory, split="validation")
    assert rec["attribution_origin"] == "engineered_canonicalizer"
    assert independent_rescore_hitchhiker(suite["cohorts"]["factor-entity-permutation"], rec) == rec["metrics"]

    canon = ExternalSlotCanonicalizer(assignment="first-mention", strip_aliases=True)
    orig, perm = suite["cohorts"]["factor-entity-permutation"][:2]
    m_o, _ = canon.build_mapping(CognitiveState(orig.goal).observe(orig.environment.initial), orig.surface)
    m_p, inv_p = canon.build_mapping(CognitiveState(perm.goal).observe(perm.environment.initial), perm.surface)
    s_o, surf_o, _ = canon.canonicalize_public(CognitiveState(orig.goal).observe(orig.environment.initial), orig.surface, m_o)
    s_p, surf_p, _ = canon.canonicalize_public(CognitiveState(perm.goal).observe(perm.environment.initial), perm.surface, m_p)
    assert render_hitchhiker_public(s_o, (), surf_o, False) == render_hitchhiker_public(s_p, (), surf_p, False)
    assert canon.dereference_output("ANSWER", m_o[orig.truth.expected], inv_p) == ("ANSWER", perm.truth.expected)
