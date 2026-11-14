"""Unit, adversarial, and end-to-end tests for the Elendril Hitchhiker experiment suite."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest
import torch

from anra_v5.elendril import hitchhiker_diagnostics as hdiag
from anra_v5.elendril.codec import canonical
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
    verify_custody_receipt,
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
    cases_identity,
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
from tools.elendril_hitchhiker_export import handoff_hitchhiker_archive, restore_hitchhiker_archive
from tools.elendril_hitchhiker_runner import (
    _evaluate_backend_decoders,
    main as runner_main,
    resolve_executing_source_commit,
    run_cpu_mechanical_check,
    run_stage0_audit,
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


def _create_temp_upstream_repo(repo_dir: Path) -> tuple[str, str]:
    """Create a local upstream git repo with current workspace tracked files + two distinct commits."""
    repo_root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "-C", str(repo_root), "ls-files"], text=True).splitlines()
    for rel in tracked:
        src = repo_root / rel
        if src.is_file():
            dst = repo_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    subprocess.run(["git", "init", str(repo_dir)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "ci@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "CI"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "pin"], check=True, capture_output=True)
    pin_sha = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    readme = repo_dir / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo_dir), "add", "README.md"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "other"], check=True, capture_output=True)
    other_sha = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    return pin_sha, other_sha


def test_r1_notebook_setup_executes_unmocked_clone_checkout_and_rejects_mismatch_or_dirty(tmp_path: Path):
    from tools.elendril_cloud import check_source

    upstream = tmp_path / "upstream_repo"
    pin_sha, other_sha = _create_temp_upstream_repo(upstream)
    nb = build_hh_notebook(pin_sha)
    setup_src = "".join(nb["cells"][1]["source"]) if isinstance(nb["cells"][1]["source"], list) else nb["cells"][1]["source"]
    run_src = "".join(nb["cells"][2]["source"]) if isinstance(nb["cells"][2]["source"], list) else nb["cells"][2]["source"]
    compile(setup_src, "<notebook-setup>", "exec")
    compile(run_src, "<notebook-run>", "exec")
    assert "sys.executable" in run_src
    assert "PYTHONDONTWRITEBYTECODE" in run_src
    assert "campaign_status.json" in run_src

    work_root = tmp_path / "work_root"
    code_root = tmp_path / "code_root"
    patched = setup_src.replace(
        'REPO_URL = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"',
        f"REPO_URL = {upstream.as_uri()!r}",
    ).replace(
        'WORK_ROOT = Path(f"/kaggle/working/elendril_hitchhiker_{PHASE}")',
        f"WORK_ROOT = Path({str(work_root)!r})",
    ).replace(
        'CODE_ROOT = Path("/kaggle/working/anra_src")',
        f"CODE_ROOT = Path({str(code_root)!r})",
    )
    assert "git_mirror" not in patched
    subprocess.run([sys.executable, "-c", patched], check=True)

    assert (code_root / ".git").exists()
    assert not (work_root / ".git").exists()
    head_checked = subprocess.check_output(["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True).strip()
    assert head_checked == pin_sha
    assert check_source(code_root, pin_sha) is None

    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    if shutil.disk_usage(work_root).free >= (14 << 30):
        dry = subprocess.run(
            [sys.executable, "-m", "tools.elendril_hitchhiker_campaign", "--output", str(work_root), "--source", pin_sha, "--phase", "stage1", "--dry-run"],
            cwd=code_root,
            env=env,
            capture_output=True,
            text=True,
        )
        assert dry.returncode == 0, dry.stderr
        assert json.loads(dry.stdout)["status"] == "DRY_RUN_READY"

    bad_pin = subprocess.run(
        [sys.executable, "-m", "tools.elendril_hitchhiker_campaign", "--output", str(work_root), "--source", other_sha, "--phase", "stage1", "--dry-run"],
        cwd=code_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert bad_pin.returncode != 0

    (code_root / "README.md").write_text("dirty\n", encoding="utf-8")
    dirty_run = subprocess.run(
        [sys.executable, "-m", "tools.elendril_hitchhiker_campaign", "--output", str(work_root), "--source", pin_sha, "--phase", "stage1", "--dry-run"],
        cwd=code_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert dirty_run.returncode != 0
    with pytest.raises(ValueError, match="clean source checkout"):
        check_source(code_root, pin_sha)


def test_r2_forecast_accounts_all_eval_components_and_deadline_interrupts_cleanly(tmp_path: Path):
    cohort_ms = {
        "factor-entity-permutation": 10.0,
        "factor-source-renaming": 12.0,
        "factor-answer-edge-swap": 15.0,
    }
    cohort_counts = {
        "factor-entity-permutation": 512,
        "factor-source-renaming": 512,
        "factor-answer-edge-swap": 512,
    }
    profile_receipt = {
        "schema": "elendril-hitchhiker-profile-v1",
        "status": "PASS",
        "stream_probe": {
            "base_stream_overhead_s": 4.0,
            "ms_per_materialized_update": 50.0,
            "materialized_updates_measured": 4,
        },
        "profiles": {
            "S3": {
                "ms_per_update": 100.0,
                "checkpoint_bytes": 10 * 1024 * 1024,
                "checkpoint_write_per_save_ms": 200.0,
                "eval_components": {
                    "suite": {"cohort_ms_per_case": cohort_ms, "cohort_full_counts": cohort_counts},
                    "orig": {"ms_per_case": 5.0, "full_cases": 200},
                    "probes": {"ms_per_case": 8.0, "full_cases": 384},
                    "structural": {"ms_per_case": 9.0, "full_cases": 256},
                    "equivariance": {"ms_per_case": 10.0, "full_cases": 512},
                },
            },
        },
    }
    prepared_manifest = {
        "val_suite_cohort_counts": cohort_counts,
        "conf_suite_cohort_counts": cohort_counts,
        "val_orig_cases": 200,
        "val_probes_cases": 384,
        "struct_suite_cases": 256,
        "val_equivariance_cases": 512,
        "conf_equivariance_cases": 512,
    }
    jobs = [
        {"job_id": "J1", "stage": "stage1_s3_screen", "arm": "B0", "profile": "S3", "training_updates": 40},
        {"job_id": "J2", "stage": "stage1_s3_screen", "arm": "AB", "profile": "S3", "training_updates": 40},
    ]
    forecast = hh_campaign._forecast_phase_seconds(
        jobs,
        profile_receipt,
        train_samples=100,
        prepared_manifest=prepared_manifest,
        gpu_workers=2,
        remaining_budget_s=1000.0,
        elapsed_prep_and_profile_s=12.0,
    )
    expected_eval = (
        (512 * 10.0 + 512 * 12.0 + 512 * 15.0)
        + 200 * 5.0
        + 384 * 8.0
        + 512 * 10.0
    ) / 1000.0
    assert abs(forecast["per_job"][0]["eval_seconds"] - expected_eval) < 1e-6
    assert forecast["per_job"][1]["stream_build_seconds"] > forecast["per_job"][0]["stream_build_seconds"]
    assert forecast["projected_wall_seconds"] > forecast["worker_wall_seconds"]
    assert len(forecast["lane_seconds"]) == 2 and all(s > 0 for s in forecast["lane_seconds"])

    # Hard session deadline during Stage 1 canary forces safe INTERRUPTED export + recovery to COMPLETE.
    head = resolve_executing_source_commit()
    out = tmp_path / "deadline_run"
    part = hh_campaign.run(
        output=out,
        source=head,
        phase="stage1",
        minutes=60.0,
        session_deadline_s=0.05,
        worlds_per_hop=2,
        train_bytes=12_000,
        updates_override=4,
        cpu_canary=True,
    )
    assert part["status"] == "INTERRUPTED"
    full_zip = Path(part["delivery"]["exports"]["full"]["zip_path"])
    assert full_zip.is_file()
    restored = tmp_path / "deadline_restored"
    restore_hitchhiker_archive(full_zip, restored, source=head, phase="stage1")
    finished = hh_campaign.run(
        output=restored,
        source=head,
        phase="stage1",
        minutes=60.0,
        worlds_per_hop=2,
        train_bytes=12_000,
        updates_override=2,
        cpu_canary=True,
    )
    assert finished["status"] == "COMPLETE"


def test_r3_handoff_vs_resume_enforcement_and_stage3_winner_propagation(tmp_path: Path):
    head = resolve_executing_source_commit()
    s1_out = tmp_path / "s1"
    s1_res = hh_campaign.run(
        output=s1_out,
        source=head,
        phase="stage1",
        minutes=60.0,
        worlds_per_hop=2,
        train_bytes=12_000,
        updates_override=2,
        cpu_canary=True,
    )
    assert s1_res["status"] == "COMPLETE"
    s1_zip = Path(s1_res["delivery"]["exports"]["full"]["zip_path"])

    # Same-phase restore must reject Stage 1 zip when restoring into Stage 3.
    with pytest.raises(ValueError):
        restore_hitchhiker_archive(s1_zip, tmp_path / "bad_same_phase", source=head, phase="stage3")

    # Reject reverse direction handoff (stage3 -> stage1).
    with pytest.raises(ValueError, match="disallowed cross-phase handoff"):
        handoff_hitchhiker_archive(
            s1_zip,
            tmp_path / "bad_reverse",
            source=head,
            from_phase="stage3",
            to_phase="stage1",
        )

    # Incomplete Stage 1 archive must be rejected for Stage 3 handoff.
    s1_incomplete = hh_campaign.run(
        output=tmp_path / "s1_int",
        source=head,
        phase="stage1",
        minutes=60.0,
        worlds_per_hop=2,
        train_bytes=12_000,
        updates_override=2,
        cpu_interrupt_after=1,
        cpu_canary=True,
    )
    assert s1_incomplete["status"] == "INTERRUPTED"
    with pytest.raises(ValueError, match="EXECUTED|COMPLETE"):
        handoff_hitchhiker_archive(
            Path(s1_incomplete["delivery"]["exports"]["full"]["zip_path"]),
            tmp_path / "bad_inc_handoff",
            source=head,
            from_phase="stage1",
            to_phase="stage3",
        )

    # Construct a genuinely finalized COMPLETE Stage 1 archive with winning_arm="AB" for Stage 3 handoff tests.
    s1_passed_dir = tmp_path / "s1_passed_dir"
    shutil.copytree(s1_out, s1_passed_dir)
    if (s1_passed_dir / "exports").exists():
        shutil.rmtree(s1_passed_dir / "exports")
    for pattern in ("campaign-*.json", "delivery-*.json", "FINAL_MANIFEST.json"):
        for f in (s1_passed_dir / "analysis").glob(pattern):
            f.unlink()
    sel_file = s1_passed_dir / "analysis" / "selection_receipt.json"
    sel_doc = json.loads(sel_file.read_text(encoding="utf-8"))
    sel_doc["screening_status"] = "PASS_SCREENING"
    sel_doc["winning_arm"] = "AB"
    sel_doc["model_profile"] = "S8"
    sel_doc["admitted_followups"] = ["AB+C", "D-external"]
    sel_file.write_text(json.dumps(sel_doc, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    s1_passed_camp = dict(s1_res)
    s1_passed_camp["status"] = "EXECUTED"
    s1_passed_camp["selection_receipt"] = sel_doc
    delivery_passed = hh_campaign.finalize_hitchhiker(
        s1_passed_dir,
        source=head,
        protocol_sha=PROTOCOL_SHA,
        campaign=s1_passed_camp,
        required_jobs=[j["job_id"] for j in stage_jobs("stage1_s3_screen")],
        required_checkpoints=[],
        complete=True,
    )
    s1_mod_zip = Path(delivery_passed["exports"]["full"]["zip_path"])

    # Wrong winning_arm override must be rejected.
    with pytest.raises(ValueError, match="disagrees with locked receipt"):
        handoff_hitchhiker_archive(
            s1_mod_zip,
            tmp_path / "bad_win",
            source=head,
            from_phase="stage1",
            to_phase="stage3",
            winning_arm="A",
        )

    # Valid Stage 1 -> Stage 3 handoff + Stage 3 canary without passing winning_arm again.
    s3_out = tmp_path / "s3"
    s3_res = hh_campaign.run(
        output=s3_out,
        source=head,
        phase="stage3",
        minutes=60.0,
        handoff_archive=s1_mod_zip,
        worlds_per_hop=2,
        train_bytes=12_000,
        updates_override=2,
        cpu_canary=True,
    )
    assert s3_res["status"] == "COMPLETE"
    assert (s3_out / "analysis" / "handoff_receipt.json").is_file()
    for s3_job in stage_jobs("stage3_scale_confirmation", winning_arm="AB"):
        rr = json.loads((s3_out / "analysis" / "jobs" / s3_job["job_id"] / "run_receipt.json").read_text(encoding="utf-8"))
        assert rr["arm"] == s3_job["arm"]


def test_r4_stage3_mixed_s8_s20_gate_and_replay_binding(tmp_path: Path):
    head = resolve_executing_source_commit()
    prepared = tmp_path / "prepared"
    hh_campaign.prepare_hitchhiker_data(
        tmp_path,
        train_bytes=12_000,
        val_worlds=2,
        conf_worlds=2,
        struct_worlds=2,
        probe_worlds=2,
        orig_val_bytes=6_000,
    )
    from anra_v5.elendril.backend import BackendIdentity, NeuralBackend
    from v5_model.core import initialize

    configure_cpu()
    spec = _tiny_utf8_spec()
    model = initialize(spec, 7)
    val_orig = make_dataset("validation", target_bytes=6_000, seed=707)
    conf_suite = hh_campaign.load_suite_bundle(prepared / "confirmation_suite.json")
    struct_suite = hh_campaign.load_suite_bundle(prepared / "structural_suite.json")

    out = tmp_path / "out"
    jobs = stage_jobs("stage3_scale_confirmation", winning_arm="A")
    # Write all 12 genuinely evaluated Stage 3 confirmation jobs (testing profile mismatch, duplicate check, and mixed S8/S20 gate).
    for j in jobs:
        jdir = out / "analysis" / "jobs" / j["job_id"]
        jdir.mkdir(parents=True, exist_ok=True)
        ident = sha256(j["job_id"].encode("utf-8")).hexdigest()
        be = NeuralBackend(model, BackendIdentity(spec.sha256(), UTF8.identity, UTF8.vocabulary_size, spec.context_length, ident))
        pol = RawHitchhikerPolicy(be)
        suite_eval = {
            **evaluate_counterfactual_suite(conf_suite, pol),
            "source_commit": head,
        }
        orig_eval = evaluate_original_kinds(val_orig, pol, split="validation")
        struct_eval = evaluate_counterfactual_suite(struct_suite, pol)
        eval_doc = {
            "protocol_sha256": PROTOCOL_SHA,
            "source_commit": head,
            "job_id": j["job_id"],
            "arm": j["arm"],
            "seed": j["seed"],
            "profile": j["profile"],
            "split": "confirmation",
            "checkpoint_sha256": ident,
            "suite_eval": suite_eval,
            "original_eval": orig_eval,
            "structural_eval": struct_eval,
            "equivariance_report": {
                "model_identity": ident,
                "cases_sha256": suite_eval["records"]["factor-entity-permutation"]["cases_sha256"],
                "complete_pairs": len(conf_suite["cohorts"]["factor-entity-permutation"]),
                "argmax_equivariance_rate": 0.99,
            },
        }
        (jdir / "final_eval.json").write_text(json.dumps(eval_doc), encoding="utf-8")
        (jdir / "run_receipt.json").write_text(json.dumps({"job_id": j["job_id"], "checkpoint_sha256": ident}), encoding="utf-8")

    # Profile mismatch must be caught in replay!
    bad_job_path = out / "analysis" / "jobs" / jobs[0]["job_id"] / "final_eval.json"
    bad_doc = json.loads(bad_job_path.read_text(encoding="utf-8"))
    bad_doc["profile"] = "S8" if jobs[0]["profile"] == "S20" else "S20"
    bad_job_path.write_text(json.dumps(bad_doc), encoding="utf-8")
    with pytest.raises(ValueError, match="profile mismatch"):
        hh_campaign._evaluate_stage_outcomes("stage3", jobs, out, prepared, source=head, winning_arm="A")
    bad_doc["profile"] = jobs[0]["profile"]
    bad_job_path.write_text(json.dumps(bad_doc), encoding="utf-8")

    # Duplicate job_id fails loudly.
    with pytest.raises(ValueError, match="duplicate job_id"):
        hh_campaign._evaluate_stage_outcomes("stage3", [*jobs, jobs[0]], out, prepared, source=head, winning_arm="A")

    # Simulate mixed scale gate outcome: S8 -> FALSIFIED, S20 -> FORMATION_QUALIFIED via patched evaluate_formation_gate.
    ident_to_job = {sha256(j["job_id"].encode("utf-8")).hexdigest(): j for j in jobs}
    orig_eval_fg = hh_campaign.evaluate_formation_gate
    try:
        def _fake_fg(suite_eval_arg, *, equivariance_report=None):
            mid = suite_eval_arg["model_identity"]
            prof = ident_to_job[mid]["profile"]
            return {"status": "FALSIFIED" if prof == "S8" else "FORMATION_QUALIFIED", "model_identity": mid}
        hh_campaign.evaluate_formation_gate = _fake_fg
        shuffled_jobs = list(reversed(jobs))
        _, gate, _ = hh_campaign._evaluate_stage_outcomes("stage3", shuffled_jobs, out, prepared, source=head, winning_arm="A")
    finally:
        hh_campaign.evaluate_formation_gate = orig_eval_fg

    assert gate["treatment_arm"] == "A"
    assert gate["per_scale"]["S8"]["status"] == "FALSIFIED"
    assert gate["per_scale"]["S20"]["status"] == "FORMATION_QUALIFIED"
    assert gate["status"] == "NOT_QUALIFIED"


def test_r5_stage0_admitted_checkpoint_non_null_historical_source_end_to_end(tmp_path: Path, monkeypatch):
    from anra_v5.elendril import hitchhiker_diagnostics as hd
    from tools import elendril_hitchhiker_runner as hh_runner
    from tools import elendril_hitchhiker_worker as hh_worker

    # Bad source_commit in manifest is rejected when mounted checkpoint files exist.
    bad_root = tmp_path / "bad_ckpts"
    bad_lineage = HISTORICAL_S20_LOWLR_MANIFESTS[7]["lineage"]
    bad_sha = HISTORICAL_S20_LOWLR_MANIFESTS[7]["checkpoint_sha256"]
    (bad_root / bad_lineage / "objects" / bad_sha).mkdir(parents=True, exist_ok=True)
    (bad_root / bad_lineage / "LATEST").write_text(bad_sha + "\n", encoding="ascii")
    bad_manifest = dict(HISTORICAL_S20_LOWLR_MANIFESTS[7], source_commit="not-a-sha")
    monkeypatch.setitem(hd.HISTORICAL_S20_LOWLR_MANIFESTS, 7, bad_manifest)
    with pytest.raises(ValueError, match="source_commit"):
        admit_historical_s20_checkpoint(bad_root, 7)

    # Restore valid manifest and test both runner & worker Stage 0 execution with an ADMITTED synthetic backend.
    monkeypatch.undo()
    configure_cpu()
    spec = _tiny_utf8_spec()
    from anra_v5.elendril.backend import BackendIdentity, NeuralBackend
    from v5_model.core import initialize

    manifest_7 = dict(HISTORICAL_S20_LOWLR_MANIFESTS[7])
    expected_hist_src = manifest_7["source_commit"]
    assert len(expected_hist_src) == 40
    backend = NeuralBackend(
        initialize(spec, 7),
        BackendIdentity(spec.sha256(), UTF8.identity, UTF8.vocabulary_size, spec.context_length, manifest_7["checkpoint_sha256"]),
    )

    def _mock_admit(checkpoint_root, seed, *, expected_data_sha=None, device="cuda:0"):
        return {
            "status": "ADMITTED",
            "fabricate_permitted": False,
            "seed": seed,
            "job_id": manifest_7["lineage"],
            "checkpoint_sha256": manifest_7["checkpoint_sha256"],
            "expected_checkpoint_sha256": manifest_7["checkpoint_sha256"],
            "lineage": manifest_7["lineage"],
            "global_update": manifest_7["expected_update"],
            "model_spec_sha256": spec.sha256(),
            "tokenizer_sha256": UTF8.identity,
            "parameters": 1234,
            "tensor_sha256s": {},
            "metadata": {"seed": seed, "profile": "S20"},
            "manifest": manifest_7,
            "historical_training_source": expected_hist_src,
            "backend": backend,
            "admission_scope": "historical_s20_low_lr",
        }

    monkeypatch.setattr(hh_runner, "admit_historical_s20_checkpoint", _mock_admit)
    monkeypatch.setattr(hh_worker, "admit_historical_s20_checkpoint", _mock_admit)

    pkg_dir = tmp_path / "runner_pkg"
    runner_manifest = run_stage0_audit(
        checkpoint_root=str(tmp_path / "ckpts"),
        seed=7,
        worlds_per_hop=2,
        package_dir=pkg_dir,
    )
    assert runner_manifest["historical_training_source"] == expected_hist_src
    assert (pkg_dir / "stage0_package_manifest.json").is_file()
    pkg_manifest = json.loads((pkg_dir / "stage0_package_manifest.json").read_text(encoding="utf-8"))
    assert pkg_manifest["historical_training_source"] == expected_hist_src

    w_out = tmp_path / "worker_s0"
    hh_campaign.prepare_hitchhiker_data(
        w_out,
        train_bytes=12_000,
        val_worlds=2,
        conf_worlds=2,
        struct_worlds=2,
        probe_worlds=2,
        orig_val_bytes=6_000,
    )
    w_receipt = w_out / "analysis" / "stage0" / "worker_receipt.json"
    summary = hh_worker.mode_stage0(
        type("Args", (), {
            "source": resolve_executing_source_commit(),
            "prepared": w_out / "prepared",
            "output": w_out,
            "checkpoint_root": str(tmp_path / "ckpts"),
            "seed": 7,
            "receipt": w_receipt,
            "expected_data_sha": None,
            "cpu_canary": True,
        })()
    )
    assert summary["status"] == "EXECUTED"
    assert summary["historical_training_source"] == expected_hist_src
    assert json.loads((w_out / "analysis" / "stage0" / "seed_7" / "stage0_package" / "stage0_package_manifest.json").read_text(encoding="utf-8"))["historical_training_source"] == expected_hist_src


def test_r6_familiar_primary_sources_and_source_renaming_cohort():
    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=4, split="validation")
    verify_counterfactual_suite(suite)
    for hop in ("hop-1", "hop-2", "hop-3", "hop-4"):
        for c in suite["cohorts"][hop]:
            assert all(ev.source == "map" for ev in c.environment.initial)
    for c in suite["cohorts"]["factor-answer-query-swap"]:
        assert all(ev.source == "map" for ev in c.environment.initial)

    sr = suite["cohorts"]["factor-source-renaming"]
    assert len(sr) == 4 * 4 * 2
    fam_sources = {ev.source for c in sr if c.panel == "source-familiar" for ev in c.environment.initial}
    ren_sources = {ev.source for c in sr if c.panel == "source-renamed" for ev in c.environment.initial}
    assert fam_sources == {"map"}
    assert all(s.startswith("src-") for s in ren_sources)

    class _MapOnlyPolicy(ControlPolicy):
        def __init__(self, *, surface=None, memory_available=False):
            super().__init__("recency", surface=surface, memory_available=memory_available)
            self.mode = "fixed_baseline"
            self.decoder_schema = "utf8-raw"

        def for_case(self, case, *, memory_available=None):
            return _MapOnlyPolicy(
                surface=case.surface,
                memory_available=case.memory is not None if memory_available is None else memory_available,
            )

        def propose(self, state, legal, channels, positions):
            if any(ev.source == "map" for ev in state.events):
                return super().propose(state, legal, channels, positions)
            return OperationCandidate("ABSTAIN")

    ev_rep = evaluate_counterfactual_suite(suite, _MapOnlyPolicy())
    sr_sum = ev_rep["factor_summary"]["factor-source-renaming"]
    assert sr_sum["familiar_source_accuracy"] == 1.0
    assert sr_sum["renamed_source_accuracy"] == 0.0
    assert sr_sum["accuracy_gap"] == 1.0


def test_r7_fail_closed_custody_receipt_and_cuda_guard(tmp_path: Path):
    configure_cpu()
    train_base = make_dataset("train", target_bytes=12_000, seed=606)
    val_orig = make_dataset("validation", target_bytes=6_000, seed=707)
    val_suite = build_counterfactual_suite(seed=17071, worlds_per_hop=2, split="validation")

    test_job = next(j for j in stage_jobs("stage1_s3_screen") if j["arm"] == "AB")
    full_updates = math.ceil(len(train_base.samples) / test_job["batch_size"]) * test_job["epochs"]
    no_heldout_full = build_training_stream(
        "AB",
        base_data=train_base,
        total_updates=full_updates,
        batch_size=test_job["batch_size"],
        data_seed=606,
        order_seed=7,
        heldout_suites=(),
    )
    assert no_heldout_full["data_custody_verified"] is False
    assert no_heldout_full["custody_receipt"]["passed"] is False
    assert no_heldout_full["custody_receipt"]["status"] == "UNVERIFIED_NO_HELDOUT"
    with pytest.raises(ValueError, match="custody_receipt"):
        verify_custody_receipt(no_heldout_full, heldout_suites=(val_suite,))
    with pytest.raises(ValueError, match="heldout_suites|data_custody_verified"):
        create_hitchhiker_trainer(
            test_job,
            source_commit=resolve_executing_source_commit(),
            root=tmp_path / "ckpt_no_heldout",
            stream=no_heldout_full,
            device="cuda:0",
            heldout_suites=(),
        )
    with pytest.raises(ValueError, match="overlaps heldout suites"):
        build_training_stream("A", base_data=train_base, total_updates=2, batch_size=4, data_seed=606, order_seed=7, heldout_suites=(train_base,))

    good = build_training_stream(
        "AB",
        base_data=train_base,
        total_updates=2,
        batch_size=4,
        data_seed=606,
        order_seed=7,
        heldout_suites=(val_suite, val_orig),
    )
    assert good["data_custody_verified"] is True
    assert good["custody_receipt"]["passed"] is True
    assert verify_custody_receipt(good, heldout_suites=(val_suite, val_orig))["passed"] is True

    # Tampered receipt fails validation.
    tampered = {**good, "custody_receipt": {**good["custody_receipt"], "receipt_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="receipt_sha256"):
        verify_custody_receipt(tampered, heldout_suites=(val_suite, val_orig))
    with pytest.raises(ValueError, match="receipt_sha256"):
        validate_stream_contract(tampered, expected_arm="AB", total_updates=2, batch_size=4)

    # Mismatched heldout_suites fails in create_hitchhiker_trainer.
    with pytest.raises(ValueError):
        create_hitchhiker_trainer(
            test_job,
            source_commit=resolve_executing_source_commit(),
            root=tmp_path / "ckpt",
            stream=good,
            device="cpu",
            cpu_canary=True,
            spec_override=_tiny_utf8_spec(),
            updates_override=2,
            batch_override=4,
            heldout_suites=(val_suite,),
        )


def test_r8_ab_plus_c_matches_c_exposure_marginals_across_seeds():
    train_base = make_dataset("train", target_bytes=20_000, seed=606)
    val_orig = make_dataset("validation", target_bytes=6_000, seed=707)
    for seed in (7, 19, 41):
        stream_c = build_training_stream(
            "C",
            base_data=train_base,
            total_updates=12,
            batch_size=4,
            data_seed=606,
            order_seed=seed,
            heldout_suites=(val_orig,),
        )
        stream_abc = build_training_stream(
            "AB+C",
            base_data=train_base,
            total_updates=12,
            batch_size=4,
            data_seed=606,
            order_seed=seed,
            heldout_suites=(val_orig,),
        )
        table = {row["arm"]: row for row in arm_comparison_table({"C": stream_c, "AB+C": stream_abc})}
        assert table["AB+C"]["stage_marginals"] == table["C"]["stage_marginals"]
        assert table["AB+C"]["task_kind_marginals"] == table["C"]["task_kind_marginals"]
        assert table["AB+C"]["operation_marginals"] == table["C"]["operation_marginals"]
        assert table["AB+C"]["supervised_tokens"] == table["C"]["supervised_tokens"]
        assert table["AB+C"]["total_examples_served"] == table["C"]["total_examples_served"]
        assert table["AB+C"]["sampling_method"] == table["C"]["sampling_method"] == "blocked"
        assert table["AB+C"]["coverage_recipe"] == table["C"]["coverage_recipe"] == "14-stratum-coverage"


def test_r9_stage2_admission_policy_and_subset_validation():
    sel_pass = {
        "schema": "elendril-hitchhiker-selection-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "source_commit": "0" * 40,
        "stage1_completed": True,
        "screening_status": "PASS_SCREENING",
        "split": "validation",
        "winning_arm": "AB",
        "admitted_followups": ["AB+C", "D-external"],
        "excluded_followups": {"C+penalty": "Withheld", "S20-contrast": "Withheld"},
    }
    sub = hh_campaign._resolve_stage_jobs(
        "stage2",
        selection_receipt=sel_pass,
        winning_arm=None,
        admitted_followups=["D-external"],
    )
    assert {j["arm"] for j in sub} == {"D-external"}
    with pytest.raises(ValueError, match="not admitted"):
        hh_campaign._resolve_stage_jobs(
            "stage2",
            selection_receipt=sel_pass,
            winning_arm=None,
            admitted_followups=["C+penalty"],
        )
    with pytest.raises(ValueError, match="non-empty"):
        hh_campaign._resolve_stage_jobs(
            "stage2",
            selection_receipt=sel_pass,
            winning_arm=None,
            admitted_followups=[],
        )
    with pytest.raises(ValueError, match="duplicate"):
        hh_campaign._resolve_stage_jobs(
            "stage2",
            selection_receipt=sel_pass,
            winning_arm=None,
            admitted_followups=["D-external", "D-external"],
        )

    # Stage 1 no-winner: _build_stage2_admission derives evidence-bound list & explicit exclusions (no silent all-grant).
    arm_gates = {
        "B0": {"status": "BASELINE", "mean_primary_gain": 0.0},
        "A": {"status": "FALSIFIED", "mean_primary_gain": -0.05},
        "B": {"status": "FALSIFIED", "mean_primary_gain": -0.05},
        "AB": {"status": "FALSIFIED", "mean_primary_gain": -0.05},
        "C": {"status": "FALSIFIED", "mean_primary_gain": -0.05},
    }
    fake_jobs = [{"job_id": f"{a}-s7", "arm": a} for a in ("B0", "A", "B", "AB", "C")]
    fake_evals = {
        f"{a}-s7": {"cohort_metrics": {"primary": {"invalid_rate": 0.0}}, "equivariance_report": {"argmax_equivariance_rate": 0.60}}
        for a in ("B0", "A", "B", "AB", "C")
    }
    adm, _, exc = hh_campaign._build_stage2_admission(arm_gates, fake_jobs, fake_evals, locked=None, top_status="FALSIFIED")
    assert adm == ["D-external"]
    assert "AB+C" in exc and "S20-contrast" in exc and "C+penalty" in exc
    assert validate_stage_selection("stage2_followups", {"stage1_completed": True, "screening_status": "FALSIFIED", "admitted_followups": adm})["admitted_followups"] == ["D-external"]


def test_r10_interrupt_resume_reconciles_update_ledger_and_tokens(tmp_path: Path):
    from tools import elendril_hitchhiker_worker as hh_worker

    head = resolve_executing_source_commit()
    # Uninterrupted 2-update Stage 1 canary baseline.
    clean_out = tmp_path / "clean"
    clean_res = hh_campaign.run(
        output=clean_out,
        source=head,
        phase="stage1",
        minutes=60.0,
        worlds_per_hop=2,
        train_bytes=12_000,
        updates_override=2,
        cpu_canary=True,
    )
    assert clean_res["status"] == "COMPLETE"

    # Interrupted at update 1 -> restore -> inject stale uncommitted update=2 row -> resume to update 2.
    part_out = tmp_path / "part"
    part_res = hh_campaign.run(
        output=part_out,
        source=head,
        phase="stage1",
        minutes=60.0,
        worlds_per_hop=2,
        train_bytes=12_000,
        updates_override=2,
        cpu_interrupt_after=1,
        cpu_canary=True,
    )
    assert part_res["status"] == "INTERRUPTED"
    restored = tmp_path / "restored"
    restore_hitchhiker_archive(Path(part_res["delivery"]["exports"]["full"]["zip_path"]), restored, source=head, phase="stage1")

    sample_jid = stage_jobs("stage1_s3_screen")[0]["job_id"]
    sample_ledger = restored / "analysis" / "jobs" / sample_jid / "update_logs.jsonl"
    hh_worker._append_update_ledger(
        sample_ledger,
        {"update": 2, "loss": 99.0, "supervised_tokens": 999, "input_positions": 999, "examples": 999, "lr": 0.0, "step_ms": 1.0, "grad_norm": 0.0},
    )

    resumed_res = hh_campaign.run(
        output=restored,
        source=head,
        phase="stage1",
        minutes=60.0,
        worlds_per_hop=2,
        train_bytes=12_000,
        updates_override=2,
        cpu_canary=True,
    )
    assert resumed_res["status"] == "COMPLETE"

    s1_jobs = stage_jobs("stage1_s3_screen")
    started_in_part1 = {s1_jobs[0]["job_id"], s1_jobs[1]["job_id"]}
    for j in s1_jobs:
        jid = j["job_id"]
        c_rec = json.loads((clean_out / "analysis" / "jobs" / jid / "run_receipt.json").read_text(encoding="utf-8"))
        r_rec = json.loads((restored / "analysis" / "jobs" / jid / "run_receipt.json").read_text(encoding="utf-8"))
        assert r_rec["updates_committed"] == c_rec["updates_committed"] == 2
        assert r_rec["supervised_tokens"] == c_rec["supervised_tokens"]
        assert r_rec["input_positions"] == c_rec["input_positions"]
        assert r_rec["examples_processed"] == c_rec["examples_processed"]
        expected_this_proc = 1 if jid in started_in_part1 else 2
        assert r_rec["updates_executed_this_process"] == expected_this_proc
    sample_rec = json.loads((restored / "analysis" / "jobs" / sample_jid / "run_receipt.json").read_text(encoding="utf-8"))
    assert sample_rec["replayed_uncommitted_updates"] == 1

    # Already-complete resume preserves prior totals instead of overwriting with zeros.
    skip_out = tmp_path / "skip_restored"
    restore_hitchhiker_archive(Path(resumed_res["delivery"]["exports"]["full"]["zip_path"]), skip_out, source=head, phase="stage1")
    skipped_res = hh_campaign.run(
        output=skip_out,
        source=head,
        phase="stage1",
        minutes=60.0,
        worlds_per_hop=2,
        train_bytes=12_000,
        updates_override=2,
        cpu_canary=True,
    )
    assert skipped_res["status"] == "COMPLETE"
    assert all(w["resumed_skipped"] is True and w["updates_executed_this_process"] == 0 for w in skipped_res["worker_receipts"])
    after_skip = json.loads((skip_out / "analysis" / "jobs" / sample_jid / "run_receipt.json").read_text(encoding="utf-8"))
    assert after_skip == sample_rec
    assert after_skip["supervised_tokens"] == sample_rec["supervised_tokens"]
    assert after_skip["updates_committed"] == 2

