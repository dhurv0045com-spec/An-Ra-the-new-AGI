"""Verification tests for the Planthir Universal Research Infrastructure."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.contracts import SYMBOLS, build_v8_model_spec
from anra_v5.elendril.v8.data import D1_TRAIN_SEED, build_d1_training_dataset
from anra_v5.elendril.v8.evaluation import run_episode as v8_run_episode
from anra_v5.elendril.v8.models import build_v8_model
from anra_v5.elendril.v8.protocol import jobs as protocol_jobs
from anra_v5.elendril.v8.training import (
    V8TypedTrainer,
    create_v8_trainer,
    prepare_v8_training_data,
)
from anra_v5.elendril.foundation_training import Recipe
from anra_v5.planthir import (
    CartridgeArmSpec,
    CartridgeRunner,
    CausalContrastSpec,
    CheckpointCatalog,
    DIVERGENCE_TAXONOMY,
    ExperimentCartridge,
    ExperimentRegistration,
    FalsifierSpec,
    HypothesisSpec,
    InternalTelemetryCollector,
    PlanthirRegistry,
    ProcessFunnelAnalyzer,
    UniversalObservation,
    V8EnvironmentKernel,
    V8ObservablePolicy,
    audit_experimental_design,
    build_historical_atlas,
    classify_parameter_group,
    compute_effective_rank,
    compute_representation_isotropy,
    export_result_bundle,
    package_three_tier_evidence,
    probe_from_capsule,
    run_compute_depth_scaling_intervention,
    run_fault_injection_suite,
    run_invariance_counterfactuals,
    run_oracle_information_intervention,
    run_representation_linear_probe,
    run_subgoal_hop_decomposition,
    run_surgical_mechanism_ablation,
    synthesize_scientific_interpretation,
    validate_environment_kernel,
    verify_result_bundle,
    verify_single_planthir_zip,
    verify_tier2_capsule,
)
from anra_v5.planthir.instrument import _OracleV8Actor

_REPO_ROOT = Path(__file__).resolve().parents[1]
_COMMIT = "21f57160343a54053278360e056b3ba4347f7c03"


def _tiny_base_spec():
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


def _sample_stratified_worlds(limit_multihop: int = 4) -> tuple[list[dict], list[dict]]:
    d1 = build_d1_training_dataset(seed=D1_TRAIN_SEED + 7, quota_per_family=16)
    by_family: dict[str, list[dict]] = {}
    multihop: list[dict] = []
    for w in d1.worlds:
        by_family.setdefault(str(w["family"]), []).append(w)
        if int(w.get("hops", 1)) >= 2 and len(multihop) < limit_multihop:
            multihop.append(w)
    eval_worlds = [bucket[0] for _, bucket in sorted(by_family.items())]
    eval_worlds.extend(bucket[1] for _, bucket in sorted(by_family.items()) if len(bucket) > 1)
    return eval_worlds[:12], multihop


def test_observation_invariants_and_truth_leak_detection() -> None:
    clean = UniversalObservation(
        obs_id="obs-clean-1",
        rendered_text="State: A=B@1/s0\nAnswer:",
        available_tools={"query": True, "retrieve": False},
        consumed_tools=frozenset(),
        decision_index=0,
        budget_remaining=4,
        metadata={"family": "known"},
    )
    assert clean.validate_invariants() == []

    bad = UniversalObservation(
        obs_id="",
        rendered_text="State: expected=B",
        available_tools={"query": True},
        consumed_tools=frozenset({"query"}),
        decision_index=-1,
        budget_remaining=0,
        metadata={"hidden": {"expected": "B"}},
    )
    violations = bad.validate_invariants()
    assert any("empty_obs_id" == v for v in violations)
    assert any(v.startswith("truth_leak_marker:") for v in violations)
    assert any(v == "consumed_tool_still_available:query" for v in violations)
    assert any(v == "forbidden_metadata_key:hidden" for v in violations)


def test_v8_consumed_tool_evaluator_fix_and_kernel_parity() -> None:
    d1 = build_d1_training_dataset(quota_per_family=16)
    t0_tool = [
        w
        for w in d1.worlds
        if int(w["renderer_template"]) == 0
        and w["family"] in ("missing_query", "conflict_query", "memory_retrieval")
    ][:6]

    kernel_fixed = V8EnvironmentKernel(enforce_tool_consumption=True)
    actor = _OracleV8Actor(kernel_fixed)
    for w in t0_tool:
        ep_ok = v8_run_episode(actor, dict(w), max_actions=4, enforce_tool_consumption=True)
        assert ep_ok.outcome.correct is True
        assert ep_ok.query_count <= 1 and ep_ok.retrieve_count <= 1

    # Reproducing legacy bug on at least one multi-tool world causes repeated consumed tool calls
    repeated_failures = 0
    for w in t0_tool:
        ep_bug = v8_run_episode(actor, dict(w), max_actions=4, enforce_tool_consumption=False)
        if not ep_bug.outcome.correct and (ep_bug.query_count > 1 or ep_bug.retrieve_count > 1):
            repeated_failures += 1
    assert repeated_failures >= 1

    healthy_report = validate_environment_kernel(
        kernel_fixed, t0_tool, check_v8_evaluator_parity=True, v8_enforce_tool_consumption=True
    )
    assert healthy_report["passed"] is True

    faulty_report = validate_environment_kernel(
        V8EnvironmentKernel(enforce_tool_consumption=False),
        t0_tool,
        check_v8_evaluator_parity=True,
        v8_enforce_tool_consumption=False,
    )
    assert faulty_report["status"] == "INSTRUMENT_DEFECT_DETECTED"


def test_controlled_fault_injection_suite_detects_all_six() -> None:
    report = run_fault_injection_suite()
    assert report["healthy_v8_kernel_passed"] is True
    assert report["total_faults"] == 6
    assert report["detected_faults"] == 6
    assert report["all_detected"] is True


def test_observatory_telemetry_and_process_funnel() -> None:
    # Effective rank & isotropy sanity
    eye = torch.eye(8, dtype=torch.float32)
    assert compute_effective_rank(eye) > 6.5
    collapsed = torch.ones((8, 4), dtype=torch.float32)
    assert compute_effective_rank(collapsed) == pytest.approx(1.0, abs=1e-4)
    assert compute_representation_isotropy(collapsed)["isotropy"] < 0.05
    assert classify_parameter_group("q_proj.weight") == "pointer_or_memory"
    assert classify_parameter_group("u_gru.weight_ih") == "recurrent"

    eval_worlds, _ = _sample_stratified_worlds()
    kernel = V8EnvironmentKernel(enforce_tool_consumption=True)
    probe_obs = [kernel.observe(kernel.initial_state(w)) for w in eval_worlds[:6]]

    base = _tiny_base_spec()
    collector = InternalTelemetryCollector()

    spec_p = build_v8_model_spec("P", base_spec=base, pointer_dim=16)
    model_p = build_v8_model(spec_p, 7, torch_module=torch)
    pol_p = V8ObservablePolicy(model_p, spec_p, model_identity="test-P", mode="raw")
    snap_p = collector.measure_snapshot(model_p, step=0, lr=1e-4, policy=pol_p, probe_observations=probe_obs)
    assert "pointer_gate_mean" in snap_p["mechanism_telemetry"]
    assert snap_p["representation"]["effective_rank"] >= 1.0

    spec_u = build_v8_model_spec("U", base_spec=base, recurrent_dim=16, recurrent_steps=2)
    model_u = build_v8_model(spec_u, 7, torch_module=torch)
    pol_u = V8ObservablePolicy(model_u, spec_u, model_identity="test-U", mode="raw")
    snap_u = collector.measure_snapshot(model_u, step=0, lr=1e-4, policy=pol_u, probe_observations=probe_obs)
    assert "recurrent_mean_delta" in snap_u["mechanism_telemetry"]

    funnel = ProcessFunnelAnalyzer().evaluate_suite(pol_p, kernel, eval_worlds, include_subgoals=True)
    assert funnel["episodes"] == len(eval_worlds)
    assert funnel["funnel"]["stage3_single_hop_grounding"] is not None
    assert funnel["funnel"]["stage4_epistemic_abstention"] is not None
    assert set(funnel["divergence_taxonomy"]) == set(DIVERGENCE_TAXONOMY)
    for k in ("brier_score", "ece", "aurc"):
        assert 0.0 <= funnel["calibration"][k] <= 1.0


def test_causal_interventions_and_epistemic_firewall() -> None:
    eval_worlds, multihop_worlds = _sample_stratified_worlds(limit_multihop=4)
    kernel = V8EnvironmentKernel(enforce_tool_consumption=True)
    base = _tiny_base_spec()
    spec_p = build_v8_model_spec("P", base_spec=base, pointer_dim=16)
    model_p = build_v8_model(spec_p, 7, torch_module=torch)
    pol_raw = V8ObservablePolicy(model_p, spec_p, model_identity="test-P", mode="raw")
    pol_mask = V8ObservablePolicy(model_p, spec_p, model_identity="test-P", mode="availability-mask")

    oracle = run_oracle_information_intervention(pol_raw, kernel, eval_worlds)
    subgoal = run_subgoal_hop_decomposition(pol_raw, kernel, multihop_worlds)
    inv = run_invariance_counterfactuals(
        pol_raw, kernel, eval_worlds[:6], mode_policies={"raw": pol_raw, "availability-mask": pol_mask}
    )
    probe = run_representation_linear_probe(pol_raw, kernel, eval_worlds[:6], eval_worlds[6:10])
    funnel = ProcessFunnelAnalyzer().evaluate_suite(pol_raw, kernel, eval_worlds)

    assert "delta_oracle" in oracle and len(oracle["delta_oracle_95ci"]) == 2
    assert subgoal["worlds_decomposed"] >= 1
    assert set(inv["decoder_modes"]) == {"raw", "availability-mask"}
    assert 0.0 <= probe["probe_operation_accuracy"] <= 1.0

    # Epistemic firewall MUST refuse to certify claims if instrument validation failed
    halted = synthesize_scientific_interpretation(
        experiment_id="TEST-HALT",
        instrument_validation={"passed": False, "status": "INSTRUMENT_DEFECT_DETECTED", "defects": [{"check": "x"}]},
        process_funnel=funnel,
        oracle_intervention=oracle,
    )
    assert halted.status == "INVALIDATED_BY_INSTRUMENT_DEFECT"
    assert halted.confirmed_causes == ()
    assert halted.recommended_next_experiments[0].action_id == "REPAIR_INSTRUMENT_DEFECT"

    # When instrument passes, falsifiers & unresolved scope guard are synthesized
    hyp = HypothesisSpec(
        hypothesis_id="H-TEST",
        statement="Probe accuracy is finite",
        mechanism_target="linear_readout",
        falsifiers=(FalsifierSpec("F1", "Nonnegative probe acc", "probe_op", ">=", 0.0),),
    )
    report = synthesize_scientific_interpretation(
        experiment_id="TEST-OK",
        instrument_validation={"passed": True, "status": "PASS", "defects": []},
        process_funnel=funnel,
        oracle_intervention=oracle,
        subgoal_decomposition=subgoal,
        invariance_counterfactuals=inv,
        linear_probe=probe,
        hypotheses=(hyp,),
        observed_metrics_for_falsifiers={"probe_op": probe["probe_operation_accuracy"]},
    )
    assert report.status == "INTERPRETED_WITH_INSTRUMENT_VERIFIED"
    assert any(u.claim_id == "U-SCALE-AND-HORIZON" for u in report.unresolved_questions)
    assert report.falsified_hypotheses == ()


def test_checkpoint_intelligence_and_three_tier_evidence(tmp_path: Path) -> None:
    td = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=12)
    base = _tiny_base_spec()
    ckpt_root = tmp_path / "ckpts"

    # Train 1 update each for J, F, and B0
    recipe = Recipe(seed=7, total_updates=1, batch_size=4, micro_batch_size=2, lr=1e-4, schedule="cosine", precision="fp32")
    shas: dict[str, str] = {}
    specs: dict[str, object] = {}
    for arm in ("J", "F"):
        sp = build_v8_model_spec(arm, base_spec=base, pointer_dim=16)
        specs[arm] = sp
        tr = V8TypedTrainer(
            td.typed_samples,
            v8_spec=sp,
            recipe=recipe,
            order_seed=7,
            source_commit=_COMMIT,
            root=ckpt_root,
            lineage=f"test-{arm}",
            data_provenance=td.manifest,
            device="cpu",
            cpu_canary=True,
        )
        tr.step()
        shas[arm] = tr.save()

    b0_job = next(j for j in protocol_jobs("data-control") if j["arm"] == "B0" and j["dataset"] == "D1" and j["init_seed"] == 7)
    b0_spec = build_v8_model_spec("B0", base_spec=base)
    tr_b0 = create_v8_trainer(
        td, b0_job, root=ckpt_root, source=_COMMIT, device="cpu", cpu_canary=True, v8_spec_override=b0_spec, updates_override=1
    )
    tr_b0.step()
    shas["B0"] = tr_b0.save()

    catalog = CheckpointCatalog.discover_checkpoints([ckpt_root])
    assert len(catalog.records) == 3
    f_recs = catalog.query_compatible_checkpoints(architecture="F")
    j_recs = catalog.query_compatible_checkpoints(architecture="J")
    b0_recs = catalog.query_compatible_checkpoints(architecture="B0")
    assert len(f_recs) == 1 and len(j_recs) == 1 and len(b0_recs) == 1

    diff = CheckpointCatalog.diff_checkpoints(j_recs[0], f_recs[0])
    assert diff["single_architecture_contrast"] is True

    loaded_f = CheckpointCatalog.load_checkpoint_for_causal_probing(f_recs[0], mode="raw")
    loaded_b0 = CheckpointCatalog.load_checkpoint_for_causal_probing(b0_recs[0], mode="raw")
    assert loaded_f["policy"] is not None
    assert loaded_b0["policy"] is not None

    eval_worlds, _ = _sample_stratified_worlds()
    kernel = V8EnvironmentKernel(enforce_tool_consumption=True)
    funnel = ProcessFunnelAnalyzer().evaluate_suite(loaded_f["policy"], kernel, eval_worlds[:6])

    reg = ExperimentRegistration(
        experiment_id="TEST-EVIDENCE-F",
        program="Planthir-Test",
        architecture_family="F",
        parameter_count=f_recs[0].parameter_count or 0,
        dataset_identity=td.data_identity,
        source_commit=_COMMIT,
        hypotheses=(),
        contrasts=(CausalContrastSpec("C1", "arch", "J", "F", ("seed",), "op_vs_val"),),
    )
    interp = synthesize_scientific_interpretation(
        experiment_id="TEST-EVIDENCE-F",
        instrument_validation={"passed": True, "status": "PASS", "defects": []},
        process_funnel=funnel,
    )
    obs_batch = [kernel.observe(kernel.initial_state(w)) for w in eval_worlds]
    hidden = loaded_f["policy"].extract_representations(obs_batch)["decision_hidden"]
    bundle_dir = tmp_path / "evidence_F"
    packaged = package_three_tier_evidence(
        bundle_dir,
        registration=reg,
        instrument_validation={"passed": True, "status": "PASS", "kernel_id": kernel.kernel_id, "checks": {}, "defects": []},
        process_funnel=funnel,
        causal_diagnostics={},
        interpretation=interp,
        probe_arrays={"decision_hidden": hidden, "y_final": torch.arange(len(obs_batch)) % len(SYMBOLS)},
        checkpoint_root=Path(f_recs[0].checkpoint_root),
        checkpoint_lineage=f_recs[0].lineage_id,
    )
    assert packaged["tier3_binding"]["status"] == "BOUND_VERIFIED"
    manifest = verify_tier2_capsule(bundle_dir / "tier2_capsule")
    assert manifest["total_bytes"] < 2 * 1024 * 1024

    offline = probe_from_capsule(bundle_dir / "tier2_capsule")
    assert 0.0 <= offline["probes"]["y_final"] <= 1.0

    # Tampering Tier-2 capsule must fail closed
    npz_file = bundle_dir / "tier2_capsule" / "representations.npz"
    npz_file.write_bytes(npz_file.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="corruption"):
        verify_tier2_capsule(bundle_dir / "tier2_capsule")


def test_registry_idempotence_and_historical_atlas(tmp_path: Path) -> None:
    registry = PlanthirRegistry()
    reg = ExperimentRegistration(
        experiment_id="EXP-REG-1",
        program="V8",
        architecture_family="F",
        parameter_count=18340,
        dataset_identity="abc",
        source_commit=_COMMIT,
        hypotheses=(),
        contrasts=(),
    )
    sha = registry.register(reg)
    assert registry.register(reg) == sha

    conflict = replace(reg, parameter_count=19999)
    with pytest.raises(ValueError, match="conflicting registration"):
        registry.register(conflict)

    saved = registry.save(tmp_path / "reg.json")
    reloaded = PlanthirRegistry.load(saved)
    assert reloaded.get("EXP-REG-1")["registration_sha256"] == sha

    atlas = build_historical_atlas(_REPO_ROOT)
    assert atlas["generations_indexed"] == 7
    assert atlas["total_archived_runs_indexed"] >= 50
    defect_ids = {d["id"] for d in atlas["historical_instrument_defects"]}
    assert {"HIST-DEFECT-V716-SHADOW", "HIST-DEFECT-V8-UNCONSUMED-TOOL"}.issubset(defect_ids)


def test_result_zip_and_full_checkpoint_bundle_export_verify_and_load(tmp_path: Path) -> None:
    td = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=12)
    base = _tiny_base_spec()
    ckpt_root = tmp_path / "ckpts"
    recipe = Recipe(
        seed=7,
        total_updates=1,
        batch_size=4,
        micro_batch_size=2,
        lr=1e-4,
        schedule="cosine",
        precision="fp32",
    )

    trained_shas: dict[str, str] = {}
    lineage_by_arm: dict[str, str] = {}
    for arm in ("J", "F"):
        sp = build_v8_model_spec(arm, base_spec=base, pointer_dim=16)
        tr = V8TypedTrainer(
            td.typed_samples,
            v8_spec=sp,
            recipe=recipe,
            order_seed=7,
            source_commit=_COMMIT,
            root=ckpt_root,
            lineage=f"bundle-{arm}",
            data_provenance=td.manifest,
            device="cpu",
            cpu_canary=True,
        )
        tr.step()
        trained_shas[arm] = tr.save()
        lineage_by_arm[arm] = f"bundle-{arm}"

    b0_job = next(
        j for j in protocol_jobs("data-control") if j["arm"] == "B0" and j["dataset"] == "D1" and j["init_seed"] == 7
    )
    b0_spec = build_v8_model_spec("B0", base_spec=base)
    tr_b0 = create_v8_trainer(
        td,
        b0_job,
        root=ckpt_root,
        source=_COMMIT,
        device="cpu",
        cpu_canary=True,
        v8_spec_override=b0_spec,
        updates_override=1,
    )
    tr_b0.step()
    trained_shas["B0"] = tr_b0.save()
    lineage_by_arm["B0"] = str(b0_job["job_id"])

    catalog = CheckpointCatalog.discover_checkpoints([ckpt_root])
    assert len(catalog.records) == 3

    eval_worlds, _ = _sample_stratified_worlds()
    kernel = V8EnvironmentKernel(enforce_tool_consumption=True)
    obs_batch = [kernel.observe(kernel.initial_state(w)) for w in eval_worlds[:6]]

    exp_bundles: dict[str, dict] = {}
    for arm in ("B0", "J", "F"):
        rec = [r for r in catalog.records if r.lineage_id == lineage_by_arm[arm]][0]
        loaded = CheckpointCatalog.load_checkpoint_for_causal_probing(rec, mode="raw")
        funnel = ProcessFunnelAnalyzer().evaluate_suite(loaded["policy"], kernel, eval_worlds[:4])
        interp = synthesize_scientific_interpretation(
            experiment_id=f"EXP-BUNDLE-{arm}",
            instrument_validation={"passed": True, "status": "PASS", "defects": []},
            process_funnel=funnel,
        )
        reg = ExperimentRegistration(
            experiment_id=f"EXP-BUNDLE-{arm}",
            program="Planthir-Bundle-Test",
            architecture_family=arm,
            parameter_count=rec.parameter_count or 0,
            dataset_identity=td.data_identity,
            source_commit=_COMMIT,
            hypotheses=(),
            contrasts=(),
        )
        hidden = loaded["policy"].extract_representations(obs_batch)["decision_hidden"]
        pkg = package_three_tier_evidence(
            tmp_path / "tiers" / arm,
            registration=reg,
            instrument_validation={
                "passed": True,
                "status": "PASS",
                "kernel_id": kernel.kernel_id,
                "checks": {},
                "defects": [],
            },
            process_funnel=funnel,
            causal_diagnostics={},
            interpretation=interp,
            probe_arrays={"decision_hidden": hidden, "y_final": torch.arange(len(obs_batch)) % len(SYMBOLS)},
            checkpoint_root=ckpt_root,
            checkpoint_lineage=rec.lineage_id,
        )
        exp_bundles[f"EXP-BUNDLE-{arm}"] = {
            **pkg,
            "lineage_id": rec.lineage_id,
            "checkpoint_sha256": trained_shas[arm],
        }

    # 1. Default 20 MB cap embeds full checkpoints directly inside result.zip
    full_bundle_dir = tmp_path / "bundle_full"
    receipt_full = export_result_bundle(
        full_bundle_dir,
        campaign_id="TEST-BUNDLE-CAMPAIGN",
        source_commit=_COMMIT,
        status="COMPLETE",
        experiment_bundles=exp_bundles,
        checkpoint_root=ckpt_root,
    )
    assert receipt_full["whole_checkpoint_embedded"] is True
    assert receipt_full["checkpoint_count"] == 3
    assert receipt_full["experiment_count"] == 3
    assert receipt_full["result_zip_bytes"] <= 20 * 1024 * 1024

    verified = verify_result_bundle(full_bundle_dir, expected_source_commit=_COMMIT)
    assert verified["verified"] is True

    # Load F and B0 directly from result.zip in full_embedded mode
    loaded_full_f = CheckpointCatalog.load_from_result_zip(
        receipt_full["result_zip"],
        experiment_id="EXP-BUNDLE-F",
        expected_source_commit=_COMMIT,
        mode="raw",
    )
    assert loaded_full_f["source_archive"] == "result.zip (full_embedded)"
    assert loaded_full_f["architecture"] == "F"
    assert loaded_full_f["policy"] is not None
    full_f_funnel = ProcessFunnelAnalyzer().evaluate_suite(loaded_full_f["policy"], kernel, eval_worlds[:4])
    assert 0.0 <= full_f_funnel["overall_accuracy"] <= 1.0

    loaded_full_b0 = CheckpointCatalog.load_from_result_zip(
        receipt_full["result_zip"],
        experiment_id="EXP-BUNDLE-B0",
        expected_source_commit=_COMMIT,
        mode="raw",
    )
    assert loaded_full_b0["architecture"] == "B0"
    assert loaded_full_b0["policy"] is not None

    # Restore full CheckpointStore archive from checkpoint_full.zip
    restored = CheckpointCatalog.restore_full_checkpoint_zip(
        receipt_full["checkpoint_full_zip"],
        tmp_path / "restored_full_store",
        expected_source_commit=_COMMIT,
    )
    assert restored["verified"] is True
    assert restored["checkpoint_count"] == 3
    assert len(restored["catalog"].records) == 3

    # 2. Force compact_fp16 mode by capping max_result_zip_bytes strictly below full size
    compact_dir = tmp_path / "bundle_compact"
    receipt_compact = export_result_bundle(
        compact_dir,
        campaign_id="TEST-BUNDLE-CAMPAIGN",
        source_commit=_COMMIT,
        status="COMPLETE",
        experiment_bundles=exp_bundles,
        checkpoint_root=ckpt_root,
        max_result_zip_bytes=int(receipt_full["result_zip_bytes"]) - 1024,
    )
    assert receipt_compact["whole_checkpoint_embedded"] is False
    assert receipt_compact["result_zip_bytes"] < receipt_full["result_zip_bytes"]

    loaded_compact_f = CheckpointCatalog.load_from_result_zip(
        receipt_compact["result_zip"],
        experiment_id="EXP-BUNDLE-F",
        expected_source_commit=_COMMIT,
        mode="raw",
    )
    assert loaded_compact_f["source_archive"] == "result.zip (compact_fp16)"
    assert loaded_compact_f["architecture"] == "F"
    compact_hidden = loaded_compact_f["policy"].extract_representations(obs_batch)["decision_hidden"]
    full_hidden = loaded_full_f["policy"].extract_representations(obs_batch)["decision_hidden"]
    assert torch.allclose(compact_hidden, full_hidden, atol=5e-2)

    loaded_compact_b0 = CheckpointCatalog.load_from_result_zip(
        receipt_compact["result_zip"],
        experiment_id="EXP-BUNDLE-B0",
        expected_source_commit=_COMMIT,
        mode="raw",
    )
    assert loaded_compact_b0["source_archive"] == "result.zip (compact_fp16)"
    assert loaded_compact_b0["architecture"] == "B0"

    # 3. Tampering inside result.zip must fail closed
    tampered_zip = tmp_path / "tampered_result.zip"
    with zipfile.ZipFile(receipt_compact["result_zip"], "r") as zsrc, zipfile.ZipFile(
        tampered_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as zdst:
        for item in zsrc.infolist():
            data = zsrc.read(item.filename)
            if item.filename == "FINAL_STATUS.json":
                doc = json.loads(data.decode("utf-8"))
                doc["status"] = "PARTIAL"
                data = (json.dumps(doc) + "\n").encode("utf-8")
            zdst.writestr(item.filename, data)

    with pytest.raises(ValueError, match="checksum/size mismatch"):
        verify_single_planthir_zip(tampered_zip)
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        CheckpointCatalog.load_from_result_zip(tampered_zip, experiment_id="EXP-BUNDLE-F")


def test_level5_cartridge_runner_5_gates_and_surgical_interventions(tmp_path: Path) -> None:
    td = prepare_v8_training_data("D1", d1_worlds_per_family=16, sample_limit_for_cpu=12)
    base = _tiny_base_spec()
    eval_worlds, multihop_worlds = _sample_stratified_worlds()
    probe_worlds = list(eval_worlds[:8])
    kernel = V8EnvironmentKernel(enforce_tool_consumption=True)

    recipe_ctrl = Recipe(
        seed=7,
        total_updates=2,
        batch_size=4,
        micro_batch_size=2,
        lr=1e-4,
        schedule="cosine",
        precision="fp32",
    )
    recipe_bad = Recipe(
        seed=11,
        total_updates=2,
        batch_size=4,
        micro_batch_size=2,
        lr=5e-4,
        schedule="cosine",
        precision="fp32",
    )

    ctrl_arm = CartridgeArmSpec(
        arm_id="J",
        architecture_family="J",
        is_control=True,
        v8_spec=build_v8_model_spec("J", base_spec=base, pointer_dim=16),
        recipe=recipe_ctrl,
        dataset_identity=td.data_identity,
        order_seed=7,
        milestone_updates=(1, 2),
    )
    bad_treat = CartridgeArmSpec(
        arm_id="P",
        architecture_family="P",
        is_control=False,
        v8_spec=build_v8_model_spec("P", base_spec=base, pointer_dim=16),
        recipe=recipe_bad,
        dataset_identity=td.data_identity,
        order_seed=9,
        contrasts=(CausalContrastSpec("C-BAD", "arch", "J", "P", ("seed", "lr"), "copy"),),
    )
    bad_audit = audit_experimental_design(ctrl_arm, [bad_treat])
    assert bad_audit["passed"] is False
    assert {"SEED_MISMATCH", "LR_MISMATCH"}.issubset(set(bad_audit["violations"][0]["codes"]))

    p_arm = CartridgeArmSpec(
        arm_id="P",
        architecture_family="P",
        is_control=False,
        v8_spec=build_v8_model_spec("P", base_spec=base, pointer_dim=16),
        recipe=recipe_ctrl,
        dataset_identity=td.data_identity,
        order_seed=7,
        milestone_updates=(1, 2),
        contrasts=(CausalContrastSpec("C-P-VS-J", "arch", "J", "P", ("seed", "lr", "params"), "pointer_copy"),),
    )
    u_arm = CartridgeArmSpec(
        arm_id="U",
        architecture_family="U",
        is_control=False,
        v8_spec=build_v8_model_spec("U", base_spec=base, pointer_dim=16, recurrent_dim=16, recurrent_steps=2),
        recipe=recipe_ctrl,
        dataset_identity=td.data_identity,
        order_seed=7,
        milestone_updates=(1, 2),
        contrasts=(CausalContrastSpec("C-U-VS-J", "arch", "J", "U", ("seed", "lr", "params"), "recurrence"),),
    )

    cartridge = ExperimentCartridge(
        campaign_id="TEST-LEVEL5",
        program="Planthir-Level5-Test",
        source_commit=_COMMIT,
        kernel=kernel,
        control_arm=ctrl_arm,
        treatment_arms=(p_arm, u_arm),
        train_data=td,
        eval_worlds=tuple(eval_worlds[:8]),
        multihop_worlds=tuple(multihop_worlds[:4]),
        probe_train_worlds=tuple(probe_worlds),
        parameter_tolerance_ratio=2.0,
    )

    bundle_dir = tmp_path / "level5_bundle"
    out = CartridgeRunner().run(
        cartridge,
        work_root=tmp_path / "work",
        bundle_out_dir=bundle_dir,
        device="cpu",
        cpu_canary=True,
    )
    assert out["gates"]["all_5_gates_passed"] is True
    for arm_id in ("J", "P", "U"):
        parity = out["gates"]["gate5_bundle_and_live_parity_self_audit"]["parity_by_arm"][arm_id]
        assert parity["prediction_parity_rate"] == 1.0
        assert out["arms"][arm_id]["causal_diagnostics"]["learning_trajectory"]["checkpoints_evaluated"] == 2

    sa_j = out["arms"]["J"]["causal_diagnostics"]["surgical_ablation"]
    sa_p = out["arms"]["P"]["causal_diagnostics"]["surgical_ablation"]
    sa_u = out["arms"]["U"]["causal_diagnostics"]["surgical_ablation"]
    assert sa_j["ablation_supported"] is False and sa_j["mechanism_type"] == "none_baseline_control"
    assert sa_p["ablation_supported"] is True and sa_p["mechanism_type"] == "pointer_copy_gate"
    assert sa_u["ablation_supported"] is True and sa_u["mechanism_type"] == "recurrent_refinement"

    cds_u = out["arms"]["U"]["causal_diagnostics"]["compute_depth_scaling"]
    assert cds_u["supports_variable_depth"] is True
    assert set(cds_u["accuracy_by_steps"].keys()) == {"0", "1", "2", "4"}
    assert out["arms"]["J"]["causal_diagnostics"]["compute_depth_scaling"]["supports_variable_depth"] is False

    brief = out["ai_handoff_brief"]
    assert brief["all_5_gates_passed"] is True
    assert len(json.dumps(brief).encode("utf-8")) <= 8192
    assert out["token_fuel_report"]["estimated_token_reduction_pct"] >= 80.0
    assert (bundle_dir / "AI_HANDOFF_BRIEF.json").is_file()
    assert (bundle_dir / "LAUNCH_GATE_CERTIFICATE.json").is_file()
    assert verify_result_bundle(bundle_dir, expected_source_commit=_COMMIT)["verified"] is True
