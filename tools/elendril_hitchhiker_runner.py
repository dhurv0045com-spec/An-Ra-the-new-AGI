"""Command-line harness for Hitchhiker suite verification, Stage 0 audit, and CPU mechanical checks.

Never fabricates parent checkpoints or logits when weights are absent, never promotes
tiny CPU mechanical runs into scientific capability claims, and keeps engineered
decoders/canonicalizers attributed separately from learned representations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from typing import Any

from anra_v5.elendril.hitchhiker_data import (
    V6Dataset,
    assert_all_stage_disjoint,
    build_counterfactual_suite,
    build_health_probes,
    verify_counterfactual_suite,
)
from anra_v5.elendril.hitchhiker_diagnostics import (
    AvailabilityOnlyMaskPolicy,
    ExternalSlotCanonicalizerPolicy,
    GrammarControlHitchhikerPolicy,
    RawHitchhikerPolicy,
    admit_frozen_checkpoint,
    classify_diagnostic_outcomes,
    inspect_first_byte_distribution,
    score_entity_equivariance,
)
from anra_v5.elendril.hitchhiker_evaluation import (
    build_experiment_manifest,
    evaluate_counterfactual_suite,
    evaluate_formation_gate,
    evaluate_health_probes,
    evaluate_screening_gate,
    holm_adjust,
    paired_world_bootstrap_contrast,
)
from anra_v5.elendril.hitchhiker_protocol import (
    AUDITED_V717_COMMIT,
    PROTOCOL_SHA,
    resolve_scale_profile,
    stage_jobs,
)
from anra_v5.elendril.hitchhiker_training import (
    aux_penalty_gradient_probe,
    create_hitchhiker_trainer,
    run_hitchhiker_updates,
)
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import MODEL, configure_cpu
from anra_v5.elendril.v6_data import make_dataset
from anra_v5.elendril.v6_evaluation import ControlPolicy


DEFAULT_MECHANICAL_OUTPUT = Path("artifacts/elendril/hitchhiker/mechanical_verification.json")


def _compact_inspection(report: dict[str, Any]) -> dict[str, Any]:
    """Strip bulky full-260 probability vectors while retaining exact scalar & top-k diagnostics."""
    return {
        k: v
        for k, v in report.items()
        if k not in ("raw_logits_260", "raw_probs_260", "availability_masked_probs_260")
    }


def _evaluate_backend_decoders(
    backend: Any,
    suite: dict[str, Any],
    probes: dict[str, Any],
) -> dict[str, Any]:
    """Run raw, availability-only, and grammar decoders plus 260-way first-byte diagnostics."""
    decoder_specs = (
        ("raw", RawHitchhikerPolicy),
        ("availability-only", AvailabilityOnlyMaskPolicy),
        ("grammar", GrammarControlHitchhikerPolicy),
    )
    decoder_reports: dict[str, Any] = {}
    for name, policy_cls in decoder_specs:
        factory = policy_cls(backend)
        suite_eval = evaluate_counterfactual_suite(suite, factory)
        probes_eval = evaluate_health_probes(probes, factory, split=suite["split"])
        decoder_reports[name] = {
            "attribution_origin": suite_eval["attribution_origin"],
            "decoder_schema": suite_eval["decoder_schema"],
            "cohort_metrics": suite_eval["cohort_metrics"],
            "factor_summary": suite_eval["factor_summary"],
            "health_probe_scores": probes_eval["scores"],
            "_suite_eval": suite_eval,
        }

    representative_cases = (
        suite["cohorts"]["hop-1"][0],
        suite["cohorts"]["factor-entity-permutation"][0],
        probes["missing-bridge"][0],
        probes["missing-bridge"][1],
    )
    first_byte = [
        _compact_inspection(inspect_first_byte_distribution(backend, case, top_k=5))
        for case in representative_cases
    ]
    equivariance = score_entity_equivariance(
        backend,
        suite["cohorts"]["factor-entity-permutation"],
        suite["permutations"],
    )
    raw_metrics = decoder_reports["raw"]["cohort_metrics"]["hop-1"]
    masked_metrics = decoder_reports["availability-only"]["cohort_metrics"]["hop-1"]
    outcome = classify_diagnostic_outcomes(
        raw_metrics,
        masked_metrics,
        factor_metrics=decoder_reports["raw"]["factor_summary"],
        equivariance=equivariance,
    )
    return {
        "decoders": decoder_reports,
        "first_byte_inspections": first_byte,
        "entity_equivariance": equivariance,
        "diagnostic_classification": outcome,
    }


def run_verify_suite(*, worlds_per_hop: int = 16) -> dict[str, Any]:
    """Verify counterfactual suite construction, health probes, split custody, and oracle baseline."""
    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=worlds_per_hop, split="validation")
    probes = build_health_probes(seed=27173, worlds_per_probe=max(2, worlds_per_hop), split="validation")
    suite_check = verify_counterfactual_suite(suite)

    train_base = make_dataset("train", target_bytes=20_000, seed=606)
    probe_cases = tuple(c for name in ("distractor-trap", "belief-revision", "missing-bridge") for c in probes[name])
    val_bundle = V6Dataset(
        split="validation",
        cases=(*(c for cohort in suite["cohorts"].values() for c in cohort), *probe_cases),
        examples=(),
        samples=(),
        manifest={"version": "hitchhiker-verify-v1"},
    )
    split_check = assert_all_stage_disjoint(train_base, val_bundle)

    teacher_factory = ControlPolicy("fixed")
    teacher_suite = evaluate_counterfactual_suite(suite, teacher_factory)
    teacher_probes = evaluate_health_probes(probes, teacher_factory, stage_records=None, split="validation")
    return {
        "schema": "elendril-hitchhiker-suite-verify-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "source_commit": AUDITED_V717_COMMIT,
        "claim_status": "IMPLEMENTED",
        "worlds_per_hop": worlds_per_hop,
        "suite_verification": suite_check,
        "split_disjointness": split_check,
        "teacher_cohort_metrics": teacher_suite["cohort_metrics"],
        "teacher_health_probe_scores": teacher_probes["scores"],
    }


def run_stage0_audit(
    *,
    checkpoint_root: str | None,
    lineage: str,
    worlds_per_hop: int = 16,
) -> dict[str, Any]:
    """Audit frozen S20 weights if present, or emit honest CHECKPOINT_AUDIT_PENDING receipt."""
    admission = admit_frozen_checkpoint(
        checkpoint_root,
        lineage,
        expected_spec=resolve_scale_profile("S20"),
        device="cpu",
    )
    admission_receipt = {k: v for k, v in admission.items() if k != "backend"}
    if admission["status"] != "ADMITTED":
        decision_table = [
            {
                "mechanism": "Stage 0 frozen S20-low-lr audit",
                "status": "PENDING",
                "attribution": "model",
                "evidence_summary": f"Checkpoint weights unavailable at {checkpoint_root!r} for lineage {lineage!r}.",
                "next_action": "Mount archived S20-low-lr checkpoints and rerun stage0-audit.",
            }
        ]
        return build_experiment_manifest(
            source_commit=AUDITED_V717_COMMIT,
            stage="stage0_checkpoint_audit",
            stage0_status=admission_receipt,
            arm_runs=[],
            decision_table=decision_table,
            claim_status="PENDING",
        )

    configure_cpu()
    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=worlds_per_hop, split="validation")
    probes = build_health_probes(seed=27173, worlds_per_probe=max(2, worlds_per_hop), split="validation")
    diag = _evaluate_backend_decoders(admission["backend"], suite, probes)
    raw_summary = {
        k: v for k, v in diag["decoders"].items()
    }
    clean_decoders = {
        k: {dk: dv for dk, dv in v.items() if dk != "_suite_eval"}
        for k, v in raw_summary.items()
    }
    decision_table = [
        {
            "mechanism": f"Stage 0 frozen S20-low-lr audit ({lineage})",
            "status": "OBSERVED",
            "attribution": "model_and_decoder_comparison",
            "evidence_summary": f"Findings: {diag['diagnostic_classification']['findings']}",
            "next_action": "Proceed to Stage 1 15-run S3 comparison on GPU.",
        }
    ]
    manifest = build_experiment_manifest(
        source_commit=AUDITED_V717_COMMIT,
        stage="stage0_checkpoint_audit",
        stage0_status=admission_receipt,
        arm_runs=[],
        decision_table=decision_table,
        claim_status="OBSERVED",
    )
    manifest["stage0_diagnostics"] = {
        "decoders": clean_decoders,
        "first_byte_inspections": diag["first_byte_inspections"],
        "entity_equivariance": diag["entity_equivariance"],
        "diagnostic_classification": diag["diagnostic_classification"],
    }
    return manifest


def run_cpu_mechanical_check(*, worlds_per_hop: int = 2) -> dict[str, Any]:
    """Run bounded tiny CPU mechanical verification across protocol, trainers, and gates."""
    configure_cpu()
    tiny_spec = replace(MODEL, vocabulary_size=UTF8.vocabulary_size, context_length=512)
    param_total = tiny_spec.parameter_receipt().total
    if param_total > 100_000:
        raise ValueError(f"CPU mechanical model unexpectedly large: {param_total}")

    custom_spec = resolve_scale_profile(
        "S9",
        {
            "spec": tiny_spec,
            "parameters": param_total,
            "model_sha256": tiny_spec.sha256(),
        },
    )

    suite = build_counterfactual_suite(seed=17071, worlds_per_hop=worlds_per_hop, split="validation")
    probes = build_health_probes(seed=27173, worlds_per_probe=max(2, worlds_per_hop), split="validation")
    suite_check = verify_counterfactual_suite(suite)
    probe_bundle = {"split": "validation", "cohorts": probes}
    train_base = make_dataset("train", target_bytes=12_000, seed=606)
    heldout = (suite, probe_bundle)

    s1_jobs = [j for j in stage_jobs("stage1_s3_screen") if j["seed"] == 7]
    s2_jobs = [j for j in stage_jobs("stage2_followups") if j["seed"] == 7 and j["arm"] != "S20-contrast"]
    jobs = [*s1_jobs, *s2_jobs]

    arm_runs: list[dict[str, Any]] = []
    trainers: dict[str, Any] = {}
    aux_probe_result: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        missing_audit = admit_frozen_checkpoint(tmp_path / "missing_checkpoints", "S20-low-lr-i7-o7")
        if missing_audit["status"] != "CHECKPOINT_AUDIT_PENDING":
            raise RuntimeError("expected CHECKPOINT_AUDIT_PENDING for missing parent checkpoint root")

        ckpt_root = tmp_path / "mechanical_checkpoints"
        for job in jobs:
            trainer, stream = create_hitchhiker_trainer(
                job,
                source_commit=AUDITED_V717_COMMIT,
                root=ckpt_root,
                base_data=train_base,
                heldout_suites=heldout,
                device="cpu",
                cpu_canary=True,
                spec_override=custom_spec,
                updates_override=2,
                batch_override=4,
            )
            if job["arm"] == "C+penalty":
                aux_probe_result = aux_penalty_gradient_probe(trainer)
                if aux_probe_result["status"] != "PASS":
                    raise RuntimeError("aux_penalty_gradient_probe failed on C+penalty")
            run_receipt = run_hitchhiker_updates(trainer, save_every=2)
            ckpt_sha = trainer.store.latest_sha256()
            trainers[job["arm"]] = trainer
            arm_runs.append(
                {
                    "job_id": job["job_id"],
                    "arm": job["arm"],
                    "stage": job["stage"],
                    "attribution_origin": job["attribution_origin"],
                    "parameters": param_total,
                    "updates_executed": run_receipt["updates_executed"],
                    "supervised_tokens": run_receipt["supervised_tokens"],
                    "input_positions": run_receipt["input_positions"],
                    "examples_processed": run_receipt["examples_processed"],
                    "final_loss": run_receipt["final_loss"],
                    "checkpoint_sha256": ckpt_sha,
                    "stream_counts": {
                        "accepted_candidates": stream["stream_metrics"]["accepted_candidates"],
                        "rejected_candidates": stream["stream_metrics"]["rejected_candidates"],
                    },
                }
            )

        # Verify local frozen-checkpoint admission + Stage 0 three-decoder pipeline on B0 checkpoint.
        b0_lineage = s1_jobs[0]["job_id"]
        admitted_b0 = admit_frozen_checkpoint(
            ckpt_root,
            b0_lineage,
            expected_spec=tiny_spec,
            expected_update=2,
            device="cpu",
        )
        if admitted_b0["status"] != "ADMITTED":
            raise RuntimeError("expected local tiny checkpoint admission to succeed")
        b0_stage0 = _evaluate_backend_decoders(admitted_b0["backend"], suite, probes)

        # Evaluate AB raw and D-external canonicalizer wrapper for contrast & gate checks.
        ab_backend = trainers["AB"].backend
        ab_suite_eval = evaluate_counterfactual_suite(suite, RawHitchhikerPolicy(ab_backend))
        ab_probes_eval = evaluate_health_probes(probes, RawHitchhikerPolicy(ab_backend), split="validation")
        ab_equiv = score_entity_equivariance(
            ab_backend,
            suite["cohorts"]["factor-entity-permutation"],
            suite["permutations"],
        )

        dext_backend = trainers["D-external"].backend
        dext_factory = ExternalSlotCanonicalizerPolicy(dext_backend, assignment="first-mention")
        dext_suite_eval = evaluate_counterfactual_suite(suite, dext_factory)

    screening_gate = evaluate_screening_gate(
        {7: ab_suite_eval},
        {7: b0_stage0["decoders"]["raw"]["_suite_eval"]},
        extra_contrasts={
            "ab_vs_b0_perm": paired_world_bootstrap_contrast(
                ab_suite_eval["records"]["factor-entity-permutation"]["episodes"],
                b0_stage0["decoders"]["raw"]["_suite_eval"]["records"]["factor-entity-permutation"]["episodes"],
            )["p_value"],
            "dext_vs_b0_perm": paired_world_bootstrap_contrast(
                dext_suite_eval["records"]["factor-entity-permutation"]["episodes"],
                b0_stage0["decoders"]["raw"]["_suite_eval"]["records"]["factor-entity-permutation"]["episodes"],
            )["p_value"],
        },
    )
    formation_gate = evaluate_formation_gate(
        ab_suite_eval,
        equivariance_report=ab_equiv,
        confirmation_worlds_per_cohort=worlds_per_hop,
    )
    if formation_gate["status"] != "MECHANICAL_ONLY_NOT_FORMATION_QUALIFIED":
        raise RuntimeError("tiny CPU mechanical check was unexpectedly formation-qualified")

    decision_table = [
        {
            "mechanism": "Stage 0 frozen S20 audit harness",
            "status": "IMPLEMENTED",
            "attribution": "model_vs_mask_vs_grammar",
            "evidence_summary": f"Missing-weights receipt={missing_audit['status']}; local admission={admitted_b0['status']}.",
            "next_action": "Execute against real S20-low-lr checkpoints when mounted.",
        },
        {
            "mechanism": "A (Procedural world diversity)",
            "status": "IMPLEMENTED",
            "attribution": "model",
            "evidence_summary": "Fresh split-disjoint stream built and ran 2 tiny CPU updates.",
            "next_action": "Execute 3-seed S3 screen on GPU.",
        },
        {
            "mechanism": "B (Entity permutation invariance)",
            "status": "IMPLEMENTED",
            "attribution": "model",
            "evidence_summary": "Episode-consistent A-P relabeling stream ran 2 tiny CPU updates.",
            "next_action": "Execute 3-seed S3 screen on GPU.",
        },
        {
            "mechanism": "AB (Fresh + Entity relabeling)",
            "status": "IMPLEMENTED",
            "attribution": "model",
            "evidence_summary": f"2 tiny CPU updates verified; formation_gate={formation_gate['status']}.",
            "next_action": "Execute 3-seed S3 screen on GPU.",
        },
        {
            "mechanism": "C (Matched coverage across availability)",
            "status": "IMPLEMENTED",
            "attribution": "model",
            "evidence_summary": "Matched completeness x availability stream ran 2 tiny CPU updates.",
            "next_action": "Execute 3-seed S3 screen on GPU.",
        },
        {
            "mechanism": "C+penalty (Aux illegal-action penalty)",
            "status": "IMPLEMENTED",
            "attribution": "model",
            "evidence_summary": f"Gradient probe={aux_probe_result['status']}; ran 2 tiny CPU updates.",
            "next_action": "Conditional Stage 2 arm if availability illegality persists under C.",
        },
        {
            "mechanism": "D-external (Public canonicalizer control)",
            "status": "IMPLEMENTED",
            "attribution": dext_suite_eval["attribution_origin"],
            "evidence_summary": "Public-only canonicalization + dereferencing ran and replayed cleanly.",
            "next_action": "Run as non-learned diagnostic baseline in Stage 2.",
        },
        {
            "mechanism": "Second-wave E/F/G/H",
            "status": "PENDING",
            "attribution": "deferred",
            "evidence_summary": "Explicitly deferred until Stage 1-3 empirical bottlenecks justify.",
            "next_action": "Remain deferred until Stage 1-3 screening completes.",
        },
    ]

    manifest = build_experiment_manifest(
        source_commit=AUDITED_V717_COMMIT,
        stage="cpu_mechanical_verification",
        stage0_status=missing_audit,
        arm_runs=arm_runs,
        decision_table=decision_table,
        claim_status="IMPLEMENTED",
    )
    manifest["mechanical_checks"] = {
        "cpu_model_spec": asdict(tiny_spec),
        "cpu_model_parameters": param_total,
        "suite_verification": suite_check,
        "aux_penalty_gradient_probe": aux_probe_result,
        "admitted_local_b0_decoders": {
            k: {dk: dv for dk, dv in v.items() if dk != "_suite_eval"}
            for k, v in b0_stage0["decoders"].items()
        },
        "admitted_local_b0_first_byte": b0_stage0["first_byte_inspections"],
        "admitted_local_b0_classification": b0_stage0["diagnostic_classification"],
        "ab_mechanical_probes": ab_probes_eval["scores"],
        "ab_entity_equivariance": ab_equiv,
        "d_external_factor_summary": dext_suite_eval["factor_summary"],
        "d_external_attribution_origin": dext_suite_eval["attribution_origin"],
        "screening_gate_mechanical": screening_gate,
        "formation_gate_mechanical": formation_gate,
        "holm_example": holm_adjust([0.01, 0.04, 0.20]),
        "dataset_smoke_worlds": len(train_base.cases),
    }
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Elendril Hitchhiker evidence runner")
    parser.add_argument(
        "--mode",
        choices=("verify-suite", "stage0-audit", "cpu-mechanical-check"),
        default="verify-suite",
        help="Execution mode (verify-suite, stage0-audit, or bounded cpu-mechanical-check).",
    )
    parser.add_argument("--checkpoint-root", default=None, help="Parent checkpoint directory for stage0-audit.")
    parser.add_argument("--lineage", default="S20-low-lr-i7-o7", help="Parent lineage for stage0-audit.")
    parser.add_argument("--worlds-per-hop", type=int, default=None, help="Override worlds per hop.")
    parser.add_argument("--output", default=None, help="Path to write JSON output artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.mode == "verify-suite":
        payload = run_verify_suite(worlds_per_hop=args.worlds_per_hop or 16)
        out_path = args.output
    elif args.mode == "stage0-audit":
        payload = run_stage0_audit(
            checkpoint_root=args.checkpoint_root,
            lineage=args.lineage,
            worlds_per_hop=args.worlds_per_hop or 16,
        )
        out_path = args.output
    else:
        worlds = args.worlds_per_hop or 2
        if not 1 <= worlds <= 8:
            raise ValueError("cpu-mechanical-check limits --worlds-per-hop to 1..8")
        payload = run_cpu_mechanical_check(worlds_per_hop=worlds)
        out_path = args.output or str(DEFAULT_MECHANICAL_OUTPUT)

    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if out_path:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(serialized, encoding="utf-8")
        print(json.dumps({"status": "OK", "mode": args.mode, "output": str(dest), "claim_status": payload.get("claim_status")}))
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
