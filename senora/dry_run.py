"""End-to-end dry-run and plumbing certification for Senora P35 experiments.

Executes 2 synthetic mock steps through the data pipeline, WSD scheduler,
state transition engine, atomic CAS checkpoint store, and 4-tier evaluator
without consuming research compute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from e0_cognition.evaluation_generators import Split, build_evaluation_suite
from senora.data_pipeline import (
    CURSOR_SCHEMA,
    DataPipeline,
    MIXTURE_COGNITION_15,
)
from senora.evaluator import CasePrediction, SenoraEvaluator
from senora.experiment_design import P35_MODEL_SPEC, build_p35_cms1_plan
from senora.experiment_identity import ExperimentIdentity, SCHEMA as IDENTITY_SCHEMA
from senora.trainer import P35Trainer, P35TrainerConfig
from v5_training.checkpoint import CheckpointStore, REQUIRED_COMPONENTS
from v5_training.state import CursorState, IdentityBindings, TrainingState


def execute_dry_run(*, output_receipt: Path | None = None) -> dict[str, Any]:
    start_time = time.perf_counter()
    print("============================================================")
    print("SENORA P35-CMS-1: END-TO-END PLUMBING DRY-RUN")
    print("============================================================")

    # 1. Plan and Identity
    print("\n[Stage 1/5] Freezing experiment plan and cryptographic identity...")
    plan = build_p35_cms1_plan()
    sha_dummy = "0" * 63 + "1"
    identity = ExperimentIdentity(
        schema=IDENTITY_SCHEMA,
        experiment_id=plan.experiment_id,
        source_commit_sha="a" * 40,
        model_spec_sha256=sha_dummy,
        model_constructor_sha256=sha_dummy,
        tokenizer_artifact_sha256=sha_dummy,
        corpus_manifest_sha256=sha_dummy,
        data_manifest_sha256=sha_dummy,
        pack_manifest_sha256=sha_dummy,
        generator_version="e0-train/0.2.0",
        split_identities={"training": sha_dummy, "fresh": sha_dummy},
        optimizer_spec={"family": "AdamW", "lr": 3e-4, "weight_decay": 0.1},
        schedule_spec={"family": "WSD", "warmup_tokens": 1_500_000},
        precision="bf16-mixed-fp32-master",
        token_budget=plan.arms[1]["token_budget"],
        tokens_per_update=131_072,
        random_seeds=(42, 43),
        evaluator_spec={"suite_version": "e0-eval/0.4.0"},
        scorer_firewall_status="BYPASS_CANDIDATE_LOGPROB_RAW_CORE_ONLY",
        statistical_protocol={"paired_sign_test": True, "resamples": 10_000},
        promotion_criteria={"min_ood_delta": 0.25},
        abort_criteria={
            "max_loss_regression_fraction": 0.03,
            "fail_on_nan_loss": True,
            "fail_on_gradient_explosion": True,
            "fail_on_stagnation": True,
        },
    )
    identity.assert_valid()
    print(f"  Plan: {plan.title} (ID: {plan.experiment_id})")
    print(f"  Plan SHA-256: {plan.sha256()[:16]}...")
    print(f"  Identity SHA-256: {identity.sha256()[:16]}...")

    # 2. Data Pipeline & Contamination Guard
    print("\n[Stage 2/5] Initializing data curriculum and testing contamination guards...")
    pipeline = DataPipeline(
        pack_manifest=None,
        recipe=MIXTURE_COGNITION_15,
        sequence_length=2048,
        batch_size=64,
        allow_synthetic_mock=True,
    )
    pipeline.assert_no_contamination(
        training_template_ids=["train.causal.registry", "train.causal.revision"],
        evaluation_template_ids={"dev.eval.binding", "fresh.eval.state"},
    )
    init_cursor = CursorState(
        schema=CURSOR_SCHEMA,
        pack_manifest_sha256=sha_dummy,
        shard_ordinal=0,
        sequence_ordinal=0,
        token_offset=0,
    )
    batches = list(pipeline.mock_stream(initial_cursor=init_cursor, total_batches=2))
    print(f"  Batches produced: {len(batches)}")
    print(f"  Batch 1 token count: {batches[0].batch_token_count:,}")
    print(f"  Batch 1 source allocation: {batches[0].tokens_by_source}")

    # 3. Trainer & Checkpointing
    print("\n[Stage 3/5] Initializing P35 trainer and executing 2 synthetic state transitions...")
    with tempfile.TemporaryDirectory(prefix="senora-dry-run-") as temp_dir:
        checkpoint_dir = Path(temp_dir)
        trainer_config = P35TrainerConfig(
            model_spec=P35_MODEL_SPEC,
            token_budget=batches[0].batch_token_count * 2,
            tokens_per_update=batches[0].batch_token_count,
            learning_rate=3e-4,
            weight_decay=0.1,
            gradient_clip_norm=1.0,
            query_swap_lambda=0.10,
            remote_authorized=True,  # authorized for this synthetic dry run
        )
        identities = IdentityBindings(
            schema="anra-v5-identity-bindings/v1",
            source_commit="a" * 40,
            model_spec_sha256=sha_dummy,
            tokenizer_sha256=sha_dummy,
            data_manifest_sha256=sha_dummy,
            pack_manifest_sha256=sha_dummy,
            run_spec_sha256=sha_dummy,
            optimizer_spec_sha256=sha_dummy,
            schedule_spec_sha256=sha_dummy,
            curriculum_spec_sha256=sha_dummy,
        )
        trainer = P35Trainer(
            trainer_config,
            identity_bindings=identities,
            checkpoint_directory=checkpoint_dir,
        )

        # Initialize State (Step 0)
        state = trainer.initialize_training_state(
            initial_cursor=init_cursor,
            rng_state_sha256=sha_dummy,
        )
        print(f"  Initial State: update={state.global_update}, tokens={state.cumulative_tokens}")

        # Execute Step 1
        rng_step1 = hashlib.sha256(b"rng_step_1").hexdigest()
        state = trainer.advance_step(
            state,
            tokens_by_source=batches[0].tokens_by_source,
            new_cursor=batches[0].new_cursor,
            new_rng_state_sha256=rng_step1,
            loss_value=3.1415,
            gradient_norm=0.72,
        )
        print(f"  Step 1 Completed: update={state.global_update}, tokens={state.cumulative_tokens:,}, loss=3.1415, grad_norm=0.72")

        def _make_payloads(s: TrainingState, label: str) -> dict[str, bytes]:
            from dataclasses import asdict
            return {
                "model.bin": f"mock_model_weights_{label}".encode("utf-8"),
                "optimizer.bin": f"mock_optimizer_moments_{label}".encode("utf-8"),
                "rng.bin": f"mock_rng_state_{label}".encode("utf-8"),
                "scheduler.json": json.dumps({"lr": 3e-4, "tokens": s.cumulative_tokens}, sort_keys=True).encode("utf-8"),
                "cursor.json": json.dumps(asdict(s.cursor), sort_keys=True).encode("utf-8"),
                "ledger.json": json.dumps(dict(s.tokens_by_source), sort_keys=True).encode("utf-8"),
                "training_state.json": json.dumps(s.canonical(), sort_keys=True, separators=(",", ":")).encode("utf-8"),
            }

        # Save Checkpoint 1
        dummy_payloads = _make_payloads(state, "step1")
        ckpt1_sha = trainer.save_checkpoint(state, payloads=dummy_payloads, expected_parent_sha256=None)
        print(f"  CAS Checkpoint 1 Published: {ckpt1_sha[:16]}...")

        # Execute Step 2
        rng_step2 = hashlib.sha256(b"rng_step_2").hexdigest()
        state = trainer.advance_step(
            state,
            tokens_by_source=batches[1].tokens_by_source,
            new_cursor=batches[1].new_cursor,
            new_rng_state_sha256=rng_step2,
            loss_value=2.8540,
            gradient_norm=0.68,
            parent_checkpoint_sha256=ckpt1_sha,
        )
        print(f"  Step 2 Completed: update={state.global_update}, tokens={state.cumulative_tokens:,}, loss=2.8540, grad_norm=0.68")

        # Save Checkpoint 2
        dummy_payloads2 = _make_payloads(state, "step2")
        ckpt2_sha = trainer.save_checkpoint(state, payloads=dummy_payloads2, expected_parent_sha256=ckpt1_sha)
        print(f"  CAS Checkpoint 2 Published: {ckpt2_sha[:16]}...")

        # Test Clean Restoration from Checkpoint 2
        print("\n[Stage 4/5] Testing clean state restoration from atomic CAS store...")
        store = CheckpointStore(root=checkpoint_dir, lineage_id=state.lineage_id)
        restored_state, restored_payloads = store.restore()
        assert restored_state == state, "Restored state does not match memory state"
        assert set(restored_payloads.keys()) == REQUIRED_COMPONENTS
        print(f"  State restoration: EXACT MATCH (verified {len(REQUIRED_COMPONENTS)} components)")

    # 5. 4-Tier Evaluator Verification
    print("\n[Stage 5/5] Testing 4-tier evaluator harness on synthetic test cases...")
    suite = build_evaluation_suite(Split.DEVELOPMENT, seed=101, groups_per_family=1)
    evaluator = SenoraEvaluator(suite, scorer_firewall_status="FAIL_DEVELOPMENT_POLICY")
    predictions = [
        CasePrediction(
            case_id=case.case_id,
            raw_output=case.answer,
            constrained_output=case.answer,
            assisted_output=case.answer,
            candidate_logprobs={cand: 1.0 for cand in case.candidates},
        )
        for case in suite.cases
    ]
    summary = evaluator.evaluate_predictions(predictions, general_substrate_loss=2.15)
    print(f"  Cases evaluated: {summary.case_count}")
    print(f"  Raw Core Accuracy: {summary.raw_core_accuracy * 100:.1f}%")
    print(f"  Constrained Accuracy: {summary.constrained_accuracy * 100:.1f}%")
    print(f"  Assisted Accuracy: {summary.assisted_accuracy * 100:.1f}%")
    print(f"  Natural Analogue Macro Accuracy: {summary.natural_analogue_macro_accuracy * 100:.1f}%")
    print(f"  Scorer Firewall Status: {summary.candidate_scoring_status}")
    assert "BLOCKED_BY_SCORER_FIREWALL" in summary.candidate_scoring_status

    duration = time.perf_counter() - start_time
    receipt = {
        "schema": "senora-p35-dry-run-receipt/v1",
        "status": "PASS_PLUMBING_CERTIFIED",
        "experiment_id": plan.experiment_id,
        "identity_sha256": identity.sha256(),
        "stages_passed": [
            "plan_and_identity",
            "data_pipeline_and_contamination_guard",
            "two_step_training_transitions",
            "atomic_cas_checkpoint_save_and_restore",
            "four_tier_evaluator_certification",
        ],
        "step_1_loss": 3.1415,
        "step_2_loss": 2.8540,
        "checkpoint_restored_clean": True,
        "scorer_firewall_gate_enforced": True,
        "execution_duration_seconds": round(duration, 3),
    }

    if output_receipt:
        output_receipt.parent.mkdir(parents=True, exist_ok=True)
        output_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nSaved dry-run execution receipt to: {output_receipt}")

    print("\n============================================================")
    print(f"DRY-RUN RESULT: PASS ({duration:.2f}s) - PLUMBING FULLY CERTIFIED")
    print("============================================================")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Senora P35 end-to-end dry run")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v5/p35_cms1_dry_run_receipt.json"),
        help="Path to emit the execution receipt",
    )
    args = parser.parse_args()
    execute_dry_run(output_receipt=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())