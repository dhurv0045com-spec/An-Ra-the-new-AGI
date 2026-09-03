"""Evaluation protocol engine and experiment/preregistration tests + red-team."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v5_evaluation.checkpoint_adapter import CheckpointBackedV5Adapter
from v5_evaluation.protocol import (
    EvaluationProtocol,
    run_evaluation,
)
from v5_experiments.preregistration import (
    ExperimentChronology,
    ExperimentSpec,
    PreregistrationReceipt,
    TrainingInterventionRecord,
    assert_matched_arms,
)
from v5_registry.subject import CoreSubjectManifest
from tests.test_v5_checkpoint_adapter import TINY_SPEC, _adapter, _identity


def _subject() -> CoreSubjectManifest:
    return CoreSubjectManifest.create(
        checkpoint_sha256="a" * 64,
        checkpoint_file_sha256="a" * 64,
        parameter_sha256="b" * 64,
        model_spec_sha256=TINY_SPEC.sha256(),
        tokenizer_artifact_sha256="d" * 64,
        tokenizer_identity_sha256="e" * 64,
        training_spec_sha256="1" * 64,
        data_manifest_sha256="2" * 64,
        pack_manifest_sha256="3" * 64,
        optimizer_spec_sha256="4" * 64,
        schedule_spec_sha256="5" * 64,
        curriculum_spec_sha256="6" * 64,
        source_commit="0123456789abcdef0123456789abcdef01234567",
        parent_checkpoint_sha256=None,
        global_update=2,
        cumulative_training_tokens=8_192,
        stage="SOFTWARE_EVAL",
        seed=3,
        custody="local-ephemeral-checkpoint-store",
        creation_receipt_sha256="7" * 64,
    )


def _protocol(mode: str = "CONSTRAINED_SELECTION") -> EvaluationProtocol:
    return EvaluationProtocol(
        protocol_id="proto-mini",
        generator_id="miniature-binding-generator",
        generator_sha256="a" * 64,
        split="software_eval",
        seed=1,
        n_cases=2,
        decoding_mode=mode,
        candidate_scoring_mode="summed-suffix-logprob",
        metrics=("exact_match",),
        statistical_rule="wilson-95-lcb",
    )


TASKS = [
    {
        "task_id": "t1",
        "cluster_id": "c1",
        "family": "query_binding",
        "difficulty": "easy",
        "prompt": "The zibble is crimson. What color is the zibble?",
        "candidates": (" crimson", " blue"),
        "gold": " crimson",
    },
    {
        "task_id": "t2",
        "cluster_id": "c1",
        "family": "query_binding",
        "difficulty": "easy",
        "prompt": "The woggle is blue. What color is the woggle?",
        "candidates": (" crimson", " blue"),
        "gold": " blue",
    },
]


class ProtocolTest(unittest.TestCase):
    def test_protocol_validation(self) -> None:
        protocol = _protocol()
        protocol.assert_valid()
        self.assertEqual(protocol.sha256(), protocol.sha256())
        with self.assertRaises(ValueError):
            EvaluationProtocol(
                protocol_id="p",
                generator_id="g",
                generator_sha256="a" * 64,
                split="fresh-ish",
                seed=1,
                n_cases=1,
                decoding_mode="RAW_FREE_GENERATION",
                candidate_scoring_mode="sum",
                metrics=("m",),
                statistical_rule="wilson",
            ).assert_valid()

    def test_run_evaluation_produces_task_level_evidence(self) -> None:
        adapter = _adapter()
        receipt, evidence = run_evaluation(
            protocol=_protocol(),
            subject=_subject(),
            adapter=adapter,
            tasks=TASKS,
        )
        self.assertEqual(receipt.n_tasks, 2)
        self.assertEqual(len(evidence), 2)
        self.assertTrue(receipt.derived_from_task_evidence)
        for record in evidence:
            self.assertEqual(record.evaluation_mode, "CONSTRAINED_SELECTION")
            self.assertIn(record.raw_output, (" crimson", " blue"))
            self.assertEqual(record.checkpoint_sha256, "a" * 64)
        # aggregates derive from the evidence, not asserted independently
        expected = sum(1 for r in evidence if r.correct) / len(evidence)
        self.assertEqual(receipt.aggregate_correct_rate, expected)

    def test_raw_scoring_mode(self) -> None:
        adapter = _adapter()
        receipt, evidence = run_evaluation(
            protocol=_protocol("RAW_CANDIDATE_SCORING"),
            subject=_subject(),
            adapter=adapter,
            tasks=TASKS,
        )
        for record in evidence:
            self.assertEqual(len(record.candidate_scores), 2)

    def test_assisted_mode_rejected_without_support(self) -> None:
        adapter = _adapter()
        with self.assertRaises(ValueError):
            run_evaluation(
                protocol=_protocol("ORACLE_ASSISTED"),
                subject=_subject(),
                adapter=adapter,
                tasks=TASKS,
            )


def _spec(experiment_id: str, **overrides) -> ExperimentSpec:
    fields = dict(
        experiment_id=experiment_id,
        hypothesis="verified cognition data improves binding beyond matched language modeling",
        intervention=TrainingInterventionRecord(
            hypothesis="cognition mixture causes binding gains",
            mechanism_target="query-conditioned binding",
            treatment_definition="15% verified cognition mixture",
            control_definition="0% cognition mixture",
            expected_behavioral_effect="binding accuracy increases at matched compute",
            expected_failure_profile_effect="binding failures shift from absent to partial",
            risks=("template memorization", "substrate regression"),
        ),
        parent_checkpoint_sha256s=("a" * 64,),
        model_spec_sha256="b" * 64,
        tokenizer_artifact_sha256="c" * 64,
        training_spec_sha256="d" * 64,
        data_manifest_sha256="e" * 64,
        optimizer_spec_sha256="1" * 64,
        schedule_spec_sha256="2" * 64,
        token_budget=100_000,
        seeds=(1, 2),
        evaluation_protocol_sha256="3" * 64,
        promotion_rule="conjunctive gates per frozen spec",
        stop_rule="two consecutive tier-1 declines",
    )
    fields.update(overrides)
    return ExperimentSpec(**fields)


class ExperimentTest(unittest.TestCase):
    def test_freeze_and_tamper_detection(self) -> None:
        spec = _spec("p35-a")
        receipt = PreregistrationReceipt.freeze(spec)
        self.assertTrue(receipt.verify())
        tampered = dict(receipt.spec)
        tampered["token_budget"] = 999_999_999
        forged = PreregistrationReceipt(
            receipt_schema=receipt.receipt_schema, spec=tampered, spec_sha256=receipt.spec_sha256
        )
        self.assertFalse(forged.verify())

    def test_chronology_enforces_order(self) -> None:
        spec = _spec("chrono")
        chronology = ExperimentChronology.begin(spec, question_payload_sha256="a" * 64)
        chronology = chronology.record(stage="PREREGISTRATION", payload_sha256="b" * 64)
        chronology = chronology.record(stage="CODE_FREEZE", payload_sha256="c" * 64)
        with self.assertRaises(ValueError):
            chronology.record(stage="PREREGISTRATION", payload_sha256="d" * 64)
        with self.assertRaises(ValueError):
            chronology.record(stage="ANALYSIS", payload_sha256="d" * 64)

    def test_matched_arms_fail_closed_on_undeclared_drift(self) -> None:
        control = _spec("arm-control")
        treatment = _spec("arm-treatment")
        # same everything (allowed: experiment_id differs only as identity)
        result = assert_matched_arms(control, treatment, allowed_differences=("experiment_id",))
        self.assertTrue(result["matched"])
        drifted = _spec("arm-treatment", token_budget=999_999)
        with self.assertRaises(ValueError):
            assert_matched_arms(control, drifted, allowed_differences=("experiment_id",))

    def test_intervention_record_requires_risks(self) -> None:
        with self.assertRaises(ValueError):
            TrainingInterventionRecord(
                hypothesis="h",
                mechanism_target="m",
                treatment_definition="t",
                control_definition="c",
                expected_behavioral_effect="e",
                expected_failure_profile_effect="f",
                risks=(),
            )


if __name__ == "__main__":
    unittest.main()
