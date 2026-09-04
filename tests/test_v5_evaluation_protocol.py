"""Evaluation protocol engine and experiment/preregistration tests + red-team."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v5_evaluation.checkpoint_adapter import (
    SCORING_CONTRACT_SHA256,
    CheckpointBackedV5Adapter,
)
from v5_evaluation.fixture import TaskFixtureBatch
from v5_evaluation.protocol import (
    EvaluationProtocol,
    run_evaluation,
    verify_evidence_artifact,
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
        candidate_scoring_mode=SCORING_CONTRACT_SHA256,
        metrics=("EXACT_ACCURACY",),
        statistical_rule="WILSON_BINOMIAL",
    )


def _fixture(**overrides) -> TaskFixtureBatch:
    fields: dict[str, object] = {
        "generator_id": "miniature-binding-generator",
        "generator_sha256": "a" * 64,
        "generator_config_sha256": "b" * 64,
        "seed": 1,
        "split": "software_eval",
        "cases": [dict(task, split="software_eval") for task in TASKS],
    }
    fields.update(overrides)
    return TaskFixtureBatch.freeze(**fields)  # type: ignore[arg-type]


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

    def _run(self, protocol=None, fixture=None, adapter=None, clock=None):
        if adapter is None:
            adapter = _adapter()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            receipt, evidence = run_evaluation(
                protocol=protocol or _protocol(),
                subject=_subject(),
                adapter=adapter,
                fixture=fixture or _fixture(),
                evidence_path=path,
                clock=clock,
            )
            artifact = path.read_bytes()
        return receipt, evidence, artifact

    def test_run_evaluation_produces_task_level_evidence(self) -> None:
        receipt, evidence, _artifact = self._run()
        self.assertEqual(receipt.n_tasks, 2)
        self.assertEqual(len(evidence), 2)
        self.assertTrue(receipt.derived_from_task_evidence)
        for record in evidence:
            self.assertEqual(record.evaluation_mode, "CONSTRAINED_SELECTION")
            self.assertIn(record.raw_output, (" crimson", " blue"))
            self.assertEqual(record.checkpoint_sha256, "a" * 64)
        expected = sum(1 for r in evidence if r.correct) / len(evidence)
        self.assertEqual(receipt.aggregate_correct_rate, expected)
        self.assertEqual(dict(receipt.metric_values)["EXACT_ACCURACY"], expected)
        self.assertEqual(receipt.statistical_rule, "WILSON_BINOMIAL")
        self.assertEqual(len(receipt.evidence_artifact_sha256), 64)

    def test_raw_scoring_mode(self) -> None:
        import dataclasses

        protocol = dataclasses.replace(
            _protocol("RAW_CANDIDATE_SCORING"), metrics=("CANDIDATE_RANK1",)
        )
        receipt, evidence, _artifact = self._run(protocol=protocol)
        for record in evidence:
            self.assertEqual(len(record.candidate_scores), 2)

    def test_assisted_mode_rejected_without_support(self) -> None:
        adapter = _adapter()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_evaluation(
                    protocol=_protocol("ORACLE_ASSISTED"),
                    subject=_subject(),
                    adapter=adapter,
                    fixture=_fixture(),
                    evidence_path=Path(directory) / "evidence.jsonl",
                )

    def test_n_must_be_exact(self) -> None:
        import dataclasses

        adapter = _adapter()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            for bad_n in (1, 3):
                with self.assertRaises(ValueError, msg=f"n={bad_n}"):
                    run_evaluation(
                        protocol=dataclasses.replace(_protocol(), n_cases=bad_n),
                        subject=_subject(),
                        adapter=adapter,
                        fixture=_fixture(),
                        evidence_path=path,
                    )

    def test_split_seed_generator_scoring_must_match(self) -> None:
        import dataclasses

        adapter = _adapter()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            with self.assertRaises(ValueError):
                run_evaluation(
                    protocol=_protocol(), subject=_subject(), adapter=adapter,
                    fixture=_fixture(split="development"), evidence_path=path,
                )
            with self.assertRaises(ValueError):
                run_evaluation(
                    protocol=_protocol(), subject=_subject(), adapter=adapter,
                    fixture=_fixture(seed=2), evidence_path=path,
                )
            with self.assertRaises(ValueError):
                run_evaluation(
                    protocol=_protocol(), subject=_subject(), adapter=adapter,
                    fixture=_fixture(generator_sha256="f" * 64), evidence_path=path,
                )
            with self.assertRaises(ValueError):
                run_evaluation(
                    protocol=dataclasses.replace(
                        _protocol(), candidate_scoring_mode="other-contract"
                    ),
                    subject=_subject(), adapter=adapter,
                    fixture=_fixture(), evidence_path=path,
                )

    def test_unknown_metric_and_rule_fail_before_run(self) -> None:
        import dataclasses

        adapter = _adapter()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            with self.assertRaises(ValueError):
                run_evaluation(
                    protocol=dataclasses.replace(_protocol(), metrics=("something",)),
                    subject=_subject(), adapter=adapter,
                    fixture=_fixture(), evidence_path=path,
                )
            with self.assertRaises(ValueError):
                run_evaluation(
                    protocol=dataclasses.replace(
                        _protocol(), statistical_rule="whatever"
                    ),
                    subject=_subject(), adapter=adapter,
                    fixture=_fixture(), evidence_path=path,
                )

    def test_tampered_artifact_fails_verification(self) -> None:
        import dataclasses

        receipt, _evidence, artifact = self._run()
        with tempfile.TemporaryDirectory() as directory:
            forged = Path(directory) / "evidence.jsonl"
            forged.write_bytes(artifact + b'{"forged": true}\n')
            with self.assertRaises(ValueError):
                verify_evidence_artifact(forged, receipt.evidence_artifact_sha256)

    def test_reproduction_is_identical(self) -> None:
        def make_clock():
            state = {"t": 1000.0}

            def tick() -> float:
                state["t"] += 0.25
                return state["t"]

            return tick

        first = self._run(clock=make_clock())
        second = self._run(clock=make_clock())
        self.assertEqual(first[0].sha256(), second[0].sha256())
        self.assertEqual(first[2], second[2])


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
