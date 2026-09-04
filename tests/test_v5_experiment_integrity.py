"""Experiment integrity: code freeze, run manifests, matched-arm red-team."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from v5_experiments.codefreeze import CodeFreezeReceipt, freeze_code
from v5_experiments.preregistration import ExperimentSpec, TrainingInterventionRecord, assert_matched_arms
from v5_experiments.runmanifest import RunManifest

ROOT = Path(__file__).resolve().parents[1]


def _intervention() -> TrainingInterventionRecord:
    return TrainingInterventionRecord(
        hypothesis="cognition mixture causes binding gains",
        mechanism_target="query-conditioned binding",
        treatment_definition="15% verified cognition mixture",
        control_definition="0% cognition mixture",
        expected_behavioral_effect="binding accuracy increases at matched compute",
        expected_failure_profile_effect="binding failures shift from absent to partial",
        risks=("template memorization", "substrate regression"),
    )


def _spec(experiment_id: str, **overrides) -> ExperimentSpec:
    fields: dict[str, object] = {
        "experiment_id": experiment_id,
        "hypothesis": "verified cognition data improves binding beyond matched language modeling",
        "intervention": _intervention(),
        "parent_checkpoint_sha256s": ("a" * 64,),
        "model_spec_sha256": "b" * 64,
        "tokenizer_artifact_sha256": "c" * 64,
        "training_spec_sha256": "d" * 64,
        "data_manifest_sha256": "e" * 64,
        "optimizer_spec_sha256": "1" * 64,
        "schedule_spec_sha256": "2" * 64,
        "token_budget": 100_000,
        "seeds": (1, 2),
        "evaluation_protocol_sha256": "3" * 64,
        "promotion_rule": "conjunctive gates per frozen spec",
        "stop_rule": "two consecutive tier-1 declines",
    }
    fields.update(overrides)
    return ExperimentSpec(**fields)  # type: ignore[arg-type]


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


class CodeFreezeTests(unittest.TestCase):
    def test_freeze_binds_real_implementation_bytes(self) -> None:
        receipt = freeze_code(ROOT, source_commit=_head(), experiment_spec_sha256="f" * 64)
        self.assertEqual(len(dict(receipt.component_shas)), 6)
        self.assertEqual(len(receipt.sha256()), 64)
        # Deterministic: freezing twice yields the same receipt.
        again = freeze_code(ROOT, source_commit=_head(), experiment_spec_sha256="f" * 64)
        self.assertEqual(again.sha256(), receipt.sha256())

    def test_freeze_is_source_bound(self) -> None:
        first = freeze_code(ROOT, source_commit="a" * 40, experiment_spec_sha256="f" * 64)
        second = freeze_code(ROOT, source_commit="b" * 40, experiment_spec_sha256="f" * 64)
        self.assertNotEqual(first.sha256(), second.sha256())
        with self.assertRaises(ValueError):
            freeze_code(ROOT, source_commit="short", experiment_spec_sha256="f" * 64)


class RunManifestTests(unittest.TestCase):
    def _manifest(self, **overrides) -> RunManifest:
        fields: dict[str, object] = {
            "schema": "anra-v5-run-manifest/v1",
            "experiment_spec_sha256": "e" * 64,
            "arm_id": "control-seed-1",
            "seed": 1,
            "parent_subject_manifest_sha256": None,
            "source_commit": _head(),
            "runtime": "python-3.11-torch-2.11-cu128",
            "accelerator": "rtx-4050-6gb",
            "topology": "single",
            "data_stream_manifest_sha256": "d" * 64,
            "start_checkpoint_sha256": None,
            "token_budget": 200_000_000,
            "expected_evaluation_schedule": ("tier0@25M", "tier1@100M", "tier1@200M"),
        }
        fields.update(overrides)
        return RunManifest(**fields)  # type: ignore[arg-type]

    def test_manifest_round_trip(self) -> None:
        manifest = self._manifest()
        clone = RunManifest.from_dict(json.loads(json.dumps({
            "schema": manifest.schema,
            "experiment_spec_sha256": manifest.experiment_spec_sha256,
            "arm_id": manifest.arm_id,
            "seed": manifest.seed,
            "parent_subject_manifest_sha256": manifest.parent_subject_manifest_sha256,
            "source_commit": manifest.source_commit,
            "runtime": manifest.runtime,
            "accelerator": manifest.accelerator,
            "topology": manifest.topology,
            "data_stream_manifest_sha256": manifest.data_stream_manifest_sha256,
            "start_checkpoint_sha256": manifest.start_checkpoint_sha256,
            "token_budget": manifest.token_budget,
            "expected_evaluation_schedule": list(manifest.expected_evaluation_schedule),
        })))
        self.assertEqual(clone.sha256(), manifest.sha256())

    def test_manifest_rejects_gaps(self) -> None:
        with self.assertRaises(ValueError):
            self._manifest(arm_id="")
        with self.assertRaises(ValueError):
            self._manifest(token_budget=0)
        with self.assertRaises(ValueError):
            self._manifest(source_commit="short")


class MatchedArmRedTeamTests(unittest.TestCase):
    def _arms(self):
        control = _spec("arm-control", treatment_fields=("data_manifest_sha256",))
        treatment = _spec(
            "arm-treatment",
            data_manifest_sha256="9" * 64,
            treatment_fields=("data_manifest_sha256",),
        )
        return control, treatment

    def test_declared_treatment_passes(self) -> None:
        control, treatment = self._arms()
        result = assert_matched_arms(control, treatment, allowed_differences=("experiment_id",))
        self.assertTrue(result["matched"])
        self.assertIn("data_manifest_sha256", result["declared_differences"])

    def test_undeclared_tokenizer_drift_fails(self) -> None:
        control, _treatment = self._arms()
        drifted = _spec(
            "arm-treatment",
            data_manifest_sha256="9" * 64,
            tokenizer_artifact_sha256="8" * 64,
            treatment_fields=("data_manifest_sha256",),
        )
        with self.assertRaises(ValueError):
            assert_matched_arms(control, drifted, allowed_differences=("experiment_id",))

    def test_undeclared_schedule_drift_fails(self) -> None:
        control, _treatment = self._arms()
        drifted = _spec(
            "arm-treatment",
            data_manifest_sha256="9" * 64,
            schedule_spec_sha256="7" * 64,
            treatment_fields=("data_manifest_sha256",),
        )
        with self.assertRaises(ValueError):
            assert_matched_arms(control, drifted, allowed_differences=("experiment_id",))

    def test_undeclared_base_corpus_drift_fails(self) -> None:
        control = _spec("arm-control")
        drifted = _spec("arm-treatment", data_manifest_sha256="9" * 64)
        with self.assertRaises(ValueError):
            assert_matched_arms(control, drifted, allowed_differences=("experiment_id",))


if __name__ == "__main__":
    unittest.main()
