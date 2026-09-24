from __future__ import annotations

import copy
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import pytest

from signac_100m.phase1_eval import (
    INTERFERENCE_RETRIEVAL_GRID,
    PHASE1_METRICS,
    build_development_fixture,
    build_phase1_protocol,
    run_phase1_evaluation,
    summarize_phase1_evidence,
    verify_phase1_summary,
    _assert_checkpoint_source_binding,
    _sha256,
    _training_recipe_sha256,
    wilson_interval,
)
from signac_100m.spec import MODEL_SPEC
from v5_evaluation.adapter import GenerationResult
from v5_evaluation.checkpoint_adapter import (
    ADAPTER_IMPLEMENTATION_SHA256,
    ADAPTER_SCHEMA,
    CheckpointBackedV5Adapter,
    DECODING_RULE,
    SCORING_CONTRACT_SHA256,
    SCORING_RULE,
    AdapterIdentity,
)
from v5_evaluation.fixture import TaskFixtureBatch
from v5_evaluation.protocol import run_evaluation
from v5_registry.subject import CoreSubjectManifest


def _subject() -> CoreSubjectManifest:
    return CoreSubjectManifest.create(
        checkpoint_sha256="a" * 64,
        checkpoint_file_sha256="a" * 64,
        parameter_sha256="b" * 64,
        model_spec_sha256=MODEL_SPEC.sha256(),
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
        seed=41,
        custody="local-test-only",
        creation_receipt_sha256="7" * 64,
        source_tree_sha256="8" * 64,
    )


class _OracleForHarnessTest:
    scoring_contract_sha256 = SCORING_CONTRACT_SHA256

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        subject = _subject()
        self.identity = AdapterIdentity(
            schema=ADAPTER_SCHEMA,
            checkpoint_sha256=subject.checkpoint_sha256,
            model_payload_sha256="c" * 64,
            parameter_sha256=subject.parameter_sha256,
            model_spec_sha256=subject.model_spec_sha256,
            tokenizer_artifact_sha256=subject.tokenizer_artifact_sha256,
            scoring_rule=SCORING_RULE,
            decoding_rule=DECODING_RULE,
            implementation_sha256=ADAPTER_IMPLEMENTATION_SHA256,
            source_tree_sha256=subject.source_tree_sha256,
        )

    def generate_free_with_status(self, prompt: str) -> GenerationResult:
        return GenerationResult(
            text=self.answers[prompt],
            terminated_eos=True,
            generated_tokens=1,
            stop_reason="eos",
        )


def test_signac_phase1_uses_actual_causal_suite_and_candidate_free_metrics(tmp_path: Path):
    suite, fixture = build_development_fixture(seed=41, groups_per_family=1)
    protocol = build_phase1_protocol(fixture, seed=41)
    assert protocol.decoding_mode == "RAW_FREE_GENERATION"
    assert protocol.metrics == PHASE1_METRICS
    assert protocol.fixture_sha256 == fixture.sha256()
    assert len(fixture.cases) == len(suite.cases)
    assert any(case.get("causal_pairs") for case in fixture.cases)

    answers = {
        case.model_view()["prompt"]: case.answer
        for case in suite.cases
    }
    assert len(answers) == len(suite.cases)
    receipt, evidence = run_evaluation(
        protocol=protocol,
        subject=_subject(),
        adapter=_OracleForHarnessTest(answers),
        fixture=fixture,
        evidence_path=tmp_path / "phase1-evidence.jsonl",
    )
    summary = summarize_phase1_evidence(
        receipt=receipt,
        evidence=evidence,
        protocol=protocol,
        fixture=fixture,
        suite=suite,
        subject=_subject(),
    )

    assert summary["status"] == "MEASUREMENT_ONLY"
    assert summary["training_authorized"] is False
    assert summary["metrics"]["valid_eos"] == 1.0
    assert summary["metrics"]["exact_and_eos"] == 1.0
    assert summary["metrics"]["causal_pair_sensitivity"] == 1.0
    assert summary["metrics"]["causal_pair_invariance"] == 1.0
    assert summary["identities"]["receipt_subject_manifest_sha256"] == _subject().sha256()
    assert summary["identities"]["subject_manifest_sha256"] == _subject().sha256()
    assert summary["identities"]["source_tree_sha256"] == _subject().source_tree_sha256
    verify_phase1_summary(summary)
    contradictory = copy.deepcopy(summary)
    axis_record = contradictory["metrics"]["by_skill_axis"]["identity"]["exact_and_eos"]
    altered_successes = (axis_record["successes"] + 1) % (axis_record["cases"] + 1)
    axis_record["successes"] = altered_successes
    axis_record["rate"] = altered_successes / axis_record["cases"]
    axis_record["wilson95"] = list(wilson_interval(altered_successes, axis_record["cases"]))
    contradictory["sha256"] = _sha256({
        key: value for key, value in contradictory.items() if key != "sha256"
    })
    with pytest.raises(ValueError, match="disagrees with family metrics"):
        verify_phase1_summary(contradictory)
    interference_grid = summary["metrics"]["interference_retrieval_grid"]
    assert set(interference_grid) == {str(dose) for dose in INTERFERENCE_RETRIEVAL_GRID}
    for dose, quartiles in INTERFERENCE_RETRIEVAL_GRID.items():
        assert set(interference_grid[str(dose)]) == {str(quartile) for quartile in quartiles}
        for quartile in quartiles:
            condition = interference_grid[str(dose)][str(quartile)]
            assert condition["exact_and_eos"]["successes"] == 2
            assert condition["exact_and_eos"]["cases"] == 2
            assert condition["valid_eos"]["successes"] == 2
    counterfactual_pair = summary["metrics"]["causal_pairs_by_family_and_kind"][
        "counterfactual_premise"
    ]["relevant_fact_swap"]
    assert counterfactual_pair["successes"] == counterfactual_pair["cases"]
    assert counterfactual_pair["valid_eos_pair_rate"]["successes"] == counterfactual_pair["cases"]
    realization_pair = summary["metrics"]["causal_pairs_by_family_and_kind"][
        "faithful_realization"
    ]["relevant_fact_swap"]
    assert realization_pair["successes"] == realization_pair["cases"]
    assert summary["metrics"]["by_family"]["faithful_realization"]["exact_and_eos"]["successes"] > 0
    assert set(summary["metrics"]["by_skill_axis"]) == {
        "identity", "binding", "interference_retrieval", "state_order", "composition",
        "faithful_realization", "missing_information",
    }
    assert len(summary["sha256"]) == 64


def test_phase1_checkpoint_and_subject_source_tree_must_match():
    subject = _subject()
    _assert_checkpoint_source_binding(
        subject=subject,
        adapter_identity=SimpleNamespace(source_tree_sha256=subject.source_tree_sha256),
    )
    with pytest.raises(ValueError, match="source tree identity disagrees"):
        _assert_checkpoint_source_binding(
            subject=subject,
            adapter_identity=SimpleNamespace(source_tree_sha256="9" * 64),
        )
    changed_subject = replace(subject, source_tree_sha256="9" * 64)
    assert _training_recipe_sha256(changed_subject) != _training_recipe_sha256(subject)


def test_subject_manifest_source_extension_keeps_v1_hash_compatibility():
    subject = _subject()
    restored = CoreSubjectManifest.from_dict(subject.canonical())
    assert restored.sha256() == subject.sha256()

    legacy = replace(subject, schema="anra-v5-core-subject-manifest/v1", source_tree_sha256=None)
    legacy_restored = CoreSubjectManifest.from_dict(legacy.canonical())
    assert legacy_restored.source_tree_sha256 is None
    assert legacy_restored.sha256() == legacy.sha256()


def test_phase1_refuses_a_sealed_fixture():
    _suite, development = build_development_fixture(seed=41, groups_per_family=1)
    sealed_cases = [dict(case, split="sealed") for case in development.cases]
    sealed = TaskFixtureBatch.freeze(
        generator_id=development.generator_id,
        generator_sha256=development.generator_sha256,
        generator_config_sha256=development.generator_config_sha256,
        seed=41,
        split="sealed",
        cases=sealed_cases,
    )
    with pytest.raises(ValueError, match="only its matching development fixture"):
        build_phase1_protocol(sealed, seed=41)


def test_production_phase1_entry_refuses_an_unbound_test_adapter(tmp_path: Path):
    suite, fixture = build_development_fixture(seed=41, groups_per_family=1)
    protocol = build_phase1_protocol(fixture, seed=41)
    with pytest.raises(ValueError, match="checkpoint-backed V5 adapter"):
        run_phase1_evaluation(
            protocol=protocol,
            subject=_subject(),
            adapter=_OracleForHarnessTest({}),
            fixture=fixture,
            evidence_path=tmp_path / "refused.jsonl",
        )


def test_production_phase1_entry_rejects_checkpoint_adapter_subclasses(tmp_path: Path):
    class OracleSubclass(CheckpointBackedV5Adapter):
        def generate_free_with_status(self, prompt: str) -> GenerationResult:
            return GenerationResult(
                text="oracle", terminated_eos=True, generated_tokens=1, stop_reason="eos"
            )

    _suite, fixture = build_development_fixture(seed=41, groups_per_family=1)
    protocol = build_phase1_protocol(fixture, seed=41)
    adapter = object.__new__(OracleSubclass)
    with pytest.raises(ValueError, match="canonical checkpoint-backed V5 adapter type"):
        run_phase1_evaluation(
            protocol=protocol,
            subject=_subject(),
            adapter=adapter,
            fixture=fixture,
            evidence_path=tmp_path / "subclass-refused.jsonl",
        )


def test_production_phase1_entry_rejects_instance_shadowed_generation(tmp_path: Path):
    _suite, fixture = build_development_fixture(seed=41, groups_per_family=1)
    protocol = build_phase1_protocol(fixture, seed=41)
    adapter = object.__new__(CheckpointBackedV5Adapter)
    adapter.generate_free_with_status = lambda prompt: GenerationResult(
        text="oracle", terminated_eos=True, generated_tokens=1, stop_reason="eos"
    )
    with pytest.raises(ValueError, match="inference method is overridden: generate_free_with_status"):
        run_phase1_evaluation(
            protocol=protocol,
            subject=_subject(),
            adapter=adapter,
            fixture=fixture,
            evidence_path=tmp_path / "shadow-refused.jsonl",
        )
