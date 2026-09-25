from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from x_factor.ibq_v2 import _t_state_table
from x_factor.x1_real_1 import (
    ANALYSIS_SCHEMA,
    BASIS_SCHEMA,
    PREDICTION_BASELINES,
    PREDICTION_SCHEMA,
    REVEAL_SCHEMA,
    X1ChronologyError,
    X1EvidenceError,
    _sha_json,
    build_baseline_commitment,
    build_execution_plan,
    commit_phase_record,
    commit_prediction_receipt,
    commit_reveal_receipt,
    intake_checkpoint_candidate,
    inventory_checkpoint_candidates,
    main,
    paired_cluster_bootstrap,
    preflight_x1_real_1,
    qualify_intervention_basis,
    validate_frozen_protocol,
    validate_basis_qualification,
    validate_intervention_registry,
    validate_prediction_receipt,
    validate_public_task_bundle,
    validate_reveal_receipt,
    validate_subject_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "x_factor" / "protocols" / "x1_real_1_v1.json"
REGISTRY_PATH = ROOT / "x_factor" / "registry" / "checkpoints.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _subject(registry_path: Path) -> dict:
    checkpoint_path = registry_path.parent / "checkpoint.pt"
    checkpoint_path.write_bytes(b"fixture-checkpoint")
    checkpoint_sha = hashlib.sha256(b"fixture-checkpoint").hexdigest()
    readiness_path = registry_path.parent / "readiness.json"
    cohort_plan_path = registry_path.parent / "cohort-plan.json"
    cross_checkpoint_plan_path = registry_path.parent / "cross-checkpoint-plan.json"
    readiness_path.write_text("readiness", encoding="utf-8")
    cohort_plan_path.write_text("cohort-plan", encoding="utf-8")
    cross_checkpoint_plan_path.write_text("cross-checkpoint-plan", encoding="utf-8")
    subject = {
        "schema": "anra-x1-real-1-subject/v1",
        "subject_id": "external-subject-001",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_sha256": checkpoint_sha,
        "parameter_sha256": "b" * 64,
        "model_config_sha256": "c" * 64,
        "tokenizer_artifact_sha256": "d" * 64,
        "tokenizer_identity_sha256": "e" * 64,
        "runtime_source_revision": "external-runtime-revision-001",
        "source_commit": "2" * 40,
        "training_lineage": "external-training-lineage",
        "stage": "qualified_stage",
        "global_step": 1234,
        "research_subject": True,
        "eligibility": {
            "state": "READY_SCOPED",
            "readiness_receipt_path": str(readiness_path),
            "readiness_receipt_sha256": hashlib.sha256(readiness_path.read_bytes()).hexdigest(),
            "cohort_plan_path": str(cohort_plan_path),
            "cohort_plan_sha256": hashlib.sha256(cohort_plan_path.read_bytes()).hexdigest(),
            "cross_checkpoint_plan_path": str(cross_checkpoint_plan_path),
            "cross_checkpoint_plan_sha256": hashlib.sha256(cross_checkpoint_plan_path.read_bytes()).hexdigest(),
        },
        "registry_entry_sha256": "6" * 64,
        "identity_attestation_path": "identity-attestation.json",
        "identity_attestation_sha256": "7" * 64,
    }
    registry_entry = {
        "path": subject["checkpoint_path"],
        "file_sha256": subject["checkpoint_file_sha256"],
        "parameter_sha256": subject["parameter_sha256"],
        "config_sha256": subject["model_config_sha256"],
        "tokenizer_artifact_sha256": subject["tokenizer_artifact_sha256"],
        "tokenizer_identity_sha256": subject["tokenizer_identity_sha256"],
        "runtime_source_revision": subject["runtime_source_revision"],
        "source_commit": subject["source_commit"],
        "global_step": subject["global_step"],
        "stage": subject["stage"],
        "role": "RESEARCH_SUBJECT",
        "status": "READY_SCOPED",
        "research_subject": True,
        "readiness": dict(subject["eligibility"]),
    }
    subject["registry_entry_sha256"] = _sha_json(registry_entry)
    registry = {"schema": "anra-checkpoint-registry/v2", "checkpoints": [registry_entry]}
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    attestation = {
        "schema": "anra-checkpoint-identity/v1",
        "subject_id": subject["subject_id"],
        "checkpoint_file_sha256": subject["checkpoint_file_sha256"],
        "parameter_sha256": subject["parameter_sha256"],
        "model_config_sha256": subject["model_config_sha256"],
        "tokenizer_artifact_sha256": subject["tokenizer_artifact_sha256"],
        "tokenizer_identity_sha256": subject["tokenizer_identity_sha256"],
        "runtime_source_revision": subject["runtime_source_revision"],
        "source_commit": subject["source_commit"],
        "training_lineage": subject["training_lineage"],
        "stage": subject["stage"],
        "global_step": subject["global_step"],
        "registry_entry_sha256": subject["registry_entry_sha256"],
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    }
    attestation_path = registry_path.parent / "identity-attestation.json"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    subject["identity_attestation_path"] = str(attestation_path)
    subject["identity_attestation_sha256"] = hashlib.sha256(attestation_path.read_bytes()).hexdigest()
    return subject


def _basis_rows(protocol: dict, count: int = 60) -> list[dict]:
    intervention_ids = [item["id"] for item in protocol["interventions"]]
    return [
        {
            "task_id": f"task-{index:03d}",
            "cluster_id": f"cluster-{index // 3:03d}",
            "source_id": f"source-{index // 3:03d}",
            "baseline_failed": True,
            "outcomes": {
                intervention_id: (0 if index < 4 or item_index < 2 else int((index >> (item_index - 2)) & 1))
                for item_index, intervention_id in enumerate(intervention_ids)
            },
        }
        for index in range(count)
    ]


def _prefix_rows(rows: list[dict], prefix: str) -> list[dict]:
    return [
        {
            **row,
            "task_id": f"{prefix}-{row['task_id']}",
            "cluster_id": f"{prefix}-{row['cluster_id']}",
            "source_id": f"{prefix}-{row['source_id']}",
            "content_variant": f"{prefix}-{row['task_id']}",
        }
        for row in rows
    ]


def _public_task_record(row: dict, index: int) -> dict:
    return {
        "task_id": row["task_id"],
        "context": f"Visible context for {row.get('content_variant', 'the reference task')}.",
        "query": "Return the visible reference.",
        "visible_candidates": ["AAA-001", "BBB-002"],
        "format": "json",
        "features": {
            "confidence": -0.2 - index / 100,
            "entropy": 0.4,
            "margin": 0.1,
            "output_len": 4,
            "distinct_ratio": 1.0,
            "prompt_tokens": 20 + index,
        },
    }


def _split(
    protocol_sha: str, rows: list[dict], *, cohort_id: str = "PRIMARY_EVAL",
    cohort_role: str = "EVALUATION", seed: int = 81819, split_label: str = "PRIMARY_EVAL",
) -> dict:
    clusters = {}
    row_indices = {row["task_id"]: index for index, row in enumerate(rows)}
    for row in rows:
        cluster = clusters.setdefault(row["cluster_id"], {
            "cluster_id": row["cluster_id"],
            "source_id": row["source_id"],
            "split": split_label,
            "task_ids": [],
            "task_content_sha256s": {},
        })
        cluster["task_ids"].append(row["task_id"])
        public_record = _public_task_record(row, row_indices[row["task_id"]])
        cluster["task_content_sha256s"][row["task_id"]] = _sha_json({
            key: value for key, value in public_record.items()
            if key not in {"task_id", "observation_hash"}
        })
    body = {
        "schema": "anra-x1-real-1-split/v1",
        "cohort_id": cohort_id,
        "cohort_role": cohort_role,
        "seed": seed,
        "cohort_plan_sha256": {
            "BASIS_QUALIFICATION": "21c186b22edaf60abd0387fa2b2ed29104b8a13f953be1250a58bc7b15663336",
            "DEV_COHORT_V1": "f69d9f14fc97dba864c4e7d29d1151ffb141d7a9a11b86d04c427b856afcbb56",
            "PRIMARY_EVAL": "3aa54d2ec406cd0d7b7576ea991833c03658ba58b8ab17a289eab77b97ac55bb",
            "INDEPENDENT_TASK_EVAL": "cf4123533170fbe535c70c385d184ec76064b732a2ce9e892f1686c8a6744213",
            "CHECKPOINT_REPLICATION": "6459c5107121766a4ac65db8b6d7f83b1b51bfd5fde368089a6ae78956948e29",
        }[cohort_id],
        "protocol_sha256": protocol_sha,
        "clusters": list(clusters.values()),
    }
    return {**body, "manifest_sha256": _sha_json(body)}


def _public_tasks(rows: list[dict]) -> list[dict]:
    return [_public_task_record(row, index) for index, row in enumerate(rows)]


def _outcome_rows(rows: list[dict], intervention_ids: list[str], verifier_artifact_path: Path) -> list[dict]:
    output = []
    for row in rows:
        for intervention_id in intervention_ids:
            repaired = bool(row["outcomes"][intervention_id])
            raw_output = f"raw-{row['task_id']}-{intervention_id}"
            transformed_prompt = f"transformed-{row['task_id']}-{intervention_id}"
            raw_hash = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
            prompt_hash = hashlib.sha256(transformed_prompt.encode("utf-8")).hexdigest()
            gold_hash = "8" * 64
            artifact_path = verifier_artifact_path.with_name(
                f"{verifier_artifact_path.stem}-{row['task_id']}-{intervention_id}.json"
            )
            artifact_path.write_text(json.dumps({
                "schema": "anra-x1-real-1-verifier/v1",
                "task_id": row["task_id"],
                "intervention_id": intervention_id,
                "repaired": repaired,
                "gold_sha256": gold_hash,
                "raw_output_sha256": raw_hash,
                "transformed_prompt_sha256": prompt_hash,
            }), encoding="utf-8")
            artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            output.append({
                "task_id": row["task_id"],
                "intervention_id": intervention_id,
                "cluster_id": row["cluster_id"],
                "source_id": row["source_id"],
                "repaired": repaired,
                "baseline_failed": True,
                "effect": 1.0 if repaired else 0.0,
                "raw_output": raw_output,
                "verifier_receipt_sha256": artifact_hash,
                "gold_sha256": gold_hash,
                "transformed_prompt_sha256": prompt_hash,
                "transformed_prompt": transformed_prompt,
                "verifier_artifact_path": str(artifact_path),
                "verifier_artifact_sha256": artifact_hash,
            })
    return output


def _predictor(rows: list[dict], intervention_ids: list[str]) -> dict:
    return {
        row["task_id"]: {
            "predicted_repair_probability": [0.0] * len(intervention_ids),
            "predicted_best_intervention": "NO_CHANGE",
            "uncertainty": 0.0,
        }
        for row in rows
    }


def _receipt_fixture(tmp_path: Path) -> tuple:
    protocol = _protocol()
    registry_path = tmp_path / "registry.json"
    subject = _subject(registry_path)
    basis_rows = _prefix_rows(_basis_rows(protocol), "basis")
    development_rows = _prefix_rows(_basis_rows(protocol), "dev")
    rows = [
        {**row, "cluster_id": f"eval-cluster-{index}", "source_id": f"eval-source-{index}"}
        for index, row in enumerate(_prefix_rows(_basis_rows(protocol), "eval"))
    ]
    basis_source_path = tmp_path / "basis-source.json"
    basis_source_path.write_text(json.dumps({"rows": basis_rows}), encoding="utf-8")
    verifier_artifact_path = tmp_path / "verifier.bin"
    verifier_artifact_path.write_bytes(b"fixture-verifier")
    basis_phase_path = tmp_path / "basis-phase.json"
    basis_probe = qualify_intervention_basis(
        basis_rows, protocol["interventions"],
        protocol_sha256=protocol["identity"]["protocol_sha256"],
        thresholds=protocol["decision_thresholds"]["basis"],
    )
    basis_matrix_sha = basis_probe["matrix_sha256"]
    basis_source_sha = hashlib.sha256(basis_source_path.read_bytes()).hexdigest()
    basis_phase_body = {
        "schema": "anra-x1-real-1-phase-commit/v1",
        "phase": "BASIS_QUALIFIED",
        "phase_index": 0,
        "receipt_sha256": basis_source_sha,
        "artifact_path": str(basis_source_path),
        "artifact_sha256": basis_source_sha,
        "sequence": 0,
        "committed_at": "2026-01-01T00:00:00+00:00",
        "previous_commit_sha256": None,
        "custodian": "test-custodian",
        "attestation_sha256": "b1" * 32,
    }
    basis_phase_body["phase_commit_sha256"] = _sha_json(basis_phase_body)
    basis_phase_path.write_text(json.dumps(basis_phase_body), encoding="utf-8")
    basis_split = _split(
        protocol["identity"]["protocol_sha256"], basis_rows,
        cohort_id="BASIS_QUALIFICATION", cohort_role="QUALIFICATION", seed=81817,
        split_label="QUALIFICATION",
    )
    development_split = _split(
        protocol["identity"]["protocol_sha256"], development_rows,
        cohort_id="DEV_COHORT_V1", cohort_role="DEVELOPMENT", seed=81818,
        split_label="DEVELOPMENT",
    )
    split = _split(protocol["identity"]["protocol_sha256"], rows)
    basis = qualify_intervention_basis(
        basis_rows,
        protocol["interventions"],
        protocol_sha256=protocol["identity"]["protocol_sha256"],
        subject_manifest_sha256=_sha_json(subject),
        thresholds=protocol["decision_thresholds"]["basis"],
        binding={
            "basis_cohort_id": "BASIS_QUALIFICATION",
            "basis_split_manifest_sha256": basis_split["manifest_sha256"],
            "basis_source_artifact_path": str(basis_source_path),
            "basis_source_artifact_sha256": basis_source_sha,
            "basis_matrix_sha256": basis_matrix_sha,
            "basis_phase_artifact_path": str(basis_phase_path),
            "basis_phase_commit_sha256": hashlib.sha256(basis_phase_path.read_bytes()).hexdigest(),
        },
    )
    assert basis["status"] == "QUALIFIED"
    development_artifact_path = tmp_path / "development.json"
    development_artifact_path.write_text(json.dumps(development_rows), encoding="utf-8")
    development = [
        {"task_id": row["task_id"], "outcomes": row["outcomes"], "surface_key": f"surface-{index % 2}"}
        for index, row in enumerate(development_rows)
    ]
    baselines = build_baseline_commitment(
        development,
        [row["task_id"] for row in rows],
        protocol["interventions"],
        development_artifact_sha256=hashlib.sha256(development_artifact_path.read_bytes()).hexdigest(),
        development_artifact_path=str(development_artifact_path),
        development_cohort_id="DEV_COHORT_V1",
        development_split_manifest_sha256=development_split["manifest_sha256"],
        selection_lambda=0.25,
        seed=81818,
        surface_keys={row["task_id"]: f"surface-{index % 2}" for index, row in enumerate(rows)},
        development_split_manifest=development_split,
    )
    predictor_artifact_path = tmp_path / "predictor.bin"
    predictor_artifact_path.write_bytes(b"predictor-training-artifact")
    predictor_training_manifest_path = tmp_path / "predictor-training.json"
    predictor_training_manifest_path.write_text(json.dumps({
        "schema": "anra-x1-real-1-predictor-training/v1",
        "training_split_manifest_sha256": development_split["manifest_sha256"],
        "development_task_ids": [row["task_id"] for row in development_rows],
        "evaluation_outcomes_used": False,
        "source_sha256": "a1" * 32,
        "feature_schema_sha256": _sha_json(protocol["observations"]["feature_fields"]),
    }), encoding="utf-8")
    predictor_identity = {
        "name": "test-predictor",
        "version": "1",
        "source_sha256": "a1" * 32,
        "training_artifact_sha256": hashlib.sha256(predictor_artifact_path.read_bytes()).hexdigest(),
        "training_artifact_path": str(predictor_artifact_path),
        "training_manifest_path": str(predictor_training_manifest_path),
        "training_manifest_sha256": hashlib.sha256(predictor_training_manifest_path.read_bytes()).hexdigest(),
        "training_split_manifest_sha256": development_split["manifest_sha256"],
        "feature_schema_sha256": _sha_json(protocol["observations"]["feature_fields"]),
        "evaluation_outcomes_used": False,
    }
    release_artifact_path = tmp_path / "release.bin"
    release_artifact_path.write_bytes(b"release-attestation")
    release_attestation_path = tmp_path / "release-attestation.json"
    release_attestation_path.write_text(json.dumps({
        "schema": "anra-x1-real-1-release-attestation/v1",
        "protocol_sha256": protocol["identity"]["protocol_sha256"],
        "source_closure_sha256": protocol["identity"]["source_closure_sha256"],
        "custodian": "test-custodian",
        "source_tree_status": "CLEAN",
    }), encoding="utf-8")
    release_manifest = {
        "schema": "anra-x1-real-1-release/v1",
        "protocol_sha256": protocol["identity"]["protocol_sha256"],
        "source_closure_sha256": protocol["identity"]["source_closure_sha256"],
        "base_revision": protocol["identity"]["base_revision"],
        "release_revision": protocol["identity"]["base_revision"],
        "source_tree_status": "CLEAN",
        "source_files": protocol["identity"]["source_files"],
        "protocol_artifact_path": str(PROTOCOL_PATH),
        "protocol_artifact_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "release_artifact_path": str(release_artifact_path),
        "release_artifact_sha256": hashlib.sha256(release_artifact_path.read_bytes()).hexdigest(),
        "attestation_path": str(release_attestation_path),
        "attestation_sha256": hashlib.sha256(release_attestation_path.read_bytes()).hexdigest(),
        "custodian": "test-custodian",
    }
    release_manifest["release_manifest_sha256"] = _sha_json(release_manifest)
    public_tasks = _public_tasks(rows)
    prediction = commit_prediction_receipt(
        protocol,
        subject,
        split,
        public_tasks,
        _predictor(rows, [item["id"] for item in protocol["interventions"]]),
        baselines,
        predictor_identity,
        basis,
        cohort_id="PRIMARY_EVAL",
        registry_path=registry_path,
        artifact_root=tmp_path,
        development_split_manifest=development_split,
        basis_split_manifest=basis_split,
        release_manifest=release_manifest,
        release_root=tmp_path,
    )
    prediction_path = tmp_path / "prediction.json"
    prediction_commit = commit_phase_record(
        prediction, receipt_artifact_path=prediction_path, phase="PREDICT_COMMITTED",
        custodian="test-custodian", attestation_sha256="c2" * 32,
        previous_commit_sha256=basis["basis_binding"]["basis_phase_commit_sha256"],
    )
    reveal = commit_reveal_receipt(
        prediction,
        _outcome_rows(rows, [item["id"] for item in protocol["interventions"]], verifier_artifact_path),
        evaluator_identity={
            "schema": "anra-x1-real-1-evaluator/v1",
            "source_sha256": "d1" * 32,
            "verifier_source_sha256": "e1" * 32,
            "execution_artifact_sha256": "f1" * 32,
        },
        split_manifest=split,
        prediction_commit=prediction_commit,
        protocol=protocol,
        subject_manifest=subject,
        registry_path=registry_path,
        checkpoint_root=registry_path.parent,
        release_manifest=release_manifest,
        release_root=registry_path.parent,
        public_tasks=public_tasks,
        basis_qualification=basis,
    )
    reveal_path = tmp_path / "reveal.json"
    reveal_commit = commit_phase_record(
        reveal, receipt_artifact_path=reveal_path, phase="REVEAL_ACCEPTED",
        custodian="test-custodian", attestation_sha256="d2" * 32,
        previous_commit_sha256=prediction_commit["phase_commit_sha256"],
    )
    return (
        protocol, subject, basis, rows, split, prediction, reveal, registry_path,
        basis_split, development_split, public_tasks, prediction_commit, reveal_commit,
        release_manifest,
    )


def test_protocol_is_frozen_and_source_bound() -> None:
    verdict = validate_frozen_protocol(_protocol(), repo_root=ROOT, check_source_closure=True)
    assert verdict["valid"] is True
    assert verdict["protocol_sha256"] == _protocol()["identity"]["protocol_sha256"]


def test_inventory_reports_no_eligible_current_checkpoint() -> None:
    inventory = inventory_checkpoint_candidates(REGISTRY_PATH)
    assert inventory["n_candidates"] == 5
    assert inventory["eligible_candidates"] == 0
    assert inventory["status"] == "NO_ELIGIBLE_CHECKPOINT"


def test_candidate_intake_hashes_files_without_promoting(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    config = tmp_path / "config.py"
    tokenizer = tmp_path / "tokenizer.json"
    checkpoint.write_bytes(b"checkpoint")
    config.write_bytes(b"config")
    tokenizer.write_bytes(b"tokenizer")
    result = intake_checkpoint_candidate(
        checkpoint, config_path=config, tokenizer_path=tokenizer,
        runtime_source_revision="runtime-1", source_commit="a" * 40,
        global_step=10, stage="qualified_stage",
    )
    assert result["status"] == "UNQUALIFIED_NEW"
    assert result["research_subject"] is False
    assert result["eligible_for_x1"] is False
    assert result["promotion_status"] == "BLOCKED"
    assert result["files"]["checkpoint"]["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert "parameter_sha256" in result["missing_fields"]


def test_intervention_registry_rejects_gold_input() -> None:
    protocol = _protocol()
    protocol["interventions"][2]["legality_inputs"] = ["gold_answer"]
    verdict = validate_intervention_registry(protocol["interventions"])
    assert verdict["valid"] is False
    assert any("forbidden legality input" in item for item in verdict["errors"])


def test_basis_requires_clustered_failures_and_null_controls(tmp_path: Path) -> None:
    protocol = _protocol()
    unqualified = qualify_intervention_basis(
        [{"task_id": f"t-{index}", "outcomes": {item["id"]: 0 for item in protocol["interventions"]}} for index in range(50)],
        protocol["interventions"],
        thresholds=protocol["decision_thresholds"]["basis"],
    )
    assert unqualified["status"] == "NOT_QUALIFIED"
    assert unqualified["checks"]["independent_cluster_identity"] is False
    basis_probe = qualify_intervention_basis(
        _basis_rows(protocol), protocol["interventions"],
        thresholds=protocol["decision_thresholds"]["basis"],
    )
    source_path = tmp_path / "basis-source.json"
    source_path.write_text(json.dumps({"rows": _basis_rows(protocol)}), encoding="utf-8")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    phase_path = tmp_path / "basis-phase.json"
    phase_body = {
        "schema": "anra-x1-real-1-phase-commit/v1",
        "phase": "BASIS_QUALIFIED",
        "phase_index": 0,
        "receipt_sha256": source_sha,
        "artifact_path": str(source_path),
        "artifact_sha256": source_sha,
        "sequence": 0,
        "committed_at": "2026-01-01T00:00:00+00:00",
        "previous_commit_sha256": None,
        "custodian": "test-custodian",
        "attestation_sha256": "b1" * 32,
    }
    phase_body["phase_commit_sha256"] = _sha_json(phase_body)
    phase_path.write_text(json.dumps(phase_body), encoding="utf-8")
    qualified = qualify_intervention_basis(
        _basis_rows(protocol), protocol["interventions"],
        thresholds=protocol["decision_thresholds"]["basis"],
        binding={
            "basis_cohort_id": "BASIS_QUALIFICATION",
            "basis_split_manifest_sha256": "1" * 64,
            "basis_source_artifact_path": str(source_path),
            "basis_source_artifact_sha256": source_sha,
            "basis_matrix_sha256": basis_probe["matrix_sha256"],
            "basis_phase_artifact_path": str(phase_path),
            "basis_phase_commit_sha256": hashlib.sha256(phase_path.read_bytes()).hexdigest(),
        },
    )
    assert qualified["status"] == "QUALIFIED"
    assert qualified["null_analysis"]["geometry_assumption"].startswith("none")


def test_public_task_schema_rejects_nested_truth() -> None:
    task = {
        "task_id": "task-001",
        "context": "visible",
        "query": "query",
        "visible_candidates": ["A-001"],
        "features": {"confidence": 0.1},
        "nested": {"gold": "A-001"},
    }
    with pytest.raises(X1EvidenceError):
        validate_public_task_bundle([task])


def test_prediction_receipt_has_no_truth_and_is_deterministic(tmp_path: Path) -> None:
    (protocol, subject, basis, rows, split, prediction, reveal, registry_path,
     basis_split, development_split, public_tasks, prediction_commit, reveal_commit,
     release_manifest) = _receipt_fixture(tmp_path)
    assert prediction["schema"] == PREDICTION_SCHEMA
    assert prediction["phase_index"] == 1
    assert validate_prediction_receipt(
        prediction, protocol=protocol, subject_manifest=subject, split_manifest=split,
        registry_path=registry_path, checkpoint_root=registry_path.parent,
        release_manifest=release_manifest, release_root=registry_path.parent,
        basis_qualification=basis, basis_subject_manifest_sha256=_sha_json(subject),
        public_tasks=public_tasks,
    )["valid"] is True
    tampered = copy.deepcopy(prediction)
    tampered["predictions"][0]["gold"] = "AAA-001"
    assert validate_prediction_receipt(tampered)["valid"] is False


def test_predictor_training_manifest_is_file_bound(tmp_path: Path) -> None:
    (protocol, subject, basis, rows, split, prediction, reveal, registry_path,
     basis_split, development_split, public_tasks, prediction_commit, reveal_commit,
     release_manifest) = _receipt_fixture(tmp_path)
    manifest_path = Path(prediction["predictor"]["training_manifest_path"])
    manifest_path.write_text("{}", encoding="utf-8")
    assert validate_prediction_receipt(
        prediction, protocol=protocol, subject_manifest=subject, split_manifest=split,
        registry_path=registry_path, checkpoint_root=registry_path.parent,
        release_manifest=release_manifest, release_root=registry_path.parent,
        basis_qualification=basis, basis_subject_manifest_sha256=_sha_json(subject),
        public_tasks=public_tasks, development_split_manifest=development_split,
        basis_split_manifest=basis_split,
    )["valid"] is False


def test_reveal_requires_prediction_parent_and_exact_coverage(tmp_path: Path) -> None:
    (protocol, subject, basis, rows, split, prediction, reveal, registry_path,
     basis_split, development_split, public_tasks, prediction_commit, reveal_commit,
     release_manifest) = _receipt_fixture(tmp_path)
    assert validate_reveal_receipt(
        reveal, prediction_receipt=prediction, split_manifest=split,
        prediction_commit=prediction_commit, reveal_commit=reveal_commit,
        repo_root=registry_path.parent,
    )["valid"] is True
    wrong_parent = copy.deepcopy(reveal)
    wrong_parent["prediction_receipt_sha256"] = "0" * 64
    assert validate_reveal_receipt(
        wrong_parent, prediction_receipt=prediction, prediction_commit=prediction_commit,
        reveal_commit=reveal_commit, repo_root=registry_path.parent,
    )["valid"] is False
    with pytest.raises(X1EvidenceError):
        commit_reveal_receipt(
            prediction,
            _outcome_rows(rows, [item["id"] for item in protocol["interventions"]], tmp_path / "verifier.bin")[:-1],
            evaluator_identity={
                "schema": "anra-x1-real-1-evaluator/v1",
                "source_sha256": "d1" * 32,
                "verifier_source_sha256": "e1" * 32,
                "execution_artifact_sha256": "f1" * 32,
            },
            split_manifest=split,
            prediction_commit=prediction_commit,
            protocol=protocol,
            subject_manifest=subject,
            registry_path=registry_path,
            checkpoint_root=registry_path.parent,
            release_manifest=release_manifest,
            release_root=registry_path.parent,
            public_tasks=public_tasks,
            basis_qualification=basis,
        )


def test_analysis_blocks_or_reports_not_supported_without_replication(tmp_path: Path) -> None:
    from x_factor.x1_real_1 import analyze_x1_real_1

    (protocol, subject, basis, rows, split, prediction, reveal, registry_path,
     basis_split, development_split, public_tasks, prediction_commit, reveal_commit,
     release_manifest) = _receipt_fixture(tmp_path)
    analysis = analyze_x1_real_1(
        protocol, subject, split, prediction, reveal,
        basis_qualification=basis, registry_path=registry_path,
        checkpoint_root=registry_path.parent, release_manifest=release_manifest,
        release_root=registry_path.parent, public_tasks=public_tasks,
        prediction_commit=prediction_commit, reveal_commit=reveal_commit,
        basis_split_manifest=basis_split, development_split_manifest=development_split,
    )
    assert analysis["schema"] == ANALYSIS_SCHEMA
    assert analysis["decision"]["supported"] is False
    assert analysis["decision"]["status"] in {"NOT_SUPPORTED", "PRIMARY_GATE_PASS_REPLICATION_PENDING"}


def test_always_negative_is_not_a_valid_raw_accuracy_promotion() -> None:
    from x_factor.x1_real_1 import _cell_metrics, _normalize_prediction_map

    intervention_ids = ["NO_CHANGE", "CANONICAL_CONTEXT"]
    predictions = _normalize_prediction_map(
        {"task-001": [0.0, 0.0], "task-002": [0.0, 0.0]},
        ["task-001", "task-002"], intervention_ids, name="test",
    )
    outcomes = {
        ("task-001", "NO_CHANGE"): False, ("task-001", "CANONICAL_CONTEXT"): False,
        ("task-002", "NO_CHANGE"): False, ("task-002", "CANONICAL_CONTEXT"): True,
    }
    metrics = _cell_metrics(predictions, outcomes, intervention_ids)
    assert metrics["raw_accuracy_diagnostic_only"] == 0.75
    assert metrics["auprc_lift_over_prevalence"] <= 0.0
    assert metrics["brier_skill_vs_prevalence"] <= 0.0
    assert metrics["score_variance"] == 0.0


def test_paired_cluster_bootstrap_preserves_cluster_unit() -> None:
    predictor = [
        {"task_id": "t1", "cluster_id": "c1", "selected_utility": 1.0},
        {"task_id": "t2", "cluster_id": "c1", "selected_utility": 0.5},
        {"task_id": "t3", "cluster_id": "c2", "selected_utility": 0.0},
    ]
    baseline = [
        {"task_id": "t1", "cluster_id": "c1", "selected_utility": 0.0},
        {"task_id": "t2", "cluster_id": "c1", "selected_utility": 0.0},
        {"task_id": "t3", "cluster_id": "c2", "selected_utility": 0.0},
    ]
    result = paired_cluster_bootstrap(predictor, baseline, n_resamples=100, seed=3)
    assert result["n_clusters"] == 2
    assert result["paired_unit"] == "independent_latent_world_or_source_cluster"
    assert result["ci95"][0] <= result["point_estimate"] <= result["ci95"][1]


def test_import_is_model_free() -> None:
    code = "import sys; import x_factor.x1_real_1; assert 'torch' not in sys.modules; assert 'anra_core' not in sys.modules; print('ok')"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "ok"


def test_legacy_receipt_is_not_a_prediction_receipt() -> None:
    legacy = json.loads((ROOT / "output" / "x1_real_receipt.json").read_text(encoding="utf-8"))
    assert legacy["schema"] != PREDICTION_SCHEMA
    assert validate_prediction_receipt(legacy)["valid"] is False


def test_malformed_registry_entries_are_not_silently_dropped() -> None:
    protocol = _protocol()
    verdict = validate_intervention_registry(protocol["interventions"] + ["malformed"])
    assert verdict["valid"] is False


def test_split_overlap_is_rejected() -> None:
    protocol = _protocol()
    rows = _prefix_rows(_basis_rows(protocol)[:20], "same")
    primary = _split(protocol["identity"]["protocol_sha256"], rows)
    replication = _split(
        protocol["identity"]["protocol_sha256"], rows,
        cohort_id="CHECKPOINT_REPLICATION", cohort_role="REPLICATION", seed=81821,
        split_label="CHECKPOINT_REPLICATION",
    )
    from x_factor.x1_real_1 import validate_split_independence
    assert validate_split_independence(primary, replication)["valid"] is False


def test_relabelled_task_content_is_not_independent() -> None:
    protocol = _protocol()
    rows = _prefix_rows(_basis_rows(protocol)[:20], "content")
    relabelled = []
    for index, row in enumerate(rows):
        relabelled.append({
            **row,
            "task_id": f"new-task-{index}",
            "cluster_id": f"new-cluster-{index // 3}",
            "source_id": f"new-source-{index // 3}",
        })
    from x_factor.x1_real_1 import validate_split_independence
    primary = _split(protocol["identity"]["protocol_sha256"], rows)
    other = _split(
        protocol["identity"]["protocol_sha256"], relabelled,
        cohort_id="CHECKPOINT_REPLICATION", cohort_role="REPLICATION", seed=81821,
        split_label="CHECKPOINT_REPLICATION",
    )
    assert validate_split_independence(primary, other)["valid"] is False


def test_phase_commit_rejects_sequence_tampering(tmp_path: Path) -> None:
    protocol = _protocol()
    (protocol, subject, basis, rows, split, prediction, reveal, registry_path,
     basis_split, development_split, public_tasks, prediction_commit, reveal_commit,
     release_manifest) = _receipt_fixture(tmp_path)
    tampered = copy.deepcopy(prediction_commit)
    tampered["sequence"] = 2
    from x_factor.x1_real_1 import validate_phase_commit
    verdict = validate_phase_commit(
        tampered, expected_phase="PREDICT_COMMITTED",
        expected_receipt_sha256=prediction["prediction_receipt_sha256"],
        repo_root=tmp_path,
    )
    assert verdict["valid"] is False


def test_phase_commit_resumes_idempotently(tmp_path: Path) -> None:
    (protocol, subject, basis, rows, split, prediction, reveal, registry_path,
     basis_split, development_split, public_tasks, prediction_commit, reveal_commit,
     release_manifest) = _receipt_fixture(tmp_path)
    path = tmp_path / "prediction-resume.json"
    first = commit_phase_record(
        prediction, receipt_artifact_path=path, phase="PREDICT_COMMITTED",
        custodian="test-custodian", attestation_sha256="c2" * 32,
        previous_commit_sha256=basis["basis_binding"]["basis_phase_commit_sha256"],
        committed_at="2026-01-01T00:00:00+00:00",
    )
    second = commit_phase_record(
        prediction, receipt_artifact_path=path, phase="PREDICT_COMMITTED",
        custodian="test-custodian", attestation_sha256="c2" * 32,
        previous_commit_sha256=basis["basis_binding"]["basis_phase_commit_sha256"],
        committed_at="2026-01-01T00:00:00+00:00",
    )
    assert second["phase_commit_sha256"] == first["phase_commit_sha256"]


def test_subject_requires_file_bound_identity_attestation(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    subject = _subject(registry_path)
    assert validate_subject_manifest(subject, registry_path=registry_path, checkpoint_root=tmp_path)["valid"] is True
    Path(subject["identity_attestation_path"]).write_text("{}", encoding="utf-8")
    assert validate_subject_manifest(subject, registry_path=registry_path, checkpoint_root=tmp_path)["valid"] is False


def test_basis_binding_rejects_source_artifact_tampering(tmp_path: Path) -> None:
    (protocol, subject, basis, rows, split, prediction, reveal, registry_path,
     basis_split, development_split, public_tasks, prediction_commit, reveal_commit,
     release_manifest) = _receipt_fixture(tmp_path)
    source_path = Path(basis["basis_binding"]["basis_source_artifact_path"])
    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    verdict = validate_basis_qualification(
        basis, expected_protocol_sha256=protocol["identity"]["protocol_sha256"],
        expected_subject_manifest_sha256=_sha_json(subject),
        expected_registry_sha256=protocol["intervention_registry_sha256"],
        registry=protocol["interventions"],
    )
    assert verdict["valid"] is False


def test_state_table_preserves_long_visible_fact_lines() -> None:
    fact = "A" * 64
    rendered = _t_state_table({"block": fact, "query": "Q"})
    assert fact in rendered


def test_reveal_rejects_verifier_artifact_tampering(tmp_path: Path) -> None:
    (protocol, subject, basis, rows, split, prediction, reveal, registry_path,
     basis_split, development_split, public_tasks, prediction_commit, reveal_commit,
     release_manifest) = _receipt_fixture(tmp_path)
    verifier_path = Path(reveal["rows"][0]["verifier_artifact_path"])
    verifier_path.write_text("{}", encoding="utf-8")
    assert validate_reveal_receipt(
        reveal, prediction_receipt=prediction, split_manifest=split,
        prediction_commit=prediction_commit, reveal_commit=reveal_commit,
        repo_root=tmp_path,
    )["valid"] is False


def test_preflight_dry_run_accepts_complete_fixture(tmp_path: Path) -> None:
    (protocol, subject, basis, rows, split, prediction, reveal, registry_path,
     basis_split, development_split, public_tasks, prediction_commit, reveal_commit,
     release_manifest) = _receipt_fixture(tmp_path)
    result = preflight_x1_real_1(
        protocol,
        subject_manifest=subject,
        basis_qualification=basis,
        release_manifest=release_manifest,
        registry_path=registry_path,
        checkpoint_root=registry_path.parent,
        release_root=registry_path.parent,
        source_root=ROOT,
        primary_split_manifest=split,
        basis_split_manifest=basis_split,
        development_split_manifest=development_split,
        public_tasks=public_tasks,
        predictor_identity=prediction["predictor"],
        baseline_commitment=prediction["baseline_commitment"],
        predictor_predictions=prediction["predictions"],
        prediction_receipt=prediction,
        prediction_commit=prediction_commit,
        reveal_receipt=reveal,
        reveal_commit=reveal_commit,
    )
    assert result["status"] == "READY_FOR_EXTERNAL_PREDICTION"
    assert result["check_summary"]["warning"] > 0
    assert result["dry_run"]["model_execution"] is False
    assert result["dry_run"]["training_execution"] is False
    intake_path = tmp_path / "candidate.pt"
    intake_path.write_bytes(b"candidate")
    intake = intake_checkpoint_candidate(intake_path)
    plan = build_execution_plan(protocol, result, intake=intake)
    assert plan["status"] == "BLOCKED_POWER_DEVIATION_REQUIRED"
    assert "POWER_DEVIATION_REQUIRED" in plan["blockers"]
    assert plan["candidate_intake"]["sha256"] == intake["intake_sha256"]
    assert plan["preflight_sha256"] == result["preflight_sha256"]


def test_cli_errors_are_structured(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["preflight", "--protocol", str(ROOT / "missing-protocol.json")])
    output = json.loads(capsys.readouterr().out)
    assert code == 2
    assert output["schema"] == "anra-x1-real-1-cli-error/v1"
    assert output["status"] == "ERROR"
    assert output["next_action"]
    code = main(["not-a-command"])
    usage_output = json.loads(capsys.readouterr().out)
    assert code == 2
    assert usage_output["schema"] == "anra-x1-real-1-cli-error/v1"