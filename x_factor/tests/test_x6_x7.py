from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from x_factor.x6_x7 import (
    ARM_IDS,
    READINESS_SCHEMA,
    X6X7Error,
    X7_READINESS_SCHEMA,
    assess_x6_readiness,
    audit_x1_entry,
    build_dataset,
    build_evaluation_commitment,
    build_source_release,
    build_x6_result_receipt,
    build_x6_run_manifest,
    build_x7_readiness,
    build_x7_receipt,
    load_json,
    make_arm_manifest,
    make_continuation_manifest,
    make_parent_identity,
    make_split_bundle,
    make_split_manifest,
    protocol_sha256,
    sha_file,
    sha_json,
    source_closure_sha256,
    validate_arm_manifest,
    validate_protocol,
    validate_split_bundle,
    validate_x6_result_receipt,
    validate_x7_receipt,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "x_factor" / "protocols" / "x6_x7_v1.json"
LEGACY_X1_PATH = ROOT / "output" / "x1_real_receipt.json"
INTERVENTION_IDS = ("NO_CHANGE", "CANONICAL_CONTEXT")
TARGET_INTERVENTION = "CANONICAL_CONTEXT"


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _protocol(authorized: bool = True) -> dict:
    files = [{"path": "x_factor/ladder.py", "sha256": "a" * 64}]
    thresholds = {
        "authority_status": "AUTHORIZED" if authorized else "PROPOSED_PENDING_REVIEW",
        "min_unaided_gain_vs_parent": 0.10,
        "min_unaided_gain_vs_control": 0.05,
        "min_lift_reduction_vs_parent": 0.10,
        "min_lift_reduction_vs_control": 0.05,
        "minimum_independent_worlds": 2,
        "minimum_paired_seeds": 2,
        "bootstrap_resamples": 100,
        "require_ci_above_zero": True,
        "min_transfer_gain_by_axis": {axis: 0.05 for axis in (
            "KNOWN_STRUCTURE_INSTANCE", "SURFACE_RENDERING", "CAUSAL_STRUCTURE", "TASK_FAMILY"
        )},
        "x7_min_transfer_gain_by_axis": {axis: 0.05 for axis in (
            "KNOWN_STRUCTURE_INSTANCE", "SURFACE_RENDERING", "CAUSAL_STRUCTURE", "TASK_FAMILY"
        )},
        "x7_min_unaided_retention_vs_parent": 0.05,
        "x7_min_unaided_gain_vs_control": 0.05,
        "x7_min_target_lift_reduction_vs_parent": 0.10,
        "x7_min_target_lift_reduction_vs_control": 0.05,
        "x7_max_unrelated_lift_change": 0.10,
    }
    body = {
        "schema": "anra-x6-x7-protocol/v1",
        "phase": "FROZEN_CONTRACT",
        "execution_authorized": False,
        "canonical_authority": {
            "base_revision": "a" * 40,
            "retention_floor_absolute_accuracy": 0.10,
            "rehearsal_min_fraction": 0.50,
        },
        "identity": {
            "base_revision": "a" * 40,
            "source_paths": [item["path"] for item in files],
            "source_closure_sha256": source_closure_sha256("a" * 40, files),
            "protocol_sha256": "",
        },
        "decision_thresholds": thresholds,
        "invalid_run_conditions": ["altered_protocol", "split_collision", "incomplete_arm"],
    }
    body["identity"]["protocol_sha256"] = protocol_sha256(body)
    validate_protocol(body)
    return body


def _x1_receipt(protocol: dict) -> dict:
    body = {
        "schema": "anra-x1-real-1-analysis/v1",
        "phase": "ANALYSIS_ACCEPTED",
        "protocol_sha256": "1" * 64,
        "subject_manifest_sha256": "1" * 64,
        "release_manifest_sha256": "3" * 64,
        "basis_qualification_sha256": "4" * 64,
        "basis_split_manifest_sha256": "5" * 64,
        "development_split_manifest_sha256": "6" * 64,
        "prediction_receipt_sha256": "7" * 64,
        "prediction_phase_commit_sha256": "8" * 64,
        "reveal_receipt_sha256": "9" * 64,
        "reveal_phase_commit_sha256": "b" * 64,
        "prediction_verdict": {"status": "PASS", "checks": {"gate": True}},
        "decision_verdict": {"status": "PASS", "checks": {"gate": True}},
        "replication": {"valid": True},
        "transfer_verdict": {
            "status": "SUPPORTED_SCOPED_CHECKPOINT_PENDING",
            "independent_task_cohort": True,
            "cross_checkpoint": False,
        },
        "decision": {
            "status": "SUPPORTED_SCOPED",
            "supported": True,
            "primary_gate_pass": True,
            "prediction_gate_pass": True,
            "decision_gate_pass": True,
        },
    }
    body["analysis_receipt_sha256"] = sha_json(body)
    return body


def _entry_audit(protocol: dict) -> dict:
    return audit_x1_entry(_x1_receipt(protocol), "c" * 64, "1" * 64)


def _parent() -> dict:
    hashes = {key: character * 64 for key, character in {
        "subject_manifest_sha256": "1",
        "checkpoint_file_sha256": "2",
        "parameter_sha256": "3",
        "parent_state_tree_sha256": "4",
        "model_config_sha256": "5",
        "tokenizer_artifact_sha256": "6",
        "tokenizer_identity_sha256": "7",
        "runtime_source_sha256": "8",
        "readiness_receipt_sha256": "9",
        "qualification_receipt_sha256": "a",
        "identity_attestation_sha256": "b",
        "continuation_contract_sha256": "c",
    }.items()}
    return make_parent_identity({
        **hashes,
        "checkpoint_path": "checkpoints/qualified-parent.pt",
        "runtime_source_revision": "runtime-revision-001",
        "source_commit": "b" * 40,
        "training_lineage": "qualified-lineage",
        "stage": "qualified-stage",
        "global_step": 1000,
        "research_subject": True,
        "qualification_state": "QUALIFIED",
        "checkpoint_file_verified": True,
        "continuation_code": "continuation-v1",
    })


def _confirmation_clusters(prefix: str) -> list[dict]:
    clusters = []
    for axis in ("KNOWN_STRUCTURE_INSTANCE", "SURFACE_RENDERING", "CAUSAL_STRUCTURE", "TASK_FAMILY"):
        for repeat in range(2):
            task_id = f"{prefix}-task-{axis.lower()}-{repeat}"
            clusters.append({
                "cluster_id": f"cluster-{prefix}-{axis.lower()}-{repeat}",
                "latent_world_id": f"world-{prefix}-{axis.lower()}-{repeat}",
                "causal_structure_id": f"structure-{prefix}-{axis.lower()}-{repeat}",
                "source_id": f"source-{prefix}-{axis.lower()}-{repeat}",
                "task_ids": [task_id],
                "task_content_sha256s": {task_id: _hash_text(f"content-{prefix}-{axis}-{repeat}")},
                "transfer_axes": [axis],
            })
    return clusters


def _split(role: str, protocol_sha: str) -> dict:
    prefix = role.lower().replace("_", "-")
    if role == "X6_EXPERIENCE":
        task_ids = ["experience-repair-task", "experience-raw-task"]
        axes = ["KNOWN_STRUCTURE_INSTANCE"]
    elif role in {"X6_CONFIRMATION", "X7_FOLLOWUP"}:
        return make_split_manifest({
            "split_id": prefix,
            "role": role,
            "protocol_sha256": protocol_sha,
            "generator_id": f"generator-{prefix}",
            "generator_sha256": _hash_text(prefix),
            "seed": 1000 + len(role),
            "clusters": _confirmation_clusters(prefix),
            "freeze": {
                "status": "FROZEN_SEALED",
                "frozen_at": "2026-09-25T00:00:00+00:00",
                "frozen_before_training": True,
                "inspection_count": 0,
                "retired_if_inspected": True,
            },
        })
    else:
        task_ids = [f"{prefix}-task"]
        axes = ["KNOWN_STRUCTURE_INSTANCE"]
    cluster = {
        "cluster_id": f"cluster-{prefix}",
        "latent_world_id": f"world-{prefix}",
        "causal_structure_id": f"structure-{prefix}",
        "source_id": f"source-{prefix}",
        "task_ids": task_ids,
        "task_content_sha256s": {task_id: _hash_text(task_id) for task_id in task_ids},
        "transfer_axes": axes,
    }
    return make_split_manifest({
        "split_id": prefix,
        "role": role,
        "protocol_sha256": protocol_sha,
        "generator_id": f"generator-{prefix}",
        "generator_sha256": _hash_text(prefix),
        "seed": 1000 + len(role),
        "clusters": [cluster],
        "freeze": {
            "status": "FROZEN_SEALED",
            "frozen_at": "2026-09-25T00:00:00+00:00",
            "frozen_before_training": True,
            "inspection_count": 0,
            "retired_if_inspected": True,
        },
    })


def _split_bundle(protocol: dict) -> dict:
    protocol_sha = protocol["identity"]["protocol_sha256"]
    return make_split_bundle(protocol_sha, {
        role: _split(role, protocol_sha)
        for role in (
            "X6_EXPERIENCE", "X6_REPLAY", "X6_DEVELOPMENT",
            "X6_RETENTION", "X6_CONFIRMATION", "X7_FOLLOWUP",
        )
    })


def _experience(
    experience_class: str, task_id: str, split_bundle: dict,
    intervention_id: str = "CANONICAL_CONTEXT", registry_sha256: str = "d" * 64,
) -> dict:
    role = "X6_REPLAY" if experience_class == "REPLAY" else "X6_EXPERIENCE"
    split = split_bundle["splits"][role]
    cluster = split["clusters"][0]
    prompt = f"Unaided prompt for {task_id}."
    target = f"Verified target for {task_id}."
    assisted_prompt = f"Assisted prompt for {task_id}."
    row = {
        "schema": "anra-x6-repair-experience/v1",
        "experience_id": f"experience-{experience_class.lower()}-{task_id}",
        "experience_class": experience_class,
        "source_task_id": task_id,
        "cluster_id": cluster["cluster_id"],
        "latent_world_id": cluster["latent_world_id"],
        "causal_structure_id": cluster["causal_structure_id"],
        "source_split_id": split["split_id"],
        "intervention_id": intervention_id,
        "intervention_version": 1,
        "intervention_registry_sha256": registry_sha256,
        "legality_receipt_sha256": "1" * 64,
        "intervention_information_preserving": True,
        "intervention_answer_revealing": False,
        "baseline_failed": True,
        "baseline_raw_output_sha256": _hash_text(f"baseline-{task_id}"),
        "assisted_prompt_sha256": _hash_text(assisted_prompt),
        "assisted_raw_output_sha256": _hash_text(f"assisted-output-{task_id}"),
        "assisted_verified_correct": experience_class == "REPAIR_SUCCESS",
        "verifier_receipt_sha256": "2" * 64,
        "verifier_source_sha256": "3" * 64,
        "verification_independent": True,
        "target_text": target,
        "target_sha256": _hash_text(target),
        "unaided_prompt": prompt,
        "unaided_prompt_sha256": _hash_text(prompt),
        "support_removal_transformation_id": "support-removal-v1",
        "support_removal_receipt_sha256": "4" * 64,
        "external_support_removed": True,
        "unaided_token_count": 5,
        "target_token_count": 2,
        "sealed_outcome": False,
        "evaluation_label": False,
        "experience_sha256": "",
    }
    row["experience_sha256"] = sha_json({key: value for key, value in row.items() if key != "experience_sha256"})
    return row


def _interventions() -> list[dict]:
    return [
        {
            "id": "NO_CHANGE", "version": 1, "role": "NULL_CONTROL", "assistance": "A0",
            "cost": 0, "information_preserving": True, "answer_revealing": False,
        },
        {
            "id": "CANONICAL_CONTEXT", "version": 1, "role": "REPAIR", "assistance": "A1",
            "cost": 1, "information_preserving": True, "answer_revealing": False,
        },
    ]


def _datasets(protocol: dict, split_bundle: dict) -> dict[str, dict]:
    registry_sha = "d" * 64
    experience = [
        _experience("REPAIR_SUCCESS", "experience-repair-task", split_bundle, registry_sha256=registry_sha),
        _experience("RAW_FAILURE", "experience-raw-task", split_bundle, registry_sha256=registry_sha),
    ]
    replay = [
        _experience("REPLAY", "x6-replay-task", split_bundle, intervention_id="NO_CHANGE", registry_sha256=registry_sha),
    ]
    protocol_sha = protocol["identity"]["protocol_sha256"]
    return {
        "REPAIR_INTERNALIZATION": build_dataset(
            protocol_sha, "REPAIR_INTERNALIZATION", experience, replay,
            split_bundle, _interventions(), registry_sha,
        )["manifest"],
        "RAW_FAILURE_SFT_CONTROL": build_dataset(
            protocol_sha, "RAW_FAILURE_SFT_CONTROL", experience, replay,
            split_bundle, _interventions(), registry_sha,
        )["manifest"],
    }


def _exposure() -> dict:
    return {
        "optimizer_updates": 10,
        "training_examples": 2,
        "unaided_tokens": 10,
        "target_tokens": 4,
        "replay_examples": 1,
        "replay_target_tokens": 2,
        "replay_fraction": 0.5,
        "optimizer_sha256": "1" * 64,
        "schedule_sha256": "2" * 64,
        "checkpoint_cadence_sha256": "3" * 64,
        "evaluation_timing_sha256": "4" * 64,
        "data_opportunity_sha256": "5" * 64,
    }


def _arm_manifest(protocol: dict, parent: dict, datasets: dict[str, dict]) -> dict:
    return make_arm_manifest(
        protocol["identity"]["protocol_sha256"], 17, parent, datasets, _exposure(),
    )


def _continuations(protocol: dict, parent: dict, seed: int = 17) -> list[dict]:
    result = []
    for index, arm_id in enumerate(("REPAIR_INTERNALIZATION", "RAW_FAILURE_SFT_CONTROL"), start=1):
        hashes = {key: character * 64 for key, character in {
            "checkpoint_file_sha256": "1", "parameter_sha256": "2",
            "parent_checkpoint_file_sha256": "3", "optimizer_state_sha256": "4",
            "scheduler_state_sha256": "5", "rng_state_sha256": "6",
            "sampler_state_sha256": "7", "dataloader_cursor_sha256": "8",
            "token_ledger_sha256": "9", "source_tree_sha256": "a",
            "resume_contract_sha256": "b", "fresh_process_restore_receipt_sha256": "c",
        }.items()}
        result.append(make_continuation_manifest({
            **hashes,
            "parent_checkpoint_file_sha256": parent["checkpoint_file_sha256"],
            "protocol_sha256": protocol["identity"]["protocol_sha256"],
            "arm_id": arm_id,
            "seed": seed,
            "global_step": seed * 100 + index * 10,
            "checkpoint_id": "d" * 64,
            "resume_validated": True,
        }))
    return result


def _source_release(protocol: dict) -> dict:
    identity = protocol["identity"]
    return build_source_release({
        "protocol_sha256": identity["protocol_sha256"],
        "source_closure_sha256": identity["source_closure_sha256"],
        "base_revision": identity["base_revision"],
        "release_revision": "c" * 40,
        "source_tree_status": "CLEAN",
        "files": [{"path": "x_factor/ladder.py", "sha256": "a" * 64}],
        "custodian": "test-custodian",
    })


def _run_manifest(
    protocol: dict, entry_audit: dict, parent: dict, split_bundle: dict,
    arm_manifest: dict, datasets: dict[str, dict],
) -> dict:
    second_arm = copy.deepcopy(arm_manifest)
    second_arm["seed"] = 18
    second_arm["arm_manifest_sha256"] = sha_json({
        key: value for key, value in second_arm.items() if key != "arm_manifest_sha256"
    })
    seeds = (17, 18)
    return build_x6_run_manifest(
        protocol,
        entry_audit,
        parent,
        split_bundle,
        [arm_manifest, second_arm],
        {str(seed): datasets for seed in seeds},
        {
            str(seed): {
                manifest["arm_id"]: manifest
                for manifest in _continuations(protocol, parent, seed)
            }
            for seed in seeds
        },
        _source_release(protocol),
        {
            "registry_schema": "anra-x1-intervention-registry/v1",
            "registry_sha256": "d" * 64,
            "intervention_ids": list(INTERVENTION_IDS),
            "transformation_hashes": {intervention_id: "6" * 64 for intervention_id in INTERVENTION_IDS},
            "information_class": "INFORMATION_PRESERVING",
            "legality_inputs": "VISIBLE_TASK_ONLY",
            "cost_model_sha256": "7" * 64,
            "target_intervention": TARGET_INTERVENTION,
        },
        {
            "optimizer_sha256": _exposure()["optimizer_sha256"],
            "schedule_sha256": _exposure()["schedule_sha256"],
            "data_opportunity_sha256": _exposure()["data_opportunity_sha256"],
            "checkpoint_cadence_sha256": _exposure()["checkpoint_cadence_sha256"],
            "evaluation_timing_sha256": _exposure()["evaluation_timing_sha256"],
            "resume_contract_sha256": "8" * 64,
        },
        {"kind": "fixed", "identity_sha256": "9" * 64},
        "2026-09-25T02:00:00+00:00",
    )


def _evaluation_commitment(
    protocol: dict, split_bundle: dict, arm_manifest: dict,
    role: str = "X6_CONFIRMATION",
) -> dict:
    task_ids = [
        task_id
        for cluster in split_bundle["splits"][role]["clusters"]
        for task_id in cluster["task_ids"]
    ]
    return build_evaluation_commitment({
        "protocol_sha256": protocol["identity"]["protocol_sha256"],
        "run_manifest_sha256": "1" * 64,
        "split_bundle_sha256": split_bundle["bundle_sha256"],
        "arm_manifest_sha256": arm_manifest["arm_manifest_sha256"],
        "checkpoint_identity_sha256s": {arm_id: "2" * 64 for arm_id in ARM_IDS},
        "intervention_registry_sha256": "3" * 64,
        "x1_policy_receipt_sha256": None,
        "sealed_outcome_receipt_sha256": None,
        "evaluator_source_sha256": "4" * 64,
        "verifier_source_sha256": "5" * 64,
        "task_ids": task_ids,
        "retention_task_ids": [
            task_id
            for cluster in split_bundle["splits"]["X6_RETENTION"]["clusters"]
            for task_id in cluster["task_ids"]
        ],
        "intervention_ids": list(INTERVENTION_IDS),
        "frozen_at": "2026-09-25T01:00:00+00:00",
        "outcomes_loaded": False,
        "checkpoint_selection_uses_evaluation_labels": False,
    })


def _outcome(correct: bool, cost: float) -> dict:
    return {
        "correct": correct,
        "valid_stop": True,
        "invalid_output": False,
        "abstained": False,
        "predicted_probability_correct": 0.8 if correct else 0.2,
        "cost": cost,
    }


def _observations(split_bundle: dict, role: str) -> list[dict]:
    split = split_bundle["splits"][role]
    task_rows = [
        (cluster["cluster_id"], task_id, cluster["transfer_axes"][0])
        for cluster in split["clusters"]
        for task_id in cluster["task_ids"]
    ]
    result = []
    for seed in (17, 18):
        for arm_id in ARM_IDS:
            for index, (cluster_id, task_id, axis) in enumerate(task_rows):
                if arm_id == "PARENT":
                    unaided_correct = index in {1, 3}
                elif arm_id == "RAW_FAILURE_SFT_CONTROL":
                    unaided_correct = False
                else:
                    unaided_correct = True
                no_change_correct = unaided_correct
                target_correct = True
                result.append({
                    "arm_id": arm_id,
                    "seed": seed,
                    "task_id": task_id,
                    "cluster_id": cluster_id,
                    "transfer_axis": axis,
                    "unaided": _outcome(unaided_correct, 0.0),
                    "assisted": [
                        {**_outcome(no_change_correct, 0.0), "intervention_id": "NO_CHANGE"},
                        {**_outcome(target_correct, 1.0), "intervention_id": "CANONICAL_CONTEXT"},
                    ],
                })
    return result


def _repair_choice_rows(observations: list[dict]) -> list[dict]:
    return [{
        "schema": "anra-x6-repair-choice/v1",
        "arm_id": row["arm_id"],
        "seed": row["seed"],
        "task_id": row["task_id"],
        "cluster_id": row["cluster_id"],
        "policy_receipt_sha256": "a" * 64,
        "selected_intervention_id": "CANONICAL_CONTEXT",
        "unaided_correct": row["unaided"]["correct"],
        "selected_correct": row["assisted"][1]["correct"],
        "intervention_cost": 1.0,
    } for row in observations]


def _retention() -> list[dict]:
    rows = []
    for seed in (17, 18):
        for arm_id, child_score in (
            ("REPAIR_INTERNALIZATION", 0.75), ("RAW_FAILURE_SFT_CONTROL", 0.78),
        ):
            rows.append({
                "arm_id": arm_id,
                "seed": seed,
                "capability_id": "qualified-binding",
                "parent_score": 0.80,
                "child_score": child_score,
                "n_independent_worlds": 8,
                "score_semantics": "exact_answer_accuracy",
            })
    return rows


def _costs(include_training: bool) -> dict:
    costs = {"evaluation": 2.0, "intervention": 1.0, "checkpoint": 0.5}
    if include_training:
        costs["training"] = 3.0
    return costs


def test_historical_x1_receipt_never_unlocks_x6() -> None:
    receipt = load_json(LEGACY_X1_PATH)
    audit = audit_x1_entry(receipt, sha_file(LEGACY_X1_PATH))
    assert audit["status"] == "BLOCKED"
    assert audit["verdict"] == "X6_ENTRY_BLOCKED"
    assert any("schema" in error for error in audit["errors"])


def test_valid_x1_analysis_receipt_passes_entry_contract() -> None:
    protocol = _protocol()
    audit = _entry_audit(protocol)
    assert audit["status"] == "PASS"
    assert audit["checkpoint_replication_status"] == "PENDING"


def test_parent_requires_qualification_and_exact_continuation_identity() -> None:
    parent = _parent()
    assert parent["qualification_state"] == "QUALIFIED"
    parent["qualification_state"] = "CALIBRATION_ONLY"
    parent["parent_identity_sha256"] = sha_json({
        key: value for key, value in parent.items() if key != "parent_identity_sha256"
    })
    with pytest.raises(X6X7Error, match="qualification state"):
        from x_factor.x6_x7 import validate_parent_identity
        validate_parent_identity(parent)


def test_split_bundle_rejects_relabelled_causal_structure_collision() -> None:
    protocol = _protocol()
    bundle = _split_bundle(protocol)
    x7 = copy.deepcopy(bundle["splits"]["X7_FOLLOWUP"])
    x7["clusters"][0]["causal_structure_id"] = bundle["splits"]["X6_CONFIRMATION"]["clusters"][0]["causal_structure_id"]
    x7["manifest_sha256"] = sha_json({key: value for key, value in x7.items() if key != "manifest_sha256"})
    bundle["splits"]["X7_FOLLOWUP"] = x7
    bundle["bundle_sha256"] = sha_json({key: value for key, value in bundle.items() if key != "bundle_sha256"})
    with pytest.raises(X6X7Error, match="causal_structure_ids collision"):
        validate_split_bundle(bundle)


def test_dataset_firewall_rejects_evaluation_labels() -> None:
    protocol = _protocol()
    bundle = _split_bundle(protocol)
    row = _experience("REPAIR_SUCCESS", "experience-repair-task", bundle)
    row["evaluation_label"] = True
    row["experience_sha256"] = sha_json({key: value for key, value in row.items() if key != "experience_sha256"})
    with pytest.raises(X6X7Error, match="sealed outcomes"):
        build_dataset(
            protocol["identity"]["protocol_sha256"], "REPAIR_INTERNALIZATION",
            [row], [_experience("REPLAY", "x6-replay-task", bundle, intervention_id="NO_CHANGE")],
            bundle, _interventions(), "d" * 64,
        )


def test_repair_dataset_requires_external_support_removal() -> None:
    protocol = _protocol()
    bundle = _split_bundle(protocol)
    row = _experience("REPAIR_SUCCESS", "experience-repair-task", bundle)
    row["external_support_removed"] = False
    row["experience_sha256"] = sha_json({key: value for key, value in row.items() if key != "experience_sha256"})
    with pytest.raises(X6X7Error, match="support removal"):
        build_dataset(
            protocol["identity"]["protocol_sha256"], "REPAIR_INTERNALIZATION",
            [row], [_experience("REPLAY", "x6-replay-task", bundle, intervention_id="NO_CHANGE")],
            bundle, _interventions(), "d" * 64,
        )


def test_arm_manifest_rejects_unequal_training_dose() -> None:
    protocol = _protocol()
    parent = _parent()
    datasets = _datasets(protocol, _split_bundle(protocol))
    manifest = _arm_manifest(protocol, parent, datasets)
    control = next(arm for arm in manifest["arms"] if arm["arm_id"] == "RAW_FAILURE_SFT_CONTROL")
    control["exposure"]["optimizer_updates"] += 1
    manifest["arm_manifest_sha256"] = sha_json({key: value for key, value in manifest.items() if key != "arm_manifest_sha256"})
    with pytest.raises(X6X7Error, match="not exposure-matched"):
        validate_arm_manifest(manifest)


def test_evaluation_commitment_rejects_loaded_outcomes() -> None:
    protocol = _protocol()
    bundle = _split_bundle(protocol)
    arm = _arm_manifest(protocol, _parent(), _datasets(protocol, bundle))
    commitment = _evaluation_commitment(protocol, bundle, arm)
    body = {key: value for key, value in commitment.items() if key not in {"schema", "evaluation_commitment_sha256"}}
    body["outcomes_loaded"] = True
    with pytest.raises(X6X7Error, match="boundary"):
        build_evaluation_commitment(body)


def test_readiness_accepts_complete_valid_fixture_but_never_authorizes_execution() -> None:
    protocol = _protocol()
    parent = _parent()
    bundle = _split_bundle(protocol)
    arm = _arm_manifest(protocol, parent, _datasets(protocol, bundle))
    commitment = _evaluation_commitment(protocol, bundle, arm)
    readiness = assess_x6_readiness(
        protocol, _entry_audit(protocol), parent, bundle, arm,
        _continuations(protocol, parent), commitment, _source_release(protocol),
        _datasets(protocol, bundle),
        _run_manifest(protocol, _entry_audit(protocol), parent, bundle, arm, _datasets(protocol, bundle)),
    )
    assert readiness["status"] == "READY"
    assert readiness["execution_authorized"] is False


def test_current_unresolved_threshold_and_invalid_x1_block_readiness() -> None:
    protocol = _protocol(authorized=False)
    parent = _parent()
    bundle = _split_bundle(protocol)
    arm = _arm_manifest(protocol, parent, _datasets(protocol, bundle))
    commitment = _evaluation_commitment(protocol, bundle, arm)
    invalid_x1 = audit_x1_entry(load_json(LEGACY_X1_PATH), sha_file(LEGACY_X1_PATH))
    readiness = assess_x6_readiness(
        protocol, invalid_x1, parent, bundle, arm,
        _continuations(protocol, parent), commitment, _source_release(protocol),
        _datasets(protocol, bundle),
        _run_manifest(protocol, _entry_audit(protocol), parent, bundle, arm, _datasets(protocol, bundle)),
    )
    assert readiness["status"] == "PREREQUISITE_BLOCKED"
    assert "X1_ENTRY_NOT_SUPPORTED" in readiness["blockers"]
    assert "RAW_GAIN_AND_LIFT_REDUCTION_THRESHOLDS_UNRESOLVED" in readiness["blockers"]


def test_x6_receipt_requires_raw_gain_lift_reduction_control_and_retention() -> None:
    protocol = _protocol()
    parent = _parent()
    bundle = _split_bundle(protocol)
    arm = _arm_manifest(protocol, parent, _datasets(protocol, bundle))
    readiness = assess_x6_readiness(
        protocol, _entry_audit(protocol), parent, bundle, arm,
        _continuations(protocol, parent), _evaluation_commitment(protocol, bundle, arm),
        _source_release(protocol), _datasets(protocol, bundle),
        _run_manifest(protocol, _entry_audit(protocol), parent, bundle, arm, _datasets(protocol, bundle)),
    )
    receipt = build_x6_result_receipt(
        protocol, readiness, readiness["evidence"]["run_manifest_sha256"], bundle,
        _observations(bundle, "X6_CONFIRMATION"), INTERVENTION_IDS,
        TARGET_INTERVENTION, _retention(), _costs(True),
        {"kind": "fixed", "identity_sha256": "a" * 64}, n_resamples=100,
    )
    assert receipt["decision"]["status"] == "SUPPORTED_SCOPED_WITH_TRANSFER"
    assert receipt["claims"] == {
        "assisted_repair": "measured separately",
        "predictive_selection": "inherited only from a valid X1 receipt; not re-established here",
        "internalization": True,
        "retention": True,
        "transfer": True,
        "agi": False,
    }
    assert validate_x6_result_receipt(receipt, protocol)["status"] == "SUPPORTED_SCOPED_WITH_TRANSFER"
    tampered = copy.deepcopy(receipt)
    cluster = next(iter(tampered["cluster_results"]["REPAIR_INTERNALIZATION"]["17"]))
    tampered["cluster_results"]["REPAIR_INTERNALIZATION"]["17"][cluster]["unaided"] = 0.0
    tampered["x6_result_receipt_sha256"] = sha_json({
        key: value for key, value in tampered.items() if key != "x6_result_receipt_sha256"
    })
    with pytest.raises(X6X7Error, match="paired-effect arithmetic"):
        validate_x6_result_receipt(tampered, protocol)


def test_repair_choice_utility_is_reported_only_with_a_frozen_policy() -> None:
    protocol = _protocol()
    parent = _parent()
    bundle = _split_bundle(protocol)
    datasets = _datasets(protocol, bundle)
    arm = _arm_manifest(protocol, parent, datasets)
    entry = _entry_audit(protocol)
    run = _run_manifest(protocol, entry, parent, bundle, arm, datasets)
    readiness = assess_x6_readiness(
        protocol, entry, parent, bundle, arm,
        _continuations(protocol, parent), _evaluation_commitment(protocol, bundle, arm),
        _source_release(protocol), datasets, run,
    )
    observations = _observations(bundle, "X6_CONFIRMATION")
    receipt = build_x6_result_receipt(
        protocol, readiness, run["run_manifest_sha256"], bundle,
        observations, INTERVENTION_IDS, TARGET_INTERVENTION,
        _retention(), _costs(True),
        {"kind": "fixed", "identity_sha256": "a" * 64},
        n_resamples=100,
        repair_choice_rows=_repair_choice_rows(observations),
        repair_choice_cost_lambda=0.25,
    )
    assert receipt["repair_choice_utility"]["status"] == "AVAILABLE"
    assert receipt["repair_choice_utility"]["policy_receipt_sha256"] == "a" * 64
    assert receipt["repair_choice_utility"]["cost_lambda"] == 0.25


def test_assisted_only_improvement_is_not_internalization() -> None:
    protocol = _protocol()
    parent = _parent()
    bundle = _split_bundle(protocol)
    arm = _arm_manifest(protocol, parent, _datasets(protocol, bundle))
    readiness = assess_x6_readiness(
        protocol, _entry_audit(protocol), parent, bundle, arm,
        _continuations(protocol, parent), _evaluation_commitment(protocol, bundle, arm),
        _source_release(protocol), _datasets(protocol, bundle),
        _run_manifest(protocol, _entry_audit(protocol), parent, bundle, arm, _datasets(protocol, bundle)),
    )
    observations = _observations(bundle, "X6_CONFIRMATION")
    for row in observations:
        if row["arm_id"] == "REPAIR_INTERNALIZATION":
            control = next(
                item for item in observations
                if item["arm_id"] == "RAW_FAILURE_SFT_CONTROL"
                and item["seed"] == row["seed"] and item["task_id"] == row["task_id"]
            )
            row["unaided"] = copy.deepcopy(control["unaided"])
            row["assisted"][0] = copy.deepcopy(control["assisted"][0])
    receipt = build_x6_result_receipt(
        protocol, readiness, readiness["evidence"]["run_manifest_sha256"], bundle, observations, INTERVENTION_IDS,
        TARGET_INTERVENTION, _retention(), _costs(True),
        {"kind": "fixed", "identity_sha256": "a" * 64}, n_resamples=100,
    )
    assert receipt["decision"]["status"] == "NOT_SUPPORTED"
    assert receipt["claims"]["internalization"] is False


def test_x7_is_separately_gated_and_requires_fresh_cohort() -> None:
    protocol = _protocol()
    parent = _parent()
    bundle = _split_bundle(protocol)
    arm = _arm_manifest(protocol, parent, _datasets(protocol, bundle))
    x6_readiness = assess_x6_readiness(
        protocol, _entry_audit(protocol), parent, bundle, arm,
        _continuations(protocol, parent), _evaluation_commitment(protocol, bundle, arm),
        _source_release(protocol), _datasets(protocol, bundle),
        _run_manifest(protocol, _entry_audit(protocol), parent, bundle, arm, _datasets(protocol, bundle)),
    )
    x6 = build_x6_result_receipt(
        protocol, x6_readiness, x6_readiness["evidence"]["run_manifest_sha256"], bundle,
        _observations(bundle, "X6_CONFIRMATION"), INTERVENTION_IDS,
        TARGET_INTERVENTION, _retention(), _costs(True),
        {"kind": "fixed", "identity_sha256": "a" * 64}, n_resamples=100,
    )
    x7_commitment = _evaluation_commitment(protocol, bundle, arm, "X7_FOLLOWUP")
    readiness = build_x7_readiness(protocol, x6, bundle, x7_commitment, _source_release(protocol))
    assert readiness["schema"] == X7_READINESS_SCHEMA
    assert readiness["status"] == "READY"
    receipt = build_x7_receipt(
        protocol, readiness, x6, bundle,
        _observations(bundle, "X7_FOLLOWUP"), INTERVENTION_IDS,
        TARGET_INTERVENTION, _retention(), _costs(False),
        {"kind": "fixed", "identity_sha256": "a" * 64}, n_resamples=100,
    )
    assert receipt["decision"]["status"] == "DEPENDENCE_FELL"
    assert validate_x7_receipt(receipt, protocol)["status"] == "DEPENDENCE_FELL"
    assert receipt["agi"] is False


def test_module_import_does_not_load_torch_or_anra_runtime() -> None:
    code = (
        "import sys; import x_factor.x6_x7; "
        "assert 'torch' not in sys.modules; assert 'anra_core' not in sys.modules; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "ok"


def test_committed_protocol_is_source_bound_when_present() -> None:
    if not PROTOCOL_PATH.exists():
        pytest.skip("protocol artifact is added after interface tests")
    protocol = load_json(PROTOCOL_PATH)
    verdict = validate_protocol(protocol, ROOT, check_source_closure=True)
    assert verdict["valid"] is True
