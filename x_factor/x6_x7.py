from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence


PROTOCOL_SCHEMA = "anra-x6-x7-protocol/v1"
ENTRY_AUDIT_SCHEMA = "anra-x1-to-x6-entry-audit/v1"
PARENT_SCHEMA = "anra-x6-parent-identity/v1"
SPLIT_SCHEMA = "anra-x6-x7-split/v1"
SPLIT_BUNDLE_SCHEMA = "anra-x6-x7-split-bundle/v1"
EXPERIENCE_SCHEMA = "anra-x6-repair-experience/v1"
TRAINING_RECORD_SCHEMA = "anra-x6-training-record/v1"
DATASET_SCHEMA = "anra-x6-dataset-manifest/v1"
ARM_SCHEMA = "anra-x6-arm-manifest/v1"
CONTINUATION_SCHEMA = "anra-x6-continuation-manifest/v1"
EVALUATION_COMMITMENT_SCHEMA = "anra-x6-evaluation-commitment/v1"
SOURCE_RELEASE_SCHEMA = "anra-x6-source-release/v1"
RUN_MANIFEST_SCHEMA = "anra-x6-run-manifest/v1"
READINESS_SCHEMA = "anra-x6-readiness/v1"
X6_RECEIPT_SCHEMA = "anra-x6-result-receipt/v1"
REPAIR_CHOICE_SCHEMA = "anra-x6-repair-choice/v1"
X7_READINESS_SCHEMA = "anra-x7-readiness/v1"
X7_RECEIPT_SCHEMA = "anra-x7-followup-receipt/v1"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")

SPLIT_ROLES = (
    "X6_EXPERIENCE",
    "X6_REPLAY",
    "X6_DEVELOPMENT",
    "X6_RETENTION",
    "X6_CONFIRMATION",
    "X7_FOLLOWUP",
)
TRAINING_ARMS = ("REPAIR_INTERNALIZATION", "RAW_FAILURE_SFT_CONTROL")
ARM_IDS = ("PARENT", *TRAINING_ARMS)
TRANSFER_AXES = (
    "KNOWN_STRUCTURE_INSTANCE",
    "SURFACE_RENDERING",
    "CAUSAL_STRUCTURE",
    "TASK_FAMILY",
)
FORBIDDEN_POLICY_KEYS = frozenset({
    "answer_key", "cluster_id", "correct", "correctness", "correctness_label",
    "evaluation_label", "evaluator_metadata", "family_id", "gold", "gold_answer",
    "hidden_label", "is_correct", "latent_world_id", "outcome", "outcomes",
    "sealed_outcome", "source_id", "target", "transfer_axis", "world_id",
})
ALLOWED_TRAINING_POLICY_KEYS = frozenset({"unaided_prompt", "target_text"})


class X6X7Error(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise X6X7Error(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise X6X7Error(f"nonstandard JSON constant is forbidden: {value}")


def _plain(value: Any, name: str = "value") -> Any:
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise X6X7Error(f"{name} contains invalid Unicode") from exc
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise X6X7Error(f"{name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise X6X7Error(f"{name} contains a non-string key")
            result[key] = _plain(item, f"{name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{name}[{index}]") for index, item in enumerate(value)]
    raise X6X7Error(f"{name} contains unsupported value {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="strict")
        return _plain(json.loads(
            text, object_pairs_hook=_strict_object, parse_constant=_reject_constant,
        ))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise X6X7Error(f"unable to load strict JSON {path}: {exc}") from exc


def _without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    body = dict(value)
    for key in keys:
        body.pop(key, None)
    return body


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise X6X7Error(f"{name} must be an object")
    return dict(value)


def _require_keys(value: Mapping[str, Any], required: set[str], name: str, optional: set[str] | None = None) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - set(optional or set()))
    if missing or unknown:
        raise X6X7Error(f"{name} keys invalid; missing={missing}, unknown={unknown}")


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise X6X7Error(f"{name} must be a lowercase SHA256 digest")
    return value


def _id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not OPAQUE_ID.fullmatch(value):
        raise X6X7Error(f"{name} must be an opaque identifier")
    return value


def _relative_path(value: Any, name: str) -> str:
    path = Path(_text(value, name))
    if path.is_absolute() or ".." in path.parts:
        raise X6X7Error(f"{name} must be a relative path without parent traversal")
    return path.as_posix()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise X6X7Error(f"{name} must be non-empty text")
    value.encode("utf-8", errors="strict")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise X6X7Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise X6X7Error(f"{name} must be in [0, 1]")
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise X6X7Error(f"{name} must be an integer >= {minimum}")
    return value


def _timestamp(value: Any, name: str) -> str:
    _text(value, name)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise X6X7Error(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise X6X7Error(f"{name} must include a timezone")
    return str(value)


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_keys(item)


def assert_training_policy_visible(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != ALLOWED_TRAINING_POLICY_KEYS:
        raise X6X7Error("training policy-visible record must contain only unaided_prompt and target_text")
    hits = sorted({key.lower() for key in _walk_keys(value)} & FORBIDDEN_POLICY_KEYS)
    if hits:
        raise X6X7Error(f"training policy-visible record contains evaluator fields: {hits}")


def _self_hash(value: Mapping[str, Any], field: str, name: str) -> None:
    identity = _hash(value.get(field), f"{name}.{field}")
    if sha_json(_without(value, field)) != identity:
        raise X6X7Error(f"{name} hash mismatch")


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    body = _without(protocol, "protocol_sha256")
    identity = body.get("identity")
    if isinstance(identity, Mapping):
        body["identity"] = _without(identity, "protocol_sha256")
    return sha_json(body)


def source_closure_sha256(base_revision: str, files: Sequence[Mapping[str, str]]) -> str:
    normalized = sorted(
        ({"path": str(item["path"]), "sha256": str(item["sha256"])} for item in files),
        key=lambda item: item["path"],
    )
    return sha_json({"base_revision": base_revision, "files": normalized})


def validate_protocol(
    protocol: Mapping[str, Any], repo_root: str | Path | None = None,
    check_source_closure: bool = False,
) -> dict[str, Any]:
    protocol = _require_mapping(protocol, "protocol")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise X6X7Error("X6/X7 protocol schema mismatch")
    if protocol.get("execution_authorized") is not False:
        raise X6X7Error("protocol cannot authorize execution")
    identity = _require_mapping(protocol.get("identity"), "protocol.identity")
    for key in ("base_revision", "source_paths", "source_closure_sha256", "protocol_sha256"):
        _hash(identity.get(key), f"identity.{key}") if key.endswith("sha256") else None
    if not isinstance(identity.get("base_revision"), str) or not HEX40.fullmatch(identity["base_revision"]):
        raise X6X7Error("identity.base_revision must be a full commit SHA")
    paths = identity.get("source_paths")
    if not isinstance(paths, list) or not paths or len(paths) != len(set(paths)):
        raise X6X7Error("identity.source_paths must be a non-empty unique list")
    for path in paths:
        _relative_path(path, "identity.source_paths item")
    expected = protocol_sha256(protocol)
    if identity.get("protocol_sha256") != expected:
        raise X6X7Error("protocol identity hash mismatch")
    authority = _require_mapping(protocol.get("canonical_authority"), "canonical_authority")
    if authority.get("retention_floor_absolute_accuracy") != 0.10:
        raise X6X7Error("retention floor differs from current canonical X6 authority")
    if authority.get("rehearsal_min_fraction") != 0.50:
        raise X6X7Error("rehearsal minimum differs from current canonical X6 authority")
    thresholds = _require_mapping(protocol.get("decision_thresholds"), "decision_thresholds")
    threshold_status = thresholds.get("authority_status")
    if threshold_status not in {"PROPOSED_PENDING_REVIEW", "AUTHORIZED"}:
        raise X6X7Error("decision threshold authority status is invalid")
    if threshold_status == "AUTHORIZED":
        probability_keys = (
            "min_unaided_gain_vs_parent", "min_unaided_gain_vs_control",
            "min_lift_reduction_vs_parent", "min_lift_reduction_vs_control",
            "x7_min_unaided_retention_vs_parent", "x7_min_unaided_gain_vs_control",
            "x7_min_target_lift_reduction_vs_parent", "x7_min_target_lift_reduction_vs_control",
            "x7_max_unrelated_lift_change",
        )
        for key in probability_keys:
            _probability(thresholds.get(key), f"decision_thresholds.{key}")
        for key in ("minimum_independent_worlds", "minimum_paired_seeds", "bootstrap_resamples"):
            _integer(thresholds.get(key), f"decision_thresholds.{key}", 2 if key != "bootstrap_resamples" else 1)
        if thresholds.get("require_ci_above_zero") is not True:
            raise X6X7Error("authorized protocol must require a positive paired interval")
        for threshold_name in ("min_transfer_gain_by_axis", "x7_min_transfer_gain_by_axis"):
            transfer = _require_mapping(thresholds.get(threshold_name), threshold_name)
            if set(transfer) != set(TRANSFER_AXES):
                raise X6X7Error("authorized transfer thresholds do not cover every axis")
            for axis, value in transfer.items():
                _probability(value, f"{threshold_name}.{axis}")
    result: dict[str, Any] = {"valid": True, "protocol_sha256": expected, "errors": []}
    if check_source_closure:
        root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
        observed: list[dict[str, str]] = []
        errors: list[str] = []
        for relative in paths:
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                errors.append(f"source path escapes repository: {relative}")
                continue
            path = (root / candidate).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                errors.append(f"source path escapes repository: {relative}")
                continue
            if not path.is_file():
                errors.append(f"source file missing: {relative}")
                continue
            observed.append({"path": candidate.as_posix(), "sha256": sha_file(path)})
        calculated = source_closure_sha256(identity["base_revision"], observed)
        if calculated != identity["source_closure_sha256"]:
            errors.append("source closure hash mismatch")
        result = {"valid": not errors, "protocol_sha256": expected, "errors": errors, "files": observed}
    return result


def audit_x1_entry(
    receipt: Mapping[str, Any], receipt_file_sha256: str,
    expected_subject_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    receipt = _require_mapping(receipt, "X1 receipt")
    _hash(receipt_file_sha256, "x1_receipt_file_sha256")
    if receipt.get("schema") != "anra-x1-real-1-analysis/v1":
        errors.append("X1 receipt schema is not anra-x1-real-1-analysis/v1")
    if receipt.get("phase") != "ANALYSIS_ACCEPTED":
        errors.append("X1 primary analysis phase is not ANALYSIS_ACCEPTED")
    try:
        _self_hash(receipt, "analysis_receipt_sha256", "X1 analysis")
    except X6X7Error as exc:
        errors.append(str(exc))
    decision = receipt.get("decision")
    if not isinstance(decision, Mapping):
        errors.append("X1 decision is missing")
        decision = {}
    if decision.get("status") != "SUPPORTED_SCOPED" or decision.get("supported") is not True:
        errors.append("X1 decision is not SUPPORTED_SCOPED with supported=true")
    for field in ("primary_gate_pass", "prediction_gate_pass", "decision_gate_pass"):
        if decision.get(field) is not True:
            errors.append(f"X1 decision field is not true: {field}")
    for verdict_name in ("prediction_verdict", "decision_verdict"):
        verdict = receipt.get(verdict_name)
        if not isinstance(verdict, Mapping) or verdict.get("status") != "PASS":
            errors.append(f"X1 {verdict_name} is not PASS")
        elif isinstance(verdict.get("checks"), Mapping) and not all(value is True for value in verdict["checks"].values()):
            errors.append(f"X1 {verdict_name} contains a failed check")
    for field in (
        "protocol_sha256", "subject_manifest_sha256", "release_manifest_sha256",
        "basis_qualification_sha256", "basis_split_manifest_sha256",
        "development_split_manifest_sha256", "prediction_receipt_sha256",
        "prediction_phase_commit_sha256", "reveal_receipt_sha256",
        "reveal_phase_commit_sha256",
    ):
        try:
            _hash(receipt.get(field), f"X1 {field}")
        except X6X7Error as exc:
            errors.append(str(exc))
    if expected_subject_manifest_sha256 is not None and receipt.get("subject_manifest_sha256") != expected_subject_manifest_sha256:
        errors.append("X1 subject manifest does not match the proposed X6 parent")
    replication = receipt.get("replication")
    if not isinstance(replication, Mapping) or replication.get("valid") is not True:
        errors.append("X1 independent replication is not valid")
    transfer = receipt.get("transfer_verdict")
    allowed_transfer = {"SUPPORTED_SCOPED", "SUPPORTED_SCOPED_CHECKPOINT_PENDING"}
    if not isinstance(transfer, Mapping) or transfer.get("status") not in allowed_transfer:
        errors.append("X1 transfer verdict is unresolved")
    if isinstance(transfer, Mapping) and transfer.get("independent_task_cohort") is not True:
        errors.append("X1 independent-task replication is not confirmed")
    body = {
        "schema": ENTRY_AUDIT_SCHEMA,
        "status": "PASS" if not errors else "BLOCKED",
        "x1_receipt_schema": receipt.get("schema"),
        "x1_analysis_receipt_sha256": receipt.get("analysis_receipt_sha256"),
        "x1_receipt_file_sha256": receipt_file_sha256,
        "x1_decision_status": decision.get("status"),
        "x1_supported": decision.get("supported") is True,
        "subject_manifest_sha256": receipt.get("subject_manifest_sha256"),
        "checkpoint_replication_status": (
            "PENDING" if isinstance(transfer, Mapping) and transfer.get("status") == "SUPPORTED_SCOPED_CHECKPOINT_PENDING"
            else "SATISFIED" if isinstance(transfer, Mapping) and transfer.get("status") == "SUPPORTED_SCOPED"
            else "UNRESOLVED"
        ),
        "verdict": "X6_ENTRY_AUTHORIZED" if not errors else "X6_ENTRY_BLOCKED",
        "errors": errors,
        "next_action": "Bind the accepted X1 receipt to an eligible parent and freeze the X6 run manifest." if not errors else "Do not train or mark X6 ready; satisfy every listed X1 evidence error first.",
    }
    body["entry_audit_sha256"] = sha_json(body)
    return body


def validate_entry_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    audit = _require_mapping(audit, "entry audit")
    if audit.get("schema") != ENTRY_AUDIT_SCHEMA:
        raise X6X7Error("entry audit schema mismatch")
    _self_hash(audit, "entry_audit_sha256", "entry audit")
    if audit.get("status") not in {"PASS", "BLOCKED"}:
        raise X6X7Error("entry audit status is invalid")
    if (audit.get("status") == "PASS") != (audit.get("verdict") == "X6_ENTRY_AUTHORIZED"):
        raise X6X7Error("entry audit status/verdict mismatch")
    if audit.get("status") == "PASS" and audit.get("errors"):
        raise X6X7Error("passing entry audit contains errors")
    return {"valid": True, "authorized": audit.get("status") == "PASS", "errors": list(audit.get("errors", []))}


PARENT_FIELDS = {
    "schema", "subject_manifest_sha256", "checkpoint_path", "checkpoint_file_sha256",
    "parameter_sha256", "parent_state_tree_sha256", "model_config_sha256",
    "tokenizer_artifact_sha256", "tokenizer_identity_sha256", "runtime_source_revision",
    "runtime_source_sha256", "source_commit", "training_lineage", "stage", "global_step",
    "research_subject", "qualification_state", "readiness_receipt_sha256",
    "qualification_receipt_sha256", "identity_attestation_sha256",
    "checkpoint_file_verified", "continuation_code", "continuation_contract_sha256",
    "parent_identity_sha256",
}


def validate_parent_identity(parent: Mapping[str, Any]) -> dict[str, Any]:
    parent = _require_mapping(parent, "parent identity")
    _require_keys(parent, PARENT_FIELDS, "parent identity")
    if parent.get("schema") != PARENT_SCHEMA or parent.get("research_subject") is not True:
        raise X6X7Error("parent is not an eligible research subject")
    if parent.get("qualification_state") not in {"READY_SCOPED", "READY", "QUALIFIED"}:
        raise X6X7Error("parent qualification state is ineligible")
    if parent.get("checkpoint_file_verified") is not True:
        raise X6X7Error("parent checkpoint file was not verified")
    for key in sorted(PARENT_FIELDS - {
        "schema", "checkpoint_path", "runtime_source_revision", "source_commit",
        "training_lineage", "stage", "global_step", "research_subject",
        "qualification_state", "checkpoint_file_verified", "continuation_code",
        "parent_identity_sha256",
    }):
        _hash(parent.get(key), f"parent.{key}")
    if not isinstance(parent.get("source_commit"), str) or not HEX40.fullmatch(parent["source_commit"]):
        raise X6X7Error("parent.source_commit must be a full commit SHA")
    _text(parent.get("checkpoint_path"), "parent.checkpoint_path")
    _text(parent.get("runtime_source_revision"), "parent.runtime_source_revision")
    _text(parent.get("training_lineage"), "parent.training_lineage")
    _text(parent.get("stage"), "parent.stage")
    _id(parent.get("continuation_code"), "parent.continuation_code")
    _integer(parent.get("global_step"), "parent.global_step")
    _self_hash(parent, "parent_identity_sha256", "parent identity")
    return {"valid": True, "parent_identity_sha256": parent["parent_identity_sha256"]}


def make_parent_identity(body: Mapping[str, Any]) -> dict[str, Any]:
    parent = {"schema": PARENT_SCHEMA, **dict(body)}
    parent["parent_identity_sha256"] = sha_json(parent)
    validate_parent_identity(parent)
    return parent


SPLIT_FIELDS = {
    "schema", "split_id", "role", "protocol_sha256", "generator_id",
    "generator_sha256", "seed", "clusters", "freeze", "manifest_sha256",
}
CLUSTER_FIELDS = {
    "cluster_id", "latent_world_id", "causal_structure_id", "source_id",
    "task_ids", "task_content_sha256s", "transfer_axes",
}
FREEZE_FIELDS = {"status", "frozen_at", "frozen_before_training", "inspection_count", "retired_if_inspected"}


def validate_split_manifest(manifest: Mapping[str, Any], expected_role: str | None = None) -> dict[str, Any]:
    manifest = _require_mapping(manifest, "split manifest")
    _require_keys(manifest, SPLIT_FIELDS, "split manifest")
    if manifest.get("schema") != SPLIT_SCHEMA or manifest.get("role") not in SPLIT_ROLES:
        raise X6X7Error("split schema or role is invalid")
    if expected_role is not None and manifest.get("role") != expected_role:
        raise X6X7Error("split role mismatch")
    _id(manifest.get("split_id"), "split.split_id")
    _id(manifest.get("generator_id"), "split.generator_id")
    _hash(manifest.get("protocol_sha256"), "split.protocol_sha256")
    _hash(manifest.get("generator_sha256"), "split.generator_sha256")
    _integer(manifest.get("seed"), "split.seed")
    freeze = _require_mapping(manifest.get("freeze"), "split.freeze")
    _require_keys(freeze, FREEZE_FIELDS, "split.freeze")
    if freeze.get("status") != "FROZEN_SEALED" or freeze.get("frozen_before_training") is not True:
        raise X6X7Error("split is not frozen and sealed before training")
    if freeze.get("inspection_count") != 0 or freeze.get("retired_if_inspected") is not True:
        raise X6X7Error("split retirement-on-inspection contract is invalid")
    _timestamp(freeze.get("frozen_at"), "split.freeze.frozen_at")
    clusters = manifest.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise X6X7Error("split must contain clusters")
    cluster_ids: set[str] = set()
    world_ids: set[str] = set()
    structure_ids: set[str] = set()
    source_ids: set[str] = set()
    task_ids: set[str] = set()
    content_hashes: set[str] = set()
    for cluster in clusters:
        cluster = _require_mapping(cluster, "split cluster")
        _require_keys(cluster, CLUSTER_FIELDS, "split cluster")
        for key in ("cluster_id", "latent_world_id", "causal_structure_id", "source_id"):
            _id(cluster.get(key), f"cluster.{key}")
        for value, collection, name in (
            (cluster["cluster_id"], cluster_ids, "cluster"),
            (cluster["latent_world_id"], world_ids, "latent world"),
            (cluster["causal_structure_id"], structure_ids, "causal structure"),
            (cluster["source_id"], source_ids, "source"),
        ):
            if value in collection:
                raise X6X7Error(f"duplicate {name} identity within split: {value}")
            collection.add(value)
        rows = cluster.get("task_ids")
        hashes = cluster.get("task_content_sha256s")
        axes = cluster.get("transfer_axes")
        if not isinstance(rows, list) or not rows or len(rows) != len(set(rows)):
            raise X6X7Error("cluster task_ids must be non-empty and unique")
        if not isinstance(hashes, dict) or set(hashes) != set(rows):
            raise X6X7Error("cluster content hashes must cover task_ids exactly")
        if not isinstance(axes, list) or not axes or any(axis not in TRANSFER_AXES for axis in axes):
            raise X6X7Error("cluster transfer axes are invalid")
        for task_id in rows:
            _id(task_id, "task_id")
            if task_id in task_ids:
                raise X6X7Error(f"duplicate task identity within split: {task_id}")
            task_ids.add(task_id)
            _hash(hashes[task_id], f"task content {task_id}")
            if hashes[task_id] in content_hashes:
                raise X6X7Error("duplicate task content within split")
            content_hashes.add(hashes[task_id])
    _self_hash(manifest, "manifest_sha256", "split manifest")
    return {
        "valid": True,
        "split_id": manifest["split_id"],
        "role": manifest["role"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cluster_ids": cluster_ids,
        "latent_world_ids": world_ids,
        "causal_structure_ids": structure_ids,
        "source_ids": source_ids,
        "task_ids": task_ids,
        "content_hashes": content_hashes,
    }


def make_split_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    manifest = {"schema": SPLIT_SCHEMA, **dict(body)}
    manifest["manifest_sha256"] = sha_json(manifest)
    validate_split_manifest(manifest)
    return manifest


def validate_split_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    bundle = _require_mapping(bundle, "split bundle")
    _require_keys(bundle, {"schema", "protocol_sha256", "splits", "bundle_sha256"}, "split bundle")
    if bundle.get("schema") != SPLIT_BUNDLE_SCHEMA:
        raise X6X7Error("split bundle schema mismatch")
    _hash(bundle.get("protocol_sha256"), "split bundle.protocol_sha256")
    splits = _require_mapping(bundle.get("splits"), "split bundle.splits")
    if set(splits) != set(SPLIT_ROLES):
        raise X6X7Error(f"split bundle roles must be exactly {sorted(SPLIT_ROLES)}")
    verdicts = {role: validate_split_manifest(splits[role], role) for role in SPLIT_ROLES}
    identity_fields = ("cluster_ids", "latent_world_ids", "causal_structure_ids", "source_ids", "task_ids", "content_hashes")
    for field in identity_fields:
        seen: dict[str, str] = {}
        for role in SPLIT_ROLES:
            for value in verdicts[role][field]:
                if value in seen:
                    raise X6X7Error(f"{field} collision between {seen[value]} and {role}: {value}")
                seen[value] = role
    confirmation_axes = {
        axis
        for cluster in splits["X6_CONFIRMATION"]["clusters"]
        for axis in cluster["transfer_axes"]
    }
    missing_axes = sorted(set(TRANSFER_AXES) - confirmation_axes)
    if missing_axes:
        raise X6X7Error(f"X6 confirmation transfer axes missing: {missing_axes}")
    _self_hash(bundle, "bundle_sha256", "split bundle")
    return {
        "valid": True,
        "bundle_sha256": bundle["bundle_sha256"],
        "split_manifest_sha256": {role: verdicts[role]["manifest_sha256"] for role in SPLIT_ROLES},
    }


def make_split_bundle(protocol_sha: str, splits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    body = {
        "schema": SPLIT_BUNDLE_SCHEMA,
        "protocol_sha256": protocol_sha,
        "splits": {role: dict(splits[role]) for role in SPLIT_ROLES},
    }
    body["bundle_sha256"] = sha_json(body)
    validate_split_bundle(body)
    return body


def _intervention_map(interventions: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in interventions:
        item = _require_mapping(raw, "intervention")
        intervention_id = _id(item.get("id"), "intervention.id")
        if intervention_id in result:
            raise X6X7Error(f"duplicate intervention: {intervention_id}")
        if item.get("role") not in {"NULL_CONTROL", "DIAGNOSTIC", "REPAIR", "ASSISTANCE"}:
            raise X6X7Error(f"invalid intervention role: {intervention_id}")
        if item.get("information_preserving") is not True or item.get("answer_revealing") is not False:
            raise X6X7Error(f"intervention is not legal and answer-blind: {intervention_id}")
        if item.get("assistance") == "A4":
            raise X6X7Error(f"A4 oracle cannot supply training experience: {intervention_id}")
        _integer(item.get("cost"), f"intervention {intervention_id}.cost")
        result[intervention_id] = dict(item)
    if "NO_CHANGE" not in result or result["NO_CHANGE"].get("role") != "NULL_CONTROL":
        raise X6X7Error("NO_CHANGE null control is required")
    return result


EXPERIENCE_FIELDS = {
    "schema", "experience_id", "experience_class", "source_task_id", "cluster_id",
    "latent_world_id", "causal_structure_id", "source_split_id", "intervention_id",
    "intervention_version", "intervention_registry_sha256", "legality_receipt_sha256",
    "intervention_information_preserving", "intervention_answer_revealing",
    "baseline_failed", "baseline_raw_output_sha256", "assisted_prompt_sha256",
    "assisted_raw_output_sha256", "assisted_verified_correct", "verifier_receipt_sha256",
    "verifier_source_sha256", "verification_independent", "target_text", "target_sha256",
    "unaided_prompt", "unaided_prompt_sha256", "support_removal_transformation_id",
    "support_removal_receipt_sha256", "external_support_removed",
    "unaided_token_count", "target_token_count",
    "sealed_outcome", "evaluation_label", "experience_sha256",
}
EXPERIENCE_CLASSES = {"REPAIR_SUCCESS", "RAW_FAILURE", "REPLAY"}


def _validate_experience(
    row: Mapping[str, Any], split: Mapping[str, Any], interventions: Mapping[str, Mapping[str, Any]],
    registry_sha256: str,
) -> dict[str, Any]:
    row = _require_mapping(row, "experience row")
    _require_keys(row, EXPERIENCE_FIELDS, "experience row")
    if row.get("schema") != EXPERIENCE_SCHEMA or row.get("experience_class") not in EXPERIENCE_CLASSES:
        raise X6X7Error("experience schema or class is invalid")
    _id(row.get("experience_id"), "experience_id")
    for field in (
        "intervention_registry_sha256", "legality_receipt_sha256", "baseline_raw_output_sha256",
        "assisted_prompt_sha256", "assisted_raw_output_sha256", "verifier_receipt_sha256",
        "verifier_source_sha256", "target_sha256", "unaided_prompt_sha256",
        "support_removal_receipt_sha256", "experience_sha256",
    ):
        _hash(row.get(field), f"experience.{field}")
    if row["intervention_registry_sha256"] != registry_sha256:
        raise X6X7Error("experience intervention registry hash mismatch")
    intervention = interventions.get(row.get("intervention_id"))
    if intervention is None:
        raise X6X7Error("experience references an unknown intervention")
    if row.get("intervention_version") != intervention.get("version"):
        raise X6X7Error("experience intervention version mismatch")
    if row.get("intervention_information_preserving") is not True or row.get("intervention_answer_revealing") is not False:
        raise X6X7Error("experience intervention legality flags are invalid")
    if row.get("verification_independent") is not True:
        raise X6X7Error("experience target lacks independent verification")
    if row.get("sealed_outcome") is not False or row.get("evaluation_label") is not False:
        raise X6X7Error("sealed outcomes and evaluation labels are forbidden in training")
    _text(row.get("unaided_prompt"), "experience.unaided_prompt")
    _text(row.get("target_text"), "experience.target_text")
    if hashlib.sha256(row["unaided_prompt"].encode("utf-8")).hexdigest() != row["unaided_prompt_sha256"]:
        raise X6X7Error("unaided prompt hash mismatch")
    if hashlib.sha256(row["target_text"].encode("utf-8")).hexdigest() != row["target_sha256"]:
        raise X6X7Error("training target hash mismatch")
    _integer(row.get("unaided_token_count"), "unaided_token_count", 1)
    _integer(row.get("target_token_count"), "target_token_count", 1)
    _id(row.get("support_removal_transformation_id"), "support_removal_transformation_id")
    expected_role = "X6_REPLAY" if row["experience_class"] == "REPLAY" else "X6_EXPERIENCE"
    if row.get("source_split_id") != split.get("split_id") or split.get("role") != expected_role:
        raise X6X7Error("experience source split does not match its class")
    cluster = next(
        (item for item in split["clusters"] if item["cluster_id"] == row.get("cluster_id")),
        None,
    )
    if cluster is None or row.get("source_task_id") not in cluster["task_ids"]:
        raise X6X7Error("experience task is not covered by its split cluster")
    if row.get("latent_world_id") != cluster["latent_world_id"] or row.get("causal_structure_id") != cluster["causal_structure_id"]:
        raise X6X7Error("experience world or causal structure identity mismatch")
    if row["experience_class"] == "REPAIR_SUCCESS":
        if row.get("baseline_failed") is not True or row.get("assisted_verified_correct") is not True:
            raise X6X7Error("repair success must be a verified baseline failure repaired by assistance")
        if row.get("intervention_id") == "NO_CHANGE" or row.get("external_support_removed") is not True:
            raise X6X7Error("repair success lacks an active intervention or support removal")
    elif row["experience_class"] == "RAW_FAILURE":
        if row.get("baseline_failed") is not True or row.get("assisted_verified_correct") is not False:
            raise X6X7Error("raw-failure control must be a baseline failure not repaired in collection")
    elif row.get("intervention_id") != "NO_CHANGE" or row.get("baseline_failed") is not True:
        raise X6X7Error("replay exposure must be unaided baseline-failure supervision")
    _self_hash(row, "experience_sha256", "experience row")
    return dict(row)


def build_training_record(
    row: Mapping[str, Any], arm_id: str, protocol_sha256: str,
) -> dict[str, Any]:
    if arm_id not in TRAINING_ARMS:
        raise X6X7Error(f"unknown training arm: {arm_id}")
    policy_visible = {"unaided_prompt": row["unaided_prompt"], "target_text": row["target_text"]}
    assert_training_policy_visible(policy_visible)
    record = {
        "schema": TRAINING_RECORD_SCHEMA,
        "record_id": f"{arm_id}:{row['experience_id']}",
        "arm_id": arm_id,
        "experience_class": row["experience_class"],
        "protocol_sha256": protocol_sha256,
        "source_task_id": row["source_task_id"],
        "source_cluster_id": row["cluster_id"],
        "latent_world_id": row["latent_world_id"],
        "causal_structure_id": row["causal_structure_id"],
        "source_split_id": row["source_split_id"],
        "intervention_id": row["intervention_id"],
        "intervention_version": row["intervention_version"],
        "intervention_registry_sha256": row["intervention_registry_sha256"],
        "legality_receipt_sha256": row["legality_receipt_sha256"],
        "baseline_raw_output_sha256": row["baseline_raw_output_sha256"],
        "assisted_prompt_sha256": row["assisted_prompt_sha256"],
        "assisted_raw_output_sha256": row["assisted_raw_output_sha256"],
        "verifier_receipt_sha256": row["verifier_receipt_sha256"],
        "verifier_source_sha256": row["verifier_source_sha256"],
        "support_removal_transformation_id": row["support_removal_transformation_id"],
        "support_removal_receipt_sha256": row["support_removal_receipt_sha256"],
        "external_support_removed": row["external_support_removed"],
        "unaided_token_count": row["unaided_token_count"],
        "target_token_count": row["target_token_count"],
        "policy_visible": policy_visible,
    }
    record["record_sha256"] = sha_json(record)
    return record


def build_dataset_manifest(
    protocol_sha256: str, arm_id: str, records: Sequence[Mapping[str, Any]],
    split_bundle: Mapping[str, Any], replay_fraction_minimum: float,
) -> dict[str, Any]:
    validate_split_bundle(split_bundle)
    if split_bundle["protocol_sha256"] != protocol_sha256 or arm_id not in TRAINING_ARMS:
        raise X6X7Error("dataset protocol or arm mismatch")
    normalized = sorted((dict(record) for record in records), key=lambda item: item["record_id"])
    if not normalized:
        raise X6X7Error("training dataset cannot be empty")
    ids = [record["record_id"] for record in normalized]
    task_ids = [record["source_task_id"] for record in normalized]
    if len(ids) != len(set(ids)) or len(task_ids) != len(set(task_ids)):
        raise X6X7Error("training dataset contains duplicate record or source task IDs")
    replay = [record for record in normalized if record["experience_class"] == "REPLAY"]
    expected_classes = {"REPAIR_SUCCESS"} if arm_id == "REPAIR_INTERNALIZATION" else {"RAW_FAILURE"}
    non_replay_classes = {record["experience_class"] for record in normalized if record["experience_class"] != "REPLAY"}
    if non_replay_classes != expected_classes:
        raise X6X7Error("training arm contains the wrong repair-specific experience class")
    replay_fraction = len(replay) / len(normalized)
    if replay_fraction < replay_fraction_minimum:
        raise X6X7Error("replay fraction is below the frozen canonical minimum")
    training_splits = {
        split_bundle["splits"]["X6_EXPERIENCE"]["split_id"]: split_bundle["splits"]["X6_EXPERIENCE"],
        split_bundle["splits"]["X6_REPLAY"]["split_id"]: split_bundle["splits"]["X6_REPLAY"],
    }
    for record in normalized:
        _hash(record.get("record_sha256"), "training record hash")
        if record.get("protocol_sha256") != protocol_sha256 or record.get("arm_id") != arm_id:
            raise X6X7Error("training record protocol or arm mismatch")
        split = training_splits.get(record.get("source_split_id"))
        if split is None:
            raise X6X7Error("training record references a non-training split")
        cluster = next(
            (item for item in split["clusters"] if item["cluster_id"] == record.get("source_cluster_id")),
            None,
        )
        if cluster is None or record.get("source_task_id") not in cluster["task_ids"]:
            raise X6X7Error("training record source identity is not covered by its split")
        if record.get("latent_world_id") != cluster["latent_world_id"] or record.get("causal_structure_id") != cluster["causal_structure_id"]:
            raise X6X7Error("training record world or causal structure identity mismatch")
        assert_training_policy_visible(record.get("policy_visible"))
    manifest = {
        "schema": DATASET_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "arm_id": arm_id,
        "split_bundle_sha256": split_bundle["bundle_sha256"],
        "record_count": len(normalized),
        "target_token_count": sum(record["target_token_count"] for record in normalized),
        "unaided_token_count": sum(record["unaided_token_count"] for record in normalized),
        "replay_count": len(replay),
        "replay_fraction": round(replay_fraction, 12),
        "replay_unaided_token_count": sum(record["unaided_token_count"] for record in replay),
        "replay_target_token_count": sum(record["target_token_count"] for record in replay),
        "repair_specific_count": len(normalized) - len(replay),
        "intervention_ids": sorted({record["intervention_id"] for record in normalized}),
        "record_sha256s": [record["record_sha256"] for record in normalized],
        "world_ids": sorted({record["latent_world_id"] for record in normalized}),
        "causal_structure_ids": sorted({record["causal_structure_id"] for record in normalized}),
        "evaluation_artifacts_used": False,
        "sealed_outcomes_used": False,
        "evaluator_answer_keys_used": False,
        "external_support_removed_from_all_repair_records": all(
            record["external_support_removed"] is True
            for record in normalized if record["experience_class"] == "REPAIR_SUCCESS"
        ),
    }
    manifest["dataset_manifest_sha256"] = sha_json(manifest)
    validate_dataset_manifest(manifest)
    return manifest


def validate_dataset_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _require_mapping(manifest, "dataset manifest")
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("arm_id") not in TRAINING_ARMS:
        raise X6X7Error("dataset manifest schema or arm is invalid")
    _hash(manifest.get("protocol_sha256"), "dataset.protocol_sha256")
    _hash(manifest.get("split_bundle_sha256"), "dataset.split_bundle_sha256")
    _self_hash(manifest, "dataset_manifest_sha256", "dataset manifest")
    record_count = _integer(manifest.get("record_count"), "dataset.record_count", 1)
    if _integer(manifest.get("replay_count"), "dataset.replay_count", 1) >= record_count:
        raise X6X7Error("dataset replay count is invalid")
    if manifest.get("replay_count") + manifest.get("repair_specific_count") != record_count:
        raise X6X7Error("dataset class counts do not sum to record count")
    for key in ("unaided_token_count", "target_token_count", "replay_unaided_token_count", "replay_target_token_count"):
        _integer(manifest.get(key), f"dataset.{key}")
    if _probability(manifest.get("replay_fraction"), "dataset.replay_fraction") < 0.5:
        raise X6X7Error("dataset replay fraction is below canonical minimum")
    hashes = manifest.get("record_sha256s")
    if not isinstance(hashes, list) or len(hashes) != record_count or len(hashes) != len(set(hashes)):
        raise X6X7Error("dataset record hash inventory is incomplete")
    for value in hashes:
        _hash(value, "dataset record hash")
    for key in ("evaluation_artifacts_used", "sealed_outcomes_used", "evaluator_answer_keys_used"):
        if manifest.get(key) is not False:
            raise X6X7Error(f"dataset firewall violation: {key}")
    if manifest.get("external_support_removed_from_all_repair_records") is not True:
        raise X6X7Error("dataset contains a repair record without support removal")
    return {"valid": True, "dataset_manifest_sha256": manifest["dataset_manifest_sha256"]}


def build_dataset(
    protocol_sha256: str, arm_id: str, experience_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]], split_bundle: Mapping[str, Any],
    interventions: Sequence[Mapping[str, Any]], intervention_registry_sha256: str,
    replay_fraction_minimum: float = 0.5,
) -> dict[str, Any]:
    intervention_map = _intervention_map(interventions)
    bundle_verdict = validate_split_bundle(split_bundle)
    if not bundle_verdict["valid"] or split_bundle["protocol_sha256"] != protocol_sha256:
        raise X6X7Error("split bundle is invalid or protocol-mismatched")
    experience_split = split_bundle["splits"]["X6_EXPERIENCE"]
    replay_split = split_bundle["splits"]["X6_REPLAY"]
    selected_class = "REPAIR_SUCCESS" if arm_id == "REPAIR_INTERNALIZATION" else "RAW_FAILURE"
    validated_experience = [
        _validate_experience(row, experience_split, intervention_map, intervention_registry_sha256)
        for row in experience_rows
    ]
    validated_replay = [
        _validate_experience(row, replay_split, intervention_map, intervention_registry_sha256)
        for row in replay_rows
    ]
    if any(row["experience_class"] not in {"REPAIR_SUCCESS", "RAW_FAILURE"} for row in validated_experience):
        raise X6X7Error("experience collection contains an invalid class for X6 experience data")
    if any(row["experience_class"] != "REPLAY" for row in validated_replay):
        raise X6X7Error("replay collection contains a non-replay record")
    selected_experience = [row for row in validated_experience if row["experience_class"] == selected_class]
    selected_replay = validated_replay
    if not selected_experience or not selected_replay:
        raise X6X7Error("both repair-specific and replay experience are required")
    records = [
        build_training_record(row, arm_id, protocol_sha256)
        for row in [*selected_experience, *selected_replay]
    ]
    manifest = build_dataset_manifest(
        protocol_sha256, arm_id, records, split_bundle, replay_fraction_minimum,
    )
    return {"manifest": manifest, "records": records}


EXPOSURE_FIELDS = {
    "optimizer_updates", "training_examples", "unaided_tokens", "target_tokens",
    "replay_examples", "replay_target_tokens", "replay_fraction", "optimizer_sha256",
    "schedule_sha256", "checkpoint_cadence_sha256", "evaluation_timing_sha256",
    "data_opportunity_sha256",
}
NONZERO_EXPOSURE_FIELDS = {
    "optimizer_updates", "training_examples", "unaided_tokens", "target_tokens",
    "replay_examples", "replay_target_tokens",
}


def _parent_exposure() -> dict[str, Any]:
    return {
        **{key: 0 for key in NONZERO_EXPOSURE_FIELDS},
        "replay_fraction": 0.0,
        **{key: None for key in EXPOSURE_FIELDS - NONZERO_EXPOSURE_FIELDS - {"replay_fraction"}},
    }


def validate_arm_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _require_mapping(manifest, "arm manifest")
    _require_keys(
        manifest,
        {"schema", "protocol_sha256", "seed", "parent_identity_sha256", "parent_state_tree_sha256", "arms", "arm_manifest_sha256"},
        "arm manifest",
    )
    if manifest.get("schema") != ARM_SCHEMA:
        raise X6X7Error("arm manifest schema mismatch")
    _hash(manifest.get("protocol_sha256"), "arm.protocol_sha256")
    _hash(manifest.get("parent_identity_sha256"), "arm.parent_identity_sha256")
    _hash(manifest.get("parent_state_tree_sha256"), "arm.parent_state_tree_sha256")
    _integer(manifest.get("seed"), "arm.seed")
    arms = manifest.get("arms")
    if not isinstance(arms, list) or {arm.get("arm_id") for arm in arms} != set(ARM_IDS) or len(arms) != len(ARM_IDS):
        raise X6X7Error("arm manifest must contain exactly the canonical arms")
    parent_rows = []
    trained_rows = []
    for arm in arms:
        _require_keys(
            arm,
            {"arm_id", "role", "parent_identity_sha256", "parent_state_tree_sha256", "dataset_manifest_sha256", "exposure"},
            "arm",
        )
        if arm["parent_identity_sha256"] != manifest["parent_identity_sha256"] or arm["parent_state_tree_sha256"] != manifest["parent_state_tree_sha256"]:
            raise X6X7Error("arm parent state is not byte-identical")
        if arm["arm_id"] == "PARENT":
            parent_rows.append(arm)
            if arm.get("dataset_manifest_sha256") is not None or arm.get("exposure") != _parent_exposure():
                raise X6X7Error("parent arm must have zero exposure and no dataset")
        else:
            trained_rows.append(arm)
            _hash(arm.get("dataset_manifest_sha256"), "arm.dataset_manifest_sha256")
            exposure = _require_mapping(arm.get("exposure"), "arm.exposure")
            _require_keys(exposure, EXPOSURE_FIELDS, "arm.exposure")
            for key in (
                "optimizer_updates", "training_examples", "unaided_tokens", "target_tokens",
                "replay_examples", "replay_target_tokens",
            ):
                _integer(exposure.get(key), f"exposure.{key}")
            _probability(exposure.get("replay_fraction"), "exposure.replay_fraction")
            for key in (
                "optimizer_sha256", "schedule_sha256", "checkpoint_cadence_sha256",
                "evaluation_timing_sha256", "data_opportunity_sha256",
            ):
                _hash(exposure.get(key), f"exposure.{key}")
    if len(trained_rows) != 2:
        raise X6X7Error("both trained arms are required")
    reference = trained_rows[0]["exposure"]
    for key in EXPOSURE_FIELDS:
        if trained_rows[1]["exposure"][key] != reference[key]:
            raise X6X7Error(f"trained arms are not exposure-matched: {key}")
    _self_hash(manifest, "arm_manifest_sha256", "arm manifest")
    return {"valid": True, "arm_manifest_sha256": manifest["arm_manifest_sha256"]}


def make_arm_manifest(
    protocol_sha256: str, seed: int, parent: Mapping[str, Any],
    dataset_manifests: Mapping[str, Mapping[str, Any]], exposure: Mapping[str, Any],
) -> dict[str, Any]:
    _require_keys(exposure, EXPOSURE_FIELDS, "exposure")
    for arm_id in TRAINING_ARMS:
        manifest = validate_dataset_manifest(dataset_manifests.get(arm_id, {}))
        if manifest["dataset_manifest_sha256"] != dataset_manifests[arm_id]["dataset_manifest_sha256"]:
            raise X6X7Error("dataset identity changed during validation")
        if dataset_manifests[arm_id].get("protocol_sha256") != protocol_sha256:
            raise X6X7Error("dataset protocol mismatch")
        expected = {
            "training_examples": dataset_manifests[arm_id]["record_count"],
            "unaided_tokens": dataset_manifests[arm_id]["unaided_token_count"],
            "target_tokens": dataset_manifests[arm_id]["target_token_count"],
            "replay_examples": dataset_manifests[arm_id]["replay_count"],
            "replay_target_tokens": dataset_manifests[arm_id]["replay_target_token_count"],
            "replay_fraction": dataset_manifests[arm_id]["replay_fraction"],
        }
        if any(exposure[key] != value for key, value in expected.items()):
            raise X6X7Error(f"arm exposure does not match dataset manifest: {arm_id}")
    arms = [{
        "arm_id": "PARENT",
        "role": "UNCHANGED_PARENT",
        "parent_identity_sha256": parent["parent_identity_sha256"],
        "parent_state_tree_sha256": parent["parent_state_tree_sha256"],
        "dataset_manifest_sha256": None,
        "exposure": _parent_exposure(),
    }]
    roles = {
        "REPAIR_INTERNALIZATION": "VERIFIED_REPAIR_EXTERNAL_CONTEXT_REMOVED",
        "RAW_FAILURE_SFT_CONTROL": "DOSE_MATCHED_ORDINARY_SUPERVISED_REPLAY",
    }
    for arm_id in TRAINING_ARMS:
        if arm_id not in dataset_manifests or dataset_manifests[arm_id].get("arm_id") != arm_id:
            raise X6X7Error(f"dataset manifest missing for {arm_id}")
        arms.append({
            "arm_id": arm_id,
            "role": roles[arm_id],
            "parent_identity_sha256": parent["parent_identity_sha256"],
            "parent_state_tree_sha256": parent["parent_state_tree_sha256"],
            "dataset_manifest_sha256": dataset_manifests[arm_id]["dataset_manifest_sha256"],
            "exposure": dict(exposure),
        })
    manifest = {
        "schema": ARM_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "seed": seed,
        "parent_identity_sha256": parent["parent_identity_sha256"],
        "parent_state_tree_sha256": parent["parent_state_tree_sha256"],
        "arms": arms,
    }
    manifest["arm_manifest_sha256"] = sha_json(manifest)
    validate_arm_manifest(manifest)
    return manifest


CONTINUATION_FIELDS = {
    "schema", "protocol_sha256", "arm_id", "seed", "global_step", "checkpoint_id",
    "checkpoint_file_sha256", "parameter_sha256", "parent_checkpoint_file_sha256",
    "optimizer_state_sha256", "scheduler_state_sha256", "rng_state_sha256",
    "sampler_state_sha256", "dataloader_cursor_sha256", "token_ledger_sha256",
    "source_tree_sha256", "resume_contract_sha256", "fresh_process_restore_receipt_sha256",
    "resume_validated", "continuation_manifest_sha256",
}


def make_continuation_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    manifest = {"schema": CONTINUATION_SCHEMA, **dict(body)}
    manifest["continuation_manifest_sha256"] = sha_json(manifest)
    validate_continuation_manifest(manifest)
    return manifest


def validate_continuation_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _require_mapping(manifest, "continuation manifest")
    _require_keys(manifest, CONTINUATION_FIELDS, "continuation manifest")
    if manifest.get("schema") != CONTINUATION_SCHEMA or manifest.get("arm_id") not in TRAINING_ARMS:
        raise X6X7Error("continuation schema or arm is invalid")
    if manifest.get("resume_validated") is not True:
        raise X6X7Error("continuation lacks fresh-process restore validation")
    _integer(manifest.get("seed"), "continuation.seed")
    _integer(manifest.get("global_step"), "continuation.global_step", 1)
    for key in sorted(CONTINUATION_FIELDS - {"schema", "arm_id", "seed", "global_step", "resume_validated", "continuation_manifest_sha256"}):
        _hash(manifest.get(key), f"continuation.{key}")
    _self_hash(manifest, "continuation_manifest_sha256", "continuation manifest")
    return {"valid": True, "continuation_manifest_sha256": manifest["continuation_manifest_sha256"]}


EVALUATION_COMMITMENT_FIELDS = {
    "schema", "protocol_sha256", "run_manifest_sha256", "split_bundle_sha256",
    "arm_manifest_sha256", "checkpoint_identity_sha256s", "intervention_registry_sha256",
    "x1_policy_receipt_sha256", "sealed_outcome_receipt_sha256", "evaluator_source_sha256",
    "verifier_source_sha256", "task_ids", "retention_task_ids", "intervention_ids", "frozen_at",
    "outcomes_loaded", "checkpoint_selection_uses_evaluation_labels",
    "evaluation_commitment_sha256",
}


def build_evaluation_commitment(body: Mapping[str, Any]) -> dict[str, Any]:
    commitment = {
        "schema": EVALUATION_COMMITMENT_SCHEMA,
        **dict(body),
        "evaluation_commitment_sha256": body.get("evaluation_commitment_sha256", ""),
    }
    _require_keys(commitment, EVALUATION_COMMITMENT_FIELDS, "evaluation commitment")
    for key in (
        "protocol_sha256", "run_manifest_sha256", "split_bundle_sha256", "arm_manifest_sha256",
        "intervention_registry_sha256", "verifier_source_sha256",
        "evaluation_commitment_sha256",
    ):
        if key == "evaluation_commitment_sha256":
            continue
        _hash(commitment.get(key), f"evaluation_commitment.{key}")
    checkpoint_hashes = commitment.get("checkpoint_identity_sha256s")
    if not isinstance(checkpoint_hashes, dict) or set(checkpoint_hashes) != set(ARM_IDS) or any(not HEX64.fullmatch(str(value)) for value in checkpoint_hashes.values()):
        raise X6X7Error("evaluation commitment must bind every arm checkpoint identity")
    for field in ("x1_policy_receipt_sha256", "sealed_outcome_receipt_sha256"):
        value = commitment.get(field)
        if value is not None:
            _hash(value, f"evaluation_commitment.{field}")
    evaluator = commitment.get("evaluator_source_sha256")
    _hash(evaluator, "evaluation_commitment.evaluator_source_sha256")
    task_ids = commitment.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids or len(task_ids) != len(set(task_ids)):
        raise X6X7Error("evaluation commitment task IDs must be non-empty and unique")
    retention_task_ids = commitment.get("retention_task_ids")
    if not isinstance(retention_task_ids, list) or not retention_task_ids or len(retention_task_ids) != len(set(retention_task_ids)):
        raise X6X7Error("evaluation commitment retention task IDs must be non-empty and unique")
    if set(task_ids) & set(retention_task_ids):
        raise X6X7Error("evaluation and retention task sets overlap")
    intervention_ids = commitment.get("intervention_ids")
    if not isinstance(intervention_ids, list) or "NO_CHANGE" not in intervention_ids or len(intervention_ids) != len(set(intervention_ids)):
        raise X6X7Error("evaluation commitment interventions are invalid")
    _timestamp(commitment.get("frozen_at"), "evaluation_commitment.frozen_at")
    if commitment.get("outcomes_loaded") is not False or commitment.get("checkpoint_selection_uses_evaluation_labels") is not False:
        raise X6X7Error("evaluation commitment boundary is violated")
    commitment["evaluation_commitment_sha256"] = sha_json(_without(commitment, "evaluation_commitment_sha256"))
    return commitment


SOURCE_RELEASE_FIELDS = {
    "schema", "protocol_sha256", "source_closure_sha256", "base_revision",
    "release_revision", "source_tree_status", "files", "custodian",
    "release_manifest_sha256",
}


def build_source_release(body: Mapping[str, Any]) -> dict[str, Any]:
    release = {
        "schema": SOURCE_RELEASE_SCHEMA,
        **dict(body),
        "release_manifest_sha256": body.get("release_manifest_sha256", ""),
    }
    _require_keys(release, SOURCE_RELEASE_FIELDS, "source release")
    if release.get("source_tree_status") != "CLEAN":
        raise X6X7Error("source release tree is not clean")
    for key in ("protocol_sha256", "source_closure_sha256", "release_manifest_sha256"):
        if key == "release_manifest_sha256":
            continue
        _hash(release.get(key), f"source_release.{key}")
    for key in ("base_revision", "release_revision"):
        if not isinstance(release.get(key), str) or not HEX40.fullmatch(release[key]):
            raise X6X7Error(f"source_release.{key} must be a full commit SHA")
    files = release.get("files")
    if not isinstance(files, list) or not files:
        raise X6X7Error("source release file inventory is empty")
    for item in files:
        item = _require_mapping(item, "source release file")
        _require_keys(item, {"path", "sha256"}, "source release file")
        _relative_path(item.get("path"), "source release path")
        _hash(item.get("sha256"), "source release file hash")
    _text(release.get("custodian"), "source_release.custodian")
    release["release_manifest_sha256"] = sha_json(_without(release, "release_manifest_sha256"))
    return release


RUN_MANIFEST_FIELDS = {
    "schema", "protocol_sha256", "x1_entry_audit_sha256", "x1_receipt_file_sha256",
    "parent_identity_sha256", "parent_checkpoint_file_sha256", "parent_parameter_sha256",
    "parent_state_tree_sha256", "model_identity", "source_release_manifest_sha256",
    "split_bundle_sha256", "confirmation_frozen_at", "confirmation_inspection_count",
    "generators", "intervention_binding", "training_binding", "paired_seeds",
    "arm_manifest_sha256s", "dataset_manifest_sha256s", "continuation_manifest_sha256s",
    "controller_provenance", "prepared_at", "execution_authorized", "run_manifest_sha256",
}
INTERVENTION_BINDING_FIELDS = {
    "registry_schema", "registry_sha256", "intervention_ids", "transformation_hashes",
    "information_class", "legality_inputs", "cost_model_sha256", "target_intervention",
}
TRAINING_BINDING_FIELDS = {
    "optimizer_sha256", "schedule_sha256", "data_opportunity_sha256",
    "checkpoint_cadence_sha256", "evaluation_timing_sha256", "resume_contract_sha256",
}


def _validate_controller_provenance(value: Any, name: str) -> dict[str, Any]:
    controller = _require_mapping(value, name)
    kind = controller.get("kind")
    if kind not in {"fixed", "learned", "human_supplied", "externally_assisted", "none"}:
        raise X6X7Error(f"{name}.kind is invalid")
    if kind == "human_supplied":
        _text(controller.get("custodian"), f"{name}.custodian")
    else:
        _hash(controller.get("identity_sha256"), f"{name}.identity_sha256")
    return dict(controller)


def build_x6_run_manifest(
    protocol: Mapping[str, Any], entry_audit: Mapping[str, Any], parent: Mapping[str, Any],
    split_bundle: Mapping[str, Any], arm_manifests: Sequence[Mapping[str, Any]],
    dataset_manifests: Mapping[str, Mapping[str, Mapping[str, Any]]],
    continuation_manifests: Mapping[str, Mapping[str, Mapping[str, Any]]],
    source_release: Mapping[str, Any], intervention_binding: Mapping[str, Any],
    training_binding: Mapping[str, Any], controller_provenance: Mapping[str, Any],
    prepared_at: str,
) -> dict[str, Any]:
    validate_protocol(protocol)
    if validate_entry_audit(entry_audit)["authorized"] is not True:
        raise X6X7Error("run manifest cannot bind a blocked X1 entry")
    parent_verdict = validate_parent_identity(parent)
    if entry_audit.get("subject_manifest_sha256") != parent.get("subject_manifest_sha256"):
        raise X6X7Error("run manifest parent does not match the X1 subject")
    split_verdict = validate_split_bundle(split_bundle)
    if split_bundle.get("protocol_sha256") != protocol["identity"]["protocol_sha256"]:
        raise X6X7Error("run manifest split protocol mismatch")
    if not arm_manifests:
        raise X6X7Error("run manifest requires paired arm manifests")
    if source_release.get("protocol_sha256") != protocol["identity"]["protocol_sha256"] or source_release.get("source_closure_sha256") != protocol["identity"]["source_closure_sha256"]:
        raise X6X7Error("run manifest source release protocol mismatch")
    _self_hash(source_release, "release_manifest_sha256", "run manifest source release")
    seed_names = sorted({str(arm.get("seed")) for arm in arm_manifests})
    if len(seed_names) != len(arm_manifests):
        raise X6X7Error("run manifest contains duplicate paired seeds")
    arm_hashes: dict[str, str] = {}
    dataset_hashes: dict[str, dict[str, str]] = {}
    continuation_hashes: dict[str, dict[str, str]] = {}
    training_rows: list[Mapping[str, Any]] = []
    for seed_name in seed_names:
        matching_arms = [arm for arm in arm_manifests if str(arm.get("seed")) == seed_name]
        if len(matching_arms) != 1:
            raise X6X7Error("run manifest seed does not have one arm manifest")
        arm_manifest = matching_arms[0]
        validate_arm_manifest(arm_manifest)
        if arm_manifest.get("protocol_sha256") != protocol["identity"]["protocol_sha256"]:
            raise X6X7Error("run manifest arm protocol mismatch")
        if arm_manifest.get("parent_identity_sha256") != parent_verdict["parent_identity_sha256"]:
            raise X6X7Error("run manifest arm parent mismatch")
        arm_hashes[seed_name] = arm_manifest["arm_manifest_sha256"]
        if set(dataset_manifests.get(seed_name, {})) != set(TRAINING_ARMS):
            raise X6X7Error("run manifest seed dataset inventory is incomplete")
        if set(continuation_manifests.get(seed_name, {})) != set(TRAINING_ARMS):
            raise X6X7Error("run manifest seed continuation inventory is incomplete")
        for arm_id in TRAINING_ARMS:
            dataset = validate_dataset_manifest(dataset_manifests[seed_name][arm_id])
            manifest = dataset_manifests[seed_name][arm_id]
            if manifest.get("protocol_sha256") != protocol["identity"]["protocol_sha256"] or manifest.get("split_bundle_sha256") != split_bundle["bundle_sha256"]:
                raise X6X7Error("run manifest dataset binding mismatch")
            if next(arm for arm in arm_manifest["arms"] if arm["arm_id"] == arm_id)["dataset_manifest_sha256"] != dataset["dataset_manifest_sha256"]:
                raise X6X7Error("run manifest arm-dataset binding mismatch")
            continuation = validate_continuation_manifest(continuation_manifests[seed_name][arm_id])
            if continuation_manifests[seed_name][arm_id].get("protocol_sha256") != protocol["identity"]["protocol_sha256"]:
                raise X6X7Error("run manifest continuation protocol mismatch")
            if continuation_manifests[seed_name][arm_id].get("parent_checkpoint_file_sha256") != parent.get("checkpoint_file_sha256"):
                raise X6X7Error("run manifest continuation parent mismatch")
            dataset_hashes.setdefault(seed_name, {})[arm_id] = dataset["dataset_manifest_sha256"]
            continuation_hashes.setdefault(seed_name, {})[arm_id] = continuation["continuation_manifest_sha256"]
        for arm in arm_manifest["arms"]:
            if arm["arm_id"] in TRAINING_ARMS:
                training_rows.append(arm["exposure"])
    _require_keys(training_binding, TRAINING_BINDING_FIELDS, "training binding")
    for key in TRAINING_BINDING_FIELDS:
        _hash(training_binding.get(key), f"training_binding.{key}")
    for exposure in training_rows:
        for exposure_key, binding_key in (
            ("optimizer_sha256", "optimizer_sha256"),
            ("schedule_sha256", "schedule_sha256"),
            ("data_opportunity_sha256", "data_opportunity_sha256"),
            ("checkpoint_cadence_sha256", "checkpoint_cadence_sha256"),
            ("evaluation_timing_sha256", "evaluation_timing_sha256"),
        ):
            if exposure[exposure_key] != training_binding[binding_key]:
                raise X6X7Error("training binding differs from paired arm exposure")
    _require_keys(intervention_binding, INTERVENTION_BINDING_FIELDS, "intervention binding")
    _hash(intervention_binding.get("registry_sha256"), "intervention_binding.registry_sha256")
    _hash(intervention_binding.get("cost_model_sha256"), "intervention_binding.cost_model_sha256")
    intervention_ids = intervention_binding.get("intervention_ids")
    if not isinstance(intervention_ids, list) or "NO_CHANGE" not in intervention_ids or len(intervention_ids) != len(set(intervention_ids)):
        raise X6X7Error("run manifest intervention IDs are invalid")
    if intervention_binding.get("target_intervention") not in intervention_ids or intervention_binding.get("target_intervention") == "NO_CHANGE":
        raise X6X7Error("run manifest target intervention is invalid")
    if intervention_binding.get("information_class") != "INFORMATION_PRESERVING" or intervention_binding.get("legality_inputs") != "VISIBLE_TASK_ONLY":
        raise X6X7Error("run manifest intervention binding is not answer-blind")
    if not isinstance(intervention_binding.get("transformation_hashes"), dict) or set(intervention_binding["transformation_hashes"]) != set(intervention_ids):
        raise X6X7Error("run manifest transformation coverage is incomplete")
    for value in intervention_binding["transformation_hashes"].values():
        _hash(value, "intervention transformation hash")
    controller_provenance = _validate_controller_provenance(
        controller_provenance, "run_manifest.controller_provenance",
    )
    generators = {
        role: {
            "split_id": split_bundle["splits"][role]["split_id"],
            "manifest_sha256": split_bundle["splits"][role]["manifest_sha256"],
            "generator_id": split_bundle["splits"][role]["generator_id"],
            "generator_sha256": split_bundle["splits"][role]["generator_sha256"],
        }
        for role in SPLIT_ROLES
    }
    manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "protocol_sha256": protocol["identity"]["protocol_sha256"],
        "x1_entry_audit_sha256": entry_audit["entry_audit_sha256"],
        "x1_receipt_file_sha256": entry_audit["x1_receipt_file_sha256"],
        "parent_identity_sha256": parent["parent_identity_sha256"],
        "parent_checkpoint_file_sha256": parent["checkpoint_file_sha256"],
        "parent_parameter_sha256": parent["parameter_sha256"],
        "parent_state_tree_sha256": parent["parent_state_tree_sha256"],
        "model_identity": {
            key: parent[key]
            for key in (
                "model_config_sha256", "tokenizer_artifact_sha256",
                "tokenizer_identity_sha256", "runtime_source_revision",
                "runtime_source_sha256", "source_commit", "training_lineage",
                "stage", "global_step", "identity_attestation_sha256",
            )
        },
        "source_release_manifest_sha256": source_release["release_manifest_sha256"],
        "split_bundle_sha256": split_bundle["bundle_sha256"],
        "confirmation_frozen_at": split_bundle["splits"]["X6_CONFIRMATION"]["freeze"]["frozen_at"],
        "confirmation_inspection_count": split_bundle["splits"]["X6_CONFIRMATION"]["freeze"]["inspection_count"],
        "generators": generators,
        "intervention_binding": dict(intervention_binding),
        "training_binding": dict(training_binding),
        "paired_seeds": seed_names,
        "arm_manifest_sha256s": arm_hashes,
        "dataset_manifest_sha256s": dataset_hashes,
        "continuation_manifest_sha256s": continuation_hashes,
        "controller_provenance": dict(controller_provenance),
        "prepared_at": _timestamp(prepared_at, "run_manifest.prepared_at"),
        "execution_authorized": False,
    }
    manifest["run_manifest_sha256"] = sha_json(manifest)
    validate_x6_run_manifest(manifest)
    return manifest


def validate_x6_run_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _require_mapping(manifest, "X6 run manifest")
    _require_keys(manifest, RUN_MANIFEST_FIELDS, "X6 run manifest")
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA or manifest.get("execution_authorized") is not False:
        raise X6X7Error("X6 run manifest schema or execution boundary is invalid")
    for key in (
        "protocol_sha256", "x1_entry_audit_sha256", "x1_receipt_file_sha256",
        "parent_identity_sha256", "parent_checkpoint_file_sha256", "parent_parameter_sha256",
        "parent_state_tree_sha256", "source_release_manifest_sha256", "split_bundle_sha256",
        "run_manifest_sha256",
    ):
        _hash(manifest.get(key), f"run_manifest.{key}")
    _timestamp(manifest.get("prepared_at"), "run_manifest.prepared_at")
    _timestamp(manifest.get("confirmation_frozen_at"), "run_manifest.confirmation_frozen_at")
    model = _require_mapping(manifest.get("model_identity"), "run_manifest.model_identity")
    for key in (
        "model_config_sha256", "tokenizer_artifact_sha256", "tokenizer_identity_sha256",
        "runtime_source_sha256", "source_commit", "identity_attestation_sha256",
    ):
        if key == "source_commit":
            if not isinstance(model.get(key), str) or not HEX40.fullmatch(model[key]):
                raise X6X7Error("run manifest source commit is malformed")
        else:
            _hash(model.get(key), f"run_manifest.model_identity.{key}")
    _integer(model.get("global_step"), "run_manifest.model_identity.global_step")
    generators = _require_mapping(manifest.get("generators"), "run_manifest.generators")
    if set(generators) != set(SPLIT_ROLES):
        raise X6X7Error("run manifest generator role coverage is incomplete")
    for role, generator in generators.items():
        generator = _require_mapping(generator, f"run_manifest.generators.{role}")
        _require_keys(generator, {"split_id", "manifest_sha256", "generator_id", "generator_sha256"}, "run manifest generator")
        _hash(generator.get("manifest_sha256"), "run manifest split manifest hash")
        _hash(generator.get("generator_sha256"), "run manifest generator hash")
        _id(generator.get("split_id"), "run manifest split_id")
        _id(generator.get("generator_id"), "run manifest generator_id")
    if manifest.get("confirmation_inspection_count") != 0:
        raise X6X7Error("confirmation cohort has already been inspected")
    _validate_controller_provenance(manifest.get("controller_provenance"), "run_manifest.controller_provenance")
    seeds = manifest.get("paired_seeds")
    if not isinstance(seeds, list) or not seeds or seeds != sorted(set(seeds)):
        raise X6X7Error("run manifest paired seeds are invalid")
    for map_name in ("arm_manifest_sha256s", "dataset_manifest_sha256s", "continuation_manifest_sha256s"):
        mapping = manifest.get(map_name)
        if not isinstance(mapping, dict) or set(mapping) != set(map(str, seeds)):
            raise X6X7Error(f"run manifest {map_name} seed coverage is invalid")
        for value in mapping.values():
            if map_name == "arm_manifest_sha256s":
                _hash(value, map_name)
            elif set(value) != set(TRAINING_ARMS):
                raise X6X7Error(f"run manifest {map_name} arm coverage is invalid")
            else:
                for arm_hash in value.values():
                    _hash(arm_hash, map_name)
    _self_hash(manifest, "run_manifest_sha256", "X6 run manifest")
    return {"valid": True, "run_manifest_sha256": manifest["run_manifest_sha256"]}


def _readiness_body(
    schema: str, prefix: str, protocol: Mapping[str, Any], checks: Mapping[str, bool],
    blockers: Sequence[str], evidence: Mapping[str, Any],
) -> dict[str, Any]:
    status = "READY" if not blockers else "PREREQUISITE_BLOCKED"
    body = {
        "schema": schema,
        "status": status,
        "protocol_sha256": protocol.get("identity", {}).get("protocol_sha256"),
        "checks": dict(checks),
        "blockers": list(blockers),
        "evidence": dict(evidence),
        "execution_authorized": False,
    }
    body[f"{prefix}_sha256"] = sha_json(body)
    return body


def assess_x6_readiness(
    protocol: Mapping[str, Any], entry_audit: Mapping[str, Any], parent: Mapping[str, Any],
    split_bundle: Mapping[str, Any], arm_manifest: Mapping[str, Any],
    continuation_manifests: Sequence[Mapping[str, Any]], evaluation_commitment: Mapping[str, Any],
    source_release: Mapping[str, Any], dataset_manifests: Mapping[str, Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        protocol_verdict = validate_protocol(protocol)
    except X6X7Error:
        protocol_verdict = {"valid": False, "errors": ["protocol invalid"]}
    blockers: list[str] = []
    checks: dict[str, bool] = {}
    try:
        entry = validate_entry_audit(entry_audit)
        checks["x1_entry"] = entry["authorized"]
        if not entry["authorized"]:
            blockers.append("X1_ENTRY_NOT_SUPPORTED")
    except X6X7Error:
        checks["x1_entry"] = False
        blockers.append("X1_ENTRY_AUDIT_INVALID")
    try:
        validate_parent_identity(parent)
        if entry_audit.get("subject_manifest_sha256") != parent.get("subject_manifest_sha256"):
            raise X6X7Error("X1 subject manifest does not match the X6 parent")
        checks["parent_identity"] = True
    except X6X7Error:
        checks["parent_identity"] = False
        blockers.append("ELIGIBLE_PARENT_IDENTITY_MISSING")
    try:
        validate_split_bundle(split_bundle)
        if split_bundle.get("protocol_sha256") != protocol.get("identity", {}).get("protocol_sha256"):
            raise X6X7Error("split protocol mismatch")
        checks["split_isolation"] = True
    except X6X7Error:
        checks["split_isolation"] = False
        blockers.append("SPLIT_BUNDLE_INVALID")
    try:
        validate_arm_manifest(arm_manifest)
        if arm_manifest.get("protocol_sha256") != protocol.get("identity", {}).get("protocol_sha256"):
            raise X6X7Error("arm protocol mismatch")
        if arm_manifest.get("parent_identity_sha256") != parent.get("parent_identity_sha256"):
            raise X6X7Error("arm parent mismatch")
        checks["paired_arms"] = True
    except X6X7Error:
        checks["paired_arms"] = False
        blockers.append("PAIRED_ARM_MANIFEST_INVALID")
    try:
        if set(dataset_manifests) != set(TRAINING_ARMS):
            raise X6X7Error("both training dataset manifests are required")
        arm_by_id = {arm["arm_id"]: arm for arm in arm_manifest.get("arms", [])}
        for arm_id in TRAINING_ARMS:
            verdict = validate_dataset_manifest(dataset_manifests[arm_id])
            manifest = dataset_manifests[arm_id]
            if manifest.get("protocol_sha256") != protocol.get("identity", {}).get("protocol_sha256"):
                raise X6X7Error("dataset protocol mismatch")
            if manifest.get("split_bundle_sha256") != split_bundle.get("bundle_sha256"):
                raise X6X7Error("dataset split bundle mismatch")
            if arm_by_id[arm_id].get("dataset_manifest_sha256") != verdict["dataset_manifest_sha256"]:
                raise X6X7Error("arm does not bind dataset manifest")
        checks["dataset_binding"] = True
    except (KeyError, X6X7Error):
        checks["dataset_binding"] = False
        blockers.append("VERIFIED_DATASET_MANIFESTS_INVALID")
    try:
        run = validate_x6_run_manifest(run_manifest)
        expected = {
            "protocol_sha256": protocol.get("identity", {}).get("protocol_sha256"),
            "x1_entry_audit_sha256": entry_audit.get("entry_audit_sha256"),
            "parent_identity_sha256": parent.get("parent_identity_sha256"),
            "parent_checkpoint_file_sha256": parent.get("checkpoint_file_sha256"),
            "parent_parameter_sha256": parent.get("parameter_sha256"),
            "parent_state_tree_sha256": parent.get("parent_state_tree_sha256"),
            "source_release_manifest_sha256": source_release.get("release_manifest_sha256"),
            "split_bundle_sha256": split_bundle.get("bundle_sha256"),
        }
        if any(run_manifest.get(key) != value for key, value in expected.items()):
            raise X6X7Error("run manifest cross-artifact binding mismatch")
        if str(arm_manifest.get("seed")) not in set(run_manifest.get("paired_seeds", [])):
            raise X6X7Error("run manifest does not bind the supplied paired arm")
        threshold_status = protocol.get("decision_thresholds", {}).get("authority_status")
        if threshold_status == "AUTHORIZED" and len(run_manifest.get("paired_seeds", [])) < int(protocol["decision_thresholds"].get("minimum_paired_seeds", 2)):
            raise X6X7Error("run manifest has too few paired seeds")
        if run["run_manifest_sha256"] != run_manifest.get("run_manifest_sha256"):
            raise X6X7Error("run manifest identity changed")
        checks["run_manifest"] = True
    except (KeyError, X6X7Error):
        checks["run_manifest"] = False
        blockers.append("SOURCE_BOUND_RUN_MANIFEST_INVALID")
    continuation_ok = len(continuation_manifests) == len(TRAINING_ARMS)
    continuation_arms: set[str] = set()
    for manifest in continuation_manifests:
        try:
            validate_continuation_manifest(manifest)
            if manifest.get("protocol_sha256") != protocol.get("identity", {}).get("protocol_sha256"):
                raise X6X7Error("continuation protocol mismatch")
            if manifest.get("parent_checkpoint_file_sha256") != parent.get("checkpoint_file_sha256"):
                raise X6X7Error("continuation parent mismatch")
            continuation_arms.add(manifest["arm_id"])
        except X6X7Error:
            continuation_ok = False
    if continuation_arms != set(TRAINING_ARMS):
        continuation_ok = False
    checks["continuation_contract"] = continuation_ok
    if not continuation_ok:
        blockers.append("EXACT_CONTINUATION_CONTRACT_MISSING")
    try:
        expected_protocol = protocol["identity"]["protocol_sha256"]
        if evaluation_commitment.get("protocol_sha256") != expected_protocol:
            raise X6X7Error("evaluation protocol mismatch")
        if evaluation_commitment.get("outcomes_loaded") is not False:
            raise X6X7Error("outcomes were loaded before commitment")
        if evaluation_commitment.get("checkpoint_selection_uses_evaluation_labels") is not False:
            raise X6X7Error("evaluation labels influenced checkpoint selection")
        if evaluation_commitment.get("split_bundle_sha256") != split_bundle.get("bundle_sha256"):
            raise X6X7Error("evaluation split binding mismatch")
        if evaluation_commitment.get("arm_manifest_sha256") != arm_manifest.get("arm_manifest_sha256"):
            raise X6X7Error("evaluation arm binding mismatch")
        expected_tasks = {
            task_id
            for cluster in split_bundle["splits"]["X6_CONFIRMATION"]["clusters"]
            for task_id in cluster["task_ids"]
        }
        if set(evaluation_commitment.get("task_ids", [])) != expected_tasks:
            raise X6X7Error("evaluation commitment task set mismatch")
        expected_retention = {
            task_id
            for cluster in split_bundle["splits"]["X6_RETENTION"]["clusters"]
            for task_id in cluster["task_ids"]
        }
        if set(evaluation_commitment.get("retention_task_ids", [])) != expected_retention:
            raise X6X7Error("evaluation commitment retention task set mismatch")
        _self_hash(evaluation_commitment, "evaluation_commitment_sha256", "evaluation commitment")
        checks["evaluation_commitment"] = True
    except (KeyError, X6X7Error):
        checks["evaluation_commitment"] = False
        blockers.append("PREDICTED_EVALUATION_BOUNDARY_INVALID")
    try:
        identity = protocol["identity"]
        if source_release.get("protocol_sha256") != identity["protocol_sha256"] or source_release.get("source_closure_sha256") != identity["source_closure_sha256"]:
            raise X6X7Error("source release protocol binding mismatch")
        if source_release.get("source_tree_status") != "CLEAN":
            raise X6X7Error("source release is not clean")
        if source_release.get("base_revision") != identity["base_revision"]:
            raise X6X7Error("source release base revision mismatch")
        if not isinstance(source_release.get("release_revision"), str) or not HEX40.fullmatch(source_release["release_revision"]):
            raise X6X7Error("source release revision is malformed")
        if source_closure_sha256(identity["base_revision"], source_release.get("files", [])) != identity["source_closure_sha256"]:
            raise X6X7Error("source release closure does not match protocol")
        _self_hash(source_release, "release_manifest_sha256", "source release")
        checks["source_release"] = True
    except (KeyError, X6X7Error):
        checks["source_release"] = False
        blockers.append("CLEAN_SOURCE_RELEASE_MISSING")
    threshold_status = protocol.get("decision_thresholds", {}).get("authority_status")
    checks["decision_thresholds_authorized"] = threshold_status == "AUTHORIZED"
    if threshold_status != "AUTHORIZED":
        blockers.append("RAW_GAIN_AND_LIFT_REDUCTION_THRESHOLDS_UNRESOLVED")
    checks["protocol_valid"] = protocol_verdict["valid"]
    if not protocol_verdict["valid"]:
        blockers.append("PROTOCOL_INVALID")
    return _readiness_body(
        READINESS_SCHEMA, "readiness", protocol, checks, blockers,
        {
            "entry_audit_sha256": entry_audit.get("entry_audit_sha256"),
            "parent_identity_sha256": parent.get("parent_identity_sha256"),
            "split_bundle_sha256": split_bundle.get("bundle_sha256"),
            "arm_manifest_sha256": arm_manifest.get("arm_manifest_sha256"),
            "dataset_manifest_sha256s": {
                arm_id: dataset_manifests.get(arm_id, {}).get("dataset_manifest_sha256")
                for arm_id in TRAINING_ARMS
            },
            "run_manifest_sha256": run_manifest.get("run_manifest_sha256"),
            "paired_seeds": run_manifest.get("paired_seeds"),
            "evaluation_commitment_sha256": evaluation_commitment.get("evaluation_commitment_sha256"),
            "source_release_manifest_sha256": source_release.get("release_manifest_sha256"),
        },
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise X6X7Error("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def paired_cluster_effect(
    treatment: Mapping[str, float], control: Mapping[str, float], seed: int,
    n_resamples: int = 2000,
) -> dict[str, Any]:
    if set(treatment) != set(control) or not treatment:
        raise X6X7Error("paired effect requires identical non-empty clusters")
    clusters = sorted(treatment)
    deltas = [float(treatment[cluster]) - float(control[cluster]) for cluster in clusters]
    point = _mean(deltas)
    if len(clusters) < 2:
        return {"status": "INSUFFICIENT_CLUSTERS", "n_clusters": len(clusters), "point_estimate": point, "ci95": None}
    rng = random.Random(seed)
    draws = [_mean([deltas[rng.randrange(len(deltas))] for _ in deltas]) for _ in range(n_resamples)]
    return {
        "status": "OK",
        "n_clusters": len(clusters),
        "n_resamples": n_resamples,
        "point_estimate": point,
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "paired_unit": "independent_latent_world",
    }


def _validate_observation_split(
    observations: Sequence[Mapping[str, Any]], split_bundle: Mapping[str, Any], role: str,
) -> None:
    split = split_bundle["splits"][role]
    task_map = {
        task_id: (cluster["cluster_id"], set(cluster["transfer_axes"]))
        for cluster in split["clusters"]
        for task_id in cluster["task_ids"]
    }
    observed_tasks = {row["task_id"] for row in observations}
    if observed_tasks != set(task_map):
        raise X6X7Error(f"{role} observations do not exactly cover the frozen task set")
    coverage: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in observations:
        expected = task_map.get(row["task_id"])
        if expected is None or expected[0] != row["cluster_id"] or row["transfer_axis"] not in expected[1]:
            raise X6X7Error(f"{role} observation cluster or transfer axis mismatch")
        coverage[(row["arm_id"], int(row["seed"]))].add(row["task_id"])
    if any(tasks != observed_tasks for tasks in coverage.values()):
        raise X6X7Error(f"{role} arm/seed task coverage is incomplete")


def _outcome_summary(rows: Sequence[Mapping[str, bool | float]]) -> dict[str, float]:
    if not rows:
        return {"accuracy": 0.0, "valid_stop_rate": 0.0, "invalid_output_rate": 0.0, "abstention_rate": 0.0, "brier": 0.0, "calibration_error": 0.0}
    accuracies = [float(bool(row["correct"])) for row in rows]
    brier = _mean([
        (float(row["predicted_probability_correct"]) - float(bool(row["correct"]))) ** 2
        for row in rows
    ])
    calibration_error = 0.0
    for probability in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        bucket = [row for row in rows if min(0.9, max(0.0, float(row["predicted_probability_correct"]))) == probability]
        if bucket:
            calibration_error += len(bucket) / len(rows) * abs(_mean([float(row["predicted_probability_correct"]) for row in bucket]) - _mean([float(bool(row["correct"])) for row in bucket]))
    return {
        "accuracy": _mean(accuracies),
        "valid_stop_rate": _mean([float(bool(row["valid_stop"])) for row in rows]),
        "invalid_output_rate": _mean([float(bool(row["invalid_output"])) for row in rows]),
        "abstention_rate": _mean([float(bool(row["abstained"])) for row in rows]),
        "brier": brier,
        "calibration_error": calibration_error,
    }


def summarize_evaluation_observations(
    observations: Sequence[Mapping[str, Any]], intervention_ids: Sequence[str],
) -> dict[str, Any]:
    if not observations:
        raise X6X7Error("evaluation observations cannot be empty")
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    expected_interventions = list(intervention_ids)
    seen_cells: set[tuple[str, int, str]] = set()
    task_sets: dict[tuple[str, int], set[str]] = defaultdict(set)
    cluster_sets: dict[tuple[str, int], set[str]] = defaultdict(set)
    for raw in observations:
        row = _require_mapping(raw, "evaluation observation")
        _require_keys(
            row,
            {"arm_id", "seed", "task_id", "cluster_id", "transfer_axis", "unaided", "assisted"},
            "evaluation observation",
        )
        if row["arm_id"] not in ARM_IDS:
            raise X6X7Error("unknown evaluation arm")
        if row["transfer_axis"] not in TRANSFER_AXES:
            raise X6X7Error("invalid transfer axis")
        _id(row["task_id"], "observation.task_id")
        _id(row["cluster_id"], "observation.cluster_id")
        _integer(row["seed"], "observation.seed")
        cell = (row["arm_id"], row["seed"], row["task_id"])
        if cell in seen_cells:
            raise X6X7Error("duplicate evaluation observation")
        seen_cells.add(cell)
        unaided = _require_mapping(row["unaided"], "unaided outcome")
        outcome_fields = {"correct", "valid_stop", "invalid_output", "abstained", "predicted_probability_correct", "cost"}
        _require_keys(unaided, outcome_fields, "unaided outcome")
        assisted = row["assisted"]
        if not isinstance(assisted, list) or [item.get("intervention_id") for item in assisted] != expected_interventions:
            raise X6X7Error("assisted outcomes must exactly cover frozen intervention order")
        for outcome in [unaided, *[_require_mapping(item, "assisted outcome") for item in assisted]]:
            _require_keys(outcome, outcome_fields | ({"intervention_id"} if outcome is not unaided else set()), "outcome")
            for key in ("correct", "valid_stop", "invalid_output", "abstained"):
                if not isinstance(outcome[key], bool):
                    raise X6X7Error(f"outcome.{key} must be boolean")
            _probability(outcome["predicted_probability_correct"], "predicted_probability_correct")
            if not isinstance(outcome["cost"], (int, float)) or isinstance(outcome["cost"], bool) or not math.isfinite(float(outcome["cost"])) or float(outcome["cost"]) < 0:
                raise X6X7Error("outcome cost must be finite and non-negative")
        grouped[(row["arm_id"], row["seed"])].append(row)
        task_sets[(row["arm_id"], row["seed"])].add(row["task_id"])
        cluster_sets[(row["arm_id"], row["seed"])].add(row["cluster_id"])
    reference_tasks = next(iter(task_sets.values()))
    reference_clusters = next(iter(cluster_sets.values()))
    for key, tasks in task_sets.items():
        if tasks != reference_tasks or cluster_sets[key] != reference_clusters:
            raise X6X7Error(f"arm/seed task or cluster coverage differs: {key}")
    by_arm_seed: dict[str, dict[str, Any]] = {}
    by_arm_cluster: dict[str, dict[str, dict[str, float]]] = {}
    for (arm_id, seed), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {"unaided": _outcome_summary([row["unaided"] for row in rows])}
        for index, intervention_id in enumerate(intervention_ids):
            summary[f"assisted:{intervention_id}"] = _outcome_summary([row["assisted"][index] for row in rows])
            summary[f"lift:{intervention_id}"] = summary[f"assisted:{intervention_id}"]["accuracy"] - summary["unaided"]["accuracy"]
        summary["seed"] = seed
        by_arm_seed.setdefault(arm_id, {})[str(seed)] = summary
        clusters: dict[str, dict[str, float]] = {}
        for cluster_id in sorted({row["cluster_id"] for row in rows}):
            cluster_rows = [row for row in rows if row["cluster_id"] == cluster_id]
            cluster_summary = {"unaided": _outcome_summary([row["unaided"] for row in cluster_rows])["accuracy"]}
            for index, intervention_id in enumerate(intervention_ids):
                assisted_accuracy = _outcome_summary([row["assisted"][index] for row in cluster_rows])["accuracy"]
                cluster_summary[f"assisted:{intervention_id}"] = assisted_accuracy
                cluster_summary[f"lift:{intervention_id}"] = assisted_accuracy - cluster_summary["unaided"]
            clusters[cluster_id] = cluster_summary
        by_arm_cluster.setdefault(arm_id, {})[str(seed)] = clusters
        summary["cluster_macro_unaided_accuracy"] = _mean([
            values["unaided"] for values in clusters.values()
        ])
        for intervention_id in intervention_ids:
            summary[f"cluster_macro_lift:{intervention_id}"] = _mean([
                values[f"lift:{intervention_id}"] for values in clusters.values()
            ])
    return {"by_arm_seed": by_arm_seed, "by_arm_cluster": by_arm_cluster}


def _effect_bundle(
    summaries: Mapping[str, Any], treatment_arm: str, control_arm: str,
    target_intervention: str, n_resamples: int,
) -> dict[str, Any]:
    seeds = sorted(set(summaries["by_arm_cluster"][treatment_arm]) & set(summaries["by_arm_cluster"][control_arm]))
    if not seeds:
        raise X6X7Error("no paired seeds")
    effects: dict[str, Any] = {}
    for seed in seeds:
        treatment = summaries["by_arm_cluster"][treatment_arm][seed]
        control = summaries["by_arm_cluster"][control_arm][seed]
        effects[seed] = {
            "unaided_vs_parent": paired_cluster_effect(
                {cluster: values["unaided"] for cluster, values in treatment.items()},
                {cluster: values["unaided"] for cluster, values in control.items()},
                int(seed) + 1,
                n_resamples,
            ),
            "lift_reduction_vs_parent": paired_cluster_effect(
                {cluster: values[f"lift:{target_intervention}"] for cluster, values in control.items()},
                {cluster: values[f"lift:{target_intervention}"] for cluster, values in treatment.items()},
                int(seed) + 2,
                n_resamples,
            ),
        }
    return {
        "treatment_arm": treatment_arm,
        "control_arm": control_arm,
        "target_intervention": target_intervention,
        "by_seed": effects,
    }


def _x6_effects(
    summaries: Mapping[str, Any], target_intervention: str, n_resamples: int,
) -> dict[str, Any]:
    effects = _effect_bundle(
        summaries, "REPAIR_INTERNALIZATION", "PARENT", target_intervention, n_resamples,
    )
    effects["by_seed"] = {
        seed: {
            "unaided_vs_parent": effects["by_seed"][seed]["unaided_vs_parent"],
            "unaided_vs_control": paired_cluster_effect(
                {cluster: values["unaided"] for cluster, values in summaries["by_arm_cluster"]["REPAIR_INTERNALIZATION"][seed].items()},
                {cluster: values["unaided"] for cluster, values in summaries["by_arm_cluster"]["RAW_FAILURE_SFT_CONTROL"][seed].items()},
                int(seed) + 11,
                n_resamples,
            ),
            "lift_reduction_vs_parent": effects["by_seed"][seed]["lift_reduction_vs_parent"],
            "lift_reduction_vs_control": paired_cluster_effect(
                {cluster: values[f"lift:{target_intervention}"] for cluster, values in summaries["by_arm_cluster"]["RAW_FAILURE_SFT_CONTROL"][seed].items()},
                {cluster: values[f"lift:{target_intervention}"] for cluster, values in summaries["by_arm_cluster"]["REPAIR_INTERNALIZATION"][seed].items()},
                int(seed) + 12,
                n_resamples,
            ),
        }
        for seed in sorted(summaries["by_arm_cluster"]["REPAIR_INTERNALIZATION"])
    }
    return effects


def _x7_effects(
    summaries: Mapping[str, Any], target_intervention: str, n_resamples: int,
) -> dict[str, Any]:
    effects = _effect_bundle(
        summaries, "REPAIR_INTERNALIZATION", "PARENT", target_intervention, n_resamples,
    )
    effects["by_seed"] = {
        seed: {
            "unaided_vs_parent": effects["by_seed"][seed]["unaided_vs_parent"],
            "unaided_vs_control": paired_cluster_effect(
                {cluster: values["unaided"] for cluster, values in summaries["by_arm_cluster"]["REPAIR_INTERNALIZATION"][seed].items()},
                {cluster: values["unaided"] for cluster, values in summaries["by_arm_cluster"]["RAW_FAILURE_SFT_CONTROL"][seed].items()},
                int(seed) + 21,
                n_resamples,
            ),
            "lift_reduction_vs_parent": effects["by_seed"][seed]["lift_reduction_vs_parent"],
            "lift_reduction_vs_control": paired_cluster_effect(
                {cluster: values[f"lift:{target_intervention}"] for cluster, values in summaries["by_arm_cluster"]["RAW_FAILURE_SFT_CONTROL"][seed].items()},
                {cluster: values[f"lift:{target_intervention}"] for cluster, values in summaries["by_arm_cluster"]["REPAIR_INTERNALIZATION"][seed].items()},
                int(seed) + 22,
                n_resamples,
            ),
        }
        for seed in sorted(summaries["by_arm_cluster"]["REPAIR_INTERNALIZATION"])
    }
    return effects


def _retention_report(
    retention_rows: Sequence[Mapping[str, Any]], treatment_arm: str, control_arm: str,
    floor: float,
) -> dict[str, Any]:
    required = {
        "arm_id", "seed", "capability_id", "parent_score", "child_score",
        "n_independent_worlds", "score_semantics",
    }
    normalized = []
    for row in retention_rows:
        row = _require_mapping(row, "retention row")
        _require_keys(row, required, "retention row")
        if row["arm_id"] not in {treatment_arm, control_arm}:
            raise X6X7Error("retention row arm is not a paired arm")
        for key in ("parent_score", "child_score"):
            _probability(row[key], f"retention.{key}")
        _integer(row["n_independent_worlds"], "retention.n_independent_worlds", 2)
        normalized.append(dict(row))
    if not normalized:
        raise X6X7Error("retention rows cannot be empty")
    reports = []
    for row in sorted(normalized, key=lambda item: (item["arm_id"], int(item["seed"]), item["capability_id"])):
        delta = float(row["child_score"]) - float(row["parent_score"])
        reports.append({
            **row,
            "delta": delta,
            "floor": -abs(floor),
            "passed": delta >= -abs(floor),
        })
    return {
        "per_capability": reports,
        "all_passed": all(item["passed"] for item in reports),
        "material_regressions": [item["capability_id"] for item in reports if not item["passed"]],
    }


def _transfer_report(
    summaries: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
    treatment_arm: str, control_arm: str, target_intervention: str,
    n_resamples: int,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    seeds = sorted(summaries["by_arm_cluster"][treatment_arm])
    for axis in TRANSFER_AXES:
        by_seed: dict[str, Any] = {}
        for seed in seeds:
            axis_clusters: dict[str, set[str]] = defaultdict(set)
            for row in observations:
                if row["transfer_axis"] == axis and int(row["seed"]) == int(seed):
                    axis_clusters[str(row["arm_id"])].add(row["cluster_id"])
            common_clusters = sorted(
                axis_clusters[treatment_arm] & axis_clusters[control_arm]
            )
            treatment: dict[str, float] = {}
            control: dict[str, float] = {}
            for cluster_id in common_clusters:
                treatment_rows = [
                    row for row in observations
                    if row["transfer_axis"] == axis and int(row["seed"]) == int(seed)
                    and row["arm_id"] == treatment_arm and row["cluster_id"] == cluster_id
                ]
                control_rows = [
                    row for row in observations
                    if row["transfer_axis"] == axis and int(row["seed"]) == int(seed)
                    and row["arm_id"] == control_arm and row["cluster_id"] == cluster_id
                ]
                treatment[cluster_id] = _mean([float(bool(row["unaided"]["correct"])) for row in treatment_rows])
                control[cluster_id] = _mean([float(bool(row["unaided"]["correct"])) for row in control_rows])
            by_seed[seed] = (
                paired_cluster_effect(treatment, control, int(seed) + 3, n_resamples)
                if treatment else {"status": "INSUFFICIENT_CLUSTERS", "n_clusters": 0, "ci95": None}
            )
        reports[axis] = {
            "target_intervention": target_intervention,
            "by_seed": by_seed,
        }
    return reports


def summarize_repair_choice(
    rows: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]],
    intervention_ids: Sequence[str, str], cost_lambda: float,
) -> dict[str, Any]:
    _probability(cost_lambda, "repair-choice cost_lambda")
    required = {

        "schema", "arm_id", "seed", "task_id", "cluster_id", "policy_receipt_sha256",
        "selected_intervention_id", "unaided_correct", "selected_correct", "intervention_cost",
    }
    expected = {
        (row["arm_id"], int(row["seed"]), row["task_id"]): row
        for row in observations
    }
    seen: set[tuple[str, int, str]] = set()
    policies: set[str] = set()
    clusters: dict[tuple[str, int, str], dict[str, Any]] = defaultdict(list)
    for raw in rows:
        row = _require_mapping(raw, "repair-choice row")
        _require_keys(row, required, "repair-choice row")
        if row.get("schema") != REPAIR_CHOICE_SCHEMA:
            raise X6X7Error("repair-choice schema mismatch")
        key = (row.get("arm_id"), int(row.get("seed", -1)), row.get("task_id"))
        if key not in expected or key in seen:
            raise X6X7Error("repair-choice rows do not exactly match evaluation cells")
        seen.add(key)
        observation = expected[key]
        if row.get("cluster_id") != observation["cluster_id"]:
            raise X6X7Error("repair-choice cluster identity mismatch")
        if row.get("selected_intervention_id") not in intervention_ids:
            raise X6X7Error("repair choice selected an illegal intervention")
        for field in ("unaided_correct", "selected_correct"):
            if not isinstance(row.get(field), bool):
                raise X6X7Error(f"repair-choice {field} must be boolean")
        if not isinstance(row.get("intervention_cost"), (int, float)) or isinstance(row.get("intervention_cost"), bool) or not math.isfinite(float(row["intervention_cost"])) or float(row["intervention_cost"]) < 0:
            raise X6X7Error("repair-choice intervention cost is invalid")
        _hash(row.get("policy_receipt_sha256"), "repair-choice policy receipt")
        policies.add(row["policy_receipt_sha256"])
        clusters[(row["arm_id"], int(row["seed"]), row["cluster_id"])].append(row)
    if seen != set(expected) or len(policies) != 1:
        raise X6X7Error("repair-choice coverage or policy identity is incomplete")
    by_arm_seed: dict[str, dict[str, Any]] = {}
    for arm_id in sorted({key[0] for key in clusters}):
        for seed in sorted({key[1] for key in clusters if key[0] == arm_id}):
            cell_rows = [row for (row_arm, row_seed, _), values in clusters.items() for row in values if row_arm == arm_id and row_seed == seed]
            utility_by_cluster: dict[str, list[float]] = defaultdict(list)
            for row in cell_rows:
                baseline_utility = float(row["unaided_correct"])
                selected_utility = float(row["selected_correct"]) - cost_lambda * float(row["intervention_cost"])
                utility_by_cluster[row["cluster_id"]].append(selected_utility - baseline_utility)
            by_arm_seed.setdefault(arm_id, {})[str(seed)] = {
                "cluster_macro_utility_gain": _mean([
                    _mean(values) for values in utility_by_cluster.values()
                ]),
                "mean_intervention_cost": _mean([float(row["intervention_cost"]) for row in cell_rows]),
                "false_intervention_rate": _mean([
                    float(row["unaided_correct"] and row["selected_intervention_id"] != "NO_CHANGE")
                    for row in cell_rows
                ]),
            }
    return {
        "status": "AVAILABLE",
        "policy_receipt_sha256": next(iter(policies)),
        "by_arm_seed": by_arm_seed,
        "cost_lambda": cost_lambda,
    }


def _gate_from_effect(effect: Mapping[str, Any], threshold: float, require_ci: bool) -> bool:
    ci = effect.get("ci95")
    return (
        isinstance(ci, list)
        and len(ci) == 2
        and float(effect.get("point_estimate", -math.inf)) >= threshold
        and (not require_ci or float(ci[0]) > 0.0)
    )


def _x6_decision(
    protocol: Mapping[str, Any], readiness: Mapping[str, Any], effects: Mapping[str, Any],
    retention: Mapping[str, Any], transfer: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = protocol.get("decision_thresholds", {})
    required = {
        "min_unaided_gain_vs_parent", "min_unaided_gain_vs_control",
        "min_lift_reduction_vs_parent", "min_lift_reduction_vs_control",
        "minimum_independent_worlds", "minimum_paired_seeds",
        "bootstrap_resamples", "require_ci_above_zero",
        "min_transfer_gain_by_axis",
    }
    if readiness.get("status") != "READY" or thresholds.get("authority_status") != "AUTHORIZED" or any(thresholds.get(key) is None for key in required):
        return {"status": "PREREQUISITE_BLOCKED", "supported": False, "failed_gates": ["frozen decision thresholds or readiness"]}
    by_seed = effects["by_seed"]
    if len(by_seed) < int(thresholds["minimum_paired_seeds"]):
        return {"status": "INCONCLUSIVE", "supported": False, "failed_gates": ["paired-seed sample size"]}
    min_clusters = int(thresholds["minimum_independent_worlds"])
    if any(seed_result["unaided_vs_parent"].get("n_clusters", 0) < min_clusters for seed_result in by_seed.values()):
        return {"status": "INCONCLUSIVE", "supported": False, "failed_gates": ["independent-world sample size"]}
    if any(row["n_independent_worlds"] < min_clusters for row in retention["per_capability"]):
        return {"status": "INCONCLUSIVE", "supported": False, "failed_gates": ["retention independent-world sample size"]}
    gates = {
        "unaided_vs_parent": all(_gate_from_effect(seed_result["unaided_vs_parent"], float(thresholds["min_unaided_gain_vs_parent"]), bool(thresholds["require_ci_above_zero"])) for seed_result in by_seed.values()),
        "unaided_vs_control": all(_gate_from_effect(seed_result["unaided_vs_parent"], float(thresholds["min_unaided_gain_vs_control"]), bool(thresholds["require_ci_above_zero"])) for seed_result in by_seed.values()),
        "lift_reduction_vs_parent": all(_gate_from_effect(seed_result["lift_reduction_vs_parent"], float(thresholds["min_lift_reduction_vs_parent"]), bool(thresholds["require_ci_above_zero"])) for seed_result in by_seed.values()),
        "lift_reduction_vs_control": all(_gate_from_effect(seed_result["lift_reduction_vs_parent"], float(thresholds["min_lift_reduction_vs_control"]), bool(thresholds["require_ci_above_zero"])) for seed_result in by_seed.values()),
        "retention": retention["all_passed"],
    }
    if not all(gates.values()):
        return {"status": "NOT_SUPPORTED", "supported": False, "failed_gates": [key for key, value in gates.items() if not value], "gates": gates}
    transfer_gates = {}
    for axis, report in transfer.items():
        minimum = float(thresholds["min_transfer_gain_by_axis"][axis])
        transfer_gates[axis] = all(_gate_from_effect(seed_result, minimum, bool(thresholds["require_ci_above_zero"])) for seed_result in report["by_seed"].values())
    if all(transfer_gates.values()):
        status = "SUPPORTED_SCOPED_WITH_TRANSFER"
    else:
        status = "SUPPORTED_SCOPED_NO_TRANSFER"
    return {
        "status": status,
        "supported": True,
        "transfer_supported": all(transfer_gates.values()),
        "gates": gates,
        "transfer_gates": transfer_gates,
        "claim_ceiling": "unaided internalization and reduced intervention dependence in the frozen tested regime; never AGI",
    }


def build_x6_result_receipt(
    protocol: Mapping[str, Any], readiness: Mapping[str, Any], run_manifest_sha256: str,
    split_bundle: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
    intervention_ids: Sequence[str, Any], target_intervention: str,
    retention_rows: Sequence[Mapping[str, Any]], costs: Mapping[str, Any],
    controller_provenance: Mapping[str, Any], n_resamples: int = 2000,
    repair_choice_rows: Sequence[Mapping[str, Any]] | None = None,
    repair_choice_cost_lambda: float | None = None,
) -> dict[str, Any]:
    validate_protocol(protocol)
    if readiness.get("schema") != READINESS_SCHEMA or readiness.get("status") not in {"READY", "PREREQUISITE_BLOCKED"}:
        raise X6X7Error("X6 readiness receipt is invalid")
    _self_hash(readiness, "readiness_sha256", "X6 readiness")
    if readiness.get("protocol_sha256") != protocol["identity"]["protocol_sha256"]:
        raise X6X7Error("X6 readiness protocol mismatch")
    if readiness.get("evidence", {}).get("run_manifest_sha256") != run_manifest_sha256:
        raise X6X7Error("X6 result does not bind the ready run manifest")
    expected_seeds = {int(seed) for seed in readiness.get("evidence", {}).get("paired_seeds", [])}
    observed_seeds = {int(row["seed"]) for row in observations}
    if observed_seeds != expected_seeds:
        raise X6X7Error("X6 evaluation seeds do not match the ready run manifest")
    if not intervention_ids or len(intervention_ids) != len(set(intervention_ids)) or "NO_CHANGE" not in intervention_ids:
        raise X6X7Error("frozen intervention set is invalid")
    if target_intervention not in intervention_ids:
        raise X6X7Error("target intervention is not in the frozen intervention set")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise X6X7Error("bootstrap resamples must be positive")
    if protocol.get("decision_thresholds", {}).get("authority_status") == "AUTHORIZED" and n_resamples != protocol["decision_thresholds"].get("bootstrap_resamples"):
        raise X6X7Error("bootstrap resample count differs from the frozen protocol")
    validate_split_bundle(split_bundle)
    if split_bundle.get("protocol_sha256") != protocol["identity"]["protocol_sha256"]:
        raise X6X7Error("X6 result split protocol mismatch")
    _validate_observation_split(observations, split_bundle, "X6_CONFIRMATION")
    summaries = summarize_evaluation_observations(observations, intervention_ids)
    effects = _x6_effects(summaries, target_intervention, n_resamples)
    retention = _retention_report(
        retention_rows, "REPAIR_INTERNALIZATION", "RAW_FAILURE_SFT_CONTROL",
        float(protocol["canonical_authority"]["retention_floor_absolute_accuracy"]),
    )
    transfer = _transfer_report(
        summaries, observations, "REPAIR_INTERNALIZATION", "RAW_FAILURE_SFT_CONTROL",
        target_intervention, n_resamples,
    )
    if not isinstance(costs, Mapping) or set(costs) != {"training", "evaluation", "intervention", "checkpoint"}:
        raise X6X7Error("X6 cost receipt must separate training, evaluation, intervention, and checkpoint costs")
    controller_provenance = _validate_controller_provenance(
        controller_provenance, "X6.controller_provenance",
    )
    if repair_choice_rows is None:
        repair_choice = {"status": "NOT_AVAILABLE_NO_FROZEN_X1_POLICY"}
    else:
        if repair_choice_cost_lambda is None:
            raise X6X7Error("repair-choice cost lambda is required when policy utility is reported")
        repair_choice = summarize_repair_choice(
            repair_choice_rows, observations, intervention_ids, repair_choice_cost_lambda,
        )
    decision = _x6_decision(protocol, readiness, effects, retention, transfer)
    body = {
        "schema": X6_RECEIPT_SCHEMA,
        "phase": "RESULT_COMMITTED" if decision["status"] != "PREREQUISITE_BLOCKED" else "RESULT_PREREQUISITE_BLOCKED",
        "protocol_sha256": protocol["identity"]["protocol_sha256"],
        "readiness_sha256": readiness.get("readiness_sha256"),
        "run_manifest_sha256": _hash(run_manifest_sha256, "run_manifest_sha256"),
        "split_bundle_sha256": split_bundle["bundle_sha256"],
        "evaluation_commitment_sha256": readiness.get("evidence", {}).get("evaluation_commitment_sha256"),
        "arms": list(ARM_IDS),
        "intervention_ids": list(intervention_ids),
        "seeds": sorted({int(row["seed"]) for row in observations}),
        "raw_outcomes": summaries["by_arm_seed"],
        "cluster_results": summaries["by_arm_cluster"],
        "paired_effects": effects,
        "retention": retention,
        "transfer": transfer,
        "repair_choice_utility": repair_choice,
        "uncertainty": {
            "unit": "independent_latent_world",
            "paired_method": "cluster_bootstrap",
            "bootstrap_resamples": n_resamples,
        },
        "costs": dict(costs),
        "controller_provenance": dict(controller_provenance),
        "claims": {
            "assisted_repair": "measured separately",
            "predictive_selection": "inherited only from a valid X1 receipt; not re-established here",
            "internalization": decision["supported"],
            "retention": retention["all_passed"],
            "transfer": bool(decision.get("transfer_supported")),
            "agi": False,
        },
        "decision": decision,
        "invalid_conditions": list(protocol.get("invalid_run_conditions", [])),
    }
    body["x6_result_receipt_sha256"] = sha_json(body)
    return body


def validate_x6_result_receipt(
    receipt: Mapping[str, Any], protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = _require_mapping(receipt, "X6 result receipt")
    if receipt.get("schema") != X6_RECEIPT_SCHEMA:
        raise X6X7Error("X6 result schema mismatch")
    _self_hash(receipt, "x6_result_receipt_sha256", "X6 result")
    allowed = {
        "PREREQUISITE_BLOCKED", "INCONCLUSIVE", "NOT_SUPPORTED",
        "SUPPORTED_SCOPED_NO_TRANSFER", "SUPPORTED_SCOPED_WITH_TRANSFER",
    }
    if receipt.get("decision", {}).get("status") not in allowed:
        raise X6X7Error("X6 decision status is invalid")
    if receipt.get("claims", {}).get("agi") is not False:
        raise X6X7Error("X6 receipt cannot claim AGI")
    _validate_controller_provenance(receipt.get("controller_provenance"), "X6.controller_provenance")
    if protocol is not None:
        validate_protocol(protocol)
        if receipt.get("protocol_sha256") != protocol["identity"]["protocol_sha256"]:
            raise X6X7Error("X6 receipt protocol mismatch")
        uncertainty = receipt.get("uncertainty", {})
        if uncertainty.get("unit") != "independent_latent_world" or uncertainty.get("paired_method") != "cluster_bootstrap":
            raise X6X7Error("X6 uncertainty contract mismatch")
        n_resamples = _integer(uncertainty.get("bootstrap_resamples"), "X6 bootstrap_resamples", 1)
        target_intervention = receipt.get("paired_effects", {}).get("target_intervention")
        expected_effects = _x6_effects(
            {"by_arm_cluster": receipt.get("cluster_results", {})},
            target_intervention,
            n_resamples,
        )
        if receipt.get("paired_effects") != expected_effects:
            raise X6X7Error("X6 paired-effect arithmetic mismatch")
        retention_rows = [
            {key: row[key] for key in (
                "arm_id", "seed", "capability_id", "parent_score",
                "child_score", "n_independent_worlds", "score_semantics",
            )}
            for row in receipt.get("retention", {}).get("per_capability", [])
        ]
        expected_retention = _retention_report(
            retention_rows,
            "REPAIR_INTERNALIZATION",
            "RAW_FAILURE_SFT_CONTROL",
            float(protocol["canonical_authority"]["retention_floor_absolute_accuracy"]),
        )
        if receipt.get("retention") != expected_retention:
            raise X6X7Error("X6 retention arithmetic mismatch")
        readiness = {
            "status": "PREREQUISITE_BLOCKED" if receipt["decision"]["status"] == "PREREQUISITE_BLOCKED" else "READY"
        }
        expected_decision = _x6_decision(
            protocol, readiness, expected_effects,
            expected_retention, receipt.get("transfer", {}),
        )
        if receipt.get("decision") != expected_decision:
            raise X6X7Error("X6 decision arithmetic mismatch")
    return {"valid": True, "status": receipt["decision"]["status"]}


def build_x7_readiness(
    protocol: Mapping[str, Any], x6_receipt: Mapping[str, Any], split_bundle: Mapping[str, Any],
    x7_evaluation_commitment: Mapping[str, Any], source_release: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    checks: dict[str, bool] = {}
    try:
        validate_protocol(protocol)
        checks["protocol"] = True
    except X6X7Error:
        checks["protocol"] = False
        blockers.append("X7_PROTOCOL_INVALID")
    try:
        x6 = validate_x6_result_receipt(x6_receipt, protocol)
        checks["x6_result"] = x6["status"] in {"SUPPORTED_SCOPED_NO_TRANSFER", "SUPPORTED_SCOPED_WITH_TRANSFER"}
        if not checks["x6_result"]:
            blockers.append("X6_SCOPED_RESULT_REQUIRED")
    except X6X7Error:
        checks["x6_result"] = False
        blockers.append("X6_RESULT_INVALID")
    try:
        validate_split_bundle(split_bundle)
        if split_bundle.get("protocol_sha256") != protocol.get("identity", {}).get("protocol_sha256"):
            raise X6X7Error("X7 split protocol mismatch")
        checks["fresh_x7_split"] = True
    except X6X7Error:
        checks["fresh_x7_split"] = False
        blockers.append("FRESH_DISJOINT_X7_SPLIT_MISSING")
    try:
        if x7_evaluation_commitment.get("protocol_sha256") != protocol.get("identity", {}).get("protocol_sha256"):
            raise X6X7Error("X7 commitment protocol mismatch")
        if x7_evaluation_commitment.get("outcomes_loaded") is not False:
            raise X6X7Error("X7 outcomes loaded before commitment")
        if x7_evaluation_commitment.get("split_bundle_sha256") != split_bundle.get("bundle_sha256"):
            raise X6X7Error("X7 commitment split mismatch")
        expected_tasks = {
            task_id
            for cluster in split_bundle["splits"]["X7_FOLLOWUP"]["clusters"]
            for task_id in cluster["task_ids"]
        }
        if set(x7_evaluation_commitment.get("task_ids", [])) != expected_tasks:
            raise X6X7Error("X7 commitment task set mismatch")
        expected_retention = {
            task_id
            for cluster in split_bundle["splits"]["X6_RETENTION"]["clusters"]
            for task_id in cluster["task_ids"]
        }
        if set(x7_evaluation_commitment.get("retention_task_ids", [])) != expected_retention:
            raise X6X7Error("X7 commitment retention task set mismatch")
        _self_hash(x7_evaluation_commitment, "evaluation_commitment_sha256", "X7 evaluation commitment")
        checks["outcome_blind_commitment"] = True
    except X6X7Error:
        checks["outcome_blind_commitment"] = False
        blockers.append("X7_OUTCOME_BLIND_COMMITMENT_MISSING")
    try:
        if source_release.get("protocol_sha256") != protocol.get("identity", {}).get("protocol_sha256"):
            raise X6X7Error("X7 source release protocol mismatch")
        identity = protocol.get("identity", {})
        if source_release.get("base_revision") != identity.get("base_revision"):
            raise X6X7Error("X7 source release base mismatch")
        if not isinstance(source_release.get("release_revision"), str) or not HEX40.fullmatch(source_release["release_revision"]):
            raise X6X7Error("X7 source release revision is malformed")
        if source_closure_sha256(identity.get("base_revision", ""), source_release.get("files", [])) != identity.get("source_closure_sha256"):
            raise X6X7Error("X7 source release closure mismatch")
        _self_hash(source_release, "release_manifest_sha256", "X7 source release")
        checks["source_release"] = True
    except X6X7Error:
        checks["source_release"] = False
        blockers.append("X7_SOURCE_RELEASE_MISSING")
    threshold_status = protocol.get("decision_thresholds", {}).get("authority_status")
    checks["x7_thresholds_authorized"] = threshold_status == "AUTHORIZED"
    if threshold_status != "AUTHORIZED":
        blockers.append("X7_DEPENDENCE_THRESHOLDS_UNRESOLVED")
    return _readiness_body(
        X7_READINESS_SCHEMA, "readiness", protocol, checks, blockers,
        {
            "x6_result_receipt_sha256": x6_receipt.get("x6_result_receipt_sha256"),
            "paired_seeds": x6_receipt.get("seeds"),
            "split_bundle_sha256": split_bundle.get("bundle_sha256"),
            "x7_evaluation_commitment_sha256": x7_evaluation_commitment.get("evaluation_commitment_sha256"),
            "source_release_manifest_sha256": source_release.get("release_manifest_sha256"),
        },
    )


def _x7_decision(
    protocol: Mapping[str, Any], readiness: Mapping[str, Any], effects: Mapping[str, Any],
    retention: Mapping[str, Any], unrelated: Mapping[str, Any], transfer: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = protocol.get("decision_thresholds", {})
    required = {
        "x7_min_unaided_retention_vs_parent", "x7_min_unaided_gain_vs_control",
        "x7_min_target_lift_reduction_vs_parent", "x7_min_target_lift_reduction_vs_control",
        "x7_max_unrelated_lift_change", "minimum_independent_worlds",
        "minimum_paired_seeds", "x7_min_transfer_gain_by_axis", "require_ci_above_zero",
    }
    if readiness.get("status") != "READY" or thresholds.get("authority_status") != "AUTHORIZED" or any(thresholds.get(key) is None for key in required):
        return {"status": "PREREQUISITE_BLOCKED", "supported": False}
    by_seed = effects["by_seed"]
    if len(by_seed) < int(thresholds["minimum_paired_seeds"]):
        return {"status": "INCONCLUSIVE", "supported": False, "failed_gates": ["paired-seed sample size"]}
    if any(result["unaided_vs_parent"].get("n_clusters", 0) < int(thresholds["minimum_independent_worlds"]) for result in by_seed.values()):
        return {"status": "INCONCLUSIVE", "supported": False, "failed_gates": ["independent-world sample size"]}
    if any(row["n_independent_worlds"] < int(thresholds["minimum_independent_worlds"]) for row in retention["per_capability"]):
        return {"status": "INCONCLUSIVE", "supported": False, "failed_gates": ["retention independent-world sample size"]}
    transfer_gates = {
        axis: all(
            _gate_from_effect(seed_result, float(thresholds["x7_min_transfer_gain_by_axis"][axis]), True)
            for seed_result in report["by_seed"].values()
        )
        for axis, report in transfer.items()
    }
    gates = {
        "unaided_retained_vs_parent": all(_gate_from_effect(result["unaided_vs_parent"], float(thresholds["x7_min_unaided_retention_vs_parent"]), bool(thresholds["require_ci_above_zero"])) for result in by_seed.values()),
        "unaided_gain_vs_control": all(_gate_from_effect(result["unaided_vs_control"], float(thresholds["x7_min_unaided_gain_vs_control"]), bool(thresholds["require_ci_above_zero"])) for result in by_seed.values()),
        "target_lift_reduced_vs_parent": all(_gate_from_effect(result["lift_reduction_vs_parent"], float(thresholds["x7_min_target_lift_reduction_vs_parent"]), bool(thresholds["require_ci_above_zero"])) for result in by_seed.values()),
        "target_lift_reduced_vs_control": all(_gate_from_effect(result["lift_reduction_vs_control"], float(thresholds["x7_min_target_lift_reduction_vs_control"]), bool(thresholds["require_ci_above_zero"])) for result in by_seed.values()),
        "retention": retention["all_passed"],
        "unrelated_interventions_stable": all(abs(float(value)) <= float(thresholds["x7_max_unrelated_lift_change"]) for value in unrelated.values()),
        "fresh_transfer": all(transfer_gates.values()),
    }
    if all(gates.values()):
        return {"status": "DEPENDENCE_FELL", "supported": True, "gates": gates, "transfer_gates": transfer_gates, "claim_ceiling": "reduced target-intervention dependence with retained unaided capability in the frozen X7 regime"}
    return {"status": "DEPENDENCE_DID_NOT_FALL", "supported": False, "gates": gates}


def build_x7_receipt(
    protocol: Mapping[str, Any], x7_readiness: Mapping[str, Any], x6_receipt: Mapping[str, Any],
    split_bundle: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
    intervention_ids: Sequence[str, Any], target_intervention: str,
    retention_rows: Sequence[Mapping[str, Any]], costs: Mapping[str, Any],
    controller_provenance: Mapping[str, Any], n_resamples: int = 2000,
) -> dict[str, Any]:
    validate_x6_result_receipt(x6_receipt, protocol)
    if x7_readiness.get("schema") != X7_READINESS_SCHEMA:
        raise X6X7Error("X7 readiness schema mismatch")
    _self_hash(x7_readiness, "readiness_sha256", "X7 readiness")
    if x7_readiness.get("protocol_sha256") != protocol["identity"]["protocol_sha256"]:
        raise X6X7Error("X7 readiness protocol mismatch")
    if not intervention_ids or len(intervention_ids) != len(set(intervention_ids)) or "NO_CHANGE" not in intervention_ids:
        raise X6X7Error("frozen X7 intervention set is invalid")
    if target_intervention not in intervention_ids:
        raise X6X7Error("X7 target intervention is not frozen")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise X6X7Error("bootstrap resamples must be positive")
    if protocol.get("decision_thresholds", {}).get("authority_status") == "AUTHORIZED" and n_resamples != protocol["decision_thresholds"].get("bootstrap_resamples"):
        raise X6X7Error("bootstrap resample count differs from the frozen protocol")
    validate_split_bundle(split_bundle)
    if split_bundle.get("protocol_sha256") != protocol["identity"]["protocol_sha256"]:
        raise X6X7Error("X7 result split protocol mismatch")
    if x7_readiness.get("evidence", {}).get("split_bundle_sha256") != split_bundle.get("bundle_sha256"):
        raise X6X7Error("X7 readiness does not bind this split bundle")
    expected_seeds = {int(seed) for seed in x7_readiness.get("evidence", {}).get("paired_seeds", [])}
    if {int(row["seed"]) for row in observations} != expected_seeds:
        raise X6X7Error("X7 evaluation seeds do not match the accepted X6 run")
    if x6_receipt.get("protocol_sha256") != protocol["identity"]["protocol_sha256"]:
        raise X6X7Error("X6/X7 protocol mismatch")
    _validate_observation_split(observations, split_bundle, "X7_FOLLOWUP")
    summaries = summarize_evaluation_observations(observations, intervention_ids)
    effects = _x7_effects(summaries, target_intervention, n_resamples)
    retention = _retention_report(
        retention_rows, "REPAIR_INTERNALIZATION", "RAW_FAILURE_SFT_CONTROL",
        float(protocol["canonical_authority"]["retention_floor_absolute_accuracy"]),
    )
    transfer = _transfer_report(
        summaries, observations, "REPAIR_INTERNALIZATION", "RAW_FAILURE_SFT_CONTROL",
        target_intervention, n_resamples,
    )
    unrelated = {}
    for intervention_id in intervention_ids:
        if intervention_id in {target_intervention, "NO_CHANGE"}:
            continue
        deltas = []
        for seed in sorted(set(summaries["by_arm_cluster"]["REPAIR_INTERNALIZATION"]) & set(summaries["by_arm_cluster"]["PARENT"])):
            treatment = summaries["by_arm_cluster"]["REPAIR_INTERNALIZATION"][seed]
            parent = summaries["by_arm_cluster"]["PARENT"][seed]
            for cluster in sorted(set(treatment) & set(parent)):
                deltas.append(
                    float(treatment[cluster][f"lift:{intervention_id}"])
                    - float(parent[cluster][f"lift:{intervention_id}"])
                )
        unrelated[intervention_id] = _mean(deltas)
    if not isinstance(costs, Mapping) or set(costs) != {"evaluation", "intervention", "checkpoint"}:
        raise X6X7Error("X7 costs must separately report evaluation, intervention, and checkpoint costs")
    controller_provenance = _validate_controller_provenance(
        controller_provenance, "X7.controller_provenance",
    )
    decision = _x7_decision(protocol, x7_readiness, effects, retention, unrelated, transfer)
    body = {
        "schema": X7_RECEIPT_SCHEMA,
        "phase": "FOLLOWUP_COMMITTED" if decision["status"] != "PREREQUISITE_BLOCKED" else "FOLLOWUP_PREREQUISITE_BLOCKED",
        "protocol_sha256": protocol["identity"]["protocol_sha256"],
        "x6_result_receipt_sha256": x6_receipt["x6_result_receipt_sha256"],
        "x7_readiness_sha256": x7_readiness.get("readiness_sha256"),
        "split_bundle_sha256": split_bundle["bundle_sha256"],
        "assistance_removed": "all external target-intervention support absent from unaided runs",
        "reduced_assistance_scope": sorted(set(intervention_ids) - {target_intervention, "NO_CHANGE"}),
        "raw_outcomes": summaries["by_arm_seed"],
        "cluster_results": summaries["by_arm_cluster"],
        "paired_effects": effects,
        "retention": retention,
        "transfer": transfer,
        "unrelated_intervention_lift_change": unrelated,
        "uncertainty": {
            "unit": "independent_latent_world",
            "paired_method": "cluster_bootstrap",
            "bootstrap_resamples": n_resamples,
        },
        "memorization_diagnostic": {
            "training_templates_only": (
                not decision.get("gates", {})["fresh_transfer"]
                if "gates" in decision else None
            ),
            "fresh_worlds": True,
            "fresh_task_instances": True,
            "evaluation_labels_used_for_training_or_selection": False,
        },
        "costs": dict(costs),
        "controller_provenance": dict(controller_provenance),
        "decision": decision,
        "agi": False,
    }
    body["x7_followup_receipt_sha256"] = sha_json(body)
    return body


def validate_x7_receipt(
    receipt: Mapping[str, Any], protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = _require_mapping(receipt, "X7 receipt")
    if receipt.get("schema") != X7_RECEIPT_SCHEMA:
        raise X6X7Error("X7 receipt schema mismatch")
    _self_hash(receipt, "x7_followup_receipt_sha256", "X7 receipt")
    allowed = {"PREREQUISITE_BLOCKED", "INCONCLUSIVE", "DEPENDENCE_DID_NOT_FALL", "DEPENDENCE_FELL"}
    if receipt.get("decision", {}).get("status") not in allowed or receipt.get("agi") is not False:
        raise X6X7Error("X7 decision or claim boundary is invalid")
    _validate_controller_provenance(receipt.get("controller_provenance"), "X7.controller_provenance")
    if protocol is not None:
        validate_protocol(protocol)
        if receipt.get("protocol_sha256") != protocol["identity"]["protocol_sha256"]:
            raise X6X7Error("X7 receipt protocol mismatch")
        n_resamples = _integer(receipt.get("uncertainty", {}).get("bootstrap_resamples"), "X7 bootstrap_resamples", 1)
        target_intervention = receipt.get("paired_effects", {}).get("target_intervention")
        expected_effects = _x7_effects(
            {"by_arm_cluster": receipt.get("cluster_results", {})},
            target_intervention,
            n_resamples,
        )
        if receipt.get("paired_effects") != expected_effects:
            raise X6X7Error("X7 paired-effect arithmetic mismatch")
        retention_rows = [
            {key: row[key] for key in (
                "arm_id", "seed", "capability_id", "parent_score",
                "child_score", "n_independent_worlds", "score_semantics",
            )}
            for row in receipt.get("retention", {}).get("per_capability", [])
        ]
        expected_retention = _retention_report(
            retention_rows,
            "REPAIR_INTERNALIZATION",
            "RAW_FAILURE_SFT_CONTROL",
            float(protocol["canonical_authority"]["retention_floor_absolute_accuracy"]),
        )
        if receipt.get("retention") != expected_retention:
            raise X6X7Error("X7 retention arithmetic mismatch")
        readiness = {
            "status": "PREREQUISITE_BLOCKED" if receipt["decision"]["status"] == "PREREQUISITE_BLOCKED" else "READY"
        }
        expected_decision = _x7_decision(
            protocol,
            readiness,
            expected_effects,
            expected_retention,
            receipt.get("unrelated_intervention_lift_change", {}),
            receipt.get("transfer", {}),
        )
        if receipt.get("decision") != expected_decision:
            raise X6X7Error("X7 decision arithmetic mismatch")
    return {"valid": True, "status": receipt["decision"]["status"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-protocol")
    validate.add_argument("--protocol", required=True)
    validate.add_argument("--repo-root")
    validate.add_argument("--check-source-closure", action="store_true")
    audit = subparsers.add_parser("audit-x1")
    audit.add_argument("--receipt", required=True)
    subparsers.add_parser("constants")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-protocol":
        result = validate_protocol(load_json(args.protocol), args.repo_root, args.check_source_closure)
    elif args.command == "audit-x1":
        receipt = load_json(args.receipt)
        result = audit_x1_entry(receipt, sha_file(args.receipt))
    else:
        result = {
            name: value for name, value in globals().items()
            if name.endswith("_SCHEMA") and isinstance(value, str)
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("valid", True) and result.get("status", "PASS") not in {"BLOCKED", "PREREQUISITE_BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
