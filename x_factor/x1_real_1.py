"""Source-bound, model-free execution contract for An-Ra X1-REAL-1.

The module coordinates externally executed checkpoint observations and
interventions. It never imports Torch, an An-Ra runtime, or a checkpoint.
The custody phases are basis qualification, a committed prediction receipt,
and a parent-bound evaluator reveal receipt. All promotion decisions are made
by pure receipt and metric functions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from datetime import datetime, timezone
import json
import math
import os
import platform
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_SCHEMA = "anra-x1-real-1-protocol/v1"
PREDICTION_SCHEMA = "anra-x1-real-1-prediction/v1"
REVEAL_SCHEMA = "anra-x1-real-1-reveal/v1"
ANALYSIS_SCHEMA = "anra-x1-real-1-analysis/v1"
SPLIT_SCHEMA = "anra-x1-real-1-split/v1"
SUBJECT_SCHEMA = "anra-x1-real-1-subject/v1"
BASIS_SCHEMA = "anra-x1-real-1-basis-qualification/v1"
REPLICATION_SCHEMA = "anra-x1-real-1-replication/v1"
RELEASE_SCHEMA = "anra-x1-real-1-release/v1"
PHASE_COMMIT_SCHEMA = "anra-x1-real-1-phase-commit/v1"
VERIFIER_SCHEMA = "anra-x1-real-1-verifier/v1"
PREDICTOR_TRAINING_SCHEMA = "anra-x1-real-1-predictor-training/v1"
PREFLIGHT_SCHEMA = "anra-x1-real-1-preflight/v1"
EXECUTION_PLAN_SCHEMA = "anra-x1-real-1-execution-plan/v1"
INTAKE_SCHEMA = "anra-x1-real-1-candidate-intake/v1"
CLI_ERROR_SCHEMA = "anra-x1-real-1-cli-error/v1"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
OPAQUE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")

FORBIDDEN_POLICY_KEYS = frozenset({
    "gold", "gold_answer", "gold_code", "gold_sha256", "answer", "answer_key",
    "target", "labels", "correct", "correctness", "correctness_label", "is_correct",
    "correctness_score", "label", "outcome", "outcomes", "intervention_outcome",
    "intervention_outcomes", "effect", "raw_output", "raw_outputs", "verifier",
    "verifier_receipt", "oracle", "oracle_metadata", "family", "family_id",
    "task_family", "cluster", "cluster_id", "cluster_ids", "source_id", "source_ids",
    "world_id", "latent_world", "hidden_label", "required_factors", "template_id",
})

FORBIDDEN_INTERVENTION_INPUTS = frozenset({
    "gold", "gold_answer", "gold_code", "answer", "correctness", "correctness_label",
    "is_correct", "hidden_failure_category", "future_intervention_outcome",
    "intervention_outcome", "oracle_selected_relevant_fact", "evaluator_latent_variables",
    "family", "family_id", "task_family", "cluster_id", "source_id", "world_id",
})

ALLOWED_INTERVENTION_INPUTS = frozenset({
    "visible_context", "visible_query", "visible_candidates", "visible_format",
    "pre_intervention_observations",
})

ALLOWED_OBSERVATION_FIELDS = frozenset({
    "task_id", "context", "query", "visible_candidates", "format", "features",
    "observation_hash",
})
ALLOWED_FEATURE_FIELDS = frozenset({
    "confidence", "entropy", "margin", "output_len", "distinct_ratio", "prompt_tokens",
})

SUBJECT_FIELDS = frozenset({
    "schema", "subject_id", "checkpoint_path", "checkpoint_file_sha256",
    "parameter_sha256", "model_config_sha256", "tokenizer_artifact_sha256",
    "tokenizer_identity_sha256", "runtime_source_revision", "source_commit",
    "training_lineage", "stage", "global_step", "research_subject", "eligibility",
    "registry_entry_sha256", "identity_attestation_path", "identity_attestation_sha256",
})
SUBJECT_ELIGIBILITY_FIELDS = frozenset({
    "state", "readiness_receipt_path", "readiness_receipt_sha256",
    "cohort_plan_path", "cohort_plan_sha256",
    "cross_checkpoint_plan_path", "cross_checkpoint_plan_sha256",
})
SUBJECT_ATTESTATION_FIELDS = frozenset({
    "schema", "subject_id", "checkpoint_file_sha256", "parameter_sha256",
    "model_config_sha256", "tokenizer_artifact_sha256", "tokenizer_identity_sha256",
    "runtime_source_revision", "source_commit", "training_lineage", "stage",
    "global_step", "registry_entry_sha256", "registry_sha256",
})

BASIS_DEFAULTS = {
    "min_oracle_coverage": 0.15,
    "min_active_prevalence": 0.03,
    "max_active_prevalence": 0.90,
    "max_pairwise_redundancy": 0.40,
    "min_signature_entropy_bits": 0.50,
    "min_unique_signatures": 4,
    "min_failures": 40,
    "explicit_failure_required": True,
    "min_independent_clusters": 20,
    "max_null_p_value": 0.05,
    "max_no_change_repair_rate": 0.05,
    "max_control_excess_rate": 0.10,
    "null_resamples": 200,
    "null_seed": 81818,
}

PREDICTION_BASELINES = (
    "ALWAYS_NEGATIVE", "PREVALENCE_ONLY", "BEST_FIXED_DEV", "COST_AWARE_FIXED",
    "SURFACE_SHORTCUT", "SPARSITY_RANDOM",
)

REQUIRED_SOURCE_PATHS = frozenset({
    "AN_RA_PROGRAM.md",
    "pyproject.toml",
    "x_factor/REAL_MODEL_CAUSAL_SPEC.md",
    "x_factor/SPEC.md",
    "x_factor/__init__.py",
    "x_factor/ladder.py",
    "x_factor/contracts.py",
    "x_factor/execution_policy.py",
    "x_factor/ibq.py",
    "x_factor/ibq_v2.py",
    "x_factor/observed.py",
    "x_factor/registry/checkpoints.json",
    "x_factor/tests/test_x1_real_1.py",
    "x_factor/tests/test_x1_external_runner.py",
    "x_factor/tests/test_x1_cuda_canary.py",
    "x_factor/x1_real_1.py",
    "x_factor/x1_external_runner.py",
    "x_factor/x1_cuda_canary.py",
})

BASIS_BINDING_FIELDS = frozenset({
    "basis_cohort_id", "basis_split_manifest_sha256", "basis_source_artifact_path",
    "basis_source_artifact_sha256", "basis_matrix_sha256", "basis_phase_artifact_path",
    "basis_phase_commit_sha256",
})

BASIS_CHECK_NAMES = frozenset({
    "registry_valid", "matrix_rectangular", "min_failures", "explicit_baseline_failure_cohort", "independent_cluster_identity",
    "independent_source_identity", "oracle_coverage", "no_degenerate_active_probes",
    "no_universal_solver", "response_signature_variation", "unique_signatures",
    "bounded_pairwise_redundancy", "no_change_control_null", "matched_controls_null",
    "sparsity_matched_diversity", "binding_complete",
})
FROZEN_COHORT_PLAN_HASHES = {
    "BASIS_QUALIFICATION": "21c186b22edaf60abd0387fa2b2ed29104b8a13f953be1250a58bc7b15663336",
    "DEV_COHORT_V1": "f69d9f14fc97dba864c4e7d29d1151ffb141d7a9a11b86d04c427b856afcbb56",
    "PRIMARY_EVAL": "3aa54d2ec406cd0d7b7576ea991833c03658ba58b8ab17a289eab77b97ac55bb",
    "INDEPENDENT_TASK_EVAL": "cf4123533170fbe535c70c385d184ec76064b732a2ce9e892f1686c8a6744213",
    "CHECKPOINT_REPLICATION": "6459c5107121766a4ac65db8b6d7f83b1b51bfd5fde368089a6ae78956948e29",
}


class _CLIArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise X1EvidenceError(f"CLI usage error: {message}")


class X1ContractError(ValueError):
    pass


class X1ProtocolError(X1ContractError):
    pass


class X1ChronologyError(X1ContractError):
    pass


class X1EvidenceError(X1ContractError):
    pass


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted(_plain(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _without(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    body = copy.deepcopy(dict(mapping))
    for key in keys:
        body.pop(key, None)
    return body


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_keys(item)


def _assert_no_policy_leak(value: Any) -> None:
    hits = sorted({key for key in _walk_keys(value) if key.lower() in FORBIDDEN_POLICY_KEYS})
    if hits:
        raise X1EvidenceError(f"policy-visible record contains forbidden fields: {hits}")


def _assert_exact_keys(value: Mapping[str, Any], required: set[str], optional: set[str] = frozenset()) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - set(optional))
    if missing or unknown:
        raise X1EvidenceError(f"schema keys invalid; missing={missing}, unknown={unknown}")


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise X1EvidenceError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise X1EvidenceError(f"{name} must be a non-empty string")
    lowered = value.strip().lower()
    if any(token in lowered for token in ("unverified", "uncomputed", "unknown", "unfilled", "pending", "placeholder")):
        raise X1EvidenceError(f"{name} contains a placeholder value")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise X1EvidenceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise X1EvidenceError(f"{name} must be finite")
    return result


def _probability(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise X1EvidenceError(f"{name} must be in [0, 1]")
    return result


def _protocol_hash(protocol: Mapping[str, Any]) -> str:
    body = _without(protocol, )
    identity = body.get("identity")
    if isinstance(identity, Mapping):
        identity_copy = dict(identity)
        identity_copy.pop("protocol_sha256", None)
        body["identity"] = identity_copy
    return _sha_json(body)


def source_closure_sha256(
    repo_root: str | Path,
    source_files: Sequence[str],
    base_revision: str | None = None,
) -> str:
    root = Path(repo_root).resolve()
    normalized = sorted({str(Path(item).as_posix()) for item in source_files})
    if not normalized:
        raise X1EvidenceError("source closure cannot be empty")
    records = []
    for relative in normalized:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise X1EvidenceError(f"source path escapes repository: {relative}")
        path = (root / candidate).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise X1EvidenceError(f"source path escapes repository: {relative}") from exc
        if not path.is_file():
            raise X1EvidenceError(f"source file missing: {relative}")
        records.append({"path": candidate.as_posix(), "sha256": _sha_file(path)})
    return _sha_json({"base_revision": base_revision or "", "files": records})


def validate_source_closure(
    protocol: Mapping[str, Any], repo_root: str | Path | None = None
) -> dict[str, Any]:
    identity = protocol.get("identity")
    if not isinstance(identity, Mapping):
        return {"valid": False, "errors": ["protocol identity is missing"]}
    files = identity.get("source_files")
    if not isinstance(files, list) or not files:
        return {"valid": False, "errors": ["identity.source_files is missing"]}
    errors: list[str] = []
    declared_paths = {str(item.get("path")) for item in files if isinstance(item, Mapping)}
    if declared_paths != set(REQUIRED_SOURCE_PATHS):
        errors.append("source closure path allowlist mismatch")
    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    observed = []
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            errors.append("each source file must contain exactly path and sha256")
            continue
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str) or not HEX64.fullmatch(expected):
            errors.append(f"invalid source file identity: {relative!r}")
            continue
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
        actual = _sha_file(path)
        observed.append({"path": Path(relative).as_posix(), "sha256": actual})
        if actual != expected:
            errors.append(f"source hash mismatch: {relative}")
    expected_closure = identity.get("source_closure_sha256")
    if not isinstance(expected_closure, str) or not HEX64.fullmatch(expected_closure):
        errors.append("identity.source_closure_sha256 is invalid")
    else:
        base_revision = identity.get("base_revision", "")
        calculated = _sha_json({
            "base_revision": base_revision,
            "files": sorted(observed, key=lambda item: item["path"]),
        })
        if calculated != expected_closure:
            errors.append("source closure digest mismatch")
    return {"valid": not errors, "errors": errors, "files": observed}


def validate_release_manifest(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    required = {
        "schema", "protocol_sha256", "source_closure_sha256", "base_revision",
        "source_tree_status", "source_files", "protocol_artifact_path",
        "protocol_artifact_sha256", "release_artifact_path",
        "release_artifact_sha256", "release_revision",
        "attestation_path", "attestation_sha256", "custodian",
        "release_manifest_sha256",
    }
    errors: list[str] = []
    try:
        _assert_exact_keys(manifest, required)
    except X1EvidenceError as exc:
        errors.append(str(exc))
    if manifest.get("schema") != RELEASE_SCHEMA:
        errors.append("release manifest schema mismatch")
    if manifest.get("protocol_sha256") != protocol.get("identity", {}).get("protocol_sha256"):
        errors.append("release protocol identity mismatch")
    if manifest.get("source_closure_sha256") != protocol.get("identity", {}).get("source_closure_sha256"):
        errors.append("release source closure mismatch")
    if manifest.get("base_revision") != protocol.get("identity", {}).get("base_revision"):
        errors.append("release base revision mismatch")
    release_revision = manifest.get("release_revision")
    if not isinstance(release_revision, str) or not HEX40.fullmatch(release_revision):
        errors.append("release revision is malformed")
    if manifest.get("source_tree_status") != "CLEAN":
        errors.append("release source tree is not clean")
    source_root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    if (source_root / ".git").exists():
        try:
            git_status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=str(source_root),
                check=True, capture_output=True, text=True,
            ).stdout
            if git_status.strip():
                errors.append("release repository has uncommitted changes")
            git_head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(source_root),
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            if git_head != release_revision:
                errors.append("release revision does not match repository HEAD")
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"release repository cleanliness could not be verified: {exc}")
    if manifest.get("source_files") != protocol.get("identity", {}).get("source_files"):
        errors.append("release source file inventory mismatch")
    for key in ("protocol_artifact_sha256", "release_artifact_sha256", "attestation_sha256"):
        try:
            _hash(manifest.get(key), f"release.{key}")
        except X1EvidenceError as exc:
            errors.append(str(exc))
    attestation_path_value = manifest.get("attestation_path")
    if not isinstance(attestation_path_value, str) or not attestation_path_value:
        errors.append("release attestation path is missing")
    else:
        attestation_path = Path(attestation_path_value)
        if not attestation_path.is_absolute():
            attestation_path = Path(repo_root or Path.cwd()) / attestation_path
        if not attestation_path.is_file():
            errors.append("release attestation file is missing")
        elif _sha_file(attestation_path) != manifest.get("attestation_sha256"):
            errors.append("release attestation file hash mismatch")
        else:
            try:
                attestation = _load_json(attestation_path)
                _assert_exact_keys(attestation, {
                    "schema", "protocol_sha256", "source_closure_sha256",
                    "custodian", "source_tree_status",
                })
                if attestation.get("schema") != "anra-x1-real-1-release-attestation/v1":
                    errors.append("release attestation schema mismatch")
                for key in ("protocol_sha256", "source_closure_sha256", "custodian", "source_tree_status"):
                    if attestation.get(key) != manifest.get(key):
                        errors.append(f"release attestation field mismatch: {key}")
            except X1EvidenceError as exc:
                errors.append(str(exc))
    try:
        _nonempty(manifest.get("custodian"), "release.custodian")
    except X1EvidenceError as exc:
        errors.append(str(exc))
    protocol_artifact_path = manifest.get("protocol_artifact_path")
    if not isinstance(protocol_artifact_path, str) or not protocol_artifact_path:
        errors.append("protocol artifact path is missing")
    else:
        candidate = Path(protocol_artifact_path)
        if not candidate.is_absolute():
            candidate = Path(repo_root or Path.cwd()) / candidate
        if not candidate.is_file():
            errors.append(f"protocol artifact is missing: {candidate}")
        elif _sha_file(candidate) != manifest.get("protocol_artifact_sha256"):
            errors.append("protocol artifact hash mismatch")
        else:
            try:
                protocol_artifact = _load_json(candidate)
                if not isinstance(protocol_artifact, Mapping) or _protocol_hash(protocol_artifact) != protocol.get("identity", {}).get("protocol_sha256"):
                    errors.append("release protocol artifact is not the supplied frozen protocol")
            except X1EvidenceError as exc:
                errors.append(str(exc))
    artifact_path = manifest.get("release_artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        errors.append("release artifact path is missing")
    else:
        candidate = Path(artifact_path)
        if not candidate.is_absolute():
            candidate = Path(repo_root or Path.cwd()) / candidate
        if not candidate.is_file():
            errors.append(f"release artifact is missing: {candidate}")
        elif _sha_file(candidate) != manifest.get("release_artifact_sha256"):
            errors.append("release artifact hash mismatch")
    identity = manifest.get("release_manifest_sha256")
    if not isinstance(identity, str) or not HEX64.fullmatch(identity):
        errors.append("release manifest hash is malformed")
    elif _sha_json(_without(manifest, "release_manifest_sha256")) != identity:
        errors.append("release manifest hash mismatch")
    return {"valid": not errors, "errors": errors}


def _registry_items(registry: Any) -> list[dict[str, Any]]:
    if isinstance(registry, list):
        if any(not isinstance(item, Mapping) for item in registry):
            raise X1EvidenceError("intervention registry contains a malformed record")
        return [dict(item) for item in registry]
    if isinstance(registry, Mapping):
        items = registry.get("interventions", registry.get("probes", []))
        if isinstance(items, list):
            if any(not isinstance(item, Mapping) for item in items):
                raise X1EvidenceError("intervention registry contains a malformed record")
            return [dict(item) for item in items]
    raise X1EvidenceError("intervention registry must be a list or contain interventions")


def intervention_registry_sha256(registry: Any) -> str:
    items = _registry_items(registry)
    return _sha_json(items)


def validate_intervention_registry(registry: Any) -> dict[str, Any]:
    errors: list[str] = []
    try:
        items = _registry_items(registry)
    except X1EvidenceError as exc:
        return {"valid": False, "errors": [str(exc)], "registry_sha256": None}
    ids: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    required = {
        "id", "version", "family", "role", "assistance", "cost",
        "information_class", "legality_inputs", "transformation", "transformation_id",
        "control_pair", "preserves_task_answer_information", "semantics",
    }
    for item in items:
        missing = required - set(item)
        unknown = set(item) - required
        if missing or unknown:
            errors.append(f"intervention schema invalid; missing={sorted(missing)}, unknown={sorted(unknown)}")
            continue
        intervention_id = item["id"]
        if not isinstance(intervention_id, str) or not intervention_id:
            errors.append("intervention id must be a non-empty string")
            continue
        if intervention_id in ids:
            errors.append(f"duplicate intervention id: {intervention_id}")
            continue
        ids.append(intervention_id)
        by_id[intervention_id] = item
        if item["role"] not in {"NULL_CONTROL", "DIAGNOSTIC", "REPAIR", "ASSISTANCE"}:
            errors.append(f"unknown intervention role: {intervention_id}")
        if item["assistance"] not in {"A0", "A1", "A2", "A3", "A4"}:
            errors.append(f"unknown assistance class: {intervention_id}")
        if item["information_class"] != "INFORMATION_PRESERVING":
            errors.append(f"non-preserving intervention is not eligible: {intervention_id}")
        if item["preserves_task_answer_information"] is not True:
            errors.append(f"answer information preservation not declared: {intervention_id}")
        if not isinstance(item["cost"], int) or isinstance(item["cost"], bool) or item["cost"] < 0:
            errors.append(f"invalid intervention cost: {intervention_id}")
        inputs = item["legality_inputs"]
        if not isinstance(inputs, list) or not inputs:
            errors.append(f"legality inputs missing: {intervention_id}")
        else:
            bad_inputs = set(inputs) & FORBIDDEN_INTERVENTION_INPUTS
            unknown_inputs = set(inputs) - ALLOWED_INTERVENTION_INPUTS
            if bad_inputs:
                errors.append(f"forbidden legality input {intervention_id}: {sorted(bad_inputs)}")
            if unknown_inputs:
                errors.append(f"unknown legality input {intervention_id}: {sorted(unknown_inputs)}")
        for key in ("transformation", "transformation_id", "semantics"):
            if not isinstance(item[key], str) or not item[key].strip():
                errors.append(f"missing {key}: {intervention_id}")
    if "NO_CHANGE" not in by_id:
        errors.append("NO_CHANGE control is required")
    else:
        no_change = by_id["NO_CHANGE"]
        if no_change["role"] != "NULL_CONTROL" or no_change["cost"] != 0:
            errors.append("NO_CHANGE must be a zero-cost NULL_CONTROL")
        if no_change["transformation_id"] != "identity":
            errors.append("NO_CHANGE transformation_id must be identity")
    null_controls = {key for key, value in by_id.items() if value["role"] == "NULL_CONTROL"}
    if "NULL_REFORMAT" not in null_controls:
        errors.append("NULL_REFORMAT matched control is required")
    for intervention_id, item in by_id.items():
        pair = item["control_pair"]
        if intervention_id == "NO_CHANGE":
            if pair != "NULL_REFORMAT":
                errors.append("NO_CHANGE must point to NULL_REFORMAT")
        elif not isinstance(pair, str) or pair not in null_controls:
            errors.append(f"intervention has no valid matched control: {intervention_id}")
    if not any(item["role"] != "NULL_CONTROL" for item in items):
        errors.append("at least one active intervention is required")
    return {
        "valid": not errors,
        "errors": errors,
        "registry_sha256": intervention_registry_sha256(items) if not errors else None,
        "intervention_ids": ids,
    }


def _basis_rows(artifact: Any) -> tuple[list[dict[str, Any]], list[str] | None]:
    if isinstance(artifact, Mapping) and "rows" in artifact:
        raw_rows = artifact["rows"]
    elif isinstance(artifact, Mapping) and "outcome_matrix" in artifact:
        raw_rows = []
        matrix = artifact["outcome_matrix"]
        if isinstance(matrix, Mapping):
            for task_id, row in matrix.items():
                if isinstance(row, Mapping):
                    raw_rows.append({"task_id": task_id, "outcomes": row})
                else:
                    raw_rows.append({"task_id": task_id, "outcomes": row})
        else:
            raw_rows = matrix
    elif isinstance(artifact, list):
        raw_rows = artifact
    else:
        raise X1EvidenceError("basis artifact must contain rows or outcome_matrix")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise X1EvidenceError("basis artifact has no rows")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        if isinstance(raw, Mapping) and "outcomes" in raw:
            row = dict(raw)
        elif isinstance(raw, Mapping):
            row = {"task_id": str(raw.get("task_id", f"row-{index:04d}")), "outcomes": raw}
        elif isinstance(raw, (list, tuple)):
            row = {"task_id": f"row-{index:04d}", "outcomes": list(raw)}
        else:
            raise X1EvidenceError(f"invalid basis row {index}")
        if "task_id" not in row:
            row["task_id"] = f"row-{index:04d}"
        row["task_id"] = str(row["task_id"])
        if not isinstance(row["outcomes"], Mapping):
            raise X1EvidenceError(f"basis row outcomes must be a mapping: {row['task_id']}")
        normalized: dict[str, Any] = {}
        for intervention_id, value in row["outcomes"].items():
            if isinstance(value, Mapping):
                if "repaired" not in value:
                    raise X1EvidenceError(f"outcome row lacks repaired flag: {row['task_id']}")
                value = value["repaired"]
            if not isinstance(value, (bool, int)) or isinstance(value, float) or int(value) not in (0, 1):
                raise X1EvidenceError(f"basis outcomes must be boolean or 0/1: {row['task_id']}/{intervention_id}")
            normalized[str(intervention_id)] = bool(value)
        row["outcomes"] = normalized
        rows.append(row)
    intervention_ids = None
    if isinstance(artifact, Mapping):
        candidate = artifact.get("intervention_ids")
        if isinstance(candidate, list):
            intervention_ids = [str(item) for item in candidate]
    return rows, intervention_ids


def _matrix_from_artifact(artifact: Any, intervention_ids: Sequence[str]) -> tuple[list[list[int]], list[dict[str, Any]]]:
    rows, declared_ids = _basis_rows(artifact)
    if declared_ids is not None and list(intervention_ids) != declared_ids:
        raise X1EvidenceError("basis intervention order does not match protocol")
    matrix: list[list[int]] = []
    normalized_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        task_id = row["task_id"]
        if task_id in seen:
            raise X1EvidenceError(f"duplicate basis task: {task_id}")
        seen.add(task_id)
        missing = set(intervention_ids) - set(row["outcomes"])
        extra = set(row["outcomes"]) - set(intervention_ids)
        if missing or extra:
            raise X1EvidenceError(f"basis coverage mismatch for {task_id}: missing={sorted(missing)}, extra={sorted(extra)}")
        values = []
        for intervention_id in intervention_ids:
            value = row["outcomes"][intervention_id]
            if not isinstance(value, bool):
                raise X1EvidenceError(f"basis outcomes must be boolean: {task_id}/{intervention_id}")
            values.append(int(value))
        matrix.append(values)
        normalized_rows.append({
            "task_id": task_id,
            "cluster_id": row.get("cluster_id"),
            "source_id": row.get("source_id"),
            "baseline_failed": row.get("baseline_failed"),
            "outcomes": {key: int(value) for key, value in zip(intervention_ids, values)},
        })
    paired = sorted(
        zip(matrix, normalized_rows),
        key=lambda pair: (str(pair[1].get("cluster_id", "")), pair[1]["task_id"]),
    )
    matrix = [item[0] for item in paired]
    normalized_rows = [item[1] for item in paired]
    return matrix, normalized_rows


def _columns(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[int(row[column]) for row in matrix] for column in range(len(matrix[0]))]


def _signature_entropy(matrix: Sequence[Sequence[int]]) -> float:
    counts = Counter(tuple(int(value) for value in row) for row in matrix)
    total = len(matrix)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _pairwise_redundancy(matrix: Sequence[Sequence[int]]) -> float:
    columns = _columns(matrix)
    pairs = [(left, right) for left in range(len(columns)) for right in range(left + 1, len(columns))]
    if not pairs:
        return 0.0
    identical = sum(1 for left, right in pairs if columns[left] == columns[right])
    return identical / len(pairs)


def _null_matrix(matrix: Sequence[Sequence[int]], kind: str, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    rows = [list(map(int, row)) for row in matrix]
    if kind == "GLOBAL":
        prevalence = sum(sum(row) for row in rows) / (len(rows) * len(rows[0]))
        return [[int(rng.random() < prevalence) for _ in row] for row in rows]
    if kind == "COLUMN":
        columns = _columns(rows)
        shuffled = []
        for column in columns:
            values = column[:]
            rng.shuffle(values)
            shuffled.append(values)
        return [[shuffled[column][row] for column in range(len(shuffled))] for row in range(len(rows))]
    if kind == "ROW":
        output = []
        for row in rows:
            values = row[:]
            rng.shuffle(values)
            output.append(values)
        return output
    raise X1EvidenceError(f"unknown null family: {kind}")


def sparsity_matched_null_report(
    matrix: Sequence[Sequence[int]], *, n_resamples: int = 200, seed: int = 81818,
    max_p_value: float = 0.05, cluster_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not matrix or not matrix[0]:
        raise X1EvidenceError("null analysis requires a non-empty matrix")
    if n_resamples < 1:
        raise X1EvidenceError("n_resamples must be positive")
    real_entropy = _signature_entropy(matrix)
    real_unique = len({tuple(row) for row in matrix})
    families: dict[str, Any] = {}
    for offset, kind in enumerate(("GLOBAL", "COLUMN", "ROW", "CLUSTER_MARGINAL")):
        entropy_values = []
        unique_values = []
        for index in range(n_resamples):
            null_seed = (seed + offset * 1000003 + index * 7919) & 0xFFFFFFFFFFFFFFFF
            if kind == "CLUSTER_MARGINAL":
                if cluster_ids is None or len(cluster_ids) != len(matrix):
                    raise X1EvidenceError("cluster null requires one cluster ID per matrix row")
                null = [list(map(int, row)) for row in matrix]
                rng_cluster = random.Random(null_seed)
                for cluster_id in sorted(set(cluster_ids)):
                    indices = [index for index, row_cluster in enumerate(cluster_ids) if row_cluster == cluster_id]
                    for column in range(len(matrix[0])):
                        values = [null[index][column] for index in indices]
                        rng_cluster.shuffle(values)
                        for index, value in zip(indices, values):
                            null[index][column] = value
            else:
                null = _null_matrix(matrix, kind, null_seed)
            entropy_values.append(_signature_entropy(null))
            unique_values.append(len({tuple(row) for row in null}))
        structure_p = sum(value <= real_entropy for value in entropy_values) / len(entropy_values)
        diversity_p = sum(value >= real_entropy for value in entropy_values) / len(entropy_values)
        unique_p = sum(value >= real_unique for value in unique_values) / len(unique_values)
        families[kind] = {
            "structure_p_value": round(structure_p, 6),
            "entropy_p_value": round(structure_p, 6),
            "diversity_p_value": round(diversity_p, 6),
            "unique_signature_p_value": round(unique_p, 6),
            "entropy_mean": round(sum(entropy_values) / len(entropy_values), 6),
            "unique_mean": round(sum(unique_values) / len(unique_values), 6),
            "entropy_p95": round(sorted(entropy_values)[max(0, int(0.95 * len(entropy_values)) - 1)], 6),
            "unique_p95": sorted(unique_values)[max(0, int(0.95 * len(unique_values)) - 1)],
        }
    return {
        "real_signature_entropy_bits": round(real_entropy, 6),
        "real_unique_signatures": real_unique,
        "null_families": families,
        "all_families_separated": all(
            item["diversity_p_value"] <= max_p_value and
            item["unique_signature_p_value"] <= max_p_value
            for item in families.values()
        ),
        "structure_families_separated": all(
            item["structure_p_value"] <= max_p_value
            for item in families.values()
        ),
        "max_p_value": max_p_value,
        "geometry_assumption": "none; this is a diversity diagnostic, not a low-rank claim",
    }


def qualify_intervention_basis(
    artifact: Any,
    registry: Any,
    *,
    protocol_sha256: str | None = None,
    subject_manifest_sha256: str | None = None,
    thresholds: Mapping[str, Any] | None = None,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry_verdict = validate_intervention_registry(registry)
    if not registry_verdict["valid"]:
        receipt = {
            "schema": BASIS_SCHEMA,
            "status": "NOT_QUALIFIED",
            "protocol_sha256": protocol_sha256,
            "subject_manifest_sha256": subject_manifest_sha256,
            "registry_sha256": None,
            "basis_binding": dict(binding or {}),
            "checks": {"registry_valid": False},
            "errors": registry_verdict["errors"],
        }
        receipt["basis_qualification_sha256"] = _sha_json(receipt)
        return receipt
    intervention_ids = list(registry_verdict["intervention_ids"])
    matrix, normalized_rows = _matrix_from_artifact(artifact, intervention_ids)
    matrix_sha = _sha_json(normalized_rows)
    settings = dict(BASIS_DEFAULTS)
    if thresholds:
        settings.update({key: value for key, value in thresholds.items() if key in settings})
    n = len(matrix)
    columns = _columns(matrix)
    prevalence = [sum(column) / n for column in columns]
    no_change_index = intervention_ids.index("NO_CHANGE")
    null_indices = [
        index for index, intervention_id in enumerate(intervention_ids)
        if next(item for item in _registry_items(registry) if item["id"] == intervention_id)["role"] == "NULL_CONTROL"
    ]
    active_indices = [index for index in range(len(intervention_ids)) if index not in null_indices]
    oracle_coverage = sum(any(row[index] for index in active_indices) for row in matrix) / n
    no_change_rate = prevalence[no_change_index]
    sham_rates = [prevalence[index] for index in null_indices if index != no_change_index]
    control_excess = max((rate - no_change_rate for rate in sham_rates), default=1.0)
    cluster_values = [row.get("cluster_id") for row in normalized_rows]
    source_values = [row.get("source_id") for row in normalized_rows]
    try:
        null_report = sparsity_matched_null_report(
            matrix,
            n_resamples=int(settings["null_resamples"]),
            seed=int(settings["null_seed"]),
            max_p_value=float(settings["max_null_p_value"]),
            cluster_ids=cluster_values,
        )
    except X1EvidenceError as exc:
        null_report = {
            "real_signature_entropy_bits": None,
            "real_unique_signatures": None,
            "null_families": {},
            "all_families_separated": False,
            "structure_families_separated": False,
            "max_p_value": float(settings["max_null_p_value"]),
            "error": str(exc),
        }
    cluster_count = len(set(value for value in cluster_values if value))
    source_count = len(set(value for value in source_values if value))
    source_artifact_bound = False
    phase_artifact_bound = False
    if isinstance(binding, Mapping):
        source_path_value = binding.get("basis_source_artifact_path")
        phase_path_value = binding.get("basis_phase_artifact_path")
        if isinstance(source_path_value, str) and Path(source_path_value).is_file():
            try:
                source_document = _load_json(source_path_value)
                _, source_rows = _matrix_from_artifact(source_document, intervention_ids)
                source_artifact_bound = _sha_json(source_rows) == matrix_sha
            except X1EvidenceError:
                source_artifact_bound = False
        if isinstance(phase_path_value, str) and Path(phase_path_value).is_file():
            try:
                phase_document = _load_json(phase_path_value)
                phase_artifact_bound = (
                    isinstance(phase_document, Mapping)
                    and phase_document.get("schema") == PHASE_COMMIT_SCHEMA
                    and phase_document.get("phase") == "BASIS_QUALIFIED"
                    and phase_document.get("phase_index") == 0
                    and phase_document.get("receipt_sha256") == binding.get("basis_source_artifact_sha256")
                    and phase_document.get("artifact_sha256") == binding.get("basis_source_artifact_sha256")
                )
            except X1EvidenceError:
                phase_artifact_bound = False
    binding_complete = (
        isinstance(binding, Mapping) and set(binding) == BASIS_BINDING_FIELDS and
        all(isinstance(binding.get(key), str) and HEX64.fullmatch(binding.get(key, ""))
            for key in ("basis_split_manifest_sha256", "basis_source_artifact_sha256", "basis_matrix_sha256", "basis_phase_commit_sha256")) and
        all(isinstance(binding.get(key), str) and bool(binding.get(key))
            for key in ("basis_source_artifact_path", "basis_phase_artifact_path")) and
        binding.get("basis_matrix_sha256") == matrix_sha and source_artifact_bound and phase_artifact_bound and
        all(Path(binding.get(path_key, "")).is_file() and
            _sha_file(Path(binding.get(path_key, ""))) == binding.get(hash_key)
            for path_key, hash_key in (
                ("basis_source_artifact_path", "basis_source_artifact_sha256"),
                ("basis_phase_artifact_path", "basis_phase_commit_sha256"),
            )) and
        isinstance(binding.get("basis_cohort_id"), str) and bool(binding.get("basis_cohort_id"))
    )
    checks = {
        "registry_valid": True,
        "matrix_rectangular": True,
        "min_failures": n >= int(settings["min_failures"]),
        "explicit_baseline_failure_cohort": all(row.get("baseline_failed") is True for row in normalized_rows),
        "independent_cluster_identity": cluster_count >= int(settings["min_independent_clusters"]),
        "independent_source_identity": all(source_values) and source_count >= int(settings["min_independent_clusters"]),
        "oracle_coverage": oracle_coverage >= float(settings["min_oracle_coverage"]),
        "no_degenerate_active_probes": all(
            float(settings["min_active_prevalence"]) <= prevalence[index] <= float(settings["max_active_prevalence"])
            for index in active_indices
        ),
        "no_universal_solver": all(prevalence[index] < 0.95 for index in active_indices) and oracle_coverage < 0.95,
        "response_signature_variation": _signature_entropy(matrix) >= float(settings["min_signature_entropy_bits"]),
        "unique_signatures": len({tuple(row) for row in matrix}) >= int(settings["min_unique_signatures"]),
        "bounded_pairwise_redundancy": _pairwise_redundancy(matrix) <= float(settings["max_pairwise_redundancy"]),
        "no_change_control_null": no_change_rate <= float(settings["max_no_change_repair_rate"]),
        "matched_controls_null": control_excess <= float(settings["max_control_excess_rate"]),
        "sparsity_matched_diversity": null_report["all_families_separated"],
        "binding_complete": binding_complete,
    }
    quality = {
        "n_failures": n,
        "n_interventions": len(intervention_ids),
        "independent_clusters": cluster_count,
        "independent_sources": source_count,
        "oracle_coverage": round(oracle_coverage, 6),
        "cell_prevalence": round(sum(prevalence) / len(prevalence), 6),
        "per_intervention_prevalence": {
            intervention_id: round(prevalence[index], 6)
            for index, intervention_id in enumerate(intervention_ids)
        },
        "no_change_repair_rate": round(no_change_rate, 6),
        "matched_control_excess": round(control_excess, 6),
        "signature_entropy_bits": round(_signature_entropy(matrix), 6),
        "unique_signatures": len({tuple(row) for row in matrix}),
        "pairwise_redundancy": round(_pairwise_redundancy(matrix), 6),
    }
    body = {
        "schema": BASIS_SCHEMA,
        "status": "QUALIFIED" if all(checks.values()) else "NOT_QUALIFIED",
        "protocol_sha256": protocol_sha256,
        "subject_manifest_sha256": subject_manifest_sha256,
        "registry_sha256": registry_verdict["registry_sha256"],
        "basis_binding": dict(binding or {}),
        "basis_source_artifact_sha256": binding.get("basis_source_artifact_sha256") if isinstance(binding, Mapping) else None,
        "matrix_sha256": matrix_sha,
        "thresholds": settings,
        "checks": checks,
        "quality": quality,
        "null_analysis": null_report,
    }
    body["basis_qualification_sha256"] = _sha_json(body)
    return body


def validate_basis_qualification(
    receipt: Mapping[str, Any], *, expected_protocol_sha256: str | None = None,
    expected_subject_manifest_sha256: str | None = None,
    expected_registry_sha256: str | None = None, require_binding: bool = True,
    registry: Any | None = None, basis_split_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if receipt.get("schema") != BASIS_SCHEMA:
        errors.append("basis qualification schema mismatch")
    if receipt.get("status") != "QUALIFIED":
        errors.append(f"basis status is {receipt.get('status')!r}, not QUALIFIED")
    for key, expected in (
        ("protocol_sha256", expected_protocol_sha256),
        ("subject_manifest_sha256", expected_subject_manifest_sha256),
        ("registry_sha256", expected_registry_sha256),
    ):
        if expected is not None and receipt.get(key) != expected:
            errors.append(f"{key} mismatch")
    identity = receipt.get("basis_qualification_sha256")
    if not isinstance(identity, str) or not HEX64.fullmatch(identity):
        errors.append("basis qualification hash missing or malformed")
    else:
        body = _without(receipt, "basis_qualification_sha256")
        if _sha_json(body) != identity:
            errors.append("basis qualification hash mismatch")
    binding = receipt.get("basis_binding")
    if require_binding and (not isinstance(binding, Mapping) or set(binding) != BASIS_BINDING_FIELDS):
        errors.append("basis qualification binding is missing or incomplete")
    elif isinstance(binding, Mapping):
        try:
            _nonempty(binding.get("basis_cohort_id"), "basis_binding.basis_cohort_id")
        except X1EvidenceError as exc:
            errors.append(str(exc))
        for key in ("basis_split_manifest_sha256", "basis_source_artifact_sha256", "basis_matrix_sha256", "basis_phase_commit_sha256"):
            try:
                _hash(binding.get(key), f"basis_binding.{key}")
            except X1EvidenceError as exc:
                errors.append(str(exc))
        if binding.get("basis_matrix_sha256") != receipt.get("matrix_sha256"):
            errors.append("basis binding matrix hash does not match qualification receipt")
        if registry is not None and isinstance(binding.get("basis_source_artifact_path"), str):
            try:
                source_document = _load_json(binding["basis_source_artifact_path"])
                intervention_ids = validate_intervention_registry(registry)["intervention_ids"]
                _, source_rows = _matrix_from_artifact(source_document, intervention_ids)
                if _sha_json(source_rows) != receipt.get("matrix_sha256"):
                    errors.append("basis source artifact does not reproduce the qualification matrix")
                if basis_split_manifest is not None:
                    expected_basis_ids = {
                        task_id for cluster in basis_split_manifest.get("clusters", [])
                        for task_id in cluster.get("task_ids", [])
                    }
                    if {row.get("task_id") for row in source_rows} != expected_basis_ids:
                        errors.append("basis source task IDs do not match the basis split")
            except X1EvidenceError as exc:
                errors.append(str(exc))
        for path_key, hash_key in (
            ("basis_source_artifact_path", "basis_source_artifact_sha256"),
            ("basis_phase_artifact_path", "basis_phase_commit_sha256"),
        ):
            path_value = binding.get(path_key)
            if not isinstance(path_value, str) or not path_value:
                errors.append(f"basis binding path is missing: {path_key}")
            else:
                path = Path(path_value)
                if not path.is_file():
                    errors.append(f"basis binding artifact is missing: {path_value}")
                elif _sha_file(path) != binding.get(hash_key):
                    errors.append(f"basis binding artifact hash mismatch: {path_key}")
        phase_path_value = binding.get("basis_phase_artifact_path")
        if isinstance(phase_path_value, str) and Path(phase_path_value).is_file():
            try:
                phase_document = _load_json(phase_path_value)
                if not isinstance(phase_document, Mapping):
                    raise X1EvidenceError("basis phase artifact is not an object")
                if _sha_json(_without(phase_document, "phase_commit_sha256")) != phase_document.get("phase_commit_sha256"):
                    raise X1EvidenceError("basis phase artifact hash mismatch")
                if phase_document.get("schema") != PHASE_COMMIT_SCHEMA or phase_document.get("phase") != "BASIS_QUALIFIED":
                    raise X1EvidenceError("basis phase artifact schema mismatch")
                if phase_document.get("receipt_sha256") != binding.get("basis_source_artifact_sha256"):
                    raise X1EvidenceError("basis phase artifact receipt binding mismatch")
            except X1EvidenceError as exc:
                errors.append(str(exc))
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != BASIS_CHECK_NAMES or any(value is not True for value in checks.values()):
        errors.append("basis qualification check set is incomplete or contains a failed check")
    return {"valid": not errors, "errors": errors}


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parent / "registry" / "checkpoints.json"


def _load_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise X1EvidenceError(f"unable to read JSON artifact {path}: {exc}") from exc


def _verify_verifier_artifact(
    path: Path, *, task_id: str, intervention_id: str, repaired: bool,
    raw_output_sha256: str, gold_sha256: str, transformed_prompt_sha256: str,
) -> None:
    document = _load_json(path)
    if not isinstance(document, Mapping):
        raise X1EvidenceError("verifier artifact is not an object")
    _assert_exact_keys(document, {
        "schema", "task_id", "intervention_id", "repaired",
        "gold_sha256", "raw_output_sha256", "transformed_prompt_sha256",
    })
    if document.get("schema") != VERIFIER_SCHEMA:
        raise X1EvidenceError("verifier artifact schema mismatch")
    expected = {
        "task_id": task_id,
        "intervention_id": intervention_id,
        "repaired": repaired,
        "gold_sha256": gold_sha256,
        "raw_output_sha256": raw_output_sha256,
        "transformed_prompt_sha256": transformed_prompt_sha256,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise X1EvidenceError(f"verifier artifact field mismatch: {key}")


def inventory_checkpoint_candidates(
    registry_path: str | Path | None = None, *, verify_files: bool = False
) -> dict[str, Any]:
    path = Path(registry_path or _default_registry_path())
    document = _load_json(path)
    entries = document.get("checkpoints", []) if isinstance(document, Mapping) else []
    if not isinstance(entries, list):
        raise X1EvidenceError("checkpoint registry checkpoints must be a list")
    candidates = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        required = {
            "path", "file_sha256", "parameter_sha256", "config_sha256",
            "tokenizer_artifact_sha256", "tokenizer_identity_sha256",
            "runtime_source_revision", "source_commit", "identity_attestation_sha256", "identity_attestation_path", "global_step", "stage",
            "role", "status", "research_subject", "readiness",
        }
        missing = sorted(required - set(entry))
        placeholders = sorted(
            key for key in required & set(entry)
            if not isinstance(entry[key], (int, float, Mapping)) and (
                not isinstance(entry[key], str) or not entry[key].strip() or
                any(token in entry[key].lower() for token in ("unverified", "uncomputed", "unknown", "unfilled", "pending", "not_applicable", "n/a"))
            )
        )
        path_value = entry.get("path")
        file_verified = None
        if verify_files and isinstance(path_value, str):
            candidate = Path(path_value)
            if not candidate.is_absolute():
                candidate = path.parent.parent / candidate
            file_verified = candidate.is_file() and _sha_file(candidate) == entry.get("file_sha256")
        attestation_verified = None
        attestation_value = entry.get("identity_attestation_path")
        if verify_files and isinstance(attestation_value, str):
            attestation_path = Path(attestation_value)
            if not attestation_path.is_absolute():
                attestation_path = path.parent.parent / attestation_path
            attestation_verified = attestation_path.is_file() and _sha_file(attestation_path) == entry.get("identity_attestation_sha256")
        identity_complete = not missing and not placeholders
        eligible = bool(
            identity_complete and entry.get("research_subject") is True and
            entry.get("status") in {"QUALIFIED", "READY", "READY_SCOPED"} and
            file_verified is True and attestation_verified is True
        )
        candidates.append({
            "path": path_value,
            "file_sha256": entry.get("file_sha256"),
            "parameter_sha256": entry.get("parameter_sha256"),
            "config_sha256": entry.get("config_sha256"),
            "tokenizer_artifact_sha256": entry.get("tokenizer_artifact_sha256"),
            "tokenizer_identity_sha256": entry.get("tokenizer_identity_sha256"),
            "runtime_source_revision": entry.get("runtime_source_revision"),
            "source_commit": entry.get("source_commit"),
            "identity_attestation_sha256": entry.get("identity_attestation_sha256"),
            "identity_attestation_path": entry.get("identity_attestation_path"),
            "file_verified": file_verified,
            "attestation_verified": attestation_verified,
            "registry_entry_sha256": _sha_json(entry),
            "global_step": entry.get("global_step"),
            "role": entry.get("role"),
            "status": entry.get("status"),
            "research_subject": entry.get("research_subject"),
            "identity_complete": identity_complete,
            "missing_fields": missing,
            "placeholder_fields": placeholders,
            "eligible": eligible,
        })
    return {
        "schema": "anra-x1-real-1-inventory/v1",
        "registry_path": str(path),
        "registry_sha256": _sha_file(path),
        "n_candidates": len(candidates),
        "eligible_candidates": sum(1 for item in candidates if item["eligible"]),
        "candidates": candidates,
        "status": "ELIGIBLE_SUBJECT_AVAILABLE" if any(item["eligible"] for item in candidates) else "NO_ELIGIBLE_CHECKPOINT",
    }


def prerequisite_gate(
    *, registry_path: str | Path | None = None, basis_qualification: Mapping[str, Any] | None = None,
    release_manifest: Mapping[str, Any] | None = None, protocol: Mapping[str, Any] | None = None,
    release_root: str | Path | None = None,
) -> dict[str, Any]:
    inventory = inventory_checkpoint_candidates(registry_path, verify_files=True)
    blockers = []
    if protocol is not None:
        protocol_verdict = validate_frozen_protocol(protocol, check_source_closure=True)
        if not protocol_verdict["valid"]:
            blockers.append("PROTOCOL_INVALID")
    if inventory["eligible_candidates"] == 0:
        blockers.append("NO_ELIGIBLE_CHECKPOINT")
    if basis_qualification is None or basis_qualification.get("status") != "QUALIFIED":
        blockers.append("BASIS_NOT_QUALIFIED")
    if release_manifest is None:
        blockers.append("RELEASE_MANIFEST_REQUIRED")
    elif protocol is None:
        blockers.append("RELEASE_MANIFEST_NOT_VALIDATED")
    else:
        release_verdict = validate_release_manifest(release_manifest, protocol, repo_root=release_root)
        if not release_verdict["valid"]:
            blockers.append("RELEASE_MANIFEST_INVALID")
    if "PROTOCOL_INVALID" in blockers:
        status = "PROTOCOL_INVALID"
    elif "NO_ELIGIBLE_CHECKPOINT" in blockers:
        status = "NO_ELIGIBLE_CHECKPOINT"
    elif "BASIS_NOT_QUALIFIED" in blockers:
        status = "BASIS_NOT_QUALIFIED"
    elif blockers:
        status = "RELEASE_MANIFEST_REQUIRED"
    else:
        status = "READY_FOR_EXTERNAL_PHASE_1"
    return {
        "schema": "anra-x1-real-1-prerequisite-gate/v1",
        "status": status,
        "blockers": blockers,
        "inventory": inventory,
        "next_operator_action": (
            "repair and freeze the protocol/source identity before execution"
            if status == "PROTOCOL_INVALID" else
            "supply a complete research_subject registry entry and verified checkpoint manifest"
            if status == "NO_ELIGIBLE_CHECKPOINT" else
            "supply an independently split, pre-training QUALIFIED basis receipt"
            if status == "BASIS_NOT_QUALIFIED" else
            "publish and validate a clean source release manifest and custody receipts"
            if status in {"RELEASE_MANIFEST_REQUIRED", "RELEASE_MANIFEST_INVALID"} else
            "run external prediction commitment only; do not train or infer locally"
        ),
    }


def validate_subject_manifest(
    manifest: Mapping[str, Any], *, registry_path: str | Path | None = None,
    checkpoint_root: str | Path | None = None, verify_checkpoint_file: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        _assert_exact_keys(manifest, SUBJECT_FIELDS)
    except X1EvidenceError as exc:
        errors.append(str(exc))
    if manifest.get("schema") != SUBJECT_SCHEMA:
        errors.append("subject manifest schema mismatch")
    for key in (
        "subject_id", "checkpoint_path", "runtime_source_revision", "training_lineage", "stage",
    ):
        try:
            _nonempty(manifest.get(key), key)
        except X1EvidenceError as exc:
            errors.append(str(exc))
    for key in (
        "checkpoint_file_sha256", "parameter_sha256", "model_config_sha256",
        "tokenizer_artifact_sha256", "tokenizer_identity_sha256", "registry_entry_sha256",
        "identity_attestation_sha256",
    ):
        try:
            _hash(manifest.get(key), key)
        except X1EvidenceError as exc:
            errors.append(str(exc))
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not HEX40.fullmatch(source_commit):
        errors.append("source_commit must be a 40-character lowercase commit hash")
    if manifest.get("research_subject") is not True:
        errors.append("research_subject must be true")
    if not isinstance(manifest.get("global_step"), int) or isinstance(manifest.get("global_step"), bool) or manifest["global_step"] < 0:
        errors.append("global_step must be a non-negative integer")
    eligibility = manifest.get("eligibility")
    if not isinstance(eligibility, Mapping):
        errors.append("eligibility block is missing")
    else:
        try:
            _assert_exact_keys(eligibility, SUBJECT_ELIGIBILITY_FIELDS)
        except X1EvidenceError as exc:
            errors.append(str(exc))
        if eligibility.get("state") not in {"READY_SCOPED", "READY", "QUALIFIED"}:
            errors.append("eligibility state is not READY_SCOPED, READY, or QUALIFIED")
        for key in ("readiness_receipt_sha256", "cohort_plan_sha256", "cross_checkpoint_plan_sha256"):
            try:
                _hash(eligibility.get(key), f"eligibility.{key}")
            except X1EvidenceError as exc:
                errors.append(str(exc))
        for path_key, hash_key in (
            ("readiness_receipt_path", "readiness_receipt_sha256"),
            ("cohort_plan_path", "cohort_plan_sha256"),
            ("cross_checkpoint_plan_path", "cross_checkpoint_plan_sha256"),
        ):
            path_value = eligibility.get(path_key)
            if not isinstance(path_value, str) or not path_value:
                errors.append(f"eligibility artifact path is missing: {path_key}")
                continue
            path = Path(path_value)
            if not path.is_absolute():
                path = Path(checkpoint_root or Path.cwd()) / path
            if not path.is_file():
                errors.append(f"eligibility artifact is missing: {path_value}")
            elif _sha_file(path) != eligibility.get(hash_key):
                errors.append(f"eligibility artifact hash mismatch: {path_key}")
    file_verified = None
    checkpoint_path = manifest.get("checkpoint_path")
    if not verify_checkpoint_file:
        errors.append("checkpoint file verification is required for eligibility")
    elif isinstance(checkpoint_path, str) and checkpoint_path.strip():
        candidate = Path(checkpoint_path)
        if not candidate.is_absolute():
            candidate = Path(checkpoint_root or Path.cwd()) / candidate
        file_verified = candidate.is_file()
        if not file_verified:
            errors.append(f"checkpoint file is not available for verification: {candidate}")
        elif _sha_file(candidate) != manifest.get("checkpoint_file_sha256"):
            errors.append("checkpoint file hash does not match manifest")
    else:
        errors.append("checkpoint path is missing")
    registry_match = None
    try:
        registry_document = _load_json(registry_path or _default_registry_path())
        registry_entries = registry_document.get("checkpoints", [])
        if not isinstance(registry_entries, list):
            raise X1EvidenceError("checkpoint registry entries are not a list")
        expected_registry = {
            "path": checkpoint_path,
            "file_sha256": manifest.get("checkpoint_file_sha256"),
            "parameter_sha256": manifest.get("parameter_sha256"),
            "config_sha256": manifest.get("model_config_sha256"),
            "tokenizer_artifact_sha256": manifest.get("tokenizer_artifact_sha256"),
            "tokenizer_identity_sha256": manifest.get("tokenizer_identity_sha256"),
            "runtime_source_revision": manifest.get("runtime_source_revision"),
            "source_commit": manifest.get("source_commit"),
            "global_step": manifest.get("global_step"),
            "stage": manifest.get("stage"),
            "research_subject": True,
        }
        matches = [
            entry for entry in registry_entries
            if isinstance(entry, Mapping) and all(entry.get(key) == value for key, value in expected_registry.items())
        ]
        if len(matches) != 1:
            errors.append(f"subject must match exactly one complete registry entry; matches={len(matches)}")
        else:
            registry_match = matches[0]
            if registry_match.get("status") not in {"QUALIFIED", "READY", "READY_SCOPED"}:
                errors.append("registry entry status is not eligible")
            if any(token in str(registry_match.get("role", "")).lower() for token in ("historical", "control", "weak", "non_model")):
                errors.append("historical/control registry role is not a research subject")
            readiness = registry_match.get("readiness")
            if not isinstance(readiness, Mapping) or dict(readiness) != dict(manifest.get("eligibility", {})):
                errors.append("registry readiness binding mismatch")
            if _sha_json(registry_match) != manifest.get("registry_entry_sha256"):
                errors.append("registry entry hash mismatch")
    except X1EvidenceError as exc:
        errors.append(str(exc))
    attestation_path_value = manifest.get("identity_attestation_path")
    if not isinstance(attestation_path_value, str) or not attestation_path_value:
        errors.append("identity attestation path is missing")
    else:
        attestation_path = Path(attestation_path_value)
        if not attestation_path.is_absolute():
            attestation_path = Path(checkpoint_root or Path.cwd()) / attestation_path
        if not attestation_path.is_file():
            errors.append(f"identity attestation is missing: {attestation_path}")
        elif _sha_file(attestation_path) != manifest.get("identity_attestation_sha256"):
            errors.append("identity attestation hash mismatch")
        else:
            try:
                attestation = _load_json(attestation_path)
                if not isinstance(attestation, Mapping):
                    raise X1EvidenceError("identity attestation is not an object")
                _assert_exact_keys(attestation, SUBJECT_ATTESTATION_FIELDS)
                if attestation.get("schema") != "anra-checkpoint-identity/v1":
                    raise X1EvidenceError("identity attestation schema mismatch")
                for key in (
                    "subject_id", "checkpoint_file_sha256", "parameter_sha256", "model_config_sha256",
                    "tokenizer_artifact_sha256", "tokenizer_identity_sha256", "runtime_source_revision",
                    "source_commit", "training_lineage", "stage", "global_step", "registry_entry_sha256",
                ):
                    if attestation.get(key) != manifest.get(key):
                        raise X1EvidenceError(f"identity attestation mismatch: {key}")
                registry_path_for_attestation = Path(registry_path or _default_registry_path())
                if attestation.get("registry_sha256") != _sha_file(registry_path_for_attestation):
                    raise X1EvidenceError("identity attestation registry hash mismatch")
            except X1EvidenceError as exc:
                errors.append(str(exc))
    return {
        "valid": not errors,
        "errors": errors,
        "status": "ELIGIBLE" if not errors else "BLOCKED",
        "checkpoint_file_verified": file_verified,
        "registry_match": bool(registry_match),
        "subject_manifest_sha256": _sha_json(manifest),
    }


def _subject_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "subject_id", "checkpoint_file_sha256", "parameter_sha256", "model_config_sha256",
        "tokenizer_artifact_sha256", "tokenizer_identity_sha256", "runtime_source_revision",
        "source_commit", "training_lineage", "stage", "global_step",
    )
    return {key: manifest.get(key) for key in keys}


def validate_frozen_protocol(
    protocol: Mapping[str, Any], *, repo_root: str | Path | None = None,
    check_source_closure: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    required = {
        "schema", "brand", "version", "phase", "authority", "claim_ceiling", "execution_boundary",
        "preflight_contract", "power_policy", "subject_binding", "interventions", "intervention_registry_sha256", "cost_model", "execution_budget", "cohorts",
        "splits", "observations", "predictor", "baselines", "metrics", "decision_thresholds",
        "release_binding", "custody_phases", "prediction_parent_phase", "reveal_evidence",
        "stop_conditions", "next_rung_contract", "decision_table", "identity",
    }
    missing = sorted(required - set(protocol))
    unknown = sorted(set(protocol) - required)
    if missing or unknown:
        errors.append(f"protocol keys invalid; missing={missing}, unknown={unknown}")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        errors.append("protocol schema mismatch")
    if protocol.get("brand") != "euler":
        errors.append("protocol brand must be euler")
    if protocol.get("version") != 1:
        errors.append("unsupported protocol version")
    if protocol.get("phase") != "FROZEN_TEMPLATE":
        errors.append("protocol is not a frozen template")
    boundary = protocol.get("execution_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("local_model_execution") is not False:
        errors.append("local model execution must be disabled")
    power_policy = protocol.get("power_policy")
    if not isinstance(power_policy, Mapping) or power_policy.get("target_clusters_are_preferred") is not True or power_policy.get("minimum_clusters_are_hard_gate") is not True or power_policy.get("below_target_requires_explicit_deviation") is not True:
        errors.append("power policy is missing or not fail-closed")
    preflight_contract = protocol.get("preflight_contract")
    if not isinstance(preflight_contract, Mapping) or preflight_contract.get("schema") != PREFLIGHT_SCHEMA or preflight_contract.get("required_before_external_execution") is not True or preflight_contract.get("transformation_smoke_required") is not True or preflight_contract.get("candidate_intake_required") is not True or preflight_contract.get("candidate_intake_schema") != INTAKE_SCHEMA or preflight_contract.get("platform_profiles") != ["COLAB", "KAGGLE", "PERSISTENT_GPU_RUNNER"] or preflight_contract.get("failure_policy") != "BLOCKED":
        errors.append("preflight contract is missing or not fail-closed")
    if not isinstance(protocol.get("claim_ceiling"), str) or not protocol["claim_ceiling"].strip():
        errors.append("claim ceiling is missing")
    cost_model = protocol.get("cost_model")
    if not isinstance(cost_model, Mapping) or cost_model.get("unit") != "abstract_intervention_unit" or cost_model.get("all_costs_predeclared") is not True:
        errors.append("cost comparison contract is missing or invalid")
    subject_binding = protocol.get("subject_binding")
    if not isinstance(subject_binding, Mapping) or subject_binding.get("release_manifest_required") is not True:
        errors.append("release manifest binding must be required")
    execution_budget = protocol.get("execution_budget")
    if not isinstance(execution_budget, Mapping) or execution_budget.get("local_execution_authorized") is not False:
        errors.append("execution budget must prohibit local execution")
    if isinstance(execution_budget, Mapping) and execution_budget.get("cell_budget_fail_closed") is not True:
        errors.append("execution cell budget must be fail-closed")
    if isinstance(execution_budget, Mapping):
        budget_values = [execution_budget.get(key) for key in ("primary_cells", "replication_cells", "maximum_external_cells")]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in budget_values):
            errors.append("execution budget cell counts must be integers")
        elif budget_values[0] + budget_values[1] > budget_values[2]:
            errors.append("execution budget cells exceed declared maximum")
    if isinstance(execution_budget, Mapping) and isinstance(protocol.get("interventions"), list):
        try:
            expected_cells = len(protocol["interventions"]) * int(protocol.get("cohorts", {}).get("primary", {}).get("target_independent_clusters", 0))
            expected_replication_cells = len(protocol["interventions"]) * int(protocol.get("cohorts", {}).get("checkpoint_replication", {}).get("target_independent_clusters", 0))
            if execution_budget.get("primary_cells") != expected_cells:
                errors.append("primary execution budget does not match cohort x intervention count")
            if execution_budget.get("replication_cells") != expected_replication_cells:
                errors.append("replication execution budget does not match cohort x intervention count")
        except (TypeError, ValueError):
            errors.append("execution budget cohort counts are malformed")
    cohorts = protocol.get("cohorts", {})
    if not isinstance(cohorts, Mapping):
        errors.append("cohort definitions are missing")
    else:
        cohort_ids = set()
        for name, cohort in cohorts.items():
            if not isinstance(cohort, Mapping):
                errors.append(f"cohort definition is malformed: {name}")
                continue
            cohort_id = cohort.get("cohort_id")
            if not isinstance(cohort_id, str) or cohort_id in cohort_ids:
                errors.append(f"cohort identity is invalid: {name}")
            cohort_ids.add(cohort_id)
            for key in ("seed", "target_independent_clusters", "minimum_independent_clusters"):
                if not isinstance(cohort.get(key), int) or isinstance(cohort.get(key), bool) or cohort.get(key) < 0:
                    errors.append(f"cohort {name}.{key} must be a non-negative integer")
            if isinstance(cohort.get("minimum_independent_clusters"), int) and isinstance(cohort.get("target_independent_clusters"), int) and cohort["minimum_independent_clusters"] > cohort["target_independent_clusters"]:
                errors.append(f"cohort minimum exceeds target: {name}")
    interventions = protocol.get("interventions")
    registry_verdict = validate_intervention_registry(interventions)
    if not registry_verdict["valid"]:
        errors.extend(registry_verdict["errors"])
    elif protocol.get("intervention_registry_sha256") != registry_verdict["registry_sha256"]:
        errors.append("intervention registry digest mismatch")
    thresholds = protocol.get("decision_thresholds")
    if isinstance(cost_model, Mapping) and isinstance(thresholds, Mapping):
        predictor_thresholds = thresholds.get("predictor", {})
        if isinstance(predictor_thresholds, Mapping) and predictor_thresholds.get("cost_lambda") != cost_model.get("utility_lambda"):
            errors.append("cost lambda is not frozen consistently")
    if not isinstance(thresholds, Mapping):
        errors.append("decision thresholds are missing")
    else:
        for key in ("basis", "predictor", "replication"):
            if not isinstance(thresholds.get(key), Mapping) or not thresholds[key]:
                errors.append(f"decision_thresholds.{key} is missing")
    if protocol.get("custody_phases") != ["BASIS_QUALIFIED", "PREDICT_COMMITTED", "REVEAL_ACCEPTED"]:
        errors.append("custody phase sequence is invalid")
    if protocol.get("prediction_parent_phase") != "BASIS_QUALIFIED":
        errors.append("prediction parent phase is invalid")
    release_binding = protocol.get("release_binding")
    if not isinstance(release_binding, Mapping) or not all(
        release_binding.get(key) is True for key in (
            "release_revision_required", "protocol_artifact_must_match_supplied_protocol", "attestation_artifact_required",
        )
    ):
        errors.append("release binding contract is incomplete")
    reveal_evidence = protocol.get("reveal_evidence")
    if not isinstance(reveal_evidence, Mapping) or not isinstance(reveal_evidence.get("required_row_fields"), list):
        errors.append("reveal evidence contract is incomplete")
    observations = protocol.get("observations")
    if not isinstance(observations, Mapping):
        errors.append("observation contract is missing")
    else:
        if observations.get("closed_schema") is not True:
            errors.append("observation schema must be closed")
        if observations.get("prediction_receipt_before_reveal") is not True:
            errors.append("prediction-before-reveal rule is not enabled")
    baselines = protocol.get("baselines")
    if not isinstance(baselines, list) or any(item not in PREDICTION_BASELINES for item in baselines):
        errors.append("baseline registry is incomplete or contains oracle policy")
    identity = protocol.get("identity")
    if not isinstance(identity, Mapping):
        errors.append("protocol identity is missing")
    else:
        protocol_identity = identity.get("protocol_sha256")
        if not isinstance(protocol_identity, str) or not HEX64.fullmatch(protocol_identity):
            errors.append("protocol identity hash is malformed")
        elif _protocol_hash(protocol) != protocol_identity:
            errors.append("protocol identity hash mismatch")
        try:
            _hash(identity.get("source_closure_sha256"), "identity.source_closure_sha256")
        except X1EvidenceError as exc:
            errors.append(str(exc))
        source_files = identity.get("source_files")
        if not isinstance(source_files, list) or len(source_files) != len(REQUIRED_SOURCE_PATHS) or {item.get("path") for item in source_files if isinstance(item, Mapping)} != set(REQUIRED_SOURCE_PATHS):
            errors.append("protocol source path allowlist mismatch")
        if check_source_closure:
            closure = validate_source_closure(protocol, repo_root)
            errors.extend(closure["errors"])
    if errors:
        return {"valid": False, "errors": errors, "protocol_sha256": _protocol_hash(protocol)}
    return {"valid": True, "errors": [], "protocol_sha256": protocol["identity"]["protocol_sha256"]}


def require_frozen_protocol(
    protocol: Mapping[str, Any], *, repo_root: str | Path | None = None,
    check_source_closure: bool = False,
) -> str:
    verdict = validate_frozen_protocol(protocol, repo_root=repo_root, check_source_closure=check_source_closure)
    if not verdict["valid"]:
        raise X1ProtocolError("; ".join(verdict["errors"]))
    return str(protocol["identity"]["protocol_sha256"])


def load_protocol(path: str | Path, *, repo_root: str | Path | None = None,
                  check_source_closure: bool = False) -> dict[str, Any]:
    document = _load_json(path)
    if not isinstance(document, Mapping):
        raise X1ProtocolError("protocol document must be an object")
    require_frozen_protocol(document, repo_root=repo_root, check_source_closure=check_source_closure)
    return dict(document)


def validate_split_manifest(
    manifest: Mapping[str, Any], *, expected_protocol_sha256: str | None = None,
    expected_cohort_id: str | None = None, expected_cohort_role: str | None = None,
    expected_seed: int | None = None, expected_cohort_plan_sha256: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    required = {
        "schema", "cohort_id", "cohort_role", "seed", "cohort_plan_sha256",
        "clusters", "manifest_sha256",
    }
    try:
        _assert_exact_keys(manifest, required, {"protocol_sha256"})
    except X1EvidenceError as exc:
        errors.append(str(exc))
    if manifest.get("schema") != SPLIT_SCHEMA:
        errors.append("split manifest schema mismatch")
    cohort_id = manifest.get("cohort_id")
    try:
        _nonempty(cohort_id, "cohort_id")
    except X1EvidenceError as exc:
        errors.append(str(exc))
    if manifest.get("cohort_role") not in {"DEVELOPMENT", "QUALIFICATION", "EVALUATION", "REPLICATION"}:
        errors.append("invalid cohort role")
    if not isinstance(manifest.get("seed"), int) or isinstance(manifest.get("seed"), bool):
        errors.append("split seed must be an integer")
    try:
        _hash(manifest.get("cohort_plan_sha256"), "cohort_plan_sha256")
    except X1EvidenceError as exc:
        errors.append(str(exc))
    if expected_cohort_id is not None and manifest.get("cohort_id") != expected_cohort_id:
        errors.append("split cohort identity mismatch")
    if expected_cohort_role is not None and manifest.get("cohort_role") != expected_cohort_role:
        errors.append("split cohort role mismatch")
    if expected_seed is not None and manifest.get("seed") != expected_seed:
        errors.append("split seed mismatch")
    expected_plan = expected_cohort_plan_sha256 or FROZEN_COHORT_PLAN_HASHES.get(cohort_id)
    if expected_plan is not None and manifest.get("cohort_plan_sha256") != expected_plan:
        errors.append("split cohort plan mismatch")
    clusters = manifest.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        errors.append("split clusters are missing")
        clusters = []
    seen_clusters: set[str] = set()
    seen_sources: set[str] = set()
    seen_tasks: set[str] = set()
    task_to_split: dict[str, str] = {}
    for cluster in clusters:
        if not isinstance(cluster, Mapping) or set(cluster) != {"cluster_id", "source_id", "split", "task_ids", "task_content_sha256s"}:
            errors.append("invalid cluster record")
            continue
        cluster_id = cluster.get("cluster_id")
        source_id = cluster.get("source_id")
        split = cluster.get("split")
        if split not in {"DEVELOPMENT", "QUALIFICATION", "PRIMARY_EVAL", "INDEPENDENT_TASK_EVAL", "CHECKPOINT_REPLICATION"}:
            errors.append(f"invalid split label: {split}")
        expected_split = {
            "BASIS_QUALIFICATION": "QUALIFICATION",
            "DEV_COHORT_V1": "DEVELOPMENT",
            "PRIMARY_EVAL": "PRIMARY_EVAL",
            "INDEPENDENT_TASK_EVAL": "INDEPENDENT_TASK_EVAL",
            "CHECKPOINT_REPLICATION": "CHECKPOINT_REPLICATION",
        }.get(cohort_id)
        if expected_split is not None and split != expected_split:
            errors.append(f"split label does not match cohort role: {split}")
        task_ids = cluster.get("task_ids")
        if not isinstance(cluster_id, str) or not cluster_id or cluster_id in seen_clusters:
            errors.append(f"duplicate or invalid cluster_id: {cluster_id}")
            continue
        if not isinstance(source_id, str) or not source_id or source_id in seen_sources:
            errors.append(f"duplicate or invalid source_id: {source_id}")
        if not isinstance(task_ids, list) or not task_ids:
            errors.append(f"cluster has no task_ids: {cluster_id}")
            continue
        content_hashes = cluster.get("task_content_sha256s")
        if not isinstance(content_hashes, Mapping) or set(content_hashes) != set(task_ids):
            errors.append(f"cluster task content hash coverage mismatch: {cluster_id}")
        else:
            for task_id, content_hash in content_hashes.items():
                try:
                    _hash(content_hash, f"task_content_sha256s.{cluster_id}.{task_id}")
                except X1EvidenceError as exc:
                    errors.append(str(exc))
        seen_clusters.add(cluster_id)
        seen_sources.add(source_id)
        for task_id in task_ids:
            if not isinstance(task_id, str) or not OPAQUE_TASK_ID.fullmatch(task_id) or task_id.lower().startswith(("gold", "answer", "family", "cluster", "source")) or task_id in seen_tasks:
                errors.append(f"duplicate or invalid task_id: {task_id}")
            seen_tasks.add(task_id)
            task_to_split[task_id] = str(split)
    expected_hash = manifest.get("manifest_sha256")
    body = _without(manifest, "manifest_sha256")
    if not isinstance(expected_hash, str) or not HEX64.fullmatch(expected_hash):
        errors.append("split manifest hash is malformed")
    elif _sha_json(body) != expected_hash:
        errors.append("split manifest hash mismatch")
    if expected_protocol_sha256 is not None:
        bound = manifest.get("protocol_sha256")
        if bound != expected_protocol_sha256:
            errors.append("split protocol binding mismatch")
    return {
        "valid": not errors,
        "errors": errors,
        "n_clusters": len(seen_clusters),
        "n_tasks": len(seen_tasks),
        "task_to_split": task_to_split,
        "manifest_sha256": expected_hash,
    }


def validate_split_independence(
    primary: Mapping[str, Any], replication: Mapping[str, Any]
) -> dict[str, Any]:
    primary_verdict = validate_split_manifest(primary)
    replication_verdict = validate_split_manifest(replication)
    errors = list(primary_verdict["errors"]) + list(replication_verdict["errors"])
    if primary_verdict["valid"] and replication_verdict["valid"]:
        primary_clusters = {item["cluster_id"] for item in primary.get("clusters", [])}
        replication_clusters = {item["cluster_id"] for item in replication.get("clusters", [])}
        primary_sources = {item["source_id"] for item in primary.get("clusters", [])}
        replication_sources = {item["source_id"] for item in replication.get("clusters", [])}
        primary_tasks = set(primary_verdict["task_to_split"])
        replication_tasks = set(replication_verdict["task_to_split"])
        primary_content = {
            content_hash for cluster in primary.get("clusters", [])
            for content_hash in cluster.get("task_content_sha256s", {}).values()
        }
        replication_content = {
            content_hash for cluster in replication.get("clusters", [])
            for content_hash in cluster.get("task_content_sha256s", {}).values()
        }
        for name, overlap in (
            ("cluster", primary_clusters & replication_clusters),
            ("source", primary_sources & replication_sources),
            ("task", primary_tasks & replication_tasks),
            ("task_content", primary_content & replication_content),
        ):
            if overlap:
                errors.append(f"{name} overlap between evaluation splits: {sorted(overlap)[:5]}")
    return {"valid": not errors, "errors": errors}


def _validate_public_task(task: Mapping[str, Any]) -> dict[str, Any]:
    try:
        _assert_exact_keys(
            task,
            {"task_id", "context", "query", "visible_candidates"},
            {"format", "features", "observation_hash"},
        )
    except X1EvidenceError as exc:
        raise X1EvidenceError(str(exc)) from exc
    _assert_no_policy_leak(task)
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not OPAQUE_TASK_ID.fullmatch(task_id) or task_id.lower().startswith(("gold", "answer", "family", "cluster", "source")):
        raise X1EvidenceError(f"task_id is not an opaque valid identifier: {task_id}")
    for key in ("context", "query"):
        if not isinstance(task.get(key), str) or not task[key]:
            raise X1EvidenceError(f"public task field must be non-empty text: {key}")
    candidates = task.get("visible_candidates")
    if not isinstance(candidates, list) or not candidates or any(not isinstance(item, str) for item in candidates):
        raise X1EvidenceError("visible_candidates must be a non-empty string list")
    if task.get("format") is not None and not isinstance(task.get("format"), str):
        raise X1EvidenceError("format must be text when present")
    features = task.get("features", {})
    if not isinstance(features, Mapping):
        raise X1EvidenceError("features must be an object")
    unknown_features = set(features) - ALLOWED_FEATURE_FIELDS
    if unknown_features:
        raise X1EvidenceError(f"unknown observation features: {sorted(unknown_features)}")
    for key, value in features.items():
        if key in {"output_len", "prompt_tokens"}:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise X1EvidenceError(f"invalid count feature: {key}")
        else:
            _finite(value, f"features.{key}")
    normalized = dict(task)
    if "observation_hash" in normalized:
        expected = normalized.pop("observation_hash")
        if not isinstance(expected, str) or not HEX64.fullmatch(expected) or _sha_json(normalized) != expected:
            raise X1EvidenceError(f"observation hash mismatch: {task_id}")
    return normalized


def validate_public_task_bundle(
    tasks: Sequence[Mapping[str, Any]], *, split_manifest: Mapping[str, Any] | None = None,
    expected_task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)) or not tasks:
        raise X1EvidenceError("public task bundle is empty")
    normalized = [_validate_public_task(task) for task in tasks]
    ids = [task["task_id"] for task in normalized]
    content_hashes = {}
    for task in normalized:
        content_hashes[task["task_id"]] = _sha_json({
            key: value for key, value in task.items()
            if key not in {"task_id", "observation_hash"}
        })
    if len(set(ids)) != len(ids):
        raise X1EvidenceError("public task bundle contains duplicate task IDs")
    if len(set(content_hashes.values())) != len(content_hashes):
        raise X1EvidenceError("public task bundle contains repeated task content under different IDs")
    normalized.sort(key=lambda task: task["task_id"])
    if expected_task_ids is not None and set(ids) != set(expected_task_ids):
        raise X1EvidenceError("public task IDs do not match the prediction task manifest")
    if split_manifest is not None:
        verdict = validate_split_manifest(split_manifest)
        if not verdict["valid"]:
            raise X1EvidenceError("split manifest is invalid")
        expected = set(verdict["task_to_split"])
        if expected != set(ids):
            raise X1EvidenceError("public task bundle does not exactly cover the split")
        declared_content = {
            task_id: content_hash
            for cluster in split_manifest.get("clusters", [])
            for task_id, content_hash in cluster.get("task_content_sha256s", {}).items()
        }
        if declared_content != content_hashes:
            raise X1EvidenceError("public task content does not match split content hashes")
    return {
        "valid": True,
        "tasks": normalized,
        "task_ids": ids,
        "bundle_sha256": _sha_json(normalized),
    }


def _protocol_intervention_ids(protocol: Mapping[str, Any]) -> list[str]:
    return [item["id"] for item in _registry_items(protocol.get("interventions"))]


def _normalize_prediction_map(
    predictions: Any, task_ids: Sequence[str], intervention_ids: Sequence[str], *, name: str
) -> dict[str, dict[str, Any]]:
    if isinstance(predictions, Mapping):
        raw_items = []
        for task_id, value in predictions.items():
            if isinstance(value, Mapping):
                item = dict(value)
                item.setdefault("task_id", task_id)
            else:
                item = {"task_id": task_id, "predicted_repair_probability": value}
            raw_items.append(item)
    elif isinstance(predictions, Sequence) and not isinstance(predictions, (str, bytes)):
        if any(not isinstance(item, Mapping) for item in predictions):
            raise X1EvidenceError(f"{name} contains a malformed prediction record")
        raw_items = [dict(item) for item in predictions]
    else:
        raise X1EvidenceError(f"{name} must be a task mapping or record list")
    by_task: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        task_id = item.get("task_id")
        if task_id not in task_ids or task_id in by_task:
            raise X1EvidenceError(f"{name} has duplicate or unknown task: {task_id}")
        values = item.get("predicted_repair_probability", item.get("predicted_response_distribution", item.get("probabilities")))
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != len(intervention_ids):
            raise X1EvidenceError(f"{name} probability length mismatch for {task_id}")
        probabilities = [_probability(value, f"{name}.{task_id}") for value in values]
        best = item.get("predicted_best_intervention", item.get("best_intervention"))
        if best is None:
            best = intervention_ids[max(range(len(probabilities)), key=lambda index: (probabilities[index], -index))]
        if best not in intervention_ids:
            raise X1EvidenceError(f"{name} best intervention is not a candidate: {task_id}")
        uncertainty = _probability(item.get("uncertainty", 0.0), f"{name}.{task_id}.uncertainty")
        normalized = {
            "task_id": task_id,
            "candidate_interventions": list(intervention_ids),
            "predicted_repair_probability": probabilities,
            "predicted_best_intervention": best,
            "uncertainty": uncertainty,
        }
        if "predicted_utility" in item:
            utility = item["predicted_utility"]
            if not isinstance(utility, Sequence) or len(utility) != len(intervention_ids):
                raise X1EvidenceError(f"{name} utility length mismatch for {task_id}")
            normalized["predicted_utility"] = [_finite(value, f"{name}.{task_id}.predicted_utility") for value in utility]
        by_task[task_id] = normalized
    if set(by_task) != set(task_ids):
        raise X1EvidenceError(f"{name} does not cover every task exactly once")
    return {task_id: by_task[task_id] for task_id in task_ids}


def _validate_predictor_identity(
    identity: Mapping[str, Any], *, artifact_root: str | Path | None = None,
    expected_feature_schema_sha256: str | None = None,
    expected_training_split_manifest_sha256: str | None = None,
    expected_development_task_ids: Sequence[str] | None = None,
) -> None:
    required = {
        "name", "version", "source_sha256", "training_artifact_sha256",
        "training_artifact_path",
        "training_manifest_path", "training_manifest_sha256",
        "training_split_manifest_sha256", "feature_schema_sha256",
        "evaluation_outcomes_used",
    }
    try:
        _assert_exact_keys(identity, required)
    except X1EvidenceError as exc:
        raise X1EvidenceError(str(exc)) from exc
    _nonempty(identity.get("name"), "predictor.name")
    _nonempty(identity.get("version"), "predictor.version")
    if identity.get("evaluation_outcomes_used") is not False:
        raise X1EvidenceError("predictor must declare evaluation_outcomes_used=false")
    for key in ("source_sha256", "training_artifact_sha256", "training_manifest_sha256", "training_split_manifest_sha256", "feature_schema_sha256"):
        _hash(identity.get(key), f"predictor.{key}")
    if expected_feature_schema_sha256 is not None and identity.get("feature_schema_sha256") != expected_feature_schema_sha256:
        raise X1EvidenceError("predictor feature schema does not match protocol")
    if expected_training_split_manifest_sha256 is not None and identity.get("training_split_manifest_sha256") != expected_training_split_manifest_sha256:
        raise X1EvidenceError("predictor training split does not match receipt development split")
    artifact_path = identity.get("training_artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise X1EvidenceError("predictor training artifact path is missing")
    candidate = Path(artifact_path)
    if not candidate.is_absolute():
        candidate = Path(artifact_root or Path.cwd()) / candidate
    if not candidate.is_file():
        raise X1EvidenceError(f"predictor training artifact is missing: {candidate}")
    if _sha_file(candidate) != identity.get("training_artifact_sha256"):
        raise X1EvidenceError("predictor training artifact hash mismatch")
    training_manifest_path = identity.get("training_manifest_path")
    if not isinstance(training_manifest_path, str) or not training_manifest_path:
        raise X1EvidenceError("predictor training manifest path is missing")
    training_manifest_candidate = Path(training_manifest_path)
    if not training_manifest_candidate.is_absolute():
        training_manifest_candidate = Path(artifact_root or Path.cwd()) / training_manifest_candidate
    if not training_manifest_candidate.is_file():
        raise X1EvidenceError("predictor training manifest is missing")
    if _sha_file(training_manifest_candidate) != identity.get("training_manifest_sha256"):
        raise X1EvidenceError("predictor training manifest hash mismatch")
    training_manifest = _load_json(training_manifest_candidate)
    if not isinstance(training_manifest, Mapping):
        raise X1EvidenceError("predictor training manifest is not an object")
    _assert_exact_keys(training_manifest, {
        "schema", "training_split_manifest_sha256", "development_task_ids",
        "evaluation_outcomes_used", "source_sha256", "feature_schema_sha256",
    })
    if training_manifest.get("schema") != PREDICTOR_TRAINING_SCHEMA:
        raise X1EvidenceError("predictor training manifest schema mismatch")
    if training_manifest.get("training_split_manifest_sha256") != identity.get("training_split_manifest_sha256"):
        raise X1EvidenceError("predictor training manifest split mismatch")
    if training_manifest.get("evaluation_outcomes_used") is not False:
        raise X1EvidenceError("predictor training manifest used evaluation outcomes")
    for key in ("source_sha256", "feature_schema_sha256"):
        if training_manifest.get(key) != identity.get(key):
            raise X1EvidenceError(f"predictor training manifest identity mismatch: {key}")
    development_task_ids = training_manifest.get("development_task_ids")
    if not isinstance(development_task_ids, list) or not development_task_ids or any(
        not isinstance(task_id, str) or not OPAQUE_TASK_ID.fullmatch(task_id) for task_id in development_task_ids
    ) or len(set(development_task_ids)) != len(development_task_ids):
        raise X1EvidenceError("predictor training manifest development task IDs are invalid")
    if expected_development_task_ids is not None and set(development_task_ids) != set(expected_development_task_ids):
        raise X1EvidenceError("predictor training task IDs do not match the development split")


def _validate_baseline_commitment(
    commitment: Mapping[str, Any], task_ids: Sequence[str], intervention_ids: Sequence[str], *,
    expected_development_cohort_id: str | None = None,
    expected_selection_lambda: float | None = None, artifact_root: str | Path | None = None,
    development_split_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = {
        "schema", "development_artifact_sha256", "development_artifact_path",
        "development_cohort_id", "development_split_manifest_sha256", "development_prevalence",
        "development_prevalence_by_intervention", "selection_lambda",
        "baselines", "fit_scope", "surface_shortcut_diagnostic_only", "commitment_sha256",
    }
    try:
        _assert_exact_keys(commitment, required)
    except X1EvidenceError as exc:
        raise X1EvidenceError(str(exc)) from exc
    if commitment.get("schema") != "anra-x1-real-1-baseline-commitment/v1":
        raise X1EvidenceError("baseline commitment schema mismatch")
    _hash(commitment.get("development_artifact_sha256"), "baseline.development_artifact_sha256")
    artifact_path = commitment.get("development_artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise X1EvidenceError("baseline development artifact path is missing")
    candidate = Path(artifact_path)
    if not candidate.is_absolute():
        candidate = Path(artifact_root or Path.cwd()) / candidate
    if not candidate.is_file() or _sha_file(candidate) != commitment.get("development_artifact_sha256"):
        raise X1EvidenceError("baseline development artifact hash verification failed")
    _nonempty(commitment.get("development_cohort_id"), "baseline.development_cohort_id")
    _hash(commitment.get("development_split_manifest_sha256"), "baseline.development_split_manifest_sha256")
    development_prevalence = _probability(commitment.get("development_prevalence"), "baseline.development_prevalence")
    prevalence_by_intervention = commitment.get("development_prevalence_by_intervention")
    if not isinstance(prevalence_by_intervention, Mapping) or set(prevalence_by_intervention) != set(intervention_ids):
        raise X1EvidenceError("baseline per-intervention prevalence is missing or incomplete")
    for intervention_id, value in prevalence_by_intervention.items():
        _probability(value, f"baseline.development_prevalence_by_intervention.{intervention_id}")
    if abs(sum(float(value) for value in prevalence_by_intervention.values()) / len(prevalence_by_intervention) - development_prevalence) > 1e-12:
        raise X1EvidenceError("baseline aggregate prevalence does not match per-intervention prevalence")
    try:
        development_document = _load_json(candidate)
        development_values = development_document.get("rows") if isinstance(development_document, Mapping) else development_document
        if not isinstance(development_values, list) or not development_values:
            raise X1EvidenceError("development artifact has no rows")
        if development_split_manifest is not None:
            expected_task_ids = {
                task_id for cluster in development_split_manifest.get("clusters", [])
                for task_id in cluster.get("task_ids", [])
            }
            actual_task_ids = {
                row.get("task_id") for row in development_values if isinstance(row, Mapping)
            }
            if actual_task_ids != expected_task_ids:
                raise X1EvidenceError("development artifact task IDs do not match the development split")
        observed_values: list[int] = []
        for row in development_values:
            outcomes = row.get("outcomes") if isinstance(row, Mapping) else None
            if not isinstance(outcomes, Mapping) or not outcomes:
                raise X1EvidenceError("development artifact has an invalid outcome row")
            for value in outcomes.values():
                if isinstance(value, Mapping):
                    value = value.get("repaired")
                if isinstance(value, bool):
                    value = int(value)
                if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
                    raise X1EvidenceError("development artifact outcome is not binary")
                observed_values.append(value)
        observed_prevalence = sum(observed_values) / len(observed_values)
        if abs(observed_prevalence - development_prevalence) > 1e-12:
            raise X1EvidenceError("declared development prevalence does not match the development artifact")
    except X1EvidenceError as exc:
        raise X1EvidenceError("baseline development artifact validation failed: " + str(exc)) from exc
    if expected_development_cohort_id is not None and commitment.get("development_cohort_id") != expected_development_cohort_id:
        raise X1EvidenceError("baseline development cohort is not the frozen development cohort")
    selection_lambda = _finite(commitment.get("selection_lambda"), "baseline.selection_lambda")
    if expected_selection_lambda is not None and selection_lambda != expected_selection_lambda:
        raise X1EvidenceError("baseline selection lambda does not match protocol")
    if commitment.get("fit_scope") != "DEVELOPMENT_ONLY":
        raise X1EvidenceError("baselines must be fit on development data only")
    if commitment.get("surface_shortcut_diagnostic_only") is not True:
        raise X1EvidenceError("surface shortcut must be diagnostic-only")
    baselines = commitment.get("baselines")
    if not isinstance(baselines, Mapping) or set(baselines) != set(PREDICTION_BASELINES):
        raise X1EvidenceError("baseline set does not match the frozen baseline registry")
    normalized = {name: _normalize_prediction_map(value, task_ids, intervention_ids, name=f"baseline.{name}") for name, value in baselines.items()}
    body = _without(commitment, "commitment_sha256")
    expected = commitment.get("commitment_sha256")
    if not isinstance(expected, str) or not HEX64.fullmatch(expected) or _sha_json(body) != expected:
        raise X1EvidenceError("baseline commitment hash mismatch")
    result = dict(commitment)
    result["baselines"] = normalized
    return result


def _subject_and_basis_hashes(
    protocol: Mapping[str, Any], subject_manifest: Mapping[str, Any], basis_qualification: Mapping[str, Any],
    *, registry_path: str | Path | None = None, checkpoint_root: str | Path | None = None,
    release_manifest: Mapping[str, Any] | None = None, release_root: str | Path | None = None,
    source_root: str | Path | None = None, basis_split_manifest: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    protocol_sha = require_frozen_protocol(protocol, repo_root=source_root, check_source_closure=True)
    if release_manifest is None:
        raise X1EvidenceError("release manifest is required before a scientific receipt")
    release_verdict = validate_release_manifest(release_manifest, protocol, repo_root=release_root)
    if not release_verdict["valid"]:
        raise X1EvidenceError("release manifest is invalid: " + "; ".join(release_verdict["errors"]))
    subject_verdict = validate_subject_manifest(
        subject_manifest, registry_path=registry_path, checkpoint_root=checkpoint_root,
        verify_checkpoint_file=True,
    )
    if not subject_verdict["valid"]:
        raise X1EvidenceError("subject manifest is not eligible: " + "; ".join(subject_verdict["errors"]))
    subject_sha = subject_verdict["subject_manifest_sha256"]
    basis_verdict = validate_basis_qualification(
        basis_qualification,
        expected_protocol_sha256=protocol_sha,
        expected_subject_manifest_sha256=subject_sha,
        expected_registry_sha256=protocol.get("intervention_registry_sha256"),
        registry=protocol.get("interventions"),
        basis_split_manifest=basis_split_manifest,
    )
    if not basis_verdict["valid"]:
        raise X1EvidenceError("basis qualification is not promotion-grade: " + "; ".join(basis_verdict["errors"]))
    if basis_qualification.get("thresholds") != protocol.get("decision_thresholds", {}).get("basis"):
        raise X1EvidenceError("basis qualification thresholds do not match the frozen protocol")
    return protocol_sha, subject_sha


def commit_prediction_receipt(
    protocol: Mapping[str, Any], subject_manifest: Mapping[str, Any], split_manifest: Mapping[str, Any],
    public_tasks: Sequence[Mapping[str, Any]], predictor_predictions: Any,
    baseline_commitment: Mapping[str, Any], predictor_identity: Mapping[str, Any],
    basis_qualification: Mapping[str, Any], *, cohort_id: str,
    registry_path: str | Path | None = None, checkpoint_root: str | Path | None = None,
    release_manifest: Mapping[str, Any] | None = None, release_root: str | Path | None = None,
    artifact_root: str | Path | None = None, development_split_manifest: Mapping[str, Any] | None = None,
    basis_split_manifest: Mapping[str, Any] | None = None, source_root: str | Path | None = None,
) -> dict[str, Any]:
    protocol_sha, subject_sha = _subject_and_basis_hashes(
        protocol, subject_manifest, basis_qualification, registry_path=registry_path,
        checkpoint_root=checkpoint_root, release_manifest=release_manifest,
        release_root=release_root, source_root=source_root,
        basis_split_manifest=basis_split_manifest,
    )
    cohort_spec = next((value for value in protocol.get("cohorts", {}).values() if value.get("cohort_id") == cohort_id), None)
    expected_role = "REPLICATION" if cohort_spec and cohort_spec.get("role") == "REPLICATION" else "EVALUATION"
    if cohort_spec is None or cohort_spec.get("role") not in {"EVALUATION", "REPLICATION"}:
        raise X1EvidenceError("prediction cohort is not a frozen evaluation or replication cohort")
    split_verdict = validate_split_manifest(
        split_manifest, expected_protocol_sha256=protocol_sha,
        expected_cohort_id=cohort_id, expected_cohort_role=expected_role,
        expected_seed=int(cohort_spec["seed"]), expected_cohort_plan_sha256=cohort_spec.get("cohort_plan_sha256"),
    )
    if not split_verdict["valid"]:
        raise X1EvidenceError("split manifest is invalid: " + "; ".join(split_verdict["errors"]))
    if split_manifest.get("cohort_id") != cohort_id:
        raise X1EvidenceError("prediction cohort does not match split manifest")
    if split_manifest.get("cohort_role") == "DEVELOPMENT":
        raise X1EvidenceError("X1 prediction receipt cannot target the development cohort")
    development_spec = protocol.get("cohorts", {}).get("development", {})
    basis_spec = protocol.get("cohorts", {}).get("basis_qualification", {})
    if development_split_manifest is None:
        raise X1EvidenceError("development split manifest is required")
    development_verdict = validate_split_manifest(
        development_split_manifest, expected_protocol_sha256=protocol_sha,
        expected_cohort_id=development_spec.get("cohort_id"), expected_cohort_role="DEVELOPMENT",
        expected_seed=int(development_spec.get("seed")),
        expected_cohort_plan_sha256=development_spec.get("cohort_plan_sha256"),
    )
    if not development_verdict["valid"]:
        raise X1EvidenceError("development split is invalid: " + "; ".join(development_verdict["errors"]))
    if development_verdict["n_clusters"] < int(development_spec.get("minimum_independent_clusters", 0)):
        raise X1EvidenceError("development split is below the frozen independent-cluster minimum")
    if basis_split_manifest is None:
        raise X1EvidenceError("basis qualification split manifest is required")
    basis_verdict_split = validate_split_manifest(
        basis_split_manifest, expected_protocol_sha256=protocol_sha,
        expected_cohort_id=basis_spec.get("cohort_id"), expected_cohort_role="QUALIFICATION",
        expected_seed=int(basis_spec.get("seed")),
        expected_cohort_plan_sha256=basis_spec.get("cohort_plan_sha256"),
    )
    if not basis_verdict_split["valid"]:
        raise X1EvidenceError("basis split is invalid: " + "; ".join(basis_verdict_split["errors"]))
    if basis_verdict_split["n_clusters"] < int(basis_spec.get("minimum_independent_clusters", 0)):
        raise X1EvidenceError("basis split is below the frozen independent-cluster minimum")
    for prior_name, prior_split in (("development", development_split_manifest), ("basis", basis_split_manifest)):
        independence = validate_split_independence(prior_split, split_manifest)
        if not independence["valid"]:
            raise X1EvidenceError(f"{prior_name}/evaluation split overlap: " + "; ".join(independence["errors"]))
    basis_independence = validate_split_independence(basis_split_manifest, development_split_manifest)
    if not basis_independence["valid"]:
        raise X1EvidenceError("basis/development split overlap: " + "; ".join(basis_independence["errors"]))
    basis_binding = basis_qualification.get("basis_binding", {})
    if basis_binding.get("basis_split_manifest_sha256") != basis_split_manifest.get("manifest_sha256"):
        raise X1EvidenceError("basis qualification is not bound to the supplied basis split")
    basis_phase_path = basis_binding.get("basis_phase_artifact_path")
    if not isinstance(basis_phase_path, str):
        raise X1EvidenceError("basis qualification phase artifact is missing")
    basis_phase_document = _load_json(basis_phase_path)
    basis_phase_verdict = validate_phase_commit(
        basis_phase_document, expected_phase="BASIS_QUALIFIED",
        expected_receipt_sha256=basis_binding.get("basis_source_artifact_sha256"),
        repo_root=release_root,
    )
    if not basis_phase_verdict["valid"]:
        raise X1EvidenceError("basis qualification phase commitment is invalid")
    if _sha_file(basis_phase_path) != basis_binding.get("basis_phase_commit_sha256"):
        raise X1EvidenceError("basis phase file is not bound to the basis qualification")
    if baseline_commitment.get("development_split_manifest_sha256") != development_split_manifest.get("manifest_sha256"):
        raise X1EvidenceError("baseline commitment is not bound to the supplied development split")
    task_bundle = validate_public_task_bundle(public_tasks, split_manifest=split_manifest)
    intervention_ids = _protocol_intervention_ids(protocol)
    predictions = _normalize_prediction_map(predictor_predictions, task_bundle["task_ids"], intervention_ids, name="predictor")
    _validate_predictor_identity(
        predictor_identity, artifact_root=artifact_root,
        expected_feature_schema_sha256=_sha_json(protocol.get("observations", {}).get("feature_fields", [])),
        expected_training_split_manifest_sha256=development_split_manifest.get("manifest_sha256"),
        expected_development_task_ids=[
            task_id for cluster in development_split_manifest.get("clusters", [])
            for task_id in cluster.get("task_ids", [])
        ],
    )
    if predictor_identity.get("training_split_manifest_sha256") != development_split_manifest.get("manifest_sha256"):
        raise X1EvidenceError("predictor training split does not match the frozen development split")
    baselines = _validate_baseline_commitment(
        baseline_commitment, task_bundle["task_ids"], intervention_ids,
        expected_development_cohort_id=protocol.get("cohorts", {}).get("development", {}).get("cohort_id"),
        expected_selection_lambda=float(protocol.get("cost_model", {}).get("utility_lambda")),
        artifact_root=artifact_root,
        development_split_manifest=development_split_manifest,
    )
    identity = protocol.get("identity")
    body = {
        "schema": PREDICTION_SCHEMA,
        "phase": "PREDICT_COMMITTED",
        "phase_index": 1,
        "protocol_sha256": protocol_sha,
        "subject_manifest_sha256": subject_sha,
        "subject_identity": _subject_identity(subject_manifest),
        "cohort_id": cohort_id,
        "split_manifest_sha256": split_manifest.get("manifest_sha256"),
        "basis_split_manifest_sha256": basis_split_manifest.get("manifest_sha256"),
        "development_split_manifest_sha256": development_split_manifest.get("manifest_sha256"),
        "public_task_bundle_sha256": task_bundle["bundle_sha256"],
        "intervention_registry_sha256": protocol.get("intervention_registry_sha256"),
        "basis_qualification_sha256": basis_qualification.get("basis_qualification_sha256"),
        "basis_phase_commit_sha256": basis_binding.get("basis_phase_commit_sha256"),
        "source_revision": identity.get("base_revision"),
        "source_closure_sha256": identity.get("source_closure_sha256"),
        "release_manifest_sha256": release_manifest.get("release_manifest_sha256"),
        "predictor": dict(predictor_identity),
        "baseline_commitment_sha256": baselines["commitment_sha256"],
        "baseline_commitment": dict(baseline_commitment),
        "baselines": baselines["baselines"],
        "task_manifest": [
            {"task_id": task_id, "observation_sha256": _sha_json(task_bundle["tasks"][index])}
            for index, task_id in enumerate(task_bundle["task_ids"])
        ],
        "predictions": [predictions[task_id] for task_id in task_bundle["task_ids"]],
    }
    _assert_no_policy_leak(body)
    body["prediction_receipt_sha256"] = _sha_json(body)
    body["receipt_sha256"] = body["prediction_receipt_sha256"]
    return body


def validate_prediction_receipt(
    receipt: Mapping[str, Any], *, protocol: Mapping[str, Any] | None = None,
    subject_manifest: Mapping[str, Any] | None = None,
    public_tasks: Sequence[Mapping[str, Any]] | None = None,
    split_manifest: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None, checkpoint_root: str | Path | None = None,
    release_manifest: Mapping[str, Any] | None = None, release_root: str | Path | None = None,
    basis_qualification: Mapping[str, Any] | None = None,
    basis_subject_manifest_sha256: str | None = None, source_root: str | Path | None = None,
    development_split_manifest: Mapping[str, Any] | None = None,
    basis_split_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        _assert_no_policy_leak(receipt)
    except X1EvidenceError as exc:
        errors.append(str(exc))
    try:
        _assert_exact_keys(receipt, {
            "schema", "phase", "phase_index", "protocol_sha256", "subject_manifest_sha256",
            "subject_identity", "cohort_id", "split_manifest_sha256", "basis_split_manifest_sha256",
            "development_split_manifest_sha256", "public_task_bundle_sha256", "intervention_registry_sha256",
            "basis_qualification_sha256", "basis_phase_commit_sha256", "source_revision", "source_closure_sha256",
            "release_manifest_sha256", "predictor", "baseline_commitment_sha256",
            "baseline_commitment", "baselines", "task_manifest", "predictions",
            "prediction_receipt_sha256", "receipt_sha256",
        })
    except X1EvidenceError as exc:
        errors.append(str(exc))
    if receipt.get("schema") != PREDICTION_SCHEMA or receipt.get("phase") != "PREDICT_COMMITTED":
        errors.append("prediction receipt schema or phase mismatch")
    if receipt.get("phase_index") != 1:
        errors.append("prediction phase index must be 1")
    prediction_hash = receipt.get("prediction_receipt_sha256")
    if not isinstance(prediction_hash, str) or not HEX64.fullmatch(prediction_hash):
        errors.append("prediction receipt hash is malformed")
    else:
        body = _without(receipt, "prediction_receipt_sha256", "receipt_sha256")
        if _sha_json(body) != prediction_hash:
            errors.append("prediction receipt hash mismatch")
    if receipt.get("receipt_sha256") != prediction_hash:
        errors.append("prediction receipt alias hash mismatch")
    if protocol is not None:
        try:
            expected = require_frozen_protocol(protocol, repo_root=source_root, check_source_closure=True)
            if receipt.get("protocol_sha256") != expected:
                errors.append("prediction protocol binding mismatch")
            if receipt.get("intervention_registry_sha256") != protocol.get("intervention_registry_sha256"):
                errors.append("prediction intervention registry mismatch")
            if receipt.get("source_revision") != protocol.get("identity", {}).get("base_revision"):
                errors.append("prediction source revision mismatch")
            if receipt.get("source_closure_sha256") != protocol.get("identity", {}).get("source_closure_sha256"):
                errors.append("prediction source closure mismatch")
        except X1EvidenceError as exc:
            errors.append(str(exc))
    if release_manifest is not None and protocol is not None:
        release_verdict = validate_release_manifest(release_manifest, protocol, repo_root=release_root)
        if not release_verdict["valid"]:
            errors.extend(release_verdict["errors"])
        if receipt.get("release_manifest_sha256") != release_manifest.get("release_manifest_sha256"):
            errors.append("prediction release binding mismatch")
    if basis_qualification is not None and protocol is not None:
        basis_verdict = validate_basis_qualification(
            basis_qualification, expected_protocol_sha256=receipt.get("protocol_sha256"),
            expected_subject_manifest_sha256=basis_subject_manifest_sha256,
            expected_registry_sha256=protocol.get("intervention_registry_sha256"),
            registry=protocol.get("interventions"),
            basis_split_manifest=basis_split_manifest,
        )
        if not basis_verdict["valid"]:
            errors.append("prediction basis binding is invalid")
        if receipt.get("basis_qualification_sha256") != basis_qualification.get("basis_qualification_sha256"):
            errors.append("prediction basis qualification mismatch")
        if receipt.get("basis_phase_commit_sha256") != basis_qualification.get("basis_binding", {}).get("basis_phase_commit_sha256"):
            errors.append("prediction basis phase parent mismatch")
        try:
            _hash(receipt.get("basis_phase_commit_sha256"), "prediction.basis_phase_commit_sha256")
        except X1EvidenceError as exc:
            errors.append(str(exc))
    if protocol is not None and isinstance(receipt.get("predictor"), Mapping):
        try:
            _validate_predictor_identity(
                receipt["predictor"], artifact_root=release_root,
                expected_feature_schema_sha256=_sha_json(protocol.get("observations", {}).get("feature_fields", [])),
                expected_training_split_manifest_sha256=receipt.get("development_split_manifest_sha256"),
                expected_development_task_ids=(
                    [task_id for cluster in development_split_manifest.get("clusters", []) for task_id in cluster.get("task_ids", [])]
                    if development_split_manifest is not None else None
                ),
            )
        except X1EvidenceError as exc:
            errors.append("prediction predictor identity is invalid: " + str(exc))
    if subject_manifest is not None:
        try:
            subject = validate_subject_manifest(
                subject_manifest, registry_path=registry_path, checkpoint_root=checkpoint_root,
                verify_checkpoint_file=True,
            )
            if not subject["valid"]:
                errors.append("prediction subject is not eligible")
            if receipt.get("subject_manifest_sha256") != subject["subject_manifest_sha256"]:
                errors.append("prediction subject binding mismatch")
        except X1EvidenceError as exc:
            errors.append(str(exc))
    if split_manifest is not None:
        split_cohort_spec = next((
            value for value in (protocol or {}).get("cohorts", {}).values()
            if value.get("cohort_id") == split_manifest.get("cohort_id")
        ), {})
        split = validate_split_manifest(
            split_manifest, expected_protocol_sha256=receipt.get("protocol_sha256"),
            expected_cohort_id=split_manifest.get("cohort_id"),
            expected_cohort_role=split_manifest.get("cohort_role"),
            expected_seed=split_manifest.get("seed"),
            expected_cohort_plan_sha256=split_cohort_spec.get("cohort_plan_sha256"),
        )
        if not split["valid"]:
            errors.append("prediction split manifest is invalid")
        if receipt.get("split_manifest_sha256") != split_manifest.get("manifest_sha256"):
            errors.append("prediction split binding mismatch")
    task_manifest = receipt.get("task_manifest")
    predictions = receipt.get("predictions")
    baselines = receipt.get("baselines")
    baseline_commitment = receipt.get("baseline_commitment")
    if not isinstance(baseline_commitment, Mapping):
        errors.append("baseline commitment is missing")
    else:
        commitment_body = _without(baseline_commitment, "commitment_sha256")
        commitment_hash = baseline_commitment.get("commitment_sha256")
        if not isinstance(commitment_hash, str) or not HEX64.fullmatch(commitment_hash) or _sha_json(commitment_body) != commitment_hash:
            errors.append("baseline commitment hash mismatch")
        if commitment_hash != receipt.get("baseline_commitment_sha256"):
            errors.append("baseline commitment binding mismatch")
    if not isinstance(task_manifest, list) or not isinstance(predictions, list) or not task_manifest:
        errors.append("prediction task manifest or rows are missing")
        return {"valid": False, "errors": errors}
    task_ids = []
    for item in task_manifest:
        if not isinstance(item, Mapping) or set(item) != {"task_id", "observation_sha256"}:
            errors.append("invalid prediction task manifest row")
            continue
        task_ids.append(item.get("task_id"))
        try:
            _hash(item.get("observation_sha256"), "observation_sha256")
        except X1EvidenceError as exc:
            errors.append(str(exc))
    if len(set(task_ids)) != len(task_ids):
        errors.append("duplicate prediction task IDs")
    if len(predictions) != len(task_ids):
        errors.append("prediction row count does not match task manifest")
    intervention_ids = _protocol_intervention_ids(protocol) if protocol is not None else None
    if intervention_ids is None:
        try:
            candidate = predictions[0].get("candidate_interventions")
            if not isinstance(candidate, list) or not candidate:
                raise X1EvidenceError("prediction candidate registry is missing")
            intervention_ids = [str(item) for item in candidate]
        except (AttributeError, IndexError, TypeError) as exc:
            errors.append("prediction candidate registry is missing")
    if intervention_ids is not None:
        try:
            normalized = _normalize_prediction_map(predictions, task_ids, intervention_ids, name="prediction")
            if [row["task_id"] for row in normalized.values()] != task_ids:
                errors.append("prediction rows are not in task manifest order")
            if baselines is not None:
                if not isinstance(baselines, Mapping):
                    raise X1EvidenceError("baseline prediction registry is not an object")
                normalized_baselines = {}
                for baseline_name, baseline_rows in baselines.items():
                    normalized_baselines[baseline_name] = _normalize_prediction_map(
                        baseline_rows, task_ids, intervention_ids,
                        name=f"baseline.{baseline_name}",
                    )
                if isinstance(baseline_commitment, Mapping):
                    validated_baseline = _validate_baseline_commitment(
                        baseline_commitment, task_ids, intervention_ids,
                        expected_development_cohort_id=(protocol or {}).get("cohorts", {}).get("development", {}).get("cohort_id"),
                        expected_selection_lambda=float((protocol or {}).get("cost_model", {}).get("utility_lambda", 0.25)),
                        artifact_root=release_root,
                        development_split_manifest=development_split_manifest,
                    )
                    if _sha_json(validated_baseline["baselines"]) != _sha_json(normalized_baselines):
                        errors.append("baseline predictions do not match the committed baseline artifact")
        except X1EvidenceError as exc:
            errors.append(str(exc))
    if public_tasks is not None:
        try:
            bundle = validate_public_task_bundle(public_tasks, split_manifest=split_manifest)
            if receipt.get("public_task_bundle_sha256") != bundle["bundle_sha256"]:
                errors.append("public task bundle hash mismatch")
            for index, task_id in enumerate(bundle["task_ids"]):
                if receipt["task_manifest"][index].get("observation_sha256") != _sha_json(bundle["tasks"][index]):
                    errors.append(f"observation hash mismatch: {task_id}")
        except X1EvidenceError as exc:
            errors.append(str(exc))
    return {"valid": not errors, "errors": errors, "prediction_receipt_sha256": prediction_hash}


def _normalize_outcome_rows(
    outcome_rows: Any, task_ids: Sequence[str], intervention_ids: Sequence[str],
    split_task_to_cluster: Mapping[str, str] | None = None,
    split_task_to_source: Mapping[str, str] | None = None,
    artifact_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    if isinstance(outcome_rows, Mapping):
        raw_items = []
        for task_id, values in outcome_rows.items():
            if not isinstance(values, Mapping):
                raise X1EvidenceError("outcome mapping values must be objects")
            for intervention_id, value in values.items():
                item = {"task_id": task_id, "intervention_id": intervention_id, "value": value}
                if isinstance(value, Mapping):
                    item.update(value)
                else:
                    item["repaired"] = bool(value)
                raw_items.append(item)
    elif isinstance(outcome_rows, Sequence) and not isinstance(outcome_rows, (str, bytes)):
        if any(not isinstance(item, Mapping) for item in outcome_rows):
            raise X1EvidenceError("outcome rows contain a malformed record")
        raw_items = [dict(item) for item in outcome_rows]
    else:
        raise X1EvidenceError("outcome rows must be a mapping or record list")
    normalized = []
    seen: set[tuple[str, str]] = set()
    task_set = set(task_ids)
    for item in raw_items:
        task_id = item.get("task_id")
        intervention_id = item.get("intervention_id")
        if task_id not in task_set or intervention_id not in intervention_ids:
            raise X1EvidenceError(f"outcome row has unknown task/intervention: {task_id}/{intervention_id}")
        key = (task_id, intervention_id)
        if key in seen:
            raise X1EvidenceError(f"duplicate outcome row: {task_id}/{intervention_id}")
        seen.add(key)
        repaired = item.get("repaired")
        if not isinstance(repaired, bool):
            raise X1EvidenceError(f"outcome repaired flag must be boolean: {task_id}/{intervention_id}")
        baseline_failed = item.get("baseline_failed")
        if not isinstance(baseline_failed, bool) or baseline_failed is not True:
            raise X1EvidenceError(f"outcome row is not an explicit baseline failure: {task_id}/{intervention_id}")
        effect = _finite(item.get("effect", 0.0), f"effect:{task_id}/{intervention_id}")
        raw_output = item.get("raw_output")
        raw_hash = item.get("raw_output_sha256")
        if raw_output is not None:
            if not isinstance(raw_output, str):
                raise X1EvidenceError("raw_output must be text")
            computed = _sha_bytes(raw_output.encode("utf-8"))
            if raw_hash is not None and raw_hash != computed:
                raise X1EvidenceError(f"raw output hash mismatch: {task_id}/{intervention_id}")
            raw_hash = computed
        try:
            _hash(raw_hash, f"raw_output_sha256:{task_id}/{intervention_id}")
            _hash(item.get("verifier_receipt_sha256"), f"verifier_receipt_sha256:{task_id}/{intervention_id}")
            _hash(item.get("gold_sha256"), f"gold_sha256:{task_id}/{intervention_id}")
            transformed_prompt = item.get("transformed_prompt")
            if not isinstance(transformed_prompt, str) or not transformed_prompt:
                raise X1EvidenceError(f"transformed prompt is missing: {task_id}/{intervention_id}")
            if _sha_bytes(transformed_prompt.encode("utf-8")) != item.get("transformed_prompt_sha256"):
                raise X1EvidenceError(f"transformed prompt hash mismatch: {task_id}/{intervention_id}")
            _hash(item.get("transformed_prompt_sha256"), f"transformed_prompt_sha256:{task_id}/{intervention_id}")
            _hash(item.get("verifier_artifact_sha256"), f"verifier_artifact_sha256:{task_id}/{intervention_id}")
        except X1EvidenceError as exc:
            raise X1EvidenceError(str(exc)) from exc
        verifier_artifact_path = item.get("verifier_artifact_path")
        if not isinstance(verifier_artifact_path, str) or not verifier_artifact_path:
            raise X1EvidenceError(f"verifier artifact path is missing: {task_id}/{intervention_id}")
        verifier_path = Path(verifier_artifact_path)
        if not verifier_path.is_absolute():
            verifier_path = Path(artifact_root or Path.cwd()) / verifier_path
        if not verifier_path.is_file() or _sha_file(verifier_path) != item.get("verifier_artifact_sha256"):
            raise X1EvidenceError(f"verifier artifact hash verification failed: {task_id}/{intervention_id}")
        if item.get("verifier_receipt_sha256") != item.get("verifier_artifact_sha256"):
            raise X1EvidenceError(f"verifier receipt is not the file-bound artifact: {task_id}/{intervention_id}")
        _verify_verifier_artifact(
            verifier_path, task_id=task_id, intervention_id=intervention_id,
            repaired=repaired, raw_output_sha256=raw_hash,
            gold_sha256=item.get("gold_sha256"),
            transformed_prompt_sha256=item.get("transformed_prompt_sha256"),
        )
        cluster_id = item.get("cluster_id")
        source_id = item.get("source_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise X1EvidenceError(f"outcome row lacks cluster_id: {task_id}/{intervention_id}")
        if not isinstance(source_id, str) or not source_id:
            raise X1EvidenceError(f"outcome row lacks source_id: {task_id}/{intervention_id}")
        if split_task_to_cluster is not None and split_task_to_cluster.get(task_id) != cluster_id:
            raise X1EvidenceError(f"cluster identity mismatch: {task_id}")
        if split_task_to_source is not None and split_task_to_source.get(task_id) != source_id:
            raise X1EvidenceError(f"source identity mismatch: {task_id}")
        normalized.append({
            "task_id": task_id,
            "intervention_id": intervention_id,
            "cluster_id": cluster_id,
            "source_id": source_id,
            "repaired": repaired,
            "baseline_failed": baseline_failed,
            "effect": effect,
            "raw_output_sha256": raw_hash,
            "verifier_receipt_sha256": item.get("verifier_receipt_sha256"),
            "gold_sha256": item.get("gold_sha256"),
            "transformed_prompt_sha256": item.get("transformed_prompt_sha256"),
            "transformed_prompt": transformed_prompt,
            "verifier_artifact_path": str(verifier_path),
            "verifier_artifact_sha256": item.get("verifier_artifact_sha256"),
        })
    expected = {(task_id, intervention_id) for task_id in task_ids for intervention_id in intervention_ids}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise X1EvidenceError(f"outcome coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")
    return sorted(normalized, key=lambda item: (item["task_id"], intervention_ids.index(item["intervention_id"])))


def _split_identity_maps(split_manifest: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    task_to_cluster: dict[str, str] = {}
    task_to_source: dict[str, str] = {}
    for cluster in split_manifest.get("clusters", []):
        if not isinstance(cluster, Mapping):
            continue
        for task_id in cluster.get("task_ids", []):
            task_to_cluster[task_id] = cluster.get("cluster_id")
            task_to_source[task_id] = cluster.get("source_id")
    return task_to_cluster, task_to_source


def commit_reveal_receipt(
    prediction_receipt: Mapping[str, Any], outcome_rows: Any, *, evaluator_identity: Mapping[str, Any],
    split_manifest: Mapping[str, Any], prediction_commit: Mapping[str, Any],
    protocol: Mapping[str, Any], subject_manifest: Mapping[str, Any],
    registry_path: str | Path | None = None, checkpoint_root: str | Path | None = None,
    release_manifest: Mapping[str, Any] | None = None, release_root: str | Path | None = None,
    public_tasks: Sequence[Mapping[str, Any]] | None = None,
    basis_qualification: Mapping[str, Any] | None = None, source_root: str | Path | None = None,
) -> dict[str, Any]:
    if public_tasks is None or basis_qualification is None or release_manifest is None:
        raise X1EvidenceError("public tasks, basis qualification, and release manifest are required at reveal")
    prediction_verdict = validate_prediction_receipt(
        prediction_receipt, protocol=protocol, subject_manifest=subject_manifest,
        public_tasks=public_tasks, split_manifest=split_manifest,
        registry_path=registry_path, checkpoint_root=checkpoint_root,
        release_manifest=release_manifest, release_root=release_root,
        basis_qualification=basis_qualification,
        basis_subject_manifest_sha256=_sha_json(subject_manifest), source_root=source_root,
    )
    if not prediction_verdict["valid"]:
        raise X1ChronologyError("prediction receipt is invalid: " + "; ".join(prediction_verdict["errors"]))
    basis_phase_path = basis_qualification.get("basis_binding", {}).get("basis_phase_artifact_path")
    basis_phase_document = _load_json(basis_phase_path) if isinstance(basis_phase_path, str) else None
    commit_verdict = validate_phase_commit(
        prediction_commit, expected_phase="PREDICT_COMMITTED",
        expected_receipt_sha256=prediction_receipt.get("prediction_receipt_sha256"),
        expected_previous_commit_sha256=prediction_receipt.get("basis_phase_commit_sha256"),
        repo_root=release_root, previous_commit=basis_phase_document,
    )
    if not commit_verdict["valid"]:
        raise X1ChronologyError("prediction phase commitment is invalid: " + "; ".join(commit_verdict["errors"]))
    if evaluator_identity.get("schema") != "anra-x1-real-1-evaluator/v1":
        raise X1EvidenceError("evaluator identity schema mismatch")
    _assert_exact_keys(
        evaluator_identity,
        {"schema", "source_sha256", "verifier_source_sha256", "execution_artifact_sha256"},
    )
    for key in ("source_sha256", "verifier_source_sha256", "execution_artifact_sha256"):
        _hash(evaluator_identity.get(key), f"evaluator.{key}")
    split_verdict = validate_split_manifest(split_manifest, expected_protocol_sha256=prediction_receipt.get("protocol_sha256"))
    if not split_verdict["valid"]:
        raise X1EvidenceError("split manifest is invalid at reveal")
    if split_manifest.get("cohort_id") != prediction_receipt.get("cohort_id"):
        raise X1EvidenceError("reveal cohort does not match prediction cohort")
    if split_manifest.get("manifest_sha256") != prediction_receipt.get("split_manifest_sha256"):
        raise X1EvidenceError("reveal split does not match the prediction split")
    task_manifest = prediction_receipt.get("task_manifest", [])
    task_ids = [item.get("task_id") for item in task_manifest]
    intervention_ids = [row.get("candidate_interventions", []) for row in prediction_receipt.get("predictions", [])]
    if not intervention_ids or any(item != intervention_ids[0] for item in intervention_ids[1:]):
        raise X1EvidenceError("prediction candidate registries are inconsistent")
    task_to_cluster, task_to_source = _split_identity_maps(split_manifest)
    rows = _normalize_outcome_rows(
        outcome_rows, task_ids, intervention_ids[0], task_to_cluster, task_to_source,
        artifact_root=release_root,
    )
    body = {
        "schema": REVEAL_SCHEMA,
        "phase": "REVEAL_ACCEPTED",
        "phase_index": 2,
        "prediction_receipt_sha256": prediction_receipt.get("prediction_receipt_sha256"),
        "prediction_phase_commit_sha256": prediction_commit.get("phase_commit_sha256"),
        "protocol_sha256": prediction_receipt.get("protocol_sha256"),
        "subject_manifest_sha256": prediction_receipt.get("subject_manifest_sha256"),
        "cohort_id": prediction_receipt.get("cohort_id"),
        "split_manifest_sha256": split_manifest.get("manifest_sha256"),
        "intervention_registry_sha256": prediction_receipt.get("intervention_registry_sha256"),
        "evaluator": dict(evaluator_identity),
        "rows": rows,
        "rows_sha256": _sha_json(rows),
    }
    body["reveal_receipt_sha256"] = _sha_json(body)
    body["receipt_sha256"] = body["reveal_receipt_sha256"]
    return body


def validate_reveal_receipt(
    receipt: Mapping[str, Any], *, prediction_receipt: Mapping[str, Any] | None = None,
    split_manifest: Mapping[str, Any] | None = None,
    prediction_commit: Mapping[str, Any] | None = None,
    reveal_commit: Mapping[str, Any] | None = None, repo_root: str | Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        _assert_exact_keys(receipt, {
            "schema", "phase", "phase_index", "prediction_receipt_sha256",
            "prediction_phase_commit_sha256", "protocol_sha256", "subject_manifest_sha256",
            "cohort_id", "split_manifest_sha256", "intervention_registry_sha256",
            "evaluator", "rows", "rows_sha256", "reveal_receipt_sha256", "receipt_sha256",
        })
    except X1EvidenceError as exc:
        errors.append(str(exc))
    if receipt.get("schema") != REVEAL_SCHEMA or receipt.get("phase") != "REVEAL_ACCEPTED":
        errors.append("reveal receipt schema or phase mismatch")
    if prediction_commit is None:
        errors.append("prediction phase commitment is required")
    elif prediction_receipt is not None:
        commit_verdict = validate_phase_commit(
            prediction_commit, expected_phase="PREDICT_COMMITTED",
            expected_receipt_sha256=prediction_receipt.get("prediction_receipt_sha256"),
            expected_previous_commit_sha256=prediction_receipt.get("basis_phase_commit_sha256"),
            repo_root=repo_root,
        )
        errors.extend(commit_verdict["errors"])
        if receipt.get("prediction_phase_commit_sha256") != prediction_commit.get("phase_commit_sha256"):
            errors.append("reveal does not bind the prediction phase commitment")
    if reveal_commit is None:
        errors.append("reveal phase commitment is required")
    else:
        reveal_commit_verdict = validate_phase_commit(
            reveal_commit, expected_phase="REVEAL_ACCEPTED",
            expected_receipt_sha256=receipt.get("reveal_receipt_sha256"),
            expected_previous_commit_sha256=prediction_commit.get("phase_commit_sha256") if prediction_commit else None,
            repo_root=repo_root, previous_commit=prediction_commit,
        )
        errors.extend(reveal_commit_verdict["errors"])
    if receipt.get("phase_index") != 2:
        errors.append("reveal phase index must be 2")
    reveal_hash = receipt.get("reveal_receipt_sha256")
    if not isinstance(reveal_hash, str) or not HEX64.fullmatch(reveal_hash):
        errors.append("reveal receipt hash is malformed")
    elif _sha_json(_without(receipt, "reveal_receipt_sha256", "receipt_sha256")) != reveal_hash:
        errors.append("reveal receipt hash mismatch")
    if receipt.get("receipt_sha256") != reveal_hash:
        errors.append("reveal receipt alias hash mismatch")
    if prediction_receipt is not None:
        if receipt.get("prediction_receipt_sha256") != prediction_receipt.get("prediction_receipt_sha256"):
            errors.append("reveal does not point to the committed prediction receipt")
        for key in ("protocol_sha256", "subject_manifest_sha256", "cohort_id", "split_manifest_sha256", "intervention_registry_sha256"):
            if receipt.get(key) != prediction_receipt.get(key):
                errors.append(f"reveal {key} does not match prediction")
    if split_manifest is not None:
        expected_plan = FROZEN_COHORT_PLAN_HASHES.get(split_manifest.get("cohort_id"))
        split = validate_split_manifest(
            split_manifest, expected_protocol_sha256=receipt.get("protocol_sha256"),
            expected_cohort_id=split_manifest.get("cohort_id"),
            expected_cohort_role=split_manifest.get("cohort_role"),
            expected_seed=split_manifest.get("seed"),
            expected_cohort_plan_sha256=expected_plan,
        )
        if not split["valid"]:
            errors.append("reveal split manifest is invalid")
    rows = receipt.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("reveal rows are missing")
    elif receipt.get("rows_sha256") != _sha_json(rows):
        errors.append("reveal rows hash mismatch")
    if prediction_receipt is not None and isinstance(rows, list) and rows:
        try:
            task_manifest = prediction_receipt.get("task_manifest", [])
            task_ids = [item.get("task_id") for item in task_manifest]
            candidate_rows = [item.get("candidate_interventions", []) for item in prediction_receipt.get("predictions", [])]
            if not candidate_rows or any(item != candidate_rows[0] for item in candidate_rows[1:]):
                raise X1EvidenceError("prediction candidate registry is inconsistent")
            task_to_cluster, task_to_source = _split_identity_maps(split_manifest) if split_manifest is not None else ({}, {})
            normalized_rows = _normalize_outcome_rows(
                rows, task_ids, candidate_rows[0], task_to_cluster or None, task_to_source or None,
                artifact_root=repo_root,
            )
            if _sha_json(normalized_rows) != _sha_json(rows):
                errors.append("reveal row normalization or ordering mismatch")
        except X1EvidenceError as exc:
            errors.append(str(exc))
    evaluator = receipt.get("evaluator")
    if not isinstance(evaluator, Mapping) or evaluator.get("schema") != "anra-x1-real-1-evaluator/v1":
        errors.append("evaluator identity is missing or invalid")
    else:
        for key in ("source_sha256", "verifier_source_sha256", "execution_artifact_sha256"):
            try:
                _hash(evaluator.get(key), f"evaluator.{key}")
            except X1EvidenceError as exc:
                errors.append(str(exc))
    return {"valid": not errors, "errors": errors, "reveal_receipt_sha256": reveal_hash}


def _tie_safe_ap(scores: Sequence[float], labels: Sequence[int]) -> float:
    if not scores:
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda item: (-item[0], item[1]))
    positives = sum(labels)
    if positives == 0:
        return 0.0
    hits = 0
    average_precision = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        group_positive = sum(label for _, label in pairs[index:end])
        average_precision += (hits + group_positive) / end * group_positive
        hits += group_positive
        index = end
    return min(1.0, average_precision / positives)


def _brier_skill(
    probabilities: Sequence[float], labels: Sequence[int], reference_prevalence: float | None = None
) -> float:
    if not labels:
        return 0.0
    prevalence = sum(labels) / len(labels) if reference_prevalence is None else reference_prevalence
    baseline = sum((prevalence - label) ** 2 for label in labels) / len(labels)
    observed = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(labels)
    return 1.0 - observed / baseline if baseline > 0 else 0.0


def _mcc(predictions: Sequence[int], labels: Sequence[int]) -> float:
    tp = sum(1 for prediction, label in zip(predictions, labels) if prediction and label)
    tn = sum(1 for prediction, label in zip(predictions, labels) if not prediction and not label)
    fp = sum(1 for prediction, label in zip(predictions, labels) if prediction and not label)
    fn = sum(1 for prediction, label in zip(predictions, labels) if not prediction and label)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def _cell_metrics(
    prediction_map: Mapping[str, Mapping[str, Any]], outcome_map: Mapping[tuple[str, str], bool],
    intervention_ids: Sequence[str], reference_prevalence: float | None = None,
    reference_prevalence_by_intervention: Mapping[str, float] | None = None,
    task_clusters: Mapping[str, str] | None = None, _macro: bool = False,
) -> dict[str, Any]:
    if task_clusters is not None and not _macro:
        cluster_ids = sorted(set(task_clusters.values()))
        cluster_reports = []
        for cluster_id in cluster_ids:
            task_ids = {task_id for task_id, value in task_clusters.items() if value == cluster_id}
            cluster_reports.append(_cell_metrics(
                {task_id: prediction_map[task_id] for task_id in task_ids},
                outcome_map, intervention_ids, reference_prevalence,
                reference_prevalence_by_intervention, None, True,
            ))
        if not cluster_reports:
            return {"n_cells": 0, "n_clusters": 0, "score_variance": 0.0}
        aggregate_keys = (
            "positive_prevalence", "raw_accuracy_diagnostic_only", "auprc", "auprc_lift_over_prevalence",
            "brier", "brier_skill_vs_prevalence", "mcc", "calibration_error", "score_variance",
        )
        aggregated = {
            key: round(sum(float(report[key]) for report in cluster_reports if report.get(key) is not None) /
                       sum(report.get(key) is not None for report in cluster_reports), 6)
            if any(report.get(key) is not None for report in cluster_reports) else None
            for key in aggregate_keys
        }
        aggregated.update({
            "n_cells": sum(report["n_cells"] for report in cluster_reports),
            "n_clusters": len(cluster_reports),
            "brier_reference_prevalence": reference_prevalence,
            "brier_reference_prevalence_by_intervention": reference_prevalence_by_intervention,
        })
        return aggregated
    scores: list[float] = []
    labels: list[int] = []
    for task_id in sorted(prediction_map):
        row = prediction_map[task_id]
        for intervention_id, probability in zip(row["candidate_interventions"], row["predicted_repair_probability"]):
            scores.append(float(probability))
            labels.append(int(outcome_map[(task_id, intervention_id)]))
    prevalence = sum(labels) / len(labels) if labels else 0.0
    predictions = [int(score >= 0.5) for score in scores]
    calibration_error = 0.0
    if scores:
        bins = [[] for _ in range(10)]
        for score, label in zip(scores, labels):
            bins[min(9, int(score * 10))].append(label)
        calibration_error = sum(
            abs((sum(bin_labels) / len(bin_labels)) - (sum(labels) / len(labels))) * len(bin_labels)
            for bin_labels in bins if bin_labels
        ) / len(labels)
    brier_skills = []
    for intervention_id in intervention_ids:
        intervention_scores = []
        intervention_labels = []
        for task_id in sorted(prediction_map):
            row = prediction_map[task_id]
            if intervention_id in row["candidate_interventions"]:
                index = row["candidate_interventions"].index(intervention_id)
                intervention_scores.append(float(row["predicted_repair_probability"][index]))
                intervention_labels.append(int(outcome_map[(task_id, intervention_id)]))
        reference = (
            reference_prevalence_by_intervention.get(intervention_id)
            if reference_prevalence_by_intervention is not None else reference_prevalence
        )
        brier_skills.append(_brier_skill(intervention_scores, intervention_labels, reference))
    brier_skill = sum(brier_skills) / len(brier_skills) if brier_skills else 0.0
    return {
        "n_cells": len(scores),
        "positive_prevalence": round(prevalence, 6),
        "raw_accuracy_diagnostic_only": round(
            sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels), 6
        ) if labels else None,
        "auprc": round(_tie_safe_ap(scores, labels), 6),
        "auprc_lift_over_prevalence": round(_tie_safe_ap(scores, labels) - prevalence, 6),
        "brier": round(sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels), 6) if labels else None,
        "brier_skill_vs_prevalence": round(brier_skill, 6),
        "brier_reference_prevalence": round(reference_prevalence, 6) if reference_prevalence is not None else round(prevalence, 6),
        "brier_reference_prevalence_by_intervention": {
            intervention_id: round(float(reference_prevalence_by_intervention.get(intervention_id, reference_prevalence or 0.0)), 6)
            for intervention_id in intervention_ids
        } if reference_prevalence_by_intervention is not None else None,
        "mcc": round(_mcc(predictions, labels), 6),
        "calibration_error": round(calibration_error, 6),
        "score_variance": round(sum((score - sum(scores) / len(scores)) ** 2 for score in scores) / len(scores), 6) if scores else 0.0,
    }


def _policy_metrics(
    prediction_map: Mapping[str, Mapping[str, Any]], outcome_map: Mapping[tuple[str, str], bool],
    task_clusters: Mapping[str, str], intervention_ids: Sequence[str], repair_intervention_ids: Sequence[str],
    costs: Mapping[str, int], lambda_cost: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_task: list[dict[str, Any]] = []
    for task_id in sorted(prediction_map):
        row = prediction_map[task_id]
        probabilities = row["predicted_repair_probability"]
        utilities = [
            probability - lambda_cost * costs[intervention_id]
            for intervention_id, probability in zip(row["candidate_interventions"], probabilities)
        ]
        selected_index = max(
            range(len(utilities)),
            key=lambda index: (utilities[index], probabilities[index], -index),
        )
        selected = row["candidate_interventions"][selected_index]
        no_change = outcome_map[(task_id, "NO_CHANGE")]
        selected_observed = outcome_map[(task_id, selected)]
        selected_repaired = bool(selected in repair_intervention_ids and selected_observed)
        successful = [
            intervention_id for intervention_id in row["candidate_interventions"]
            if intervention_id in repair_intervention_ids and outcome_map[(task_id, intervention_id)]
        ]
        cheapest = min((costs[item] for item in successful), default=None)
        oracle_utility = 1.0 - lambda_cost * cheapest if cheapest is not None else 0.0
        selected_utility = (1.0 if selected_repaired else 0.0) - lambda_cost * costs[selected]
        per_task.append({
            "task_id": task_id,
            "cluster_id": task_clusters[task_id],
            "selected": selected,
            "selected_repaired": bool(selected_repaired),
            "no_change_repaired": bool(no_change),
            "repairable": bool(successful),
            "repair_lift": int(bool(selected_repaired)) - int(bool(no_change)),
            "capture": int(bool(successful) and bool(selected_repaired)),
            "false_intervention": int(not successful and selected != "NO_CHANGE"),
            "selected_cost": costs[selected],
            "oracle_cost": cheapest,
            "selected_utility": selected_utility,
            "oracle_utility": oracle_utility,
            "regret": oracle_utility - selected_utility,
        })
    n = len(per_task)
    repairable = [item for item in per_task if item["repairable"]]
    unrepairable = [item for item in per_task if not item["repairable"]]
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in per_task:
        clusters[item["cluster_id"]].append(item)
    cluster_repairable = [items for items in clusters.values() if any(item["repairable"] for item in items)]
    cluster_unrepairable = [items for items in clusters.values() if any(not item["repairable"] for item in items)]
    summary = {
        "n_tasks": n,
        "n_clusters": len(clusters),
        "repair_capture": round(sum(
            sum(item["capture"] for item in items if item["repairable"]) /
            sum(1 for item in items if item["repairable"])
            for items in cluster_repairable
        ) / len(cluster_repairable), 6) if cluster_repairable else None,
        "false_intervention_rate": round(sum(
            sum(item["false_intervention"] for item in items if not item["repairable"]) /
            sum(1 for item in items if not item["repairable"])
            for items in cluster_unrepairable
        ) / len(cluster_unrepairable), 6) if cluster_unrepairable else None,
        "mean_repair_lift_over_no_change": round(sum(
            sum(item["repair_lift"] for item in items) / len(items) for items in clusters.values()
        ) / len(clusters), 6) if clusters else None,
        "mean_cost_adjusted_utility": round(sum(
            sum(item["selected_utility"] for item in items) / len(items) for items in clusters.values()
        ) / len(clusters), 6) if clusters else None,
        "mean_oracle_normalized_regret": round(sum(
            sum(item["regret"] for item in items) / len(items) for items in clusters.values()
        ) / len(clusters), 6) if clusters else None,
        "mean_excess_cost_vs_oracle": round(sum(
            (sum(item["selected_cost"] - item["oracle_cost"] for item in items if item["oracle_cost"] is not None) /
             sum(1 for item in items if item["oracle_cost"] is not None))
            for items in cluster_repairable
            if any(item["oracle_cost"] is not None for item in items)
        ) / len([items for items in cluster_repairable if any(item["oracle_cost"] is not None for item in items)]), 6) if cluster_repairable else None,
        "no_change_rate": round(sum(
            sum(item["selected"] == "NO_CHANGE" for item in items) / len(items) for items in clusters.values()
        ) / len(clusters), 6) if clusters else None,
        "task_weighted_mean_cost_adjusted_utility": round(sum(item["selected_utility"] for item in per_task) / n, 6) if n else None,
        "task_weighted_mean_repair_lift_over_no_change": round(sum(item["repair_lift"] for item in per_task) / n, 6) if n else None,
    }
    return summary, per_task


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_cluster_bootstrap(
    predictor_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]],
    *, n_resamples: int = 2000, seed: int = 81818,
) -> dict[str, Any]:
    if len(predictor_rows) != len(baseline_rows):
        raise X1EvidenceError("paired bootstrap rows have different lengths")
    if n_resamples < 1:
        raise X1EvidenceError("bootstrap resample count must be positive")
    clusters: dict[str, list[tuple[float, float]]] = defaultdict(list)
    seen_tasks: set[str] = set()
    for predictor, baseline in zip(predictor_rows, baseline_rows):
        if predictor.get("task_id") != baseline.get("task_id"):
            raise X1EvidenceError("paired bootstrap task identities differ")
        if predictor.get("task_id") in seen_tasks:
            raise X1EvidenceError("paired bootstrap contains duplicate task IDs")
        seen_tasks.add(predictor.get("task_id"))
        if predictor.get("cluster_id") != baseline.get("cluster_id"):
            raise X1EvidenceError("paired bootstrap cluster identities differ")
        clusters[predictor["cluster_id"]].append((
            float(predictor["selected_utility"]) - float(baseline["selected_utility"]),
            float(predictor["selected_utility"]),
        ))
    if not clusters:
        return {"status": "INSUFFICIENT_CLUSTERS", "n_clusters": 0, "n_resamples": 0}
    cluster_ids = sorted(clusters)
    cluster_deltas = [sum(item[0] for item in clusters[cluster_id]) / len(clusters[cluster_id]) for cluster_id in cluster_ids]
    point = sum(cluster_deltas) / len(cluster_deltas)
    if len(cluster_ids) < 2:
        return {
            "status": "INSUFFICIENT_CLUSTERS", "n_clusters": len(cluster_ids),
            "n_resamples": 0, "point_estimate": round(point, 6),
            "ci95": None, "seed": seed,
        }
    rng = random.Random(seed)
    draws = []
    for _ in range(n_resamples):
        sampled = [cluster_deltas[rng.randrange(len(cluster_deltas))] for _ in cluster_deltas]
        draws.append(sum(sampled) / len(sampled))
    return {
        "status": "OK",
        "n_clusters": len(cluster_ids),
        "n_resamples": n_resamples,
        "point_estimate": round(point, 6),
        "ci95": [round(_percentile(draws, 0.025), 6), round(_percentile(draws, 0.975), 6)],
        "seed": seed,
        "paired_unit": "independent_latent_world_or_source_cluster",
    }


def _outcome_map(reveal: Mapping[str, Any]) -> dict[tuple[str, str], bool]:
    return {
        (row["task_id"], row["intervention_id"]): bool(row["repaired"])
        for row in reveal.get("rows", [])
    }


def _cluster_map(split_manifest: Mapping[str, Any], task_ids: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cluster in split_manifest.get("clusters", []):
        for task_id in cluster.get("task_ids", []):
            if task_id in task_ids:
                mapping[task_id] = cluster["cluster_id"]
    if set(mapping) != set(task_ids):
        raise X1EvidenceError("cluster manifest does not cover prediction tasks")
    return mapping


def _policy_threshold(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return dict(protocol.get("decision_thresholds", {}).get("predictor", {}))


def _protocol_costs(protocol: Mapping[str, Any]) -> dict[str, int]:
    return {item["id"]: int(item["cost"]) for item in _registry_items(protocol.get("interventions"))}


def _lambda_cost(protocol: Mapping[str, Any]) -> float:
    return float(protocol.get("decision_thresholds", {}).get("predictor", {}).get("cost_lambda", 0.25))


def _replication_valid(
    replication: Mapping[str, Any] | None, *, protocol_sha: str, subject_sha: str,
    primary_analysis_sha: str | None = None,
) -> dict[str, Any]:
    if replication is None:
        return {"valid": False, "errors": ["independent replication receipt is missing"]}
    errors = []
    required = {
        "schema", "phase", "protocol_sha256", "subject_manifest_sha256", "cohort_id",
        "seed", "independent_task_cohort", "cross_checkpoint", "primary_split_manifest_sha256",
        "replication_split_manifest_sha256", "replication_subject_manifest_sha256",
        "replication_prediction_receipt_sha256", "replication_reveal_receipt_sha256",
        "primary_analysis_sha256", "replication_analysis_sha256", "status", "replication_receipt_sha256",
    }
    try:
        _assert_exact_keys(replication, required)
    except X1EvidenceError as exc:
        errors.append(str(exc))
    if replication.get("schema") != REPLICATION_SCHEMA or replication.get("phase") != "REPLICATION_ACCEPTED":
        errors.append("replication schema or phase mismatch")
    if replication.get("protocol_sha256") != protocol_sha:
        errors.append("replication protocol mismatch")
    if replication.get("subject_manifest_sha256") != subject_sha:
        errors.append("replication subject mismatch")
    if replication.get("independent_task_cohort") is not True:
        errors.append("independent task cohort is not confirmed")
    if replication.get("cross_checkpoint") is not True:
        errors.append("cross-checkpoint replication is not confirmed")
    if replication.get("replication_subject_manifest_sha256") == subject_sha:
        errors.append("cross-checkpoint replication must use a distinct subject manifest")
    if replication.get("status") != "QUALIFIED":
        errors.append("replication status is not QUALIFIED")
    for key in (
        "primary_split_manifest_sha256", "replication_split_manifest_sha256",
        "replication_subject_manifest_sha256", "replication_prediction_receipt_sha256",
        "replication_reveal_receipt_sha256", "primary_analysis_sha256", "replication_analysis_sha256",
    ):
        try:
            _hash(replication.get(key), key)
        except X1EvidenceError as exc:
            errors.append(str(exc))
    if primary_analysis_sha is not None and replication.get("primary_analysis_sha256") != primary_analysis_sha:
        errors.append("replication does not point to the supplied primary analysis")
    if replication.get("cohort_id") != "CHECKPOINT_REPLICATION":
        errors.append("replication cohort is not the frozen checkpoint-replication cohort")
    replication_hash = replication.get("replication_receipt_sha256")
    if not isinstance(replication_hash, str) or not HEX64.fullmatch(replication_hash):
        errors.append("replication receipt hash is malformed")
    elif _sha_json(_without(replication, "replication_receipt_sha256")) != replication_hash:
        errors.append("replication receipt hash mismatch")
    return {"valid": not errors, "errors": errors}


def validate_replication_receipt(
    replication: Mapping[str, Any], *, protocol_sha256: str, subject_manifest_sha256: str,
    primary_analysis_sha256: str | None = None,
) -> dict[str, Any]:
    return _replication_valid(
        replication, protocol_sha=protocol_sha256, subject_sha=subject_manifest_sha256,
        primary_analysis_sha=primary_analysis_sha256,
    )


def validate_replication_bundle(
    primary_bundle: Mapping[str, Any], replication_bundle: Mapping[str, Any], replication_receipt: Mapping[str, Any],
    *, protocol: Mapping[str, Any], registry_path: str | Path | None = None,
    checkpoint_root: str | Path | None = None, release_manifest: Mapping[str, Any] | None = None,
    release_root: str | Path | None = None, source_root: str | Path | None = None,
) -> dict[str, Any]:
    required = {
        "subject_manifest", "split_manifest", "prediction_receipt", "reveal_receipt",
        "basis_qualification", "prediction_commit", "reveal_commit", "public_tasks",
        "analysis_receipt",
    }
    errors: list[str] = []
    for name, bundle in (("primary", primary_bundle), ("replication", replication_bundle)):
        if not isinstance(bundle, Mapping) or set(bundle) != required:
            errors.append(f"{name} replication bundle schema is incomplete")
    if not isinstance(replication_receipt, Mapping):
        errors.append("replication receipt is not an object")
    if errors:
        return {"valid": False, "errors": errors}
    try:
        protocol_sha = require_frozen_protocol(protocol, repo_root=source_root, check_source_closure=True)
        replication_verdict = _replication_valid(
            replication_receipt, protocol_sha=protocol_sha,
            subject_sha=_sha_json(primary_bundle["subject_manifest"]),
        )
        errors.extend(replication_verdict["errors"])
        primary_split = primary_bundle["split_manifest"]
        replication_split = replication_bundle["split_manifest"]
        if not isinstance(primary_split, Mapping) or not isinstance(replication_split, Mapping):
            raise X1ContractError("replication split manifests must be objects")
        independence = validate_split_independence(primary_split, replication_split)
        errors.extend(independence["errors"])
        subject = replication_bundle["subject_manifest"]
        subject_verdict = validate_subject_manifest(
            subject, registry_path=registry_path, checkpoint_root=checkpoint_root,
            verify_checkpoint_file=True,
        )
        if not subject_verdict["valid"]:
            errors.extend(subject_verdict["errors"])
        primary_subject = primary_bundle.get("subject_manifest", {})
        primary_subject_verdict = validate_subject_manifest(
            primary_subject, registry_path=registry_path, checkpoint_root=checkpoint_root,
            verify_checkpoint_file=True,
        )
        if not primary_subject_verdict["valid"]:
            errors.extend(primary_subject_verdict["errors"])
        if subject.get("checkpoint_file_sha256") == primary_subject.get("checkpoint_file_sha256"):
            errors.append("replication subject reuses the primary checkpoint file")
        primary_analysis = primary_bundle.get("analysis_receipt", {})
        if not isinstance(primary_analysis, Mapping):
            errors.append("primary analysis receipt is not an object")
            primary_analysis = {}
        if primary_analysis.get("schema") != ANALYSIS_SCHEMA or primary_analysis.get("decision", {}).get("primary_gate_pass") is not True:
            errors.append("primary analysis did not pass its frozen gate")
        primary_analysis_hash = primary_analysis.get("analysis_receipt_sha256")
        if not isinstance(primary_analysis_hash, str) or _sha_json(_without(primary_analysis, "analysis_receipt_sha256")) != primary_analysis_hash:
            errors.append("primary analysis hash mismatch")
        if primary_analysis.get("protocol_sha256") != protocol_sha or primary_analysis.get("subject_manifest_sha256") != _sha_json(primary_subject):
            errors.append("primary analysis is not bound to its subject and protocol")
        if not isinstance(primary_analysis.get("primary_checks"), Mapping) or not all(
            value is True for value in primary_analysis["primary_checks"].values()
        ):
            errors.append("primary analysis checks are not all true")
        primary_prediction = primary_bundle.get("prediction_receipt", {})
        primary_reveal = primary_bundle.get("reveal_receipt", {})
        primary_basis = primary_bundle.get("basis_qualification", {})
        for key, expected in (
            ("prediction_receipt_sha256", primary_prediction.get("prediction_receipt_sha256")),
            ("reveal_receipt_sha256", primary_reveal.get("reveal_receipt_sha256")),
            ("basis_qualification_sha256", primary_basis.get("basis_qualification_sha256")),
            ("release_manifest_sha256", (release_manifest or {}).get("release_manifest_sha256")),
        ):
            if primary_analysis.get(key) != expected:
                errors.append(f"primary analysis binding mismatch: {key}")
        primary_predictor = primary_bundle.get("prediction_receipt", {}).get("predictor", {})
        replication_predictor = replication_bundle.get("prediction_receipt", {}).get("predictor", {})
        for key in ("source_sha256", "training_artifact_sha256", "feature_schema_sha256"):
            if replication_predictor.get(key) != primary_predictor.get(key):
                errors.append(f"replication predictor identity mismatch: {key}")
        split_verdict = validate_split_manifest(
            replication_split, expected_protocol_sha256=protocol_sha,
            expected_cohort_id="CHECKPOINT_REPLICATION", expected_cohort_role="REPLICATION",
            expected_seed=int(protocol.get("cohorts", {}).get("checkpoint_replication", {}).get("seed")),
            expected_cohort_plan_sha256=protocol.get("cohorts", {}).get("checkpoint_replication", {}).get("cohort_plan_sha256"),
        )
        if not split_verdict["valid"]:
            errors.extend(split_verdict["errors"])
        if replication_split.get("cohort_id") != replication_receipt.get("cohort_id"):
            errors.append("replication split and receipt cohort differ")
        if replication_split.get("manifest_sha256") != replication_receipt.get("replication_split_manifest_sha256"):
            errors.append("replication split hash is not bound to replication receipt")
        if replication_split.get("seed") != replication_receipt.get("seed"):
            errors.append("replication split seed is not bound to replication receipt")
        if primary_split.get("manifest_sha256") != replication_receipt.get("primary_split_manifest_sha256"):
            errors.append("primary split hash is not bound to replication receipt")
        if split_verdict["n_clusters"] < int(protocol.get("cohorts", {}).get("checkpoint_replication", {}).get("minimum_independent_clusters", 0)):
            errors.append("replication split is below the frozen independent-cluster minimum")
        basis = replication_bundle["basis_qualification"]
        prediction = replication_bundle["prediction_receipt"]
        reveal = replication_bundle["reveal_receipt"]
        prediction_verdict = validate_prediction_receipt(
            prediction, protocol=protocol, subject_manifest=subject, split_manifest=replication_split,
            public_tasks=replication_bundle["public_tasks"], registry_path=registry_path,
            checkpoint_root=checkpoint_root, release_manifest=release_manifest, release_root=release_root,
            basis_qualification=basis,
            basis_subject_manifest_sha256=subject_verdict.get("subject_manifest_sha256"),
            source_root=source_root,
        )
        if not prediction_verdict["valid"]:
            errors.extend(prediction_verdict["errors"])
        reveal_verdict = validate_reveal_receipt(
            reveal, prediction_receipt=prediction, split_manifest=replication_split,
            prediction_commit=replication_bundle["prediction_commit"],
            reveal_commit=replication_bundle["reveal_commit"], repo_root=release_root,
        )
        if not reveal_verdict["valid"]:
            errors.extend(reveal_verdict["errors"])
        analysis = replication_bundle["analysis_receipt"]
        if not isinstance(analysis, Mapping):
            errors.append("replication analysis receipt is not an object")
            analysis = {}
        if analysis.get("schema") != ANALYSIS_SCHEMA or analysis.get("decision", {}).get("primary_gate_pass") is not True:
            errors.append("replication analysis did not pass its primary gate")
        analysis_hash = analysis.get("analysis_receipt_sha256")
        if not isinstance(analysis_hash, str) or _sha_json(_without(analysis, "analysis_receipt_sha256")) != analysis_hash:
            errors.append("replication analysis hash mismatch")
        if analysis.get("protocol_sha256") != protocol_sha or analysis.get("subject_manifest_sha256") != _sha_json(subject):
            errors.append("replication analysis is not bound to its subject and protocol")
        for key, expected in (
            ("prediction_receipt_sha256", prediction.get("prediction_receipt_sha256")),
            ("reveal_receipt_sha256", reveal.get("reveal_receipt_sha256")),
            ("basis_qualification_sha256", basis.get("basis_qualification_sha256")),
            ("release_manifest_sha256", (release_manifest or {}).get("release_manifest_sha256")),
        ):
            if analysis.get(key) != expected:
                errors.append(f"replication analysis binding mismatch: {key}")
        if not isinstance(analysis.get("primary_checks"), Mapping) or not all(
            value is True for value in analysis["primary_checks"].values()
        ):
            errors.append("replication analysis primary checks are not all true")
        for name, candidate in (("primary", primary_analysis), ("replication", analysis)):
            comparison = candidate.get("comparisons", {}).get("predictor_minus_BEST_FIXED_DEV", {})
            ci = comparison.get("paired_cluster_bootstrap", {}).get("ci95")
            if not isinstance(ci, list) or len(ci) != 2 or float(ci[0]) <= 0.0:
                errors.append(f"{name} replication effect CI does not clear zero")
        if primary_bundle.get("analysis_receipt", {}).get("analysis_receipt_sha256") != replication_receipt.get("primary_analysis_sha256"):
            errors.append("replication receipt does not bind the primary analysis")
        expected_hashes = {
            "replication_subject_manifest_sha256": _sha_json(subject),
            "replication_prediction_receipt_sha256": prediction.get("prediction_receipt_sha256"),
            "replication_reveal_receipt_sha256": reveal.get("reveal_receipt_sha256"),
            "replication_analysis_sha256": analysis.get("analysis_receipt_sha256"),
        }
        for key, expected in expected_hashes.items():
            if replication_receipt.get(key) != expected:
                errors.append(f"replication bundle does not match receipt field {key}")
    except X1ContractError as exc:
        errors.append(str(exc))
    return {"valid": not errors, "errors": errors}


def commit_replication_receipt(
    primary_analysis: Mapping[str, Any], *, protocol_sha256: str, subject_manifest_sha256: str,
    replication_cohort_id: str, replication_seed: int, independent_task_cohort: bool,
    cross_checkpoint: bool, replication_analysis_sha256: str,
    primary_split_manifest: Mapping[str, Any], replication_split_manifest: Mapping[str, Any],
    replication_subject_manifest_sha256: str, replication_prediction_receipt_sha256: str,
    replication_reveal_receipt_sha256: str,
) -> dict[str, Any]:
    if primary_analysis.get("schema") != ANALYSIS_SCHEMA or primary_analysis.get("decision", {}).get("primary_gate_pass") is not True:
        raise X1EvidenceError("primary analysis did not pass its frozen gate")
    primary_analysis_hash = primary_analysis.get("analysis_receipt_sha256")
    if not isinstance(primary_analysis_hash, str) or _sha_json(_without(primary_analysis, "analysis_receipt_sha256")) != primary_analysis_hash:
        raise X1EvidenceError("primary analysis hash is invalid")
    if replication_cohort_id != "CHECKPOINT_REPLICATION":
        raise X1EvidenceError("replication cohort must be the frozen checkpoint-replication cohort")
    if independent_task_cohort is not True or cross_checkpoint is not True:
        raise X1EvidenceError("replication must confirm independent-task and cross-checkpoint status")
    if replication_seed == primary_analysis.get("design", {}).get("seed"):
        raise X1EvidenceError("replication seed must differ from primary seed")
    independence = validate_split_independence(primary_split_manifest, replication_split_manifest)
    if not independence["valid"]:
        raise X1EvidenceError("replication split is not independent: " + "; ".join(independence["errors"]))
    for value, name in (
        (replication_analysis_sha256, "replication_analysis_sha256"),
        (primary_split_manifest.get("manifest_sha256"), "primary_split_manifest_sha256"),
        (replication_split_manifest.get("manifest_sha256"), "replication_split_manifest_sha256"),
        (replication_subject_manifest_sha256, "replication_subject_manifest_sha256"),
        (replication_prediction_receipt_sha256, "replication_prediction_receipt_sha256"),
        (replication_reveal_receipt_sha256, "replication_reveal_receipt_sha256"),
    ):
        _hash(value, name)
    body = {
        "schema": REPLICATION_SCHEMA,
        "phase": "REPLICATION_ACCEPTED",
        "protocol_sha256": protocol_sha256,
        "subject_manifest_sha256": subject_manifest_sha256,
        "cohort_id": replication_cohort_id,
        "seed": replication_seed,
        "independent_task_cohort": bool(independent_task_cohort),
        "cross_checkpoint": bool(cross_checkpoint),
        "primary_split_manifest_sha256": primary_split_manifest.get("manifest_sha256"),
        "replication_split_manifest_sha256": replication_split_manifest.get("manifest_sha256"),
        "replication_subject_manifest_sha256": replication_subject_manifest_sha256,
        "replication_prediction_receipt_sha256": replication_prediction_receipt_sha256,
        "replication_reveal_receipt_sha256": replication_reveal_receipt_sha256,
        "primary_analysis_sha256": primary_analysis.get("analysis_receipt_sha256"),
        "replication_analysis_sha256": replication_analysis_sha256,
        "status": "QUALIFIED" if independent_task_cohort and cross_checkpoint else "NOT_QUALIFIED",
    }
    _hash(body["replication_subject_manifest_sha256"], "replication_subject_manifest_sha256")
    _hash(body["replication_prediction_receipt_sha256"], "replication_prediction_receipt_sha256")
    _hash(body["replication_reveal_receipt_sha256"], "replication_reveal_receipt_sha256")
    _hash(body["primary_analysis_sha256"], "primary_analysis_sha256")
    _hash(body["replication_analysis_sha256"], "replication_analysis_sha256")
    body["replication_receipt_sha256"] = _sha_json(body)
    return body


def analyze_x1_real_1(
    protocol: Mapping[str, Any], subject_manifest: Mapping[str, Any], split_manifest: Mapping[str, Any],
    prediction_receipt: Mapping[str, Any], reveal_receipt: Mapping[str, Any], *,
    basis_qualification: Mapping[str, Any], replication_receipt: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None, primary_analysis_sha256: str | None = None,
    checkpoint_root: str | Path | None = None, release_manifest: Mapping[str, Any] | None = None,
    release_root: str | Path | None = None, public_tasks: Sequence[Mapping[str, Any]] | None = None,
    prediction_commit: Mapping[str, Any] | None = None,
    reveal_commit: Mapping[str, Any] | None = None,
    basis_split_manifest: Mapping[str, Any] | None = None,
    development_split_manifest: Mapping[str, Any] | None = None,
    replication_bundle: Mapping[str, Any] | None = None,
    primary_bundle: Mapping[str, Any] | None = None, source_root: str | Path | None = None,
) -> dict[str, Any]:
    protocol_sha = require_frozen_protocol(protocol, repo_root=source_root, check_source_closure=True)
    if public_tasks is None:
        raise X1EvidenceError("public task bundle is required for analysis")
    if release_manifest is None:
        raise X1EvidenceError("release manifest is required for analysis")
    release_verdict = validate_release_manifest(release_manifest, protocol, repo_root=release_root)
    if not release_verdict["valid"]:
        raise X1EvidenceError("release manifest is invalid: " + "; ".join(release_verdict["errors"]))
    if basis_split_manifest is None or development_split_manifest is None:
        raise X1EvidenceError("basis and development split manifests are required for analysis")
    basis_spec = protocol.get("cohorts", {}).get("basis_qualification", {})
    development_spec = protocol.get("cohorts", {}).get("development", {})
    basis_split_verdict = validate_split_manifest(
        basis_split_manifest, expected_protocol_sha256=protocol_sha,
        expected_cohort_id=basis_spec.get("cohort_id"), expected_cohort_role="QUALIFICATION",
        expected_seed=int(basis_spec.get("seed")),
        expected_cohort_plan_sha256=basis_spec.get("cohort_plan_sha256"),
    )
    development_split_verdict = validate_split_manifest(
        development_split_manifest, expected_protocol_sha256=protocol_sha,
        expected_cohort_id=development_spec.get("cohort_id"), expected_cohort_role="DEVELOPMENT",
        expected_seed=int(development_spec.get("seed")),
        expected_cohort_plan_sha256=development_spec.get("cohort_plan_sha256"),
    )
    if not basis_split_verdict["valid"] or not development_split_verdict["valid"]:
        raise X1EvidenceError("basis or development split is invalid")
    if basis_split_verdict["n_clusters"] < int(basis_spec.get("minimum_independent_clusters", 0)):
        raise X1EvidenceError("basis split is below the frozen independent-cluster minimum")
    if development_split_verdict["n_clusters"] < int(development_spec.get("minimum_independent_clusters", 0)):
        raise X1EvidenceError("development split is below the frozen independent-cluster minimum")
    for name, prior in (("basis", basis_split_manifest), ("development", development_split_manifest)):
        independence = validate_split_independence(prior, split_manifest)
        if not independence["valid"]:
            raise X1EvidenceError(f"{name}/evaluation split overlap: " + "; ".join(independence["errors"]))
    basis_development_independence = validate_split_independence(basis_split_manifest, development_split_manifest)
    if not basis_development_independence["valid"]:
        raise X1EvidenceError("basis/development split overlap: " + "; ".join(basis_development_independence["errors"]))
    subject_verdict = validate_subject_manifest(
        subject_manifest, registry_path=registry_path, checkpoint_root=checkpoint_root,
        verify_checkpoint_file=True,
    )
    if not subject_verdict["valid"]:
        raise X1EvidenceError("subject is not eligible: " + "; ".join(subject_verdict["errors"]))
    subject_sha = subject_verdict["subject_manifest_sha256"]
    basis_verdict = validate_basis_qualification(
        basis_qualification, expected_protocol_sha256=protocol_sha,
        expected_subject_manifest_sha256=subject_sha,
        expected_registry_sha256=protocol.get("intervention_registry_sha256"),
        registry=protocol.get("interventions"),
        basis_split_manifest=basis_split_manifest,
    )
    if basis_qualification.get("thresholds") != protocol.get("decision_thresholds", {}).get("basis"):
        basis_verdict = {"valid": False, "errors": ["basis thresholds do not match protocol"]}
    if prediction_receipt.get("basis_split_manifest_sha256") != basis_split_manifest.get("manifest_sha256"):
        raise X1EvidenceError("prediction basis split binding mismatch")
    if prediction_receipt.get("development_split_manifest_sha256") != development_split_manifest.get("manifest_sha256"):
        raise X1EvidenceError("prediction development split binding mismatch")
    prediction_verdict = validate_prediction_receipt(
        prediction_receipt, protocol=protocol, subject_manifest=subject_manifest, split_manifest=split_manifest,
        registry_path=registry_path, checkpoint_root=checkpoint_root,
        release_manifest=release_manifest, release_root=release_root,
        basis_qualification=basis_qualification, public_tasks=public_tasks,
        basis_subject_manifest_sha256=subject_sha, source_root=source_root,
        development_split_manifest=development_split_manifest,
        basis_split_manifest=basis_split_manifest,
    )
    if not prediction_verdict["valid"]:
        raise X1ChronologyError("prediction receipt failed validation: " + "; ".join(prediction_verdict["errors"]))
    reveal_verdict = validate_reveal_receipt(
        reveal_receipt, prediction_receipt=prediction_receipt, split_manifest=split_manifest,
        prediction_commit=prediction_commit, reveal_commit=reveal_commit, repo_root=release_root,
    )
    if not reveal_verdict["valid"]:
        raise X1ChronologyError("reveal receipt failed validation: " + "; ".join(reveal_verdict["errors"]))
    task_manifest = prediction_receipt["task_manifest"]
    task_ids = [item["task_id"] for item in task_manifest]
    intervention_ids = _protocol_intervention_ids(protocol)
    outcome_map = _outcome_map(reveal_receipt)
    matrix_artifact = [
        {
            "task_id": row["task_id"],
            "cluster_id": row["cluster_id"],
            "source_id": row["source_id"],
            "baseline_failed": row["baseline_failed"],
            "outcomes": {
                intervention_id: int(outcome_map[(row["task_id"], intervention_id)])
                for intervention_id in intervention_ids
            },
        }
        for row in sorted(reveal_receipt["rows"], key=lambda item: (item["task_id"], intervention_ids.index(item["intervention_id"])))
    ]
    unique_matrix = {}
    for item in matrix_artifact:
        unique_matrix[item["task_id"]] = item
    matrix_artifact = list(unique_matrix.values())
    recalculated_basis = qualify_intervention_basis(
        matrix_artifact, protocol.get("interventions"), protocol_sha256=protocol_sha,
        subject_manifest_sha256=subject_sha,
        thresholds=protocol.get("decision_thresholds", {}).get("basis"),
        binding=basis_qualification.get("basis_binding"),
    )
    if not basis_verdict["valid"]:
        blocked = {
            "schema": ANALYSIS_SCHEMA,
            "phase": "ANALYSIS_BLOCKED",
            "protocol_sha256": protocol_sha,
            "subject_manifest_sha256": subject_sha,
            "prediction_receipt_sha256": prediction_receipt.get("prediction_receipt_sha256"),
            "reveal_receipt_sha256": reveal_receipt.get("reveal_receipt_sha256"),
            "basis_status": recalculated_basis.get("status"),
            "basis_errors": [key for key, value in recalculated_basis.get("checks", {}).items() if value is not True],
            "decision": {"status": "BLOCKED_BASIS_NOT_QUALIFIED", "primary_gate_pass": False, "supported": False},
            "claim_ceiling": protocol.get("claim_ceiling"),
        }
        blocked["analysis_receipt_sha256"] = _sha_json(blocked)
        return blocked
    task_clusters = _cluster_map(split_manifest, task_ids)
    predictor_map = _normalize_prediction_map(
        prediction_receipt["predictions"], task_ids, intervention_ids, name="predictor"
    )
    baseline_maps = {
        name: _normalize_prediction_map(rows, task_ids, intervention_ids, name=f"baseline.{name}")
        for name, rows in prediction_receipt["baselines"].items()
    }
    costs = _protocol_costs(protocol)
    registry_items = _registry_items(protocol.get("interventions"))
    repair_intervention_ids = [
        item["id"] for item in registry_items if item["role"] in {"DIAGNOSTIC", "REPAIR", "ASSISTANCE"}
    ]
    lambda_cost = _lambda_cost(protocol)
    policy_reports: dict[str, Any] = {}
    per_task_by_policy: dict[str, list[dict[str, Any]]] = {}
    reference_prevalence = prediction_receipt.get("baseline_commitment", {}).get("development_prevalence")
    reference_prevalence_by_intervention = prediction_receipt.get("baseline_commitment", {}).get("development_prevalence_by_intervention")
    for name, prediction_map in {"PREDICTOR": predictor_map, **baseline_maps}.items():
        cell_report = _cell_metrics(
            prediction_map, outcome_map, intervention_ids, reference_prevalence=reference_prevalence,
            reference_prevalence_by_intervention=reference_prevalence_by_intervention,
            task_clusters=task_clusters,
        )
        task_report, per_task = _policy_metrics(
            prediction_map, outcome_map, task_clusters, intervention_ids, repair_intervention_ids,
            costs, lambda_cost
        )
        policy_reports[name] = {"cell": cell_report, "task": task_report}
        per_task_by_policy[name] = per_task
    oracle_rows = []
    for task_id in task_ids:
        successful = [item for item in repair_intervention_ids if outcome_map[(task_id, item)]]
        selected = min(successful, key=lambda item: (costs[item], intervention_ids.index(item))) if successful else "NO_CHANGE"
        cheapest_success = min((costs[item] for item in successful), default=None)
        oracle_utility = 1.0 - lambda_cost * cheapest_success if cheapest_success is not None else 0.0
        selected_repaired = bool(selected in repair_intervention_ids and outcome_map[(task_id, selected)])
        oracle_rows.append({
            "task_id": task_id, "cluster_id": task_clusters[task_id], "selected": selected,
            "selected_repaired": selected_repaired,
            "selected_cost": costs[selected], "oracle_cost": cheapest_success,
            "selected_utility": (1.0 if selected_repaired else 0.0) - lambda_cost * costs[selected],
            "oracle_utility": oracle_utility,
            "repairable": bool(successful), "no_change_repaired": outcome_map[(task_id, "NO_CHANGE")],
            "capture": int(bool(successful) and selected_repaired),
            "false_intervention": int(not successful and selected != "NO_CHANGE"),
            "repair_lift": int(selected_repaired) - int(outcome_map[(task_id, "NO_CHANGE")]),
        })
    oracle_report = {
        "mean_cost_adjusted_utility": round(sum(row["selected_utility"] for row in oracle_rows) / len(oracle_rows), 6),
        "mean_repair_lift_over_no_change": round(sum(row["repair_lift"] for row in oracle_rows) / len(oracle_rows), 6),
        "evaluator_only": True,
    }
    thresholds = _policy_threshold(protocol)
    comparisons = {}
    for baseline in PREDICTION_BASELINES:
        comparisons[f"predictor_minus_{baseline}"] = {
            "cost_adjusted_utility": round(
                policy_reports["PREDICTOR"]["task"]["mean_cost_adjusted_utility"] -
                policy_reports[baseline]["task"]["mean_cost_adjusted_utility"], 6
            ),
            "paired_cluster_bootstrap": paired_cluster_bootstrap(
                per_task_by_policy["PREDICTOR"], per_task_by_policy[baseline],
                n_resamples=int(thresholds.get("bootstrap_resamples", 2000)),
                seed=int(thresholds.get("bootstrap_seed", 81818)),
            ),
        }
    predictor = policy_reports["PREDICTOR"]
    best_fixed = max(
        policy_reports[name]["task"]["mean_cost_adjusted_utility"]
        for name in ("BEST_FIXED_DEV", "COST_AWARE_FIXED", "PREVALENCE_ONLY", "ALWAYS_NEGATIVE")
    )
    best_ci = comparisons["predictor_minus_BEST_FIXED_DEV"]["paired_cluster_bootstrap"].get("ci95")
    cost_ci = comparisons["predictor_minus_COST_AWARE_FIXED"]["paired_cluster_bootstrap"].get("ci95")
    surface_ci = comparisons["predictor_minus_SURFACE_SHORTCUT"]["paired_cluster_bootstrap"].get("ci95")
    primary_checks = {
        "subject_eligible": subject_verdict["valid"],
        "basis_qualified": recalculated_basis.get("status") == "QUALIFIED",
        "prediction_reveal_order_valid": reveal_verdict["valid"],
        "non_degenerate_predictor": predictor["cell"]["score_variance"] > float(thresholds.get("min_score_variance", 1e-6)),
        "auprc_lift": predictor["cell"]["auprc_lift_over_prevalence"] >= float(thresholds.get("min_auprc_lift", 0.10)),
        "brier_skill": predictor["cell"]["brier_skill_vs_prevalence"] >= float(thresholds.get("min_brier_skill", 0.05)),
        "calibration": predictor["cell"]["calibration_error"] <= float(thresholds.get("max_calibration_error", 0.10)),
        "mcc": predictor["cell"]["mcc"] >= float(thresholds.get("min_mcc", 0.05)),
        "utility_over_best_fixed": predictor["task"]["mean_cost_adjusted_utility"] > best_fixed + float(thresholds.get("min_utility_margin", 0.0)),
        "utility_over_surface_shortcut": predictor["task"]["mean_cost_adjusted_utility"] > policy_reports["SURFACE_SHORTCUT"]["task"]["mean_cost_adjusted_utility"] + float(thresholds.get("min_utility_margin", 0.0)),
        "utility_ci_over_best_fixed": (not thresholds.get("require_ci_above_zero", True)) or (best_ci is not None and best_ci[0] > 0.0),
        "utility_ci_over_cost_aware": (not thresholds.get("require_ci_above_zero", True)) or (cost_ci is not None and cost_ci[0] > 0.0),
        "utility_ci_over_surface_shortcut": (not thresholds.get("require_ci_above_zero", True)) or (surface_ci is not None and surface_ci[0] > 0.0),
        "repair_lift": predictor["task"]["mean_repair_lift_over_no_change"] >= float(thresholds.get("min_repair_lift", 0.05)),
        "independent_clusters": len(set(task_clusters.values())) >= int(thresholds.get("min_independent_clusters", 20)),
    }
    primary_pass = all(primary_checks.values())
    if replication_receipt is not None and primary_analysis_sha256 is None:
        replication_verdict = {"valid": False, "errors": ["primary analysis hash is required for replication"]}
    else:
        replication_verdict = _replication_valid(
            replication_receipt, protocol_sha=protocol_sha, subject_sha=subject_sha,
            primary_analysis_sha=primary_analysis_sha256,
        )
    if replication_receipt is not None and replication_bundle is not None and primary_bundle is not None:
        bundle_verdict = validate_replication_bundle(
            primary_bundle, replication_bundle, replication_receipt, protocol=protocol,
            registry_path=registry_path, checkpoint_root=checkpoint_root,
            release_manifest=release_manifest, release_root=release_root,
            source_root=source_root,
        )
        if not bundle_verdict["valid"]:
            replication_verdict = {"valid": False, "errors": bundle_verdict["errors"]}
    elif replication_receipt is not None:
        replication_verdict = {"valid": False, "errors": ["replication bundle is required"]}
    if primary_pass and replication_verdict["valid"]:
        decision_status = "SUPPORTED_SCOPED"
        supported = True
    elif primary_pass:
        decision_status = "PRIMARY_GATE_PASS_REPLICATION_PENDING"
        supported = False
    elif any(value is False for key, value in primary_checks.items() if key not in {"independent_clusters"}):
        decision_status = "NOT_SUPPORTED"
        supported = False
    else:
        decision_status = "INCONCLUSIVE"
        supported = False
    design = protocol.get("cohorts", {}).get("primary", {})
    analysis_body = {
        "schema": ANALYSIS_SCHEMA,
        "phase": "ANALYSIS_ACCEPTED" if primary_pass else "ANALYSIS_COMPLETE",
        "protocol_sha256": protocol_sha,
        "subject_manifest_sha256": subject_sha,
        "release_manifest_sha256": release_manifest.get("release_manifest_sha256"),
        "basis_qualification_sha256": basis_qualification.get("basis_qualification_sha256"),
        "basis_recomputed_on_evaluation_status": recalculated_basis.get("status"),
        "basis_split_manifest_sha256": basis_split_manifest.get("manifest_sha256"),
        "development_split_manifest_sha256": development_split_manifest.get("manifest_sha256"),
        "prediction_receipt_sha256": prediction_receipt.get("prediction_receipt_sha256"),
        "prediction_phase_commit_sha256": prediction_commit.get("phase_commit_sha256") if prediction_commit else None,
        "reveal_receipt_sha256": reveal_receipt.get("reveal_receipt_sha256"),
        "reveal_phase_commit_sha256": reveal_commit.get("phase_commit_sha256") if reveal_commit else None,
        "cohort_id": prediction_receipt.get("cohort_id"),
        "n_independent_clusters": len(set(task_clusters.values())),
        "policy_reports": policy_reports,
        "oracle_ceiling": oracle_report,
        "comparisons": comparisons,
        "decision_thresholds": thresholds,
        "primary_checks": primary_checks,
        "replication": replication_verdict,
        "design": {"cohort_id": prediction_receipt.get("cohort_id"), "seed": design.get("seed")},
        "decision": {
            "status": decision_status,
            "primary_gate_pass": primary_pass,
            "supported": supported,
            "claim_ceiling": protocol.get("claim_ceiling"),
        },
    }
    analysis_body["analysis_receipt_sha256"] = _sha_json(analysis_body)
    return analysis_body


def _preflight_record(
    checks: list[dict[str, Any]], check_id: str, callback: Any, remedy: str,
) -> Any:
    try:
        value = callback()
        errors: list[str] = []
        if isinstance(value, Mapping):
            if value.get("valid") is False:
                errors = [str(item) for item in value.get("errors", [])]
            elif str(value.get("status", "")).startswith(("NOT_", "BLOCKED", "NO_ELIGIBLE", "RELEASE_MANIFEST")):
                errors = [str(value.get("status"))]
        checks.append({
            "id": check_id,
            "status": "FAIL" if errors else "PASS",
            "errors": errors,
            "remedy": remedy,
        })
        return value
    except Exception as exc:
        checks.append({
            "id": check_id,
            "status": "FAIL",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "remedy": remedy,
        })
        return None


def _preflight_missing(
    checks: list[dict[str, Any]], check_id: str, artifact: str, remedy: str,
) -> None:
    checks.append({
        "id": check_id,
        "status": "BLOCKED",
        "errors": [f"required artifact is missing: {artifact}"],
        "remedy": remedy,
    })


def _preflight_warning(
    checks: list[dict[str, Any]], check_id: str, message: str, remedy: str,
) -> None:
    checks.append({
        "id": check_id,
        "status": "WARNING",
        "errors": [message],
        "remedy": remedy,
    })


def intake_checkpoint_candidate(
    checkpoint_path: str | Path, *, config_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None, runtime_source_revision: str | None = None,
    source_commit: str | None = None, global_step: int | None = None,
    stage: str | None = None, parameter_sha256: str | None = None,
    tokenizer_identity_sha256: str | None = None, readiness_receipt_path: str | Path | None = None,
    readiness_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    config = Path(config_path) if config_path else None
    tokenizer = Path(tokenizer_path) if tokenizer_path else None
    files = {
        "checkpoint": checkpoint,
        "model_config": config,
        "tokenizer_artifact": tokenizer,
        "readiness_receipt": Path(readiness_receipt_path) if readiness_receipt_path else None,
    }
    file_records: dict[str, Any] = {}
    missing: list[str] = []
    for name, candidate in files.items():
        if candidate is None:
            file_records[name] = {"path": None, "exists": False, "bytes": None, "sha256": None}
            missing.append(name)
            continue
        exists = candidate.is_file()
        file_records[name] = {
            "path": str(candidate),
            "exists": exists,
            "bytes": candidate.stat().st_size if exists else None,
            "sha256": _sha_file(candidate) if exists else None,
        }
        if not exists:
            missing.append(name)
    metadata_missing = [
        key for key, value in {
            "runtime_source_revision": runtime_source_revision,
            "source_commit": source_commit,
            "global_step": global_step,
            "stage": stage,
            "tokenizer_identity_sha256": tokenizer_identity_sha256,
            "parameter_sha256": parameter_sha256,
            "readiness_receipt_sha256": readiness_receipt_sha256,
        }.items() if value is None
    ]
    missing.extend(metadata_missing)
    checkpoint_bytes = file_records["checkpoint"].get("bytes") or 0
    body = {
        "schema": INTAKE_SCHEMA,
        "status": "INVENTORIED" if not missing else "UNQUALIFIED_NEW",
        "research_subject": False,
        "eligible_for_x1": False,
        "promotion_status": "BLOCKED",
        "model_execution": False,
        "files": file_records,
        "checkpoint_file_sha256": file_records["checkpoint"].get("sha256"),
        "parameter_sha256": parameter_sha256,
        "model_config_sha256": file_records["model_config"].get("sha256"),
        "tokenizer_artifact_sha256": file_records["tokenizer_artifact"].get("sha256"),
        "tokenizer_identity_sha256": tokenizer_identity_sha256,
        "readiness_receipt_path": str(readiness_receipt_path) if readiness_receipt_path else None,
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "runtime_source_revision": runtime_source_revision,
        "source_commit": source_commit,
        "global_step": global_step,
        "stage": stage,
        "missing_fields": sorted(set(missing)),
        "next_actions": [
            "do not promote this candidate from file presence alone",
            "obtain parameter, tokenizer identity, runtime, source, and readiness attestations",
            "create a subject-bound identity attestation and eligibility artifact set",
            "run preflight before uploading or executing anything",
        ],
        "platform_guidance": (
            "persistent GPU runner preferred" if checkpoint_bytes >= 1_000_000_000
            else "Colab/Kaggle can host only after storage and session checks"
        ),
        "storage_budget_check": checkpoint_bytes <= 2_000_000_000,
        "local_execution_authorized": False,
    }
    body["intake_sha256"] = _sha_json(body)
    return body


def _intervention_transformation_smoke(protocol: Mapping[str, Any]) -> dict[str, Any]:
    from x_factor import ibq_v2
    task = {
        "block": "A: alpha value\nB: beta value",
        "query": "Return the reference.",
        "answer_marker": "Answer:",
    }
    outputs: dict[str, str] = {}
    for item in protocol.get("interventions", []):
        transformation = item.get("transformation")
        if transformation == "identity":
            rendered = f"{task['block']}\n{task['query']}\n{task['answer_marker']}"
        elif isinstance(transformation, str) and transformation.startswith("ibq_v2."):
            function_name = transformation.split(".", 1)[1]
            function = getattr(ibq_v2, function_name, None)
            if not callable(function):
                raise X1EvidenceError(f"transformation function is missing: {transformation}")
            rendered = function(task)
        else:
            raise X1EvidenceError(f"unsupported transformation binding: {transformation}")
        probe = ibq_v2.V2_BASIS.get(str(item.get("id")))
        if probe is not None or item.get("id") == "NO_CHANGE":
            dispatched = ibq_v2.apply_probe(str(item.get("id")), task)
            if not isinstance(dispatched, str) or not dispatched.strip():
                raise X1EvidenceError(f"transformation dispatcher returned no text: {item.get('id')}")
        if not isinstance(rendered, str) or not rendered.strip() or task["query"] not in rendered:
            raise X1EvidenceError(f"transformation did not preserve the visible query: {item.get('id')}")
        outputs[str(item.get("id"))] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return {"valid": True, "n_transformations": len(outputs), "output_hashes": outputs}


def _preflight_input_hashes(values: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        name: _sha_json(value) if value is not None else None
        for name, value in values.items()
    }


def preflight_x1_real_1(
    protocol: Mapping[str, Any], *, subject_manifest: Mapping[str, Any] | None = None,
    basis_qualification: Mapping[str, Any] | None = None,
    release_manifest: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    release_root: str | Path | None = None,
    source_root: str | Path | None = None,
    primary_split_manifest: Mapping[str, Any] | None = None,
    basis_split_manifest: Mapping[str, Any] | None = None,
    development_split_manifest: Mapping[str, Any] | None = None,
    public_tasks: Sequence[Mapping[str, Any]] | None = None,
    predictor_identity: Mapping[str, Any] | None = None,
    baseline_commitment: Mapping[str, Any] | None = None,
    predictor_predictions: Any | None = None,
    cohort_id: str = "PRIMARY_EVAL",
    replication_task_count: int | None = None,
    prediction_receipt: Mapping[str, Any] | None = None,
    prediction_commit: Mapping[str, Any] | None = None,
    reveal_receipt: Mapping[str, Any] | None = None,
    reveal_commit: Mapping[str, Any] | None = None,
    outcomes: Any | None = None,
    evaluator_identity: Mapping[str, Any] | None = None,
    replication_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    protocol_sha = None
    protocol_result = _preflight_record(
        checks, "protocol",
        lambda: validate_frozen_protocol(protocol, repo_root=source_root, check_source_closure=True),
        "freeze the protocol and make its source closure match the release source",
    )
    if isinstance(protocol_result, Mapping) and protocol_result.get("valid"):
        protocol_sha = protocol_result.get("protocol_sha256")
    _preflight_record(
        checks, "intervention_transformations",
        lambda: _intervention_transformation_smoke(protocol),
        "repair the frozen intervention transformation binding before external execution",
    )
    subject_result = None
    if subject_manifest is None:
        _preflight_missing(checks, "subject", "subject manifest", "supply a complete file-verifiable research_subject manifest")
    else:
        subject_result = _preflight_record(
            checks, "subject",
            lambda: validate_subject_manifest(
                subject_manifest, registry_path=registry_path, checkpoint_root=checkpoint_root,
                verify_checkpoint_file=True,
            ),
            "make the checkpoint, registry entry, attestation, and eligibility artifacts agree",
        )
    if release_manifest is None:
        _preflight_missing(checks, "release", "release manifest", "publish a clean release manifest with protocol and attestation artifacts")
    else:
        _preflight_record(
            checks, "release",
            lambda: validate_release_manifest(release_manifest, protocol, repo_root=release_root),
            "publish a clean release whose revision, source closure, protocol artifact, and attestation agree",
        )
    subject_sha = subject_result.get("subject_manifest_sha256") if isinstance(subject_result, Mapping) else None
    basis_result = None
    if basis_qualification is None:
        _preflight_missing(checks, "basis", "basis qualification", "run the independent pre-training basis qualification")
    else:
        basis_result = _preflight_record(
            checks, "basis",
            lambda: validate_basis_qualification(
                basis_qualification,
                expected_protocol_sha256=protocol_sha,
                expected_subject_manifest_sha256=subject_sha,
                expected_registry_sha256=protocol.get("intervention_registry_sha256"),
                registry=protocol.get("interventions"),
                basis_split_manifest=basis_split_manifest,
            ),
            "qualify the basis on an independent split before fitting any predictor",
        )
    evaluation_spec = next(
        (value for value in protocol.get("cohorts", {}).values() if value.get("cohort_id") == cohort_id),
        protocol.get("cohorts", {}).get("primary", {}),
    )
    split_specs = {
        "evaluation": (
            primary_split_manifest,
            evaluation_spec.get("cohort_id", "PRIMARY_EVAL"),
            evaluation_spec.get("role", "EVALUATION"),
            evaluation_spec.get("seed"),
            evaluation_spec,
        ),
        "basis": (
            basis_split_manifest,
            protocol.get("cohorts", {}).get("basis_qualification", {}).get("cohort_id", "BASIS_QUALIFICATION"),
            protocol.get("cohorts", {}).get("basis_qualification", {}).get("role", "QUALIFICATION"),
            protocol.get("cohorts", {}).get("basis_qualification", {}).get("seed"),
            protocol.get("cohorts", {}).get("basis_qualification", {}),
        ),
        "development": (
            development_split_manifest,
            protocol.get("cohorts", {}).get("development", {}).get("cohort_id", "DEV_COHORT_V1"),
            protocol.get("cohorts", {}).get("development", {}).get("role", "DEVELOPMENT"),
            protocol.get("cohorts", {}).get("development", {}).get("seed"),
            protocol.get("cohorts", {}).get("development", {}),
        ),
    }
    split_results: dict[str, Any] = {}
    for name, (manifest, expected_id, expected_role, expected_seed, cohort_spec) in split_specs.items():
        if manifest is None:
            _preflight_missing(checks, f"split_{name}", f"{name} split manifest", "create the frozen cohort split manifest with content hashes")
            continue
        split_results[name] = _preflight_record(
            checks, f"split_{name}",
            lambda manifest=manifest, expected_id=expected_id, expected_role=expected_role, expected_seed=expected_seed, cohort_spec=cohort_spec: validate_split_manifest(
                manifest, expected_protocol_sha256=protocol_sha,
                expected_cohort_id=expected_id, expected_cohort_role=expected_role,
                expected_seed=expected_seed,
                expected_cohort_plan_sha256=cohort_spec.get("cohort_plan_sha256"),
            ),
            "correct the split identity, seed, plan hash, task coverage, and content hashes",
        )
    if primary_split_manifest is not None and development_split_manifest is not None:
        _preflight_record(
            checks, "split_independence_development",
            lambda: validate_split_independence(development_split_manifest, primary_split_manifest),
            "use disjoint task content, source, and cluster identities for development and evaluation",
        )
    if primary_split_manifest is not None and basis_split_manifest is not None:
        _preflight_record(
            checks, "split_independence_basis",
            lambda: validate_split_independence(basis_split_manifest, primary_split_manifest),
            "use disjoint task content, source, and cluster identities for basis and evaluation",
        )
    if basis_split_manifest is not None and development_split_manifest is not None:
        _preflight_record(
            checks, "split_independence_basis_development",
            lambda: validate_split_independence(basis_split_manifest, development_split_manifest),
            "use disjoint task content, source, and cluster identities for basis and development",
        )
    power_advisories = []
    for name, result in split_results.items():
        cohort_name = "basis_qualification" if name == "basis" else "development" if name == "development" else next((
            key for key, value in protocol.get("cohorts", {}).items()
            if value.get("cohort_id") == cohort_id
        ), "primary")
        target = protocol.get("cohorts", {}).get(cohort_name, {}).get("target_independent_clusters")
        if isinstance(result, Mapping) and result.get("valid") and isinstance(target, int) and result.get("n_clusters", 0) < target:
            message = f"{name} split has {result.get('n_clusters')} independent clusters; frozen target is {target}"
            remedy = "reach the frozen target or obtain an explicit protocol deviation before spending the full compute budget"
            _preflight_warning(checks, f"power_{name}", message, remedy)
            power_advisories.append({"split": name, "actual_clusters": result.get("n_clusters"), "target_clusters": target})
        minimum = protocol.get("cohorts", {}).get(cohort_name, {}).get("minimum_independent_clusters")
        if isinstance(result, Mapping) and result.get("valid") and isinstance(minimum, int) and result.get("n_clusters", 0) < minimum:
            checks.append({
                "id": f"minimum_clusters_{name}",
                "status": "FAIL",
                "errors": [f"{name} split has {result.get('n_clusters')} clusters; hard minimum is {minimum}"],
                "remedy": "add independent clusters before external execution; do not spend compute on an underpowered split",
            })
    task_result = None
    if public_tasks is None:
        _preflight_missing(checks, "tasks", "public task bundle", "export the closed public task bundle for the evaluation split")
    else:
        task_result = _preflight_record(
            checks, "tasks",
            lambda: validate_public_task_bundle(public_tasks, split_manifest=primary_split_manifest),
            "remove forbidden truth fields and make public task content match the split",
        )
    task_ids = list(task_result.get("task_ids", [])) if isinstance(task_result, Mapping) else []
    try:
        intervention_ids = _protocol_intervention_ids(protocol)
    except Exception as exc:
        intervention_ids = []
        checks.append({
            "id": "intervention_registry",
            "status": "FAIL",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "remedy": "repair the frozen intervention registry before validating predictions",
        })
    if predictor_identity is None:
        _preflight_missing(checks, "predictor_identity", "predictor identity", "provide a file-bound predictor training manifest produced from development data only")
    else:
        _preflight_record(
            checks, "predictor_identity",
            lambda: _validate_predictor_identity(
                predictor_identity, artifact_root=release_root,
                expected_feature_schema_sha256=_sha_json(protocol.get("observations", {}).get("feature_fields", [])),
                expected_training_split_manifest_sha256=(development_split_manifest or {}).get("manifest_sha256"),
                expected_development_task_ids=[
                    task_id for cluster in (development_split_manifest or {}).get("clusters", [])
                    for task_id in cluster.get("task_ids", [])
                ] or None,
            ),
            "bind the predictor artifact and training manifest to the frozen development split",
        )
    if baseline_commitment is None:
        _preflight_missing(checks, "baselines", "baseline commitment", "fit all frozen baselines on development rows only")
    elif task_ids:
        _preflight_record(
            checks, "baselines",
            lambda: _validate_baseline_commitment(
                baseline_commitment, task_ids, intervention_ids,
                expected_development_cohort_id=protocol.get("cohorts", {}).get("development", {}).get("cohort_id"),
                expected_selection_lambda=float(protocol.get("cost_model", {}).get("utility_lambda", 0.25)),
                artifact_root=release_root,
                development_split_manifest=development_split_manifest,
            ),
            "rebuild baselines from a file-bound development artifact and split",
        )
    else:
        _preflight_missing(checks, "baselines", "baseline commitment and public tasks", "provide tasks before validating baseline coverage")
    if predictor_predictions is None:
        _preflight_missing(checks, "predictions", "predictor predictions", "freeze predictor predictions before committing the phase receipt")
    elif task_ids:
        _preflight_record(
            checks, "predictions",
            lambda: _normalize_prediction_map(predictor_predictions, task_ids, intervention_ids, name="preflight.predictor"),
            "provide one calibrated probability vector for every task and intervention",
        )
    else:
        _preflight_missing(checks, "predictions", "predictor predictions and public tasks", "provide tasks before validating prediction coverage")
    dry_run_prediction = None
    prediction_inputs_ready = all(value is not None for value in (
        subject_manifest, basis_qualification, release_manifest, primary_split_manifest,
        development_split_manifest, basis_split_manifest, public_tasks, predictor_identity,
        baseline_commitment, predictor_predictions,
    ))
    if prediction_inputs_ready:
        dry_run_prediction = _preflight_record(
            checks, "prediction_dry_run",
            lambda: commit_prediction_receipt(
                protocol, subject_manifest, primary_split_manifest, public_tasks,
                predictor_predictions, baseline_commitment, predictor_identity,
                basis_qualification, cohort_id=cohort_id,
                registry_path=registry_path, checkpoint_root=checkpoint_root,
                release_manifest=release_manifest, release_root=release_root,
                source_root=source_root, artifact_root=release_root,
                development_split_manifest=development_split_manifest,
                basis_split_manifest=basis_split_manifest,
            ),
            "fix the first failing artifact identity or shape before external compute",
        )
    else:
        _preflight_missing(checks, "prediction_dry_run", "complete prediction input bundle", "resolve all prediction-stage blockers before external execution")
    if prediction_receipt is not None:
        _preflight_record(
            checks, "prediction_receipt",
            lambda: validate_prediction_receipt(
                prediction_receipt, protocol=protocol, subject_manifest=subject_manifest,
                public_tasks=public_tasks, split_manifest=primary_split_manifest,
                registry_path=registry_path, checkpoint_root=checkpoint_root,
                release_manifest=release_manifest, release_root=release_root,
                basis_qualification=basis_qualification,
                basis_subject_manifest_sha256=subject_sha, source_root=source_root,
                development_split_manifest=development_split_manifest,
                basis_split_manifest=basis_split_manifest,
            ),
            "repair the prediction receipt before using it as the reveal parent",
        )
    elif prediction_commit is not None or reveal_receipt is not None or reveal_commit is not None:
        _preflight_missing(checks, "prediction_receipt", "committed prediction receipt", "commit prediction only after the prediction-stage dry run passes")
    if prediction_receipt is not None and prediction_commit is None:
        _preflight_missing(checks, "prediction_commit", "prediction phase commit", "commit the prediction phase before accepting any downstream artifact")
    if prediction_commit is not None and prediction_receipt is not None:
        _preflight_record(
            checks, "prediction_commit",
            lambda: validate_phase_commit(
                prediction_commit, expected_phase="PREDICT_COMMITTED",
                expected_receipt_sha256=prediction_receipt.get("prediction_receipt_sha256"),
                expected_previous_commit_sha256=prediction_receipt.get("basis_phase_commit_sha256"),
                repo_root=release_root,
            ),
            "make the prediction phase commit point to the basis-qualified phase",
        )
    if reveal_receipt is not None and prediction_receipt is not None:
        _preflight_record(
            checks, "reveal_receipt",
            lambda: validate_reveal_receipt(
                reveal_receipt, prediction_receipt=prediction_receipt,
                split_manifest=primary_split_manifest, prediction_commit=prediction_commit,
                reveal_commit=reveal_commit, repo_root=release_root,
            ),
            "bind reveal rows to structured verifier artifacts and the committed prediction parent",
        )
    elif outcomes is not None:
        _preflight_missing(checks, "reveal_receipt", "committed reveal receipt", "commit reveal only after all verifier artifacts are file-verifiable")
    if reveal_receipt is not None and reveal_commit is None:
        _preflight_missing(checks, "reveal_commit", "reveal phase commit", "commit the reveal phase before accepting downstream analysis")
    if outcomes is not None and evaluator_identity is not None and prediction_receipt is not None and prediction_commit is not None:
        _preflight_record(
            checks, "reveal_dry_run",
            lambda: commit_reveal_receipt(
                prediction_receipt, outcomes, evaluator_identity=evaluator_identity,
                split_manifest=primary_split_manifest, prediction_commit=prediction_commit,
                protocol=protocol, subject_manifest=subject_manifest,
                registry_path=registry_path, checkpoint_root=checkpoint_root,
                release_manifest=release_manifest, release_root=release_root,
                public_tasks=public_tasks, basis_qualification=basis_qualification,
                source_root=source_root,
            ),
            "repair the first reveal/verifier artifact mismatch before committing reveal",
        )
    if all(value is not None for value in (
        reveal_receipt, reveal_commit, prediction_receipt, prediction_commit,
        public_tasks, basis_split_manifest, development_split_manifest, release_manifest,
    )):
        _preflight_record(
            checks, "analysis_dry_run",
            lambda: analyze_x1_real_1(
                protocol, subject_manifest, primary_split_manifest, prediction_receipt, reveal_receipt,
                basis_qualification=basis_qualification,
                replication_receipt=replication_receipt,
                registry_path=registry_path, checkpoint_root=checkpoint_root,
                release_manifest=release_manifest, release_root=release_root,
                source_root=source_root, public_tasks=public_tasks,
                prediction_commit=prediction_commit, reveal_commit=reveal_commit,
                basis_split_manifest=basis_split_manifest,
                development_split_manifest=development_split_manifest,
            ),
            "repair the first analysis or chronology mismatch before accepting the result",
        )
    input_values = {
        "protocol": protocol, "subject_manifest": subject_manifest, "basis_qualification": basis_qualification,
        "release_manifest": release_manifest, "primary_split_manifest": primary_split_manifest,
        "basis_split_manifest": basis_split_manifest, "development_split_manifest": development_split_manifest,
        "public_tasks": public_tasks, "predictor_identity": predictor_identity,
        "baseline_commitment": baseline_commitment, "predictor_predictions": predictor_predictions,
        "prediction_receipt": prediction_receipt, "prediction_commit": prediction_commit,
        "reveal_receipt": reveal_receipt, "reveal_commit": reveal_commit,
    }
    task_count = len(task_ids)
    preflight_primary_cells = task_count * len(intervention_ids)
    evaluation_budget_key = "replication_cells" if cohort_id == "CHECKPOINT_REPLICATION" else "primary_cells"
    declared_evaluation_budget = protocol.get("execution_budget", {}).get(evaluation_budget_key)
    if isinstance(declared_evaluation_budget, int) and preflight_primary_cells > declared_evaluation_budget:
        checks.append({
            "id": f"{evaluation_budget_key}_cell_budget",
            "status": "FAIL",
            "errors": [f"preflight requires {preflight_primary_cells} evaluation cells; frozen budget is {declared_evaluation_budget}"],
            "remedy": "reduce the task set to the frozen cell budget or amend the protocol before external execution",
        })
    declared_replication_budget = protocol.get("execution_budget", {}).get("replication_cells")
    if replication_task_count is not None and (not isinstance(replication_task_count, int) or isinstance(replication_task_count, bool) or replication_task_count < 0):
        checks.append({
            "id": "replication_task_count",
            "status": "FAIL",
            "errors": ["replication task count must be a non-negative integer"],
            "remedy": "provide a valid replication task count before external execution",
        })
        preflight_replication_cells = None
    else:
        preflight_replication_cells = (
            replication_task_count * len(intervention_ids)
            if replication_task_count is not None else declared_replication_budget
        )
    maximum_external_cells = protocol.get("execution_budget", {}).get("maximum_external_cells")
    if (
        isinstance(declared_replication_budget, int) and isinstance(preflight_replication_cells, int)
        and preflight_replication_cells > declared_replication_budget
    ):
        checks.append({
            "id": "replication_cell_budget",
            "status": "FAIL",
            "errors": [f"preflight requires {preflight_replication_cells} replication cells; frozen budget is {declared_replication_budget}"],
            "remedy": "reduce the replication task set or amend the protocol before external execution",
        })
    if (
        isinstance(preflight_primary_cells, int) and isinstance(preflight_replication_cells, int)
        and isinstance(maximum_external_cells, int)
        and preflight_primary_cells + preflight_replication_cells > maximum_external_cells
    ):
        checks.append({
            "id": "total_cell_budget",
            "status": "FAIL",
            "errors": [f"preflight requires {preflight_primary_cells + preflight_replication_cells} total cells; frozen maximum is {maximum_external_cells}"],
            "remedy": "reduce the external task set or amend the protocol before spending compute",
        })
    failed = [item["id"] for item in checks if item["status"] in {"FAIL", "BLOCKED"}]
    status = "READY_FOR_EXTERNAL_PREDICTION" if not failed else "BLOCKED"
    execution_budget = {
        "protocol_primary_cells": protocol.get("execution_budget", {}).get("primary_cells"),
        "protocol_replication_cells": protocol.get("execution_budget", {}).get("replication_cells"),
        "preflight_primary_cells": preflight_primary_cells,
        "preflight_replication_cells": preflight_replication_cells,
        "evaluation_budget_key": evaluation_budget_key,
        "declared_evaluation_cells": declared_evaluation_budget,
        "replication_task_count": replication_task_count,
        "cell_definition": protocol.get("execution_budget", {}).get("cell_definition"),
        "local_execution_authorized": False,
    }
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "status": status,
        "protocol_sha256": protocol_sha,
        "cohort_id": cohort_id,
        "external_compute_authorized": False,
        "runtime": {
            "python": platform.python_version(),
            "model_free_coordinator": True,
            "local_model_execution": False,
            "local_training": False,
        },
        "checks": checks,
        "check_summary": {
            "pass": sum(item["status"] == "PASS" for item in checks),
            "warning": sum(item["status"] == "WARNING" for item in checks),
            "fail": sum(item["status"] == "FAIL" for item in checks),
            "blocked": sum(item["status"] == "BLOCKED" for item in checks),
        },
        "blockers": failed,
        "next_actions": list(dict.fromkeys(
            item["remedy"] for item in checks if item["status"] in {"FAIL", "BLOCKED"}
        )),
        "input_hashes": _preflight_input_hashes(input_values),
        "execution_budget": execution_budget,
        "power_advisories": power_advisories,
        "dry_run": {
            "prediction_receipt_sha256": dry_run_prediction.get("prediction_receipt_sha256") if isinstance(dry_run_prediction, Mapping) else None,
            "model_execution": False,
            "training_execution": False,
        },
    }
    body["preflight_sha256"] = _sha_json(body)
    return body


def build_execution_plan(
    protocol: Mapping[str, Any], preflight: Mapping[str, Any], *, source_root: str | Path | None = None,
    intake: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol_verdict = validate_frozen_protocol(protocol, repo_root=source_root, check_source_closure=True)
    if not protocol_verdict["valid"]:
        raise X1ProtocolError("cannot build an execution plan from an invalid protocol")
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise X1EvidenceError("preflight artifact schema mismatch")
    preflight_hash = preflight.get("preflight_sha256")
    if not isinstance(preflight_hash, str) or _sha_json(_without(preflight, "preflight_sha256")) != preflight_hash:
        raise X1EvidenceError("preflight artifact hash mismatch")
    if preflight.get("protocol_sha256") != protocol.get("identity", {}).get("protocol_sha256"):
        raise X1EvidenceError("preflight protocol binding mismatch")
    if intake is not None:
        if intake.get("schema") != INTAKE_SCHEMA:
            raise X1EvidenceError("candidate intake schema mismatch")
        intake_hash = intake.get("intake_sha256")
        if not isinstance(intake_hash, str) or _sha_json(_without(intake, "intake_sha256")) != intake_hash:
            raise X1EvidenceError("candidate intake hash mismatch")
    check_summary = preflight.get("check_summary", {})
    checks_present = isinstance(preflight.get("checks"), list) and bool(preflight.get("checks"))
    custody_ready = not (
        preflight.get("input_hashes", {}).get("prediction_receipt") is not None
        and preflight.get("input_hashes", {}).get("prediction_commit") is None
    ) and not (
        preflight.get("input_hashes", {}).get("reveal_receipt") is not None
        and preflight.get("input_hashes", {}).get("reveal_commit") is None
    )
    preflight_clean = checks_present and not preflight.get("blockers") and not check_summary.get("fail") and not check_summary.get("blocked")
    plan_status = "BLOCKED" if (
        preflight.get("status") != "READY_FOR_EXTERNAL_PREDICTION" or not preflight_clean or not custody_ready
    ) else (
        "BLOCKED_POWER_DEVIATION_REQUIRED"
        if preflight.get("power_advisories") else "READY_FOR_EXTERNAL_EXECUTION"
    )
    body = {
        "schema": EXECUTION_PLAN_SCHEMA,
        "status": plan_status,
        "protocol_sha256": protocol.get("identity", {}).get("protocol_sha256"),
        "source_closure_sha256": protocol.get("identity", {}).get("source_closure_sha256"),
        "preflight_sha256": preflight.get("preflight_sha256"),
        "candidate_intake": {
            "schema": intake.get("schema") if intake else None,
            "sha256": intake.get("intake_sha256") if intake else None,
            "status": intake.get("status") if intake else None,
            "checkpoint_file_sha256": intake.get("checkpoint_file_sha256") if intake else None,
            "checkpoint_bytes": intake.get("files", {}).get("checkpoint", {}).get("bytes") if intake else None,
            "platform_guidance": intake.get("platform_guidance") if intake else None,
        },
        "platform_profiles": {
            "COLAB": {
                "role": "interactive bring-up and external execution",
                "constraints": ["session disconnects", "storage and quota limits", "do not treat a notebook as custody"],
            },
            "KAGGLE": {
                "role": "external GPU execution when notebook persistence and runtime limits are accepted",
                "constraints": ["kernel restarts", "dataset and output quotas", "do not expose sealed artifacts"],
            },
            "PERSISTENT_GPU_RUNNER": {
                "role": "preferred for final high-value execution",
                "constraints": ["requires explicit operator custody and checkpointed phase outputs"],
            },
        },
        "claim_ceiling": protocol.get("claim_ceiling"),
        "external_execution_required": True,
        "external_compute_authorized": False,
        "local_model_execution": False,
        "local_training": False,
        "cohorts": protocol.get("cohorts"),
        "execution_budget": protocol.get("execution_budget"),
        "required_phases": [
            {"phase": "BASIS_QUALIFIED", "input": "basis source and split", "output": "basis qualification receipt and phase commit"},
            {"phase": "PREDICT_COMMITTED", "input": "frozen predictor, baselines, public tasks", "output": "prediction receipt and phase commit"},
            {"phase": "REVEAL_ACCEPTED", "input": "external outcomes and structured verifier artifacts", "output": "reveal receipt and phase commit"},
            {"phase": "ANALYSIS", "input": "primary and optional replication bundles", "output": "hash-bound analysis receipt"},
        ],
        "stop_conditions": protocol.get("stop_conditions"),
        "decision_thresholds": protocol.get("decision_thresholds"),
        "retention_required": [
            "protocol and source closure", "subject and registry evidence", "release attestation",
            "basis source and split", "predictor training manifest and artifact", "baseline commitment",
            "public task bundle", "prediction receipt and phase commit", "outcome and verifier artifacts",
            "reveal receipt and phase commit", "analysis receipt and replication evidence",
        ],
        "operator_sequence": [
            "run preflight and preserve its JSON receipt",
            "resolve every power advisory with an approved protocol deviation or a target-sized split",
            "obtain external execution authorization for the frozen cell budget",
            "run basis qualification before fitting the predictor",
            "commit prediction before exposing any outcome",
            "commit reveal with file-bound verifier artifacts",
            "run analysis and preserve all non-passing outcomes",
        ],
        "blockers": list(dict.fromkeys(
            list(preflight.get("blockers", [])) + (["POWER_DEVIATION_REQUIRED"] if preflight.get("power_advisories") else [])
        )),
        "power_advisories": preflight.get("power_advisories", []),
    }
    body["execution_plan_sha256"] = _sha_json(body)
    return body


def build_baseline_commitment(
    development_rows: Sequence[Mapping[str, Any]], evaluation_task_ids: Sequence[str], registry: Any,
    *, development_artifact_sha256: str, development_artifact_path: str,
    development_cohort_id: str, development_split_manifest_sha256: str, selection_lambda: float,
    seed: int = 81818, surface_keys: Mapping[str, str] | None = None,
    development_split_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry_verdict = validate_intervention_registry(registry)
    if not registry_verdict["valid"]:
        raise X1EvidenceError("cannot fit baselines against invalid intervention registry")
    intervention_ids = list(registry_verdict["intervention_ids"])
    if not development_rows:
        raise X1EvidenceError("development rows are required for fixed baselines")
    if development_split_manifest is not None:
        expected_ids = {
            task_id for cluster in development_split_manifest.get("clusters", [])
            for task_id in cluster.get("task_ids", [])
        }
        actual_ids = {str(row.get("task_id")) for row in development_rows if isinstance(row, Mapping)}
        if actual_ids != expected_ids:
            raise X1EvidenceError("development rows do not match the development split")
    rates = {intervention_id: [] for intervention_id in intervention_ids}
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in development_rows:
        outcomes = row.get("outcomes")
        if not isinstance(outcomes, Mapping) or set(outcomes) != set(intervention_ids):
            raise X1EvidenceError("development outcome coverage is incomplete")
        surface = row.get("surface_key")
        surface_key = str(surface) if surface is not None else "__all__"
        for intervention_id in intervention_ids:
            value = outcomes[intervention_id]
            if isinstance(value, Mapping):
                value = value.get("repaired")
            if not isinstance(value, (bool, int)) or isinstance(value, float) or int(value) not in (0, 1):
                raise X1EvidenceError("development outcomes must be boolean or 0/1")
            repaired = bool(value)
            rates[intervention_id].append(int(repaired))
            groups[surface_key][intervention_id].append(int(repaired))
    rates = {key: sum(value) / len(value) for key, value in rates.items()}
    costs = {item["id"]: int(item["cost"]) for item in _registry_items(registry)}
    order = {item: index for index, item in enumerate(intervention_ids)}
    best_fixed = min(
        intervention_ids,
        key=lambda item: (-rates[item], costs[item], order[item]),
    )
    cost_aware = min(
        intervention_ids,
        key=lambda item: (-(rates[item] - selection_lambda * costs[item]), costs[item], order[item]),
    )
    surface_choice = {}
    for key, group in groups.items():
        surface_choice[key] = min(
            intervention_ids,
            key=lambda item: (-(sum(group[item]) / len(group[item])), costs[item], order[item]),
        )
    eval_surface = surface_keys or {}
    baselines: dict[str, dict[str, Any]] = {}
    for task_id in evaluation_task_ids:
        surface_choice_id = surface_choice.get(str(eval_surface.get(task_id, "__all__")), best_fixed)
        baselines.setdefault("ALWAYS_NEGATIVE", {})[task_id] = {
            "predicted_repair_probability": [0.0] * len(intervention_ids),
            "predicted_best_intervention": "NO_CHANGE",
            "uncertainty": 0.0,
        }
        baselines.setdefault("PREVALENCE_ONLY", {})[task_id] = {
            "predicted_repair_probability": [rates[item] for item in intervention_ids],
            "predicted_best_intervention": min(
                intervention_ids, key=lambda item: (-(rates[item] - selection_lambda * costs[item]), costs[item], order[item])
            ),
            "uncertainty": 0.0,
        }
        baselines.setdefault("BEST_FIXED_DEV", {})[task_id] = {
            "predicted_repair_probability": [1.0 if item == best_fixed else 0.0 for item in intervention_ids],
            "predicted_best_intervention": best_fixed,
            "uncertainty": 0.0,
        }
        baselines.setdefault("COST_AWARE_FIXED", {})[task_id] = {
            "predicted_repair_probability": [1.0 if item == cost_aware else 0.0 for item in intervention_ids],
            "predicted_best_intervention": cost_aware,
            "uncertainty": 0.0,
        }
        baselines.setdefault("SURFACE_SHORTCUT", {})[task_id] = {
            "predicted_repair_probability": [1.0 if item == surface_choice_id else 0.0 for item in intervention_ids],
            "predicted_best_intervention": surface_choice_id,
            "uncertainty": 0.0,
        }
    rng = random.Random(seed)
    for task_id in evaluation_task_ids:
        values = [1.0 if rng.random() < rates[item] else 0.0 for item in intervention_ids]
        choice = max(range(len(values)), key=lambda index: (values[index], -index))
        baselines.setdefault("SPARSITY_RANDOM", {})[task_id] = {
            "predicted_repair_probability": values,
            "predicted_best_intervention": intervention_ids[choice],
            "uncertainty": 0.0,
        }
    body = {
        "schema": "anra-x1-real-1-baseline-commitment/v1",
        "development_artifact_sha256": development_artifact_sha256,
        "development_artifact_path": development_artifact_path,
        "development_cohort_id": development_cohort_id,
        "development_split_manifest_sha256": development_split_manifest_sha256,
        "development_prevalence": sum(rates.values()) / len(rates),
        "development_prevalence_by_intervention": {
            intervention_id: rates[intervention_id] for intervention_id in intervention_ids
        },
        "selection_lambda": selection_lambda,
        "baselines": baselines,
        "fit_scope": "DEVELOPMENT_ONLY",
        "surface_shortcut_diagnostic_only": True,
    }
    body["commitment_sha256"] = _sha_json(body)
    return body


def validate_phase_commit(
    commit: Mapping[str, Any], *, expected_phase: str, expected_receipt_sha256: str,
    expected_previous_commit_sha256: str | None = None, repo_root: str | Path | None = None,
    previous_commit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = {
        "schema", "phase", "phase_index", "receipt_sha256", "artifact_path", "artifact_sha256",
        "sequence", "committed_at", "previous_commit_sha256", "custodian", "attestation_sha256",
        "phase_commit_sha256",
    }
    errors: list[str] = []
    try:
        _assert_exact_keys(commit, required)
    except X1EvidenceError as exc:
        errors.append(str(exc))
    if commit.get("schema") != PHASE_COMMIT_SCHEMA:
        errors.append("phase commit schema mismatch")
    if commit.get("phase") != expected_phase:
        errors.append("phase commit phase mismatch")
    expected_index = {
        "BASIS_QUALIFIED": 0,
        "PREDICT_COMMITTED": 1,
        "REVEAL_ACCEPTED": 2,
    }.get(expected_phase)
    if expected_index is None:
        errors.append("unsupported expected phase")
        expected_index = -1
    if commit.get("phase_index") != expected_index or commit.get("sequence") != expected_index:
        errors.append("phase commit sequence is not monotonic")
    if commit.get("receipt_sha256") != expected_receipt_sha256:
        errors.append("phase commit receipt parent mismatch")
    if commit.get("previous_commit_sha256") != expected_previous_commit_sha256:
        errors.append("phase commit previous-parent mismatch")
    for key in ("artifact_sha256", "attestation_sha256", "phase_commit_sha256"):
        try:
            _hash(commit.get(key), f"phase_commit.{key}")
        except X1EvidenceError as exc:
            errors.append(str(exc))
    parsed_timestamp = None
    try:
        _nonempty(commit.get("custodian"), "phase_commit.custodian")
        timestamp = commit.get("committed_at")
        if not isinstance(timestamp, str) or not timestamp:
            raise X1EvidenceError("phase commit timestamp is missing")
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (ValueError, X1EvidenceError) as exc:
        errors.append(str(exc))
    if previous_commit is not None and parsed_timestamp is not None:
        previous_timestamp = previous_commit.get("committed_at")
        if isinstance(previous_timestamp, str):
            try:
                if parsed_timestamp <= datetime.fromisoformat(previous_timestamp.replace("Z", "+00:00")):
                    errors.append("phase commit timestamp is not after its parent")
            except ValueError:
                errors.append("parent phase commit timestamp is malformed")
    artifact_path = commit.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        errors.append("phase commit artifact path is missing")
    else:
        candidate = Path(artifact_path)
        if not candidate.is_absolute():
            candidate = Path(repo_root or Path.cwd()) / candidate
        if not candidate.is_file():
            errors.append(f"phase commit artifact is missing: {candidate}")
        elif _sha_file(candidate) != commit.get("artifact_sha256"):
            errors.append("phase commit artifact hash mismatch")
    if isinstance(commit.get("phase_commit_sha256"), str):
        if _sha_json(_without(commit, "phase_commit_sha256")) != commit.get("phase_commit_sha256"):
            errors.append("phase commit hash mismatch")
    return {"valid": not errors, "errors": errors}


def commit_phase_record(
    receipt: Mapping[str, Any], *, receipt_artifact_path: str | Path, phase: str,
    custodian: str, attestation_sha256: str, previous_commit_sha256: str | None = None,
    committed_at: str | None = None,
) -> dict[str, Any]:
    if phase not in {"BASIS_QUALIFIED", "PREDICT_COMMITTED", "REVEAL_ACCEPTED"}:
        raise X1EvidenceError("unsupported phase commit")
    artifact_path = Path(receipt_artifact_path)
    commit_path = artifact_path.with_suffix(artifact_path.suffix + ".commit.json")
    phase_index = {"BASIS_QUALIFIED": 0, "PREDICT_COMMITTED": 1, "REVEAL_ACCEPTED": 2}[phase]
    receipt_hash = (
        receipt.get("basis_source_artifact_sha256") if phase == "BASIS_QUALIFIED"
        else receipt.get("prediction_receipt_sha256") if phase == "PREDICT_COMMITTED"
        else receipt.get("reveal_receipt_sha256")
    )
    phase_artifact_path = artifact_path
    phase_artifact_hash = None
    if phase == "BASIS_QUALIFIED":
        phase_artifact_path = Path(str(receipt.get("basis_binding", {}).get("basis_source_artifact_path", artifact_path)))
        phase_artifact_hash = str(receipt.get("basis_source_artifact_sha256"))
    if commit_path.exists():
        if not artifact_path.is_file():
            raise X1EvidenceError(f"phase receipt is missing while commit exists: {artifact_path}")
        existing_receipt = _load_json(artifact_path)
        if _sha_json(existing_receipt) != _sha_json(receipt):
            raise X1EvidenceError("existing phase receipt differs from the requested receipt")
        existing_commit = _load_json(commit_path)
        if existing_commit.get("phase") != phase or existing_commit.get("phase_index") != phase_index:
            raise X1EvidenceError("existing phase commit has a different phase")
        if existing_commit.get("receipt_sha256") != receipt_hash:
            raise X1EvidenceError("existing phase commit has a different receipt parent")
        if existing_commit.get("custodian") != custodian or existing_commit.get("attestation_sha256") != attestation_sha256:
            raise X1EvidenceError("existing phase commit has a different custody attestation")
        if phase == "BASIS_QUALIFIED" and (
            existing_commit.get("artifact_path") != str(phase_artifact_path)
            or existing_commit.get("artifact_sha256") != phase_artifact_hash
        ):
            raise X1EvidenceError("existing basis phase commit has a different source artifact")
        existing_verdict = validate_phase_commit(
            existing_commit, expected_phase=phase, expected_receipt_sha256=receipt_hash,
            expected_previous_commit_sha256=previous_commit_sha256,
            repo_root=artifact_path.parent,
        )
        if not existing_verdict["valid"]:
            raise X1EvidenceError("existing phase commit is invalid: " + "; ".join(existing_verdict["errors"]))
        return existing_commit
    if artifact_path.exists():
        existing_receipt = _load_json(artifact_path)
        if _sha_json(existing_receipt) != _sha_json(receipt):
            raise X1EvidenceError("existing receipt differs from the requested receipt")
        artifact_hash = _sha_file(artifact_path)
    else:
        artifact_hash = write_receipt_no_overwrite(artifact_path, receipt)
    timestamp = committed_at or datetime.now(timezone.utc).isoformat()
    if phase_artifact_hash is None:
        phase_artifact_hash = artifact_hash
    body = {
        "schema": PHASE_COMMIT_SCHEMA,
        "phase": phase,
        "phase_index": phase_index,
        "receipt_sha256": receipt_hash,
        "artifact_path": str(phase_artifact_path),
        "artifact_sha256": phase_artifact_hash,
        "sequence": phase_index,
        "committed_at": timestamp,
        "previous_commit_sha256": previous_commit_sha256,
        "custodian": custodian,
        "attestation_sha256": attestation_sha256,
    }
    _hash(body["receipt_sha256"], "phase_commit.receipt_sha256")
    _hash(attestation_sha256, "phase_commit.attestation_sha256")
    body["phase_commit_sha256"] = _sha_json(body)
    write_receipt_no_overwrite(commit_path, body)
    return body


def write_receipt_no_overwrite(path: str | Path, receipt: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_plain(receipt), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with open(target, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise X1EvidenceError(f"receipt already exists; refusing overwrite: {target}") from exc
    return _sha_file(target)


def _cli_load(path: str | Path) -> Any:
    return _load_json(path)


def _main_uncaught(argv: Sequence[str] | None = None) -> int:
    parser = _CLIArgumentParser(prog="anra-x1-real-1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--registry")
    inventory.add_argument("--verify-files", action="store_true")

    intake = subparsers.add_parser("intake")
    intake.add_argument("--checkpoint", required=True)
    intake.add_argument("--config")
    intake.add_argument("--tokenizer")
    intake.add_argument("--runtime-source-revision")
    intake.add_argument("--source-commit")
    intake.add_argument("--global-step", type=int)
    intake.add_argument("--stage")
    intake.add_argument("--parameter-sha256")
    intake.add_argument("--tokenizer-identity-sha256")
    intake.add_argument("--readiness-receipt")
    intake.add_argument("--readiness-receipt-sha256")
    intake.add_argument("--out")

    gate_parser = subparsers.add_parser("gate")
    gate_parser.add_argument("--registry")
    gate_parser.add_argument("--basis-qualification")
    gate_parser.add_argument("--release-manifest")
    gate_parser.add_argument("--protocol")
    gate_parser.add_argument("--release-root")

    protocol_parser = subparsers.add_parser("validate-protocol")
    protocol_parser.add_argument("--protocol", required=True)
    protocol_parser.add_argument("--repo-root")
    protocol_parser.add_argument("--check-source-closure", action="store_true")

    subject_parser = subparsers.add_parser("validate-subject")
    subject_parser.add_argument("--manifest", required=True)
    subject_parser.add_argument("--registry")
    subject_parser.add_argument("--checkpoint-root")

    basis_parser = subparsers.add_parser("qualify-basis")
    basis_parser.add_argument("--artifact", required=True)
    basis_parser.add_argument("--protocol", required=True)
    basis_parser.add_argument("--subject", required=True)
    basis_parser.add_argument("--binding", required=True)
    basis_parser.add_argument("--out", required=True)

    prediction_parser = subparsers.add_parser("commit-prediction")
    prediction_parser.add_argument("--protocol", required=True)
    prediction_parser.add_argument("--subject", required=True)
    prediction_parser.add_argument("--split", required=True)
    prediction_parser.add_argument("--tasks", required=True)
    prediction_parser.add_argument("--predictions", required=True)
    prediction_parser.add_argument("--baselines", required=True)
    prediction_parser.add_argument("--predictor-identity", required=True)
    prediction_parser.add_argument("--basis-qualification", required=True)
    prediction_parser.add_argument("--cohort-id", required=True)
    prediction_parser.add_argument("--development-split", required=True)
    prediction_parser.add_argument("--basis-split", required=True)
    prediction_parser.add_argument("--release-manifest", required=True)
    prediction_parser.add_argument("--release-root")
    prediction_parser.add_argument("--source-root")
    prediction_parser.add_argument("--artifact-root")
    prediction_parser.add_argument("--checkpoint-root")
    prediction_parser.add_argument("--registry")
    prediction_parser.add_argument("--out", required=True)

    reveal_parser = subparsers.add_parser("commit-reveal")
    reveal_parser.add_argument("--prediction", required=True)
    reveal_parser.add_argument("--outcomes", required=True)
    reveal_parser.add_argument("--split", required=True)
    reveal_parser.add_argument("--evaluator-identity", required=True)
    reveal_parser.add_argument("--prediction-commit", required=True)
    reveal_parser.add_argument("--protocol", required=True)
    reveal_parser.add_argument("--subject", required=True)
    reveal_parser.add_argument("--tasks", required=True)
    reveal_parser.add_argument("--basis-qualification", required=True)
    reveal_parser.add_argument("--release-manifest", required=True)
    reveal_parser.add_argument("--release-root")
    reveal_parser.add_argument("--source-root")
    reveal_parser.add_argument("--checkpoint-root")
    reveal_parser.add_argument("--registry")
    reveal_parser.add_argument("--out", required=True)

    phase_parser = subparsers.add_parser("commit-phase")
    phase_parser.add_argument("--receipt", required=True)
    phase_parser.add_argument("--artifact-path", required=True)
    phase_parser.add_argument("--phase", required=True, choices=["BASIS_QUALIFIED", "PREDICT_COMMITTED", "REVEAL_ACCEPTED"])
    phase_parser.add_argument("--custodian", required=True)
    phase_parser.add_argument("--attestation-sha256", required=True)
    phase_parser.add_argument("--previous-commit")
    phase_parser.add_argument("--out", required=True)

    preflight_parser = subparsers.add_parser("preflight", aliases=["dry-run"])
    preflight_parser.add_argument("--protocol", required=True)
    preflight_parser.add_argument("--subject")
    preflight_parser.add_argument("--basis-qualification")
    preflight_parser.add_argument("--release-manifest")
    preflight_parser.add_argument("--registry")
    preflight_parser.add_argument("--checkpoint-root")
    preflight_parser.add_argument("--release-root")
    preflight_parser.add_argument("--source-root")
    preflight_parser.add_argument("--split")
    preflight_parser.add_argument("--basis-split")
    preflight_parser.add_argument("--development-split")
    preflight_parser.add_argument("--tasks")
    preflight_parser.add_argument("--predictor-identity")
    preflight_parser.add_argument("--baselines")
    preflight_parser.add_argument("--predictions")
    preflight_parser.add_argument("--cohort-id", default="PRIMARY_EVAL")
    preflight_parser.add_argument("--replication-tasks", type=int)
    preflight_parser.add_argument("--prediction")
    preflight_parser.add_argument("--prediction-commit")
    preflight_parser.add_argument("--reveal")
    preflight_parser.add_argument("--outcomes")
    preflight_parser.add_argument("--evaluator-identity")
    preflight_parser.add_argument("--reveal-commit")
    preflight_parser.add_argument("--replication")
    preflight_parser.add_argument("--out")

    plan_parser = subparsers.add_parser("execution-plan")
    plan_parser.add_argument("--protocol", required=True)
    plan_parser.add_argument("--preflight", required=True)
    plan_parser.add_argument("--source-root")
    plan_parser.add_argument("--intake")
    plan_parser.add_argument("--out")

    analysis_parser = subparsers.add_parser("analyze")
    analysis_parser.add_argument("--protocol", required=True)
    analysis_parser.add_argument("--subject", required=True)
    analysis_parser.add_argument("--split", required=True)
    analysis_parser.add_argument("--prediction", required=True)
    analysis_parser.add_argument("--reveal", required=True)
    analysis_parser.add_argument("--basis-qualification", required=True)
    analysis_parser.add_argument("--replication")
    analysis_parser.add_argument("--primary-bundle")
    analysis_parser.add_argument("--replication-bundle")
    analysis_parser.add_argument("--tasks", required=True)
    analysis_parser.add_argument("--prediction-commit", required=True)
    analysis_parser.add_argument("--reveal-commit", required=True)
    analysis_parser.add_argument("--release-manifest", required=True)
    analysis_parser.add_argument("--release-root")
    analysis_parser.add_argument("--source-root")
    analysis_parser.add_argument("--basis-split", required=True)
    analysis_parser.add_argument("--development-split", required=True)
    analysis_parser.add_argument("--checkpoint-root")
    analysis_parser.add_argument("--primary-analysis-sha256")
    analysis_parser.add_argument("--registry")
    analysis_parser.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "intake":
        result = intake_checkpoint_candidate(
            args.checkpoint, config_path=args.config, tokenizer_path=args.tokenizer,
            runtime_source_revision=args.runtime_source_revision, source_commit=args.source_commit,
            global_step=args.global_step, stage=args.stage,
            parameter_sha256=args.parameter_sha256,
            tokenizer_identity_sha256=args.tokenizer_identity_sha256,
            readiness_receipt_path=args.readiness_receipt,
            readiness_receipt_sha256=args.readiness_receipt_sha256,
        )
        if args.out:
            write_receipt_no_overwrite(args.out, result)
    elif args.command == "inventory":
        result = inventory_checkpoint_candidates(args.registry, verify_files=args.verify_files)
    elif args.command == "gate":
        result = prerequisite_gate(
            registry_path=args.registry,
            basis_qualification=_cli_load(args.basis_qualification) if args.basis_qualification else None,
            release_manifest=_cli_load(args.release_manifest) if args.release_manifest else None,
            protocol=_cli_load(args.protocol) if args.protocol else _load_json(
                Path(__file__).resolve().parent / "protocols" / "x1_real_1_v1.json"
            ),
            release_root=args.release_root,
        )
    elif args.command == "validate-protocol":
        document = _cli_load(args.protocol)
        result = validate_frozen_protocol(document, repo_root=args.repo_root, check_source_closure=args.check_source_closure)
    elif args.command == "validate-subject":
        result = validate_subject_manifest(
            _cli_load(args.manifest), registry_path=args.registry, checkpoint_root=args.checkpoint_root,
            verify_checkpoint_file=True,
        )
    elif args.command == "qualify-basis":
        document = _cli_load(args.protocol)
        result = qualify_intervention_basis(
            _cli_load(args.artifact), document.get("interventions"),
            protocol_sha256=document.get("identity", {}).get("protocol_sha256"),
            subject_manifest_sha256=_sha_json(_cli_load(args.subject)),
            thresholds=document.get("decision_thresholds", {}).get("basis"),
            binding=_cli_load(args.binding),
        )
        write_receipt_no_overwrite(args.out, result)
    elif args.command == "commit-prediction":
        protocol_document = _cli_load(args.protocol)
        result = commit_prediction_receipt(
            protocol_document,
            _cli_load(args.subject),
            _cli_load(args.split),
            _cli_load(args.tasks),
            _cli_load(args.predictions),
            _cli_load(args.baselines),
            _cli_load(args.predictor_identity),
            _cli_load(args.basis_qualification),
            cohort_id=args.cohort_id,
            registry_path=args.registry,
            checkpoint_root=args.checkpoint_root,
            release_manifest=_cli_load(args.release_manifest),
            release_root=args.release_root,
            source_root=args.source_root,
            artifact_root=args.artifact_root,
            development_split_manifest=_cli_load(args.development_split),
            basis_split_manifest=_cli_load(args.basis_split),
        )
        write_receipt_no_overwrite(args.out, result)
    elif args.command == "commit-reveal":
        result = commit_reveal_receipt(
            _cli_load(args.prediction),
            _cli_load(args.outcomes),
            evaluator_identity=_cli_load(args.evaluator_identity),
            split_manifest=_cli_load(args.split),
            prediction_commit=_cli_load(args.prediction_commit),
            protocol=_cli_load(args.protocol),
            subject_manifest=_cli_load(args.subject),
            registry_path=args.registry,
            checkpoint_root=args.checkpoint_root,
            release_manifest=_cli_load(args.release_manifest),
            release_root=args.release_root,
            source_root=args.source_root,
            public_tasks=_cli_load(args.tasks),
            basis_qualification=_cli_load(args.basis_qualification),
        )
        write_receipt_no_overwrite(args.out, result)
    elif args.command == "commit-phase":
        result = commit_phase_record(
            _cli_load(args.receipt), receipt_artifact_path=args.artifact_path,
            phase=args.phase, custodian=args.custodian,
            attestation_sha256=args.attestation_sha256,
            previous_commit_sha256=args.previous_commit,
        )
        write_receipt_no_overwrite(args.out, result)
    elif args.command in {"preflight", "dry-run"}:
        result = preflight_x1_real_1(
            _cli_load(args.protocol),
            subject_manifest=_cli_load(args.subject) if args.subject else None,
            basis_qualification=_cli_load(args.basis_qualification) if args.basis_qualification else None,
            release_manifest=_cli_load(args.release_manifest) if args.release_manifest else None,
            registry_path=args.registry,
            checkpoint_root=args.checkpoint_root,
            release_root=args.release_root,
            source_root=args.source_root,
            primary_split_manifest=_cli_load(args.split) if args.split else None,
            basis_split_manifest=_cli_load(args.basis_split) if args.basis_split else None,
            development_split_manifest=_cli_load(args.development_split) if args.development_split else None,
            public_tasks=_cli_load(args.tasks) if args.tasks else None,
            predictor_identity=_cli_load(args.predictor_identity) if args.predictor_identity else None,
            baseline_commitment=_cli_load(args.baselines) if args.baselines else None,
            predictor_predictions=_cli_load(args.predictions) if args.predictions else None,
            cohort_id=args.cohort_id,
            replication_task_count=args.replication_tasks,
            prediction_receipt=_cli_load(args.prediction) if args.prediction else None,
            prediction_commit=_cli_load(args.prediction_commit) if args.prediction_commit else None,
            reveal_receipt=_cli_load(args.reveal) if args.reveal else None,
            reveal_commit=_cli_load(args.reveal_commit) if args.reveal_commit else None,
            outcomes=_cli_load(args.outcomes) if args.outcomes else None,
            evaluator_identity=_cli_load(args.evaluator_identity) if args.evaluator_identity else None,
            replication_receipt=_cli_load(args.replication) if args.replication else None,
        )
        if args.out:
            write_receipt_no_overwrite(args.out, result)
    elif args.command == "execution-plan":
        result = build_execution_plan(
            _cli_load(args.protocol), _cli_load(args.preflight), source_root=args.source_root,
            intake=_cli_load(args.intake) if args.intake else None,
        )
        if args.out:
            write_receipt_no_overwrite(args.out, result)
    elif args.command == "analyze":
        document = _cli_load(args.protocol)
        result = analyze_x1_real_1(
            document, _cli_load(args.subject), _cli_load(args.split), _cli_load(args.prediction),
            _cli_load(args.reveal), basis_qualification=_cli_load(args.basis_qualification),
            replication_receipt=_cli_load(args.replication) if args.replication else None,
            primary_bundle=_cli_load(args.primary_bundle) if args.primary_bundle else None,
            replication_bundle=_cli_load(args.replication_bundle) if args.replication_bundle else None,
            registry_path=args.registry,
            checkpoint_root=args.checkpoint_root,
            release_manifest=_cli_load(args.release_manifest),
            release_root=args.release_root,
            source_root=args.source_root,
            public_tasks=_cli_load(args.tasks),
            prediction_commit=_cli_load(args.prediction_commit),
            reveal_commit=_cli_load(args.reveal_commit),
            basis_split_manifest=_cli_load(args.basis_split),
            development_split_manifest=_cli_load(args.development_split),
            primary_analysis_sha256=args.primary_analysis_sha256,
        )
        write_receipt_no_overwrite(args.out, result)
    else:
        raise X1ProtocolError("unsupported command")
    print(json.dumps(_plain(result), indent=2, sort_keys=True, ensure_ascii=False))
    if args.command == "intake":
        return 0 if result.get("status") in {"INVENTORIED", "UNQUALIFIED_NEW"} else 2
    if args.command in {"preflight", "dry-run", "execution-plan"}:
        return 0 if result.get("status") in {
            "READY_FOR_EXTERNAL_PREDICTION", "READY_FOR_EXTERNAL_EXECUTION",
        } else 2
    if args.command in {"commit-phase", "commit-prediction", "commit-reveal"}:
        return 0
    if args.command == "analyze" and result.get("schema") == ANALYSIS_SCHEMA:
        return 2 if result.get("phase") == "ANALYSIS_BLOCKED" or result.get("decision", {}).get("status") == "INCONCLUSIVE" else 0
    return 0 if result.get("valid", result.get("status") in {"ELIGIBLE", "ELIGIBLE_SUBJECT_AVAILABLE", "QUALIFIED", "READY_FOR_EXTERNAL_PHASE_1"}) else 2


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main_uncaught(argv)
    except SystemExit as exc:
        if exc.code == 0:
            raise
        payload = {
            "schema": CLI_ERROR_SCHEMA,
            "status": "ERROR",
            "command": str(argv[0]) if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)) and argv else None,
            "error_type": "CLIUsageError",
            "message": "command-line usage error",
            "next_action": "correct the command-line options and rerun preflight",
        }
        print(json.dumps(_plain(payload), indent=2, sort_keys=True, ensure_ascii=False))
        return 2
    except Exception as exc:
        command = None
        if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)) and argv:
            command = str(argv[0])
        if isinstance(exc, FileNotFoundError):
            remedy = "create the missing artifact or pass the correct path; no computation should start"
        elif isinstance(exc, X1ChronologyError):
            remedy = "repair the receipt chronology and rerun preflight before external execution"
        elif isinstance(exc, X1EvidenceError):
            remedy = "repair the reported artifact identity or schema, then rerun preflight"
        elif isinstance(exc, X1ProtocolError):
            remedy = "freeze a valid protocol and source closure before continuing"
        else:
            remedy = "inspect the input artifact shape and rerun preflight; do not start external compute"
        payload = {
            "schema": CLI_ERROR_SCHEMA,
            "status": "ERROR",
            "command": command,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "next_action": remedy,
        }
        print(json.dumps(_plain(payload), indent=2, sort_keys=True, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
