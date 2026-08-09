"""Fail-closed lineage and evidence contracts for post-training.

The numerical training loops live in their existing modules.  This module is
the control boundary that decides whether a curated SFT corpus, verifier
outcomes, or preference pairs are eligible to enter a canonical lineage.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SFT_LINEAGE_SCHEMA = "anra-sft-lineage/v1"
POSTTRAINING_GATE_SCHEMA = "anra-posttraining-gate/v1"
REQUIRED_SFT_CATEGORIES = (
    "instruction_following",
    "dialogue",
    "code",
    "mathematics",
    "decomposition",
    "tool_contracts",
    "uncertainty",
    "correction",
)
VERIFIABLE_DOMAINS = frozenset({"code", "mathematics", "exact", "symbolic"})
AUDITED_PREFERENCE_SOURCES = frozenset(
    {"human", "expert", "production_feedback", "verified_comparison"}
)
_HASH_LENGTH = 64


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_hash(value: object, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != _HASH_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _positive_counts(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise ValueError("SFT dataset manifest requires category_counts")
    counts: dict[str, int] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid SFT category count for {name!r}")
        counts[str(name)] = value
    missing = [name for name in REQUIRED_SFT_CATEGORIES if counts.get(name, 0) <= 0]
    if missing:
        raise ValueError(f"SFT dataset has no accepted examples for: {missing}")
    return dict(sorted(counts.items()))


def _nonnegative_counts(raw: object, *, name: str) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} requires category_counts")
    counts: dict[str, int] = {}
    for category, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid {name} category count for {category!r}")
        counts[str(category)] = value
    if not counts or sum(counts.values()) <= 0:
        raise ValueError(f"{name} has no accepted examples")
    return dict(sorted(counts.items()))


def _verified_dataset_artifacts(
    raw: object, *, manifest_dir: Path
) -> list[dict[str, object]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("SFT dataset manifest requires at least one artifact")
    verified: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("SFT dataset artifacts must be objects")
        artifact = Path(str(item.get("path", "")))
        if not artifact.is_absolute():
            artifact = manifest_dir / artifact
        artifact = artifact.resolve()
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        expected_hash = _validated_hash(item.get("sha256"), "SFT artifact hash")
        actual_hash = _sha256_file(artifact)
        if not hmac.compare_digest(expected_hash, actual_hash):
            raise ValueError(f"SFT artifact hash mismatch: {artifact}")
        expected_size = item.get("size_bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise ValueError("SFT artifact size_bytes must be an integer")
        if expected_size != artifact.stat().st_size:
            raise ValueError(f"SFT artifact size mismatch: {artifact}")
        verified.append(
            {
                "path": str(artifact),
                "sha256": actual_hash,
                "size_bytes": expected_size,
            }
        )
    return verified


def _seal(body: dict[str, Any], *, signing_key: str, key_id: str) -> dict[str, Any]:
    if not signing_key:
        raise PermissionError("a signing key is required for canonical post-training")
    if not key_id.strip():
        raise ValueError("signing key_id is required")
    manifest_hash = _sha256(body)
    return {
        **body,
        "manifest_sha256": manifest_hash,
        "signature": {
            "algorithm": "hmac-sha256",
            "key_id": key_id.strip(),
            "value": hmac.new(
                signing_key.encode("utf-8"),
                manifest_hash.encode("ascii"),
                hashlib.sha256,
            ).hexdigest(),
        },
    }


def _verify_seal(
    payload: Mapping[str, Any], *, signing_key: str, expected_schema: str
) -> dict[str, Any]:
    if payload.get("schema") != expected_schema:
        raise ValueError(f"unsupported manifest schema: {payload.get('schema')!r}")
    manifest_hash = _validated_hash(payload.get("manifest_sha256"), "manifest hash")
    body = {
        name: value
        for name, value in payload.items()
        if name not in {"manifest_sha256", "signature"}
    }
    if not hmac.compare_digest(manifest_hash, _sha256(body)):
        raise ValueError("post-training manifest content hash mismatch")
    signature = payload.get("signature")
    if not isinstance(signature, Mapping) or signature.get("algorithm") != "hmac-sha256":
        raise ValueError("canonical post-training manifest is not signed")
    if not signing_key:
        raise PermissionError("signing key is required to verify canonical post-training")
    expected = hmac.new(
        signing_key.encode("utf-8"), manifest_hash.encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(str(signature.get("value", "")), expected):
        raise ValueError("post-training manifest signature mismatch")
    return body


def _write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") == encoded:
            return target
        raise FileExistsError(f"immutable manifest already exists: {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_sft_lineage_manifest(
    output_path: str | Path,
    *,
    lineage_id: str,
    dataset_manifest_path: str | Path,
    validation_manifest_path: str | Path | None = None,
    base_checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    source_commit: str,
    signing_key: str,
    key_id: str = "owner",
) -> dict[str, Any]:
    """Verify and bind a quality-controlled SFT corpus to one immutable base."""

    dataset_path = Path(dataset_manifest_path).resolve()
    checkpoint_path = Path(base_checkpoint_path).resolve()
    tokenizer = Path(tokenizer_path).resolve()
    if not lineage_id.strip() or not source_commit.strip():
        raise ValueError("SFT lineage_id and source_commit are required")
    required_paths = (dataset_path, checkpoint_path, tokenizer)
    if validation_manifest_path is not None:
        required_paths = (*required_paths, Path(validation_manifest_path).resolve())
    for required_path in required_paths:
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, Mapping):
        raise ValueError("SFT dataset manifest must be a JSON object")
    if dataset.get("quality_gate_passed") is not True:
        raise ValueError("SFT dataset quality gate has not passed")
    if dataset.get("licenses_audited") is not True:
        raise ValueError("SFT dataset licenses have not been audited")
    if dataset.get("unregistered_local_pilot") is True:
        raise PermissionError("unregistered local SFT pilots cannot create a canonical lineage")
    receipt_hash = _validated_hash(
        dataset.get("source_receipt_sha256"), "SFT source receipt hash"
    )
    if dataset.get("split") != "train":
        raise ValueError("SFT lineage may only consume the immutable train split")
    categories = _positive_counts(dataset.get("category_counts"))
    artifacts = _verified_dataset_artifacts(
        dataset.get("artifacts"), manifest_dir=dataset_path.parent
    )
    accepted_examples = sum(categories.values())
    if dataset.get("accepted_examples") != accepted_examples:
        raise ValueError("SFT accepted_examples does not equal the category total")
    evaluation: dict[str, Any] | None = None
    if validation_manifest_path is not None:
        evaluation_path = Path(validation_manifest_path).resolve()
        raw_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if not isinstance(raw_evaluation, Mapping):
            raise ValueError("SFT validation manifest must be a JSON object")
        if raw_evaluation.get("quality_gate_passed") is not True:
            raise ValueError("SFT validation quality gate has not passed")
        if raw_evaluation.get("licenses_audited") is not True:
            raise ValueError("SFT validation licenses have not been audited")
        if raw_evaluation.get("unregistered_local_pilot") is True:
            raise PermissionError(
                "unregistered local SFT validation cannot enter a canonical lineage"
            )
        if raw_evaluation.get("split") != "validation":
            raise ValueError("SFT evaluation binding must use immutable validation split")
        evaluation_counts = _nonnegative_counts(
            raw_evaluation.get("category_counts"), name="SFT validation manifest"
        )
        evaluation_artifacts = _verified_dataset_artifacts(
            raw_evaluation.get("artifacts"), manifest_dir=evaluation_path.parent
        )
        evaluation_accepted = sum(evaluation_counts.values())
        if raw_evaluation.get("accepted_examples") != evaluation_accepted:
            raise ValueError("SFT validation accepted_examples does not equal category total")
        evaluation = {
            "manifest_path": str(evaluation_path),
            "manifest_sha256": _sha256_file(evaluation_path),
            "accepted_examples": evaluation_accepted,
            "category_counts": evaluation_counts,
            "artifacts": evaluation_artifacts,
            "quality_gate_passed": True,
            "licenses_audited": True,
            "source_receipt_sha256": _validated_hash(
                raw_evaluation.get("source_receipt_sha256"), "SFT validation source receipt hash"
            ),
            "split": "validation",
        }
        if evaluation["source_receipt_sha256"] != receipt_hash:
            raise ValueError("SFT train and validation splits must use the same source receipt")
    body: dict[str, Any] = {
        "schema": SFT_LINEAGE_SCHEMA,
        "lineage_id": lineage_id.strip(),
        "stage": "sft",
        "parent": {
            "base_checkpoint_path": str(checkpoint_path),
            "base_checkpoint_sha256": _sha256_file(checkpoint_path),
        },
        "tokenizer": {
            "contract": "v4-32768",
            "vocabulary_size": 32_768,
            "path": str(tokenizer),
            "sha256": _sha256_file(tokenizer),
        },
        "dataset": {
            "manifest_path": str(dataset_path),
            "manifest_sha256": _sha256_file(dataset_path),
            "accepted_examples": accepted_examples,
            "category_counts": categories,
            "artifacts": artifacts,
            "quality_gate_passed": True,
            "licenses_audited": True,
            "source_receipt_sha256": receipt_hash,
            "split": "train",
        },
        "evaluation": evaluation,
        "source_commit": source_commit.strip(),
        "optimizer_restart_required": True,
    }
    sealed = _seal(body, signing_key=signing_key, key_id=key_id)
    _write_immutable_json(output_path, sealed)
    return sealed


def verify_sft_lineage_manifest(
    path: str | Path,
    *,
    signing_key: str,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Verify a signed SFT lineage, optionally at a portable artifact location.

    The signed manifest records the paths used when its owner created it.  A
    Colab worker receives copies of those same, hash-bound artifacts at
    different absolute paths, so it may supply the three explicit local paths
    below.  Relocation never weakens the contract: the manifest signature and
    every declared digest are still verified before training can start.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("SFT lineage manifest must be a JSON object")
    body = _verify_seal(
        payload, signing_key=signing_key, expected_schema=SFT_LINEAGE_SCHEMA
    )
    parent = body.get("parent")
    tokenizer = body.get("tokenizer")
    dataset = body.get("dataset")
    if not all(isinstance(item, Mapping) for item in (parent, tokenizer, dataset)):
        raise ValueError("SFT lineage is missing parent, tokenizer, or dataset binding")
    assert isinstance(parent, Mapping)
    assert isinstance(tokenizer, Mapping)
    assert isinstance(dataset, Mapping)
    if tokenizer.get("contract") != "v4-32768" or tokenizer.get("vocabulary_size") != 32_768:
        raise ValueError("SFT lineage is not bound to the operational V4 tokenizer")
    _positive_counts(dataset.get("category_counts"))
    overrides = dict(artifact_paths or {})
    allowed_overrides = {
        "base_checkpoint_path",
        "tokenizer_path",
        "dataset_manifest_path",
        "validation_manifest_path",
    }
    unknown = sorted(set(overrides) - allowed_overrides)
    if unknown:
        raise ValueError(f"unsupported SFT artifact path overrides: {unknown}")
    bindings = (
        (
            overrides.get("base_checkpoint_path", parent.get("base_checkpoint_path")),
            parent.get("base_checkpoint_sha256"),
        ),
        (
            overrides.get("tokenizer_path", tokenizer.get("path")),
            tokenizer.get("sha256"),
        ),
        (
            overrides.get("dataset_manifest_path", dataset.get("manifest_path")),
            dataset.get("manifest_sha256"),
        ),
    )
    resolved_dataset_manifest = Path(str(bindings[2][0]))
    for raw_path, raw_hash in bindings:
        artifact = Path(str(raw_path))
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        expected_hash = _validated_hash(raw_hash, "bound artifact hash")
        if not hmac.compare_digest(expected_hash, _sha256_file(artifact)):
            raise ValueError(f"bound post-training artifact changed: {artifact}")
    _verified_dataset_artifacts(
        dataset.get("artifacts"), manifest_dir=resolved_dataset_manifest.parent
    )
    evaluation = body.get("evaluation")
    if evaluation is not None:
        if not isinstance(evaluation, Mapping) or evaluation.get("split") != "validation":
            raise ValueError("SFT lineage evaluation binding is invalid")
        validation_path = Path(
            str(overrides.get("validation_manifest_path", evaluation.get("manifest_path", "")))
        )
        if not validation_path.is_file():
            raise FileNotFoundError(validation_path)
        expected_validation_hash = _validated_hash(
            evaluation.get("manifest_sha256"), "validation manifest hash"
        )
        if not hmac.compare_digest(expected_validation_hash, _sha256_file(validation_path)):
            raise ValueError(f"bound SFT validation manifest changed: {validation_path}")
        _nonnegative_counts(evaluation.get("category_counts"), name="SFT validation binding")
        _verified_dataset_artifacts(
            evaluation.get("artifacts"), manifest_dir=validation_path.parent
        )
    return dict(payload)


def audit_verifiable_outcomes(
    method: str, records: Sequence[Mapping[str, object]]
) -> dict[str, Any]:
    """Gate RLVR/STaR data on reproducible, binary verifier outcomes."""

    normalized_method = method.strip().lower()
    if normalized_method not in {"rlvr", "star"}:
        raise ValueError("outcome audit method must be rlvr or star")
    issues: list[str] = []
    identities: set[tuple[str, str]] = set()
    included = 0
    passed_outcomes = 0
    for index, record in enumerate(records):
        task_id = str(record.get("task_id", "")).strip()
        domain = str(record.get("domain", "")).strip().lower()
        candidate_hash = str(record.get("candidate_sha256", "")).strip().lower()
        verifier_id = str(record.get("verifier_id", "")).strip()
        verifier_version = str(record.get("verifier_version", "")).strip()
        outcome = str(record.get("outcome", "")).strip().lower()
        reward = record.get("reward")
        include = record.get("included") is True
        prefix = f"record[{index}]"
        if not task_id or not verifier_id or not verifier_version:
            issues.append(f"{prefix}: missing task or verifier identity")
        if domain not in VERIFIABLE_DOMAINS:
            issues.append(f"{prefix}: domain {domain!r} is not mechanically verifiable")
        try:
            candidate_hash = _validated_hash(candidate_hash, "candidate hash")
            _validated_hash(record.get("verifier_evidence_sha256"), "verifier evidence hash")
        except ValueError as error:
            issues.append(f"{prefix}: {error}")
        identity = (task_id, candidate_hash)
        if identity in identities:
            issues.append(f"{prefix}: repeated task/candidate outcome")
        identities.add(identity)
        if outcome not in {"pass", "fail"}:
            issues.append(f"{prefix}: outcome must be pass or fail")
        expected_reward = 1.0 if outcome == "pass" else 0.0
        if (
            isinstance(reward, bool)
            or not isinstance(reward, int | float)
            or not math.isfinite(float(reward))
            or float(reward) != expected_reward
        ):
            issues.append(f"{prefix}: reward is not derived from the binary outcome")
        if include:
            included += 1
            if outcome == "pass":
                passed_outcomes += 1
            if normalized_method == "star" and outcome != "pass":
                issues.append(f"{prefix}: STaR may include only verifier-passing chains")
    if not records:
        issues.append("no verifier outcomes were supplied")
    if included == 0:
        issues.append("no verifier outcome is eligible for training")
    body = {
        "schema": "anra-verifiable-outcome-audit/v1",
        "method": normalized_method,
        "record_count": len(records),
        "included_count": included,
        "included_pass_count": passed_outcomes,
        "issues": issues,
        "passed": not issues,
    }
    return {**body, "report_sha256": _sha256(body)}


def audit_preference_pairs(
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Reject unaudited, ambiguous, duplicated, or heuristic DPO pairs."""

    issues: list[str] = []
    pair_ids: set[str] = set()
    for index, pair in enumerate(pairs):
        prefix = f"pair[{index}]"
        pair_id = str(pair.get("pair_id", "")).strip()
        source_kind = str(pair.get("source_kind", "")).strip().lower()
        source_id = str(pair.get("source_id", "")).strip()
        auditor_id = str(pair.get("auditor_id", "")).strip()
        decision = str(pair.get("audit_decision", "")).strip().lower()
        if not pair_id or pair_id in pair_ids:
            issues.append(f"{prefix}: pair_id is missing or duplicated")
        pair_ids.add(pair_id)
        if source_kind not in AUDITED_PREFERENCE_SOURCES:
            issues.append(f"{prefix}: preference source {source_kind!r} is not auditable")
        if not source_id or not auditor_id or decision != "approved":
            issues.append(f"{prefix}: explicit source, auditor, and approval are required")
        hashes: dict[str, str] = {}
        for name in ("prompt_sha256", "chosen_sha256", "rejected_sha256"):
            try:
                hashes[name] = _validated_hash(pair.get(name), name)
            except ValueError as error:
                issues.append(f"{prefix}: {error}")
        try:
            _validated_hash(pair.get("audit_evidence_sha256"), "audit evidence hash")
        except ValueError as error:
            issues.append(f"{prefix}: {error}")
        if hashes.get("chosen_sha256") == hashes.get("rejected_sha256"):
            issues.append(f"{prefix}: chosen and rejected responses are identical")
    if not pairs:
        issues.append("no preference pairs were supplied")
    body = {
        "schema": "anra-preference-pair-audit/v1",
        "pair_count": len(pairs),
        "approved_count": len(pairs) if not issues else 0,
        "issues": issues,
        "passed": not issues,
    }
    return {**body, "report_sha256": _sha256(body)}


def require_gate_report(
    gate_report: Mapping[str, Any], *, expected_stage: str
) -> dict[str, Any]:
    """Validate an in-memory RLVR, STaR, or DPO audit before numerical work."""

    stage = expected_stage.strip().lower()
    expected_schema = {
        "rlvr": "anra-verifiable-outcome-audit/v1",
        "star": "anra-verifiable-outcome-audit/v1",
        "dpo": "anra-preference-pair-audit/v1",
    }
    if stage not in expected_schema:
        raise ValueError("post-training gate stage must be rlvr, star, or dpo")
    if gate_report.get("schema") != expected_schema[stage]:
        raise ValueError("post-training gate report schema does not match its stage")
    if gate_report.get("method", stage) != stage:
        raise ValueError("verifiable outcome report method does not match its stage")
    if gate_report.get("passed") is not True:
        raise PermissionError("post-training evidence gate has not passed")
    report_hash = _validated_hash(gate_report.get("report_sha256"), "gate report hash")
    report_body = {
        name: value for name, value in gate_report.items() if name != "report_sha256"
    }
    if not hmac.compare_digest(report_hash, _sha256(report_body)):
        raise ValueError("post-training gate report hash mismatch")
    return dict(gate_report)


def write_posttraining_gate_manifest(
    output_path: str | Path,
    *,
    stage: str,
    parent_manifest_sha256: str,
    source_commit: str,
    gate_report: Mapping[str, Any],
    signing_key: str,
    key_id: str = "owner",
) -> dict[str, Any]:
    """Sign the evidence gate that authorizes one separate lineage stage."""

    normalized_stage = stage.strip().lower()
    validated_report = require_gate_report(gate_report, expected_stage=normalized_stage)
    body: dict[str, Any] = {
        "schema": POSTTRAINING_GATE_SCHEMA,
        "stage": normalized_stage,
        "parent_manifest_sha256": _validated_hash(
            parent_manifest_sha256, "parent manifest hash"
        ),
        "source_commit": source_commit.strip(),
        "gate_report": validated_report,
        "optimizer_restart_required": True,
    }
    if not body["source_commit"]:
        raise ValueError("source_commit is required")
    sealed = _seal(body, signing_key=signing_key, key_id=key_id)
    _write_immutable_json(output_path, sealed)
    return sealed


def verify_posttraining_gate_manifest(
    path: str | Path, *, signing_key: str, expected_stage: str
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("post-training gate manifest must be a JSON object")
    body = _verify_seal(
        payload, signing_key=signing_key, expected_schema=POSTTRAINING_GATE_SCHEMA
    )
    if body.get("stage") != expected_stage.strip().lower():
        raise ValueError("post-training gate stage mismatch")
    gate_report = body.get("gate_report")
    if not isinstance(gate_report, Mapping):
        raise PermissionError("post-training gate report has not passed")
    require_gate_report(gate_report, expected_stage=expected_stage)
    return dict(payload)
