"""External CUDA execution receipts for X1-REAL-1.

The deterministic public-task renderer is
``anra-x1-public-task-renderer/v1``. It renders context, query, and visible
candidates into the strict runtime's public ``ibq_v2`` task shape, accepts only
an absent format or the explicit ``text``, ``code``, ``json``, ``prose``, and
``table`` answer-marker contracts,
and never renders observation features, truth, or evaluator metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from x_factor import x1_real_1 as coordinator
from x_factor.ibq_v2 import apply_probe


AUTHORIZATION_PHRASE = "I AUTHORIZE X1-REAL-1 EXTERNAL CUDA COMPUTE"
EXTERNAL_AUTHORIZATION_PHRASE = AUTHORIZATION_PHRASE
RENDERER_ID = "anra-x1-public-task-renderer/v1"
CELL_RECEIPT_SCHEMA = "anra-x1-real-1-external-cell/v1"
RUN_MANIFEST_SCHEMA = "anra-x1-real-1-external-run/v1"
CLI_ERROR_SCHEMA = "anra-x1-real-1-external-runner-error/v1"

MAX_NEW_TOKENS_CEILING = 2_048
DEFAULT_MAX_NEW_TOKENS = 32
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
CUDA_DEVICE_RE = re.compile(r"^cuda(?::[0-9]+)?$")
ANSWER_MARKER_FORMATS = {
    "text": "Answer:",
    "code": "Answer:",
    "json": "Answer:",
    "prose": "Answer:",
    "table": "Answer:",
}
FEATURE_FIELDS = (
    "confidence",
    "entropy",
    "margin",
    "output_len",
    "distinct_ratio",
    "prompt_tokens",
)
PUBLIC_TASK_FIELDS = frozenset({
    "task_id", "context", "query", "visible_candidates", "format", "features",
})
FORBIDDEN_RECEIPT_KEYS = frozenset({
    "gold",
    "gold_answer",
    "truth",
    "target",
    "label",
    "correct",
    "correctness",
    "correctness_label",
    "is_correct",
    "outcome",
    "outcomes",
    "effect",
    "evaluator",
    "evaluation",
    "verifier",
    "oracle",
    "prediction",
})
CELL_RECEIPT_FIELDS = frozenset({
    "schema",
    "renderer_id",
    "task_id",
    "intervention_id",
    "transformed_prompt",
    "transformed_prompt_sha256",
    "raw_output",
    "raw_output_sha256",
    "output_token_ids",
    "features",
    "runtime_identity",
    "checkpoint_identity",
    "tokenizer_identity",
    "architecture_identity",
    "execution_profile",
    "generation_policy",
    "cell_sha256",
})
PREFLIGHT_REQUIRED_FIELDS = frozenset({
    "schema",
    "status",
    "protocol_sha256",
    "cohort_id",
    "external_compute_authorized",
    "runtime",
    "checks",
    "check_summary",
    "blockers",
    "next_actions",
    "input_hashes",
    "execution_budget",
    "power_advisories",
    "dry_run",
    "preflight_sha256",
})
CORE_CONFIG_FIELDS = frozenset({
    "architecture_version",
    "vocab_size",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "d_model",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "d_ff",
    "block_size",
    "rms_norm_eps",
    "dropout",
    "rope_base",
    "base_seq_len",
    "target_seq_len",
    "sliding_window",
    "full_attention_every",
    "qk_norm",
    "use_mtp",
    "use_moe",
    "initialization_scheme",
})


class ExternalRunnerError(RuntimeError):
    pass


class AuthorizationError(ExternalRunnerError):
    pass


class ReceiptExistsError(ExternalRunnerError):
    pass


class ReceiptIntegrityError(ExternalRunnerError):
    pass


class CLIUsageError(ExternalRunnerError):
    pass


@dataclass(frozen=True, slots=True)
class StrictRuntime:
    CoreConfig: type
    CoreExecutor: type
    torch: Any


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExternalRunnerError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ExternalRunnerError(f"nonstandard JSON constant is forbidden: {value}")


def _plain_json(value: Any, name: str = "value") -> Any:
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ExternalRunnerError(f"{name} contains invalid Unicode text") from exc
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExternalRunnerError(f"{name} contains a non-finite JSON number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExternalRunnerError(f"{name} contains a non-string object key")
            result[key] = _plain_json(item, f"{name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain_json(item, f"{name}[{index}]") for index, item in enumerate(value)]
    raise ExternalRunnerError(f"{name} contains unsupported value type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        plain = _plain_json(value)
        return json.dumps(
            plain,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except ExternalRunnerError:
        raise
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ExternalRunnerError(f"value is not canonical JSON: {exc}") from exc


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def loads_json_strict(text: str) -> Any:
    """Parse one strict JSON value with duplicate keys and constants rejected."""

    if not isinstance(text, str):
        raise ExternalRunnerError("strict JSON input must be text")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
        return _plain_json(value)
    except ExternalRunnerError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ExternalRunnerError(f"invalid strict JSON: {exc}") from exc


def load_json_strict(path: str | Path) -> Any:
    """Load strict UTF-8 JSON from a file without accepting duplicate keys."""

    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ExternalRunnerError(f"unable to read strict JSON artifact {target}: {exc}") from exc
    try:
        return loads_json_strict(text)
    except ExternalRunnerError as exc:
        raise ExternalRunnerError(f"invalid strict JSON artifact {target}: {exc}") from exc


def load_strict_json(path: str | Path) -> Any:
    return load_json_strict(path)


def safe_output_sha256(output: str) -> str:
    """Hash exact model text as UTF-8 without Unicode normalization or replacement."""

    if not isinstance(output, str):
        raise ExternalRunnerError("model output must be text")
    try:
        encoded = output.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ExternalRunnerError("model output is not valid Unicode text") from exc
    return hashlib.sha256(encoded).hexdigest()


def safe_output_hash(output: str) -> str:
    return safe_output_sha256(output)


def _validate_public_task_shape(task: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(task, Mapping):
        raise ExternalRunnerError("public task must be an object")
    unknown = set(task) - PUBLIC_TASK_FIELDS
    missing = {"task_id", "context", "query", "visible_candidates"} - set(task)
    if missing or unknown:
        raise ExternalRunnerError(
            f"public task fields are invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ExternalRunnerError("public task_id must be non-empty text")
    for field in ("context", "query"):
        if not isinstance(task.get(field), str) or not task[field]:
            raise ExternalRunnerError(f"public task {field} must be non-empty text")
    candidates = task.get("visible_candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(candidate, str) or not candidate for candidate in candidates)
    ):
        raise ExternalRunnerError("visible_candidates must be a non-empty list of non-empty strings")
    if len(set(candidates)) != len(candidates):
        raise ExternalRunnerError("visible_candidates must not contain duplicates")
    if "format" in task and task.get("format") is not None and task.get("format") not in ANSWER_MARKER_FORMATS:
        raise ExternalRunnerError(
            f"unsupported answer-marker format {task.get('format')!r}; "
            f"supported={sorted(ANSWER_MARKER_FORMATS)}"
        )
    features = task.get("features", {})
    if not isinstance(features, Mapping):
        raise ExternalRunnerError("public task features must be an object")
    unknown_features = set(features) - set(FEATURE_FIELDS)
    if unknown_features:
        raise ExternalRunnerError(f"unsupported public task features: {sorted(unknown_features)}")
    for field, value in features.items():
        if field in {"output_len", "prompt_tokens"}:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ExternalRunnerError(f"invalid public task count feature: {field}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ExternalRunnerError(f"invalid public task numeric feature: {field}")
    normalized = dict(task)
    canonical_json_bytes(normalized)
    return normalized


def render_public_task(task: Mapping[str, Any], intervention_id: str) -> str:
    """Render renderer v1 public fields and dispatch exactly through ``ibq_v2.apply_probe``."""

    normalized = _validate_public_task_shape(task)
    if not isinstance(intervention_id, str) or not intervention_id:
        raise ExternalRunnerError("intervention_id must be non-empty text")
    answer_marker = ANSWER_MARKER_FORMATS.get(normalized.get("format") or "text")
    if answer_marker is None:
        raise ExternalRunnerError("unsupported answer-marker format")
    candidate_json = json.dumps(
        normalized["visible_candidates"],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    probe_task = {
        "block": f"{normalized['context']}\nVISIBLE CANDIDATES (JSON):\n{candidate_json}",
        "query": normalized["query"],
        "answer_marker": answer_marker,
    }
    try:
        transformed = apply_probe(intervention_id, probe_task)
    except Exception as exc:
        raise ExternalRunnerError(
            f"frozen intervention transformation failed: {intervention_id}: {exc}"
        ) from exc
    if not isinstance(transformed, str) or not transformed:
        raise ExternalRunnerError(f"intervention returned no prompt text: {intervention_id}")
    if not transformed.rstrip().endswith(answer_marker):
        raise ExternalRunnerError(
            f"frozen intervention does not preserve the supported answer marker: {intervention_id}"
        )
    return transformed


def _log_softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ExternalRunnerError("logit vector cannot be empty")
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    if not math.isfinite(total) or total <= 0.0:
        raise ExternalRunnerError("logit softmax normalizer is invalid")
    normalizer = math.log(total)
    return [value - maximum - normalizer for value in logits]


def summarize_generation_features(
    step_logits: Sequence[Sequence[float]] | Sequence[float],
    output_token_ids: Sequence[int],
    prompt_token_count: int,
) -> dict[str, float | int]:
    """Summarize frozen generation features from first-step and selected-token logits.

    Confidence is mean natural-log selected-token probability, entropy is first-step
    Shannon entropy in nats, and margin is the top-two probability difference.
    """

    if not isinstance(prompt_token_count, int) or isinstance(prompt_token_count, bool) or prompt_token_count < 1:
        raise ExternalRunnerError("prompt_token_count must be a positive integer")
    if not isinstance(output_token_ids, Sequence) or isinstance(output_token_ids, (str, bytes)):
        raise ExternalRunnerError("output_token_ids must be a sequence")
    token_ids: list[int] = []
    for value in output_token_ids:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ExternalRunnerError("output token IDs must be non-negative integers")
        token_ids.append(value)
    if not isinstance(step_logits, Sequence) or isinstance(step_logits, (str, bytes)):
        raise ExternalRunnerError("step_logits must be a sequence")
    rows: list[Sequence[float]]
    if step_logits and isinstance(step_logits[0], (int, float)):
        rows = [step_logits]
    else:
        rows = list(step_logits)
    if not rows:
        raise ExternalRunnerError("at least one generation-step logit vector is required")
    if len(rows) < len(token_ids):
        raise ExternalRunnerError("each output token requires its selecting logit vector")
    normalized_rows: list[list[float]] = []
    for row in rows:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or not row:
            raise ExternalRunnerError("each logit vector must be a non-empty sequence")
        normalized: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ExternalRunnerError("logits must be finite real numbers")
            normalized.append(float(value))
        normalized_rows.append(normalized)
    first_log_probs = _log_softmax(normalized_rows[0])
    probabilities = [math.exp(value) for value in first_log_probs]
    ordered = sorted(probabilities, reverse=True)
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else 0.0
    selected_log_probs: list[float] = []
    for token_id, row in zip(token_ids, normalized_rows, strict=False):
        if token_id >= len(row):
            raise ExternalRunnerError("output token ID is outside its logit vocabulary")
        selected_log_probs.append(_log_softmax(row)[token_id])
    confidence = sum(selected_log_probs) / len(selected_log_probs) if selected_log_probs else 0.0
    distinct_ratio = len(set(token_ids)) / len(token_ids) if token_ids else 0.0
    features: dict[str, float | int] = {
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "output_len": len(token_ids),
        "distinct_ratio": distinct_ratio,
        "prompt_tokens": prompt_token_count,
    }
    canonical_json_bytes(features)
    return features


def _sha_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise ExternalRunnerError(f"unable to hash file {path}: {exc}") from exc
    return digest.hexdigest()


def _load_object(path: str | Path, name: str) -> dict[str, Any]:
    value = load_json_strict(path)
    if not isinstance(value, Mapping):
        raise ExternalRunnerError(f"{name} must be a JSON object")
    return dict(value)


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX64_RE.fullmatch(value):
        raise ExternalRunnerError(f"{name} must be a lowercase SHA256 digest")
    return value


def _require_file(path: str | Path, name: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ExternalRunnerError(f"{name} file is missing: {candidate}")
    return candidate


def _require_file_hash(path: Path, expected: Any, name: str) -> str:
    expected_hash = _require_sha256(expected, f"{name}.sha256")
    actual_hash = _sha_file(path)
    if actual_hash != expected_hash:
        raise ExternalRunnerError(f"{name} file hash does not match the subject manifest")
    return actual_hash


def _resolve_path(value: Any, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExternalRunnerError(f"{name} path is missing")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _normalize_config_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    payload: Mapping[str, Any] = document
    if set(document) == {"model_config"} and isinstance(document.get("model_config"), Mapping):
        payload = document["model_config"]
    if set(payload) != CORE_CONFIG_FIELDS:
        missing = sorted(CORE_CONFIG_FIELDS - set(payload))
        unknown = sorted(set(payload) - CORE_CONFIG_FIELDS)
        raise ExternalRunnerError(
            f"config JSON fields are invalid; missing={missing}, unknown={unknown}"
        )
    return _plain_json(payload, "model_config")


def _strict_runtime_source_identity() -> dict[str, Any]:
    runtime_root = Path(__file__).resolve().parent / "_runtime" / "anra_core"
    if not runtime_root.is_dir():
        raise ExternalRunnerError("strict An-Ra runtime source directory is missing")
    files: list[dict[str, str]] = []
    for path in sorted(runtime_root.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(runtime_root.parent.parent).as_posix()
        files.append({"path": relative, "sha256": _sha_file(path)})
    if not files:
        raise ExternalRunnerError("strict An-Ra runtime source closure is empty")
    return {
        "package": "anra_core",
        "source_closure_sha256": canonical_json_sha256(files),
        "files": files,
    }


def _validate_preflight_receipt(receipt: Mapping[str, Any], protocol_sha256: str) -> None:
    if set(receipt) != PREFLIGHT_REQUIRED_FIELDS:
        missing = sorted(PREFLIGHT_REQUIRED_FIELDS - set(receipt))
        unknown = sorted(set(receipt) - PREFLIGHT_REQUIRED_FIELDS)
        raise ExternalRunnerError(
            f"preflight receipt fields are invalid; missing={missing}, unknown={unknown}"
        )
    if receipt.get("schema") != coordinator.PREFLIGHT_SCHEMA:
        raise ExternalRunnerError("preflight receipt schema mismatch")
    expected_hash = _require_sha256(receipt.get("preflight_sha256"), "preflight_sha256")
    body = {key: value for key, value in receipt.items() if key != "preflight_sha256"}
    if canonical_json_sha256(body) != expected_hash:
        raise ExternalRunnerError("preflight receipt hash mismatch")
    if receipt.get("protocol_sha256") != protocol_sha256:
        raise ExternalRunnerError("preflight protocol binding mismatch")
    if receipt.get("status") != "READY_FOR_EXTERNAL_PREDICTION":
        raise ExternalRunnerError("supplied preflight is not READY_FOR_EXTERNAL_PREDICTION")
    blockers = receipt.get("blockers")
    if not isinstance(blockers, list) or blockers:
        raise ExternalRunnerError("supplied preflight has blockers")
    checks = receipt.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ExternalRunnerError("supplied preflight has no checks")
    if any(not isinstance(check, Mapping) or check.get("status") in {"FAIL", "BLOCKED"} for check in checks):
        raise ExternalRunnerError("supplied preflight contains failed or blocked checks")
    summary = receipt.get("check_summary")
    if (
        not isinstance(summary, Mapping)
        or not isinstance(summary.get("fail"), int)
        or not isinstance(summary.get("blocked"), int)
        or summary.get("fail") != 0
        or summary.get("blocked") != 0
    ):
        raise ExternalRunnerError("supplied preflight check summary is not clean")
    if receipt.get("external_compute_authorized") is not False:
        raise ExternalRunnerError("preflight receipt must not carry external compute authorization")
    runtime = receipt.get("runtime")
    dry_run = receipt.get("dry_run")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("local_model_execution") is not False
        or runtime.get("local_training") is not False
        or not isinstance(dry_run, Mapping)
        or dry_run.get("model_execution") is not False
        or dry_run.get("training_execution") is not False
    ):
        raise ExternalRunnerError("preflight execution boundary is invalid")


def _cohort_spec(protocol: Mapping[str, Any], cohort_id: str) -> Mapping[str, Any]:
    matches = [
        value
        for value in protocol.get("cohorts", {}).values()
        if isinstance(value, Mapping) and value.get("cohort_id") == cohort_id
    ]
    if len(matches) != 1:
        raise ExternalRunnerError("prediction cohort is not uniquely frozen")
    return matches[0]


def _expected_preflight_input_hashes(
    protocol: Mapping[str, Any],
    subject: Mapping[str, Any],
    basis: Mapping[str, Any],
    release: Mapping[str, Any],
    split: Mapping[str, Any],
    basis_split: Mapping[str, Any],
    development_split: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    prediction: Mapping[str, Any],
    prediction_commit: Mapping[str, Any],
) -> dict[str, str]:
    predictor = prediction.get("predictor")
    baseline = prediction.get("baseline_commitment")
    predictions = prediction.get("predictions")
    if not isinstance(predictor, Mapping) or not isinstance(baseline, Mapping) or not isinstance(predictions, list):
        raise ExternalRunnerError("prediction receipt lacks predictor, baseline, or prediction fields")
    values = {
        "protocol": protocol,
        "subject_manifest": subject,
        "basis_qualification": basis,
        "release_manifest": release,
        "primary_split_manifest": split,
        "basis_split_manifest": basis_split,
        "development_split_manifest": development_split,
        "public_tasks": tasks,
        "predictor_identity": predictor,
        "baseline_commitment": baseline,
        "predictor_predictions": predictions,
        "prediction_receipt": prediction,
        "prediction_commit": prediction_commit,
    }
    return {name: canonical_json_sha256(value) for name, value in values.items()}


def validate_execution_gates(
    *,
    protocol_path: str | Path,
    preflight_path: str | Path,
    subject_path: str | Path,
    registry_path: str | Path,
    release_manifest_path: str | Path,
    split_path: str | Path,
    public_tasks_path: str | Path,
    basis_qualification_path: str | Path,
    basis_split_path: str | Path,
    development_split_path: str | Path,
    prediction_path: str | Path,
    prediction_commit_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    tokenizer_path: str | Path,
    source_root: str | Path,
    release_root: str | Path,
    checkpoint_root: str | Path,
) -> dict[str, Any]:
    """Validate every custody, artifact, identity, and preflight gate before Torch import."""

    source = Path(source_root).expanduser().resolve()
    release_base = Path(release_root).expanduser().resolve()
    checkpoint_base = Path(checkpoint_root).expanduser().resolve()
    paths = {
        "protocol": _require_file(protocol_path, "protocol"),
        "preflight": _require_file(preflight_path, "preflight"),
        "subject": _require_file(subject_path, "subject manifest"),
        "registry": _require_file(registry_path, "registry"),
        "release_manifest": _require_file(release_manifest_path, "release manifest"),
        "split": _require_file(split_path, "split"),
        "public_tasks": _require_file(public_tasks_path, "public tasks"),
        "basis_qualification": _require_file(basis_qualification_path, "basis qualification"),
        "basis_split": _require_file(basis_split_path, "basis split"),
        "development_split": _require_file(development_split_path, "development split"),
        "prediction": _require_file(prediction_path, "prediction receipt"),
        "prediction_commit": _require_file(prediction_commit_path, "prediction phase commit"),
        "checkpoint": _require_file(checkpoint_path, "checkpoint"),
        "config": _require_file(config_path, "config"),
        "tokenizer": _require_file(tokenizer_path, "tokenizer"),
    }
    documents = {
        "protocol": _load_object(paths["protocol"], "protocol"),
        "preflight": _load_object(paths["preflight"], "preflight"),
        "subject": _load_object(paths["subject"], "subject manifest"),
        "registry": _load_object(paths["registry"], "registry"),
        "release_manifest": _load_object(paths["release_manifest"], "release manifest"),
        "split": _load_object(paths["split"], "split"),
        "public_tasks_document": load_json_strict(paths["public_tasks"]),
        "basis": _load_object(paths["basis_qualification"], "basis qualification"),
        "basis_split": _load_object(paths["basis_split"], "basis split"),
        "development_split": _load_object(paths["development_split"], "development split"),
        "prediction": _load_object(paths["prediction"], "prediction receipt"),
        "prediction_commit": _load_object(paths["prediction_commit"], "prediction phase commit"),
        "config": _load_object(paths["config"], "config"),
    }
    protocol = documents["protocol"]
    preflight = documents["preflight"]
    protocol_verdict = coordinator.validate_frozen_protocol(
        protocol,
        repo_root=source,
        check_source_closure=True,
    )
    if not protocol_verdict["valid"]:
        raise ExternalRunnerError("frozen protocol/source closure is invalid: " + "; ".join(protocol_verdict["errors"]))
    protocol_sha256 = str(protocol["identity"]["protocol_sha256"])
    _validate_preflight_receipt(preflight, protocol_sha256)
    subject = documents["subject"]
    registry = documents["registry"]
    if registry.get("schema") != "anra-checkpoint-registry/v2":
        raise ExternalRunnerError("checkpoint registry schema mismatch")
    subject_verdict = coordinator.validate_subject_manifest(
        subject,
        registry_path=paths["registry"],
        checkpoint_root=checkpoint_base,
        verify_checkpoint_file=True,
    )
    if not subject_verdict["valid"]:
        raise ExternalRunnerError("subject manifest is invalid: " + "; ".join(subject_verdict["errors"]))
    subject_checkpoint = _resolve_path(subject.get("checkpoint_path"), checkpoint_base, "subject.checkpoint_path")
    if subject_checkpoint != paths["checkpoint"]:
        raise ExternalRunnerError("explicit checkpoint path does not match the subject manifest")
    if subject_verdict.get("checkpoint_file_verified") is not True:
        raise ExternalRunnerError("subject checkpoint file was not verified")
    _require_file_hash(paths["config"], subject.get("model_config_sha256"), "config")
    _require_file_hash(paths["tokenizer"], subject.get("tokenizer_artifact_sha256"), "tokenizer")
    tokenizer_metadata = paths["tokenizer"].with_suffix(paths["tokenizer"].suffix + ".meta.json")
    tokenizer_metadata = _require_file(tokenizer_metadata, "tokenizer metadata")
    load_json_strict(paths["tokenizer"])
    load_json_strict(tokenizer_metadata)
    release = documents["release_manifest"]
    release_verdict = coordinator.validate_release_manifest(
        release,
        protocol,
        repo_root=release_base,
    )
    if not release_verdict["valid"]:
        raise ExternalRunnerError("release manifest is invalid: " + "; ".join(release_verdict["errors"]))
    basis = documents["basis"]
    basis_split = documents["basis_split"]
    basis_verdict = coordinator.validate_basis_qualification(
        basis,
        expected_protocol_sha256=protocol_sha256,
        expected_subject_manifest_sha256=subject_verdict["subject_manifest_sha256"],
        expected_registry_sha256=protocol.get("intervention_registry_sha256"),
        registry=protocol.get("interventions"),
        basis_split_manifest=basis_split,
    )
    if not basis_verdict["valid"]:
        raise ExternalRunnerError("basis qualification is invalid: " + "; ".join(basis_verdict["errors"]))
    development_split = documents["development_split"]
    prediction = documents["prediction"]
    cohort_id = prediction.get("cohort_id")
    if not isinstance(cohort_id, str) or not cohort_id:
        raise ExternalRunnerError("prediction receipt cohort_id is missing")
    cohort_spec = _cohort_spec(protocol, cohort_id)
    expected_role = "REPLICATION" if cohort_spec.get("role") == "REPLICATION" else "EVALUATION"
    split = documents["split"]
    split_verdict = coordinator.validate_split_manifest(
        split,
        expected_protocol_sha256=protocol_sha256,
        expected_cohort_id=cohort_id,
        expected_cohort_role=expected_role,
        expected_seed=cohort_spec.get("seed"),
        expected_cohort_plan_sha256=cohort_spec.get("cohort_plan_sha256"),
    )
    if not split_verdict["valid"]:
        raise ExternalRunnerError("evaluation split is invalid: " + "; ".join(split_verdict["errors"]))
    development_spec = protocol.get("cohorts", {}).get("development", {})
    development_verdict = coordinator.validate_split_manifest(
        development_split,
        expected_protocol_sha256=protocol_sha256,
        expected_cohort_id=development_spec.get("cohort_id"),
        expected_cohort_role="DEVELOPMENT",
        expected_seed=development_spec.get("seed"),
        expected_cohort_plan_sha256=development_spec.get("cohort_plan_sha256"),
    )
    if not development_verdict["valid"]:
        raise ExternalRunnerError("development split is invalid: " + "; ".join(development_verdict["errors"]))
    basis_spec = protocol.get("cohorts", {}).get("basis_qualification", {})
    basis_split_verdict = coordinator.validate_split_manifest(
        basis_split,
        expected_protocol_sha256=protocol_sha256,
        expected_cohort_id=basis_spec.get("cohort_id"),
        expected_cohort_role="QUALIFICATION",
        expected_seed=basis_spec.get("seed"),
        expected_cohort_plan_sha256=basis_spec.get("cohort_plan_sha256"),
    )
    if not basis_split_verdict["valid"]:
        raise ExternalRunnerError("basis split is invalid: " + "; ".join(basis_split_verdict["errors"]))
    public_tasks_document = documents["public_tasks_document"]
    if isinstance(public_tasks_document, Mapping):
        if set(public_tasks_document) != {"tasks"}:
            raise ExternalRunnerError("public task object must contain only a tasks list")
        raw_tasks = public_tasks_document.get("tasks")
    else:
        raw_tasks = public_tasks_document
    if not isinstance(raw_tasks, list) or not raw_tasks or any(not isinstance(task, Mapping) for task in raw_tasks):
        raise ExternalRunnerError("public task artifact must contain a non-empty tasks list")
    task_verdict = coordinator.validate_public_task_bundle(raw_tasks, split_manifest=split)
    tasks = list(task_verdict["tasks"])
    prediction_verdict = coordinator.validate_prediction_receipt(
        prediction,
        protocol=protocol,
        subject_manifest=subject,
        public_tasks=tasks,
        split_manifest=split,
        registry_path=paths["registry"],
        checkpoint_root=checkpoint_base,
        release_manifest=release,
        release_root=release_base,
        basis_qualification=basis,
        basis_subject_manifest_sha256=subject_verdict["subject_manifest_sha256"],
        source_root=source,
        development_split_manifest=development_split,
        basis_split_manifest=basis_split,
    )
    if not prediction_verdict["valid"]:
        raise ExternalRunnerError("prediction receipt is invalid: " + "; ".join(prediction_verdict["errors"]))
    prediction_commit = documents["prediction_commit"]
    commit_verdict = coordinator.validate_phase_commit(
        prediction_commit,
        expected_phase="PREDICT_COMMITTED",
        expected_receipt_sha256=prediction.get("prediction_receipt_sha256"),
        expected_previous_commit_sha256=prediction.get("basis_phase_commit_sha256"),
        repo_root=release_base,
    )
    if not commit_verdict["valid"]:
        raise ExternalRunnerError("PREDICT_COMMITTED phase is invalid: " + "; ".join(commit_verdict["errors"]))
    committed_prediction = _resolve_path(
        prediction_commit.get("artifact_path"),
        release_base,
        "prediction_commit.artifact_path",
    )
    if committed_prediction != paths["prediction"]:
        raise ExternalRunnerError("PREDICT_COMMITTED phase does not bind the supplied prediction receipt")
    if preflight.get("cohort_id") != cohort_id:
        raise ExternalRunnerError("supplied preflight cohort mismatch")
    preflight_input_hashes = preflight.get("input_hashes")
    if not isinstance(preflight_input_hashes, Mapping):
        raise ExternalRunnerError("supplied preflight input hashes are missing")
    expected_input_hashes = _expected_preflight_input_hashes(
        protocol,
        subject,
        basis,
        release,
        split,
        basis_split,
        development_split,
        raw_tasks,
        prediction,
        prediction_commit,
    )
    mismatched = {
        name: {"expected": expected, "actual": preflight_input_hashes.get(name)}
        for name, expected in expected_input_hashes.items()
        if preflight_input_hashes.get(name) != expected
    }
    if mismatched:
        raise ExternalRunnerError(f"supplied preflight is not bound to the supplied artifacts: {mismatched}")
    supplied_dry_run = preflight.get("dry_run")
    if not isinstance(supplied_dry_run, Mapping) or supplied_dry_run.get("prediction_receipt_sha256") != prediction.get("prediction_receipt_sha256"):
        raise ExternalRunnerError("supplied preflight dry run is not bound to the prediction receipt")
    coordinator_preflight = coordinator.preflight_x1_real_1(
        protocol,
        subject_manifest=subject,
        basis_qualification=basis,
        release_manifest=release,
        registry_path=paths["registry"],
        checkpoint_root=checkpoint_base,
        release_root=release_base,
        source_root=source,
        primary_split_manifest=split,
        basis_split_manifest=basis_split,
        development_split_manifest=development_split,
        public_tasks=raw_tasks,
        predictor_identity=prediction["predictor"],
        baseline_commitment=prediction["baseline_commitment"],
        predictor_predictions=prediction["predictions"],
        cohort_id=cohort_id,
        prediction_receipt=prediction,
        prediction_commit=prediction_commit,
    )
    if coordinator_preflight.get("status") != "READY_FOR_EXTERNAL_PREDICTION":
        raise ExternalRunnerError(
            "coordinator preflight is blocked: "
            + "; ".join(str(item) for item in coordinator_preflight.get("blockers", []))
        )
    if coordinator_preflight.get("blockers"):
        raise ExternalRunnerError("coordinator preflight returned blockers")
    rerun_summary = coordinator_preflight.get("check_summary", {})
    if rerun_summary.get("fail") or rerun_summary.get("blocked"):
        raise ExternalRunnerError("coordinator preflight rerun contains failed or blocked checks")
    if coordinator_preflight.get("external_compute_authorized") is not False:
        raise ExternalRunnerError("coordinator preflight unexpectedly granted compute authorization")
    rerun_dry_run = coordinator_preflight.get("dry_run")
    if not isinstance(rerun_dry_run, Mapping) or rerun_dry_run.get("prediction_receipt_sha256") != prediction.get("prediction_receipt_sha256"):
        raise ExternalRunnerError("coordinator preflight rerun is not bound to the prediction receipt")
    intervention_ids = tuple(str(item["id"]) for item in protocol["interventions"])
    if len(set(intervention_ids)) != len(intervention_ids) or "NO_CHANGE" not in intervention_ids:
        raise ExternalRunnerError("frozen intervention IDs are invalid")
    prompts: list[dict[str, str]] = []
    for task in tasks:
        for intervention_id in intervention_ids:
            prompts.append({
                "task_id": str(task["task_id"]),
                "intervention_id": intervention_id,
                "transformed_prompt": render_public_task(task, intervention_id),
            })
    config_payload = _normalize_config_payload(documents["config"])
    runtime_source = _strict_runtime_source_identity()
    file_hashes = {
        name: _sha_file(path)
        for name, path in paths.items()
        if name != "checkpoint"
    }
    file_hashes["checkpoint"] = str(subject["checkpoint_file_sha256"])
    file_hashes["tokenizer_metadata"] = _sha_file(tokenizer_metadata)
    return {
        "paths": paths,
        "tokenizer_metadata_path": tokenizer_metadata,
        "documents": documents,
        "protocol": protocol,
        "preflight": preflight,
        "subject": subject,
        "release": release,
        "split": split,
        "basis": basis,
        "basis_split": basis_split,
        "development_split": development_split,
        "prediction": prediction,
        "prediction_commit": prediction_commit,
        "tasks": tasks,
        "task_ids": tuple(str(task["task_id"]) for task in tasks),
        "intervention_ids": intervention_ids,
        "prompts": tuple(prompts),
        "config_payload": config_payload,
        "coordinator_preflight": coordinator_preflight,
        "runtime_source": runtime_source,
        "file_hashes": file_hashes,
        "subject_manifest_sha256": subject_verdict["subject_manifest_sha256"],
        "source_root": source,
        "release_root": release_base,
        "checkpoint_root": checkpoint_base,
    }


def validate_cuda_device(device: str) -> str:
    if not isinstance(device, str) or not CUDA_DEVICE_RE.fullmatch(device):
        raise ExternalRunnerError("device must be 'cuda' or an indexed CUDA device such as 'cuda:0'")
    return device


def validate_max_new_tokens(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_NEW_TOKENS_CEILING:
        raise ExternalRunnerError(
            f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS_CEILING}"
        )
    return value


def frozen_generation_policy(max_new_tokens: int) -> dict[str, Any]:
    return {
        "strategy": "GREEDY",
        "bos_prepended": True,
        "max_new_tokens": validate_max_new_tokens(max_new_tokens),
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": None,
        "fresh_state_per_cell": True,
    }


def _load_strict_runtime() -> StrictRuntime:
    runtime_parent = str((Path(__file__).resolve().parent / "_runtime").resolve())
    expected_package = Path(runtime_parent) / "anra_core" / "__init__.py"
    existing = sys.modules.get("anra_core")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or Path(existing_file).resolve() != expected_package.resolve():
            raise ExternalRunnerError("a different anra_core package is already loaded")
    if runtime_parent not in sys.path:
        sys.path.insert(0, runtime_parent)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    try:
        package = importlib.import_module("anra_core")
        torch = importlib.import_module("torch")
        core_config = getattr(package, "CoreConfig")
        core_executor = getattr(package, "CoreExecutor")
    except Exception as exc:
        raise ExternalRunnerError(f"strict runtime import failed: {exc}") from exc
    loaded_file = getattr(package, "__file__", None)
    if loaded_file is None or Path(loaded_file).resolve() != expected_package.resolve():
        raise ExternalRunnerError("loaded An-Ra runtime is not the repository strict runtime")
    return StrictRuntime(CoreConfig=core_config, CoreExecutor=core_executor, torch=torch)


def _configure_determinism(torch: Any) -> None:
    try:
        torch.manual_seed(0)
        if hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(0)
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None:
            cudnn.allow_tf32 = False
            cudnn.benchmark = False
            cudnn.deterministic = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        raise ExternalRunnerError(f"CUDA deterministic runtime setup failed: {exc}") from exc


def _descriptor_dict(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _plain_json(dict(value), name)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain_json(value.to_dict(), name)
    if is_dataclass(value):
        return _plain_json(asdict(value), name)
    raise ExternalRunnerError(f"{name} identity is unavailable")


def verify_strict_executor(
    executor: Any,
    *,
    subject: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    checkpoint_path: Path,
    tokenizer_path: Path,
    tokenizer_metadata_path: Path,
    runtime_source: Mapping[str, Any],
    requested_device: str,
) -> dict[str, Any]:
    """Verify the only permitted strict checkpoint loader's complete identity."""

    try:
        checkpoint_identity = _descriptor_dict(executor.checkpoint_identity, "checkpoint")
        runtime_identity = _descriptor_dict(executor.runtime_identity, "runtime")
        architecture_identity = _descriptor_dict(executor.architecture_identity(), "architecture")
        execution_profile = _descriptor_dict(executor.execution_profile, "execution_profile")
        representation_identity = _descriptor_dict(executor.representation_identity(), "representation")
        tokenizer_identity = _plain_json(executor.tokenizer.identity(), "tokenizer")
    except Exception as exc:
        raise ExternalRunnerError(f"strict executor identity extraction failed: {exc}") from exc
    expected_checkpoint = {
        "checkpoint_sha256": subject.get("checkpoint_file_sha256"),
        "parameter_sha256": subject.get("parameter_sha256"),
        "global_step": subject.get("global_step"),
        "training_stage": subject.get("stage"),
        "source_commit": subject.get("source_commit"),
    }
    for field, expected in expected_checkpoint.items():
        if checkpoint_identity.get(field) != expected:
            raise ExternalRunnerError(f"strict loader {field} does not match the subject manifest")
    if checkpoint_identity.get("legacy_unverified") is not False:
        raise ExternalRunnerError("strict loader identity reports legacy or unverified loading")
    if checkpoint_identity.get("tokenizer_contract_present") is not True:
        raise ExternalRunnerError("strict checkpoint has no tokenizer contract")
    if checkpoint_identity.get("tokenizer_contract_verified") is not True:
        raise ExternalRunnerError("strict checkpoint tokenizer contract is unverified")
    if checkpoint_identity.get("tokenizer_contract_valid") is not True:
        raise ExternalRunnerError("strict checkpoint tokenizer contract is invalid")
    if checkpoint_identity.get("artifact_class") not in {"full_resume", "model_only"}:
        raise ExternalRunnerError("strict checkpoint artifact class is unsupported")
    schema_version = checkpoint_identity.get("artifact_schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
        raise ExternalRunnerError("strict checkpoint artifact schema version is invalid")
    source_path = checkpoint_identity.get("source_path")
    if not isinstance(source_path, str) or Path(source_path).resolve() != checkpoint_path.resolve():
        raise ExternalRunnerError("strict loader source path does not match the explicit checkpoint")
    expected_runtime = {
        "engine_name": "anra-core-executor-vnext",
        "runtime_version": "0.5.0",
        "api_schema_version": 1,
        "state_schema_version": 2,
        "backend_framework": "torch",
    }
    for field, expected in expected_runtime.items():
        if runtime_identity.get(field) != expected:
            raise ExternalRunnerError(f"strict runtime identity mismatch: {field}")
    if not isinstance(runtime_identity.get("torch_version"), str) or not runtime_identity.get("torch_version"):
        raise ExternalRunnerError("strict runtime Torch version identity is missing")
    actual_device = str(getattr(executor, "device", ""))
    if not actual_device.startswith("cuda"):
        raise ExternalRunnerError("strict executor resolved to a non-CUDA device")
    if getattr(executor, "dtype_str", None) != "float32":
        raise ExternalRunnerError("strict executor is not exact FP32")
    if execution_profile.get("category") != "exact" or execution_profile.get("dtype") != "float32":
        raise ExternalRunnerError("strict executor execution profile is not exact FP32")
    if execution_profile.get("device") != actual_device or not str(execution_profile.get("device", "")).startswith("cuda"):
        raise ExternalRunnerError("strict executor execution profile device mismatch")
    expected_profile_id = f"anra-v4-executor-v2:exact:{actual_device}:float32"
    if execution_profile.get("profile_id") != expected_profile_id:
        raise ExternalRunnerError("strict executor profile identity is not the exact CUDA FP32 profile")
    model = getattr(executor, "model", None)
    model_config = getattr(model, "config", None)
    if model is None or model_config is None or getattr(model, "training", None) is not False:
        raise ExternalRunnerError("strict executor model is absent or not in evaluation mode")
    try:
        if _plain_json(model_config.immutable_fields(), "loaded_model_config") != _plain_json(config_payload):
            raise ExternalRunnerError("loaded model config differs from the verified config file")
        if architecture_identity.get("architecture_sha256") != model_config.architecture_sha256:
            raise ExternalRunnerError("architecture identity differs from the loaded model config")
    except AttributeError as exc:
        raise ExternalRunnerError("loaded model config identity is unavailable") from exc
    tokenizer_artifact_file_sha256 = _sha_file(tokenizer_path)
    if tokenizer_artifact_file_sha256 != subject.get("tokenizer_artifact_sha256"):
        raise ExternalRunnerError("loaded tokenizer artifact hash does not match the subject manifest")
    tokenizer_identity_sha256 = canonical_json_sha256(tokenizer_identity)
    if tokenizer_identity_sha256 != subject.get("tokenizer_identity_sha256"):
        raise ExternalRunnerError("tokenizer semantic identity does not match the subject manifest")
    for field in (
        "schema_version",
        "vocab_size",
        "vocabulary_sha256",
        "probe_count",
        "probe_sha256",
    ):
        if representation_identity.get(field) != tokenizer_identity.get(field):
            raise ExternalRunnerError(f"tokenizer representation identity mismatch: {field}")
    tokenizer = executor.tokenizer
    for attribute in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, attribute, None)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ExternalRunnerError(f"tokenizer contract identity is missing: {attribute}")
    sanitized_checkpoint = {
        field: checkpoint_identity.get(field)
        for field in (
            "checkpoint_sha256",
            "parameter_sha256",
            "global_step",
            "training_stage",
            "source_commit",
            "tokenizer_contract_valid",
            "tokenizer_contract_present",
            "tokenizer_contract_verified",
            "ignored_tensor_names",
            "legacy_unverified",
            "artifact_class",
            "artifact_schema_version",
        )
    }
    identity = {
        "runtime_identity": {
            **runtime_identity,
            "source_closure_sha256": runtime_source.get("source_closure_sha256"),
            "subject_runtime_source_revision": subject.get("runtime_source_revision"),
            "requested_device": requested_device,
        },
        "checkpoint_identity": sanitized_checkpoint,
        "tokenizer_identity": {
            "identity": tokenizer_identity,
            "identity_sha256": tokenizer_identity_sha256,
            "representation": representation_identity,
            "artifact_sha256": subject.get("tokenizer_artifact_sha256"),
            "artifact_file_sha256": tokenizer_artifact_file_sha256,
            "metadata_file_sha256": _sha_file(tokenizer_metadata_path),
        },
        "architecture_identity": {
            **architecture_identity,
            "model_config_file_sha256": subject.get("model_config_sha256"),
        },
        "execution_profile": execution_profile,
    }
    canonical_json_bytes(identity)
    return identity


def _validate_features(features: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(features, Mapping) or set(features) != set(FEATURE_FIELDS):
        return ["receipt features do not match the frozen six-feature schema"]
    for field in ("confidence", "entropy", "margin", "distinct_ratio"):
        value = features.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"feature is not finite: {field}")
    for field in ("output_len", "prompt_tokens"):
        value = features.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"feature count is invalid: {field}")
    if isinstance(features.get("output_len"), int) and features.get("prompt_tokens") == 0:
        errors.append("prompt_tokens must be positive")
    if isinstance(features.get("confidence"), (int, float)) and float(features.get("confidence")) > 0.0:
        errors.append("confidence must be non-positive mean log probability")
    if isinstance(features.get("entropy"), (int, float)) and float(features.get("entropy")) < 0.0:
        errors.append("entropy must be non-negative")
    if isinstance(features.get("margin"), (int, float)) and not 0.0 <= float(features.get("margin")) <= 1.0:
        errors.append("margin is outside [0, 1]")
    if isinstance(features.get("distinct_ratio"), (int, float)) and not 0.0 <= float(features.get("distinct_ratio")) <= 1.0:
        errors.append("distinct_ratio is outside [0, 1]")
    return errors


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key).lower()
            yield from _walk_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_keys(item)


def build_cell_receipt(
    *,
    task_id: str,
    intervention_id: str,
    transformed_prompt: str,
    raw_output: str,
    output_token_ids: Sequence[int],
    features: Mapping[str, Any],
    identity: Mapping[str, Any],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Build one canonical, truth-free immutable cell receipt."""

    if not isinstance(task_id, str) or not task_id:
        raise ExternalRunnerError("cell task_id must be non-empty text")
    if not isinstance(intervention_id, str) or not intervention_id:
        raise ExternalRunnerError("cell intervention_id must be non-empty text")
    if not isinstance(transformed_prompt, str) or not transformed_prompt:
        raise ExternalRunnerError("cell transformed_prompt must be non-empty text")
    if not isinstance(raw_output, str):
        raise ExternalRunnerError("cell raw_output must be text")
    feature_errors = _validate_features(features)
    if feature_errors:
        raise ExternalRunnerError("; ".join(feature_errors))
    normalized_ids: list[int] = []
    for token_id in output_token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ExternalRunnerError("cell output token IDs must be non-negative integers")
        normalized_ids.append(token_id)
    if features.get("output_len") != len(normalized_ids):
        raise ExternalRunnerError("output_len does not match output token IDs")
    expected_distinct_ratio = len(set(normalized_ids)) / len(normalized_ids) if normalized_ids else 0.0
    if float(features.get("distinct_ratio")) != expected_distinct_ratio:
        raise ExternalRunnerError("distinct_ratio does not match output token IDs")
    identity_fields = (
        "runtime_identity",
        "checkpoint_identity",
        "tokenizer_identity",
        "architecture_identity",
        "execution_profile",
    )
    if any(not isinstance(identity.get(field), Mapping) or not identity.get(field) for field in identity_fields):
        raise ExternalRunnerError("cell runtime and checkpoint identities are incomplete")
    body = {
        "schema": CELL_RECEIPT_SCHEMA,
        "renderer_id": RENDERER_ID,
        "task_id": task_id,
        "intervention_id": intervention_id,
        "transformed_prompt": transformed_prompt,
        "transformed_prompt_sha256": safe_output_sha256(transformed_prompt),
        "raw_output": raw_output,
        "raw_output_sha256": safe_output_sha256(raw_output),
        "output_token_ids": normalized_ids,
        "features": _plain_json(features, "features"),
        "runtime_identity": _plain_json(identity.get("runtime_identity"), "runtime_identity"),
        "checkpoint_identity": _plain_json(identity.get("checkpoint_identity"), "checkpoint_identity"),
        "tokenizer_identity": _plain_json(identity.get("tokenizer_identity"), "tokenizer_identity"),
        "architecture_identity": _plain_json(identity.get("architecture_identity"), "architecture_identity"),
        "execution_profile": _plain_json(identity.get("execution_profile"), "execution_profile"),
        "generation_policy": frozen_generation_policy(max_new_tokens),
    }
    body["cell_sha256"] = canonical_json_sha256(body)
    return body


def validate_cell_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_task_id: str | None = None,
    expected_intervention_id: str | None = None,
    expected_transformed_prompt: str | None = None,
    expected_identity: Mapping[str, Any] | None = None,
    expected_generation_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a cell receipt without trusting or overwriting persisted evidence."""

    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return {"valid": False, "errors": ["cell receipt must be an object"]}
    if set(receipt) != CELL_RECEIPT_FIELDS:
        errors.append(
            f"cell receipt fields invalid; missing={sorted(CELL_RECEIPT_FIELDS - set(receipt))}, "
            f"unknown={sorted(set(receipt) - CELL_RECEIPT_FIELDS)}"
        )
    if receipt.get("schema") != CELL_RECEIPT_SCHEMA or receipt.get("renderer_id") != RENDERER_ID:
        errors.append("cell receipt schema or renderer mismatch")
    if expected_task_id is not None and receipt.get("task_id") != expected_task_id:
        errors.append("cell task ID mismatch")
    if expected_intervention_id is not None and receipt.get("intervention_id") != expected_intervention_id:
        errors.append("cell intervention ID mismatch")
    if not isinstance(receipt.get("task_id"), str) or not receipt.get("task_id"):
        errors.append("cell task ID is invalid")
    if not isinstance(receipt.get("intervention_id"), str) or not receipt.get("intervention_id"):
        errors.append("cell intervention ID is invalid")
    prompt = receipt.get("transformed_prompt")
    output = receipt.get("raw_output")
    try:
        if not isinstance(prompt, str) or not prompt:
            errors.append("transformed prompt is invalid")
        elif safe_output_sha256(prompt) != receipt.get("transformed_prompt_sha256"):
            errors.append("transformed prompt hash mismatch")
        _require_sha256(receipt.get("transformed_prompt_sha256"), "transformed_prompt_sha256")
    except ExternalRunnerError as exc:
        errors.append(str(exc))
    try:
        if not isinstance(output, str):
            errors.append("raw output is invalid")
        elif safe_output_sha256(output) != receipt.get("raw_output_sha256"):
            errors.append("raw output hash mismatch")
        _require_sha256(receipt.get("raw_output_sha256"), "raw_output_sha256")
    except ExternalRunnerError as exc:
        errors.append(str(exc))
    token_ids = receipt.get("output_token_ids")
    token_ids_valid = isinstance(token_ids, list) and not any(
        not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
        for token_id in token_ids
    )
    if not token_ids_valid:
        errors.append("output token IDs are invalid")
    errors.extend(_validate_features(receipt.get("features")))
    if token_ids_valid and isinstance(receipt.get("features"), Mapping):
        token_ids_value = list(token_ids)
        if receipt["features"].get("output_len") != len(token_ids_value):
            errors.append("output_len does not match output token IDs")
        expected_distinct_ratio = len(set(token_ids_value)) / len(token_ids_value) if token_ids_value else 0.0
        observed_distinct_ratio = receipt["features"].get("distinct_ratio")
        if (
            isinstance(observed_distinct_ratio, (int, float))
            and not isinstance(observed_distinct_ratio, bool)
            and float(observed_distinct_ratio) != expected_distinct_ratio
        ):
            errors.append("distinct_ratio does not match output token IDs")
    generation_policy = receipt.get("generation_policy")
    if not isinstance(generation_policy, Mapping):
        errors.append("cell generation policy is missing")
    else:
        try:
            max_new_tokens = generation_policy.get("max_new_tokens")
            if canonical_json_bytes(generation_policy) != canonical_json_bytes(frozen_generation_policy(max_new_tokens)):
                errors.append("cell generation policy is not the frozen greedy policy")
        except ExternalRunnerError:
            errors.append("cell generation policy is invalid")
    for field in (
        "runtime_identity",
        "checkpoint_identity",
        "tokenizer_identity",
        "architecture_identity",
        "execution_profile",
    ):
        if not isinstance(receipt.get(field), Mapping) or not receipt.get(field):
            errors.append(f"cell identity is missing: {field}")
    forbidden = sorted(set(_walk_keys(receipt)) & FORBIDDEN_RECEIPT_KEYS)
    if forbidden:
        errors.append(f"cell receipt contains forbidden evaluator/truth fields: {forbidden}")
    if expected_transformed_prompt is not None and prompt != expected_transformed_prompt:
        errors.append("cell transformed prompt differs from the frozen render")
    if expected_identity is not None:
        for field in (
            "runtime_identity",
            "checkpoint_identity",
            "tokenizer_identity",
            "architecture_identity",
            "execution_profile",
        ):
            if canonical_json_bytes(receipt.get(field)) != canonical_json_bytes(expected_identity.get(field)):
                errors.append(f"cell {field} differs from the strict runtime identity")
    if expected_generation_policy is not None and canonical_json_bytes(generation_policy) != canonical_json_bytes(expected_generation_policy):
        errors.append("cell generation policy differs from the requested run")
    try:
        expected_cell_hash = _require_sha256(receipt.get("cell_sha256"), "cell_sha256")
        body = {key: value for key, value in receipt.items() if key != "cell_sha256"}
        if canonical_json_sha256(body) != expected_cell_hash:
            errors.append("canonical cell hash mismatch")
    except ExternalRunnerError as exc:
        errors.append(str(exc))
    return {
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "cell_sha256": receipt.get("cell_sha256"),
    }


def _serialized_json(value: Any) -> bytes:
    plain = _plain_json(value)
    return (
        json.dumps(plain, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8", errors="strict")


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = _serialized_json(value)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ReceiptExistsError(f"refusing to overwrite immutable receipt: {path}") from exc
    except OSError as exc:
        raise ExternalRunnerError(f"unable to persist immutable receipt {path}: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _cell_receipt_path(output_dir: Path, task_id: str, intervention_id: str) -> Path:
    task_key = hashlib.sha256(task_id.encode("utf-8", errors="strict")).hexdigest()[:16]
    intervention_key = hashlib.sha256(intervention_id.encode("utf-8", errors="strict")).hexdigest()[:16]
    return output_dir / "cells" / f"task-{task_key}-intervention-{intervention_key}.json"


def persist_cell_receipt(
    path: str | Path,
    receipt: Mapping[str, Any],
) -> str:
    """Persist a validated cell receipt with exclusive creation and no overwrite."""

    verdict = validate_cell_receipt(receipt)
    if not verdict["valid"]:
        raise ReceiptIntegrityError("refusing to persist invalid cell receipt: " + "; ".join(verdict["errors"]))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return _write_immutable_json(target, receipt)


def _execute_cell(
    executor: Any,
    torch: Any,
    *,
    task_id: str,
    intervention_id: str,
    transformed_prompt: str,
    max_new_tokens: int,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    tokenizer = executor.tokenizer
    encoded = tokenizer.encode(transformed_prompt)
    if not isinstance(encoded, list) or any(
        not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
        for token_id in encoded
    ):
        raise ExternalRunnerError("tokenizer returned malformed prompt token IDs")
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(bos_token_id, int) or not isinstance(eos_token_id, int):
        raise ExternalRunnerError("tokenizer BOS/EOS contract is invalid")
    prompt_token_ids = [bos_token_id, *encoded]
    state = None
    output_token_ids: list[int] = []
    step_logits: list[list[float]] = []
    try:
        prompt_tensor = torch.tensor([prompt_token_ids], dtype=torch.long, device=executor.device)
        state = executor.create_state(batch_size=1)
        result = executor.prefill(prompt_tensor, state)
        current_logits = result.logits[0, -1, :]
        for _ in range(max_new_tokens):
            step_logits.append(current_logits.detach().to(dtype=torch.float32).cpu().tolist())
            next_token_id = int(torch.argmax(current_logits, dim=-1).item())
            if next_token_id == eos_token_id:
                break
            output_token_ids.append(next_token_id)
            if len(output_token_ids) >= max_new_tokens:
                break
            current_logits = executor.forward_step(next_token_id, state).logits[0, -1, :]
        raw_output = tokenizer.decode(output_token_ids)
        features = summarize_generation_features(
            step_logits,
            output_token_ids,
            len(prompt_token_ids),
        )
        return build_cell_receipt(
            task_id=task_id,
            intervention_id=intervention_id,
            transformed_prompt=transformed_prompt,
            raw_output=raw_output,
            output_token_ids=output_token_ids,
            features=features,
            identity=identity,
            max_new_tokens=max_new_tokens,
        )
    except ExternalRunnerError:
        raise
    except Exception as exc:
        raise ExternalRunnerError(
            f"cell execution failed for {task_id}/{intervention_id}: {exc}"
        ) from exc
    finally:
        if state is not None:
            try:
                executor.release_state(state)
            except Exception as exc:
                raise ExternalRunnerError(
                    f"state release failed for {task_id}/{intervention_id}: {exc}"
                ) from exc


def load_existing_cell_receipt(
    path: str | Path,
    *,
    expected_task_id: str,
    expected_intervention_id: str,
    expected_transformed_prompt: str,
    expected_identity: Mapping[str, Any],
    expected_generation_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and fully validate one immutable receipt before resuming its cell."""

    target = Path(path)
    receipt = _load_object(target, "cell receipt")
    verdict = validate_cell_receipt(
        receipt,
        expected_task_id=expected_task_id,
        expected_intervention_id=expected_intervention_id,
        expected_transformed_prompt=expected_transformed_prompt,
        expected_identity=expected_identity,
        expected_generation_policy=expected_generation_policy,
    )
    if not verdict["valid"]:
        raise ReceiptIntegrityError(
            f"existing cell receipt is invalid at {target}: " + "; ".join(verdict["errors"])
        )
    return receipt


def _validate_or_get_receipt(
    path: Path,
    *,
    task_id: str,
    intervention_id: str,
    transformed_prompt: str,
    identity: Mapping[str, Any],
    generation_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    receipt = load_existing_cell_receipt(
        path,
        expected_task_id=task_id,
        expected_intervention_id=intervention_id,
        expected_transformed_prompt=transformed_prompt,
        expected_identity=identity,
        expected_generation_policy=generation_policy,
    )
    return receipt, _sha_file(path)


def _prepare_output_directory(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ExternalRunnerError("--output must be a directory")
    output.mkdir(parents=True, exist_ok=True)
    cells = output / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    allowed = {"cells", "run_manifest.json"}
    unexpected = sorted(item.name for item in output.iterdir() if item.name not in allowed)
    if unexpected:
        raise ExternalRunnerError(f"output directory contains unrelated artifacts: {unexpected}")
    return output


def _build_run_manifest(
    gates: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    cell_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    protocol = gates["protocol"]
    prediction = gates["prediction"]
    tasks = gates["tasks"]
    intervention_ids = list(gates["intervention_ids"])
    execution_order = ["NO_CHANGE", *[item for item in intervention_ids if item != "NO_CHANGE"]]
    expected_cells = len(tasks) * len(intervention_ids)
    if len(cell_records) != expected_cells:
        raise ExternalRunnerError("run manifest cell coverage is incomplete")
    input_files = []
    for name, path in gates["paths"].items():
        input_files.append({
            "name": name,
            "path": str(path),
            "sha256": gates["file_hashes"][name],
        })
    input_files.append({
        "name": "tokenizer_metadata",
        "path": str(gates["tokenizer_metadata_path"]),
        "sha256": gates["file_hashes"]["tokenizer_metadata"],
    })
    runner_source_path = Path(__file__).resolve()
    runner_source = {
        "path": "x_factor/x1_external_runner.py",
        "sha256": _sha_file(runner_source_path),
    }
    body = {
        "schema": RUN_MANIFEST_SCHEMA,
        "status": "RAW_EXTERNAL_EXECUTION_COMPLETE",
        "completion_scope": "RAW_EXTERNAL_EXECUTION_ONLY",
        "scientific_completion_claimed": False,
        "no_training": True,
        "no_weight_update": True,
        "no_weight_updates": True,
        "training_executed": False,
        "protocol_sha256": protocol["identity"]["protocol_sha256"],
        "renderer_id": RENDERER_ID,
        "runner_source": runner_source,
        "preflight_receipt_sha256": gates["preflight"]["preflight_sha256"],
        "coordinator_preflight_receipt_sha256": gates["coordinator_preflight"]["preflight_sha256"],
        "subject_manifest_sha256": gates["subject_manifest_sha256"],
        "release_manifest_sha256": gates["release"].get("release_manifest_sha256"),
        "basis_qualification_sha256": gates["basis"].get("basis_qualification_sha256"),
        "split_manifest_sha256": gates["split"].get("manifest_sha256"),
        "basis_split_manifest_sha256": gates["basis_split"].get("manifest_sha256"),
        "development_split_manifest_sha256": gates["development_split"].get("manifest_sha256"),
        "intervention_registry_sha256": protocol.get("intervention_registry_sha256"),
        "prediction_receipt_sha256": prediction["prediction_receipt_sha256"],
        "prediction_phase_commit_sha256": gates["prediction_commit"]["phase_commit_sha256"],
        "task_ids": list(gates["task_ids"]),
        "intervention_ids": intervention_ids,
        "execution_intervention_order": execution_order,
        "counts": {
            "tasks": len(tasks),
            "interventions": len(intervention_ids),
            "n_tasks": len(tasks),
            "n_interventions": len(intervention_ids),
            "expected_cells": expected_cells,
            "completed_cells": len(cell_records),
            "n_cells": len(cell_records),
        },
        "generation_policy": frozen_generation_policy(gates["max_new_tokens"]),
        "input_files": input_files,
        "cells": list(cell_records),
        "strict_runtime_identity": {
            **identity["runtime_identity"],
            "source_files": gates["runtime_source"]["files"],
        },
        "checkpoint_identity": identity["checkpoint_identity"],
        "tokenizer_identity": identity["tokenizer_identity"],
        "architecture_identity": identity["architecture_identity"],
        "device": identity["execution_profile"]["device"],
        "execution_profile": identity["execution_profile"],
        "claim": "Raw external execution receipts only; no correctness, evaluator, or scientific completion claim.",
    }
    body["run_manifest_sha256"] = canonical_json_sha256(body)
    return body


def _persist_or_validate_run_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    verdict = validate_run_manifest(manifest)
    if not verdict["valid"]:
        raise ReceiptIntegrityError("refusing to persist invalid run manifest: " + "; ".join(verdict["errors"]))
    if path.exists():
        existing = _load_object(path, "run manifest")
        if canonical_json_bytes(existing) != canonical_json_bytes(manifest):
            raise ReceiptExistsError("existing run manifest differs; refusing overwrite")
        return _sha_file(path)
    return _write_immutable_json(path, manifest)


def validate_run_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        return {"valid": False, "errors": ["run manifest must be an object"]}
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        errors.append("run manifest schema mismatch")
    if manifest.get("renderer_id") != RENDERER_ID:
        errors.append("run manifest renderer mismatch")
    runner_source = manifest.get("runner_source")
    if not isinstance(runner_source, Mapping):
        errors.append("run manifest runner source identity is missing")
    else:
        try:
            _require_sha256(runner_source.get("sha256"), "runner_source.sha256")
        except ExternalRunnerError as exc:
            errors.append(str(exc))
    if manifest.get("status") != "RAW_EXTERNAL_EXECUTION_COMPLETE":
        errors.append("run manifest status is not raw external execution complete")
    if manifest.get("scientific_completion_claimed") is not False:
        errors.append("run manifest claims scientific completion")
    if (
        manifest.get("no_training") is not True
        or manifest.get("no_weight_update") is not True
        or manifest.get("no_weight_updates") is not True
        or manifest.get("training_executed") is not False
    ):
        errors.append("run manifest no-training flags are invalid")
    generation_policy = manifest.get("generation_policy")
    try:
        if not isinstance(generation_policy, Mapping) or canonical_json_bytes(generation_policy) != canonical_json_bytes(
            frozen_generation_policy(generation_policy.get("max_new_tokens"))
        ):
            errors.append("run manifest generation policy is invalid")
    except ExternalRunnerError:
        errors.append("run manifest generation policy is invalid")
    forbidden = sorted(set(_walk_keys(manifest)) & FORBIDDEN_RECEIPT_KEYS)
    if forbidden:
        errors.append(f"run manifest contains forbidden evaluator/truth fields: {forbidden}")
    counts = manifest.get("counts")
    cells = manifest.get("cells")
    if not isinstance(counts, Mapping) or not isinstance(cells, list):
        errors.append("run manifest counts or cells are missing")
    else:
        expected = counts.get("expected_cells")
        completed = counts.get("completed_cells")
        if not isinstance(expected, int) or not isinstance(completed, int) or expected != completed or completed != len(cells):
            errors.append("run manifest cell counts do not reconcile")
    for field in (
        "protocol_sha256",
        "preflight_receipt_sha256",
        "coordinator_preflight_receipt_sha256",
        "subject_manifest_sha256",
        "release_manifest_sha256",
        "basis_qualification_sha256",
        "split_manifest_sha256",
        "basis_split_manifest_sha256",
        "development_split_manifest_sha256",
        "intervention_registry_sha256",
        "prediction_receipt_sha256",
        "prediction_phase_commit_sha256",
        "run_manifest_sha256",
    ):
        try:
            _require_sha256(manifest.get(field), field)
        except ExternalRunnerError as exc:
            errors.append(str(exc))
    identity = _require_sha256(manifest.get("run_manifest_sha256"), "run_manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "run_manifest_sha256"}
    if canonical_json_sha256(body) != identity:
        errors.append("run manifest hash mismatch")
    return {"valid": not errors, "errors": errors, "run_manifest_sha256": manifest.get("run_manifest_sha256")}


def run_external_matrix(
    *,
    protocol_path: str | Path,
    preflight_path: str | Path,
    subject_path: str | Path,
    registry_path: str | Path,
    release_manifest_path: str | Path,
    split_path: str | Path,
    public_tasks_path: str | Path,
    basis_qualification_path: str | Path,
    basis_split_path: str | Path,
    development_split_path: str | Path,
    prediction_path: str | Path,
    prediction_commit_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    tokenizer_path: str | Path,
    output: str | Path,
    authorization: str,
    device: str = "cuda",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    source_root: str | Path | None = None,
    release_root: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the frozen external matrix after all gates, with immutable cell receipts."""

    if authorization != AUTHORIZATION_PHRASE:
        raise AuthorizationError(
            f"exact external authorization phrase required: {AUTHORIZATION_PHRASE!r}"
        )
    selected_device = validate_cuda_device(device)
    selected_max_new_tokens = validate_max_new_tokens(max_new_tokens)
    source = source_root or Path(__file__).resolve().parents[1]
    gates = validate_execution_gates(
        protocol_path=protocol_path,
        preflight_path=preflight_path,
        subject_path=subject_path,
        registry_path=registry_path,
        release_manifest_path=release_manifest_path,
        split_path=split_path,
        public_tasks_path=public_tasks_path,
        basis_qualification_path=basis_qualification_path,
        basis_split_path=basis_split_path,
        development_split_path=development_split_path,
        prediction_path=prediction_path,
        prediction_commit_path=prediction_commit_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        source_root=source,
        release_root=release_root or Path.cwd(),
        checkpoint_root=checkpoint_root or Path.cwd(),
    )
    gates["max_new_tokens"] = selected_max_new_tokens
    strict_runtime = _load_strict_runtime()
    _configure_determinism(strict_runtime.torch)
    config = strict_runtime.CoreConfig(**dict(gates["config_payload"]))
    block_size = getattr(config, "block_size", None)
    if not isinstance(block_size, int) or block_size < 1:
        raise ExternalRunnerError("strict CoreConfig block_size is invalid")
    output_dir = _prepare_output_directory(output)
    try:
        executor = strict_runtime.CoreExecutor.from_checkpoint(
            gates["paths"]["checkpoint"],
            tokenizer_path=gates["paths"]["tokenizer"],
            config=config,
            device=selected_device,
            dtype="float32",
            profile_category="exact",
            enable_telemetry=False,
            allow_legacy_unverified=False,
        )
        identity = verify_strict_executor(
            executor,
            subject=gates["subject"],
            config_payload=gates["config_payload"],
            checkpoint_path=gates["paths"]["checkpoint"],
            tokenizer_path=gates["paths"]["tokenizer"],
            tokenizer_metadata_path=gates["tokenizer_metadata_path"],
            runtime_source=gates["runtime_source"],
            requested_device=selected_device,
        )
        current_runtime_source = _strict_runtime_source_identity()
        if current_runtime_source["source_closure_sha256"] != gates["runtime_source"]["source_closure_sha256"]:
            raise ExternalRunnerError("strict runtime source changed after preflight gates")
        gates["runtime_source"] = current_runtime_source
        actual_block_size = int(getattr(executor.model.config, "block_size"))
        if actual_block_size != block_size:
            raise ExternalRunnerError("loaded executor block size differs from verified config")
        prompt_records: dict[tuple[str, str], str] = {
            (row["task_id"], row["intervention_id"]): row["transformed_prompt"]
            for row in gates["prompts"]
        }
        for task in gates["tasks"]:
            task_id = str(task["task_id"])
            for intervention_id in gates["intervention_ids"]:
                transformed_prompt = prompt_records[(task_id, intervention_id)]
                prompt_ids = [executor.tokenizer.bos_token_id, *executor.tokenizer.encode(transformed_prompt)]
                if len(prompt_ids) + selected_max_new_tokens > actual_block_size:
                    raise ExternalRunnerError(
                        f"prompt plus max_new_tokens exceeds strict runtime context: {task_id}/{intervention_id}"
                    )
        execution_order = ["NO_CHANGE", *[item for item in gates["intervention_ids"] if item != "NO_CHANGE"]]
        generation_policy = frozen_generation_policy(selected_max_new_tokens)
        existing_receipts: dict[tuple[str, str], tuple[dict[str, Any], str, Path]] = {}
        for task in gates["tasks"]:
            task_id = str(task["task_id"])
            for intervention_id in execution_order:
                transformed_prompt = prompt_records[(task_id, intervention_id)]
                receipt_path = _cell_receipt_path(output_dir, task_id, intervention_id)
                if receipt_path.exists():
                    receipt, file_hash = _validate_or_get_receipt(
                        receipt_path,
                        task_id=task_id,
                        intervention_id=intervention_id,
                        transformed_prompt=transformed_prompt,
                        identity=identity,
                        generation_policy=generation_policy,
                    )
                    existing_receipts[(task_id, intervention_id)] = (receipt, file_hash, receipt_path)
        cell_records: list[dict[str, Any]] = []
        for task in gates["tasks"]:
            task_id = str(task["task_id"])
            for intervention_id in execution_order:
                transformed_prompt = prompt_records[(task_id, intervention_id)]
                receipt_path = _cell_receipt_path(output_dir, task_id, intervention_id)
                cached = existing_receipts.get((task_id, intervention_id))
                if cached is not None and receipt_path.exists():
                    receipt, file_hash = _validate_or_get_receipt(
                        receipt_path,
                        task_id=task_id,
                        intervention_id=intervention_id,
                        transformed_prompt=transformed_prompt,
                        identity=identity,
                        generation_policy=generation_policy,
                    )
                else:
                    receipt = _execute_cell(
                        executor,
                        strict_runtime.torch,
                        task_id=task_id,
                        intervention_id=intervention_id,
                        transformed_prompt=transformed_prompt,
                        max_new_tokens=selected_max_new_tokens,
                        identity=identity,
                    )
                    try:
                        file_hash = persist_cell_receipt(receipt_path, receipt)
                    except ReceiptExistsError:
                        receipt, file_hash = _validate_or_get_receipt(
                            receipt_path,
                            task_id=task_id,
                            intervention_id=intervention_id,
                            transformed_prompt=transformed_prompt,
                            identity=identity,
                            generation_policy=generation_policy,
                        )
                cell_records.append({
                    "task_id": task_id,
                    "intervention_id": intervention_id,
                    "path": str(receipt_path.relative_to(output_dir)),
                    "file_sha256": file_hash,
                    "cell_sha256": receipt["cell_sha256"],
                })
        manifest = _build_run_manifest(
            gates,
            identity=identity,
            cell_records=cell_records,
        )
        manifest_path = output_dir / "run_manifest.json"
        _persist_or_validate_run_manifest(manifest_path, manifest)
        return manifest
    finally:
        if "executor" in locals():
            del executor
        cuda_namespace = getattr(strict_runtime.torch, "cuda", None)
        if cuda_namespace is not None and hasattr(cuda_namespace, "empty_cache"):
            try:
                cuda_namespace.empty_cache()
            except Exception:
                pass


run_x1_external = run_external_matrix


class _CLIParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIUsageError(f"CLI usage error: {message}")


def _build_parser() -> argparse.ArgumentParser:
    parser = _CLIParser(prog="python -m x_factor.x1_external_runner")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--subject", "--subject-manifest", dest="subject", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--release-manifest", "--release", dest="release_manifest", required=True)
    parser.add_argument("--split", "--primary-split", dest="split", required=True)
    parser.add_argument("--public-tasks", "--tasks", dest="public_tasks", required=True)
    parser.add_argument("--basis-qualification", "--basis", dest="basis_qualification", required=True)
    parser.add_argument("--basis-split", required=True)
    parser.add_argument("--development-split", required=True)
    parser.add_argument("--prediction", "--prediction-receipt", dest="prediction", required=True)
    parser.add_argument("--prediction-commit", "--phase-commit", dest="prediction_commit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", "--model-config", dest="config", required=True, help="strict JSON CoreConfig artifact")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True, help="immutable run output directory")
    parser.add_argument(
        "--authorization",
        required=True,
        help=f"exact required phrase: {AUTHORIZATION_PHRASE}",
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--source-root")
    parser.add_argument("--release-root")
    parser.add_argument("--checkpoint-root")
    return parser


def _print_json(value: Mapping[str, Any]) -> None:
    print(_serialized_json(value).decode("utf-8"), end="")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        result = run_external_matrix(
            protocol_path=args.protocol,
            preflight_path=args.preflight,
            subject_path=args.subject,
            registry_path=args.registry,
            release_manifest_path=args.release_manifest,
            split_path=args.split,
            public_tasks_path=args.public_tasks,
            basis_qualification_path=args.basis_qualification,
            basis_split_path=args.basis_split,
            development_split_path=args.development_split,
            prediction_path=args.prediction,
            prediction_commit_path=args.prediction_commit,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            tokenizer_path=args.tokenizer,
            output=args.output,
            authorization=args.authorization,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            source_root=args.source_root,
            release_root=args.release_root,
            checkpoint_root=args.checkpoint_root,
        )
        _print_json(result)
        return 0
    except CLIUsageError as exc:
        _print_json({
            "schema": CLI_ERROR_SCHEMA,
            "status": "BLOCKED",
            "error_type": "CLIUsageError",
            "message": str(exc),
            "next_action": f"correct the command-line options and use the exact authorization phrase {AUTHORIZATION_PHRASE!r}",
        })
        return 2
    except SystemExit as exc:
        if exc.code == 0:
            raise
        _print_json({
            "schema": CLI_ERROR_SCHEMA,
            "status": "BLOCKED",
            "error_type": "CLIUsageError",
            "message": "command-line usage error",
            "next_action": f"supply every required path and the exact authorization phrase: {AUTHORIZATION_PHRASE!r}",
        })
        return 2
    except Exception as exc:
        _print_json({
            "schema": CLI_ERROR_SCHEMA,
            "status": "BLOCKED",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "next_action": "correct the blocked evidence or runtime contract and rerun; no fallback or silent completion is permitted",
        })
        return 2


__all__ = [
    "AUTHORIZATION_PHRASE",
    "ANSWER_MARKER_FORMATS",
    "CELL_RECEIPT_SCHEMA",
    "CLI_ERROR_SCHEMA",
    "CLIUsageError",
    "EXTERNAL_AUTHORIZATION_PHRASE",
    "ExternalRunnerError",
    "MAX_NEW_TOKENS_CEILING",
    "RENDERER_ID",
    "RUN_MANIFEST_SCHEMA",
    "build_cell_receipt",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "frozen_generation_policy",
    "load_existing_cell_receipt",
    "load_json_strict",
    "load_strict_json",
    "loads_json_strict",
    "main",
    "persist_cell_receipt",
    "render_public_task",
    "run_external_matrix",
    "run_x1_external",
    "safe_output_hash",
    "safe_output_sha256",
    "summarize_generation_features",
    "validate_cell_receipt",
    "validate_cuda_device",
    "validate_execution_gates",
    "validate_max_new_tokens",
    "validate_run_manifest",
    "verify_strict_executor",
]


if __name__ == "__main__":
    raise SystemExit(main())
