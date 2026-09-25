from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

import x_factor.x1_external_runner as runner
from x_factor.x1_external_runner import (
    AUTHORIZATION_PHRASE,
    CELL_RECEIPT_SCHEMA,
    RENDERER_ID,
    AuthorizationError,
    ExternalRunnerError,
    ReceiptExistsError,
    build_cell_receipt,
    frozen_generation_policy,
    load_existing_cell_receipt,
    load_json_strict,
    main,
    persist_cell_receipt,
    render_public_task,
    run_external_matrix,
    safe_output_sha256,
    summarize_generation_features,
    validate_cell_receipt,
)


def _task() -> dict:
    return {
        "task_id": "task-001",
        "context": "Visible context.",
        "query": "Return one visible candidate.",
        "visible_candidates": ["AAA-001", "BBB-002"],
        "format": "text",
        "features": {
            "confidence": -0.2,
            "entropy": 0.7,
            "margin": 0.1,
            "output_len": 4,
            "distinct_ratio": 1.0,
            "prompt_tokens": 12,
        },
    }


def _identity() -> dict:
    return {
        "runtime_identity": {"engine_name": "anra-core-executor-vnext", "runtime_version": "0.5.0"},
        "checkpoint_identity": {
            "checkpoint_sha256": "a" * 64,
            "parameter_sha256": "b" * 64,
            "legacy_unverified": False,
        },
        "tokenizer_identity": {"identity_sha256": "c" * 64},
        "architecture_identity": {"architecture_sha256": "d" * 64},
        "execution_profile": {
            "profile_id": "anra-v4-executor-v2:exact:cuda:0:float32",
            "category": "exact",
            "device": "cuda:0",
            "dtype": "float32",
        },
    }


def _receipt() -> dict:
    return build_cell_receipt(
        task_id="task-001",
        intervention_id="NO_CHANGE",
        transformed_prompt="Visible context.\nAnswer:",
        raw_output="AAA-001",
        output_token_ids=[4, 5, 6],
        features={
            "confidence": -0.2,
            "entropy": 0.4,
            "margin": 0.1,
            "output_len": 3,
            "distinct_ratio": 1.0,
            "prompt_tokens": 5,
        },
        identity=_identity(),
    )


def test_strict_json_rejects_duplicate_keys_and_nonstandard_constants(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"outer":{"value":1,"value":2}}', encoding="utf-8")
    with pytest.raises(ExternalRunnerError, match="duplicate JSON object key"):
        load_json_strict(duplicate)
    for literal in ("NaN", "Infinity", "-Infinity", "1e400"):
        artifact = tmp_path / f"constant-{literal}.json"
        artifact.write_text(f'{{"value":{literal}}}', encoding="utf-8")
        with pytest.raises(ExternalRunnerError):
            load_json_strict(artifact)


def test_renderer_is_deterministic_includes_candidates_and_rejects_unknown_format() -> None:
    task = _task()
    first = render_public_task(task, "NO_CHANGE")
    second = render_public_task(task, "NO_CHANGE")
    transformed = render_public_task(task, "STATE_TABLE")
    assert first == second
    assert first == render_public_task(copy.deepcopy(task), "NO_CHANGE")
    assert "AAA-001" in first and "BBB-002" in first
    assert "-0.2" not in first
    assert transformed.endswith("Answer:")
    assert "AAA-001" in transformed and "BBB-002" in transformed
    for supported_format in ("json", "prose", "table", "code"):
        task["format"] = supported_format
        assert render_public_task(task, "NO_CHANGE").endswith("Answer:")
    task["format"] = "unsupported"
    with pytest.raises(ExternalRunnerError, match="unsupported answer-marker format"):
        render_public_task(task, "NO_CHANGE")


def test_generation_feature_math_is_frozen_and_pure() -> None:
    features = summarize_generation_features(
        [[0.0, 0.0], [math.log(3.0), 0.0, 0.0]],
        [0, 1],
        3,
    )
    assert features["confidence"] == pytest.approx((math.log(0.5) + math.log(0.2)) / 2.0)
    assert features["entropy"] == pytest.approx(math.log(2.0))
    assert features["margin"] == pytest.approx(0.0)
    assert features["output_len"] == 2
    assert features["distinct_ratio"] == 1.0
    assert features["prompt_tokens"] == 3
    with pytest.raises(ExternalRunnerError):
        summarize_generation_features([[0.0, float("nan")]], [0], 1)


def test_authorization_and_runtime_gate_ordering_never_reaches_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_loader = Mock()
    monkeypatch.setattr(runner, "_load_strict_runtime", runtime_loader)
    arguments = {
        "protocol_path": "protocol.json",
        "preflight_path": "preflight.json",
        "subject_path": "subject.json",
        "registry_path": "registry.json",
        "release_manifest_path": "release.json",
        "split_path": "split.json",
        "public_tasks_path": "tasks.json",
        "basis_qualification_path": "basis.json",
        "basis_split_path": "basis-split.json",
        "development_split_path": "development-split.json",
        "prediction_path": "prediction.json",
        "prediction_commit_path": "prediction-commit.json",
        "checkpoint_path": "checkpoint.pt",
        "config_path": "config.json",
        "tokenizer_path": "tokenizer.json",
        "output": "output",
        "device": "cuda",
        "max_new_tokens": 8,
    }
    with pytest.raises(AuthorizationError):
        run_external_matrix(**arguments, authorization="authorized")
    runtime_loader.assert_not_called()
    gate = Mock(side_effect=ExternalRunnerError("preflight blocked"))
    monkeypatch.setattr(runner, "validate_execution_gates", gate)
    with pytest.raises(ExternalRunnerError, match="preflight blocked"):
        run_external_matrix(**arguments, authorization=AUTHORIZATION_PHRASE)
    runtime_loader.assert_not_called()
    gate.reset_mock()
    with pytest.raises(ExternalRunnerError, match="device"):
        run_external_matrix(**{**arguments, "device": "cpu"}, authorization=AUTHORIZATION_PHRASE)
    runtime_loader.assert_not_called()
    gate.assert_not_called()


def test_receipt_resume_validation_rejects_tampering_and_never_overwrites(tmp_path: Path) -> None:
    receipt = _receipt()
    path = tmp_path / "cells" / "cell.json"
    file_hash = persist_cell_receipt(path, receipt)
    assert len(file_hash) == 64
    loaded = load_json_strict(path)
    verdict = validate_cell_receipt(
        loaded,
        expected_task_id="task-001",
        expected_intervention_id="NO_CHANGE",
        expected_transformed_prompt="Visible context.\nAnswer:",
        expected_identity=_identity(),
        expected_generation_policy=frozen_generation_policy(32),
    )
    assert verdict["valid"] is True
    resumed = load_existing_cell_receipt(
        path,
        expected_task_id="task-001",
        expected_intervention_id="NO_CHANGE",
        expected_transformed_prompt="Visible context.\nAnswer:",
        expected_identity=_identity(),
        expected_generation_policy=frozen_generation_policy(32),
    )
    assert resumed == loaded
    tampered = copy.deepcopy(loaded)
    tampered["raw_output"] = "BBB-002"
    assert validate_cell_receipt(tampered)["valid"] is False
    assert validate_cell_receipt(loaded, expected_identity=_identity())["valid"] is True
    assert validate_cell_receipt(
        loaded,
        expected_generation_policy=frozen_generation_policy(8),
    )["valid"] is False
    with pytest.raises(ReceiptExistsError):
        persist_cell_receipt(path, receipt)


def test_safe_output_hash_uses_exact_utf8_without_normalization() -> None:
    composed = "é"
    decomposed = "e\u0301"
    assert safe_output_sha256(composed) != safe_output_sha256(decomposed)
    assert safe_output_sha256("") == runner.hashlib.sha256(b"").hexdigest()
    with pytest.raises(ExternalRunnerError):
        safe_output_sha256("\ud800")


def test_cli_blocked_error_is_structured(capsys: pytest.CaptureFixture[str]) -> None:
    argv = [
        "--protocol", "protocol.json",
        "--preflight", "preflight.json",
        "--subject", "subject.json",
        "--registry", "registry.json",
        "--release-manifest", "release.json",
        "--split", "split.json",
        "--public-tasks", "tasks.json",
        "--basis-qualification", "basis.json",
        "--basis-split", "basis-split.json",
        "--development-split", "development-split.json",
        "--prediction", "prediction.json",
        "--prediction-commit", "prediction-commit.json",
        "--checkpoint", "checkpoint.pt",
        "--config", "config.json",
        "--tokenizer", "tokenizer.json",
        "--output", "output",
        "--authorization", "wrong",
        "--device", "cuda",
        "--max-new-tokens", "8",
        "--source-root", "source",
        "--release-root", "release",
        "--checkpoint-root", "checkpoint",
    ]
    assert main(argv) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "anra-x1-real-1-external-runner-error/v1"
    assert payload["status"] == "BLOCKED"
    assert payload["error_type"] == "AuthorizationError"


def test_module_import_is_model_free() -> None:
    code = (
        "import sys; import x_factor.x1_external_runner; "
        "assert 'torch' not in sys.modules; assert 'anra_core' not in sys.modules; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "ok"


def test_receipt_contract_has_no_truth_fields() -> None:
    receipt = _receipt()
    assert receipt["schema"] == CELL_RECEIPT_SCHEMA
    assert receipt["renderer_id"] == RENDERER_ID
    keys = set(receipt)
    assert not keys & {"gold", "correct", "correctness", "outcome", "evaluator", "verifier"}
