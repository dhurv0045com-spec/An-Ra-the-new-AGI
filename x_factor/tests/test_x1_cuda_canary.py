from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import x_factor.x1_cuda_canary as canary


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _intake(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint = tmp_path / "candidate.pt"
    config = tmp_path / "config.json"
    tokenizer = tmp_path / "tokenizer.json"
    metadata = tmp_path / "tokenizer.json.meta.json"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text(json.dumps({"architecture_version": "v"}), encoding="utf-8")
    tokenizer.write_text(json.dumps({"token_to_id": {}}), encoding="utf-8")
    metadata.write_text(json.dumps({"schema_version": 4}), encoding="utf-8")
    body = {
        "schema": "anra-x1-real-1-candidate-intake/v1",
        "status": "INVENTORIED",
        "research_subject": False,
        "eligible_for_x1": False,
        "promotion_status": "BLOCKED",
        "model_execution": False,
        "parameter_sha256": "a" * 64,
        "tokenizer_identity_sha256": "b" * 64,
        "files": {
            "checkpoint": {"path": str(checkpoint), "exists": True, "bytes": checkpoint.stat().st_size, "sha256": _sha(checkpoint)},
            "model_config": {"path": str(config), "exists": True, "bytes": config.stat().st_size, "sha256": _sha(config)},
            "tokenizer_artifact": {"path": str(tokenizer), "exists": True, "bytes": tokenizer.stat().st_size, "sha256": _sha(tokenizer)},
        },
    }
    from x_factor.x1_external_runner import canonical_json_sha256
    body["intake_sha256"] = canonical_json_sha256(body)
    intake = tmp_path / "intake.json"
    intake.write_text(json.dumps(body), encoding="utf-8")
    return checkpoint, config, tokenizer, intake


def _args(tmp_path: Path) -> dict[str, object]:
    checkpoint, config, tokenizer, intake = _intake(tmp_path)
    return {
        "checkpoint_path": checkpoint,
        "config_path": config,
        "tokenizer_path": tokenizer,
        "intake_path": intake,
        "output_path": tmp_path / "canary.json",
        "source_root": Path(__file__).resolve().parents[2],
        "device": "cuda",
        "max_new_tokens": 1,
        "authorization": canary.AUTHORIZATION_PHRASE,
    }


def test_authorization_precedes_file_and_runtime_access(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args["authorization"] = "wrong"
    args["checkpoint_path"] = tmp_path / "missing.pt"
    with pytest.raises(canary.CanaryAuthorizationError):
        canary.run_canary(**args)


def test_cpu_and_non_one_token_are_rejected_before_runtime(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args["device"] = "cpu"
    with pytest.raises(Exception, match="device"):
        canary.run_canary(**args)
    args = _args(tmp_path)
    args["max_new_tokens"] = 2
    with pytest.raises(canary.CanaryError, match="max_new_tokens"):
        canary.run_canary(**args)


def test_intake_requires_blocked_non_subject_and_file_hashes(tmp_path: Path) -> None:
    checkpoint, config, tokenizer, intake = _intake(tmp_path)
    document = json.loads(intake.read_text(encoding="utf-8"))
    observed = canary._validate_intake(document, checkpoint, config, tokenizer)
    assert observed["promotion_status"] == "BLOCKED"
    document["eligible_for_x1"] = True
    with pytest.raises(canary.CanaryError, match="non-subject"):
        canary._validate_intake(document, checkpoint, config, tokenizer)
    document["eligible_for_x1"] = False
    document["files"]["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(canary.CanaryError, match="hash mismatch"):
        canary._validate_intake(document, checkpoint, config, tokenizer)


def test_receipt_validation_rejects_scientific_claim_and_tampering() -> None:
    body = {
        "schema": canary.CANARY_SCHEMA,
        "status": "PASS",
        "scientific_subject": False,
        "scientific_completion_claimed": False,
        "training": False,
        "weight_updates": False,
        "checkpoint_file_sha256": "a" * 64,
        "tokenizer_identity_sha256": "b" * 64,
        "device_requested": "cuda",
    }
    from x_factor.x1_external_runner import canonical_json_sha256
    body["canary_sha256"] = canonical_json_sha256(body)
    expected = {
        "checkpoint_file_sha256": "a" * 64,
        "tokenizer_identity_sha256": "b" * 64,
        "device_requested": "cuda",
    }
    assert canary._validate_receipt(body, expected)["status"] == "PASS"
    tampered = dict(body)
    tampered["scientific_subject"] = True
    with pytest.raises(canary.CanaryReceiptError):
        canary._validate_receipt(tampered, expected)


def test_cli_wrong_authorization_is_structured(capsys: pytest.CaptureFixture[str]) -> None:
    code = canary.main([
        "--checkpoint", "checkpoint",
        "--config", "config",
        "--tokenizer", "tokenizer",
        "--intake", "intake",
        "--output", "output",
        "--source-root", "source",
        "--device", "cuda",
        "--max-new-tokens", "1",
        "--authorization", "wrong",
    ])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "BLOCKED"
    assert payload["error_type"] == "CanaryAuthorizationError"
    assert payload["scientific_subject"] is False


def test_module_import_is_model_free() -> None:
    code = "import sys; import x_factor.x1_cuda_canary; assert 'torch' not in sys.modules; assert 'anra_core' not in sys.modules; print('ok')"
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "ok"
