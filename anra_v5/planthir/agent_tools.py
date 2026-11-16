"""Read-only, machine-readable discovery and run inspection for agents."""
from __future__ import annotations

from dataclasses import MISSING, fields
import inspect
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .cartridge import CartridgeArmSpec, CartridgeRunner, ExperimentCartridge


def _inputs(cls: type) -> list[dict[str, Any]]:
    return [{"name": f.name, "type": str(f.type),
             "required": f.default is MISSING and f.default_factory is MISSING}
            for f in fields(cls)]


def agent_info() -> dict[str, Any]:
    return {
        "schema": "anra-planthir-agent-info-v1",
        "backend": "V8/Elendril CartridgeRunner",
        "capabilities": ["instrument checks", "registered multi-arm training", "checkpoint resume", "diagnostics", "bundle export and verification"],
        "limitations": ["runner supports V8 backend only", "uniform cross-entropy recipes only", "custom trainer factories and b0_job are unsupported", "GPU production requires a clean checkout at source_commit", "diagnostic gates do not establish scientific causation"],
        "inputs": {"ExperimentCartridge": _inputs(ExperimentCartridge), "CartridgeArmSpec": _inputs(CartridgeArmSpec)},
        "cpu_canary_limits": {"cpu_canary_required": True, "max_training_samples": 32, "max_parameters": 100000, "max_updates": 8, "purpose": "mechanical fixtures only"},
        "commands": {"calibrate": "python -m anra_v5.planthir calibrate", "verify_directory": "python -m anra_v5.planthir verify-bundle --dir PATH", "verify_archive": "python -m anra_v5.planthir verify-bundle --zip PATH", "inspect": "python -m anra_v5.planthir inspect-run --work-root PATH"},
        "legacy_commands": {"run-gpu-validation": "NOT_QUALIFIED: legacy launcher disabled pending migration", "run-validation-study": "NOT_QUALIFIED: legacy study requires review before use"},
        "recommended_workflow": "Use the current Python runner with a registered cartridge and focused tiny CPU fixtures before any production execution.",
        "python_runner_signature": "CartridgeRunner.run" + str(inspect.signature(CartridgeRunner.run)),
        "artifacts": {"RUN_IDENTITY.json": "invocation identity binding", "RUN_STATUS.json": "latest attempt state", "ATTEMPTS.jsonl": "previous attempt states on resume", "updates/*.jsonl": "per-arm update journal", "checkpoints/*/LATEST": "latest committed checkpoint pointer", "FINAL_RECEIPT.json": "final export receipt; verify the bundle separately", "result_bundle/result.zip": "diagnostic result bundle", "result_bundle/checkpoint_full.zip": "full checkpoint archive"},
        "stop_conditions": ["instrument or design gate fails", "training or diagnostic failure", "deadline or resource admission failure", "identity mismatch on resume", "bundle verification or parity failure"],
    }


class InspectionError(Exception):
    def __init__(self, file: str, detail: str):
        self.file, self.detail = file, detail
        super().__init__(detail)


def inspect_run(work_root: str | Path) -> dict[str, Any]:
    root = Path(work_root)
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    def read_json(name: str, *, required: bool) -> dict[str, Any] | None:
        path = root / name
        if not path.exists():
            if required:
                raise InspectionError(name, "missing")
            return None
        try:
            if path.stat().st_size > 4 * 1024 * 1024:
                raise InspectionError(name, "metadata exceeds 4 MiB inspection limit")
            value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
        except (OSError, ValueError, UnicodeError) as exc:
            raise InspectionError(name, f"unreadable or invalid JSON: {type(exc).__name__}") from exc
        if not isinstance(value, dict):
            raise InspectionError(name, "expected JSON object")
        return value

    identity = read_json("RUN_IDENTITY.json", required=True)
    status = read_json("RUN_STATUS.json", required=True)
    if not isinstance(identity.get("identity"), dict) or not isinstance(identity.get("sha256"), str):
        raise InspectionError("RUN_IDENTITY.json", "missing identity or sha256")
    try:
        canonical = json.dumps(identity["identity"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise InspectionError("RUN_IDENTITY.json", "invalid identity content") from exc
    if sha256(canonical).hexdigest() != identity["sha256"]:
        raise InspectionError("RUN_IDENTITY.json", "identity digest mismatch")
    if status.get("status") not in {"RUNNING", "COMPLETE", "FAILED", "INTERRUPTED"} or status.get("identity_sha256") != identity["sha256"]:
        raise InspectionError("RUN_STATUS.json", "invalid status or identity binding")
    attempts_path = root / "ATTEMPTS.jsonl"
    attempts = []
    if attempts_path.exists():
        try:
            if attempts_path.stat().st_size > 4 * 1024 * 1024:
                raise InspectionError("ATTEMPTS.jsonl", "metadata exceeds 4 MiB inspection limit")
            with attempts_path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    row = json.loads(line, parse_constant=reject_constant)
                    if not isinstance(row, dict):
                        raise InspectionError("ATTEMPTS.jsonl", f"line {number} is not an object")
                    attempts.append(row)
        except (OSError, ValueError, UnicodeError) as exc:
            raise InspectionError("ATTEMPTS.jsonl", f"unreadable or invalid JSON: {type(exc).__name__}") from exc
    receipt = read_json("FINAL_RECEIPT.json", required=False)
    return {"schema": "anra-planthir-run-inspection-v1", "work_root": str(root),
            "identity": identity, "status": status, "attempts": attempts,
            "final_receipt_present": receipt is not None, "final_receipt": receipt,
            "verification": "NOT_CHECKED"}
