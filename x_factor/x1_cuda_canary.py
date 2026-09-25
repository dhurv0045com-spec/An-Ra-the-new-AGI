"""Non-scientific strict CUDA checkpoint canary for X1-REAL-1."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from x_factor.x1_external_runner import (
    ExternalRunnerError,
    _normalize_config_payload,
    _sha_file,
    canonical_json_bytes,
    canonical_json_sha256,
    load_json_strict,
    validate_cuda_device,
)


CANARY_SCHEMA = "anra-x1-real-1-cuda-canary/v1"
AUTHORIZATION_PHRASE = "I AUTHORIZE X1-REAL-1 CUDA CANARY"
CLI_ERROR_SCHEMA = "anra-x1-real-1-cuda-canary-error/v1"
MAX_CANARY_TOKENS = 8
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CanaryError(RuntimeError):
    pass


class CanaryAuthorizationError(CanaryError):
    pass


class CanaryReceiptError(CanaryError):
    pass


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CanaryError(f"{name} must be a JSON object")
    return dict(value)


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise CanaryError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _file(path: str | Path, name: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise CanaryError(f"{name} is not a regular file: {candidate}")
    return candidate


def _strict_json(path: str | Path, name: str) -> Any:
    try:
        return load_json_strict(path)
    except ExternalRunnerError as exc:
        raise CanaryError(f"{name} strict JSON rejected: {exc}") from exc


def _runtime_source_identity(source_root: Path) -> dict[str, Any]:
    runtime_root = source_root / "x_factor" / "_runtime" / "anra_core"
    if not runtime_root.is_dir():
        raise CanaryError("strict An-Ra runtime source directory is missing")
    records = []
    for path in sorted(runtime_root.rglob("*.py"), key=lambda item: item.as_posix()):
        records.append({
            "path": path.relative_to(source_root).as_posix(),
            "sha256": _sha_file(path),
        })
    if not records:
        raise CanaryError("strict An-Ra runtime source closure is empty")
    return {
        "source_closure_sha256": canonical_json_sha256(records),
        "files": records,
    }


def _load_runtime(source_root: Path) -> tuple[Any, Any, Any]:
    runtime_parent = str((source_root / "x_factor" / "_runtime").resolve())
    expected = Path(runtime_parent) / "anra_core" / "__init__.py"
    existing = sys.modules.get("anra_core")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or Path(existing_file).resolve() != expected.resolve():
            raise CanaryError("a different anra_core package is already loaded")
    if runtime_parent not in sys.path:
        sys.path.insert(0, runtime_parent)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    try:
        package = importlib.import_module("anra_core")
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise CanaryError(f"strict runtime import failed: {type(exc).__name__}: {exc}") from exc
    loaded = getattr(package, "__file__", None)
    if loaded is None or Path(loaded).resolve() != expected.resolve():
        raise CanaryError("loaded An-Ra runtime is not the supplied source runtime")
    return package, torch, package.CoreConfig


def _configure_torch(torch: Any) -> None:
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
        raise CanaryError(f"CUDA determinism setup failed: {type(exc).__name__}: {exc}") from exc


def _validate_intake(document: Mapping[str, Any], checkpoint: Path, config: Path, tokenizer: Path) -> dict[str, Any]:
    if document.get("schema") != "anra-x1-real-1-candidate-intake/v1":
        raise CanaryError("candidate intake schema mismatch")
    if document.get("status") not in {"INVENTORIED", "UNQUALIFIED_NEW"}:
        raise CanaryError("candidate intake is not an inventory receipt")
    if document.get("promotion_status") != "BLOCKED":
        raise CanaryError("candidate intake promotion status is not explicitly blocked")
    if document.get("research_subject") is not False or document.get("eligible_for_x1") is not False:
        raise CanaryError("canary requires a non-subject, non-eligible candidate")
    if document.get("model_execution") is not False:
        raise CanaryError("candidate intake claims model execution")
    intake_hash = _hash(document.get("intake_sha256"), "intake_sha256")
    body = {key: value for key, value in document.items() if key != "intake_sha256"}
    if canonical_json_sha256(body) != intake_hash:
        raise CanaryError("candidate intake hash mismatch")
    files = document.get("files")
    if not isinstance(files, Mapping):
        raise CanaryError("candidate intake files block is missing")
    expected_files = {"checkpoint": checkpoint, "model_config": config, "tokenizer_artifact": tokenizer}
    observed: dict[str, Any] = {}
    for name, path in expected_files.items():
        record = files.get(name)
        if not isinstance(record, Mapping):
            raise CanaryError(f"candidate intake file record is missing: {name}")
        actual = _sha_file(path)
        if record.get("sha256") != actual:
            raise CanaryError(f"candidate intake file hash mismatch: {name}")
        if record.get("path") and Path(str(record["path"])).name != path.name:
            raise CanaryError(f"candidate intake file name mismatch: {name}")
        observed[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    return {
        "intake_sha256": intake_hash,
        "status": document["status"],
        "promotion_status": document["promotion_status"],
        "parameter_sha256": document.get("parameter_sha256"),
        "tokenizer_identity_sha256": document.get("tokenizer_identity_sha256"),
        "files": observed,
    }


def _load_config(path: Path) -> tuple[Any, dict[str, Any]]:
    document = _strict_json(path, "model config")
    payload = _normalize_config_payload(_object(document, "model config"))
    return payload, payload


def _executor_identity(executor: Any, runtime: Mapping[str, Any], requested_device: str) -> dict[str, Any]:
    try:
        checkpoint = executor.checkpoint_identity.to_dict()
        runtime_identity = executor.runtime_identity.to_dict()
        architecture = executor.architecture_identity().to_dict()
        profile = executor.execution_profile.to_dict()
        tokenizer = executor.tokenizer.identity()
        representation = executor.representation_identity().to_dict()
    except Exception as exc:
        raise CanaryError(f"strict executor identity extraction failed: {type(exc).__name__}: {exc}") from exc
    if checkpoint.get("legacy_unverified") is not False:
        raise CanaryError("strict canary refused legacy or unverified checkpoint loading")
    if checkpoint.get("tokenizer_contract_verified") is not True:
        raise CanaryError("strict canary requires a verified tokenizer contract")
    if checkpoint.get("artifact_class") not in {"model_only", "full_resume"}:
        raise CanaryError("strict canary checkpoint artifact class is unsupported")
    actual_device = str(getattr(executor, "device", ""))
    if not actual_device.startswith("cuda"):
        raise CanaryError("strict canary executor did not resolve to CUDA")
    if getattr(executor, "dtype_str", None) != "float32":
        raise CanaryError("strict canary executor is not exact FP32")
    if profile.get("category") != "exact" or profile.get("dtype") != "float32":
        raise CanaryError("strict canary profile is not exact FP32")
    if profile.get("device") != actual_device:
        raise CanaryError("strict canary device profile mismatch")
    if getattr(executor.model, "training", None) is not False:
        raise CanaryError("strict canary model is not in evaluation mode")
    identity = {
        "runtime_identity": runtime_identity,
        "architecture_identity": architecture,
        "execution_profile": profile,
        "checkpoint_identity": checkpoint,
        "tokenizer_identity": tokenizer,
        "representation_identity": representation,
        "runtime_source": dict(runtime),
        "requested_device": requested_device,
    }
    canonical_json_bytes(identity)
    return identity


def _run_forward_canary(executor: Any, torch: Any) -> dict[str, Any]:
    prompt = "X1-REAL-1 CUDA RUNTIME CANARY\nAnswer:"
    tokenizer = executor.tokenizer
    ids = [tokenizer.bos_token_id, *tokenizer.encode(prompt)]
    state = None
    try:
        input_ids = torch.tensor([ids], dtype=torch.long, device=executor.device)
        state = executor.create_state(batch_size=1)
        result = executor.prefill(input_ids, state)
        logits = result.logits[0, -1, :].detach().to(dtype=torch.float32)
        if not bool(torch.isfinite(logits).all().item()):
            raise CanaryError("strict canary logits are not finite")
        selected = int(torch.argmax(logits, dim=-1).item())
        return {
            "prompt": prompt,
            "prompt_token_count": len(ids),
            "selected_token_id": selected,
            "logits_sha256": hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest(),
            "shape": list(logits.shape),
        }
    except CanaryError:
        raise
    except Exception as exc:
        raise CanaryError(f"strict CUDA forward canary failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if state is not None:
            try:
                executor.release_state(state)
            except Exception as exc:
                raise CanaryError(f"strict canary state release failed: {type(exc).__name__}: {exc}") from exc


def _validate_receipt(receipt: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") != CANARY_SCHEMA:
        raise CanaryReceiptError("canary receipt schema mismatch")
    if receipt.get("status") not in {"PASS", "PASS_WITH_IDENTITY_WARNING"}:
        raise CanaryReceiptError("canary receipt is not a pass receipt")
    if receipt.get("scientific_subject") is not False or receipt.get("scientific_completion_claimed") is not False:
        raise CanaryReceiptError("canary receipt makes a scientific claim")
    if receipt.get("training") is not False or receipt.get("weight_updates") is not False:
        raise CanaryReceiptError("canary receipt claims training or weight updates")
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise CanaryReceiptError(f"canary receipt binding mismatch: {key}")
    body = {key: value for key, value in receipt.items() if key != "canary_sha256"}
    if canonical_json_sha256(body) != receipt.get("canary_sha256"):
        raise CanaryReceiptError("canary receipt hash mismatch")
    return dict(receipt)


def run_canary(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path,
    tokenizer_path: str | Path,
    intake_path: str | Path,
    output_path: str | Path,
    source_root: str | Path,
    device: str,
    max_new_tokens: int,
    authorization: str,
) -> dict[str, Any]:
    if authorization != AUTHORIZATION_PHRASE:
        raise CanaryAuthorizationError(f"exact canary authorization phrase required: {AUTHORIZATION_PHRASE!r}")
    selected_device = validate_cuda_device(device)
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens != 1:
        raise CanaryError("max_new_tokens must equal 1 for the forward-only CUDA canary")
    source = Path(source_root).expanduser().resolve()
    checkpoint = _file(checkpoint_path, "checkpoint")
    config = _file(config_path, "model config")
    tokenizer = _file(tokenizer_path, "tokenizer")
    metadata = _file(tokenizer.with_suffix(tokenizer.suffix + ".meta.json"), "tokenizer metadata")
    intake_document = _object(_strict_json(intake_path, "candidate intake"), "candidate intake")
    intake = _validate_intake(intake_document, checkpoint, config, tokenizer)
    config_payload, _ = _load_config(config)
    runtime_source = _runtime_source_identity(source)
    package, torch, core_config_type = _load_runtime(source)
    _configure_torch(torch)
    if not torch.cuda.is_available():
        raise CanaryError("CUDA is unavailable; canary has no CPU fallback")
    config = core_config_type(**config_payload)
    try:
        executor = package.CoreExecutor.from_checkpoint(
            checkpoint,
            tokenizer_path=tokenizer,
            config=config,
            device=selected_device,
            dtype="float32",
            profile_category="exact",
            enable_telemetry=False,
            allow_legacy_unverified=False,
        )
        identity = _executor_identity(executor, runtime_source, selected_device)
        checkpoint_identity = identity["checkpoint_identity"]
        expected_checkpoint_hash = _hash(intake["files"]["checkpoint"]["sha256"], "intake checkpoint sha256")
        if checkpoint_identity.get("checkpoint_sha256") != expected_checkpoint_hash:
            raise CanaryError("strict loader checkpoint hash differs from candidate intake")
        declared_parameter = intake.get("parameter_sha256")
        loaded_parameter = checkpoint_identity.get("parameter_sha256")
        identity_warning = False
        if declared_parameter and loaded_parameter != declared_parameter:
            identity_warning = True
        tokenizer_identity_hash = canonical_json_sha256(identity["tokenizer_identity"])
        declared_tokenizer_identity = intake.get("tokenizer_identity_sha256")
        if declared_tokenizer_identity and tokenizer_identity_hash != declared_tokenizer_identity:
            raise CanaryError("strict tokenizer identity differs from candidate intake")
        canary_execution = _run_forward_canary(executor, torch)
        status = "PASS_WITH_IDENTITY_WARNING" if identity_warning else "PASS"
        body = {
            "schema": CANARY_SCHEMA,
            "status": status,
            "scope": "CUDA_RUNTIME_CANARY_ONLY",
            "scientific_subject": False,
            "scientific_completion_claimed": False,
            "promotion_authorized": False,
            "training": False,
            "weight_updates": False,
            "device_requested": selected_device,
            "max_new_tokens": max_new_tokens,
            "checkpoint_file_sha256": expected_checkpoint_hash,
            "parameter_sha256": loaded_parameter,
            "declared_parameter_sha256": declared_parameter,
            "tokenizer_identity_sha256": tokenizer_identity_hash,
            "tokenizer_metadata_sha256": _sha_file(metadata),
            "identity": identity,
            "canary_execution": canary_execution,
            "identity_warning": identity_warning,
        }
        body["canary_sha256"] = canonical_json_sha256(body)
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = _object(_strict_json(target, "existing canary receipt"), "existing canary receipt")
            return _validate_receipt(existing, {
                "checkpoint_file_sha256": expected_checkpoint_hash,
                "tokenizer_identity_sha256": tokenizer_identity_hash,
                "device_requested": selected_device,
            })
        payload = (json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        with target.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return body
    finally:
        if "executor" in locals():
            del executor
        if "torch" in locals() and hasattr(torch, "cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CanaryError(f"CLI usage error: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="python -m x_factor.x1_cuda_canary")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--intake", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--authorization", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = run_canary(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            tokenizer_path=args.tokenizer,
            intake_path=args.intake,
            output_path=args.output,
            source_root=args.source_root,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            authorization=args.authorization,
        )
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except SystemExit as exc:
        if exc.code == 0:
            raise
        print(json.dumps({"schema": CLI_ERROR_SCHEMA, "status": "BLOCKED", "error_type": "CLIUsageError", "message": "command-line usage error"}, indent=2, sort_keys=True))
        return 2
    except Exception as exc:
        print(json.dumps({"schema": CLI_ERROR_SCHEMA, "status": "BLOCKED", "error_type": type(exc).__name__, "message": str(exc), "scientific_subject": False, "training": False}, indent=2, sort_keys=True, ensure_ascii=False))
        return 2


__all__ = [
    "AUTHORIZATION_PHRASE",
    "CANARY_SCHEMA",
    "CanaryAuthorizationError",
    "CanaryError",
    "CanaryReceiptError",
    "main",
    "run_canary",
]


if __name__ == "__main__":
    raise SystemExit(main())
