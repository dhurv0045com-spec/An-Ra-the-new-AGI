#!/usr/bin/env python3
"""Fail-closed preflight for the Signac 100M-class research core.

The report is a readiness diagnostic, never a training authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Make the command usable as both `python tools/...py` and `python -m ...`
# without installing this research repository as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signac_100m.spec import MODEL_SPEC, resource_estimate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_gate(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"state": "BLOCKED", "reason": "No qualified data manifest supplied."}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"state": "BLOCKED", "reason": f"Cannot read manifest: {exc}"}
    if not isinstance(payload, dict):
        return {"state": "BLOCKED", "reason": "Data manifest must be a JSON object."}
    required = {
        "schema", "lifecycle_state", "training_manifest_sha256",
        "tokenizer_sha256", "source_ledger_sha256", "pack_manifest_sha256",
        "contamination_audit_sha256", "evaluation_manifest_sha256",
        "qualified_real_tokens",
    }
    missing = sorted(required - payload.keys())
    if missing:
        return {"state": "BLOCKED", "reason": "Manifest missing required fields.", "missing": missing}
    if payload["schema"] != "anra-v5-data-manifest/v1":
        return {"state": "BLOCKED", "reason": "Unsupported data manifest schema.",
                "schema": payload["schema"]}
    hashes = [payload[key] for key in required if key.endswith("sha256")]
    if any(not isinstance(value, str) or len(value) != 64
           or any(ch not in "0123456789abcdef" for ch in value) for value in hashes):
        return {"state": "BLOCKED", "reason": "Manifest contains invalid SHA-256 identities."}
    if payload["lifecycle_state"] != "RUNNABLE":
        return {"state": "BLOCKED", "reason": "Dataset lifecycle is not RUNNABLE.",
                "lifecycle_state": payload["lifecycle_state"]}
    tokens = payload["qualified_real_tokens"]
    if type(tokens) is not int or tokens <= 0:
        return {"state": "BLOCKED", "reason": "qualified_real_tokens must be a positive integer."}
    return {"state": "PRESENT_FOR_REVIEW", "manifest_sha256": _sha256(path),
            "qualified_real_tokens": tokens,
            "note": "Schema checks do not replace independent corpus, evaluation, licensing, or contamination review."}


def build_report(*, data_manifest: Path | None = None,
                 evaluation_receipt: Path | None = None,
                 target: str = "cpu") -> dict[str, Any]:
    MODEL_SPEC.assert_valid()
    receipts = MODEL_SPEC.parameter_receipt()
    data = _manifest_gate(data_manifest)
    if evaluation_receipt is None:
        evaluation = {"state": "BLOCKED", "reason": "No evaluation readiness receipt supplied."}
    elif not evaluation_receipt.is_file():
        evaluation = {"state": "BLOCKED", "reason": "Evaluation receipt path does not exist."}
    else:
        evaluation_error = None
        try:
            evaluation_payload = json.loads(evaluation_receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            evaluation_payload = None
            evaluation_error = str(exc)
        if not isinstance(evaluation_payload, dict):
            detail = evaluation_error or "top-level JSON value is not an object"
            evaluation = {"state": "BLOCKED", "reason": "Evaluation receipt must be a JSON object.",
                          "detail": detail}
        else:
            schema = evaluation_payload.get("receipt_schema", evaluation_payload.get("schema"))
            if not isinstance(schema, str) or not schema.strip():
                evaluation = {"state": "BLOCKED", "reason": "Evaluation receipt needs a nonempty schema identity."}
            else:
                evaluation = {"state": "PRESENT_FOR_REVIEW", "path": str(evaluation_receipt.resolve()),
                              "schema": schema, "sha256": _sha256(evaluation_receipt),
                              "note": "Receipt shape is checked; Citadel readiness and evidence bindings still require independent review."}
    runtime = {"target": target, "state": "NOT_PROBED",
               "note": "A static report cannot certify Kaggle hardware, XLA parity, memory fit, or durable resume."}
    if target == "tpu":
        runtime["state"] = "TPU_EVIDENCE_REQUIRED"
    implementation = {
        "state": "IMPLEMENTED",
        "model_spec_sha256": MODEL_SPEC.sha256(),
        "parameter_count_exact": receipts.total,
        "parameter_target_class": "100M-class (existing M102 recipe)",
    }
    blockers = []
    for name, gate in (("data", data), ("evaluation", evaluation), ("runtime", runtime)):
        if gate["state"] not in {"PRESENT_FOR_REVIEW", "SATISFIED"}:
            blockers.append(name)
    return {
        "schema": "anra-signac-100m-preflight/v1",
        "verdict": "BLOCKED" if blockers else "REVIEW_REQUIRED",
        "training_authorized": False,
        "implementation": implementation,
        "geometry_and_resources": resource_estimate(),
        "gates": {"data": data, "evaluation": evaluation, "target_runtime": runtime},
        "blockers": blockers,
        "claim_ceiling": "Architecture/preflight only. No training, capability, TPU, or AGI result is implied.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--evaluation-receipt", type=Path)
    parser.add_argument("--target", choices=("cpu", "cuda", "tpu"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(data_manifest=args.data_manifest,
                          evaluation_receipt=args.evaluation_receipt,
                          target=args.target)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["verdict"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
