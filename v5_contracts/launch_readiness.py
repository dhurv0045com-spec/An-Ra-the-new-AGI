"""Fail-closed V5 evidence-gate evaluator.

This command checks the evidence inventory for bounded prelaunch work and
independent freeze review. It is not a production launcher or a scientific
approval authority. File presence and self-declared PASS never authorize training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "anra-v5-launch-gates/v1"
RECEIPT_SCHEMA = "anra-v5-launch-readiness-receipt/v1"
EXPECTED_GATES = ("E1", "E2", "E3", "E4", "E5", "E6")
EXPECTED_IDENTITIES = {
    "data_manifest_sha256", "pack_manifest_sha256", "runtime_image_sha256",
    "sealed_evaluation_commitment_sha256", "tokenizer_artifact_sha256",
    "topology_receipt_sha256",
}
EXPECTED_DOCUMENTS = (
    "README.md",
    "V5_TRAINING_SPEC_v1.0.md",
    "V5_MASTER_BLUEPRINT.md",
    "IMPLEMENTATION_BLUEPRINT.md",
    "BENCHMARK.md",
    "EXPERIMENTS.md",
    "EXECUTION.md",
    "FREEZE_CHECKLIST.md",
    "DECISIONS.md",
    "DECISION_LOG.md",
    "OPEN_QUESTIONS.md",
    "STATUS.md",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _inside(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("receipt path must be a non-empty repository-relative string")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("receipt path escapes repository root")
    return candidate


def validate_gate_manifest(manifest: dict[str, Any], *, root: Path) -> dict[str, bool]:
    gates = manifest.get("gates")
    if not isinstance(gates, list):
        raise ValueError("launch gates must be a list")
    gate_ids = [gate.get("id") for gate in gates if isinstance(gate, dict)]
    if tuple(gate_ids) != EXPECTED_GATES:
        raise ValueError(f"launch gates must be ordered exactly as {EXPECTED_GATES}")
    for gate in gates:
        status = gate.get("status")
        if status not in {"PENDING", "PASS", "FAIL"}:
            raise ValueError(f"invalid {gate.get('id')} status: {status}")
        path_value = gate.get("receipt_path")
        hash_value = gate.get("receipt_sha256")
        if status == "PASS":
            receipt = _inside(root, path_value)
            if not receipt.is_file():
                raise ValueError(f"{gate['id']} PASS receipt does not exist: {path_value}")
            if not _is_sha256(hash_value) or _sha256_file(receipt) != hash_value:
                raise ValueError(f"{gate['id']} PASS receipt hash mismatch")
        elif path_value is not None or hash_value is not None:
            raise ValueError(f"{gate['id']} non-PASS gate must not claim a receipt")
    identities = manifest.get("external_identities")
    if not isinstance(identities, dict) or set(identities) != EXPECTED_IDENTITIES:
        raise ValueError("external identity inventory must match the six required identities")
    return {
        "schema": manifest.get("schema") == SCHEMA,
        "gate_inventory_exact": tuple(gate_ids) == EXPECTED_GATES,
        "statuses_valid": all(gate["status"] in {"PENDING", "PASS", "FAIL"} for gate in gates),
        "pass_receipts_hash_bound": True,
        "external_identity_inventory_present": bool(identities),
        "launch_request_is_boolean": type(manifest.get("main_run_requested")) is bool,
    }


def build_readiness(
    *,
    root: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    repository = (root or Path(__file__).resolve().parents[1]).resolve()
    manifest_file = (manifest_path or repository / "blueprint/LAUNCH_GATES.json").resolve()
    if not manifest_file.is_relative_to(repository):
        raise ValueError("launch manifest must be inside repository root")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    checks = validate_gate_manifest(manifest, root=repository)
    blueprint_root = repository / "blueprint"
    document_hashes: dict[str, str] = {}
    for name in EXPECTED_DOCUMENTS:
        path = blueprint_root / name
        if not path.is_file():
            raise ValueError(f"canonical blueprint document missing: {name}")
        document_hashes[name] = _text_sha256(path)
    candidate_path = _inside(repository, manifest.get("candidate_spec"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_ok = (
        candidate.get("status") == "PASS_IMPLEMENTATION_SPEC_BLOCKED_MAIN_RUN"
        and candidate.get("spec", {}).get("main_training_authorized") is False
        and bool(candidate.get("checks"))
        and all(value is True for value in candidate.get("checks", {}).values())
    )
    checks["candidate_spec_valid_and_fail_closed"] = candidate_ok
    checks["canonical_blueprint_complete"] = len(document_hashes) == len(EXPECTED_DOCUMENTS)
    gates = manifest["gates"]
    failures = [gate["id"] for gate in gates if gate["status"] == "FAIL"]
    pending = [gate["id"] for gate in gates if gate["status"] == "PENDING"]
    all_gates_pass = all(gate["status"] == "PASS" for gate in gates)
    identities = manifest["external_identities"]
    missing_identities = sorted(name for name, value in identities.items() if not _is_sha256(value))
    experiments_authorized = all(checks.values()) and not failures
    ready_for_review = (
        experiments_authorized
        and all_gates_pass
        and not missing_identities
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "status": (
            "READY_FOR_FREEZE_REVIEW"
            if ready_for_review
            else "READY_FOR_PRELAUNCH_EXPERIMENTS"
            if experiments_authorized
            else "BLOCKED_PRELAUNCH"
        ),
        "experiments_authorized": experiments_authorized,
        "main_training_authorized": False,
        "production_launcher_implemented": False,
        "independent_freeze_review_required": True,
        "candidate_spec_sha256": _sha256_file(candidate_path),
        "launch_gate_manifest_sha256": _text_sha256(manifest_file),
        "implementation_sha256": _text_sha256(Path(__file__)),
        "blueprint_document_sha256": document_hashes,
        "checks": checks,
        "passed_gates": [gate["id"] for gate in gates if gate["status"] == "PASS"],
        "pending_gates": pending,
        "failed_gates": failures,
        "missing_external_identities": missing_identities,
        "next_action": "execute E1 tokenizer/corpus gate" if pending else "independent freeze review",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_readiness(manifest_path=args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["experiments_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
