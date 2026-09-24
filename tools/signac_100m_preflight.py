#!/usr/bin/env python3
"""Fail-closed preflight for the Signac 100M-class research core.

The report is a readiness diagnostic, never a training authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path, PurePosixPath
from typing import Any

# Make the command usable as both `python tools/...py` and `python -m ...`
# without installing this research repository as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signac_100m.spec import MODEL_SPEC, resource_estimate

CANARY_SCHEMA = "anra-signac-kaggle-all-core-canary/v6"
RESTART_SCHEMA = "anra-signac-kaggle-controlled-restart/v5"
HOST_MEMORY_SCHEMA = "anra-signac-host-memory-observation/v1"
HOST_MEMORY_METRIC = "process_lifetime_high_water_rss"
HOST_MEMORY_SCOPE = "one_worker_process; not host-wide capacity"
HOST_MEMORY_FINAL_HASH_SAMPLE_POINT = "after final parameter and Adam-moment hashes"
EXPECTED_TPU_WORKLOAD = {
    "expected_global_devices": 8,
    "expected_world_size": 8,
    "sequence_length": 4096,
    "microbatch_size": 1,
    "accumulation_steps": 4,
    "seed": 73_011,
    "learning_rate": 3e-4,
    "max_grad_norm": 1.0,
}
QUALIFICATION_MAX_PEAK_MEMORY_FRACTION = 0.85
EVALUATION_READINESS_SCHEMA = "anra-signac-citadel-readiness/v1"
EVALUATION_READINESS_HASH_FIELDS = (
    "evaluator_sha256", "suite_sha256", "protocol_sha256",
    "preregistration_sha256", "sensitivity_gate_result_sha256",
    "independent_review_sha256",
)


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


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _evaluation_receipt_gate(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"state": "BLOCKED", "reason": "No evaluation readiness receipt supplied."}
    if not path.is_file():
        return {"state": "BLOCKED", "reason": "Evaluation receipt path does not exist."}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"state": "BLOCKED", "reason": f"Cannot read evaluation receipt: {exc}"}
    if not isinstance(payload, dict):
        return {"state": "BLOCKED", "reason": "Evaluation receipt must be a JSON object."}
    if payload.get("schema") != EVALUATION_READINESS_SCHEMA:
        return {
            "state": "BLOCKED",
            "reason": "Receipt is not the frozen Citadel readiness schema; development evaluation receipts do not satisfy this gate.",
            "schema": payload.get("schema"),
        }

    missing = [field for field in EVALUATION_READINESS_HASH_FIELDS if field not in payload]
    invalid_hashes = [field for field in EVALUATION_READINESS_HASH_FIELDS
                      if field in payload and not _is_sha256(payload[field])]
    required_checks = {
        "scope": "signac-100m-citadel-readiness",
        "status": "PASS",
        "firewall_status": "PASS",
        "executable_truth_status": "PASS",
        "eos_status": "PASS",
    }
    checks_not_passed = [field for field, expected in required_checks.items()
                         if payload.get(field) != expected]
    if missing or invalid_hashes or checks_not_passed:
        return {
            "state": "BLOCKED",
            "reason": "Citadel readiness receipt failed its structural evidence contract.",
            "missing_hashes": missing,
            "invalid_hashes": invalid_hashes,
            "checks_not_passed": checks_not_passed,
        }
    return {
        "state": "PRESENT_FOR_REVIEW",
        "path": str(path.resolve()),
        "schema": EVALUATION_READINESS_SCHEMA,
        "sha256": _sha256(path),
        "note": "Structural checks passed; evaluator authorship, independent review, and evidence bindings still require audit.",
    }


def _finite_positive(value: object) -> bool:
    return (type(value) in {int, float} and math.isfinite(value) and value > 0)


def _validate_runtime_receipt(payload: object, *, profile: str,
                              expected_source_tree_sha256: str) -> dict[str, Any]:
    """Validate a frozen synthetic TPU canary without treating it as production proof."""

    if not isinstance(payload, dict):
        raise ValueError("receipt must be a JSON object")
    if payload.get("schema") != CANARY_SCHEMA or payload.get("status") != "PASS":
        raise ValueError("unsupported canary schema or non-PASS status")
    if payload.get("candidate") != "m102_primary":
        raise ValueError("receipt is not for the M102 primary candidate")
    if payload.get("source_tree_sha256") != expected_source_tree_sha256:
        raise ValueError("receipt source-tree identity does not match this source snapshot")
    if payload.get("production_training_authorized") is not False:
        raise ValueError("canary receipt has an invalid production authorization claim")

    config = payload.get("config")
    aggregate = payload.get("aggregate")
    if not isinstance(config, dict) or not isinstance(aggregate, dict):
        raise ValueError("receipt is missing its config or aggregate object")
    if config.get("candidate") != "m102_primary":
        raise ValueError("receipt config is not for the M102 primary candidate")
    if config.get("source_tree_sha256") != expected_source_tree_sha256:
        raise ValueError("receipt config is not bound to this source snapshot")
    for field, expected in EXPECTED_TPU_WORKLOAD.items():
        actual = config.get(field)
        if type(expected) is int:
            matches = type(actual) is int and actual == expected
        else:
            matches = type(actual) in {int, float} and math.isfinite(actual) and actual == expected
        if not matches:
            raise ValueError(f"receipt config disagrees with the frozen {field} workload")

    if profile == "SMOKE":
        expected_updates, expected_optimizer_updates, expected_restart = 1, 2, True
    elif profile == "QUALIFICATION":
        expected_updates, expected_optimizer_updates, expected_restart = 20, 21, False
    else:
        raise ValueError("unsupported requested receipt profile")
    if (type(config.get("minimum_steady_updates")) is not int
            or config["minimum_steady_updates"] != expected_updates
            or type(config.get("optimizer_updates")) is not int
            or config["optimizer_updates"] != expected_optimizer_updates
            or config.get("verify_restart") is not expected_restart):
        raise ValueError(f"receipt config does not match the frozen {profile.lower()} profile")

    if aggregate.get("candidate") != "m102_primary":
        raise ValueError("aggregate candidate does not match the M102 receipt")
    if aggregate.get("source_tree_sha256") != expected_source_tree_sha256:
        raise ValueError("aggregate source-tree identity does not match this source snapshot")
    if aggregate.get("model_spec_sha256") != MODEL_SPEC.sha256():
        raise ValueError("aggregate model-spec identity does not match M102")
    if (type(aggregate.get("world_size")) is not int
            or aggregate["world_size"] != EXPECTED_TPU_WORKLOAD["expected_world_size"]
            or type(aggregate.get("global_device_count")) is not int
            or aggregate["global_device_count"] != EXPECTED_TPU_WORKLOAD["expected_global_devices"]
            or type(aggregate.get("addressable_device_count_per_worker")) is not int
            or aggregate["addressable_device_count_per_worker"] <= 0):
        raise ValueError("aggregate does not prove the frozen eight-device TPU topology")
    if aggregate.get("participating_ordinals") != list(range(8)):
        raise ValueError("aggregate does not account for every TPU rank")
    for field in ("all_ranks_report_tpu", "all_core_update_verified",
                  "distributed_gradient_parity_verified"):
        if aggregate.get(field) is not True:
            raise ValueError(f"aggregate is missing its {field} proof")
    rank_receipts = payload.get("rank_receipts")
    if (not isinstance(rank_receipts, list) or len(rank_receipts) != 8
            or any(not isinstance(path, str) or not path.strip() for path in rank_receipts)
            or len(set(rank_receipts)) != 8):
        raise ValueError("receipt does not name eight distinct rank receipts")

    runtime_identity = aggregate.get("runtime_identity")
    runtime_fields = ("python_version", "torch_version", "torch_xla_version", "platform")
    if not isinstance(runtime_identity, dict):
        raise ValueError("aggregate is missing its runtime identity")
    metadata: dict[str, str] = {}
    for field in runtime_fields:
        value = runtime_identity.get(field)
        if (not isinstance(value, str) or not value.strip()
                or value.strip().lower() in {"unknown", "unavailable", "n/a", "none"}):
            raise ValueError(f"aggregate runtime identity has no known {field}")
        metadata[field] = value
    identity_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8")
    if runtime_identity.get("sha256") != hashlib.sha256(identity_bytes).hexdigest():
        raise ValueError("aggregate runtime identity hash is invalid")

    if (aggregate.get("measurement_profile") != profile
            or type(aggregate.get("measured_steady_update_count")) is not int
            or aggregate["measured_steady_update_count"] != expected_updates
            or type(aggregate.get("minimum_steady_updates_required")) is not int
            or aggregate["minimum_steady_updates_required"] != expected_updates
            or type(aggregate.get("optimizer_step")) is not int
            or aggregate["optimizer_step"] != expected_optimizer_updates):
        raise ValueError(f"aggregate measurements do not match the frozen {profile.lower()} profile")
    timings = aggregate.get("steady_critical_path_seconds_by_update")
    if (not isinstance(timings, list) or len(timings) != expected_updates
            or any(not _finite_positive(value) for value in timings)
            or not _finite_positive(aggregate.get("steady_global_tokens_per_second"))):
        raise ValueError("aggregate timing evidence is missing or invalid")

    host_memory = aggregate.get("host_memory_observation")
    rank_keys = {str(rank) for rank in range(8)}
    if (not isinstance(host_memory, dict)
            or host_memory.get("schema") != HOST_MEMORY_SCHEMA
            or host_memory.get("metric") != HOST_MEMORY_METRIC
            or host_memory.get("scope") != HOST_MEMORY_SCOPE
            or host_memory.get("unit") != "bytes"
            or host_memory.get("sample_point") != HOST_MEMORY_FINAL_HASH_SAMPLE_POINT
            or type(host_memory.get("update_boundary_sample_series_aligned")) is not bool
            or host_memory.get("is_qualification_gate") is not False
            or host_memory.get("status") not in {"OBSERVED", "PARTIAL", "UNAVAILABLE"}):
        raise ValueError("aggregate host RSS observation is missing or has invalid semantics")
    host_rss_by_rank = host_memory.get("high_water_rss_bytes_by_rank")
    if (not isinstance(host_rss_by_rank, dict)
            or not set(host_rss_by_rank).issubset(rank_keys)
            or any(type(value) is not int or value <= 0
                   for value in host_rss_by_rank.values())):
        raise ValueError("aggregate host RSS observation has invalid per-rank byte counters")
    if (host_memory["status"] == "OBSERVED" and set(host_rss_by_rank) != rank_keys
            or host_memory["status"] == "PARTIAL" and not host_rss_by_rank
            or host_memory["status"] == "UNAVAILABLE" and host_rss_by_rank):
        raise ValueError("aggregate host RSS status disagrees with its per-rank counters")

    if profile == "SMOKE":
        if aggregate.get("peak_memory_status") not in {"NOT_REQUIRED", "OBSERVED_NOT_QUALIFIED"}:
            raise ValueError("smoke receipt reports an inconsistent memory profile")
        restart = payload.get("controlled_restart")
        required_restart_flags = (
            "parameters_match_uninterrupted", "optimizer_moments_match_uninterrupted",
            "rank_local_sample_streams_match_uninterrupted",
            "rank_local_rng_states_match_uninterrupted",
            "rank_local_synthetic_cursors_match_uninterrupted",
            "rank_local_xla_rng_replay_matches_uninterrupted",
        )
        if (not isinstance(restart, dict) or restart.get("schema") != RESTART_SCHEMA
                or restart.get("status") != "PASS"
                or restart.get("source_tree_sha256") != expected_source_tree_sha256
                or restart.get("runtime_identity") != runtime_identity
                or restart.get("world_size") != 8
                or restart.get("restored_optimizer_step") != 1
                or restart.get("final_optimizer_step") != expected_optimizer_updates
                or restart.get("production_exact_resume_certified") is not False
                or not _is_sha256(restart.get("checkpoint_sha256"))
                or any(restart.get(field) is not True for field in required_restart_flags)):
            raise ValueError("smoke receipt lacks the passing bounded restart evidence")
    else:
        if payload.get("controlled_restart") is not None:
            raise ValueError("qualification receipt must not claim the smoke restart profile")
        peak_fraction = aggregate.get("worst_peak_memory_fraction")
        peak_limit = aggregate.get("peak_memory_limit_fraction")
        peak_by_rank = aggregate.get("peak_memory_bytes_by_rank")
        limit_by_rank = aggregate.get("memory_limit_bytes_by_rank")
        rank_keys = {str(rank) for rank in range(8)}
        if (aggregate.get("peak_memory_status") != "PASS"
                or type(peak_fraction) not in {int, float}
                or not math.isfinite(peak_fraction)
                or not 0 < peak_fraction <= QUALIFICATION_MAX_PEAK_MEMORY_FRACTION
                or peak_limit != QUALIFICATION_MAX_PEAK_MEMORY_FRACTION
                or not isinstance(peak_by_rank, dict) or set(peak_by_rank) != rank_keys
                or not isinstance(limit_by_rank, dict) or set(limit_by_rank) != rank_keys):
            raise ValueError("qualification receipt does not pass the 85% memory headroom gate")
        ratios = []
        for rank in sorted(rank_keys, key=int):
            peak, limit = peak_by_rank[rank], limit_by_rank[rank]
            if type(peak) is not int or type(limit) is not int or peak <= 0 or limit <= 0:
                raise ValueError("qualification receipt has invalid per-rank memory counters")
            ratios.append(peak / limit)
        if not math.isclose(max(ratios), peak_fraction, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("qualification receipt memory summary disagrees with rank counters")

    return {
        "state": "PRESENT_FOR_REVIEW",
        "profile": profile,
        "candidate": "m102_primary",
        "source_tree_sha256": expected_source_tree_sha256,
        "runtime_identity_sha256": runtime_identity["sha256"],
        "measured_steady_update_count": expected_updates,
        "host_memory_observation": host_memory,
        "note": "Synthetic Kaggle evidence only; it does not satisfy production runtime gates or authorize training.",
    }


def _runtime_receipt_gate(path: Path | None, *, profile: str,
                          expected_source_tree_sha256: str | None) -> dict[str, Any]:
    if path is None:
        return {"state": "NOT_SUPPLIED", "profile": profile}
    if not path.is_file():
        return {"state": "BLOCKED", "profile": profile,
                "reason": "Canary receipt path does not exist.", "path": str(path)}
    if expected_source_tree_sha256 is None:
        return {"state": "BLOCKED", "profile": profile,
                "reason": "Current Signac source identity could not be computed.",
                "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        review = _validate_runtime_receipt(
            payload, profile=profile,
            expected_source_tree_sha256=expected_source_tree_sha256,
        )
        rank_references = payload["rank_receipts"]
        rank_names = [PurePosixPath(reference.replace("\\", "/")).name
                      for reference in rank_references]
        expected_names = [f"rank-{ordinal:02d}.json" for ordinal in range(8)]
        if rank_names != expected_names:
            raise ValueError("rank receipt references must name rank-00.json through rank-07.json in order")
        rank_rows: list[dict[str, Any]] = []
        rank_evidence: list[dict[str, Any]] = []
        for ordinal, name in enumerate(expected_names):
            rank_path = path.parent / name
            rank_payload = json.loads(rank_path.read_text(encoding="utf-8"))
            if not isinstance(rank_payload, dict) or rank_payload.get("ordinal") != ordinal:
                raise ValueError(f"rank receipt {name} is missing its matching ordinal")
            rank_rows.append(rank_payload)
            rank_evidence.append({"ordinal": ordinal, "path": str(rank_path.resolve()),
                                  "sha256": _sha256(rank_path)})
        from v5_training.kaggle_tpu_canary import aggregate_rank_receipts

        expected_optimizer_step = 2 if profile == "SMOKE" else 21
        recomputed_aggregate = aggregate_rank_receipts(
            rank_rows, expected_world_size=8, expected_global_devices=8,
            expected_optimizer_step=expected_optimizer_step,
        )
        if recomputed_aggregate != payload["aggregate"]:
            raise ValueError("embedded aggregate does not match the complete per-rank receipts")
        digest = _sha256(path)
        resolved_path = str(path.resolve())
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError,
            AttributeError, OverflowError) as exc:
        return {"state": "BLOCKED", "profile": profile,
                "reason": f"Canary receipt failed validation: {exc}", "path": str(path)}
    return {**review, "path": resolved_path, "sha256": digest,
            "rank_receipts": rank_evidence}


def build_report(*, data_manifest: Path | None = None,
                 evaluation_receipt: Path | None = None,
                 runtime_receipt: Path | None = None,
                 qualification_receipt: Path | None = None,
                 target: str = "cpu") -> dict[str, Any]:
    if target not in {"cpu", "cuda", "tpu"}:
        raise ValueError("target must be one of: cpu, cuda, tpu")
    if target != "tpu" and (runtime_receipt is not None or qualification_receipt is not None):
        raise ValueError("Kaggle TPU receipts may only be reviewed with target='tpu'")
    MODEL_SPEC.assert_valid()
    receipts = MODEL_SPEC.parameter_receipt()
    data = _manifest_gate(data_manifest)
    evaluation = _evaluation_receipt_gate(evaluation_receipt)
    runtime = {"target": target, "state": "NOT_PROBED",
               "note": "A static report cannot certify Kaggle hardware, XLA parity, memory fit, or durable resume."}
    if target == "tpu":
        runtime["state"] = "TPU_EVIDENCE_REQUIRED"
        source_tree_sha256 = None
        if runtime_receipt is not None or qualification_receipt is not None:
            try:
                from signac_100m.source_identity import build_source_identity

                source_tree_sha256 = build_source_identity(REPO_ROOT)["source_tree_sha256"]
            except (OSError, ValueError) as exc:
                runtime["receipt_review_error"] = str(exc)
        canary_evidence = _runtime_receipt_gate(
            runtime_receipt, profile="SMOKE",
            expected_source_tree_sha256=source_tree_sha256,
        )
        qualification_evidence = _runtime_receipt_gate(
            qualification_receipt, profile="QUALIFICATION",
            expected_source_tree_sha256=source_tree_sha256,
        )
        if (canary_evidence.get("state") == "PRESENT_FOR_REVIEW"
                and qualification_evidence.get("state") == "PRESENT_FOR_REVIEW"
                and canary_evidence.get("runtime_identity_sha256")
                != qualification_evidence.get("runtime_identity_sha256")):
            qualification_evidence = {
                "state": "BLOCKED",
                "profile": "QUALIFICATION",
                "path": qualification_evidence.get("path"),
                "sha256": qualification_evidence.get("sha256"),
                "reason": "Qualification runtime identity differs from the M102 smoke receipt.",
            }
        runtime["canary_evidence"] = canary_evidence
        runtime["qualification_evidence"] = qualification_evidence
        runtime["production_gate_note"] = (
            "Canary receipts are synthetic supporting evidence. Production G1-G4 still require "
            "the declared workload, production update parity, exact production resume, and durable-output verification."
        )
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
    parser.add_argument("--runtime-receipt", type=Path,
                        help="M102 Kaggle SMOKE all-core aggregate receipt for review only")
    parser.add_argument("--qualification-receipt", type=Path,
                        help="M102 Kaggle 20-update QUALIFICATION aggregate receipt for review only")
    parser.add_argument("--target", choices=("cpu", "cuda", "tpu"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(data_manifest=args.data_manifest,
                          evaluation_receipt=args.evaluation_receipt,
                          runtime_receipt=args.runtime_receipt,
                          qualification_receipt=args.qualification_receipt,
                          target=args.target)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["verdict"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
