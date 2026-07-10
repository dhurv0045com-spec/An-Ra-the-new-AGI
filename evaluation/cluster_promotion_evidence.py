"""Build signed promotion evidence for the external GPU control plane."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from anra.anra_paths import OUTPUT_V2_DIR

from evaluation.promotion import verify_release_manifest

PROMOTION_EVIDENCE_SCHEMA_VERSION = 1
REQUIRED_PROMOTION_GATES = (
    "gate6_evaluation",
    "reproducibility",
    "adversarial_audit",
    "rollback_drill",
)
DEFAULT_OUTPUT = OUTPUT_V2_DIR / "cluster_promotion_evidence.json"


def _canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(report: Mapping[str, object]) -> tuple[int, dict[str, float]]:
    sample_count = report.get("sample_count")
    raw_metrics = report.get("metrics")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or not isinstance(raw_metrics, Mapping)
        or not raw_metrics
    ):
        raise ValueError("Gate 6 report requires a non-empty finite metric set")
    metrics = {str(name): float(value) for name, value in raw_metrics.items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Gate 6 metrics must be finite")
    return sample_count, metrics


def build_cluster_promotion_evidence(
    *,
    checkpoint_path: str | Path,
    source_commit: str,
    gate6_report: Mapping[str, object],
    reproducibility_reports: Sequence[Mapping[str, object]],
    adversarial_report: Mapping[str, object],
    rollback_report: Mapping[str, object],
    output_path: str | Path | None = None,
    signing_key: str | None = None,
) -> dict[str, object]:
    """Validate local artifacts and emit the cluster's signed P4 envelope."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not source_commit or source_commit == "unknown":
        raise ValueError("a concrete source commit is required")
    checkpoint_sha256 = _file_hash(checkpoint)
    if gate6_report.get("passed") is not True:
        raise ValueError("Gate 6 evaluation did not pass")
    if gate6_report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Gate 6 report checkpoint lineage does not match")
    sample_count, metrics = _metrics(gate6_report)

    if len(reproducibility_reports) != 2:
        raise ValueError("G-C6 requires exactly two reproducibility reruns")
    first, second = (dict(report) for report in reproducibility_reports)
    manifest_hash = str(first.get("job_manifest_hash", ""))
    first_seed = first.get("seed")
    if (
        len(manifest_hash) != 64
        or second.get("job_manifest_hash") != manifest_hash
        or isinstance(first_seed, bool)
        or not isinstance(first_seed, int)
        or second.get("seed") != first_seed
    ):
        raise ValueError("G-C6 reruns must share one manifest hash and seed")
    first_metrics = first.get("metrics")
    second_metrics = second.get("metrics")
    if (
        not isinstance(first_metrics, Mapping)
        or not first_metrics
        or not isinstance(second_metrics, Mapping)
        or not second_metrics
    ):
        raise ValueError("G-C6 reruns require metric mappings")
    try:
        normalized_first = {
            str(name): float(value) for name, value in first_metrics.items()
        }
        normalized_second = {
            str(name): float(value) for name, value in second_metrics.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("G-C6 metrics must be numeric") from exc
    if not all(
        math.isfinite(value)
        for value in (*normalized_first.values(), *normalized_second.values())
    ):
        raise ValueError("G-C6 metrics must be finite")
    first_metrics_hash = _hash(normalized_first)
    second_metrics_hash = _hash(normalized_second)
    reproducible = hmac.compare_digest(first_metrics_hash, second_metrics_hash)
    if not reproducible:
        raise ValueError("G-C6 evaluation metrics are not bit-exact")

    if adversarial_report.get("passed") is not True:
        raise ValueError("adversarial promotion audit did not pass")
    rollback_payload = dict(rollback_report)
    if rollback_payload.get("passed") is not True or not verify_release_manifest(
        rollback_payload
    ):
        raise ValueError("rollback drill report is missing a valid local signature")

    evidence_hashes = {
        "gate6_evaluation": _hash(dict(gate6_report)),
        "reproducibility": _hash([first, second]),
        "adversarial_audit": _hash(dict(adversarial_report)),
        "rollback_drill": _hash(rollback_payload),
    }
    payload: dict[str, object] = {
        "schema_version": PROMOTION_EVIDENCE_SCHEMA_VERSION,
        "kind": "anra_cluster_promotion_evidence",
        "generated_at": time.time(),
        "checkpoint_sha256": checkpoint_sha256,
        "source_commit": source_commit,
        "gates": dict.fromkeys(REQUIRED_PROMOTION_GATES, True),
        "evidence_hashes": evidence_hashes,
        "evaluation": {"sample_count": sample_count, "metrics": metrics},
        "reproducibility": {
            "job_manifest_hash": manifest_hash,
            "seed": first_seed,
            "first_metrics_hash": first_metrics_hash,
            "second_metrics_hash": second_metrics_hash,
            "bit_exact": True,
        },
    }
    key = signing_key or os.environ.get("ANRA_RELEASE_SIGNING_KEY", "")
    if not key:
        raise PermissionError("ANRA_RELEASE_SIGNING_KEY is required")
    payload["signature"] = hmac.new(
        key.encode(), _canonical(payload), hashlib.sha256
    ).hexdigest()
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
    return payload


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build signed cluster promotion evidence")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gate6-report", required=True)
    parser.add_argument("--reproducibility-report", action="append", required=True)
    parser.add_argument("--adversarial-report", required=True)
    parser.add_argument("--rollback-report", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    build_cluster_promotion_evidence(
        checkpoint_path=args.checkpoint,
        source_commit=args.source_commit,
        gate6_report=_read_json(args.gate6_report),
        reproducibility_reports=[
            _read_json(path) for path in args.reproducibility_report
        ],
        adversarial_report=_read_json(args.adversarial_report),
        rollback_report=_read_json(args.rollback_report),
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
