from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest

from evaluation.cluster_promotion_evidence import build_cluster_promotion_evidence
from evaluation.promotion import run_rollback_drill


def _verify_cluster_signature(payload: dict[str, object], key: str) -> bool:
    signature = str(payload["signature"])
    unsigned = {name: value for name, value in payload.items() if name != "signature"}
    encoded = json.dumps(unsigned, separators=(",", ":"), sort_keys=True).encode()
    expected = hmac.new(key.encode(), encoded, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def test_build_cluster_promotion_evidence_is_lineage_bound_and_signed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anra.anra_paths as paths

    monkeypatch.setattr(paths, "OUTPUT_V2_DIR", tmp_path / "output")
    monkeypatch.setattr(paths, "ROLLBACK_DIR", tmp_path / "rollback")
    monkeypatch.setattr(paths, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(paths, "OPERATOR_AUDIT_LOG", tmp_path / "audit.jsonl")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rollback = run_rollback_drill(
        checkpoint,
        report_path=tmp_path / "rollback_report.json",
    )
    gate6 = {
        "passed": True,
        "checkpoint_sha256": checkpoint_hash,
        "sample_count": 200,
        "metrics": {"capability": 0.8, "safety": 1.0},
    }
    rerun = {
        "job_manifest_hash": "a" * 64,
        "seed": 1301,
        "metrics": {"capability": 0.8, "safety": 1.0},
    }

    evidence = build_cluster_promotion_evidence(
        checkpoint_path=checkpoint,
        source_commit="abcdef123",
        gate6_report=gate6,
        reproducibility_reports=[rerun, dict(rerun)],
        adversarial_report={"passed": True, "cases": 20},
        rollback_report=rollback,
        output_path=tmp_path / "cluster_evidence.json",
        signing_key="cluster-release-key",
    )

    assert evidence["checkpoint_sha256"] == checkpoint_hash
    assert evidence["gates"] == {
        "gate6_evaluation": True,
        "reproducibility": True,
        "adversarial_audit": True,
        "rollback_drill": True,
    }
    assert _verify_cluster_signature(evidence, "cluster-release-key")


def test_cluster_promotion_evidence_rejects_non_reproducible_metrics(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    first = {"job_manifest_hash": "a" * 64, "seed": 1, "metrics": {"score": 1.0}}
    second = {**first, "metrics": {"score": 0.9}}

    with pytest.raises(ValueError, match="not bit-exact"):
        build_cluster_promotion_evidence(
            checkpoint_path=checkpoint,
            source_commit="abcdef123",
            gate6_report={
                "passed": True,
                "checkpoint_sha256": checkpoint_hash,
                "sample_count": 1,
                "metrics": {"score": 1.0},
            },
            reproducibility_reports=[first, second],
            adversarial_report={"passed": True},
            rollback_report={"passed": False},
            signing_key="key",
        )


def test_cluster_promotion_evidence_rejects_non_finite_reproducibility(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rerun = {
        "job_manifest_hash": "a" * 64,
        "seed": 1,
        "metrics": {"score": float("inf")},
    }

    with pytest.raises(ValueError, match="must be finite"):
        build_cluster_promotion_evidence(
            checkpoint_path=checkpoint,
            source_commit="abcdef123",
            gate6_report={
                "passed": True,
                "checkpoint_sha256": checkpoint_hash,
                "sample_count": 1,
                "metrics": {"score": 1.0},
            },
            reproducibility_reports=[rerun, dict(rerun)],
            adversarial_report={"passed": True},
            rollback_report={"passed": False},
            signing_key="key",
        )
