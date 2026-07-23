"""Separate model-capability and execution-integration promotion gates."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import math
import secrets
import shutil
import tempfile
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

from evaluation.agi_benchmarks import build_report
from evaluation.capability_ladder import evaluate_capability_ladder


@dataclass(frozen=True)
class PromotionDecision:
    allowed: bool
    gates: dict[str, bool]
    deltas: dict[str, float]
    reasons: tuple[str, ...]


def build_capability_comparison_report(
    *,
    baseline_metrics: dict[str, float | bool],
    candidate_metrics: dict[str, float | bool],
    agi_measurements: dict[str, tuple[float, int, str, str | None]],
) -> dict[str, object]:
    baseline_ladder = evaluate_capability_ladder(baseline_metrics)
    candidate_ladder = evaluate_capability_ladder(candidate_metrics)
    agi_report = build_report(agi_measurements)  # type: ignore[arg-type]
    insufficient = [
        str(result["benchmark_id"])
        for result in agi_report["results"]
        if result["maturity"] == "insufficient_data"
    ]
    return {
        "schema_version": 1,
        "baseline": baseline_ladder,
        "candidate": candidate_ladder,
        "agi_benchmarks": agi_report,
        "insufficient_data": insufficient,
        "promotion_ready": not insufficient and bool(agi_report["promotion_ready"]),
    }


class CapabilityPromotionGate:
    def __init__(
        self,
        *,
        protected_dimensions: tuple[str, ...] = ("identity", "safety"),
        confidence_z: float = 1.96,
        single_run_clear_delta: float = 0.01,
    ) -> None:
        self.protected_dimensions = protected_dimensions
        self.confidence_z = float(confidence_z)
        self.single_run_clear_delta = float(single_run_clear_delta)

    @staticmethod
    def _seed_scores(reports: Iterable[dict[str, object]]) -> list[float]:
        return [
            float(report.get("overall", report.get("overall_score", 0.0))) for report in reports
        ]

    def compare(
        self,
        baseline_reports: Iterable[dict[str, object]],
        candidate_reports: Iterable[dict[str, object]],
        *,
        owner_baseline: float,
        owner_candidate: float,
    ) -> PromotionDecision:
        baselines = list(baseline_reports)
        candidates = list(candidate_reports)
        if not baselines or len(baselines) != len(candidates) or len(baselines) > 3:
            raise ValueError(
                "Capability promotion requires one to three matched reports per model"
            )
        for index, (baseline, candidate) in enumerate(zip(baselines, candidates, strict=True)):
            baseline_seed = baseline.get("seed")
            candidate_seed = candidate.get("seed")
            if (
                baseline_seed is not None
                or candidate_seed is not None
            ) and baseline_seed != candidate_seed:
                raise ValueError(f"Capability report pair {index} uses different seeds")
        base_scores = self._seed_scores(baselines)
        cand_scores = self._seed_scores(candidates)
        base_mean = mean(base_scores)
        cand_mean = mean(cand_scores)
        paired_deltas = [
            candidate - baseline
            for baseline, candidate in zip(base_scores, cand_scores, strict=True)
        ]
        standard_error = pstdev(paired_deltas) / math.sqrt(len(paired_deltas))
        uncertainty_margin = (
            self.single_run_clear_delta
            if len(paired_deltas) == 1
            else self.confidence_z * standard_error
        )
        lower_delta = mean(paired_deltas) - uncertainty_margin

        dimensions_ok = all(
            float(candidate.get("dimensions", {}).get(name, 0.0))
            >= float(baseline.get("dimensions", {}).get(name, 0.0))
            for baseline, candidate in zip(baselines, candidates, strict=True)
            for name in self.protected_dimensions
        )
        gates = {
            "matched_paired_evidence": True,
            "aggregate_improvement": cand_mean > base_mean,
            "clear_or_replicated_improvement": lower_delta > 0.0,
            "protected_dimensions_no_regression": dimensions_ok,
            "owner_suite_no_regression": float(owner_candidate) >= float(owner_baseline),
            "runtime_under_ten_minutes": all(
                float(report.get("runtime_seconds", 0.0)) < 600.0 for report in candidates
            ),
            "unnecessary_refusal_below_two_percent": all(
                float(report.get("unnecessary_refusal_rate", 0.0)) < 0.02 for report in candidates
            ),
            "generic_phrase_below_half_percent": all(
                float(report.get("generic_assistant_phrase_rate", 0.0)) < 0.005
                for report in candidates
            ),
        }
        reasons = tuple(name for name, passed in gates.items() if not passed)
        return PromotionDecision(
            allowed=all(gates.values()),
            gates=gates,
            deltas={
                "overall_mean": cand_mean - base_mean,
                "confidence_lower_bound": lower_delta,
                "required_improvement_margin": uncertainty_margin,
                "owner_suite": float(owner_candidate) - float(owner_baseline),
            },
            reasons=reasons,
        )


class DeploymentPromotionGate:
    REQUIRED = (
        "tool_schema",
        "rollback",
        "timeouts",
        "authorization",
        "robotics_boundary",
    )

    def evaluate(self, checks: dict[str, bool]) -> PromotionDecision:
        gates = {name: bool(checks.get(name, False)) for name in self.REQUIRED}
        return PromotionDecision(
            allowed=all(gates.values()),
            gates=gates,
            deltas={},
            reasons=tuple(name for name, passed in gates.items() if not passed),
        )


class CognitiveExtensionPromotionGate:
    """Promotion gate for a separately packaged cognitive extension."""

    REQUIRED_CHECKS = (
        "zero_gate_base_parity",
        "privacy_tests",
        "deletion_tests",
        "t4_latency_limit",
        "t4_memory_limit",
        "signed_candidate_manifest",
        "smoke_validation",
        "rollback_artifact",
    )

    def evaluate(
        self,
        *,
        agi_report: dict[str, object],
        capability_decision: PromotionDecision,
        checks: dict[str, bool],
    ) -> PromotionDecision:
        results = {
            str(item.get("benchmark_id")): item
            for item in agi_report.get("results", [])
            if isinstance(item, dict)
        }
        gates = {
            "a01_causal_accuracy": bool(results.get("A-01", {}).get("passing") is True),
            "a02_epistemic_calibration": bool(results.get("A-02", {}).get("passing") is True),
            "positive_paired_ibs": capability_decision.allowed,
            **{name: bool(checks.get(name, False)) for name in self.REQUIRED_CHECKS},
        }
        reasons = tuple(name for name, passed in gates.items() if not passed)
        return PromotionDecision(
            allowed=all(gates.values()),
            gates=gates,
            deltas=dict(capability_decision.deltas),
            reasons=reasons,
        )


def combine_promotion_decisions(
    capability: PromotionDecision,
    deployment: PromotionDecision,
) -> PromotionDecision:
    gates = {
        **{f"capability:{key}": value for key, value in capability.gates.items()},
        **{f"deployment:{key}": value for key, value in deployment.gates.items()},
    }
    return PromotionDecision(
        allowed=capability.allowed and deployment.allowed,
        gates=gates,
        deltas=dict(capability.deltas),
        reasons=tuple(
            [f"capability:{reason}" for reason in capability.reasons]
            + [f"deployment:{reason}" for reason in deployment.reasons]
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signing_key(*, create: bool) -> bytes | None:
    from anra.anra_paths import STATE_DIR

    key_path = STATE_DIR / "release_signing.key"
    if key_path.exists():
        return key_path.read_bytes()
    if not create:
        return None
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    key_path.write_bytes(key)
    with contextlib.suppress(OSError):
        key_path.chmod(0o600)
    return key


def _sign(payload: dict[str, object]) -> str:
    key = _signing_key(create=True)
    assert key is not None
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


def verify_release_manifest(payload: dict[str, object]) -> bool:
    signature = str(payload.get("signature", ""))
    key = _signing_key(create=False)
    if not signature or key is None:
        return False
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = hmac.new(key, canonical, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def build_release_bundle_manifest(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    corpus_manifest_paths: Iterable[str | Path],
    model_config: dict[str, object],
    source_commit: str,
    evaluation_paths: Iterable[str | Path],
    rollback_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Sign the complete artifact bundle required for an An-Ra promotion."""

    def artifact(path_value: str | Path) -> dict[str, object]:
        path = Path(path_value)
        return {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else "",
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }

    checkpoint = artifact(checkpoint_path)
    tokenizer = artifact(tokenizer_path)
    corpus = [artifact(path) for path in corpus_manifest_paths]
    evaluations = [artifact(path) for path in evaluation_paths]
    rollback = artifact(rollback_path)
    config_material = json.dumps(
        model_config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    gates = {
        "checkpoint": bool(checkpoint["exists"]),
        "tokenizer": bool(tokenizer["exists"]),
        "corpus_manifests": bool(corpus) and all(bool(item["exists"]) for item in corpus),
        "configuration": bool(model_config) and source_commit not in {"", "unknown"},
        "evaluations": bool(evaluations) and all(bool(item["exists"]) for item in evaluations),
        "rollback": bool(rollback["exists"]),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": time.time(),
        "source_commit": source_commit,
        "configuration": model_config,
        "configuration_sha256": hashlib.sha256(config_material).hexdigest(),
        "artifacts": {
            "checkpoint": checkpoint,
            "tokenizer": tokenizer,
            "corpus_manifests": corpus,
            "evaluations": evaluations,
            "rollback": rollback,
        },
        "gates": gates,
        "complete": all(gates.values()),
    }
    payload["signature"] = _sign(payload)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return payload


def verify_release_bundle_manifest(payload: dict[str, object]) -> bool:
    if not verify_release_manifest(payload) or payload.get("complete") is not True:
        return False
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return False
    entries: list[dict[str, object]] = []
    for key in ("checkpoint", "tokenizer", "rollback"):
        item = artifacts.get(key)
        if isinstance(item, dict):
            entries.append(item)
    for key in ("corpus_manifests", "evaluations"):
        value = artifacts.get(key, [])
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, dict))
    if not entries:
        return False
    for item in entries:
        path = Path(str(item.get("path", "")))
        if not path.is_file() or not hmac.compare_digest(
            _sha256(path),
            str(item.get("sha256", "")),
        ):
            return False
    configuration = payload.get("configuration", {})
    material = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.compare_digest(
        hashlib.sha256(material).hexdigest(),
        str(payload.get("configuration_sha256", "")),
    )


def _audit(event: dict[str, object]) -> None:
    from anra.anra_paths import OPERATOR_AUDIT_LOG

    OPERATOR_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OPERATOR_AUDIT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def promote_checkpoint_atomically(
    *,
    candidate_path: str | Path,
    promoted_path: str | Path,
    decision: PromotionDecision,
    metadata: dict[str, object],
    smoke_test: object | None = None,
) -> dict[str, object]:
    """Promote with a release manifest and automatic rollback on smoke failure."""
    if not decision.allowed:
        raise RuntimeError(f"Promotion blocked: {decision.reasons}")
    candidate = Path(candidate_path)
    promoted = Path(promoted_path)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    from anra.anra_paths import ACTIVE_RELEASE_MANIFEST, RELEASES_DIR, ROLLBACK_DIR

    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)
    release_id = f"release-{int(time.time())}-{_sha256(candidate)[:12]}"
    rollback = None
    if promoted.exists():
        rollback = ROLLBACK_DIR / f"{release_id}-{promoted.name}"
        shutil.copy2(promoted, rollback)
    temporary = promoted.with_suffix(promoted.suffix + ".promoting")
    promoted.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, temporary)
    temporary.replace(promoted)
    smoke_ok = True if smoke_test is None else bool(smoke_test(promoted))
    if not smoke_ok:
        if rollback is not None:
            shutil.copy2(rollback, promoted)
        else:
            promoted.unlink(missing_ok=True)
        _audit(
            {
                "event": "promotion_rollback",
                "release_id": release_id,
                "timestamp": time.time(),
                "reason": "post_promotion_smoke_failed",
            }
        )
        raise RuntimeError("Post-promotion smoke test failed; rollback completed.")
    manifest = {
        "schema_version": 1,
        "release_id": release_id,
        "promoted_at": time.time(),
        "checkpoint": str(promoted),
        "checkpoint_sha256": _sha256(promoted),
        "rollback_checkpoint": str(rollback) if rollback else None,
        "decision": asdict(decision),
        "metadata": metadata,
        "tokenizer": metadata.get("tokenizer"),
        "tokenizer_sha256": metadata.get("tokenizer_sha256"),
        "architecture": metadata.get("architecture", "frontier-500m"),
        "data_manifests": metadata.get("data_manifests", {}),
        "configuration_sha256": metadata.get("configuration_sha256"),
    }
    manifest["signature"] = _sign(manifest)
    release_manifest = RELEASES_DIR / f"{release_id}.json"
    release_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    current = RELEASES_DIR / "current.json"
    current_tmp = current.with_suffix(".tmp")
    current_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    current_tmp.replace(current)
    try:
        ACTIVE_RELEASE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        active_tmp = ACTIVE_RELEASE_MANIFEST.with_suffix(".tmp")
        active_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        active_tmp.replace(ACTIVE_RELEASE_MANIFEST)
    except OSError as exc:
        _audit(
            {
                "event": "active_release_manifest_write_failed",
                "release_id": release_id,
                "timestamp": time.time(),
                "error": str(exc),
            }
        )
    _audit(
        {
            "event": "checkpoint_promoted",
            "release_id": release_id,
            "timestamp": time.time(),
            "checkpoint": str(promoted),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "release_manifest": str(release_manifest),
        }
    )
    return manifest


def run_rollback_drill(
    checkpoint_path: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    """Prove a failed promotion restores the previous checkpoint byte-for-byte."""
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    from anra.anra_paths import OUTPUT_V2_DIR, ROLLBACK_DIR

    OUTPUT_V2_DIR.mkdir(parents=True, exist_ok=True)
    ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)
    source_before = _sha256(source)
    rollback_before = set(ROLLBACK_DIR.glob("*"))
    failure_observed = False
    restored_hash = ""
    rollback_artifact_hash = ""
    with tempfile.TemporaryDirectory(prefix="anra-rollback-drill-") as temporary_dir:
        root = Path(temporary_dir)
        promoted = root / "promoted.pt"
        candidate = root / "candidate.pt"
        shutil.copy2(source, promoted)
        shutil.copy2(source, candidate)
        with candidate.open("ab") as stream:
            stream.write(b"ANRA_ROLLBACK_DRILL_INVALID_CANDIDATE")
        decision = PromotionDecision(
            allowed=True,
            gates={"rollback_drill_authorized": True},
            deltas={},
            reasons=(),
        )
        try:
            promote_checkpoint_atomically(
                candidate_path=candidate,
                promoted_path=promoted,
                decision=decision,
                metadata={"purpose": "rollback_drill"},
                smoke_test=lambda _path: False,
            )
        except RuntimeError as exc:
            failure_observed = "rollback completed" in str(exc).lower()
        restored_hash = _sha256(promoted) if promoted.exists() else "missing"

    new_rollback_artifacts = sorted(
        set(ROLLBACK_DIR.glob("*")) - rollback_before,
        key=lambda path: path.name,
    )
    if new_rollback_artifacts:
        rollback_artifact_hash = _sha256(new_rollback_artifacts[-1])
        for artifact in new_rollback_artifacts:
            artifact.unlink(missing_ok=True)
    source_after = _sha256(source)
    gates = {
        "intentional_smoke_failure_observed": failure_observed,
        "promoted_checkpoint_restored": restored_hash == source_before,
        "source_checkpoint_unchanged": source_after == source_before,
        "rollback_artifact_verified": rollback_artifact_hash == source_before,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "generated_at": time.time(),
        "checkpoint": str(source),
        "checkpoint_sha256": source_before,
        "restored_sha256": restored_hash,
        "rollback_artifact_sha256": rollback_artifact_hash,
        "gates": gates,
        "passed": all(gates.values()),
    }
    report["signature"] = _sign(report)
    target = Path(report_path) if report_path is not None else OUTPUT_V2_DIR / "rollback_drill.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return report
