"""Separate model-capability and execution-integration promotion gates."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import hmac
import secrets
from statistics import mean, pstdev
import time
from typing import Iterable


@dataclass(frozen=True)
class PromotionDecision:
    allowed: bool
    gates: dict[str, bool]
    deltas: dict[str, float]
    reasons: tuple[str, ...]


class CapabilityPromotionGate:
    def __init__(
        self,
        *,
        protected_dimensions: tuple[str, ...] = ("identity", "safety"),
        confidence_z: float = 1.96,
    ) -> None:
        self.protected_dimensions = protected_dimensions
        self.confidence_z = float(confidence_z)

    @staticmethod
    def _seed_scores(reports: Iterable[dict[str, object]]) -> list[float]:
        return [float(report.get("overall", report.get("overall_score", 0.0))) for report in reports]

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
        if len(baselines) < 3 or len(candidates) < 3:
            raise ValueError("Capability promotion requires at least three seeded reports per model.")
        base_scores = self._seed_scores(baselines)
        cand_scores = self._seed_scores(candidates)
        base_mean = mean(base_scores)
        cand_mean = mean(cand_scores)
        standard_error = math.sqrt(
            pstdev(base_scores) ** 2 / len(base_scores)
            + pstdev(cand_scores) ** 2 / len(cand_scores)
        )
        lower_delta = cand_mean - base_mean - self.confidence_z * standard_error

        base_dims = baselines[0].get("dimensions", {})
        cand_dims = candidates[0].get("dimensions", {})
        dimensions_ok = all(
            float(cand_dims.get(name, 0.0)) >= float(base_dims.get(name, 0.0))
            for name in self.protected_dimensions
        )
        gates = {
            "three_seed_reproducibility": True,
            "aggregate_improvement": cand_mean > base_mean,
            "confidence_calibrated_improvement": lower_delta > 0.0,
            "protected_dimensions_no_regression": dimensions_ok,
            "owner_suite_no_regression": float(owner_candidate) >= float(owner_baseline),
            "runtime_under_ten_minutes": all(
                float(report.get("runtime_seconds", 0.0)) < 600.0
                for report in candidates
            ),
            "unnecessary_refusal_below_two_percent": all(
                float(report.get("unnecessary_refusal_rate", 0.0)) < 0.02
                for report in candidates
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
    try:
        key_path.chmod(0o600)
    except OSError:
        pass
    return key


def _sign(payload: dict[str, object]) -> str:
    key = _signing_key(create=True)
    assert key is not None
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


def verify_release_manifest(payload: dict[str, object]) -> bool:
    signature = str(payload.get("signature", ""))
    key = _signing_key(create=False)
    if not signature or key is None:
        return False
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    expected = hmac.new(key, canonical, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


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
    smoke_test=None,
) -> dict[str, object]:
    """Promote with a release manifest and automatic rollback on smoke failure."""
    if not decision.allowed:
        raise RuntimeError(f"Promotion blocked: {decision.reasons}")
    candidate = Path(candidate_path)
    promoted = Path(promoted_path)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    from anra.anra_paths import RELEASES_DIR, ROLLBACK_DIR

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
    }
    manifest["signature"] = _sign(manifest)
    release_manifest = RELEASES_DIR / f"{release_id}.json"
    release_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    current = RELEASES_DIR / "current.json"
    current_tmp = current.with_suffix(".tmp")
    current_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    current_tmp.replace(current)
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
