"""Content-addressed adapter hot-load registry with lineage checks."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock

from anra.extensions import (
    CapabilityAdapterSpec,
    detach_candidate_adapters,
    load_capability_adapter,
)
from torch import nn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_sha256(value: object, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


@dataclass(frozen=True)
class AdapterArtifact:
    adapter_id: str
    path: str
    sha256: str
    base_checkpoint_hash: str
    tokenizer_hash: str
    registered_at: float


@dataclass(frozen=True)
class AdapterPromotionEvidence:
    """Evidence needed before a candidate may become a serving capability."""

    adapter_id: str
    adapter_sha256: str
    evaluation_sha256: str
    rollback_sha256: str
    baseline_score: float
    candidate_score: float
    protected_regression: float
    rollback_target_adapter_id: str | None
    promoted_at: float


class AdapterRegistry:
    """Register adapters by digest and activate only matching base lineage."""

    def __init__(self) -> None:
        self._artifacts: dict[str, AdapterArtifact] = {}
        self._promotions: dict[str, AdapterPromotionEvidence] = {}
        self._active: str | None = None
        self._lock = RLock()

    def register(
        self,
        *,
        adapter_id: str,
        path: str | Path,
        base_checkpoint_hash: str,
        tokenizer_hash: str,
    ) -> AdapterArtifact:
        artifact_path = Path(path)
        if not adapter_id or not base_checkpoint_hash or not tokenizer_hash:
            raise ValueError("adapter id, base checkpoint hash, and tokenizer hash are required")
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        artifact = AdapterArtifact(
            adapter_id=adapter_id,
            path=str(artifact_path),
            sha256=_sha256(artifact_path),
            base_checkpoint_hash=base_checkpoint_hash,
            tokenizer_hash=tokenizer_hash,
            registered_at=time.time(),
        )
        with self._lock:
            existing = self._artifacts.get(adapter_id)
            if existing is not None and existing.sha256 != artifact.sha256:
                raise ValueError("adapter_id already refers to different content")
            self._artifacts[adapter_id] = artifact
        return artifact

    def activate(
        self,
        adapter_id: str | None,
        *,
        base_checkpoint_hash: str,
        tokenizer_hash: str,
    ) -> AdapterArtifact | None:
        with self._lock:
            if adapter_id is None:
                self._active = None
                return None
            artifact = self._artifacts[adapter_id]
            if artifact.base_checkpoint_hash != base_checkpoint_hash:
                raise ValueError("adapter base checkpoint hash does not match serving model")
            if artifact.tokenizer_hash != tokenizer_hash:
                raise ValueError("adapter tokenizer hash does not match serving tokenizer")
            if _sha256(Path(artifact.path)) != artifact.sha256:
                raise ValueError("adapter content changed after registration")
            self._active = adapter_id
            return artifact

    def provenance(self) -> dict[str, object]:
        with self._lock:
            active = self._artifacts.get(self._active) if self._active else None
            return {
                "active_adapter_id": self._active,
                "active_adapter": asdict(active) if active else None,
                "registered_adapters": len(self._artifacts),
                "promoted_adapters": len(self._promotions),
                "active_promotion": (
                    asdict(self._promotions[self._active])
                    if self._active in self._promotions
                    else None
                ),
            }

    def promote(
        self,
        adapter_id: str,
        *,
        evaluation: Mapping[str, object],
        rollback: Mapping[str, object],
        max_protected_regression: float = 0.02,
    ) -> AdapterPromotionEvidence:
        """Promote a candidate only after comparative evaluation and rollback rehearsal."""

        with self._lock:
            artifact = self._artifacts[adapter_id]
        if _sha256(Path(artifact.path)) != artifact.sha256:
            raise ValueError("adapter content changed before promotion")
        if evaluation.get("passed") is not True:
            raise PermissionError("adapter evaluation has not passed")
        baseline = evaluation.get("baseline_score")
        candidate = evaluation.get("candidate_score")
        regression = evaluation.get("protected_regression")
        numeric = (baseline, candidate, regression)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("adapter evaluation scores must be finite numbers")
        baseline_score, candidate_score, protected_regression = map(float, numeric)
        if candidate_score <= baseline_score:
            raise PermissionError("adapter did not beat its immutable base")
        if not 0.0 <= protected_regression <= max_protected_regression:
            raise PermissionError("adapter exceeds the protected regression budget")
        if evaluation.get("adapter_sha256") != artifact.sha256:
            raise ValueError("adapter evaluation is bound to different content")
        if evaluation.get("base_checkpoint_hash") != artifact.base_checkpoint_hash:
            raise ValueError("adapter evaluation is bound to a different base checkpoint")
        _required_sha256(evaluation.get("suite_sha256"), "evaluation suite hash")

        if rollback.get("passed") is not True or rollback.get("detach_restores_base") is not True:
            raise PermissionError("adapter rollback rehearsal has not passed")
        if rollback.get("adapter_sha256") != artifact.sha256:
            raise ValueError("rollback rehearsal is bound to different adapter content")
        if rollback.get("base_checkpoint_hash") != artifact.base_checkpoint_hash:
            raise ValueError("rollback rehearsal is bound to a different base checkpoint")
        _required_sha256(rollback.get("rehearsal_sha256"), "rollback rehearsal hash")
        raw_target = rollback.get("rollback_target_adapter_id")
        rollback_target = str(raw_target) if raw_target is not None else None
        with self._lock:
            if rollback_target is not None:
                target = self._artifacts.get(rollback_target)
                if target is None:
                    raise ValueError("rollback target adapter is not registered")
                if (
                    target.base_checkpoint_hash != artifact.base_checkpoint_hash
                    or target.tokenizer_hash != artifact.tokenizer_hash
                ):
                    raise ValueError("rollback target does not share the immutable base lineage")
            evidence = AdapterPromotionEvidence(
                adapter_id=adapter_id,
                adapter_sha256=artifact.sha256,
                evaluation_sha256=_evidence_sha256(dict(evaluation)),
                rollback_sha256=_evidence_sha256(dict(rollback)),
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                protected_regression=protected_regression,
                rollback_target_adapter_id=rollback_target,
                promoted_at=time.time(),
            )
            existing = self._promotions.get(adapter_id)
            if existing is not None and (
                existing.adapter_sha256 != evidence.adapter_sha256
                or existing.evaluation_sha256 != evidence.evaluation_sha256
                or existing.rollback_sha256 != evidence.rollback_sha256
            ):
                raise ValueError("adapter promotion evidence is immutable")
            self._promotions[adapter_id] = existing or evidence
            return self._promotions[adapter_id]

    def activate_on_model(
        self,
        adapter_id: str | None,
        model: nn.Module,
        *,
        base_model_profile: str,
        base_checkpoint_hash: str,
        tokenizer_hash: str,
    ) -> CapabilityAdapterSpec | None:
        """Strictly attach or remove a registered parameter-efficient capability."""

        artifact = self.activate(
            adapter_id,
            base_checkpoint_hash=base_checkpoint_hash,
            tokenizer_hash=tokenizer_hash,
        )
        if artifact is None:
            detach_candidate_adapters(model)
            return None
        try:
            return load_capability_adapter(
                model,
                artifact.path,
                expected_base_model_profile=base_model_profile,
                expected_base_checkpoint_sha256=base_checkpoint_hash,
                expected_tokenizer_sha256=tokenizer_hash,
            )
        except Exception:
            with self._lock:
                self._active = None
            detach_candidate_adapters(model)
            raise

    def activate_promoted_on_model(
        self,
        adapter_id: str,
        model: nn.Module,
        *,
        base_model_profile: str,
        base_checkpoint_hash: str,
        tokenizer_hash: str,
    ) -> CapabilityAdapterSpec:
        """Activate the production path and restore the prior state on failure."""

        with self._lock:
            promotion = self._promotions.get(adapter_id)
            previous = self._active
        if promotion is None:
            raise PermissionError("adapter has no evaluation and rollback promotion evidence")
        artifact = self._artifacts[adapter_id]
        if promotion.adapter_sha256 != artifact.sha256:
            raise ValueError("adapter promotion no longer matches registered content")
        try:
            spec = self.activate_on_model(
                adapter_id,
                model,
                base_model_profile=base_model_profile,
                base_checkpoint_hash=base_checkpoint_hash,
                tokenizer_hash=tokenizer_hash,
            )
        except Exception:
            if previous is not None and previous != adapter_id:
                self.activate_on_model(
                    previous,
                    model,
                    base_model_profile=base_model_profile,
                    base_checkpoint_hash=base_checkpoint_hash,
                    tokenizer_hash=tokenizer_hash,
                )
            raise
        if spec is None:  # pragma: no cover - adapter_id is non-null by contract.
            raise RuntimeError("promoted adapter activation unexpectedly detached the model")
        return spec

    def rollback_on_model(
        self,
        model: nn.Module,
        *,
        base_model_profile: str,
        base_checkpoint_hash: str,
        tokenizer_hash: str,
    ) -> CapabilityAdapterSpec | None:
        """Execute the rehearsed rollback target for the currently active adapter."""

        with self._lock:
            active = self._active
            promotion = self._promotions.get(active) if active else None
        if active is None or promotion is None:
            raise PermissionError("active adapter has no rehearsed rollback evidence")
        target = promotion.rollback_target_adapter_id
        if target is None:
            return self.activate_on_model(
                None,
                model,
                base_model_profile=base_model_profile,
                base_checkpoint_hash=base_checkpoint_hash,
                tokenizer_hash=tokenizer_hash,
            )
        return self.activate_on_model(
            target,
            model,
            base_model_profile=base_model_profile,
            base_checkpoint_hash=base_checkpoint_hash,
            tokenizer_hash=tokenizer_hash,
        )
