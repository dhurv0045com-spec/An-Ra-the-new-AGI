"""Content-addressed adapter hot-load registry with lineage checks."""

from __future__ import annotations

import hashlib
import time
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


@dataclass(frozen=True)
class AdapterArtifact:
    adapter_id: str
    path: str
    sha256: str
    base_checkpoint_hash: str
    tokenizer_hash: str
    registered_at: float


class AdapterRegistry:
    """Register adapters by digest and activate only matching base lineage."""

    def __init__(self) -> None:
        self._artifacts: dict[str, AdapterArtifact] = {}
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
            }

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
