"""Consent and encrypted JSON persistence for sensitive cognitive state."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class EncryptionUnavailable(RuntimeError):
    pass


class SensitiveStateStore:
    def __init__(self, root: str | Path, *, key: str | bytes | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._raw_key = key or os.environ.get("ANRA_OWNER_STATE_KEY", "")

    @property
    def available(self) -> bool:
        if not self._raw_key:
            return False
        try:
            import cryptography.fernet  # noqa: F401
        except ImportError:
            return False
        return True

    def _fernet(self):
        if not self.available:
            raise EncryptionUnavailable(
                "Sensitive persistence requires ANRA_OWNER_STATE_KEY and cryptography."
            )
        from cryptography.fernet import Fernet

        raw = self._raw_key.encode("utf-8") if isinstance(self._raw_key, str) else self._raw_key
        key = base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
        return Fernet(key)

    def write(self, name: str, payload: dict[str, Any]) -> Path:
        target = self.root / f"{name}.json.enc"
        encrypted = self._fernet().encrypt(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(encrypted)
        temporary.replace(target)
        return target

    def read(self, name: str) -> dict[str, Any] | None:
        target = self.root / f"{name}.json.enc"
        if not target.exists():
            return None
        payload = json.loads(self._fernet().decrypt(target.read_bytes()).decode("utf-8"))
        return payload if isinstance(payload, dict) else None

    def delete(self, name: str) -> None:
        (self.root / f"{name}.json.enc").unlink(missing_ok=True)

    def wipe(self) -> int:
        removed = 0
        for path in self.root.glob("*.enc"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed
