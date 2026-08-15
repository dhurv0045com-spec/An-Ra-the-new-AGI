"""Durable, privacy-aware tool trajectory records."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from anra.anra_paths import TRAJECTORY_STORE


@dataclass(frozen=True)
class TrajectoryRecord:
    goal: str
    mission_tree: dict[str, object]
    skill_sequence: tuple[dict[str, object], ...]
    artifacts: tuple[str, ...]
    success: bool
    verified: bool
    verification_method: str
    verification_evidence: dict[str, object]
    checkpoint_id: str
    tokenizer_id: str
    privacy_status: str
    timestamp: float
    content_hash: str
    approved_constraints: tuple[str, ...] = ()
    tool_results: tuple[dict[str, object], ...] = ()


class TrajectoryStore:
    def __init__(self, path: str | Path = TRAJECTORY_STORE) -> None:
        self.path = Path(path)

    @staticmethod
    def _redact(value: object) -> object:
        if isinstance(value, dict):
            return {
                str(key): (
                    "[REDACTED]"
                    if any(
                        token in str(key).lower()
                        for token in ("token", "secret", "password", "key")
                    )
                    else TrajectoryStore._redact(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [TrajectoryStore._redact(item) for item in value]
        return value

    def append(
        self,
        *,
        goal: str,
        mission_tree: dict[str, object],
        skill_sequence: list[dict[str, object]],
        artifacts: list[str],
        success: bool,
        verified: bool,
        verification_method: str,
        verification_evidence: dict[str, object],
        checkpoint_id: str = "",
        tokenizer_id: str = "",
        approved_constraints: tuple[str, ...] = (),
        tool_results: list[dict[str, object]] | None = None,
    ) -> TrajectoryRecord:
        sanitized_skills = self._redact(skill_sequence)
        body = {
            "goal": goal,
            "mission_tree": mission_tree,
            "skill_sequence": sanitized_skills,
            "artifacts": artifacts,
            "success": bool(success),
            "verified": bool(verified),
            "verification_method": verification_method,
            "verification_evidence": self._redact(verification_evidence),
            "checkpoint_id": checkpoint_id,
            "tokenizer_id": tokenizer_id,
            "privacy_status": "redacted",
            "timestamp": time.time(),
            "approved_constraints": tuple(approved_constraints),
            "tool_results": tuple(self._redact(tool_results or [])),
        }
        body["content_hash"] = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode("utf-8")
        ).hexdigest()
        record = TrajectoryRecord(**body)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        return record

    def verified_count(self) -> int:
        if not self.path.exists():
            return 0
        count = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                count += int(bool(row.get("verified", False)) and bool(row.get("success", False)))
            except Exception:
                continue
        return count
