"""Corrected-failure curriculum records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time


FAILURE_CATEGORIES = {
    "reasoning",
    "tool_selection",
    "planning",
    "memory",
    "identity_drift",
    "perception",
    "execution",
}


@dataclass(frozen=True)
class CorrectedFailure:
    prompt: str
    failed_output: str
    diagnosis: str
    corrected_target: str
    category: str
    verifier: str
    verified: bool
    provenance: dict[str, object] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class CorrectedFailureCurriculum:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: CorrectedFailure) -> None:
        if record.category not in FAILURE_CATEGORIES:
            raise ValueError(f"Unknown failure category: {record.category}")
        if not record.verified:
            raise ValueError("Unverified corrections cannot enter the curriculum.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")

    def load(self) -> list[CorrectedFailure]:
        if not self.path.exists():
            return []
        return [
            CorrectedFailure(**json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
