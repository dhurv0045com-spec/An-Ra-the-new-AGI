"""Scientific self-improvement proposals and signed evidence lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
import uuid
from typing import Literal
from pathlib import Path


FailureCategory = Literal[
    "reasoning",
    "tool_selection",
    "planning",
    "identity_drift",
    "epistemic",
    "memory",
    "workflow_execution",
    "robotics_simulation",
]


@dataclass(frozen=True)
class FailureEvidence:
    session_id: str
    category: FailureCategory
    summary: str
    content_hash: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExperimentProposal:
    experiment_id: str
    category: FailureCategory
    hypothesis: str
    falsification: str
    evidence_hashes: tuple[str, ...]
    base_checkpoint: str
    tokenizer_hash: str
    data_hash: str
    code_hash: str
    config_hash: str
    seeds: tuple[int, ...] = (1301, 1302, 1303)
    maximum_tokens: int = 0
    stop_delta: float = -0.02
    authorized: bool = False
    status: str = "proposed"
    created_at: float = field(default_factory=time.time)

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ScientificSelfImprovementEngine:
    def __init__(
        self,
        *,
        analysis_window: int = 10,
        state_path: str | Path | None = None,
    ) -> None:
        self.analysis_window = int(analysis_window)
        self.state_path = Path(state_path) if state_path is not None else None
        self.proposals: dict[str, ExperimentProposal] = {}
        self.completed: dict[str, dict[str, object]] = {}
        self._load()

    def qualifying_patterns(self, failures: list[FailureEvidence]) -> dict[str, list[FailureEvidence]]:
        sessions = sorted({row.session_id for row in failures})[-self.analysis_window :]
        window = [row for row in failures if row.session_id in sessions]
        groups: dict[str, list[FailureEvidence]] = {}
        for row in window:
            groups.setdefault(row.category, []).append(row)
        total = max(1, len(window))
        return {
            category: rows
            for category, rows in groups.items()
            if len(rows) / total > 0.05 and len({row.session_id for row in rows}) >= 3
        }

    def propose(
        self,
        category: FailureCategory,
        evidence: list[FailureEvidence],
        *,
        base_checkpoint: str,
        tokenizer_hash: str,
        data_hash: str,
        code_hash: str,
        config_hash: str,
        maximum_tokens: int,
    ) -> ExperimentProposal:
        if category not in self.qualifying_patterns(evidence):
            raise ValueError("Failure pattern does not meet frequency and session thresholds.")
        proposal = ExperimentProposal(
            experiment_id=f"ssie-{uuid.uuid4().hex[:12]}",
            category=category,
            hypothesis=f"An isolated LoRA/DoRA candidate can reduce verified {category} failures.",
            falsification=f"Reject when three-seed candidate delta is below -0.02 or protected gates regress.",
            evidence_hashes=tuple(row.content_hash for row in evidence if row.category == category),
            base_checkpoint=base_checkpoint,
            tokenizer_hash=tokenizer_hash,
            data_hash=data_hash,
            code_hash=code_hash,
            config_hash=config_hash,
            maximum_tokens=int(maximum_tokens),
        )
        self.proposals[proposal.experiment_id] = proposal
        self._save()
        return proposal

    def authorize(self, experiment_id: str, *, owner_authorized: bool) -> ExperimentProposal:
        if not owner_authorized:
            raise PermissionError("SSIE experiments require explicit owner authorization.")
        proposal = self.proposals[experiment_id]
        proposal.authorized = True
        proposal.status = "authorized"
        self._save()
        return proposal

    def record_result(self, experiment_id: str, result: dict[str, object], *, signed: bool) -> None:
        proposal = self.proposals[experiment_id]
        if not proposal.authorized:
            raise PermissionError("Unauthorized experiment result cannot enter cognition theory.")
        if not signed:
            raise ValueError("Experiment result must be signed.")
        proposal.status = "completed"
        self.completed[experiment_id] = {"proposal_digest": proposal.digest(), **result, "signed": True}
        self._save()

    def cognition_theory(self) -> list[dict[str, object]]:
        return [
            {"experiment_id": experiment_id, **result}
            for experiment_id, result in self.completed.items()
            if result.get("signed") is True
        ]

    def _save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "proposals": {
                key: asdict(value) for key, value in self.proposals.items()
            },
            "completed": self.completed,
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.proposals = {
                key: ExperimentProposal(
                    **{
                        **value,
                        "evidence_hashes": tuple(value["evidence_hashes"]),
                        "seeds": tuple(value["seeds"]),
                    }
                )
                for key, value in payload.get("proposals", {}).items()
            }
            self.completed = dict(payload.get("completed", {}))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self.proposals = {}
            self.completed = {}
