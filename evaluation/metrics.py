"""Canonical M-01 through M-12 evidence snapshots."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from anra.anra_paths import METRIC_HISTORY, METRIC_SNAPSHOT


@dataclass(frozen=True)
class MetricValue:
    metric_id: str
    name: str
    value: float
    target: float
    passed: bool
    evidence: str


@dataclass(frozen=True)
class MetricSnapshot:
    schema_version: int
    generated_at: float
    checkpoint: str
    values: tuple[MetricValue, ...]

    @property
    def all_required_passed(self) -> bool:
        return all(value.passed for value in self.values)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["all_required_passed"] = self.all_required_passed
        return payload


METRIC_NAMES = {
    "M-01": "owner_task_score",
    "M-02": "training_tokens",
    "M-03": "identity_retention",
    "M-04": "verified_trajectories",
    "M-05": "verified_reasoning_rate",
    "M-06": "truth_checking_coverage",
    "M-07": "cdr_closure_rate",
    "M-08": "memory_recall_at_3",
    "M-09": "self_improvement_success",
    "M-10": "sovereignty_accuracy",
    "M-11": "deployment_uptime",
    "M-12": "ibs_overall",
}


def build_snapshot(
    *,
    checkpoint: str,
    measurements: dict[str, float],
    targets: dict[str, float],
    evidence: dict[str, str] | None = None,
) -> MetricSnapshot:
    evidence = evidence or {}
    values = tuple(
        MetricValue(
            metric_id=metric_id,
            name=name,
            value=float(measurements.get(metric_id, 0.0)),
            target=float(targets[metric_id]),
            passed=float(measurements.get(metric_id, 0.0)) >= float(targets[metric_id]),
            evidence=evidence.get(metric_id, ""),
        )
        for metric_id, name in METRIC_NAMES.items()
        if metric_id in targets
    )
    return MetricSnapshot(1, time.time(), checkpoint, values)


def persist_snapshot(snapshot: MetricSnapshot, path: str | Path = METRIC_SNAPSHOT) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot.to_dict()
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    METRIC_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with METRIC_HISTORY.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return target
