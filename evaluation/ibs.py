"""Deterministic Intelligence Benchmark Suite (IBS-50)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Iterable


IBS_DIMENSIONS: dict[str, int] = {
    "reasoning": 10,
    "language": 6,
    "coding_science": 6,
    "tool_use": 7,
    "memory": 5,
    "identity": 6,
    "creativity": 5,
    "anti_timidity": 5,
}


@dataclass(frozen=True)
class IBSTask:
    task_id: str
    dimension: str
    prompt: str
    verifier: str
    seed: int
    expected: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class IBSResult:
    task_id: str
    dimension: str
    seed: int
    prompt_hash: str
    verifier: str
    score: float
    latency_ms: float
    tokens_used: int
    failure_class: str = ""
    response: str = ""


def _default_tasks() -> list[IBSTask]:
    tasks: list[IBSTask] = []
    templates = {
        "reasoning": ("Solve and verify reasoning problem {n}.", "symbolic"),
        "language": ("Rewrite passage {n} precisely without losing meaning.", "rubric"),
        "coding_science": ("Produce a testable code or science answer for case {n}.", "code_or_science"),
        "tool_use": ("Select and sequence tools for task {n}.", "tool_trace"),
        "memory": ("Recall and apply protected fact set {n}.", "memory"),
        "identity": ("Respond consistently with AN-RA identity anchor {n}.", "identity"),
        "creativity": ("Generate and evaluate three novel approaches for problem {n}.", "novelty"),
        "anti_timidity": ("Answer benign difficult request {n} directly and precisely.", "anti_timidity"),
    }
    seed = 1301
    for dimension, count in IBS_DIMENSIONS.items():
        prompt_template, verifier = templates[dimension]
        for index in range(1, count + 1):
            tasks.append(
                IBSTask(
                    task_id=f"{dimension}-{index:02d}",
                    dimension=dimension,
                    prompt=prompt_template.format(n=index),
                    verifier=verifier,
                    seed=seed,
                )
            )
            seed += 1
    return tasks


class IBSBenchmark:
    """Runs and serializes the fixed 50-task public capability suite."""

    def __init__(self, tasks: Iterable[IBSTask] | None = None) -> None:
        self.tasks = list(tasks or _default_tasks())
        counts: dict[str, int] = {}
        for task in self.tasks:
            counts[task.dimension] = counts.get(task.dimension, 0) + 1
        if counts != IBS_DIMENSIONS:
            raise ValueError(f"IBS task distribution mismatch: {counts}")

    @classmethod
    def from_json(cls, path: str | Path) -> "IBSBenchmark":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(IBSTask(**item) for item in payload["tasks"])

    def save_definition(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "dimensions": IBS_DIMENSIONS,
            "tasks": [asdict(task) for task in self.tasks],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target

    def run(
        self,
        generate: Callable[[str, int], str],
        score: Callable[[IBSTask, str], tuple[float, str]],
        *,
        label: str,
        output_path: str | Path | None = None,
    ) -> dict[str, object]:
        results: list[IBSResult] = []
        for task in self.tasks:
            started = time.perf_counter()
            response = str(generate(task.prompt, task.seed))
            task_score, failure_class = score(task, response)
            elapsed = (time.perf_counter() - started) * 1000.0
            results.append(
                IBSResult(
                    task_id=task.task_id,
                    dimension=task.dimension,
                    seed=task.seed,
                    prompt_hash=hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
                    verifier=task.verifier,
                    score=max(0.0, min(1.0, float(task_score))),
                    latency_ms=elapsed,
                    tokens_used=len(response.split()),
                    failure_class=str(failure_class),
                    response=response,
                )
            )

        dimensions: dict[str, float] = {}
        for dimension in IBS_DIMENSIONS:
            values = [item.score for item in results if item.dimension == dimension]
            dimensions[dimension] = sum(values) / len(values)
        report: dict[str, object] = {
            "schema_version": 1,
            "suite": "IBS-50",
            "label": label,
            "generated_at": time.time(),
            "overall": sum(item.score for item in results) / len(results),
            "dimensions": dimensions,
            "results": [asdict(item) for item in results],
        }
        if output_path is not None:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report


def load_owner_suite(path: str | Path) -> list[IBSTask]:
    """Load a private suite without embedding its prompts in source control."""
    target = Path(path)
    if not target.exists():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    rows = payload.get("tasks", payload)
    return [IBSTask(**row) for row in rows]
