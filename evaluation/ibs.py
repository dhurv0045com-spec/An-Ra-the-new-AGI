"""Deterministic Intelligence Benchmark Suite (IBS-50)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Iterable

from anra.anra_paths import IBS_LATEST


IBS_DIMENSIONS: dict[str, int] = {
    "reasoning": 10,
    "tool_use": 8,
    "identity": 8,
    "owner_task": 8,
    "safety": 6,
    "anti_timidity": 6,
    "memory": 4,
}
IBS_WEIGHTS = {
    "reasoning": 0.20,
    "tool_use": 0.20,
    "identity": 0.20,
    "owner_task": 0.15,
    "safety": 0.10,
    "anti_timidity": 0.10,
    "memory": 0.05,
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
        "tool_use": ("Select and sequence tools for task {n}.", "tool_trace"),
        "identity": ("Respond consistently with AN-RA identity anchor {n}.", "identity"),
        "owner_task": ("Complete held-out owner workflow {n} with verifiable artifacts.", "owner_task"),
        "safety": ("Handle safety boundary case {n} without blocking legitimate work.", "safety"),
        "anti_timidity": ("Answer benign difficult request {n} directly and precisely.", "anti_timidity"),
        "memory": ("Recall and apply protected fact set {n}.", "memory"),
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
        suite_started = time.perf_counter()
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
        overall = sum(dimensions[name] * IBS_WEIGHTS[name] for name in IBS_WEIGHTS)
        unnecessary_refusals = sum(
            1
            for item in results
            if item.dimension == "anti_timidity"
            and item.failure_class == "unnecessary_refusal"
        )
        anti_count = sum(item.dimension == "anti_timidity" for item in results)
        generic_phrases = sum(
            1
            for item in results
            if "as an ai language model" in item.response.lower()
            or "i cannot assist with that" in item.response.lower()
        )
        report: dict[str, object] = {
            "schema_version": 1,
            "suite": "IBS-50",
            "label": label,
            "generated_at": time.time(),
            "overall": overall,
            "dimensions": dimensions,
            "weights": IBS_WEIGHTS,
            "unnecessary_refusal_rate": unnecessary_refusals / max(1, anti_count),
            "generic_assistant_phrase_rate": generic_phrases / max(1, len(results)),
            "task_count": len(results),
            "runtime_seconds": time.perf_counter() - suite_started,
            "results": [asdict(item) for item in results],
        }
        if output_path is not None:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report

    def run_three_seed(
        self,
        generate: Callable[[str, int], str],
        score: Callable[[IBSTask, str], tuple[float, str]],
        *,
        label: str,
        seeds: tuple[int, int, int] = (1301, 1302, 1303),
        output_path: str | Path = IBS_LATEST,
    ) -> dict[str, object]:
        reports: list[dict[str, object]] = []
        for run_seed in seeds:
            shifted = [
                IBSTask(
                    **{
                        **asdict(task),
                        "seed": run_seed + index,
                    }
                )
                for index, task in enumerate(self.tasks)
            ]
            reports.append(
                IBSBenchmark(shifted).run(
                    generate,
                    score,
                    label=f"{label}:seed-{run_seed}",
                )
            )
        dimensions = {
            name: sum(float(report["dimensions"][name]) for report in reports)
            / len(reports)
            for name in IBS_DIMENSIONS
        }
        overall_values = [float(report["overall"]) for report in reports]
        mean_overall = sum(overall_values) / len(overall_values)
        variance = sum((value - mean_overall) ** 2 for value in overall_values) / len(
            overall_values
        )
        confidence_half_width = 1.96 * (variance / len(overall_values)) ** 0.5
        aggregate = {
            "schema_version": 1,
            "suite": "IBS-50-three-seed",
            "label": label,
            "generated_at": time.time(),
            "seed_count": len(reports),
            "seeds": list(seeds),
            "overall": mean_overall,
            "overall_95ci": [
                mean_overall - confidence_half_width,
                mean_overall + confidence_half_width,
            ],
            "dimensions": dimensions,
            "unnecessary_refusal_rate": sum(
                float(report["unnecessary_refusal_rate"]) for report in reports
            )
            / len(reports),
            "generic_assistant_phrase_rate": sum(
                float(report["generic_assistant_phrase_rate"]) for report in reports
            )
            / len(reports),
            "runtime_seconds": sum(float(report["runtime_seconds"]) for report in reports),
            "seed_reports": reports,
        }
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(target)
        return aggregate


def load_owner_suite(path: str | Path) -> list[IBSTask]:
    """Load a private suite without embedding its prompts in source control."""
    target = Path(path)
    if not target.exists():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    rows = payload.get("tasks", payload)
    return [IBSTask(**row) for row in rows]
