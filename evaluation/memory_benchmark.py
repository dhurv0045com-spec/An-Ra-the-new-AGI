"""Recall and provenance benchmark for BM25, semantic, and fused memory."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from anra.anra_paths import PRIVATE_EVAL_DIR


@dataclass(frozen=True)
class RetrievalTask:
    task_id: str
    query: str
    relevant_ids: tuple[str, ...]


def load_tasks(path: str | Path) -> list[RetrievalTask]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("tasks", payload)
    return [
        RetrievalTask(
            task_id=str(row["task_id"]),
            query=str(row["query"]),
            relevant_ids=tuple(str(value) for value in row["relevant_ids"]),
        )
        for row in rows
    ]


def freeze_private_owner_benchmark(
    tasks: list[RetrievalTask],
    *,
    owner_approved: bool,
    path: str | Path = PRIVATE_EVAL_DIR / "memory_owner_200.json",
) -> Path:
    if not owner_approved:
        raise PermissionError("Owner approval is required before freezing the private benchmark.")
    if len(tasks) != 200:
        raise ValueError("The private owner-memory benchmark must contain exactly 200 tasks.")
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Frozen private benchmark is immutable: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "owner_approved": True,
                "frozen_at": time.time(),
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "query": task.query,
                        "relevant_ids": list(task.relevant_ids),
                    }
                    for task in tasks
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def run_memory_benchmark(
    tasks: list[RetrievalTask],
    retrieve: Callable[[str, int], list[dict]],
    *,
    label: str,
) -> dict[str, object]:
    top1 = 0
    top3 = 0
    empty = 0
    provenance = 0
    started = time.perf_counter()
    rows = []
    for task in tasks:
        results = retrieve(task.query, 3)
        ids = [str(result.get("record_id", result.get("id", ""))) for result in results]
        if not ids:
            empty += 1
        hit1 = bool(ids and ids[0] in task.relevant_ids)
        hit3 = any(record_id in task.relevant_ids for record_id in ids[:3])
        top1 += int(hit1)
        top3 += int(hit3)
        provenance += int(all("payload" in result or "metadata" in result for result in results))
        rows.append({"task_id": task.task_id, "ids": ids, "hit1": hit1, "hit3": hit3})
    count = max(1, len(tasks))
    return {
        "schema_version": 1,
        "label": label,
        "task_count": len(tasks),
        "recall_at_1": top1 / count,
        "recall_at_3": top3 / count,
        "empty_result_rate": empty / count,
        "provenance_rate": provenance / count,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "results": rows,
    }


def run_hybrid_memory_benchmark(
    tasks: list[RetrievalTask],
    *,
    bm25: Callable[[str, int], list[dict]],
    semantic: Callable[[str, int], list[dict]],
    combined: Callable[[str, int], list[dict]],
    stale_ids: set[str] | None = None,
) -> dict[str, object]:
    if len(tasks) != 200:
        raise ValueError(
            "The frozen private owner-memory benchmark must contain exactly 200 tasks."
        )
    stale_ids = stale_ids or set()
    reports = {
        "bm25": run_memory_benchmark(tasks, bm25, label="bm25"),
        "faiss": run_memory_benchmark(tasks, semantic, label="faiss"),
        "combined": run_memory_benchmark(tasks, combined, label="combined"),
    }
    returned_ids = [record_id for row in reports["combined"]["results"] for record_id in row["ids"]]
    reports["combined"]["stale_memory_rate"] = sum(
        record_id in stale_ids for record_id in returned_ids
    ) / max(1, len(returned_ids))
    reports["targets"] = {
        "combined_recall_at_1": 0.70,
        "combined_recall_at_3": 0.85,
    }
    reports["passed"] = (
        float(reports["combined"]["recall_at_1"]) > 0.70
        and float(reports["combined"]["recall_at_3"]) > 0.85
    )
    return reports
