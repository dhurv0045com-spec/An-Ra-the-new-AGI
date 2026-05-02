from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import heapq
import json
import time


@dataclass
class GoalItem:
    goal_id: str
    text: str
    priority: int = 100
    created_at: float = 0.0
    status: str = "queued"
    parent_id: str | None = None
    attempts: int = 0
    max_attempts: int = 3
    last_error: str = ""
    completed_at: float | None = None
    metadata: dict[str, object] | None = None


class GoalQueue:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._heap: list[tuple[int, float, str]] = []
        self._items: dict[str, GoalItem] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for row in data.get("items", []):
                item = GoalItem(**row)
                self._items[item.goal_id] = item
                if item.status == "queued":
                    heapq.heappush(self._heap, (item.priority, item.created_at, item.goal_id))
        except Exception as exc:
            print(f"[goal_queue] load failed: {exc}")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": [asdict(v) for v in self._items.values()]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def push(self, goal_id: str, text: str, priority: int = 100) -> GoalItem:
        item = GoalItem(goal_id=goal_id, text=text, priority=int(priority), created_at=time.time(), metadata={})
        self._items[item.goal_id] = item
        heapq.heappush(self._heap, (item.priority, item.created_at, item.goal_id))
        self._save()
        return item

    def pop(self) -> GoalItem | None:
        while self._heap:
            _, _, gid = heapq.heappop(self._heap)
            item = self._items.get(gid)
            if item and item.status == "queued":
                item.status = "in_progress"
                self._save()
                return item
        return None

    def complete(self, goal_id: str) -> bool:
        item = self._items.get(goal_id)
        if not item:
            return False
        item.status = "done"
        item.completed_at = time.time()
        self._save()
        return True

    def fail(self, goal_id: str, error: str = "") -> bool:
        item = self._items.get(goal_id)
        if not item:
            return False
        item.attempts += 1
        item.last_error = str(error)
        item.status = "failed" if item.attempts >= item.max_attempts else "queued"
        if item.status == "queued":
            heapq.heappush(self._heap, (item.priority, time.time(), item.goal_id))
        self._save()
        return True

    def generate_successor(self, goal_id: str, text: str | None = None, priority: int | None = None) -> GoalItem:
        parent = self._items[goal_id]
        successor_id = f"{goal_id}:next:{int(time.time() * 1000)}"
        item = GoalItem(
            goal_id=successor_id,
            text=text or parent.text,
            priority=int(parent.priority if priority is None else priority),
            created_at=time.time(),
            status="queued",
            parent_id=goal_id,
            metadata={},
        )
        self._items[item.goal_id] = item
        heapq.heappush(self._heap, (item.priority, item.created_at, item.goal_id))
        self._save()
        return item

    def status_report(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        for item in self._items.values():
            counts[item.status] = counts.get(item.status, 0) + 1
        return {
            "total": len(self._items),
            "counts": counts,
            "queued": counts.get("queued", 0),
            "in_progress": counts.get("in_progress", 0),
            "failed": counts.get("failed", 0),
            "done": counts.get("done", 0),
        }

    def list(self, status: str | None = None) -> list[GoalItem]:
        vals = list(self._items.values())
        if status is None:
            return vals
        return [v for v in vals if v.status == status]
