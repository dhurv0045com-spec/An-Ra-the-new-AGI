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
        item = GoalItem(goal_id=goal_id, text=text, priority=int(priority), created_at=time.time())
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
        self._save()
        return True

    def list(self, status: str | None = None) -> list[GoalItem]:
        vals = list(self._items.values())
        if status is None:
            return vals
        return [v for v in vals if v.status == status]
