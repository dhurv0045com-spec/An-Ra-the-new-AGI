from __future__ import annotations

import asyncio
import time

from agents.message_bus import MessageBus


class OrchestratorAgent:
    def __init__(
        self,
        coder,
        researcher,
        memory,
        critic,
        bus: MessageBus | None = None,
        goal_queue=None,
    ) -> None:
        self.coder = coder
        self.researcher = researcher
        self.memory = memory
        self.critic = critic
        self.bus = bus or MessageBus()
        self.goal_queue = goal_queue

    async def dispatch(self, task: dict) -> dict:
        goal_id = task.get("goal_id")
        try:
            kind = task.get("kind", "coder")
            if kind == "coder":
                result = await self.coder.run(task)
            elif kind == "research":
                result = await self.researcher.run(task)
            elif kind == "memory":
                result = await self.memory.run(task)
            elif kind == "critic":
                result = await self.critic.run(task)
            else:
                raise ValueError(f"unknown kind: {kind}")
            if self.goal_queue is not None and goal_id:
                self.goal_queue.complete(str(goal_id))
            await self.bus.publish("results", {"task": task, "result": result}, sender="orchestrator")
            return result
        except Exception as exc:
            if self.goal_queue is not None and goal_id:
                self.goal_queue.fail(str(goal_id), str(exc))
            raise

    async def run_batch(self, tasks: list[dict]) -> list[dict]:
        return await asyncio.gather(*[self.dispatch(t) for t in tasks])

    async def run_session(self, max_goals: int = 5) -> dict:
        """
        Autonomous session loop. Pulls goals from queue, dispatches,
        marks complete or failed, generates successors.

        Called once per session after wakeup. Returns a summary dict
        for the session dashboard.
        """
        if self.goal_queue is None:
            return {
                "error": "No GoalQueue connected.",
                "goals_attempted": 0,
                "goals_completed": 0,
                "goals_failed": 0,
            }

        attempted = 0
        completed = 0
        failed = 0
        results = []

        for _ in range(int(max_goals)):
            goal = self.goal_queue.pop()
            if goal is None:
                break

            attempted += 1
            task = self._goal_to_task(goal)

            try:
                result = await self.dispatch(task)
                success = result.get("success", True)
                if success:
                    completed += 1
                    lesson = result.get("lesson", "")
                    if lesson and hasattr(self.goal_queue, "generate_successor"):
                        self.goal_queue.generate_successor(
                            goal.goal_id,
                            text=f"Follow-up: {lesson[:120]}",
                            priority=goal.priority + 10,
                        )
                else:
                    failed += 1
                    error = result.get("error", "task returned success=False")
                    self.goal_queue.fail(goal.goal_id, error=error)
                results.append({
                    "goal_id": goal.goal_id,
                    "text": goal.text[:80],
                    "success": success,
                })
            except Exception as exc:
                # dispatch() already called goal_queue.fail()
                failed += 1
                results.append({
                    "goal_id": goal.goal_id,
                    "text": goal.text[:80],
                    "success": False,
                    "error": str(exc)[:200],
                })

        return {
            "goals_attempted": attempted,
            "goals_completed": completed,
            "goals_failed": failed,
            "results": results,
            "queue_status": self.goal_queue.status_report(),
            "finished_at": time.time(),
        }

    def _goal_to_task(self, goal) -> dict:
        """Map a GoalItem to a dispatch task dict."""
        text = (goal.text or "").lower()
        if any(kw in text for kw in ("code", "implement", "write", "build", "fix", "create")):
            kind = "coder"
        elif any(kw in text for kw in ("research", "find", "read", "summarize", "analyse", "analyze")):
            kind = "research"
        elif any(kw in text for kw in ("remember", "recall", "store", "retrieve", "memory")):
            kind = "memory"
        elif any(kw in text for kw in ("review", "check", "validate", "critic")):
            kind = "critic"
        else:
            kind = "coder"

        task = {
            "kind": kind,
            "goal_id": goal.goal_id,
            "prompt": goal.text,
            "text": goal.text,
            "priority": goal.priority,
            "metadata": goal.metadata or {},
        }
        if kind == "memory":
            task["query"] = goal.text
        return task

    async def dispatch_next_goal(self, *, kind: str = "coder") -> dict | None:
        if self.goal_queue is None:
            return None
        goal = self.goal_queue.pop()
        if goal is None:
            return None
        task = {
            "kind": kind,
            "goal_id": goal.goal_id,
            "text": goal.text,
            "priority": goal.priority,
            "metadata": goal.metadata or {},
        }
        return await self.dispatch(task)
