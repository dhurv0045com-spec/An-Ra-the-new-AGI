from __future__ import annotations

import asyncio

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
