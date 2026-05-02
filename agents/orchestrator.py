from __future__ import annotations

import asyncio

from agents.message_bus import MessageBus


class OrchestratorAgent:
    def __init__(self, coder, researcher, memory, critic, bus: MessageBus | None = None) -> None:
        self.coder = coder
        self.researcher = researcher
        self.memory = memory
        self.critic = critic
        self.bus = bus or MessageBus()

    async def dispatch(self, task: dict) -> dict:
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
        await self.bus.publish("results", {"task": task, "result": result}, sender="orchestrator")
        return result

    async def run_batch(self, tasks: list[dict]) -> list[dict]:
        return await asyncio.gather(*[self.dispatch(t) for t in tasks])
