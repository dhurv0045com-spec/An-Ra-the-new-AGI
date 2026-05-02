from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class BusMessage:
    topic: str
    payload: dict
    sender: str = ""


class MessageBus:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}

    def _queue(self, topic: str) -> asyncio.Queue:
        if topic not in self._queues:
            self._queues[topic] = asyncio.Queue()
        return self._queues[topic]

    async def publish(self, topic: str, payload: dict, sender: str = "") -> None:
        await self._queue(topic).put(BusMessage(topic=topic, payload=payload, sender=sender))

    async def subscribe(self, topic: str) -> BusMessage:
        return await self._queue(topic).get()

    def pending(self, topic: str) -> int:
        return self._queue(topic).qsize()
