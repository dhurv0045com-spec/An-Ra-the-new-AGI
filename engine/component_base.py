from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class ComponentMetrics:
    name: str
    calls_total: int = 0
    calls_success: int = 0
    calls_failed: int = 0
    total_latency_ms: float = 0.0
    last_error: str | None = None

    @property
    def success_rate(self) -> float:
        return self.calls_success / self.calls_total if self.calls_total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls_total if self.calls_total else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "calls_total": self.calls_total,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "last_error": self.last_error,
        }


@runtime_checkable
class ComponentProtocol(Protocol):
    name: str
    version: str

    def health(self) -> dict[str, Any]: ...

    def metrics(self) -> ComponentMetrics: ...

    def run(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class BaseComponent:
    name: str = "unnamed"
    version: str = "0.1.0"

    def __init__(self):
        self._metrics = ComponentMetrics(name=self.name)

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "name": self.name, "version": self.version}

    def metrics(self) -> ComponentMetrics:
        return self._metrics

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _record_call(self, *, success: bool, latency_ms: float, error: str | None = None):
        self._metrics.calls_total += 1
        self._metrics.total_latency_ms += latency_ms
        if success:
            self._metrics.calls_success += 1
        else:
            self._metrics.calls_failed += 1
            self._metrics.last_error = error
