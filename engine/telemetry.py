from __future__ import annotations

import functools
import inspect
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from anra_paths import STATE_DIR

TELEMETRY_LOG = STATE_DIR / "logs" / "telemetry.jsonl"


@dataclass
class TelemetryRecord:
    module: str
    operation: str
    start_ts: float
    end_ts: float
    elapsed_ms: float
    success: bool
    error: str | None = None
    tokens_used: int | None = None
    output_size: int | None = None
    output_type: str | None = None
    confidence: float | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class TelemetryBus:
    """Write-once append bus. Thread-safe enough for single-process use."""

    def __init__(self, path: Path = TELEMETRY_LOG):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[TelemetryRecord] = []

    def record(self, rec: TelemetryRecord) -> None:
        self._buffer.append(rec)
        self._flush_one(rec)

    def _flush_one(self, rec: TelemetryRecord) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def recent(self, n: int = 50) -> list[dict]:
        """Return last n records from the JSONL file."""
        if n <= 0 or not self._path.exists():
            return []
        rows: list[dict] = []
        for line in self._path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows

    def summary_by_module(self) -> dict[str, dict]:
        """Aggregate stats per module from the JSONL file."""
        if not self._path.exists():
            return {}
        grouped: dict[str, dict[str, Any]] = {}
        for line in self._path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            module = str(row.get("module", "unknown"))
            bucket = grouped.setdefault(
                module,
                {
                    "calls_total": 0,
                    "calls_success": 0,
                    "calls_failed": 0,
                    "total_latency_ms": 0.0,
                    "error_count": 0,
                },
            )
            bucket["calls_total"] += 1
            if row.get("success"):
                bucket["calls_success"] += 1
            else:
                bucket["calls_failed"] += 1
                bucket["error_count"] += 1
            bucket["total_latency_ms"] += float(row.get("elapsed_ms") or 0.0)

        summary: dict[str, dict] = {}
        for module, bucket in grouped.items():
            total = int(bucket["calls_total"])
            summary[module] = {
                "calls_total": total,
                "success_rate": round(bucket["calls_success"] / total, 4) if total else 0.0,
                "avg_latency_ms": round(bucket["total_latency_ms"] / total, 2) if total else 0.0,
                "error_count": int(bucket["error_count"]),
            }
        return summary


_bus: TelemetryBus | None = None


def get_telemetry_bus() -> TelemetryBus:
    global _bus
    if _bus is None:
        _bus = TelemetryBus()
    return _bus


def _result_metadata(result: Any) -> tuple[int | None, int | None, str | None, float | None]:
    tokens = None
    out_size = None
    out_type = None
    confidence = None
    if isinstance(result, dict):
        tokens = result.get("tokens_used") or result.get("token_count")
        out_size = len(str(result.get("output", result.get("text", ""))))
        out_type = result.get("output_type", "dict")
        confidence = result.get("confidence")
    elif result is not None:
        out_size = len(str(result))
        out_type = type(result).__name__
        confidence = getattr(result, "confidence", None)
    return tokens, out_size, out_type, confidence


def _make_record(
    *,
    module: str,
    operation: str,
    start_ts: float,
    t0: float,
    success: bool,
    error: str | None,
    result: Any,
) -> TelemetryRecord:
    elapsed_ms = (time.perf_counter() - t0) * 1000
    tokens, out_size, out_type, confidence = _result_metadata(result)
    return TelemetryRecord(
        module=module,
        operation=operation,
        start_ts=start_ts,
        end_ts=start_ts + elapsed_ms / 1000,
        elapsed_ms=round(elapsed_ms, 2),
        success=success,
        error=error,
        tokens_used=tokens,
        output_size=out_size,
        output_type=out_type,
        confidence=confidence,
    )


def trace(module: str, operation: str = "run"):
    """Decorator. Wraps any function with automatic telemetry capture."""

    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                bus = get_telemetry_bus()
                t0 = time.perf_counter()
                start_ts = time.time()
                error = None
                success = True
                result = None
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    success = False
                    error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    bus.record(
                        _make_record(
                            module=module,
                            operation=operation,
                            start_ts=start_ts,
                            t0=t0,
                            success=success,
                            error=error,
                            result=result,
                        )
                    )

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bus = get_telemetry_bus()
            t0 = time.perf_counter()
            start_ts = time.time()
            error = None
            success = True
            result = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as exc:
                success = False
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                bus.record(
                    _make_record(
                        module=module,
                        operation=operation,
                        start_ts=start_ts,
                        t0=t0,
                        success=success,
                        error=error,
                        result=result,
                    )
                )

        return wrapper

    return decorator
