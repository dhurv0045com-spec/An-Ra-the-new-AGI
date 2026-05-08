from __future__ import annotations

import functools
import inspect
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _runs_dir() -> Path:
    """Drive-first, local fallback. Call once at MetricBus init."""
    from anra_paths import DRIVE_LOGS, STATE_DIR

    primary = DRIVE_LOGS / "runs"
    try:
        primary.mkdir(parents=True, exist_ok=True)
        return primary
    except Exception:
        fallback = STATE_DIR / "runs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


@dataclass
class ComponentMetricSnapshot:
    component: str
    run_id: str
    session_ts: float
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    loss_delta: float | None = None
    score: float | None = None
    tokens_used: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success_rate"] = round(self.success_rate, 4)
        d["avg_latency_ms"] = round(self.avg_latency_ms, 2)
        return d


class MetricBus:
    """
    MLflow/W&B-style metric collector for An-Ra.

    One bus per process. Reset at the start of each training session
    via reset_metric_bus() to get a fresh run ID.

    Every component emits here automatically via @instrument.
    Call finalize() at session end to write the run file and
    compute deltas vs the previous run.
    """

    def __init__(self) -> None:
        self._dir = _runs_dir()
        self._run_id = str(uuid.uuid4())[:8]
        self._session_ts = time.time()
        self._snaps: dict[str, ComponentMetricSnapshot] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    def emit(
        self,
        component: str,
        *,
        success: bool,
        latency_ms: float,
        tokens: int = 0,
        loss_delta: float | None = None,
        score: float | None = None,
        extra: dict | None = None,
    ) -> None:
        if component not in self._snaps:
            self._snaps[component] = ComponentMetricSnapshot(
                component=component,
                run_id=self._run_id,
                session_ts=self._session_ts,
            )
        s = self._snaps[component]
        s.calls += 1
        s.total_latency_ms += latency_ms
        s.tokens_used += tokens
        if success:
            s.successes += 1
        else:
            s.failures += 1
        if loss_delta is not None:
            s.loss_delta = loss_delta
        if score is not None:
            s.score = score
        if extra:
            s.extra.update(extra)

    def snapshot(self) -> dict[str, dict]:
        return {name: s.to_dict() for name, s in self._snaps.items()}

    def finalize(self) -> dict:
        """
        Write this run to disk and compute deltas vs the previous run.
        Returns the full run record including deltas.
        """
        ended_ts = time.time()
        run_data = {
            "run_id": self._run_id,
            "session_ts": self._session_ts,
            "ended_ts": ended_ts,
            "duration_minutes": round((ended_ts - self._session_ts) / 60, 2),
            "components": self.snapshot(),
        }
        run_file = self._dir / f"run_{self._run_id}.json"
        run_file.write_text(json.dumps(run_data, indent=2), encoding="utf-8")

        prev = self._load_previous_run()
        if prev:
            run_data["deltas"] = self._compute_deltas(
                prev["components"], run_data["components"]
            )
        else:
            run_data["deltas"] = {}

        run_file.write_text(json.dumps(run_data, indent=2), encoding="utf-8")
        return run_data

    def _load_previous_run(self) -> dict | None:
        runs = sorted(
            self._dir.glob("run_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for r in runs:
            if self._run_id in r.name:
                continue
            try:
                return json.loads(r.read_text(encoding="utf-8"))
            except Exception:
                continue
        return None

    def _compute_deltas(self, prev: dict, curr: dict) -> dict:
        deltas = {}
        for comp in set(prev) | set(curr):
            p, c, d = prev.get(comp, {}), curr.get(comp, {}), {}
            for key in (
                "success_rate",
                "avg_latency_ms",
                "calls",
                "tokens_used",
                "score",
                "loss_delta",
            ):
                pv, cv = p.get(key), c.get(key)
                if pv is not None and cv is not None:
                    d[key] = round(cv - pv, 4)
            if d:
                deltas[comp] = d
        return deltas


_bus: MetricBus | None = None


def get_metric_bus() -> MetricBus:
    global _bus
    if _bus is None:
        _bus = MetricBus()
    return _bus


def reset_metric_bus() -> MetricBus:
    """Call at the start of each training session to get a fresh run ID."""
    global _bus
    _bus = MetricBus()
    return _bus


def _emit_from_result(
    component: str, result: Any, elapsed_ms: float, success: bool
) -> None:
    tokens, score, loss_delta, extra = 0, None, None, {}
    if isinstance(result, dict):
        tokens = int(result.get("tokens_used") or result.get("tokens") or 0)
        score = result.get("score")
        loss_delta = result.get("loss_delta")
        if "error" in result:
            extra["error"] = result["error"]
    get_metric_bus().emit(
        component,
        success=success,
        latency_ms=elapsed_ms,
        tokens=tokens,
        score=score,
        loss_delta=loss_delta,
        extra=extra or None,
    )


def instrument(component: str):
    """
    Decorator that automatically emits metrics to the MetricBus.

    Works on sync and async methods. Coexists with @trace.
    """

    def decorator(fn):
        is_async = inspect.iscoroutinefunction(fn)
        if is_async:

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                t = time.perf_counter()
                try:
                    r = await fn(*args, **kwargs)
                    _emit_from_result(
                        component, r, (time.perf_counter() - t) * 1000, True
                    )
                    return r
                except Exception as exc:
                    get_metric_bus().emit(
                        component,
                        success=False,
                        latency_ms=(time.perf_counter() - t) * 1000,
                        extra={"error": str(exc)},
                    )
                    raise

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            t = time.perf_counter()
            try:
                r = fn(*args, **kwargs)
                _emit_from_result(
                    component, r, (time.perf_counter() - t) * 1000, True
                )
                return r
            except Exception as exc:
                get_metric_bus().emit(
                    component,
                    success=False,
                    latency_ms=(time.perf_counter() - t) * 1000,
                    extra={"error": str(exc)},
                )
                raise

        return sync_wrapper

    return decorator
