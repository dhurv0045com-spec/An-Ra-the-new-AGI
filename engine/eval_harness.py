from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from anra_paths import OUTPUT_V2_DIR


@dataclass
class EvalResult:
    component: str
    mode: str
    task_success_rate: float
    avg_latency_ms: float
    error_rate: float
    token_cost: float | None = None
    notes: str = ""
    raw: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegressionReport:
    component: str
    baseline: EvalResult
    current: EvalResult
    regressed: bool
    delta_success_rate: float
    delta_latency_ms: float
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


class EvalHarness:
    """Run any callable task against a component in different modes."""

    def __init__(self, output_dir: Path = OUTPUT_V2_DIR / "eval"):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def run_baseline(
        self,
        component: str,
        tasks: list[dict],
        runner: Callable[[dict], dict],
    ) -> EvalResult:
        """Run tasks with component disabled. Measures baseline performance."""
        from engine.feature_flags import load_flags, set_flag

        original = load_flags().get(component, True)
        set_flag(component, False)
        try:
            return self._execute(component, "baseline", tasks, runner)
        finally:
            set_flag(component, original)

    def run_system_on(
        self,
        component: str,
        tasks: list[dict],
        runner: Callable[[dict], dict],
    ) -> EvalResult:
        """Run tasks with component enabled. Measures full system performance."""
        from engine.feature_flags import load_flags, set_flag

        original = load_flags().get(component, True)
        set_flag(component, True)
        try:
            return self._execute(component, "system_on", tasks, runner)
        finally:
            set_flag(component, original)

    def run_ablation(
        self,
        ablated_component: str,
        tasks: list[dict],
        runner: Callable[[dict], dict],
    ) -> EvalResult:
        """Run with one component disabled, rest on."""
        from engine.feature_flags import load_flags, set_flag

        original = load_flags().get(ablated_component, True)
        set_flag(ablated_component, False)
        try:
            return self._execute(ablated_component, "ablation", tasks, runner)
        finally:
            set_flag(ablated_component, original)

    def _execute(
        self,
        component: str,
        mode: str,
        tasks: list[dict],
        runner: Callable[[dict], dict],
    ) -> EvalResult:
        results = []
        for task in tasks:
            t0 = time.perf_counter()
            try:
                out = runner(task)
                success = bool(out.get("success", True)) if isinstance(out, dict) else True
                latency = (time.perf_counter() - t0) * 1000
                tokens = out.get("tokens_used") if isinstance(out, dict) else None
                results.append({"success": success, "latency_ms": latency, "tokens": tokens, "error": None})
            except Exception as exc:
                latency = (time.perf_counter() - t0) * 1000
                results.append({"success": False, "latency_ms": latency, "tokens": None, "error": str(exc)})

        n = len(results)
        success_rate = sum(1 for r in results if r["success"]) / n if n else 0.0
        avg_latency = sum(r["latency_ms"] for r in results) / n if n else 0.0
        error_rate = sum(1 for r in results if r["error"]) / n if n else 0.0
        tokens = [r["tokens"] for r in results if r["tokens"] is not None]
        avg_tokens = sum(tokens) / len(tokens) if tokens else None

        return EvalResult(
            component=component,
            mode=mode,
            task_success_rate=round(success_rate, 4),
            avg_latency_ms=round(avg_latency, 2),
            error_rate=round(error_rate, 4),
            token_cost=avg_tokens,
            raw=results,
        )

    def compare(self, baseline: EvalResult, current: EvalResult, regression_threshold: float = 0.05) -> RegressionReport:
        delta_success = current.task_success_rate - baseline.task_success_rate
        delta_latency = current.avg_latency_ms - baseline.avg_latency_ms
        regressed = delta_success < -regression_threshold
        verdict = "regressed" if regressed else ("improved" if delta_success > regression_threshold else "neutral")
        return RegressionReport(
            component=current.component,
            baseline=baseline,
            current=current,
            regressed=regressed,
            delta_success_rate=round(delta_success, 4),
            delta_latency_ms=round(delta_latency, 2),
            verdict=verdict,
        )

    def save_report(self, report: RegressionReport) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = self._output_dir / f"eval_{report.component}_{ts}.json"
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return out

    def load_last_report(self, component: str) -> dict | None:
        """Load most recent saved report for a component."""
        files = sorted(self._output_dir.glob(f"eval_{component}_*.json"), reverse=True)
        if not files:
            return None
        return json.loads(files[0].read_text(encoding="utf-8"))
