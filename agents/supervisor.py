"""
SupervisorAgent - top-level controller for the An-Ra agent system.

Owner interface:
    supervisor = SupervisorAgent(model_size="1b")
    supervisor.start_session()
    # ... run training / inference / agent tasks ...
    summary = supervisor.end_session()
    supervisor.push_scorecard_to_drive(summary)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from anra_paths import DRIVE_SCORECARD, SCORECARD_DIR
from engine.feature_flags import is_enabled, set_flag
from engine.metric_bus import get_metric_bus, reset_metric_bus

SUCCESS_FLOOR = 0.70
LATENCY_REG_MS = 500
SCORE_REG_FLOOR = -0.05


@dataclass
class SessionSummary:
    run_id: str
    model_size: str
    started_at: float
    ended_at: float
    components_active: list[str]
    components_flagged: list[str]
    deltas: dict
    recommendations: list[str]
    raw_metrics: dict

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model_size": self.model_size,
            "started_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started_at)
            ),
            "ended_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.ended_at)
            ),
            "duration_minutes": round((self.ended_at - self.started_at) / 60, 2),
            "components_active": self.components_active,
            "components_flagged": self.components_flagged,
            "deltas": self.deltas,
            "recommendations": self.recommendations,
            "raw_metrics": self.raw_metrics,
        }


class SupervisorAgent:
    def __init__(self, model_size: str = "25m") -> None:
        self._model_size = model_size
        self._started_at: float | None = None
        self._bus = None
        SCORECARD_DIR.mkdir(parents=True, exist_ok=True)

    def start_session(self) -> str:
        """Reset the metric bus and begin a tracked session. Returns run_id."""
        self._bus = reset_metric_bus()
        self._started_at = time.time()
        print(f"[Supervisor] Session started  run_id: {self._bus.run_id}")
        print(f"[Supervisor] Model size: {self._model_size.upper()}")
        return self._bus.run_id

    def end_session(self) -> SessionSummary:
        """Finalize metrics, compute deltas, generate and write scorecard."""
        bus = self._bus or get_metric_bus()
        run_data = bus.finalize()
        ended_at = time.time()

        metrics = run_data.get("components", {})
        deltas = run_data.get("deltas", {})
        flagged = self._detect_regressions(metrics, deltas)
        recs = self._build_recommendations(metrics, deltas, flagged)

        summary = SessionSummary(
            run_id=run_data["run_id"],
            model_size=self._model_size,
            started_at=self._started_at or run_data["session_ts"],
            ended_at=ended_at,
            components_active=list(metrics.keys()),
            components_flagged=flagged,
            deltas=deltas,
            recommendations=recs,
            raw_metrics=metrics,
        )
        self._print_summary(summary)
        self._write_scorecard(summary)
        return summary

    def enable(self, component: str) -> None:
        set_flag(component, True)
        print(f"[Supervisor] {component} -> ENABLED")

    def disable(self, component: str) -> None:
        set_flag(component, False)
        print(f"[Supervisor] {component} -> DISABLED")

    def status(self) -> dict:
        """Return health snapshot for all registered components."""
        from runtime.system_registry import component_registry

        snap = (self._bus or get_metric_bus()).snapshot()
        result = {}
        for comp in component_registry():
            result[comp.name] = {
                "enabled": is_enabled(comp.name),
                "layer": comp.layer,
                "metrics": snap.get(comp.name, {}),
            }
        return result

    def push_scorecard_to_drive(self, summary: SessionSummary) -> bool:
        """
        Write scorecard locally, then push to Google Drive.
        Returns True if Drive push succeeded.
        Always succeeds locally; Drive failure is non-fatal.
        """
        data = summary.to_dict()
        fname = f"scorecard_{summary.run_id}.json"

        local = SCORECARD_DIR / fname
        local.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[Supervisor] Scorecard saved locally: {local}")

        try:
            DRIVE_SCORECARD.mkdir(parents=True, exist_ok=True)
            drive_f = DRIVE_SCORECARD / fname
            drive_f.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"[Supervisor] Scorecard -> Drive: {drive_f}")
            return True
        except Exception as exc:
            print(f"[Supervisor] Drive push failed (local copy is safe): {exc}")
            return False

    def _detect_regressions(self, metrics: dict, deltas: dict) -> list[str]:
        flagged = []
        for comp, m in metrics.items():
            if m.get("success_rate", 1.0) < SUCCESS_FLOOR:
                flagged.append(f"{comp}:low_success_rate")
        for comp, d in deltas.items():
            if d.get("avg_latency_ms", 0) > LATENCY_REG_MS:
                flagged.append(f"{comp}:latency_regression")
            if (d.get("score") or 0.0) < SCORE_REG_FLOOR:
                flagged.append(f"{comp}:score_regression")
        return flagged

    def _build_recommendations(
        self, metrics: dict, deltas: dict, flagged: list[str]
    ) -> list[str]:
        if not flagged:
            return ["All components within normal parameters. Continue training."]
        recs = []
        for f in flagged:
            comp, issue = f.split(":", 1)
            if issue == "low_success_rate":
                recs.append(
                    f"{comp}: success rate below {SUCCESS_FLOOR:.0%}. "
                    f"Check for import errors or verifier failures. "
                    f"Disable via supervisor.disable('{comp}') if blocking training."
                )
            elif issue == "latency_regression":
                recs.append(
                    f"{comp}: latency increased >500ms vs last session. "
                    f"Possible GPU contention or I/O bottleneck."
                )
            elif issue == "score_regression":
                recs.append(
                    f"{comp}: score dropped >{abs(SCORE_REG_FLOOR):.0%} vs last session. "
                    f"Inspect loss curve and data mix before next run."
                )
        return recs

    def _print_summary(self, s: SessionSummary) -> None:
        print(f"\n{'=' * 60}")
        print(
            f"[Supervisor] SESSION COMPLETE  run_id={s.run_id}  "
            f"model={s.model_size.upper()}"
        )
        print(f"  Duration   : {s.to_dict()['duration_minutes']} min")
        print(f"  Components : {len(s.components_active)} active")
        if s.components_flagged:
            print(f"  FLAGGED    : {s.components_flagged}")
        print("  Recommendations:")
        for r in s.recommendations:
            print(f"    - {r}")
        print(f"{'=' * 60}\n")

    def _write_scorecard(self, summary: SessionSummary) -> None:
        path = SCORECARD_DIR / f"scorecard_{summary.run_id}.json"
        path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
