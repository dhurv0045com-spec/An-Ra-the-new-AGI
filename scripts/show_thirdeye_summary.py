"""Print a Colab-friendly ThirdEye evidence and intelligence dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.thirdeye_adapter import PROJECT_ID, THIRDEYE_HOME, activation_snapshot, run_one_click
from training.v2_runtime import build_frontier_model, model_summary


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _project_id(result: dict[str, Any]) -> str:
    project = result.get("project", {})
    if isinstance(project, dict):
        return str(project.get("project_id", PROJECT_ID))
    return PROJECT_ID


def _report_paths(result: dict[str, Any]) -> dict[str, str]:
    paths = result.get("report_paths", result.get("reports", {}))
    if not isinstance(paths, dict):
        return {}
    return {str(key): str(value) for key, value in paths.items()}


def _summarize_intelligence(intelligence: dict[str, Any] | None) -> list[str]:
    if not intelligence:
        return ["  Intelligence report: not found yet; it is written after training finalizes."]

    lines = ["  Intelligence report: FOUND"]
    estimate = intelligence.get("estimate")
    if isinstance(estimate, dict):
        for key in ("estimate", "score", "value", "confidence", "sample_count", "checkpoint_id"):
            if key in estimate:
                lines.append(f"    {key}: {estimate[key]}")

    signals = intelligence.get("signals")
    if isinstance(signals, list):
        lines.append(f"    signals captured: {len(signals)}")

    subsystems = intelligence.get("subsystems")
    if isinstance(subsystems, list):
        lines.append(f"    subsystems tracked: {len(subsystems)}")
    elif isinstance(subsystems, dict):
        lines.append(f"    subsystems tracked: {len(subsystems)}")

    if len(lines) == 1:
        keys = ", ".join(sorted(str(key) for key in intelligence.keys())[:12])
        lines.append(f"    available fields: {keys or 'unknown'}")
    return lines


def _clean_protocol(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        kind = value.get("kind", value.get("protocol", "unknown"))
        if hasattr(kind, "value"):
            kind = kind.value
        feature_id = value.get("feature_id")
        return f"{kind}:{feature_id}" if feature_id else str(kind)
    if hasattr(value, "kind"):
        kind = getattr(value, "kind", "unknown")
        if hasattr(kind, "value"):
            kind = kind.value
        feature_id = getattr(value, "feature_id", None)
        return f"{kind}:{feature_id}" if feature_id else str(kind)
    return str(value)


def _format_recommendation(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)

    feature_id = item.get("feature_id")
    protocols = item.get("protocols")
    protocol = item.get("protocol")
    if isinstance(protocols, list) and protocols:
        protocol_text = ", ".join(_clean_protocol(value) for value in protocols[:2])
        if feature_id is None and isinstance(protocols[0], dict):
            feature_id = protocols[0].get("feature_id")
        if feature_id is None and hasattr(protocols[0], "feature_id"):
            feature_id = getattr(protocols[0], "feature_id", None)
    else:
        protocol_text = _clean_protocol(protocol or "unknown")
        if feature_id is None and isinstance(protocol, dict):
            feature_id = protocol.get("feature_id")
        if feature_id is None and hasattr(protocol, "feature_id"):
            feature_id = getattr(protocol, "feature_id", None)

    feature_id = feature_id or "unknown"
    reason = item.get("reason", "needs more evidence")
    return f"{feature_id} [{protocol_text}]: {reason}"


def render_summary(
    result: dict[str, Any],
    *,
    intelligence_path: Path | None = None,
    warning: str | None = None,
) -> str:
    snapshot = {
        str(key): bool(value)
        for key, value in dict(result.get("activation_snapshot", {})).items()
    }
    active = sum(1 for value in snapshot.values() if value)
    total = len(snapshot)
    recommendations = result.get("recommended_experiments", [])
    if not isinstance(recommendations, list):
        recommendations = []

    lines = [
        "",
        "==================================================================",
        "  THIRD EYE EVIDENCE DASHBOARD",
        "==================================================================",
        f"  Project                 : {_project_id(result)}",
        f"  Evidence profile        : {result.get('profile', 'quick')}",
        f"  Registered features     : {len(result.get('features', []))}",
        f"  Active probes           : {active}/{total}" if total else "  Active probes           : none",
        f"  Recommended experiments : {len(recommendations)}",
    ]
    if warning:
        lines.append(f"  Warning                 : {warning}")

    lines.extend(["", "  Feature Activation"])
    if snapshot:
        for feature_id in sorted(snapshot):
            status = "OK" if snapshot[feature_id] else "MISS"
            lines.append(f"    {status:<4} {feature_id}")
        if "anra.hal" not in snapshot:
            lines.append("    note: architecture probes need --with-model; training telemetry is still active.")
    else:
        lines.append("    No activation snapshot was produced.")

    lines.extend(["", "  Next Evidence Gaps"])
    if recommendations:
        for item in recommendations[:8]:
            lines.append(f"    - {_format_recommendation(item)}")
    else:
        lines.append("    No new experiments recommended by the quick profile.")

    lines.extend(["", "  Reports"])
    paths = _report_paths(result)
    if paths:
        for name, path in sorted(paths.items()):
            lines.append(f"    {name}: {path}")
    else:
        lines.append("    No report paths returned.")

    if intelligence_path is None:
        intelligence_path = THIRDEYE_HOME / "reports" / PROJECT_ID / "intelligence.json"
    lines.extend(["", "  Subsystem Intelligence"])
    lines.extend(_summarize_intelligence(_load_json(intelligence_path)))
    lines.append(f"    path: {intelligence_path}")
    lines.append("==================================================================")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a visible ThirdEye dashboard in Colab")
    parser.add_argument("--profile", choices=["quick", "standard", "exhaustive", "auto"], default="quick")
    parser.add_argument("--with-model", action="store_true", help="Build the frontier model for architecture activation probes.")
    parser.add_argument("--without-model", action="store_true", help="Compatibility flag; model construction is skipped unless --with-model is set.")
    args = parser.parse_args()

    model = None
    if args.with_model and not args.without_model:
        model = build_frontier_model()
        summary = model_summary(model)
        if not 450_000_000 <= int(summary["parameters"]) <= 600_000_000:
            raise RuntimeError(f"Unexpected 500M-class frontier parameter count: {summary}")

    warning = None
    try:
        result = run_one_click(profile=args.profile, model=model)
    except Exception as exc:
        warning = f"ThirdEye evaluation failed; showing local activation snapshot only ({type(exc).__name__}: {exc})"
        result = {
            "project": {"project_id": PROJECT_ID},
            "profile": args.profile,
            "features": [],
            "recommended_experiments": [],
            "activation_snapshot": activation_snapshot(model),
            "report_paths": {},
        }

    print(render_summary(result, warning=warning), flush=True)


if __name__ == "__main__":
    main()
