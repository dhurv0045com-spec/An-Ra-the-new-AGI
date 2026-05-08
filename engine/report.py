from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from anra_paths import OUTPUT_V2_DIR
from engine.feature_flags import load_flags
from engine.telemetry import get_telemetry_bus
from runtime.system_registry import build_system_manifest


def build_report() -> dict[str, Any]:
    """Build a full system health and performance snapshot from MetricBus + TelemetryBus."""
    manifest = build_system_manifest()
    flags = load_flags()

    try:
        from engine.metric_bus import get_metric_bus

        mbus = get_metric_bus()
        mbus_snapshot = mbus.snapshot()
        mbus_deltas = getattr(mbus, "_last_deltas", {})
    except Exception as exc:
        mbus = None
        mbus_snapshot = {}
        mbus_deltas = {}
        print(f"[report] MetricBus unavailable: {exc}")

    legacy_bus = get_telemetry_bus()
    legacy_snapshot = legacy_bus.snapshot() if hasattr(legacy_bus, "snapshot") else {}

    try:
        perf_summary = legacy_bus.summary_by_module()
    except Exception:
        perf_summary = {}

    try:
        recent = legacy_bus.recent(100)
        failures = [r for r in recent if not r.get("success")]
        recent_failures = failures[-10:]
    except Exception:
        recent_failures = []

    component_summary = []
    manifest_by_name = {comp["name"]: comp for comp in manifest["components"]}
    all_components = sorted(set(manifest_by_name) | set(legacy_snapshot) | set(mbus_snapshot))
    for name in all_components:
        comp = manifest_by_name.get(name, {})
        metrics = {**legacy_snapshot.get(name, {}), **mbus_snapshot.get(name, {})}
        component_summary.append(
            {
                "name": name,
                "component": name,
                "enabled": flags.get(name, True),
                "source_ok": comp.get("source_ok", True),
                "import_status": comp.get("import_status", "unknown"),
                "missing_paths": comp.get("missing", []),
                "metric_hooks": comp.get("metric_hooks", []),
                "metrics": metrics,
                "delta": mbus_deltas.get(name, {}),
            }
        )

    return {
        "generated_at": time.time(),
        "system_health": {
            "total_components": len(component_summary),
            "enabled_components": sum(1 for c in component_summary if c["enabled"]),
            "source_ok": sum(1 for c in component_summary if c["source_ok"]),
            "import_ok": sum(1 for c in component_summary if "degraded" not in c["import_status"]),
        },
        "manifest": manifest,
        "flags": flags,
        "metric_bus": mbus_snapshot,
        "metric_bus_deltas": mbus_deltas,
        "components": component_summary,
        "performance": perf_summary,
        "recent_failures": recent_failures,
        "training_readiness": manifest.get("training_readiness", {}),
        "artifacts": manifest.get("artifacts", {}),
        "run_id": getattr(mbus, "_run_id", None),
    }


def save_report(output_dir: Path = OUTPUT_V2_DIR / "reports") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    ts = time.strftime("%Y-%m-%d")
    out = output_dir / f"anra_report_{ts}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def print_report() -> None:
    report = build_report()
    health = report["system_health"]
    print(f"\n{'=' * 60}")
    print("  AN-RA SYSTEM REPORT")
    print(f"  {report['generated_at']}")
    print(f"{'=' * 60}")
    print(f"  Components:  {health['enabled_components']}/{health['total_components']} enabled")
    print(f"  Source OK:   {health['source_ok']}/{health['total_components']}")
    print(f"  Import OK:   {health['import_ok']}/{health['total_components']}")
    print()
    for comp in report["components"]:
        flag = "ON " if comp["enabled"] else "OFF"
        src = "OK" if comp["source_ok"] else "NO"
        imp = "OK" if "degraded" not in comp["import_status"] else "WARN"
        perf = report["performance"].get(comp["name"], {})
        latency = perf.get("avg_latency_ms", 0.0)
        success = perf.get("success_rate", 0.0)
        print(f"  [{flag}] src={src:<2} import={imp:<4} perf={success:.2f}/{latency:.2f}ms  {comp['name']}")
    if report["recent_failures"]:
        print(f"\n  Recent failures ({len(report['recent_failures'])}):")
        for failure in report["recent_failures"]:
            print(f"    {failure.get('module')} / {failure.get('operation')}: {failure.get('error')}")
    print(f"{'=' * 60}\n")
