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
    """Build a full system health and performance snapshot."""
    manifest = build_system_manifest()
    flags = load_flags()
    bus = get_telemetry_bus()

    component_summary = []
    for comp in manifest["components"]:
        name = comp["name"]
        component_summary.append(
            {
                "name": name,
                "enabled": flags.get(name, True),
                "source_ok": comp["source_ok"],
                "import_status": comp["import_status"],
                "missing_paths": comp.get("missing", []),
                "metric_hooks": comp.get("metric_hooks", []),
            }
        )

    try:
        perf_summary = bus.summary_by_module()
    except Exception:
        perf_summary = {}

    try:
        recent = bus.recent(100)
        failures = [r for r in recent if not r.get("success")]
        recent_failures = failures[-10:]
    except Exception:
        recent_failures = []

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system_health": {
            "total_components": len(component_summary),
            "enabled_components": sum(1 for c in component_summary if c["enabled"]),
            "source_ok": sum(1 for c in component_summary if c["source_ok"]),
            "import_ok": sum(1 for c in component_summary if "degraded" not in c["import_status"]),
        },
        "components": component_summary,
        "performance": perf_summary,
        "recent_failures": recent_failures,
        "training_readiness": manifest.get("training_readiness", {}),
        "artifacts": manifest.get("artifacts", {}),
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
