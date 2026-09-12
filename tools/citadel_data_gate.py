"""Citadel Data Readiness Gate CLI (CITADEL-DATA-001).

Usage:
  python tools/citadel_data_gate.py --manifest <path> --output <report.json>

Reads a candidate corpus manifest, runs all 12 gates, and emits a
machine-readable report. Fail-closed: missing evidence = INCONCLUSIVE or
FAIL, never a silent PASS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citadel_tpu.data_gate import run_all_gates, GATE_SCHEMA  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Citadel Data Readiness Gate")
    parser.add_argument("--manifest", required=True, help="Path to candidate corpus manifest JSON")
    parser.add_argument("--eval-manifest", help="Path to evaluation manifest JSON")
    parser.add_argument("--output", help="Path for machine-readable report JSON")
    parser.add_argument("--near-threshold", type=float, default=0.85,
                        help="Near-duplicate Jaccard threshold (default 0.85)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest.get("documents", manifest.get("sources", []))
    if not documents:
        print("ERROR: manifest contains no documents/sources")
        return 1

    eval_docs = []
    if args.eval_manifest:
        ep = Path(args.eval_manifest)
        if ep.is_file():
            eval_manifest = json.loads(ep.read_text(encoding="utf-8"))
            eval_docs = eval_manifest.get("documents", eval_manifest.get("sources", []))

    result = run_all_gates(
        manifest=manifest, documents=documents,
        eval_documents=eval_docs, near_dedup_threshold=args.near_threshold)

    result["manifest_path"] = str(manifest_path)
    result["eval_manifest_path"] = args.eval_manifest

    output = args.output or str(manifest_path.parent / "DATA_READINESS_REPORT.json")
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

    print(f"\n{'GATE':<30} {'STATUS':>8}")
    print("-" * 40)
    for gate, status in result["gates"].items():
        print(f"  {gate:<28} {status:>6}")
    print(f"\n{'OVERALL':<30} {result['overall']:>6}")
    print(f"total defects: {result['total_defects']}")

    if result["defects"]:
        print("\nDEFECTS:")
        for gate, defects in sorted(result["defects"].items()):
            for d in defects[:3]:
                print(f"  [{gate}] {d}")

    return 0 if result["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
