"""Execute the moonshot pilot gates from recorded evidence, fail-closed by default."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and not __package__:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_moonshot_pilots", *sys.argv[1:]],
        cwd=REPO_ROOT,
        check=False,
    )
    raise SystemExit(completed.returncode)

OUTPUT_V2_DIR = REPO_ROOT / "output" / "v2"
DEFAULT_EVIDENCE = OUTPUT_V2_DIR / "moonshot_pilot_evidence.json"
DEFAULT_REPORT = OUTPUT_V2_DIR / "moonshot_pilot_status.json"
DEFAULT_LOCAL_REPORT = OUTPUT_V2_DIR / "moonshot_local_execution.json"


def run_moonshot_pilots(evidence: dict[str, dict[str, float]]) -> dict[str, object]:
    """Evaluate every moonshot; missing metrics are blocked, never passed."""
    from training.moonshot_pilots import MOONSHOT_PILOTS, evaluate_moonshot_pilot

    rows: list[dict[str, object]] = []
    for pilot in MOONSHOT_PILOTS:
        metrics = dict(evidence.get(pilot.moonshot_id, {}))
        missing = [metric for metric in pilot.required_metrics if metric not in metrics]
        result = evaluate_moonshot_pilot(pilot.moonshot_id, metrics)
        rows.append(
            {
                "moonshot_id": pilot.moonshot_id,
                "title": pilot.title,
                "status": "blocked" if missing else "passed" if result["passed"] else "failed",
                "missing_metrics": missing,
                "result": result,
            }
        )
    statuses = {str(row["status"]) for row in rows}
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "rows": rows,
        "complete": statuses == {"passed"},
        "blocked": "blocked" in statuses,
        "failed": "failed" in statuses,
    }


def _read_evidence(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        return {}
    loaded: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("moonshot evidence must be a JSON object keyed by moonshot id")
    return {
        str(identifier): {str(key): float(value) for key, value in dict(metrics).items()}
        for identifier, metrics in loaded.items()
        if isinstance(metrics, dict)
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all moonshot pilot gates.")
    parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT))
    parser.add_argument("--execute-local", action="store_true")
    parser.add_argument("--local-report", default=str(DEFAULT_LOCAL_REPORT))
    args = parser.parse_args()
    evidence = _read_evidence(Path(args.evidence))
    if args.execute_local:
        from training.moonshot_execution import execute_local_moonshot_paths

        local_report = execute_local_moonshot_paths()
        _write_json(Path(args.local_report), local_report)
        for pilot_id, metrics in dict(local_report["acceptance_evidence"]).items():
            evidence[str(pilot_id)] = {
                str(key): float(value) for key, value in dict(metrics).items()
            }
    report = run_moonshot_pilots(evidence)
    _write_json(Path(args.json_out), report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if report["complete"] else 3 if report["blocked"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
