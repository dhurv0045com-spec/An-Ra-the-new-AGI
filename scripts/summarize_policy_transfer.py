"""Summarize frozen intervention-policy transfer from corrected audit rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def score(rows: list[dict], action: str) -> dict:
    successes = 0
    regressions = 0
    cost = 0
    costs = {"NO_CHANGE": 0, "CONSTRAINED": 1, "NORMALIZED": 2}
    for row in rows:
        outcome = {
            "NO_CHANGE": row["free_correct"],
            "CONSTRAINED": row["constrained_correct"],
            "NORMALIZED": row["normalized_correct"],
        }[action]
        successes += int(outcome)
        regressions += int(action != "NO_CHANGE" and not outcome and row["free_correct"])
        cost += costs[action]
    return {"successes": successes, "total": len(rows), "regressions": regressions, "cost": cost}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.input_dir)
    matrix = {}
    for path in sorted(root.glob("*_cf.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report["binding"]["rows"]
        fixed = {name: score(rows, name) for name in ("NO_CHANGE", "CONSTRAINED", "NORMALIZED")}
        policy_rows = [row for row in rows if "policy_action" in row]
        policy = {
            "successes": sum(int(row["policy_correct"]) for row in policy_rows),
            "total": len(policy_rows),
            "regressions": sum(int(row["policy_action"] != "NO_CHANGE" and not row["policy_correct"] and row["free_correct"]) for row in policy_rows),
            "cost": sum({"NO_CHANGE": 0, "CONSTRAINED": 1, "NORMALIZED": 2}[row["policy_action"]] for row in policy_rows),
            "action_counts": report["binding"]["frozen_policy_transfer"]["action_counts"] if report["binding"].get("frozen_policy_transfer") else {},
        }
        matrix[report["label"]] = {
            "checkpoint": report["checkpoint"],
            "step": report["identity"].get("global_step"),
            "fixed": fixed,
            "ADAPTIVE_v7_frozen": policy,
            "policy_parameter_sha256": report["binding"].get("frozen_policy_transfer", {}).get("policy_parameter_sha256"),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "anra-policy-transfer-matrix/v1", "matrix": matrix}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "checkpoints": list(matrix)}, indent=2))


if __name__ == "__main__":
    main()
