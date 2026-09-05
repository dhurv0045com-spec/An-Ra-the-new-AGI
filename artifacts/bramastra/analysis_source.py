"""Recompute development readouts and query-blind baselines from raw records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readouts(records: list[dict]) -> dict:
    groups = defaultdict(list)
    first_hits = last_hits = 0
    for row in records:
        facts_text, query_text = row["prompt"].rsplit(";Q=", 1)
        facts = [fact.split("=") for fact in facts_text.split(";")]
        query = query_text.removesuffix(";V=")
        if dict(facts)[query] != row["answer"]:
            raise ValueError("independent prompt parser disagrees with dataset truth")
        if row["correct"] != (row["stop_reason"] == "EOS" and row["prediction"] == row["answer"]):
            raise ValueError("stored correctness disagrees with raw output")
        first_hits += facts[0][1] == row["answer"]
        last_hits += facts[-1][1] == row["answer"]
        groups[row["world_id"]].append(row)
    if not records or any(len(group) != 2 for group in groups.values()):
        raise ValueError("this analysis requires complete two-query world groups")
    both_terminated = [group for group in groups.values() if all(row["stop_reason"] == "EOS" for row in group)]
    same = sum(group[0]["prediction"] == group[1]["prediction"] for group in both_terminated)
    return {"n_queries": len(records), "n_worlds": len(groups),
            "correct_queries": sum(row["correct"] for row in records),
            "both_correct_worlds": sum(all(row["correct"] for row in group) for group in groups.values()),
            "both_terminated_worlds": len(both_terminated), "same_answer_despite_query_swap_worlds": same,
            "copy_first_value_correct": first_hits, "copy_last_value_correct": last_hits,
            "baseline_scope": "explicit query-blind deterministic policies; not a free-generation chance rate"}


def analyze(root: Path) -> dict:
    runs = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "result.json").exists():
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        arms = {}
        for name in ("without_terminal", "with_terminal"):
            path = directory / f"{name}.json"
            arm = json.loads(path.read_text())
            arms[name] = {"raw_receipt_sha256": sha(path),
                          "train": readouts(arm["evaluation"]["train"]["records"]),
                          "fresh_worlds": readouts(arm["evaluation"]["development"]["records"]),
                          "rendering_shift_correct": sum(row["correct"] for row in arm["evaluation"]["development_shift"]["records"])}
        runs[directory.name] = {"manifest_sha256": sha(directory / "manifest.json"),
                                "result_sha256": sha(directory / "result.json"),
                                "parameters": manifest["parameters"], "steps_per_arm": manifest["steps_per_arm"],
                                "elapsed_seconds": result["elapsed_seconds"], "arms": arms}
    if not runs:
        raise ValueError("no completed experiments")
    return {"schema": "bramastra-development-analysis/v1", "analyzer_source_sha256": sha(Path(__file__)),
            "scope": "development only; data-diversity follow-up is exploratory; no independent natural transfer",
            "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/bramastra"))
    args = parser.parse_args()
    destination = args.root / "analysis.json"
    if destination.exists():
        raise FileExistsError("analysis already exists; preserve prior evidence")
    result = analyze(args.root)
    (args.root / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
