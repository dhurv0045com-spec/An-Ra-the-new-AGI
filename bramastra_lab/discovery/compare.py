"""Exploratory, paired comparison of two BRAMASTRA evaluation stages."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .statistics import _percentile, summarize


def _key(row: dict) -> tuple[Any, ...]:
    return (row["world_id"], row["target"], row["policy"], row["budget"])


def _ci(deltas: list[float], samples: int, seed: int) -> list[float] | None:
    if len(deltas) <= 1:
        return None
    rng = random.Random(seed)
    boot = [sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
            for _ in range(samples)]
    return [_percentile(boot, 0.025), _percentile(boot, 0.975)]


def _metric_record(parent: list[dict], child: list[dict], policy: str,
                   budget: int, family: str | None, samples: int, seed: int) -> dict:
    def in_group(rows: list[dict]) -> list[dict]:
        return [r for r in rows if r["policy"] == policy and r["budget"] == budget
                and (family is None or r["family"] == family)]

    before, after = in_group(parent), in_group(child)
    before_by_key = {_key(r): r for r in before}
    after_by_key = {_key(r): r for r in after}
    if set(before_by_key) != set(after_by_key):
        raise ValueError("stage comparison group has unmatched keys")
    if not before:
        return {"policy": policy, "budget": budget, "family": family,
                "nworlds": 0, "parent_accuracy": None, "child_accuracy": None,
                "accuracy_delta": None, "parent_brier": None, "child_brier": None,
                "brier_delta": None, "accuracy_ci95": None,
                "weighting": "equal semantic worlds"}
    worlds: dict[Any, list[tuple[float, float, float]]] = defaultdict(list)
    for key, old in before_by_key.items():
        new = after_by_key[key]
        old_brier = (old["probability"] - old["label"]) ** 2
        new_brier = (new["probability"] - new["label"]) ** 2
        worlds[old["world_id"]].append((float(old["correct"]), float(new["correct"]),
                                         old_brier, new_brier))
    world_values = []
    for world in sorted(worlds, key=str):
        values = worlds[world]
        n = len(values)
        old_acc = sum(v[0] for v in values) / n
        new_acc = sum(v[1] for v in values) / n
        old_brier = sum(v[2] for v in values) / n
        new_brier = sum(v[3] for v in values) / n
        world_values.append((old_acc, new_acc, new_acc - old_acc, old_brier, new_brier))
    parent_accuracy = sum(v[0] for v in world_values) / len(world_values)
    child_accuracy = sum(v[1] for v in world_values) / len(world_values)
    parent_brier = sum(v[3] for v in world_values) / len(world_values)
    child_brier = sum(v[4] for v in world_values) / len(world_values)
    accuracy_deltas = [v[2] for v in world_values]
    return {"policy": policy, "budget": budget, "family": family,
            "nworlds": len(world_values), "parent_accuracy": parent_accuracy,
            "child_accuracy": child_accuracy, "accuracy_delta": child_accuracy - parent_accuracy,
            "parent_brier": parent_brier, "child_brier": child_brier,
            "brier_delta": child_brier - parent_brier,
            "accuracy_ci95": _ci(accuracy_deltas, samples, seed),
            "weighting": "equal semantic worlds"}


def compare_stages(parent_rows: list[dict], child_rows: list[dict],
                   bootstrap_samples: int = 1000, seed: int = 0) -> dict:
    """Compare paired stage rows; all reported findings are descriptive."""
    if not isinstance(bootstrap_samples, int) or isinstance(bootstrap_samples, bool) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    summarize(parent_rows, bootstrap_samples=bootstrap_samples, seed=seed)
    summarize(child_rows, bootstrap_samples=bootstrap_samples, seed=seed)
    parent_keys = {_key(r) for r in parent_rows}
    child_keys = {_key(r) for r in child_rows}
    if parent_keys != child_keys:
        raise ValueError("parent and child rows must have exact matched keys")
    parent_truth = {(r["world_id"], r["target"]): (r["label"], r["family"]) for r in parent_rows}
    child_truth = {(r["world_id"], r["target"]): (r["label"], r["family"]) for r in child_rows}
    if parent_truth != child_truth:
        raise ValueError("parent and child labels/families must match for each world/target")
    policies = sorted({r["policy"] for r in parent_rows}, key=str)
    budgets = sorted({r["budget"] for r in parent_rows})
    records = []
    for policy in policies:
        for budget in budgets:
            rows = [r for r in parent_rows if r["policy"] == policy and r["budget"] == budget]
            families = sorted({r["family"] for r in rows})
            records.append(_metric_record(parent_rows, child_rows, policy, budget, None,
                                          bootstrap_samples, seed))
            records.extend(_metric_record(parent_rows, child_rows, policy, budget, family,
                                          bootstrap_samples, seed) for family in families)
    family_records = [r for r in records if r["family"] is not None and r["accuracy_delta"] is not None]
    worst = min((r["accuracy_delta"] for r in family_records), default=None)
    return {"comparisons": records,
            "retention": {"worst_family_delta": worst,
                          "interpretation": "descriptive; mean gains do not establish retention or promotion"},
            "primary_distinction": "exploratory, not confirmatory"}


def _load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON row list")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = args.run / "comparison.json"
    if destination.exists() and not args.overwrite:
        raise FileExistsError("comparison.json exists; use --overwrite explicitly")
    parent = _load(args.run / "initial" / "evaluation_rows.json")
    comparisons = {}
    for name in ("new_only", "replay"):
        comparisons[name] = compare_stages(parent, _load(args.run / name / "evaluation_rows.json"))
    destination.write_text(json.dumps(comparisons, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": str(destination)}, sort_keys=True))


if __name__ == "__main__":
    main()
