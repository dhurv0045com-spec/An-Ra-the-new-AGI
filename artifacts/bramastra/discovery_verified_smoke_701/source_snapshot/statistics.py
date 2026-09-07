"""Small, deterministic summaries for discovery-policy evaluation records."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


_REQUIRED = {
    "world_id", "family", "target", "policy", "budget", "correct",
    "probability", "label", "queries",
}


def _check(rows: list[dict], bootstrap_samples: int) -> None:
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    if not isinstance(bootstrap_samples, int) or isinstance(bootstrap_samples, bool) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if not isinstance(row, dict) or not _REQUIRED.issubset(row):
            raise ValueError("each row is missing required fields")
        for field in ("world_id", "family", "policy"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"{field} must be a nonempty string")
        key = (row["policy"], row["budget"], row["world_id"], row["target"])
        if key in seen:
            raise ValueError("duplicate policy/budget/world/target row")
        seen.add(key)
        if not isinstance(row["target"], int) or isinstance(row["target"], bool) or row["target"] < 0:
            raise ValueError("target must be a nonnegative integer")
        if not isinstance(row["budget"], int) or isinstance(row["budget"], bool) or row["budget"] < 0:
            raise ValueError("budget must be a nonnegative integer")
        if not isinstance(row["queries"], int) or isinstance(row["queries"], bool) or row["queries"] < 0 or row["queries"] > row["budget"]:
            raise ValueError("queries must be a nonnegative integer no greater than budget")
        if not isinstance(row["label"], int) or isinstance(row["label"], bool) or row["label"] not in (0, 1):
            raise ValueError("label must be 0 or 1")
        if not isinstance(row["correct"], bool):
            raise ValueError("correct must be boolean")
        probability = row["probability"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probability must be finite and in [0, 1]")
        if row["correct"] != ((probability >= 0.5) == bool(row["label"])):
            raise ValueError("correct must agree with thresholded probability and label")

    labels_families: dict[tuple[Any, Any], tuple[int, str]] = {}
    for row in rows:
        world_target = (row["world_id"], row["target"])
        value = (row["label"], row["family"])
        previous = labels_families.setdefault(world_target, value)
        if previous != value:
            raise ValueError("label and family must be consistent for each world/target")


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    worlds = {r["world_id"] for r in rows}
    result = {
        "nworlds": len(worlds), "nrows": n,
        "accuracy": (sum(r["correct"] for r in rows) / n if n else None),
        "brier": (sum((r["probability"] - r["label"]) ** 2 for r in rows) / n if n else None),
        "mean_queries": (sum(r["queries"] for r in rows) / n if n else None),
        "family": {},
    }
    by_family: dict[Any, list[dict]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    result["family"] = {str(family): _aggregate_basic(group) for family, group in sorted(by_family.items(), key=lambda x: str(x[0]))}
    return result


def _aggregate_basic(rows: list[dict]) -> dict:
    n = len(rows)
    return {"nworlds": len({r["world_id"] for r in rows}), "nrows": n,
            "accuracy": sum(r["correct"] for r in rows) / n,
            "brier": sum((r["probability"] - r["label"]) ** 2 for r in rows) / n,
            "mean_queries": sum(r["queries"] for r in rows) / n}


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def _comparison(rows: list[dict], policy: str, baseline: str, budget: int,
                bootstrap_samples: int, seed: int) -> dict:
    selected = [r for r in rows if r["budget"] == budget and r["policy"] in (policy, baseline)]
    indexed: dict[tuple[str, Any, Any], dict] = {}
    for row in selected:
        indexed[(row["policy"], row["world_id"], row["target"])] = row
    keys = {(r["world_id"], r["target"]) for r in selected if r["policy"] == policy}
    baseline_keys = {(r["world_id"], r["target"]) for r in selected if r["policy"] == baseline}
    if keys != baseline_keys:
        raise ValueError(f"missing paired keys for {policy} vs {baseline} at budget {budget}")
    by_world: dict[Any, list[float]] = defaultdict(list)
    for world, target in sorted(keys, key=lambda x: (str(x[0]), x[1])):
        by_world[world].append(float(indexed[(policy, world, target)]["correct"]) - float(indexed[(baseline, world, target)]["correct"]))
    world_deltas = [sum(values) / len(values) for values in by_world.values()]
    point = sum(world_deltas) / len(world_deltas) if world_deltas else None
    interval = None
    if len(world_deltas) > 1:
        rng = random.Random(seed)
        boot = [sum(world_deltas[rng.randrange(len(world_deltas))] for _ in world_deltas) / len(world_deltas) for _ in range(bootstrap_samples)]
        interval = [_percentile(boot, 0.025), _percentile(boot, 0.975)]
    return {"policy": policy, "baseline": baseline, "budget": budget,
            "nworlds": len(world_deltas), "delta_accuracy": point,
            "ci95": interval, "uncertainty": "insufficient worlds" if len(world_deltas) <= 1 else "cluster bootstrap percentile"}


def summarize(rows: list[dict], bootstrap_samples: int = 1000, seed: int = 0) -> dict:
    """Validate and summarize policy rows; comparisons are exploratory readouts."""
    _check(rows, bootstrap_samples)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["policy"])].setdefault(str(row["budget"]), []).append(row)
    aggregates = {policy: {budget: _aggregate(group) for budget, group in budgets.items()} for policy, budgets in grouped.items()}
    comparisons = []
    policies = sorted({r["policy"] for r in rows}, key=str)
    budgets = sorted({r["budget"] for r in rows})
    for policy in policies:
        if policy in ("random", "coverage"):
            continue
        for budget in budgets:
            for baseline in ("random", "coverage"):
                if any(r["policy"] == baseline and r["budget"] == budget for r in rows):
                    comparisons.append(_comparison(rows, policy, baseline, budget, bootstrap_samples, seed))
    return {"aggregates": aggregates, "comparisons": comparisons,
            "primary_distinction": "exploratory, not confirmatory"}
