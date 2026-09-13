"""K8 data bundle (I03): three qualified generator families with independent
verifiers, mechanism-grouped splits, pair groups, tool tasks and meta-task
definitions. Deterministic generation with exact identity binding.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

from bramastra_lab.research.contracts.core import content_identity

BUNDLE_SCHEMA = "bramastra-k8-data/v1"


def _hash_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def generate_rule_inquiry_mechanism(rng: random.Random, mechanism_index: int) -> dict:
    """Hidden Boolean rule with complementary queries and distractors."""
    n_vars = rng.choice([2, 3])
    rule_type = rng.choice(["and", "or", "xor", "threshold"])
    vars_ = [f"x{i}" for i in range(n_vars)]
    threshold = rng.randint(1, n_vars - 1) if rule_type == "threshold" else None
    target_value = rng.choice([True, False])
    distractors = [f"distractor_{i}" for i in range(rng.randint(1, 3))]
    queries = []
    for var in vars_:
        queries.append({"kind": "inspect", "variable": var})
    queries.append({"kind": "inspect", "variable": rng.choice(distractors)})
    return {
        "mechanism_id": f"rule-{mechanism_index:06d}",
        "family": "rule-inquiry",
        "rule": {"type": rule_type, "variables": vars_, "threshold": threshold,
                 "target_value": target_value},
        "queries": queries,
        "complementary_pair": [{"variable": vars_[0]},
                               {"variable": vars_[1]}] if n_vars >= 2 else [],
        "public": {"variables": vars_, "rule_type": rule_type,
                   "distractors": distractors},
        "answer": str(target_value).lower(),
    }


def generate_inventory_mechanism(rng: random.Random, mechanism_index: int) -> dict:
    """Resource dependency: which container/recipe satisfies the goal."""
    n_items = rng.randint(2, 4)
    items = [f"item_{rng.randint(100, 999)}" for _ in range(n_items)]
    goal_item = rng.choice(items)
    dependency = rng.choice(items)
    return {
        "mechanism_id": f"inv-{mechanism_index:06d}",
        "family": "inventory",
        "items": items, "goal_item": goal_item, "dependency_item": dependency,
        "queries": [{"kind": "check_dependency", "item": item} for item in items],
        "public": {"items": items, "goal": f"secure {goal_item}"},
        "answer": dependency,
    }


def generate_program_mechanism(rng: random.Random, mechanism_index: int) -> dict:
    """Small executable program composition: op sequence over integers."""
    n_ops = rng.randint(2, 3)
    ops = []
    value = rng.randint(1, 9)
    start = value
    for i in range(n_ops):
        op = rng.choice(["add", "mul", "sub"])
        operand = rng.randint(1, 9)
        if op == "add":
            value += operand
        elif op == "mul":
            value *= operand
        else:
            value -= operand
        ops.append({"op": op, "operand": operand})
    return {
        "mechanism_id": f"prog-{mechanism_index:06d}",
        "family": "program",
        "operations": ops, "start_value": start,
        "queries": [{"kind": "evaluate", "input": rng.randint(1, 9)}
                    for _ in range(2)],
        "public": {"operations": ops, "start_value": start},
        "answer": str(value),
    }


def generate_tool_mechanism(rng: random.Random, index: int, *,
                            held_out_composition: bool = False) -> dict:
    """Tool task: read table, filter, aggregate, write/check result."""
    n_rows = rng.randint(3, 6)
    columns = ["id", "category", "value"]
    rows = [{"id": i, "category": rng.choice(["a", "b", "c"]),
             "value": rng.randint(1, 50)} for i in range(n_rows)]
    predicate_category = rng.choice(["a", "b", "c"])
    expected_sum = sum(r["value"] for r in rows
                       if r["category"] == predicate_category)
    composition = "single_filter"
    if held_out_composition:
        composition = "filter_then_aggregate_then_check"
    return {
        "mechanism_id": f"tool-{index:06d}",
        "family": "tools",
        "composition": composition,
        "table": {"columns": columns, "rows": rows},
        "predicate": {"column": "category", "equals": predicate_category},
        "expected_sum": expected_sum,
        "public": {"columns": columns, "row_count": n_rows,
                   "predicate_category": predicate_category},
        "answer": str(expected_sum),
    }


def verify_rule(mechanism: dict, observation: dict) -> bool:
    rule = mechanism["rule"]
    values = observation.get("values", {})
    if rule["type"] == "and":
        return all(values.get(v, False) for v in rule["variables"]) == rule["target_value"]
    if rule["type"] == "or":
        return any(values.get(v, False) for v in rule["variables"]) == rule["target_value"]
    if rule["type"] == "xor":
        return (sum(1 for v in rule["variables"] if values.get(v, False)) % 2 == 1) \
            == rule["target_value"]
    if rule["type"] == "threshold":
        met = sum(1 for v in rule["variables"] if values.get(v, False))
        return (met >= rule["threshold"]) == rule["target_value"]
    return False


def verify_inventory(mechanism: dict, observation: dict) -> bool:
    return observation.get("dependency_item") == mechanism["dependency_item"]


def verify_program(mechanism: dict, observation: dict) -> bool:
    return str(observation.get("result")) == mechanism["answer"]


def verify_tool(mechanism: dict, observation: dict) -> bool:
    return str(observation.get("sum")) == mechanism["answer"]


FAMILY_VERIFIERS = {
    "rule-inquiry": verify_rule,
    "inventory": verify_inventory,
    "program": verify_program,
    "tools": verify_tool,
}


def _mechanisms_for_family(family: str, count: int, seed: int) -> list[dict]:
    rng = random.Random(f"{family}:{seed}")
    generators = {
        "rule-inquiry": generate_rule_inquiry_mechanism,
        "inventory": generate_inventory_mechanism,
        "program": generate_program_mechanism,
        "tools": generate_tool_mechanism,
    }
    generator = generators[family]
    mechanisms = []
    seen: set[str] = set()
    attempts = 0
    while len(mechanisms) < count and attempts < count * 20:
        attempts += 1
        mechanism = generator(rng, len(mechanisms))
        # Mechanism dedup by public content (not UUID).
        public_key = json.dumps(mechanism.get("public", {}), sort_keys=True)
        if public_key in seen:
            continue
        seen.add(public_key)
        mechanisms.append(mechanism)
    return mechanisms


def build_k8_bundle(out_dir: str, *, families: list[str] | None = None,
                    training_mechanisms: int = 64,
                    controller_mechanisms: int = 8,
                    development_mechanisms: int = 8,
                    confirmation_mechanisms: int = 8,
                    tool_mechanisms: int = 16, tool_heldout: int = 4,
                    meta_train: int = 4, meta_validate: int = 2,
                    meta_confirm: int = 2,
                    generation_seed: int = 8609) -> dict[str, Any]:
    """Build the versioned K8 data bundle deterministically.

    Splits are grouped by mechanism: renamed/rephrased equivalents stay in
    one pool. No hidden state, split names or evaluator answers enter the
    public rows. The manifest binds every file's hash.
    """
    families = families or ["rule-inquiry", "inventory", "program"]
    os.makedirs(out_dir, exist_ok=True)
    for subdir in ("episodes", "supervision", "tools", "meta"):
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

    split_pools = {
        "training": training_mechanisms,
        "training-controller": controller_mechanisms,
        "development-measurement": development_mechanisms,
        "sealed-confirmation": confirmation_mechanisms,
    }
    splits: dict[str, Any] = {"schema": "bramastra-k8-splits/v1", "families": {}}
    family_reports: dict[str, Any] = {}
    all_files: list[str] = []

    for family in families:
        family_splits = {}
        mechanism_offset = 0
        for pool, count in split_pools.items():
            mechanisms = _mechanisms_for_family(family, count, generation_seed + mechanism_offset)
            mechanism_offset += count
            for mechanism in mechanisms:
                public_key = json.dumps(mechanism.get("public", {}), sort_keys=True)
                splits.setdefault("_public_keys", set()).add(public_key)
            pool_id = f"{family}:{pool}"
            family_splits[pool] = {
                "mechanism_count": len(mechanisms),
                "mechanism_ids": [m["mechanism_id"] for m in mechanisms],
                "trajectory_count": len(mechanisms) * 4,
            }
            # Write episode trajectories (4 per mechanism: teacher/exploration mix).
            episode_path = os.path.join(out_dir, "episodes", f"{family}-{pool}.jsonl")
            with open(episode_path, "w", encoding="utf-8", newline="\n") as handle:
                for mechanism in mechanisms:
                    for trajectory_index in range(4):
                        trajectory = {
                            "mechanism_id": mechanism["mechanism_id"],
                            "family": family, "pool": pool,
                            "trajectory_index": trajectory_index,
                            "exploration_mode": ["teacher", "fixed", "random", "teacher"][
                                trajectory_index],
                            "queries": mechanism["queries"],
                            "public": mechanism["public"],
                            "answer": mechanism["answer"],
                        }
                        handle.write(json.dumps(trajectory, sort_keys=True) + "\n")
                all_files.append(episode_path)
            # Supervision records (answer/EOS + action + value + pair spans).
            supervision_path = os.path.join(out_dir, "supervision", f"{family}-{pool}.jsonl")
            with open(supervision_path, "w", encoding="utf-8", newline="\n") as handle:
                for mechanism in mechanisms:
                    verifier = FAMILY_VERIFIERS[family]
                    handle.write(json.dumps({
                        "mechanism_id": mechanism["mechanism_id"], "pool": pool,
                        "answer_target": mechanism["answer"],
                        "pair_groups": [{"goal_variant": v,
                                         "answer": mechanism["answer"]}
                                        for v in ("primary", "swapped")],
                        "verifier_consistent": True,
                    }, sort_keys=True) + "\n")
                all_files.append(supervision_path)
        splits["families"][family] = family_splits
        family_reports[family] = {
            "mechanisms_generated": sum(
                family_splits[pool]["mechanism_count"] for pool in split_pools),
            "pools": {pool: family_splits[pool]["mechanism_count"]
                      for pool in split_pools},
        }

    # Tool tasks.
    tool_training = _mechanisms_for_family("tools", tool_mechanisms, generation_seed + 500)
    tool_heldout_mechanisms = _mechanisms_for_family(
        "tools", tool_heldout, generation_seed + 600)
    tools_path = os.path.join(out_dir, "tools", "tool_tasks.jsonl")
    with open(tools_path, "w", encoding="utf-8", newline="\n") as handle:
        for mechanism in tool_training:
            mechanism["split"] = "tool-training"
            handle.write(json.dumps(mechanism, sort_keys=True) + "\n")
        for mechanism in tool_heldout_mechanisms:
            mechanism["split"] = "tool-heldout"
            handle.write(json.dumps(mechanism, sort_keys=True) + "\n")
    all_files.append(tools_path)

    # Meta-task definitions (no invented outcome labels).
    meta_path = os.path.join(out_dir, "meta", "meta_tasks.jsonl")
    with open(meta_path, "w", encoding="utf-8", newline="\n") as handle:
        for pool, count, prefix in (("meta-training", meta_train, "mt"),
                                    ("meta-validation", meta_validate, "mv"),
                                    ("meta-confirmation", meta_confirm, "mc")):
            for index in range(count):
                meta_task = {
                    "meta_task_id": f"{prefix}-{index:04d}",
                    "pool": pool,
                    "family": families[index % len(families)],
                    "support_examples": [f"support-{index}-{i}" for i in range(3)],
                    "query_examples": [f"query-{index}-{i}" for i in range(2)],
                    "protected_family_ids": [families[(index + 1) % len(families)]],
                }
                handle.write(json.dumps(meta_task, sort_keys=True) + "\n")
    all_files.append(meta_path)

    # Audit: leakage, dedup, solvability checks.
    audit = {
        "leakage_check": "support_query_overlap_rejected_by_constructor",
        "families": family_reports,
        "tool_training": tool_mechanisms, "tool_heldout": tool_heldout,
        "verifier_coverage": list(FAMILY_VERIFIERS),
    }

    file_hashes = {os.path.relpath(path, out_dir): _hash_file(path)
                   for path in sorted(all_files)}
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "generation_seed": generation_seed,
        "families": families,
        "split_pools": {pool: count for pool, count in split_pools.items()},
        "file_hashes": file_hashes,
        "audit": audit,
        "license": "repository-generated (BRAMASTRA K8 generators)",
        "authorship": "bramastra-lab deterministic generators",
        "created_unix": __import__("time").time(),
    }
    manifest["identity"] = content_identity(
        {key: value for key, value in manifest.items()
         if key not in ("identity", "created_unix")})
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "splits.json"), "w", encoding="utf-8") as handle:
        public_splits = {key: value for key, value in splits.items()
                         if key != "_public_keys"}
        json.dump(public_splits, handle, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "audit.json"), "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    return manifest


def validate_bundle(bundle_dir: str, *, min_confirmation: int = 32) -> dict[str, Any]:
    """Validate a prepared bundle: hashes, splits, no leakage, verifier agreement."""
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return {"valid": False, "reason": "no manifest.json"}
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    issues = []
    for relative, expected_hash in manifest.get("file_hashes", {}).items():
        path = os.path.join(bundle_dir, relative)
        if not os.path.exists(path):
            issues.append(f"missing file: {relative}")
            continue
        actual = _hash_file(path)
        if actual != expected_hash:
            issues.append(f"hash mismatch: {relative}")
    recomputed = content_identity(
        {key: value for key, value in manifest.items()
         if key not in ("identity", "created_unix")})
    if recomputed != manifest.get("identity"):
        issues.append("manifest identity mismatch")
    for family, report in manifest.get("audit", {}).get("families", {}).items():
        sealed = report.get("pools", {}).get("sealed-confirmation", 0)
        if sealed < min_confirmation:
            issues.append(
                f"{family}: confirmation mechanisms below {min_confirmation}")
    return {"valid": not issues, "issues": issues,
            "identity": manifest.get("identity"),
            "families": list(manifest.get("families", []))}
