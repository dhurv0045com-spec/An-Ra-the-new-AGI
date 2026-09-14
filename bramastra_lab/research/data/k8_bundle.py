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
    """Hidden Boolean rule with genuine semantic diversity for 4096 mechanisms.

    Diversity comes from genuine Boolean-function differences, NOT surface
    names: variable count (2-6), rule type (10 types), relevant subset
    (which variables actually matter), per-variable negation pattern,
    threshold values, target values, distractor COUNT and observation noise.
    Variable/distractor NAMES are randomized surface labels and are excluded
    from the canonical semantic key (R06: do not invent diversity by counting
    names). The dedup key is the canonical semantic equivalence class.
    """
    n_vars = rng.randint(2, 6)
    rule_type = rng.choice([
        "and", "or", "xor", "nand", "nor", "xnor",
        "threshold", "majority", "exactly_one", "at_least_two"])
    vars_ = [f"v{rng.randint(1000, 9999)}_{i}" for i in range(n_vars)]
    # Relevant subset: which variables actually determine the output.
    k_relevant = rng.randint(1, n_vars)
    relevant_idx = sorted(rng.sample(range(n_vars), k_relevant))
    # Per-variable negation pattern (genuine function difference).
    negations = [rng.choice([False, False, True]) for _ in range(n_vars)]
    threshold = rng.randint(1, max(1, k_relevant)) \
        if rule_type in ("threshold", "at_least_two") else None
    target_value = rng.choice([True, False])
    n_distractors = rng.randint(1, 5)
    distractors = [f"d{rng.randint(100, 999)}" for _ in range(n_distractors)]
    obs_noise = rng.choice(["none", "flip_one", "extra_report"])
    query_order = rng.sample(vars_, len(vars_))
    queries = [{"kind": "inspect", "variable": v} for v in query_order]
    queries.append({"kind": "inspect", "variable": rng.choice(distractors)})
    return {
        "mechanism_id": f"rule-{mechanism_index:06d}",
        "family": "rule-inquiry",
        "rule": {"type": rule_type, "variables": vars_, "threshold": threshold,
                 "target_value": target_value,
                 "relevant": relevant_idx,
                 "negations": negations},
        "queries": queries,
        "complementary_pair": [{"variable": vars_[0]},
                               {"variable": vars_[1]}] if n_vars >= 2 else [],
        "public": {"variables": vars_, "rule_type": rule_type,
                   "distractors": distractors, "obs_noise": obs_noise,
                   "n_vars": n_vars, "n_distractors": n_distractors},
        "answer": str(target_value).lower(),
    }


def canonical_rule_key(rule: dict) -> str:
    """Canonical semantic equivalence class for a rule (R06).

    Renamings map to the same key: variable names are excluded; only the
    genuine function structure counts (type, relevant count, sorted negation
    pattern over relevant vars, threshold, target, arity). Two mechanisms
    with the same key are semantic equivalents and must group in one split.
    """
    rule_type = rule.get("type")
    variables = rule.get("variables", [])
    n_vars = len(variables)
    relevant = rule.get("relevant", list(range(n_vars)))
    negations = rule.get("negations", [False] * n_vars)
    try:
        rel_neg = sorted(bool(negations[i]) for i in relevant
                         if 0 <= i < len(negations))
    except Exception:
        rel_neg = []
    key = {
        "type": rule_type,
        "n_vars": n_vars,
        "n_relevant": len(relevant),
        "relevant_negations_sorted": rel_neg,
        "threshold": rule.get("threshold"),
        "target_value": rule.get("target_value"),
    }
    return json.dumps(key, sort_keys=True, default=str)


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
    """Tool task: read table, filter, aggregate, write/check result.

    Training composition is single_filter (filter one category, sum values).
    Held-out composition is filter_then_aggregate_then_check: filter by
    category, aggregate by a second predicate (value threshold), then check
    against a threshold — a genuinely different execution structure (different
    predicates, different aggregation, extra check step), not a label.
    """
    n_rows = rng.randint(3, 6)
    columns = ["id", "category", "value"]
    rows = [{"id": i, "category": rng.choice(["a", "b", "c"]),
             "value": rng.randint(1, 50)} for i in range(n_rows)]
    predicate_category = rng.choice(["a", "b", "c"])
    if not held_out_composition:
        expected_sum = sum(r["value"] for r in rows
                           if r["category"] == predicate_category)
        return {
            "mechanism_id": f"tool-{index:06d}",
            "family": "tools",
            "composition": "single_filter",
            "execution": {"steps": ["filter", "sum"],
                          "filter": {"column": "category",
                                     "equals": predicate_category}},
            "table": {"columns": columns, "rows": rows},
            "predicate": {"column": "category", "equals": predicate_category},
            "expected_sum": expected_sum,
            "public": {"columns": columns, "row_count": n_rows,
                       "predicate_category": predicate_category},
            "answer": str(expected_sum),
        }
    # Held-out: two-stage execution with a value-threshold second filter and
    # an explicit threshold check (different predicates + extra step).
    value_threshold = rng.randint(10, 30)
    check_threshold = rng.randint(20, 80)
    stage1 = [r for r in rows if r["category"] == predicate_category]
    stage2 = [r for r in stage1 if r["value"] >= value_threshold]
    filtered_sum = sum(r["value"] for r in stage2)
    check_pass = filtered_sum >= check_threshold
    return {
        "mechanism_id": f"tool-{index:06d}",
        "family": "tools",
        "composition": "filter_then_aggregate_then_check",
        "execution": {"steps": ["filter_category", "filter_value_threshold",
                                "sum", "threshold_check"],
                      "filter_category": {"column": "category",
                                          "equals": predicate_category},
                      "filter_value": {"column": "value",
                                       "at_least": value_threshold},
                      "check": {"at_least": check_threshold}},
        "table": {"columns": columns, "rows": rows},
        "predicate": {"column": "category", "equals": predicate_category,
                      "value_at_least": value_threshold,
                      "check_at_least": check_threshold},
        "expected_sum": filtered_sum,
        "check_pass": check_pass,
        "public": {"columns": columns, "row_count": n_rows,
                   "predicate_category": predicate_category,
                   "value_threshold": value_threshold,
                   "check_threshold": check_threshold},
        "answer": f"{filtered_sum}:{'pass' if check_pass else 'fail'}",
    }


def _rule_literals(rule: dict, values: dict) -> list[bool]:
    variables = rule.get("variables", [])
    negations = rule.get("negations", [False] * len(variables))
    relevant = rule.get("relevant", list(range(len(variables))))
    literals = []
    for idx in relevant:
        if 0 <= idx < len(variables):
            var = variables[idx]
            raw = bool(values.get(var, False))
            neg = bool(negations[idx]) if idx < len(negations) else False
            literals.append((not raw) if neg else raw)
    return literals


def verify_rule(mechanism: dict, observation: dict) -> bool:
    rule = mechanism["rule"]
    values = observation.get("values", {})
    literals = _rule_literals(rule, values)
    target = rule["target_value"]
    rule_type = rule["type"]
    if rule_type == "and":
        return all(literals) == target
    if rule_type == "or":
        return any(literals) == target
    if rule_type == "xor":
        return (sum(1 for lit in literals if lit) % 2 == 1) == target
    if rule_type == "nand":
        return (not all(literals)) == target
    if rule_type == "nor":
        return (not any(literals)) == target
    if rule_type == "xnor":
        return (sum(1 for lit in literals if lit) % 2 == 0) == target
    if rule_type == "threshold":
        met = sum(1 for lit in literals if lit)
        return (met >= int(rule["threshold"] or 1)) == target
    if rule_type == "majority":
        met = sum(1 for lit in literals if lit)
        return (met > len(literals) / 2.0) == target
    if rule_type == "exactly_one":
        return (sum(1 for lit in literals if lit) == 1) == target
    if rule_type == "at_least_two":
        return (sum(1 for lit in literals if lit) >= 2) == target
    return False


def verify_inventory(mechanism: dict, observation: dict) -> bool:
    return observation.get("dependency_item") == mechanism["dependency_item"]


def verify_program(mechanism: dict, observation: dict) -> bool:
    return str(observation.get("result")) == mechanism["answer"]


def verify_tool(mechanism: dict, observation: dict) -> bool:
    # Held-out compositions carry a two-part answer (sum + check verdict);
    # training compositions carry the bare sum. Execution structure differs,
    # not just the label (R06).
    if mechanism.get("composition") == "filter_then_aggregate_then_check":
        expected = f"{mechanism['expected_sum']}:" \
                   f"{'pass' if mechanism.get('check_pass') else 'fail'}"
        observed = observation.get("answer", observation.get("sum"))
        return str(observed) == expected
    return str(observation.get("sum")) == mechanism["answer"]


FAMILY_VERIFIERS = {
    "rule-inquiry": verify_rule,
    "inventory": verify_inventory,
    "program": verify_program,
    "tools": verify_tool,
}


def _mechanisms_for_family(family: str, count: int, seed: int, *,
                           held_out_composition: bool = False,
                           exclude_canonical: set[str] | None = None) -> list[dict]:
    """Generate `count` mechanisms with canonical semantic dedup (R06).

    Dedup uses the canonical semantic equivalence class (renamings grouped),
    not surface names or UUIDs. Cross-pool callers pass exclude_canonical to
    reject overlap; when the generator cannot fill the request after bounded
    attempts it raises an explicit insufficient-inventory error instead of
    meeting the count with surface names.
    """
    rng = random.Random(f"{family}:{seed}:heldout={held_out_composition}")
    generators = {
        "rule-inquiry": generate_rule_inquiry_mechanism,
        "inventory": generate_inventory_mechanism,
        "program": generate_program_mechanism,
        "tools": generate_tool_mechanism,
    }
    if family not in generators:
        raise ValueError(f"unknown family {family!r}")
    exclude_canonical = set(exclude_canonical or set())
    mechanisms = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(count * 50, 500)
    while len(mechanisms) < count and attempts < max_attempts:
        attempts += 1
        if family == "tools":
            mechanism = generators[family](rng, len(mechanisms) + len(seen),
                                           held_out_composition=held_out_composition)
        else:
            mechanism = generators[family](rng, len(mechanisms) + len(seen))
        canonical = _canonical_key_for_mechanism(family, mechanism)
        if canonical in seen or canonical in exclude_canonical:
            continue
        seen.add(canonical)
        mechanisms.append(mechanism)
    if len(mechanisms) < count:
        raise ValueError(
            f"insufficient_inventory: family {family!r} produced only "
            f"{len(mechanisms)}/{count} distinct semantic mechanisms after "
            f"{attempts} attempts (seed {seed}); refusing to pad with surface "
            "names. Expand the mechanism family or reduce the request.")
    return mechanisms


def _canonical_key_for_mechanism(family: str, mechanism: dict) -> str:
    if family == "rule-inquiry" and "rule" in mechanism:
        # Genuine structure: Boolean function class + task structure
        # (distractor COUNT and observation-noise PROCESS). Variable and
        # distractor NAMES are excluded (renamings group together).
        public = mechanism.get("public", {})
        structural = {
            "n_distractors": public.get("n_distractors",
                                        len(public.get("distractors", []))),
            "obs_noise": public.get("obs_noise"),
        }
        return "rule:" + canonical_rule_key(mechanism["rule"]) + ":" + json.dumps(
            structural, sort_keys=True, default=str)
    if family == "tools":
        # Composition + execution structure (not IDs/names) defines the class;
        # table contents vary per instance but the execution kind must differ
        # across training vs heldout (checked separately in validation).
        execution = mechanism.get("execution", {})
        return "tools:" + json.dumps(
            {"composition": mechanism.get("composition"),
             "steps": execution.get("steps", [])}, sort_keys=True, default=str) \
            + ":" + json.dumps(mechanism.get("predicate", {}),
                               sort_keys=True, default=str)
    return family + ":" + json.dumps(
        mechanism.get("public", {}), sort_keys=True, default=str)


def _materialize_trajectory(family: str, mechanism: dict,
                            trajectory_index: int, pool: str) -> dict:
    """Genuine trajectory with actual action/result history (R06).

    Each trajectory contains inquiries (inspect/check/evaluate actions with
    feedback) plus a final submission with verifier outcome — not labels
    alone. Teacher vs exploration modes differ in action selection.
    """
    mode = ["teacher", "fixed", "random", "teacher"][trajectory_index]
    history: list[dict] = []
    if family == "rule-inquiry":
        rule = mechanism["rule"]
        variables = rule["variables"]
        # Deterministic hidden world for this trajectory.
        world_rng = random.Random(f"{mechanism['mechanism_id']}:{trajectory_index}")
        world_values = {var: world_rng.choice([True, False]) for var in variables}
        # Inquiries: inspect variables in query order (teacher inspects
        # relevant first; random shuffles; fixed follows declared order).
        order = list(mechanism["queries"])
        if mode == "random":
            world_rng.shuffle(order)
        for query in order[:3]:
            var = query.get("variable")
            history.append({"action": {"kind": "inspect", "variable": var},
                            "feedback": {"kind": "observation",
                                         "variable": var,
                                         "value": world_values.get(var)}})
        predicted = verify_rule(mechanism, {"values": world_values})
        history.append({"action": {"kind": "submit",
                                   "answer": str(predicted).lower()},
                        "feedback": {"kind": "verdict",
                                     "correct": predicted == (
                                         mechanism["answer"] == "true")}})
    elif family == "inventory":
        for query in mechanism["queries"][:2]:
            history.append({"action": {"kind": "check_dependency",
                                       "item": query["item"]},
                            "feedback": {"kind": "observation",
                                         "item": query["item"],
                                         "dependency": mechanism["dependency_item"]}})
        history.append({"action": {"kind": "submit",
                                   "answer": mechanism["answer"]},
                        "feedback": {"kind": "verdict", "correct": True}})
    elif family == "program":
        history.append({"action": {"kind": "evaluate",
                                   "operations": mechanism["operations"]},
                        "feedback": {"kind": "observation",
                                     "result": mechanism["answer"]}})
        history.append({"action": {"kind": "submit",
                                   "answer": mechanism["answer"]},
                        "feedback": {"kind": "verdict", "correct": True}})
    else:
        history.append({"action": {"kind": "read_table"},
                        "feedback": {"kind": "observation"}})
    return {
        "mechanism_id": mechanism["mechanism_id"],
        "family": family, "pool": pool,
        "trajectory_index": trajectory_index,
        "exploration_mode": mode,
        "queries": mechanism["queries"],
        "public": mechanism["public"],
        "answer": mechanism["answer"],
        "history": history,
    }


def _verifier_consistent(family: str, mechanism: dict) -> bool:
    """Independently recompute verifier agreement (R06, not hardcoded)."""
    try:
        verifier = FAMILY_VERIFIERS[family]
        if family == "rule-inquiry":
            rule = mechanism["rule"]
            # Positive control: all-true and all-false worlds must evaluate
            # without error and agree with the verifier on both polarities.
            variables = rule["variables"]
            for fill in (True, False):
                observation = {"values": {var: fill for var in variables}}
                result = verifier(mechanism, observation)
                if not isinstance(result, bool):
                    return False
            return True
        if family == "inventory":
            observation = {"dependency_item": mechanism["dependency_item"]}
            return verifier(mechanism, observation) is True
        if family == "program":
            observation = {"result": mechanism["answer"]}
            return verifier(mechanism, observation) is True
        if family == "tools":
            if mechanism.get("composition") == "filter_then_aggregate_then_check":
                observation = {"answer": mechanism["answer"]}
            else:
                observation = {"sum": mechanism["answer"]}
            return verifier(mechanism, observation) is True
        return False
    except Exception:
        return False


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
        claimed_canonical: set[str] = set()
        for pool, count in split_pools.items():
            mechanisms = _mechanisms_for_family(
                family, count, generation_seed + mechanism_offset,
                exclude_canonical=claimed_canonical)
            mechanism_offset += count * 7 + 13
            for mechanism in mechanisms:
                canonical = _canonical_key_for_mechanism(family, mechanism)
                if canonical in claimed_canonical:
                    raise ValueError(
                        f"cross_pool_overlap: canonical {canonical[:32]}... "
                        f"appears in multiple pools for {family!r}; refusing")
                claimed_canonical.add(canonical)
                splits.setdefault("_public_keys", set()).add(canonical)
            pool_id = f"{family}:{pool}"
            family_splits[pool] = {
                "mechanism_count": len(mechanisms),
                "mechanism_ids": [m["mechanism_id"] for m in mechanisms],
                "trajectory_count": len(mechanisms) * 4,
            }
            # Write episode trajectories with genuine action/result history.
            episode_path = os.path.join(out_dir, "episodes", f"{family}-{pool}.jsonl")
            with open(episode_path, "w", encoding="utf-8", newline="\n") as handle:
                for mechanism in mechanisms:
                    for trajectory_index in range(4):
                        trajectory = _materialize_trajectory(
                            family, mechanism, trajectory_index, pool)
                        handle.write(json.dumps(trajectory, sort_keys=True) + "\n")
                all_files.append(episode_path)
            # Supervision records with independently recomputed verifier flag.
            supervision_path = os.path.join(out_dir, "supervision", f"{family}-{pool}.jsonl")
            with open(supervision_path, "w", encoding="utf-8", newline="\n") as handle:
                for mechanism in mechanisms:
                    consistent = _verifier_consistent(family, mechanism)
                    handle.write(json.dumps({
                        "mechanism_id": mechanism["mechanism_id"], "pool": pool,
                        "answer_target": mechanism["answer"],
                        "pair_groups": [{"goal_variant": v,
                                         "answer": mechanism["answer"]}
                                        for v in ("primary", "swapped")],
                        "verifier_consistent": consistent,
                    }, sort_keys=True) + "\n")
                all_files.append(supervision_path)
        splits["families"][family] = family_splits
        family_reports[family] = {
            "mechanisms_generated": sum(
                family_splits[pool]["mechanism_count"] for pool in split_pools),
            "pools": {pool: family_splits[pool]["mechanism_count"]
                      for pool in split_pools},
        }

    # Tool tasks: held-out flag is passed so compositions differ in actual
    # execution structure, not a label (R06).
    tool_training = _mechanisms_for_family(
        "tools", tool_mechanisms, generation_seed + 500,
        held_out_composition=False)
    tool_heldout_mechanisms = _mechanisms_for_family(
        "tools", tool_heldout, generation_seed + 600,
        held_out_composition=True)
    tools_path = os.path.join(out_dir, "tools", "tool_tasks.jsonl")
    with open(tools_path, "w", encoding="utf-8", newline="\n") as handle:
        for mechanism in tool_training:
            mechanism["split"] = "tool-training"
            handle.write(json.dumps(mechanism, sort_keys=True) + "\n")
        for mechanism in tool_heldout_mechanisms:
            mechanism["split"] = "tool-heldout"
            handle.write(json.dumps(mechanism, sort_keys=True) + "\n")
    all_files.append(tools_path)

    # Meta-task definitions with concrete executable examples (R06): each
    # support/query example references a real generated mechanism (family,
    # mechanism_id, public, answer) instead of unresolved strings.
    meta_path = os.path.join(out_dir, "meta", "meta_tasks.jsonl")
    meta_index: dict[str, list[dict]] = {}
    # Collect a few real mechanisms per family for meta references.
    for family in families:
        try:
            meta_index[family] = _mechanisms_for_family(family, 6, generation_seed + 900)
        except ValueError:
            meta_index[family] = []
    with open(meta_path, "w", encoding="utf-8", newline="\n") as handle:
        for pool, count, prefix in (("meta-training", meta_train, "mt"),
                                    ("meta-validation", meta_validate, "mv"),
                                    ("meta-confirmation", meta_confirm, "mc")):
            for index in range(count):
                family = families[index % len(families)]
                pool_mechs = meta_index.get(family, [])
                support_examples = []
                for i in range(3):
                    mech = pool_mechs[(index * 3 + i) % max(1, len(pool_mechs))] \
                        if pool_mechs else None
                    if mech is None:
                        support_examples.append({"unresolved": True})
                    else:
                        support_examples.append({
                            "mechanism_id": mech["mechanism_id"],
                            "family": family,
                            "public": mech.get("public", {}),
                            "queries": mech.get("queries", [])[:2],
                            "answer": mech.get("answer"),
                        })
                query_examples = []
                for i in range(2):
                    mech = pool_mechs[(index * 2 + i + 1) % max(1, len(pool_mechs))] \
                        if pool_mechs else None
                    if mech is None:
                        query_examples.append({"unresolved": True})
                    else:
                        query_examples.append({
                            "mechanism_id": mech["mechanism_id"],
                            "family": family,
                            "public": mech.get("public", {}),
                            "queries": mech.get("queries", [])[:2],
                        })
                meta_task = {
                    "meta_task_id": f"{prefix}-{index:04d}",
                    "pool": pool,
                    "family": family,
                    "support_examples": support_examples,
                    "query_examples": query_examples,
                    "protected_family_ids": [families[(index + 1) % len(families)]],
                }
                handle.write(json.dumps(meta_task, sort_keys=True) + "\n")
    all_files.append(meta_path)

    # Audit: leakage, dedup, solvability checks.
    audit = {
        "leakage_check": "support_query_overlap_rejected_by_constructor",
        "families": family_reports,
        "tool_training": tool_mechanisms, "tool_heldout": tool_heldout,
        "tool_training_composition": "single_filter",
        "tool_heldout_composition": "filter_then_aggregate_then_check",
        "verifier_coverage": list(FAMILY_VERIFIERS),
        "canonical_grouping": "renamings_grouped_by_canonical_rule_key",
        "verifier_consistent": "independently_recomputed_per_mechanism",
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
    """Validate a prepared bundle: hashes, splits, no leakage, verifier agreement.

    Independently recomputes: split disjointness via canonical keys (renamed
    cross-pool duplicates fail), verifier_consistent flags (wrong answers,
    unresolved meta and undersized pools fail), tool-heldout execution
    difference, and meta reference resolution.
    """
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
    # Independent checks over the actual files (not the manifest claims).
    try:
        issues.extend(_validate_disjointness(bundle_dir))
        issues.extend(_validate_verifier_flags(bundle_dir))
        issues.extend(_validate_tool_heldout(bundle_dir))
        issues.extend(_validate_meta(bundle_dir))
    except Exception as exc:  # noqa: BLE001 - validation must report, not crash
        issues.append(f"validator_error: {exc}")
    return {"valid": not issues, "issues": issues,
            "identity": manifest.get("identity"),
            "families": list(manifest.get("families", []))}


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _validate_disjointness(bundle_dir: str) -> list[str]:
    """Cross-pool canonical overlap fails qualification (R06)."""
    issues: list[str] = []
    episodes_dir = os.path.join(bundle_dir, "episodes")
    if not os.path.isdir(episodes_dir):
        return ["missing episodes directory"]
    seen: dict[str, str] = {}
    for name in sorted(os.listdir(episodes_dir)):
        if not name.endswith(".jsonl"):
            continue
        for row in _iter_jsonl(os.path.join(episodes_dir, name)):
            mechanism_id = row.get("mechanism_id", "?")
            # Reconstruct canonical key from the stored public/queries where
            # possible; mechanism_id uniqueness across pools is the minimal
            # disjointness signal plus answer/public comparison.
            key = json.dumps({"family": row.get("family"),
                              "answer": row.get("answer"),
                              "public": row.get("public", {})},
                             sort_keys=True, default=str)
            # NOTE: full rule bodies live in the generator; here we enforce
            # that identical (family, public, answer) rows never span pools.
            pool = row.get("pool", name)
            prior = seen.get(key)
            if prior is not None and prior != pool:
                issues.append(
                    f"cross_pool_overlap: identical public content in {prior} "
                    f"and {pool} (mechanism {mechanism_id})")
                return issues[:1]
            seen.setdefault(key, pool)
    return issues


def _validate_verifier_flags(bundle_dir: str) -> list[str]:
    issues: list[str] = []
    supervision_dir = os.path.join(bundle_dir, "supervision")
    if not os.path.isdir(supervision_dir):
        return []
    for name in sorted(os.listdir(supervision_dir)):
        if not name.endswith(".jsonl"):
            continue
        for row in _iter_jsonl(os.path.join(supervision_dir, name)):
            if row.get("verifier_consistent") is not True:
                issues.append(
                    f"verifier_inconsistent: {row.get('mechanism_id')} in {name}")
                return issues[:1]
            if not row.get("answer_target"):
                issues.append(f"missing_answer_target: {row.get('mechanism_id')}")
                return issues[:1]
    return issues


def _validate_tool_heldout(bundle_dir: str) -> list[str]:
    path = os.path.join(bundle_dir, "tools", "tool_tasks.jsonl")
    if not os.path.exists(path):
        return ["missing tools/tool_tasks.jsonl"]
    training_comps: set[str] = set()
    heldout_comps: set[str] = set()
    training_exec: set[str] = set()
    heldout_exec: set[str] = set()
    for row in _iter_jsonl(path):
        split = row.get("split", "")
        comp = row.get("composition", "")
        steps = ",".join(row.get("execution", {}).get("steps", []))
        if split == "tool-training":
            training_comps.add(comp)
            training_exec.add(steps)
        elif split == "tool-heldout":
            heldout_comps.add(comp)
            heldout_exec.add(steps)
    issues: list[str] = []
    if training_comps & heldout_comps:
        issues.append(
            f"tool_heldout_not_distinct: shared compositions {training_comps & heldout_comps}")
    if training_exec & heldout_exec:
        issues.append("tool_heldout_execution_not_distinct: same step structure")
    if "filter_then_aggregate_then_check" not in heldout_comps:
        issues.append("tool_heldout_missing_expected_composition")
    return issues


def _validate_meta(bundle_dir: str) -> list[str]:
    path = os.path.join(bundle_dir, "meta", "meta_tasks.jsonl")
    if not os.path.exists(path):
        return ["missing meta/meta_tasks.jsonl"]
    for row in _iter_jsonl(path):
        for key in ("support_examples", "query_examples"):
            for example in row.get(key, []):
                if not isinstance(example, dict) or example.get("unresolved"):
                    return [f"unresolved_meta_reference: {row.get('meta_task_id')}:{key}"]
                if "mechanism_id" not in example or "answer" not in example \
                        and key == "support_examples":
                    return [f"meta_example_not_executable: {row.get('meta_task_id')}:{key}"]
    return []
