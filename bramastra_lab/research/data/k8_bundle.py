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

from bramastra_lab.research.contracts.core import content_identity, file_sha256

BUNDLE_SCHEMA = "bramastra-k8-data/v1"

# Registered inquiry budget: the number of permitted inspect/check actions
# within the base evaluation episode (submit excluded). Every admitted
# mechanism must carry an information-sufficiency witness within this budget.
K8_INQUIRY_BUDGET = 4


def _hash_file(path: str) -> str:
    """Hash a bundle member with bounded memory and a closed file."""
    return file_sha256(path)


def generate_rule_inquiry_mechanism(rng: random.Random, mechanism_index: int) -> dict:
    """Hidden sampled values over a PUBLIC rule function (F02).

    The public goal carries the complete rule function — rule type, which
    variables are relevant, their negation pattern, threshold and target
    polarity — while the per-episode sampled variable values stay hidden and
    are revealed only by inspect actions. Because at most
    K8_INQUIRY_BUDGET relevant variables exist, inspecting them always
    determines the unique verifier ruling within budget (proved per
    mechanism by `rule_information_witness`). Diversity comes from genuine
    Boolean-function differences, NOT surface names: variable count (2-6),
    rule type (10 types), relevant subset, negation pattern, threshold,
    target polarity, distractor COUNT and observation noise. Names are
    randomized surface labels excluded from the canonical semantic key.
    """
    n_vars = rng.randint(2, 5)
    rule_type = rng.choice([
        "and", "or", "xor", "nand", "nor", "xnor",
        "threshold", "majority", "exactly_one", "at_least_two"])
    # Short surface names: variable identities are arbitrary labels (R06),
    # and every token spent on them is taken from the evidence context.
    vars_ = [f"v{i}" for i in range(n_vars)]
    # Relevant subset: which variables actually determine the output. The
    # count is capped at the inquiry budget so the function is always
    # decidable from permitted observations (F02 sufficiency witness).
    k_relevant = rng.randint(1, min(K8_INQUIRY_BUDGET, n_vars))
    relevant_idx = sorted(rng.sample(range(n_vars), k_relevant))
    # Per-variable negation pattern (genuine function difference).
    negations = [rng.choice([False, False, True]) for _ in range(n_vars)]
    threshold = rng.randint(1, max(1, k_relevant)) \
        if rule_type in ("threshold", "at_least_two") else None
    target_value = rng.choice([True, False])
    n_distractors = rng.randint(1, 5)
    distractors = [f"d{i}" for i in range(n_distractors)]
    # Only implemented noise processes are generated (none, flip_one); the
    # live environment executes exactly what the public spec declares.
    obs_noise = rng.choice(["none", "flip_one"])
    # Query order: relevant variables first, then non-relevant, then one
    # distractor — so the declared order is a sufficient strategy.
    non_relevant = [v for i, v in enumerate(vars_) if i not in relevant_idx]
    query_order = [vars_[i] for i in relevant_idx] + non_relevant
    queries = [{"kind": "inspect", "variable": v} for v in query_order]
    queries.append({"kind": "inspect", "variable": rng.choice(distractors)})
    rule = {"type": rule_type, "variables": vars_, "threshold": threshold,
            "target_value": target_value, "relevant": relevant_idx,
            "negations": negations}
    # Public function spec (F02): rule type, relevant variables with their
    # negation flags, threshold and target polarity. Non-relevant negation
    # flags cannot affect the verifier ruling and are not part of the
    # function; sampled values are the only hidden state. Keys are compact
    # surface labels (same convention as the renderer's short keys) so the
    # spec plus the inquiry evidence fits the 512-token campaign context.
    # Cross-checked by `assert_public_function_matches` at build time.
    public = {
        "rt": rule_type,
        "rel": [vars_[i] for i in relevant_idx],
        "neg": [negations[i] for i in relevant_idx],
        "tv": target_value,
        "noise": obs_noise,
        "nd": n_distractors,
    }
    if threshold is not None:
        public["thr"] = threshold
    return {
        "mechanism_id": f"rule-{mechanism_index:06d}",
        "family": "rule-inquiry",
        "rule": rule,
        "queries": queries,
        "complementary_pair": [{"variable": vars_[relevant_idx[0]]},
                               {"variable": vars_[relevant_idx[1]]}]
        if len(relevant_idx) >= 2 else [],
        "public": public,
        "answer": str(target_value).lower(),
    }


def rule_information_witness(public: dict, *,
                             budget: int = K8_INQUIRY_BUDGET) -> dict:
    """Bounded information-sufficiency witness for a rule-inquiry task (F02).

    Proves from the PUBLIC function spec alone that some strategy within the
    inquiry budget determines the unique verifier ruling for every hidden
    world: no two hidden worlds whose permitted observation streams are
    identical can demand different unique answers. The witness strategy
    inspects exactly the public relevant variables in declared order; the
    declared observation-noise process is applied to the stream exactly as
    the live environment executes it.

    Returns a witness record; raises ValueError when the task is not
    solvable within budget (such mechanisms are refused, never admitted).
    """
    required = ("rel", "neg", "tv", "rt", "noise")
    missing = [key for key in required if key not in public]
    if missing:
        raise ValueError(
            f"public rule spec is incomplete (missing {missing}); the "
            "function must be public for the task to be solvable")
    relevant = list(public["rel"])
    if not relevant or len(relevant) > budget:
        raise ValueError(
            f"witness refused: {len(relevant)} relevant variables exceed the "
            f"{budget}-inquiry budget")
    negations = list(public["neg"])
    noise = public.get("noise", "none")
    if noise not in ("none", "flip_one"):
        raise ValueError(f"undeclared observation-noise process {noise!r}")

    def render_stream(world: dict[str, bool]) -> tuple:
        """Permitted observations under the declared noise process.

        Mirrors RuleInquiryEnv._apply_action: flip_one inverts the value of
        the first variable reported in the episode.
        """
        stream = []
        for position, var in enumerate(relevant):
            value = bool(world[var])
            if noise == "flip_one" and position == 0:
                value = not value
            stream.append(value)
        return tuple(stream)

    def rule_verdict(world: dict[str, bool]) -> bool:
        literals = []
        for position, var in enumerate(relevant):
            raw = bool(world[var])
            neg = bool(negations[position]) if position < len(negations) \
                else False
            literals.append((not raw) if neg else raw)
        # Reuse the independent verifier semantics over the relevant values
        # (non-relevant variables cannot influence the ruling).
        return verify_rule({"rule": _rule_from_public(public)},
                           {"values": dict(world)})

    stream_verdicts: dict[tuple, bool] = {}
    checked = 0
    for mask in range(1 << len(relevant)):
        world = {var: bool((mask >> position) & 1)
                 for position, var in enumerate(relevant)}
        stream = render_stream(world)
        verdict = rule_verdict(world)
        checked += 1
        if stream in stream_verdicts and stream_verdicts[stream] != verdict:
            raise ValueError(
                "information insufficiency: two hidden worlds with "
                "indistinguishable permitted observations demand different "
                "unique answers")
        stream_verdicts[stream] = verdict
    return {"sufficient": True, "strategy": [
                {"kind": "inspect", "variable": var} for var in relevant],
            "budget": budget, "checked_worlds": checked,
            "noise_process": noise}


def _rule_from_public(public: dict) -> dict:
    """Rebuild a verifier-equivalent rule object from the public function spec.

    The rebuilt rule carries only the relevant variables (non-relevant ones
    cannot influence the ruling), so `verify_rule` executes the exact public
    function the learner sees.
    """
    relevant = list(public["rel"])
    negations = [bool(flag) for flag in public.get("neg", [])]
    return {"type": public["rt"], "variables": list(relevant),
            "threshold": public.get("thr"),
            "target_value": public["tv"],
            "relevant": list(range(len(relevant))),
            "negations": negations}


def assert_public_function_matches(mechanism: dict) -> None:
    """The public function spec must equal the private rule function (F02).

    Guards against drift between the task the learner sees and the function
    the verifier executes — the base family admits no private function.
    """
    public = mechanism["public"]
    private = mechanism["rule"]
    variables = list(private["variables"])
    relevant_idx = list(private.get("relevant", range(len(variables))))
    private_relevant = [variables[i] for i in relevant_idx]
    private_negations = [bool(private["negations"][i]) for i in relevant_idx]
    if list(public["rel"]) != private_relevant:
        raise ValueError(
            f"public relevant-variable set disagrees with the private rule "
            f"for {mechanism.get('mechanism_id')}; the base family must not "
            "carry a private function")
    if [bool(flag) for flag in public["neg"]] != private_negations:
        raise ValueError(
            f"public negation pattern disagrees with the private rule for "
            f"{mechanism.get('mechanism_id')}")
    if public.get("rt") != private.get("type"):
        raise ValueError(
            f"public rule type disagrees with the private rule for "
            f"{mechanism.get('mechanism_id')}")
    for public_key, private_key in (("thr", "threshold"), ("tv", "target_value")):
        if public.get(public_key) != private.get(private_key):
            raise ValueError(
                f"public function spec disagrees with the private rule on "
                f"{private_key!r} for {mechanism.get('mechanism_id')}; the "
                "base family must not carry a private function")


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
    """Resource dependency: which container/recipe satisfies the goal.

    The dependency is the private state; one check on the (public) goal item
    reveals it, so the task is always decidable within the inquiry budget.
    Queries are ordered goal-item first so the declared order is a
    sufficient strategy.
    """
    n_items = rng.randint(2, 4)
    items = [f"item_{rng.randint(100, 999)}" for _ in range(n_items)]
    goal_item = rng.choice(items)
    dependency = rng.choice(items)
    others = [item for item in items if item != goal_item]
    queries = [{"kind": "check_dependency", "item": goal_item}] + [
        {"kind": "check_dependency", "item": item} for item in others]
    return {
        "mechanism_id": f"inv-{mechanism_index:06d}",
        "family": "inventory",
        "items": items, "goal_item": goal_item, "dependency_item": dependency,
        "queries": queries,
        "public": {"items": items, "goal": f"secure {goal_item}",
                   "goal_item": goal_item},
        "answer": dependency,
    }


def inventory_information_witness(public: dict, *,
                                  budget: int = K8_INQUIRY_BUDGET) -> dict:
    """Sufficiency witness: checking the goal item reveals the dependency."""
    goal_item = public.get("goal_item")
    items = list(public.get("items", []))
    if goal_item not in items:
        raise ValueError("inventory witness refused: goal item not public")
    if len(items) > budget + 1:
        raise ValueError(
            f"inventory witness refused: {len(items)} items cannot be "
            f"covered within the {budget}-inquiry budget")
    return {"sufficient": True, "strategy": [
                {"kind": "check_dependency", "item": goal_item}],
            "budget": budget, "checked_worlds": 2}


def program_information_witness(public: dict, *,
                                budget: int = K8_INQUIRY_BUDGET) -> dict:
    """Sufficiency witness: one evaluate at the public start value reveals
    the program result (the operations are public; the result is not)."""
    if "start_value" not in public:
        raise ValueError("program witness refused: start value not public")
    return {"sufficient": True, "strategy": [
                {"kind": "evaluate", "input": public["start_value"]}],
            "budget": budget, "checked_worlds": 1}


FAMILY_WITNESSES = {
    "rule-inquiry": rule_information_witness,
    "inventory": inventory_information_witness,
    "program": program_information_witness,
}


def information_sufficiency_witness(family: str, public: dict, *,
                                    budget: int = K8_INQUIRY_BUDGET) -> dict:
    """Dispatch the family witness over the PUBLIC spec alone (F02).

    Bundles deliberately store no mechanism bodies, so the witness must be
    computable from the public payload — which is also the acceptance form
    the build verifier replays.
    """
    witness = FAMILY_WITNESSES.get(family)
    if witness is None:
        raise ValueError(f"no information witness for family {family!r}")
    return witness(public, budget=budget)


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
    # Distinct ID namespace: held-out mechanism IDs must never collide with
    # training IDs, or the protected-exclusion proof cannot distinguish the
    # pools (leak protection would be unpassable).
    value_threshold = rng.randint(10, 30)
    check_threshold = rng.randint(20, 80)
    stage1 = [r for r in rows if r["category"] == predicate_category]
    stage2 = [r for r in stage1 if r["value"] >= value_threshold]
    filtered_sum = sum(r["value"] for r in stage2)
    check_pass = filtered_sum >= check_threshold
    return {
        "mechanism_id": f"toolh-{index:06d}",
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
        # Cross-pool separation is CLASS-level (execution structure): a held
        # -out pool must never contain a mechanism class that training saw.
        if canonical in exclude_canonical:
            continue
        # Within-pool uniqueness is INSTANCE-level for tools: the class has
        # only three single-filter predicates, so separate scored instances
        # (distinct tables/answers) of the same class are legitimate pool
        # content; every other family dedups at the semantic-class level.
        instance_key = canonical if family != "tools" else \
            "tool-instance:" + json.dumps(
                {"composition": mechanism.get("composition"),
                 "predicate": mechanism.get("predicate", {}),
                 "table": mechanism.get("table", {})}, sort_keys=True,
                default=str)
        if instance_key in seen:
            continue
        seen.add(instance_key)
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
            "nd": public.get("nd", 0),
            "noise": public.get("noise"),
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
    """Genuine trajectory replayed under live-environment semantics (F02).

    Each trajectory contains inquiries (inspect/check/evaluate actions with
    the same feedback shape the live environment emits) plus a final
    submission with the verdict the independent verifier rules on the actual
    hidden world — computed by execution, never asserted. The scored
    `answer` is the unique verifier ruling for THIS episode's hidden world
    (rule-inquiry), or the mechanism's unique answer where it is
    world-independent (inventory/program/tools). Teacher vs exploration
    modes differ in action selection only.
    """
    mode = ["teacher", "fixed", "random", "teacher"][trajectory_index]
    history: list[dict] = []
    if family == "rule-inquiry":
        rule = mechanism["rule"]
        variables = rule["variables"]
        # Deterministic hidden world for this trajectory.
        world_rng = random.Random(f"{mechanism['mechanism_id']}:{trajectory_index}")
        world_values = {var: world_rng.choice([True, False]) for var in variables}
        # Inquiries: inspect variables in query order (relevant-first for
        # teacher/fixed — the sufficient strategy; random shuffles).
        order = list(mechanism["queries"])
        if mode == "random":
            world_rng.shuffle(order)
        noise = mechanism.get("public", {}).get("noise", "none")
        reported: set[str] = set()
        for query in order[:K8_INQUIRY_BUDGET]:
            var = query.get("variable")
            value = world_values.get(var, False)
            # Mirror RuleInquiryEnv: flip_one inverts the first report.
            if noise == "flip_one" and not reported:
                value = not value
            reported.add(var)
            history.append({"action": {"kind": "inspect", "variable": var},
                            "feedback": {"kind": "observation",
                                         "variable": var,
                                         "value": value}})
        actual = verify_rule(mechanism, {"values": dict(world_values)})
        submission = bool(actual)
        # Verdict from actual verifier execution on the submitted answer —
        # the same ruling the live environment emits for this world.
        correct = submission == bool(verify_rule(
            mechanism, {"values": dict(world_values)}))
        history.append({"action": {"kind": "submit", "answer": submission},
                        "feedback": {"kind": "verdict",
                                     "submitted_answer": submission,
                                     "correct": correct}})
        answer = str(actual).lower()
    elif family == "inventory":
        for query in mechanism["queries"][:2]:
            item = query["item"]
            requires = mechanism["dependency_item"] \
                if item == mechanism["goal_item"] else "none"
            history.append({"action": {"kind": "check_dependency",
                                       "item": item},
                            "feedback": {"kind": "observation",
                                         "item": item,
                                         "requires": requires}})
        history.append({"action": {"kind": "submit",
                                   "item": mechanism["answer"]},
                        "feedback": {"kind": "verdict",
                                     "submitted_answer": mechanism["answer"],
                                     "correct": bool(verify_inventory(
                                         mechanism, {"dependency_item":
                                                     mechanism["answer"]}))}})
        answer = mechanism["answer"]
    elif family == "program":
        history.append({"action": {"kind": "evaluate",
                                   "input": mechanism["start_value"]},
                        "feedback": {"kind": "observation",
                                     "input": mechanism["start_value"],
                                     "result": mechanism["answer"]}})
        history.append({"action": {"kind": "submit",
                                   "value": int(mechanism["answer"])},
                        "feedback": {"kind": "verdict",
                                     "submitted_answer": int(
                                         mechanism["answer"]),
                                     "correct": bool(verify_program(
                                         mechanism, {"result":
                                                     mechanism["answer"]}))}})
        answer = mechanism["answer"]
    else:
        history.append({"action": {"kind": "read_table"},
                        "feedback": {"kind": "observation"}})
        answer = mechanism["answer"]
    return {
        "mechanism_id": mechanism["mechanism_id"],
        "family": family, "pool": pool,
        "trajectory_index": trajectory_index,
        "exploration_mode": mode,
        "queries": mechanism["queries"],
        "public": mechanism["public"],
        "answer": answer,
        "history": history,
    }


def _assert_public_separation(family: str, mechanism: dict, trajectory: dict) -> None:
    """Leakage separation at the actual public renderer (D3/F02).

    For rule-inquiry the function is public BY DESIGN (F02): the hidden
    state is the per-episode sampled values, which exist only in the
    environment and never in any serialized row. What must hold everywhere:
    no public payload carries an answer/verdict field, the private unique
    answer (inventory/program/tools) never appears in public, and the rule
    function the verifier executes equals the declared public function.
    """
    public = trajectory.get("public", {})
    if isinstance(public, dict) and ("answer" in public or "verdict" in public
                                     or "correct" in public):
        raise ValueError(
            f"public renderer carries an answer-like field for "
            f"{mechanism.get('mechanism_id')}")
    if family == "rule-inquiry":
        # The verifier's function must equal the public function spec.
        assert_public_function_matches(mechanism)
        # The information witness must hold from the public spec alone.
        rule_information_witness(public)
        # Per-episode sampled values are hidden state: no value map in public.
        for forbidden in ("values", "world", "sampled"):
            if forbidden in public:
                raise ValueError(
                    f"public row carries hidden sampled state under {forbidden!r}")
    elif family == "inventory":
        if "dependency_item" in public:
            raise ValueError("public row carries the private dependency item")
        if public.get("goal_item") != mechanism.get("goal_item"):
            raise ValueError(
                "inventory public must declare the goal item (witness input)")
        inventory_information_witness(public)
    elif family == "program":
        # Operations are public by design (the function); the result is not.
        if "result" in public or "answer" in public:
            raise ValueError("public row carries the program result")
        program_information_witness(public)
    elif family == "tools":
        if "expected_sum" in public or "check_pass" in public:
            raise ValueError("public row carries the tool answer")


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
        pool_mechanisms: dict[str, list[dict]] = {}
        for pool, count in split_pools.items():
            # Cross-pool separation is class-level: a pool may hold several
            # scored instances of one class (tools), but a class seen by an
            # earlier pool is never admitted here.
            previous_pools_claimed = set(claimed_canonical)
            mechanisms = _mechanisms_for_family(
                family, count, generation_seed + mechanism_offset,
                exclude_canonical=previous_pools_claimed)
            mechanism_offset += count * 7 + 13
            for mechanism in mechanisms:
                canonical = _canonical_key_for_mechanism(family, mechanism)
                if canonical in previous_pools_claimed:
                    raise ValueError(
                        f"cross_pool_overlap: canonical {canonical[:32]}... "
                        f"appears in multiple pools for {family!r}; refusing")
                claimed_canonical.add(canonical)
                mechanism["canonical_identity"] = canonical
                splits.setdefault("_public_keys", set()).add(canonical)
            pool_mechanisms[pool] = mechanisms
            pool_id = f"{family}:{pool}"
            family_splits[pool] = {
                "mechanism_count": len(mechanisms),
                "mechanism_ids": [m["mechanism_id"] for m in mechanisms],
                "canonical_identities": sorted(
                    m["canonical_identity"] for m in mechanisms),
                "trajectory_count": len(mechanisms) * 4,
            }
            # Write episode trajectories with genuine action/result history.
            # Each row carries its canonical identity (one declared identity
            # across all pools) plus a renderer-leakage check at write time.
            episode_path = os.path.join(out_dir, "episodes", f"{family}-{pool}.jsonl")
            with open(episode_path, "w", encoding="utf-8", newline="\n") as handle:
                for mechanism in mechanisms:
                    for trajectory_index in range(4):
                        trajectory = _materialize_trajectory(
                            family, mechanism, trajectory_index, pool)
                        trajectory["canonical_identity"] = mechanism["canonical_identity"]
                        _assert_public_separation(family, mechanism, trajectory)
                        handle.write(json.dumps(trajectory, sort_keys=True) + "\n")
                all_files.append(episode_path)
            # Supervision records with independently recomputed verifier flag
            # and the F02 information-sufficiency witness receipt.
            supervision_path = os.path.join(out_dir, "supervision", f"{family}-{pool}.jsonl")
            with open(supervision_path, "w", encoding="utf-8", newline="\n") as handle:
                for mechanism in mechanisms:
                    consistent = _verifier_consistent(family, mechanism)
                    if not consistent:
                        raise ValueError(
                            f"verifier_inconsistent: {mechanism['mechanism_id']} "
                            f"in {family}:{pool}; refusing unqualified bundle")
                    witness = information_sufficiency_witness(
                        family, dict(mechanism["public"]))
                    handle.write(json.dumps({
                        "mechanism_id": mechanism["mechanism_id"], "pool": pool,
                        "canonical_identity": mechanism["canonical_identity"],
                        "answer_target": mechanism["answer"],
                        "pair_groups": [{"goal_variant": v,
                                         "answer": mechanism["answer"]}
                                        for v in ("primary", "swapped")],
                        "verifier_consistent": consistent,
                        "information_witness": witness,
                    }, sort_keys=True) + "\n")
                all_files.append(supervision_path)
        splits["families"][family] = family_splits
        # Stash per-family pools for meta exclusion below.
        splits.setdefault("_pool_mechanisms", {})[family] = {
            pool: [m["canonical_identity"] for m in mechs]
            for pool, mechs in pool_mechanisms.items()}
        splits.setdefault("_claimed", {})[family] = sorted(claimed_canonical)
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

    # Meta-task definitions with concrete executable examples (D3): meta
    # mechanisms are drawn from the SAME globally grouped identity space with
    # primary-pool exclusion (never a separate unexcluded generation), and each
    # support/query example references a real mechanism. Protected families use
    # concrete mechanism references (not bare family names).
    meta_path = os.path.join(out_dir, "meta", "meta_tasks.jsonl")
    meta_excluded: set[str] = set()
    for family in families:
        meta_excluded.update(splits.get("_claimed", {}).get(family, []))
    meta_index: dict[str, list[dict]] = {}
    for family in families:
        try:
            meta_index[family] = _mechanisms_for_family(
                family, meta_train + meta_validate + meta_confirm + 6,
                generation_seed + 900, exclude_canonical=meta_excluded)
            for mech in meta_index[family]:
                meta_excluded.add(_canonical_key_for_mechanism(family, mech))
        except ValueError:
            meta_index[family] = []
    # The meta pool is generated separately from every split pool, so its
    # mechanism ids restart at 000000 and can collide by NAME with a split
    # mechanism that is a different instance entirely. Namespace the ids so
    # a meta reference can never be resolved against split episode rows.
    for family, mechs in meta_index.items():
        for mech in mechs:
            mech["mechanism_id"] = f"{mech['mechanism_id'].split('-')[0]}-meta-" \
                                   f"{mech['mechanism_id'].split('-')[-1]}"
    # Prepared labels for every referenced meta mechanism. Query examples
    # deliberately carry no answer, so the label MUST exist in published
    # prepared data (never invented by the learner, never borrowed from a
    # same-named split mechanism).
    meta_labels: dict[tuple[str, str], dict] = {}

    def _record_meta_label(family: str, mech: dict, pool: str) -> None:
        key = (family, str(mech["mechanism_id"]))
        label = {"mechanism_id": mech["mechanism_id"], "family": family,
                 "pool": pool,
                 "canonical_identity": _canonical_key_for_mechanism(
                     family, mech),
                 "public": mech.get("public", {}),
                 "queries": list(mech.get("queries", [])),
                 "answer": mech.get("answer")}
        existing = meta_labels.get(key)
        if existing is not None and (
                existing["answer"] != label["answer"]
                or existing["canonical_identity"] != label["canonical_identity"]):
            raise ValueError(
                f"meta mechanism {key[1]!r} in family {family!r} has two "
                "prepared labels; refusing an ambiguous reference")
        meta_labels[key] = label

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
                        _record_meta_label(family, mech, pool)
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
                        _record_meta_label(family, mech, pool)
                        query_examples.append({
                            "mechanism_id": mech["mechanism_id"],
                            "family": family,
                            "public": mech.get("public", {}),
                            "queries": mech.get("queries", [])[:2],
                        })
                protected_family = families[(index + 1) % len(families)]
                protected_mechs = meta_index.get(protected_family, [])
                protected_mech = protected_mechs[0] if protected_mechs else None
                if protected_mech is not None:
                    _record_meta_label(protected_family, protected_mech, pool)
                meta_task = {
                    "meta_task_id": f"{prefix}-{index:04d}",
                    "pool": pool,
                    "family": family,
                    "support_examples": support_examples,
                    "query_examples": query_examples,
                    "protected_family_ids": [protected_family],
                    "protected_references": [
                        {"family": protected_family,
                         "mechanism_id": (protected_mech or {}).get(
                             "mechanism_id", "none")}],
                }
                handle.write(json.dumps(meta_task, sort_keys=True) + "\n")
    all_files.append(meta_path)

    meta_label_path = os.path.join(out_dir, "meta", "meta_labels.jsonl")
    with open(meta_label_path, "w", encoding="utf-8", newline="\n") as handle:
        for (family, mechanism_id), label in sorted(meta_labels.items()):
            handle.write(json.dumps(label, sort_keys=True) + "\n")
    all_files.append(meta_label_path)

    # Audit: leakage, dedup, solvability checks.
    audit = {
        "leakage_check": "public_renderer_separation_plus_verifier_agreement",
        "families": family_reports,
        "tool_training": tool_mechanisms, "tool_heldout": tool_heldout,
        "tool_training_composition": "single_filter",
        "tool_heldout_composition": "filter_then_aggregate_then_check",
        "verifier_coverage": list(FAMILY_VERIFIERS),
        "canonical_grouping": "one_declared_canonical_identity_all_pools",
        "verifier_consistent": "independently_recomputed_per_mechanism",
    }

    file_hashes = {os.path.relpath(path, out_dir): _hash_file(path)
                   for path in sorted(all_files)}
    try:
        from bramastra_lab.research.runtime.provenance import source_closure_sha256

        source_bytes_identity = source_closure_sha256()
    except Exception:
        source_bytes_identity = "unavailable"
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "generation_seed": generation_seed,
        "families": families,
        "split_pools": {pool: count for pool, count in split_pools.items()},
        "file_hashes": file_hashes,
        "audit": audit,
        "source_bytes_identity": source_bytes_identity,
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
                         if not key.startswith("_") or key == "_claimed"}
        # Persist the declared canonical identities (not a weaker
        # reconstruction) for cross-pool qualification.
        json.dump(public_splits, handle, indent=2, sort_keys=True,
                  default=list)
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
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
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
        issues.extend(_validate_information_sufficiency(bundle_dir))
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
    """Cross-pool overlap via the ONE declared canonical identity (D3).

    Uses stored canonical_identity rows + splits.json _claimed sets (the same
    function as generation), never a weaker reconstruction. Any canonical
    identity in two pools fails qualification.
    """
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
            key = row.get("canonical_identity")
            if not key:
                return [f"missing_canonical_identity: {mechanism_id} in {name}"]
            pool = row.get("pool", name)
            prior = seen.get(key)
            if prior is not None and prior != pool:
                issues.append(
                    f"cross_pool_overlap: canonical {str(key)[:32]}... in "
                    f"{prior} and {pool} (mechanism {mechanism_id})")
                return issues[:1]
            seen.setdefault(key, pool)
    # Cross-check against the declared splits manifest.
    splits_path = os.path.join(bundle_dir, "splits.json")
    if os.path.exists(splits_path):
        try:
            with open(splits_path, encoding="utf-8") as handle:
                splits = json.load(handle)
            claimed = splits.get("_claimed", {})
            all_claimed: dict[str, str] = {}
            for family, identities in claimed.items():
                for identity in identities:
                    prior = all_claimed.get(identity)
                    if prior is not None and prior != family:
                        issues.append("cross_family_canonical_collision")
                        return issues[:1]
                    all_claimed[identity] = family
        except Exception as exc:  # noqa: BLE001
            issues.append(f"splits_manifest_unreadable: {exc}")
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


def _validate_information_sufficiency(bundle_dir: str) -> list[str]:
    """Replay the information-sufficiency witness from public rows (F02).

    Every episode row's public payload must carry a sufficient bounded
    strategy within the registered inquiry budget. The witness runs from
    the public spec alone (bundles store no mechanism bodies), so this
    validation is independent of generation.
    """
    issues: list[str] = []
    episodes_dir = os.path.join(bundle_dir, "episodes")
    if not os.path.isdir(episodes_dir):
        return ["missing episodes directory"]
    witnessed = 0
    for name in sorted(os.listdir(episodes_dir)):
        if not name.endswith(".jsonl"):
            continue
        for row in _iter_jsonl(os.path.join(episodes_dir, name)):
            family = str(row.get("family", ""))
            if family not in FAMILY_WITNESSES:
                issues.append(
                    f"no_witness_for_family: {family} ({row.get('mechanism_id')})")
                return issues[:1]
            try:
                information_sufficiency_witness(
                    family, dict(row.get("public") or {}))
            except ValueError as exc:
                issues.append(
                    f"information_insufficient: {row.get('mechanism_id')}: "
                    f"{exc}")
                return issues[:1]
            witnessed += 1
    return issues


def _load_meta_labels(bundle_dir: str) -> dict[tuple[str, str], dict]:
    """Prepared labels for the meta pool, keyed by (family, mechanism_id).

    The meta pool is generated independently of every split pool, so its
    mechanism ids are namespaced; these labels are the ONLY legitimate
    source for a query example's answer.
    """
    path = os.path.join(bundle_dir, "meta", "meta_labels.jsonl")
    labels: dict[tuple[str, str], dict] = {}
    if not os.path.exists(path):
        return labels
    for row in _iter_jsonl(path):
        key = (str(row.get("family", "")), str(row.get("mechanism_id", "")))
        if not key[1]:
            continue
        existing = labels.get(key)
        if existing is not None and existing.get("answer") != row.get("answer"):
            raise ValueError(
                f"meta label {key[1]!r} in family {key[0]!r} is ambiguous")
        labels[key] = row
    return labels


def _validate_meta(bundle_dir: str) -> list[str]:
    path = os.path.join(bundle_dir, "meta", "meta_tasks.jsonl")
    if not os.path.exists(path):
        return ["missing meta/meta_tasks.jsonl"]
    try:
        labels = _load_meta_labels(bundle_dir)
    except ValueError as exc:
        return [f"meta_label_ambiguous: {exc}"]
    if not labels:
        return ["missing meta/meta_labels.jsonl: no prepared meta labels"]
    issues: list[str] = []
    for row in _iter_jsonl(path):
        for key in ("support_examples", "query_examples"):
            for example in row.get(key, []):
                if not isinstance(example, dict) or example.get("unresolved"):
                    return [f"unresolved_meta_reference: {row.get('meta_task_id')}:{key}"]
                if "mechanism_id" not in example or "answer" not in example \
                        and key == "support_examples":
                    return [f"meta_example_not_executable: {row.get('meta_task_id')}:{key}"]
        # Every query label and protected reference must resolve to exactly
        # one prepared label: E5 may never invent a label or borrow one from
        # a same-named mechanism in another pool.
        referenced: list[tuple[str, str, str]] = []
        for example in row.get("query_examples", []):
            referenced.append((str(row.get("meta_task_id")), "query",
                               str(example.get("mechanism_id", ""))))
        for ref in row.get("protected_references", []):
            referenced.append((str(row.get("meta_task_id")), "protected",
                               str(ref.get("mechanism_id", ""))))
        for task_id, kind, mechanism_id in referenced:
            family = None
            for example in list(row.get("query_examples", [])) + list(
                    row.get("support_examples", [])):
                if str(example.get("mechanism_id", "")) == mechanism_id:
                    family = str(example.get("family", ""))
            for ref in row.get("protected_references", []):
                if str(ref.get("mechanism_id", "")) == mechanism_id:
                    family = str(ref.get("family", family or ""))
            label = labels.get((family or "", mechanism_id))
            if label is None:
                issues.append(
                    f"meta_label_missing: {task_id}:{kind}:{mechanism_id}")
            elif label.get("answer") in (None, ""):
                issues.append(
                    f"meta_label_empty: {task_id}:{kind}:{mechanism_id}")
    return issues
