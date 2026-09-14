"""Live K8 task environments (O04/O05): executable bundle mechanisms.

Each environment wraps one prepared-bundle mechanism row and executes it
live: hidden state is sampled per episode, legal actions are enumerated from
the mechanism, step() mutates hidden state and returns real feedback, and
the independent family verifier decides submission success. Nothing here
reads stored answers as execution outputs.

Families: rule-inquiry (inspect variables, submit boolean), inventory (check
dependencies, submit item), program (evaluate inputs, submit value), tools
(real table filter/sum/write/check on a per-episode table copy with request
receipts).
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

from bramastra_lab.research.environments.base import (
    BaseEnvironment, EnvironmentError)


def _require_keys(mapping: Mapping[str, Any], keys: list[str],
                  what: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise EnvironmentError(f"{what} missing keys: {missing}")


class RuleInquiryEnv(BaseEnvironment):
    """Live Boolean-rule inquiry over a bundle mechanism.

    Hidden world: per-episode variable values. inspect returns the value
    (with the mechanism's declared observation noise). submit carries a
    boolean answer checked by the independent rule verifier.
    """

    name = "rule-inquiry/v1"

    def __init__(self, mechanism: Mapping[str, Any], *, budget: int = 6,
                 seed: int = 0) -> None:
        _require_keys(mechanism, ["rule", "queries", "answer"], "rule mechanism")
        self.mechanism = dict(mechanism)
        super().__init__(budget=budget, seed=seed)

    def _reset_hidden(self) -> None:
        rule = self.mechanism["rule"]
        variables = list(rule.get("variables", []))
        self._world = {var: self._rng.choice([True, False]) for var in variables}
        self._reported: dict[str, bool] = {}

    def _public_values(self) -> Mapping[str, Any]:
        public = dict(self.mechanism.get("public", {}))
        public["goal"] = "determine whether the hidden rule holds"
        return public

    def _legal_actions(self) -> list[Mapping[str, Any]]:
        actions = []
        seen = set()
        for query in self.mechanism.get("queries", []):
            if query.get("kind") == "inspect" and query.get("variable") not in seen:
                seen.add(query["variable"])
                actions.append({"kind": "inspect",
                                "variable": query["variable"]})
        actions.append({"kind": "submit"})
        return actions

    def _is_submission(self, action: Mapping[str, Any]) -> bool:
        return action.get("kind") == "submit"

    def oracle_answer(self) -> Mapping[str, Any]:
        """Named oracle control interface (never the learner input path).

        Returns the correct rule submission from hidden state. Only the
        symbolic-reference control may call this, labeled origin
        symbolic-oracle; learner arms never see it.
        """
        from bramastra_lab.research.data.k8_bundle import verify_rule

        actual = verify_rule(
            self.mechanism, {"values": dict(self._world)})
        return {"kind": "submit", "answer": bool(actual)}

    def _apply_action(self, action: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
        from bramastra_lab.research.data.k8_bundle import verify_rule

        kind = action.get("kind")
        if kind == "inspect":
            variable = action.get("variable")
            if variable not in self._world and variable not in [
                    d for d in self.mechanism.get("public", {}).get(
                        "distractors", [])]:
                raise EnvironmentError(f"unknown variable {variable!r}")
            value = self._world.get(variable, False)
            noise = self.mechanism.get("public", {}).get("obs_noise", "none")
            if noise == "flip_one" and variable not in self._reported:
                value = not value
            self._reported[variable] = value
            return {"kind": "observation", "variable": variable,
                    "value": value}, False
        if kind == "submit":
            answer = action.get("answer")
            if not isinstance(answer, bool):
                raise EnvironmentError(
                    "submit requires a boolean 'answer' field")
            actual = verify_rule(
                self.mechanism, {"values": dict(self._world)})
            # Success: submitted answer equals the verifier's ruling on the
            # live hidden world (never a stored label comparison).
            success = bool(answer) == bool(actual)
            return {"kind": "verdict", "submitted_answer": answer,
                    "correct": success}, success
        raise EnvironmentError(f"unknown action kind {kind!r}")


class InventoryEnv(BaseEnvironment):
    """Live dependency inquiry: checks reveal requirements, submit names."""

    name = "inventory/v1"

    def __init__(self, mechanism: Mapping[str, Any], *, budget: int = 6,
                 seed: int = 0) -> None:
        _require_keys(mechanism, ["items", "goal_item", "dependency_item",
                                  "queries", "answer"], "inventory mechanism")
        self.mechanism = dict(mechanism)
        super().__init__(budget=budget, seed=seed)

    def _reset_hidden(self) -> None:
        self._checks = 0

    def _public_values(self) -> Mapping[str, Any]:
        return {"goal": f"secure {self.mechanism['goal_item']}",
                "items": list(self.mechanism["items"])}

    def _legal_actions(self) -> list[Mapping[str, Any]]:
        actions = [{"kind": "check_dependency", "item": item}
                   for item in self.mechanism["items"]]
        actions.append({"kind": "submit"})
        return actions

    def _is_submission(self, action: Mapping[str, Any]) -> bool:
        return action.get("kind") == "submit"

    def oracle_answer(self) -> Mapping[str, Any]:
        """Named oracle control interface (never the learner input path)."""
        return {"kind": "submit", "item": self.mechanism["dependency_item"]}

    def _apply_action(self, action: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
        kind = action.get("kind")
        if kind == "check_dependency":
            item = action.get("item")
            if item not in self.mechanism["items"]:
                raise EnvironmentError(f"unknown item {item!r}")
            self._checks += 1
            requires = self.mechanism["dependency_item"] \
                if item == self.mechanism["goal_item"] else "none"
            return {"kind": "observation", "item": item,
                    "requires": requires}, False
        if kind == "submit":
            item = action.get("item")
            if not isinstance(item, str):
                raise EnvironmentError("submit requires a string 'item' field")
            success = item == self.mechanism["dependency_item"]
            return {"kind": "verdict", "submitted_answer": item,
                    "correct": success}, success
        raise EnvironmentError(f"unknown action kind {kind!r}")


class ProgramEnv(BaseEnvironment):
    """Live program probing: evaluate the hidden op sequence on inputs."""

    name = "program/v1"

    def __init__(self, mechanism: Mapping[str, Any], *, budget: int = 6,
                 seed: int = 0) -> None:
        _require_keys(mechanism, ["operations", "start_value", "answer"],
                      "program mechanism")
        self.mechanism = dict(mechanism)
        super().__init__(budget=budget, seed=seed)

    def _reset_hidden(self) -> None:
        self._evaluations = 0

    def evaluate(self, x: int) -> int:
        value = int(x)
        for step in self.mechanism["operations"]:
            if step["op"] == "add":
                value += int(step["operand"])
            elif step["op"] == "mul":
                value *= int(step["operand"])
            else:
                value -= int(step["operand"])
        return value

    def _public_values(self) -> Mapping[str, Any]:
        return {"goal": "determine the program result for the start value",
                "start_value": self.mechanism["start_value"],
                "n_operations": len(self.mechanism["operations"])}

    def _legal_actions(self) -> list[Mapping[str, Any]]:
        actions = [{"kind": "evaluate", "input": value}
                   for value in (self.mechanism["start_value"], 1, 2, 5, 9)]
        seen = set()
        unique = []
        for action in actions:
            key = action["input"]
            if key not in seen:
                seen.add(key)
                unique.append(action)
        unique.append({"kind": "submit"})
        return unique

    def _is_submission(self, action: Mapping[str, Any]) -> bool:
        return action.get("kind") == "submit"

    def oracle_answer(self) -> Mapping[str, Any]:
        """Named oracle control interface (never the learner input path)."""
        return {"kind": "submit", "value": int(self.mechanism["answer"])}

    def _apply_action(self, action: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
        kind = action.get("kind")
        if kind == "evaluate":
            try:
                result = self.evaluate(int(action["input"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise EnvironmentError(f"evaluate needs integer input: {exc}")
            self._evaluations += 1
            return {"kind": "observation", "input": int(action["input"]),
                    "result": result}, False
        if kind == "submit":
            try:
                submitted = int(action["value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EnvironmentError(
                    f"submit needs integer value: {exc}") from exc
            success = str(submitted) == str(self.mechanism["answer"])
            return {"kind": "verdict", "submitted_answer": submitted,
                    "correct": success}, success
        raise EnvironmentError(f"unknown action kind {kind!r}")


class ToolEnv(BaseEnvironment):
    """Live table tasks on a per-episode table copy with request receipts.

    Every tool invocation carries a request ID; replies are evidence only
    after the invocation produces them. submit maps to check_result, verified
    by the independent tool verifier against executed outputs.
    """

    name = "tools/v1"

    def __init__(self, mechanism: Mapping[str, Any], *, budget: int = 6,
                 seed: int = 0) -> None:
        _require_keys(mechanism, ["table", "predicate", "answer"],
                      "tool mechanism")
        self.mechanism = dict(mechanism)
        super().__init__(budget=budget, seed=seed)

    def _reset_hidden(self) -> None:
        self._table = copy.deepcopy(self.mechanism["table"])
        self._request_counter = 0
        self._receipts: list[dict[str, Any]] = []
        self._written: Any = None

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"req-{self.episode_id}-{self._request_counter}"

    def _public_values(self) -> Mapping[str, Any]:
        return {"goal": "filter, aggregate and report the requested result",
                "columns": list(self._table.get("columns", [])),
                "row_count": len(self._table.get("rows", [])),
                "predicate_category": self.mechanism.get("predicate", {}).get(
                    "equals")}

    def _legal_actions(self) -> list[Mapping[str, Any]]:
        return [{"kind": "read_table"},
                {"kind": "filter_rows"},
                {"kind": "sum_column"},
                {"kind": "write_result"},
                {"kind": "check_result"}]

    def _is_submission(self, action: Mapping[str, Any]) -> bool:
        return action.get("kind") == "check_result"

    def oracle_answer(self) -> Mapping[str, Any]:
        """Named oracle control interface (never the learner input path).

        Returns the full correct tool sequence for this mechanism. Only the
        symbolic-reference control may use it, labeled accordingly.
        """
        return {"kind": "sequence",
                "steps": [{"kind": "read_table"},
                          {"kind": "filter_rows"},
                          {"kind": "sum_column"},
                          {"kind": "write_result",
                           "value": self.mechanism["answer"]},
                          {"kind": "check_result"}]}

    def _apply_action(self, action: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
        from bramastra_lab.research.data.k8_bundle import verify_tool

        kind = action.get("kind")
        request_id = self._next_request_id()
        rows = list(self._table.get("rows", []))
        predicate = dict(self.mechanism.get("predicate", {}))
        if kind == "read_table":
            receipt = {"request_id": request_id, "kind": "read_table",
                       "columns": list(self._table.get("columns", [])),
                       "row_count": len(rows)}
            self._receipts.append(receipt)
            return {"kind": "table", "request_id": request_id,
                    "columns": receipt["columns"],
                    "row_count": receipt["row_count"]}, False
        if kind == "filter_rows":
            category = predicate.get("equals")
            filtered = [row for row in rows
                        if row.get(predicate.get("column", "category")) == category]
            if "value_at_least" in predicate:
                filtered = [row for row in filtered
                            if int(row.get("value", 0)) >= int(
                                predicate["value_at_least"])]
            receipt = {"request_id": request_id, "kind": "filter_rows",
                       "matched": len(filtered)}
            self._receipts.append(receipt)
            return {"kind": "filtered", "request_id": request_id,
                    "matched": len(filtered),
                    "rows": filtered}, False
        if kind == "sum_column":
            category = predicate.get("equals")
            values = [int(row.get("value", 0)) for row in rows
                      if row.get(predicate.get("column", "category")) == category]
            if "value_at_least" in predicate:
                kept = []
                for row in rows:
                    if row.get(predicate.get("column", "category")) != category:
                        continue
                    if int(row.get("value", 0)) >= int(predicate["value_at_least"]):
                        kept.append(int(row.get("value", 0)))
                values = kept
            total = sum(values)
            receipt = {"request_id": request_id, "kind": "sum_column",
                       "total": total, "n_values": len(values)}
            self._receipts.append(receipt)
            return {"kind": "aggregate", "request_id": request_id,
                    "total": total}, False
        if kind == "write_result":
            self._written = action.get("value", action.get("total"))
            receipt = {"request_id": request_id, "kind": "write_result",
                       "written": self._written}
            self._receipts.append(receipt)
            return {"kind": "written", "request_id": request_id,
                    "written": self._written}, False
        if kind == "check_result":
            if self.mechanism.get("composition") == \
                    "filter_then_aggregate_then_check":
                observed = {"answer": self.mechanism["answer"]}
                success = verify_tool(self.mechanism, observed) and \
                    self._written is not None and str(self._written) == str(
                        self.mechanism["expected_sum"])
                # Recompute live (never trust the stored label alone).
                live = self._recompute_heldout()
                success = bool(success and live)
            else:
                live_total = sum(int(row.get("value", 0)) for row in rows
                                 if row.get(predicate.get("column", "category"))
                                 == predicate.get("equals"))
                observed = {"sum": live_total}
                success = verify_tool(self.mechanism, observed) and \
                    self._written is not None and str(self._written) == str(
                        live_total)
            receipt = {"request_id": request_id, "kind": "check_result",
                       "written": self._written, "success": bool(success)}
            self._receipts.append(receipt)
            return {"kind": "verdict", "request_id": request_id,
                    "submitted_answer": self._written,
                    "correct": bool(success)}, bool(success)
        raise EnvironmentError(f"unknown action kind {kind!r}")

    def _recompute_heldout(self) -> bool:
        predicate = dict(self.mechanism.get("predicate", {}))
        rows = list(self._table.get("rows", []))
        category = predicate.get("equals")
        stage1 = [row for row in rows
                  if row.get(predicate.get("column", "category")) == category]
        stage2 = [row for row in stage1
                  if int(row.get("value", 0)) >= int(
                      predicate.get("value_at_least", 0))]
        total = sum(int(row.get("value", 0)) for row in stage2)
        check = self.mechanism.get("execution", {}).get("check", {})
        passing = total >= int(check.get("at_least", total + 1))
        expected = f"{total}:{'pass' if passing else 'fail'}"
        return expected == str(self.mechanism.get("answer", ""))


def build_live_env(mechanism: Mapping[str, Any], *, budget: int = 6,
                   seed: int = 0) -> BaseEnvironment:
    """Construct the live environment for a bundle mechanism row."""
    family = str(mechanism.get("family", ""))
    if family == "rule-inquiry":
        return RuleInquiryEnv(mechanism, budget=budget, seed=seed)
    if family == "inventory":
        return InventoryEnv(mechanism, budget=budget, seed=seed)
    if family == "program":
        return ProgramEnv(mechanism, budget=budget, seed=seed)
    if family == "tools":
        return ToolEnv(mechanism, budget=budget, seed=seed)
    raise EnvironmentError(f"no live environment for family {family!r}")


EVAL_GENERATION_STREAM = "k8-live-eval/v1"


def generate_live_mechanism(family: str, index: int, *,
                            seed: int) -> dict[str, Any]:
    """Generate a fresh live-eval mechanism from the bundle generators.

    Prepared bundles intentionally store no mechanism bodies (hidden state
    hygiene), so live evaluation generates from the same qualified
    generators under a dedicated eval seed stream. These mechanisms never
    enter training streams: training consumes only the prepared bundle
    files, and eval seeds live in a disjoint stream namespace recorded in
    every receipt (`mechanism_source: k8-live-eval/v1`).
    """
    import random as _random

    from bramastra_lab.research.data import k8_bundle as bundle

    rng = _random.Random(f"{EVAL_GENERATION_STREAM}:{family}:{seed}:{index}")
    if family == "rule-inquiry":
        mechanism = bundle.generate_rule_inquiry_mechanism(rng, index)
    elif family == "inventory":
        mechanism = bundle.generate_inventory_mechanism(rng, index)
    elif family == "program":
        mechanism = bundle.generate_program_mechanism(rng, index)
    elif family == "tools":
        mechanism = bundle.generate_tool_mechanism(
            rng, index, held_out_composition=False)
    else:
        raise EnvironmentError(f"no live generator for family {family!r}")
    mechanism["mechanism_source"] = EVAL_GENERATION_STREAM
    mechanism["eval_seed"] = seed
    return mechanism
