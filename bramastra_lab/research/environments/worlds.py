"""The three B2 fixture environments: switch world, inventory, program lab."""
from __future__ import annotations

import random
from typing import Any, Mapping

from bramastra_lab.research.environments.base import BaseEnvironment, EnvironmentError


class SwitchWorld(BaseEnvironment):
    """Two switches must both be ON at submission.

    Hidden mechanism: the second switch can only be flipped while the first
    is ON. Public information: the goal, per-switch reads and action feedback.
    """

    name = "switch-world/v1"

    def _reset_hidden(self) -> None:
        self._switches = {"A": False, "B": False}

    def _public_values(self) -> Mapping[str, Any]:
        return {
            "goal": "turn both switches ON, then submit",
            "switch_names": ["A", "B"],
        }

    def _legal_actions(self) -> list[Mapping[str, Any]]:
        actions = [{"kind": "read", "switch": name} for name in ("A", "B")]
        actions += [{"kind": "flip", "switch": name} for name in ("A", "B")]
        actions.append({"kind": "submit"})
        return actions

    def _is_submission(self, action: Mapping[str, Any]) -> bool:
        return action.get("kind") == "submit"

    def _apply_action(self, action: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
        kind = action["kind"]
        if kind == "read":
            name = action["switch"]
            return {"kind": "read", "switch": name, "state": int(self._switches[name])}, False
        if kind == "flip":
            name = action["switch"]
            if name == "B" and not self._switches["A"]:
                return {"kind": "flip", "switch": name, "changed": False,
                        "note": "no effect observed"}, False
            self._switches[name] = not self._switches[name]
            return {"kind": "flip", "switch": name, "changed": True}, False
        if kind == "submit":
            success = self._switches["A"] and self._switches["B"]
            return {"kind": "submit", "submitted_answer": "both-on" if success else "not-both-on",
                    "verdict": "success" if success else "failure"}, success
        raise EnvironmentError(f"unknown action kind {kind!r}")


class InventoryWorld(BaseEnvironment):
    """One of three containers holds the item; find it, then submit its name.

    Reporting 'the item was not found' can never win: only a submission
    naming the holding container succeeds.
    """

    name = "inventory-world/v1"

    def __init__(self, *, budget: int = 6, seed: int = 0) -> None:
        self._containers = ["red", "green", "blue"]
        super().__init__(budget=budget, seed=seed)

    def _reset_hidden(self) -> None:
        self._holding = self._rng.choice(self._containers)
        self._opened: dict[str, bool] = {name: False for name in self._containers}

    def _public_values(self) -> Mapping[str, Any]:
        return {"goal": "name the container that holds the item",
                "containers": list(self._containers)}

    def _legal_actions(self) -> list[Mapping[str, Any]]:
        actions = [{"kind": "peek", "container": name} for name in self._containers]
        actions += [{"kind": "submit", "container": name} for name in self._containers]
        return actions

    def _is_submission(self, action: Mapping[str, Any]) -> bool:
        return action.get("kind") == "submit"

    def _apply_action(self, action: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
        if action["kind"] == "peek":
            name = action["container"]
            self._opened[name] = True
            holds = name == self._holding
            return {"kind": "peek", "container": name,
                    "contains_item": holds}, False
        name = action["container"]
        success = name == self._holding
        return {"kind": "submit", "submitted_answer": name,
                "verdict": "success" if success else "failure"}, success


class ProgramLab(BaseEnvironment):
    """Hidden affine function f(x) = (a*x + b) mod m on a small domain.

    Query evaluation inputs, then submit f(target). The target input is
    public; its answer is not.
    """

    name = "program-lab/v1"

    def __init__(self, *, budget: int = 6, seed: int = 0, modulus: int = 7) -> None:
        self._modulus = modulus
        self._domain = list(range(modulus))
        super().__init__(budget=budget, seed=seed)

    def _reset_hidden(self) -> None:
        self._a = self._rng.randrange(1, self._modulus)
        self._b = self._rng.randrange(0, self._modulus)
        self._target = self._rng.randrange(0, self._modulus)
        self._query_count = 0

    def evaluate(self, x: int) -> int:
        """Hidden mechanism; oracle/diagnostic use, never learner input."""
        return (self._a * x + self._b) % self._modulus

    def _public_values(self) -> Mapping[str, Any]:
        return {"goal": "predict f(target) and submit the value",
                "target_input": self._target, "domain": list(self._domain),
                "modulus": self._modulus,
                "query_count": self._query_count}

    def _legal_actions(self) -> list[Mapping[str, Any]]:
        actions = [{"kind": "evaluate", "input": value} for value in self._domain
                   if value != self._target]
        actions.append({"kind": "submit"})
        return actions

    def _is_submission(self, action: Mapping[str, Any]) -> bool:
        return action.get("kind") == "submit"

    def _apply_action(self, action: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
        if action["kind"] == "evaluate":
            value = action["input"]
            self._query_count += 1
            return {"kind": "evaluate", "input": value, "output": self.evaluate(value)}, False
        submitted = action.get("value")
        success = isinstance(submitted, int) and not isinstance(submitted, bool) \
            and submitted == self.evaluate(self._target)
        return {"kind": "submit", "submitted_answer": submitted,
                "verdict": "success" if success else "failure"}, success
