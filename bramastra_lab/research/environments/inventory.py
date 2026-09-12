"""Partially observable inventory transition world."""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts import Action, TaskSpec, content_identity
from .core import EnvironmentError, FiniteEnvironment


@dataclass(frozen=True)
class InventoryTask:
    operations: tuple[tuple[str, str, str | None, int], ...]  # name, writes, prerequisite, delay
    goal_item: str
    budget: int = 6
    def __post_init__(self):
        if not self.operations or not isinstance(self.goal_item, str) or not self.goal_item:
            raise ValueError("inventory operations and goal must be nonempty")
        names = set()
        for operation in self.operations:
            if len(operation) != 4:
                raise ValueError("inventory operation must have four fields")
            name, writes, requires, delay = operation
            if not isinstance(name,str) or not name or name in names or not isinstance(writes,str) or not writes:
                raise ValueError("invalid inventory operation")
            if requires is not None and not isinstance(requires,str): raise ValueError("invalid inventory prerequisite")
            if type(delay) is not int or delay < 0: raise ValueError("inventory delay must be a nonnegative integer")
            names.add(name)
        if type(self.budget) is not int or self.budget <= 0: raise ValueError("inventory budget must be a positive integer")
    @property
    def semantic_id(self):
        return content_identity({"family":"inventory", "operations": sorted(self.operations), "goal":self.goal_item})


class InventoryEnvironment(FiniteEnvironment):
    family = "inventory"
    def __init__(self, task: InventoryTask, split="training"):
        self._task = task
        super().__init__(TaskSpec(task.semantic_id, self.family, "inventory-generator/v1", "inventory-public/v1", split,
                                  {"chain": len(task.operations)}, {"inquiries": task.budget}))
    def _reset_state(self): self._slot = None; self._pending = []
    def _observable_values(self): return {"goal": {"slot_contains": self._task.goal_item}, "operations": [x[0] for x in self._task.operations]}
    def _legal_schema(self):
        return {"variants":[{"kind":"apply","arguments":{"operation":{"type":"string"}}},
                            {"kind":"inspect","arguments":{}},
                            {"kind":"finish","arguments":{}}]}
    def _tick(self):
        next_pending=[]
        for remaining,item in self._pending:
            if remaining <= 1: self._slot=item
            else: next_pending.append((remaining-1,item))
        self._pending=next_pending
    def _apply(self, action: Action):
        # Validate the complete action before time advances. Rejected actions are transactional.
        args=dict(action.arguments)
        if action.action_kind == "inspect":
            if args: raise EnvironmentError("inspect takes no arguments")
        elif action.action_kind == "apply":
            if set(args)!={"operation"} or not isinstance(args["operation"],str): raise EnvironmentError("apply requires operation string")
            if not any(x[0] == args["operation"] for x in self._task.operations): raise EnvironmentError("unknown operation")
        elif action.action_kind == "finish":
            if args: raise EnvironmentError("finish takes no arguments")
        else:
            raise EnvironmentError("unknown inventory action kind")
        self._tick()
        if action.action_kind == "inspect":
            return 0,False,{"kind":"inspection","slot":self._slot}
        if action.action_kind == "apply":
            matches=[x for x in self._task.operations if x[0]==args["operation"]]
            _,writes,requires,delay=matches[0]
            if requires is not None and self._slot != requires:
                return 0,False,{"kind":"operation_result","accepted":False}
            if delay: self._pending.append((delay,writes))
            else: self._slot=writes
            return 0,False,{"kind":"operation_result","accepted":True}
        if action.action_kind == "finish":
            achieved = self._slot == self._task.goal_item
            return int(achieved),True,{"kind":"goal_check","achieved":achieved}


def inventory_fixture() -> InventoryTask:
    return InventoryTask((("gather","ore",None,0),("smelt","bar","ore",1),("paint","red",None,0),("forge","key","bar",0)),"key",6)
