"""Finite hidden acyclic Boolean causal mechanisms."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ..contracts import Action, TaskSpec, content_identity
from .core import EnvironmentError, FiniteEnvironment


@dataclass(frozen=True)
class SwitchTask:
    inputs: tuple[str, ...]
    gates: tuple[tuple[str, str, tuple[str, ...]], ...]
    target_inputs: tuple[int, ...]
    budget: int = 3
    output_node: str | None = None

    def __post_init__(self):
        if not self.inputs or len(set(self.inputs)) != len(self.inputs):
            raise ValueError("switch inputs must be unique and nonempty")
        if len(self.target_inputs) != len(self.inputs) or any(type(x) is not int or x not in (0, 1) for x in self.target_inputs):
            raise ValueError("switch target must be one strict bit per input")
        if type(self.budget) is not int or self.budget <= 0:
            raise ValueError("switch budget must be a positive integer")
        names = set(self.inputs)
        for name, op, parents in self.gates:
            if name in names or not name or op not in {"and", "or", "xor", "not"}:
                raise ValueError("invalid or duplicate switch gate")
            if not parents or (op == "not" and len(parents) != 1) or (op != "not" and len(parents) < 2):
                raise ValueError("invalid switch gate arity")
            names.add(name)
        output = self.output_node or (self.gates[-1][0] if self.gates else None)
        if output not in names:
            raise ValueError("switch output node is unknown")
        object.__setattr__(self, "output_node", output)
        # Force cycle/missing-parent validation now rather than during scoring.
        self._topological_gates()

    def _topological_gates(self):
        known, pending, ordered = set(self.inputs), list(self.gates), []
        while pending:
            ready = [g for g in pending if all(parent in known for parent in g[2])]
            if not ready:
                raise ValueError("switch graph must be acyclic and all parents declared")
            for gate in ready:
                ordered.append(gate); known.add(gate[0]); pending.remove(gate)
        return tuple(ordered)

    def canonical(self):
        """Exact Boolean function in stable positional input order."""
        return {"arity": len(self.inputs), "truth_table": [self.evaluate(bits) for bits in product((0, 1), repeat=len(self.inputs))]}

    @property
    def mechanism_id(self): return content_identity({"family": "switch", "function": self.canonical()})

    @property
    def semantic_id(self): return content_identity({"mechanism_id": self.mechanism_id, "target": list(self.target_inputs)})

    def evaluate(self, values: tuple[int, ...]) -> int:
        state = dict(zip(self.inputs, values))
        for name, op, parents in self._topological_gates():
            xs = [state[p] for p in parents]
            state[name] = {"and": all, "or": any, "xor": lambda y: sum(y) % 2, "not": lambda y: not y[0]}[op](xs)
            state[name] = int(state[name])
        return state[self.output_node]


class SwitchEnvironment(FiniteEnvironment):
    family = "switch"
    def __init__(self, task: SwitchTask, split="training", surface_order=None):
        self._task = task
        self._surface_order = tuple(surface_order or task.inputs)
        if len(self._surface_order) != len(task.inputs) or set(self._surface_order) != set(task.inputs):
            raise ValueError("surface order must be a permutation of task inputs")
        spec = TaskSpec(task.semantic_id, self.family, "switch-generator/v1", "switch-public/v1", split,
                        {"inputs": len(task.inputs), "gates": len(task.gates)}, {"inquiries": task.budget})
        super().__init__(spec)

    def _reset_state(self): self._queried = set()
    def _observable_values(self):
        return {"goal": "predict_target_output", "input_names": list(self._surface_order),
                "target_input": list(self._surface_vector(self._task.target_inputs))}
    def _legal_schema(self):
        return {"variants": [
            {"kind": "query", "arguments": {"input": {"type": "bits", "length": len(self._task.inputs)}}},
            {"kind": "predict", "arguments": {"value": {"type": "integer", "enum": [0, 1]}}},
        ], "forbidden_query": list(self._surface_vector(self._task.target_inputs)), "repeat_query": False}

    def _surface_vector(self, positional):
        by_name = dict(zip(self._task.inputs, positional))
        return tuple(by_name[name] for name in self._surface_order)

    def _apply(self, action: Action):
        args = dict(action.arguments)
        if action.action_kind == "query":
            if set(args) != {"input"} or not isinstance(args["input"], (list, tuple)):
                raise EnvironmentError("query requires one input bit vector")
            surface = tuple(args["input"])
            if len(surface) != len(self._task.inputs) or any(x not in (0, 1) or isinstance(x, bool) for x in surface):
                raise EnvironmentError("query input must be a fixed-length bit vector")
            values = tuple(surface[self._surface_order.index(name)] for name in self._task.inputs)
            if values == self._task.target_inputs: raise EnvironmentError("direct target query is forbidden")
            if values in self._queried: raise EnvironmentError("repeated query is forbidden")
            self._queried.add(values)
            return 0.0, False, {"kind": "query_result", "input": list(surface), "output": self._task.evaluate(values),
                                "noise": self._rng.randrange(2)}
        if action.action_kind == "predict":
            if set(args) != {"value"} or args["value"] not in (0, 1) or isinstance(args["value"], bool):
                raise EnvironmentError("prediction must be integer 0 or 1")
            correct = args["value"] == self._task.evaluate(self._task.target_inputs)
            return float(correct), True, {"kind": "score", "correct": correct}
        raise EnvironmentError("unknown switch action kind")


def parity_synergy_task() -> SwitchTask:
    # f(1,1) requires both permitted basis queries; either alone has zero target IG.
    return SwitchTask(("a", "b"), (("target", "xor", ("a", "b")),), (1, 1), 3)


def parity_synergy_information(history=()):
    """Exact target information gains for f(x)=a*x0 XOR b*x1, a,b fair."""
    import math
    hypotheses = [(a, b) for a in (0, 1) for b in (0, 1)]
    for query, observed in history:
        hypotheses = [h for h in hypotheses if ((h[0] * query[0]) ^ (h[1] * query[1])) == observed]
    def entropy(values):
        if not values: return 0.0
        p = sum(values) / len(values)
        return 0.0 if p in (0, 1) else -p*math.log2(p)-(1-p)*math.log2(1-p)
    base = entropy([a ^ b for a,b in hypotheses])
    gains = {}
    for q in ((1,0),(0,1)):
        groups = [[h for h in hypotheses if ((h[0]*q[0])^(h[1]*q[1])) == o] for o in (0,1)]
        expected = sum(len(g)/len(hypotheses)*entropy([a^b for a,b in g]) for g in groups if g)
        gains[q] = base-expected
    return gains


def exhaustive_truth(task: SwitchTask):
    return {bits: task.evaluate(bits) for bits in product((0, 1), repeat=len(task.inputs))}
