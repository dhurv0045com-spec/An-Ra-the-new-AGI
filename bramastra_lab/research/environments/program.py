"""Bounded typed expression interpreter; generated text is never host-executed."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ..contracts import Action, TaskSpec, content_identity
from .core import EnvironmentError, FiniteEnvironment


OPS = {"add": lambda a,b:a+b, "sub":lambda a,b:a-b, "mul":lambda a,b:a*b, "xor":lambda a,b:a^b}

@dataclass(frozen=True)
class ProgramTask:
    op: str; constant: int; target_input: int; budget: int = 3
    def __post_init__(self):
        if self.op not in OPS or type(self.constant) is not int or type(self.target_input) is not int: raise ValueError("invalid finite program")
        if type(self.budget) is not int or self.budget <= 0: raise ValueError("program budget must be a positive integer")
    @property
    def mechanism_id(self):
        # Function semantics, not AST spelling: add 0 and sub 0 deduplicate.
        return content_identity({"family":"program","domain":list(range(-4,5)),"outputs":[self.run(x) for x in range(-4,5)]})
    @property
    def semantic_id(self): return content_identity({"mechanism_id":self.mechanism_id,"target":self.target_input})
    def run(self,x:int)->int: return OPS[self.op](x,self.constant)


class ProgramEnvironment(FiniteEnvironment):
    family="program"
    def __init__(self, task:ProgramTask, split="training"):
        self._task=task
        super().__init__(TaskSpec(task.semantic_id,self.family,"program-generator/v1","program-public/v1",split,
                                  {"ast_nodes":3},{"inquiries":task.budget}))
    def _reset_state(self): self._tested=set()
    def _observable_values(self): return {"goal":"predict_program_output","target_input":self._task.target_input,"input_type":"bounded_integer"}
    def _legal_schema(self): return {"variants":[{"kind":"test","arguments":{"input":{"type":"integer","minimum":-4,"maximum":4}}},
                                                       {"kind":"predict","arguments":{"value":{"type":"integer"}}}],
                                      "forbidden_test":self._task.target_input,"repeat_test":False}
    def _apply(self,action:Action):
        args=dict(action.arguments)
        if action.action_kind=="test":
            if set(args)!={"input"} or not isinstance(args["input"],int) or isinstance(args["input"],bool) or not -4<=args["input"]<=4: raise EnvironmentError("test input must be bounded integer")
            x=args["input"]
            if x==self._task.target_input: raise EnvironmentError("direct target test is forbidden")
            if x in self._tested: raise EnvironmentError("repeated test is forbidden")
            self._tested.add(x); return 0,False,{"kind":"test_result","input":x,"output":self._task.run(x)}
        if action.action_kind=="predict":
            if set(args)!={"value"} or not isinstance(args["value"],int) or isinstance(args["value"],bool): raise EnvironmentError("prediction must be integer")
            correct=args["value"]==self._task.run(self._task.target_input)
            return int(correct),True,{"kind":"score","correct":correct}
        raise EnvironmentError("unknown program action kind")
