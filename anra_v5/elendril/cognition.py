"""A bounded episode loop with public-only experience and independent verification."""

from dataclasses import asdict, dataclass
import re
from types import MappingProxyType
from typing import Mapping

from .codec import digest


SYMBOL = re.compile(r"[A-P]")
OPERATIONS = frozenset({"ANSWER", "RETRIEVE", "DELIBERATE", "QUERY", "VERIFY", "REVISE", "ABSTAIN"})


@dataclass(frozen=True, slots=True)
class ExperienceEvent:
    scope: str
    subject: str
    source: str
    tick: int
    value: str | None
    parents: tuple[str, ...] = ()

    def __post_init__(self):
        if (not self.scope or not self.source or SYMBOL.fullmatch(self.subject) is None
                or type(self.tick) is not int or self.tick < 0
                or (self.value is not None and SYMBOL.fullmatch(self.value) is None)
                or len(set(self.parents)) != len(self.parents)):
            raise ValueError("invalid public evidence event")

    @property
    def identity(self):
        return digest(["public-evidence-v1", asdict(self)])


@dataclass(frozen=True, slots=True)
class Goal:
    episode: str
    family: str
    subject: str
    scope: str
    reusable_scope: str
    slots: tuple[str, ...]
    separator: str = ";"

    def __post_init__(self):
        if (not self.episode or self.family not in {"route", "revision"}
                or SYMBOL.fullmatch(self.subject) is None or not self.scope
                or not self.reusable_scope or self.scope == self.reusable_scope
                or any(SYMBOL.fullmatch(symbol) is None for symbol in self.slots)
                or len(set(self.slots)) != len(self.slots) or self.separator not in {";", "#"}):
            raise ValueError("invalid episode goal")


@dataclass(frozen=True, slots=True)
class Budget:
    decisions: int = 6
    model_positions: int = 240
    requests: int = 2

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise ValueError("invalid episode budget")


@dataclass(frozen=True, slots=True)
class Belief:
    status: str
    value: str | None
    evidence: tuple[str, ...]


def belief(events: tuple[ExperienceEvent, ...], subject: str) -> Belief:
    relevant = [event for event in events if event.subject == subject]
    if not relevant:
        return Belief("missing", None, ())
    latest = {}
    for event in relevant:
        latest[event.source] = max(latest.get(event.source, -1), event.tick)
    current = tuple(event for event in relevant if event.tick == latest[event.source])
    values = {event.value for event in current}
    ids = tuple(sorted(event.identity for event in current))
    if len(values) != 1:
        return Belief("contradicted", None, ids)
    value = next(iter(values))
    return Belief("null" if value is None else "derived", value, ids)


@dataclass(frozen=True, slots=True)
class CognitiveState:
    goal: Goal
    events: tuple[ExperienceEvent, ...] = ()
    hypothesis: str | None = None
    verification: str | None = None

    def observe(self, events: tuple[ExperienceEvent, ...]) -> "CognitiveState":
        from dataclasses import replace

        accepted = {event.identity: event for event in self.events}
        for event in events:
            if event.scope not in (self.goal.scope, self.goal.reusable_scope):
                raise ValueError("evidence outside goal custody")
            if any(parent not in accepted for parent in event.parents):
                raise ValueError("missing public evidence ancestor")
            accepted.setdefault(event.identity, event)
        return replace(self, events=tuple(accepted.values()))

    def render(self) -> str:
        edges = []
        for subject in self.goal.slots:
            value = belief(self.events, subject).value
            if value is not None:
                edges.append(f"{subject}:{value}")
        return "B:" + self.goal.separator.join(edges) + f"?{self.goal.subject}>"


@dataclass(frozen=True, slots=True)
class OperationCandidate:
    operation: str
    argument: str | None = None
    predicted_success: float | None = None
    model_positions: int = 0
    model_calls: int = 0
    raw_tokens: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PublicEnvironment:
    initial: tuple[ExperienceEvent, ...]
    channels: Mapping[str, tuple[ExperienceEvent, ...]]

    def __post_init__(self):
        object.__setattr__(self, "channels", MappingProxyType(
            {key: tuple(events) for key, events in self.channels.items()}))


@dataclass(frozen=True, slots=True)
class DecisionReceipt:
    operation: str
    argument: str | None
    origin: str
    predicted_success: float | None
    model_positions: int
    model_calls: int
    evidence: tuple[str, ...]
    raw_tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Trace:
    state: CognitiveState
    decisions: tuple[DecisionReceipt, ...]
    status: str
    answer: str | None
    requests: int
    model_identity: str | None

    @property
    def identity(self):
        return digest(["episode-trace-v1", asdict(self)])


@dataclass(frozen=True, slots=True)
class Truth:
    episode: str
    expected: str | None
    admitted_public: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    trace_sha256: str
    correct: bool
    error: str
    admitted_public: tuple[str, ...]


def verify(trace: Trace, truth: Truth) -> EpisodeOutcome:
    if truth.episode != trace.state.goal.episode:
        raise ValueError("verifier episode identity mismatch")
    admitted = set(truth.admitted_public)
    visible = {event.identity for event in trace.state.events}
    correct = ((trace.status == "answered" and truth.expected is not None and trace.answer == truth.expected)
               or (trace.status == "abstained" and truth.expected is None))
    error = "none" if correct else ("wrong_answer" if trace.status == "answered" else trace.status)
    return EpisodeOutcome(trace.identity, correct, error, tuple(sorted(admitted & visible)))


@dataclass(frozen=True, slots=True)
class MemoryVersion:
    events: tuple[ExperienceEvent, ...] = ()
    parent: str | None = None
    capacity: int = 32

    def __post_init__(self):
        if type(self.capacity) is not int or not 1 <= self.capacity <= 128 or len(self.events) > self.capacity:
            raise ValueError("memory capacity exceeded")
        if len({event.identity for event in self.events}) != len(self.events):
            raise ValueError("duplicate memory event")
        present = set()
        for event in self.events:
            if any(parent not in present for parent in event.parents):
                raise ValueError("memory ancestry is incomplete")
            present.add(event.identity)

    @property
    def identity(self):
        return digest(["episodic-memory-v1", asdict(self)])

    @classmethod
    def from_record(cls, record: dict, expected_sha256: str) -> "MemoryVersion":
        events = tuple(ExperienceEvent(**{**item, "parents": tuple(item["parents"])})
                       for item in record["events"])
        version = cls(events, record["parent"], record["capacity"])
        if version.identity != expected_sha256:
            raise ValueError("memory version identity mismatch")
        return version

    def recall(self, goal: Goal) -> tuple[ExperienceEvent, ...]:
        admitted = []
        seen = set()
        for event in self.events:
            if (event.scope == goal.reusable_scope and event.subject in goal.slots
                    and all(parent in seen for parent in event.parents)):
                admitted.append(event)
                seen.add(event.identity)
        return tuple(admitted)

    def consolidate(self, trace: Trace, outcome: EpisodeOutcome) -> "MemoryVersion":
        if outcome.trace_sha256 != trace.identity:
            raise ValueError("memory admission outcome does not bind trace")
        old = {event.identity: event for event in self.events}
        allowed = set(outcome.admitted_public)
        for event in trace.state.events:
            if (event.scope == trace.state.goal.reusable_scope and event.identity in allowed
                    and all(parent in old for parent in event.parents)
                    and len(old) < self.capacity):
                old.setdefault(event.identity, event)
        return MemoryVersion(tuple(old.values()), self.identity, self.capacity)


def run_episode(goal: Goal, env: PublicEnvironment, policy, *, budget: Budget = Budget(),
                memory: MemoryVersion | None = None, prefetch: bool = False) -> Trace:
    from dataclasses import replace
    import math

    state = CognitiveState(goal).observe(env.initial)
    decisions, used, requests, queried = [], 0, 0, set()
    status, answer = "decision_budget", None
    for _ in range(budget.decisions):
        legal = OPERATIONS - ({"RETRIEVE"} if memory is None or any(
            row.operation == "RETRIEVE" for row in decisions) else set())
        channels = tuple(key for key in env.channels if key not in queried)
        if requests >= budget.requests:
            legal -= {"QUERY", "RETRIEVE"}
        if not channels:
            legal -= {"QUERY"}
        if prefetch and not decisions and "RETRIEVE" in legal:
            choice, origin = OperationCandidate("RETRIEVE"), "fixed_baseline"
        else:
            choice = policy.propose(state, legal, channels, budget.model_positions - used)
            if choice is None:
                status = "model_budget_or_context"
                break
            origin = policy.origin
        op, arg = choice.operation, choice.argument
        if (type(choice.model_positions) is not int or choice.model_positions < 0
                or type(choice.model_calls) is not int or choice.model_calls < 0
                or (choice.predicted_success is not None and
                    (not math.isfinite(choice.predicted_success) or not 0 <= choice.predicted_success <= 1))):
            raise ValueError("invalid proposal telemetry")
        used += choice.model_positions
        valid = (op in legal and used <= budget.model_positions
                 and (op != "QUERY" or arg in channels)
                 and (op not in {"ANSWER", "DELIBERATE", "REVISE"} or
                      (arg is not None and SYMBOL.fullmatch(arg) is not None)))
        learned = ()
        if valid and op == "RETRIEVE":
            learned = memory.recall(goal)
            requests += 1
        elif valid and op == "QUERY":
            learned = env.channels[arg]
            queried.add(arg)
            requests += 1
        if learned:
            state = state.observe(learned)
        decisions.append(DecisionReceipt(op, arg, origin, choice.predicted_success,
                                         choice.model_positions, choice.model_calls,
                                         tuple(event.identity for event in learned), choice.raw_tokens))
        if not valid:
            status = "invalid_action"
            break
        if op == "ANSWER":
            status, answer = "answered", arg
            break
        if op == "ABSTAIN":
            status = "abstained"
            break
        if op in {"DELIBERATE", "REVISE"}:
            state = replace(state, hypothesis=arg)
        if op == "VERIFY":
            state = replace(state, verification=belief(state.events, goal.subject).status)
    return Trace(state, tuple(decisions), status, answer, requests, policy.model_identity)
