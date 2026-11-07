"""Public-evidence action supervision and complete, independently scored episodes.

This deliberately small *development* instrument trains the existing decoder's
action vocabulary.  It is not evidence for language, metacognition or transfer.
The training labeler sees public state and channel availability, never Truth.
"""

from dataclasses import asdict, dataclass
from math import isfinite
import random
from typing import Callable

from .codec import BOS, CODEC_SHA, EOS, digest, encode
from .cognition import (
    Budget, CognitiveState, ExperienceEvent, Goal, OperationCandidate,
    PublicEnvironment, Truth, belief, run_episode, verify,
)


VERSION = "public-action-worlds-v1"
ACTION_GRAMMAR = "legacy-action-eos-v1"
SYMBOLS = "ABCDEFGHIJKLMNOP"
CHANNEL = "request-current-status"
KINDS = ("known", "missing", "conflict", "null", "unavailable")
DEFAULT_BUDGET = Budget(decisions=3, model_positions=384, requests=1)
QUERY_COST = 0.1


@dataclass(frozen=True, slots=True)
class ActionCase:
    world_id: str
    kind: str
    goal: Goal
    environment: PublicEnvironment
    truth: Truth


def render_action(state: CognitiveState, channels: tuple[str, ...]) -> str:
    """Serialize only visible evidence and one advertised channel's availability.

    The channel *payload*, split/world IDs, case kind and evaluator truth are
    intentionally absent.  The legacy codec's identity is unchanged.
    """
    if channels not in ((), (CHANNEL,)):
        raise ValueError("action fixture supports only its declared status channel")
    return f"C:{int(bool(channels))};" + state.render()


def public_action(state: CognitiveState, channels: tuple[str, ...]) -> tuple[str, str | None]:
    """Explicit engineered supervision, not a claimed learned policy."""
    current = belief(state.events, state.goal.subject)
    if current.status == "derived":
        return "ANSWER", current.value
    if current.status == "null":
        return "ABSTAIN", None
    if channels:
        return "QUERY", channels[0]
    return "ABSTAIN", None


def action_tokens(operation: str, argument: str | None) -> tuple[int, ...]:
    if operation == "QUERY" and argument == CHANNEL:
        content = "Q"
    elif operation == "ABSTAIN" and argument is None:
        content = "!"
    elif (operation == "ANSWER" and isinstance(argument, str)
          and len(argument) == 1 and argument in SYMBOLS):
        content = argument
    else:
        raise ValueError("unsupported supervised action")
    return (*encode(content), EOS)


@dataclass(frozen=True, slots=True)
class ActionExample:
    task_id: str
    world_id: str
    kind: str
    stage: str
    prompt: str
    target_tokens: tuple[int, ...]
    action: str

    @property
    def tokens(self) -> tuple[int, ...]:
        return (BOS, *encode(self.prompt), *self.target_tokens)

    @property
    def target_span(self) -> tuple[int, int]:
        start = len(self.tokens) - len(self.target_tokens)
        return start, len(self.tokens)

    @property
    def eligible(self) -> tuple[bool, ...]:
        start, end = self.target_span
        return tuple(start <= index < end for index in range(len(self.tokens)))


def supervision(case: ActionCase) -> tuple[ActionExample, ...]:
    """Teacher labels are derived from *observed* state, including after QUERY.

    Truth is not consulted. The second example is an answer/abstention only
    after the charged public channel has supplied its observation.
    """
    state = CognitiveState(case.goal).observe(case.environment.initial)
    channels = tuple(case.environment.channels)
    examples = []
    for stage in ("initial", "after-query"):
        operation, argument = public_action(state, channels)
        examples.append(ActionExample(
            case.world_id + "/" + stage, case.world_id, case.kind, stage,
            render_action(state, channels), action_tokens(operation, argument), operation,
        ))
        if operation != "QUERY":
            break
        state = state.observe(case.environment.channels[argument])
        channels = ()
    return tuple(examples)


def _prompt_split(prompt: str) -> str:
    bucket = int(digest([VERSION, "prompt-partition", prompt]), 16) % 10
    return "train" if bucket < 7 else "validation" if bucket < 9 else "test"


def _candidate(index: int, seed: int) -> ActionCase:
    # A private RNG cannot perturb training, model initialization or evaluator RNG.
    rng = random.Random(int(digest([VERSION, seed, index]), 16))
    kind = KINDS[index % len(KINDS)]
    subject, *others = rng.sample(SYMBOLS, 4)
    value, stale = rng.sample(SYMBOLS, 2)
    world = digest([VERSION, seed, index])
    goal = Goal(world, "revision", subject, world, "publication-" + world,
                tuple(rng.sample([subject, *others], 4)))
    events = [ExperienceEvent(world, other, "directory", 0, rng.choice(SYMBOLS))
              for other in others]
    channels = {}
    if kind == "known":
        events += [ExperienceEvent(world, subject, "operator", 0, stale),
                   ExperienceEvent(world, subject, "operator", 2, value)]
        answer = value
    elif kind == "null":
        events.append(ExperienceEvent(world, subject, "operator", 2, None))
        answer = None
    elif kind == "conflict":
        events += [ExperienceEvent(world, subject, "operator", 1, stale),
                   ExperienceEvent(world, subject, "sensor", 1, value)]
        answer = rng.choice((None, value, value, value))
    elif kind == "missing":
        answer = rng.choice((None, value, value, value))
    else:
        answer = None  # no answer is publicly obtainable; abstention is the target
    if kind != "unavailable":
        channels[CHANNEL] = (
            ExperienceEvent(world, subject, "operator", 3, answer),
            ExperienceEvent(world, subject, "sensor", 3, answer),
        )
    rng.shuffle(events)  # recency in the serialized list is not evidence authority
    environment = PublicEnvironment(tuple(events), channels)
    admitted = tuple(event.identity for event in (
        *events, *(event for payload in channels.values() for event in payload)))
    return ActionCase(world, kind, goal, environment, Truth(world, answer, admitted))


@dataclass(frozen=True, slots=True)
class ActionDataset:
    split: str
    seed: int
    cases: tuple[ActionCase, ...]
    examples: tuple[ActionExample, ...]

    @property
    def manifest(self) -> dict:
        # MappingProxyType in PublicEnvironment is intentionally handled explicitly.
        worlds = [{"world_id": case.world_id, "kind": case.kind,
                   "goal": asdict(case.goal),
                   "initial": [asdict(event) for event in case.environment.initial],
                   "channels": {key: [asdict(event) for event in events]
                                for key, events in case.environment.channels.items()},
                   "truth": asdict(case.truth)} for case in self.cases]
        return {"schema": VERSION, "split": self.split, "seed": self.seed,
                "tokenizer_sha256": CODEC_SHA, "action_grammar": ACTION_GRAMMAR,
                "source": "new deterministic synthetic worlds; no inherited corpus",
                "license": "repository LICENSE; no external training text",
                "worlds_sha256": digest(worlds),
                "examples_sha256": digest([asdict(row) for row in self.examples]),
                "world_ids": [case.world_id for case in self.cases],
                "episodes": len(self.cases), "examples": len(self.examples),
                "kinds": {kind: sum(case.kind == kind for case in self.cases) for kind in KINDS},
                "max_sequence_tokens": max(len(row.tokens) for row in self.examples)}

    @property
    def identity(self) -> str:
        return digest(self.manifest)


def action_fixture(split: str = "train", *, worlds: int = 40, seed: int = 1729) -> ActionDataset:
    """Balanced, fresh worlds with *all* decision prompts partitioned by hash.

    Even post-query prompts cannot occur in another split. Selection uses input
    hashes before any model outcomes, and removes duplicate prompts within a
    fixture. Training and evaluation remain development data, never a sealed test.
    """
    if (split not in {"train", "validation", "test"} or type(worlds) is not int
            or not 5 <= worlds <= 1000 or worlds % len(KINDS)
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("expected supported split, seed and 5..1000 worlds divisible by five")
    cases, examples, seen = [], [], set()
    quota = worlds // len(KINDS)
    counts = {kind: 0 for kind in KINDS}
    for index in range(worlds * 10000):
        case = _candidate(index, seed)
        if counts[case.kind] == quota:
            continue
        rows = supervision(case)
        prompts = {row.prompt for row in rows}
        if (len(prompts) != len(rows) or prompts & seen
                or any(_prompt_split(prompt) != split for prompt in prompts)):
            continue
        cases.append(case)
        examples.extend(rows)
        counts[case.kind] += 1
        seen.update(prompts)
        if len(cases) == worlds:
            result = ActionDataset(split, seed, tuple(cases), tuple(examples))
            assert_disjoint(result)
            return result
    raise RuntimeError("could not form the preregistered split; no partial dataset admitted")


def assert_disjoint(*datasets: ActionDataset) -> None:
    """Reject world, observation-prompt or task reuse across declared datasets."""
    worlds, prompts, tasks = set(), set(), set()
    for data in datasets:
        current_worlds = {case.world_id for case in data.cases}
        current_prompts = {row.prompt for row in data.examples}
        current_tasks = {row.task_id for row in data.examples}
        if (len(current_worlds) != len(data.cases) or len(current_tasks) != len(data.examples)
                or len(current_prompts) != len(data.examples)
                or current_worlds & worlds or current_prompts & prompts or current_tasks & tasks
                or any(_prompt_split(prompt) != data.split for prompt in current_prompts)):
            raise ValueError("action dataset contamination or duplicate decision prompt")
        worlds |= current_worlds
        prompts |= current_prompts
        tasks |= current_tasks


@dataclass(frozen=True, slots=True)
class DecodedAction:
    tokens: tuple[int, ...]
    model_positions: int
    model_calls: int
    confidence: float | None = None


class NeuralActionPolicy:
    """A strict raw-token adapter, shared by training evaluation and cognition.

    ``decode(prompt, remaining_positions)`` must actually invoke the model and
    return both content and EOS. No legality mask, oracle choice or fallback is
    applied. An unavailable model query remains an invalid action in the loop.
    """
    origin = "model"

    def __init__(self, decode: Callable[[str, int], DecodedAction | None], model_identity: str):
        if not model_identity:
            raise ValueError("model identity is required")
        self.decode = decode
        self.model_identity = model_identity

    def propose(self, state, legal, channels, positions):
        decoded = self.decode(render_action(state, channels), positions)
        if decoded is None:
            return None
        operation, argument = "INVALID", None
        if decoded.tokens == action_tokens("QUERY", CHANNEL):
            operation, argument = "QUERY", CHANNEL
        elif decoded.tokens == action_tokens("ABSTAIN", None):
            operation = "ABSTAIN"
        else:
            for symbol in SYMBOLS:
                if decoded.tokens == action_tokens("ANSWER", symbol):
                    operation, argument = "ANSWER", symbol
                    break
        return OperationCandidate(operation, argument, decoded.confidence,
                                  decoded.model_positions, decoded.model_calls, decoded.tokens)


class ControlPolicy:
    """Explicit engineered controls. Oracle is separately truth-privileged."""
    model_identity = None

    def __init__(self, mode: str = "fixed", *, seed: int = 0, oracle: Truth | None = None):
        if mode not in {"fixed", "oracle", "no-query", "always-query", "random", "query-blind", "recency"}:
            raise ValueError("unknown control")
        if (mode == "oracle") != (oracle is not None):
            raise ValueError("only the explicitly privileged oracle may receive truth")
        self.mode, self.seed, self.oracle = mode, seed, oracle
        self.origin = "privileged_oracle" if mode == "oracle" else "fixed_baseline"

    def propose(self, state, legal, channels, positions):
        if self.mode == "oracle":
            if state.goal.episode != self.oracle.episode:
                raise ValueError("oracle episode mismatch")
            return OperationCandidate("ANSWER", self.oracle.expected) if self.oracle.expected else OperationCandidate("ABSTAIN")
        if self.mode == "always-query" and channels and "QUERY" in legal:
            return OperationCandidate("QUERY", channels[0])
        if self.mode == "random":
            rng = random.Random(int(digest([self.seed, state.render(), channels]), 16))
            options = [OperationCandidate("ANSWER", symbol) for symbol in SYMBOLS]
            options.append(OperationCandidate("ABSTAIN"))
            if "QUERY" in legal and channels:
                options.append(OperationCandidate("QUERY", channels[0]))
            return rng.choice(options)
        if self.mode == "query-blind":
            return OperationCandidate("ANSWER", "A")
        if self.mode == "recency":
            events = [event for event in state.events if event.subject == state.goal.subject]
            value = events[-1].value if events else None
            return OperationCandidate("ANSWER", value) if value else OperationCandidate("ABSTAIN")
        operation, argument = public_action(state, () if self.mode == "no-query" else channels)
        return OperationCandidate(operation, argument)


def evaluate_policy(data: ActionDataset, policy, *, budget: Budget = DEFAULT_BUDGET,
                    query_cost: float = QUERY_COST) -> dict:
    """Score complete terminal actions independently, preserving every attempt."""
    if type(query_cost) not in (float, int) or not isfinite(query_cost) or not 0 < query_cost < 1:
        raise ValueError("query cost must be finite and strictly between zero and one")
    rows = []
    for case in data.cases:
        actor = ControlPolicy("oracle", oracle=case.truth) if policy == "oracle" else policy
        trace = run_episode(case.goal, case.environment, actor, budget=budget)
        outcome = verify(trace, case.truth)
        terminal = trace.decisions[-1] if trace.decisions else None
        confidence = terminal.predicted_success if terminal else None
        rows.append({"world_id": case.world_id, "kind": case.kind,
                     "trace": asdict(trace), "trace_sha256": trace.identity,
                     "outcome": asdict(outcome), "requests": trace.requests,
                     "utility": float(outcome.correct) - query_cost * trace.requests,
                     "terminal_confidence": confidence,
                     "uncalibrated_brier": None if confidence is None else (confidence - int(outcome.correct))**2})
    totals = {}
    for kind in ("all", *KINDS):
        subset = [row for row in rows if kind == "all" or row["kind"] == kind]
        totals[kind] = {"episodes": len(subset),
                        "correct": sum(row["outcome"]["correct"] for row in subset),
                        "requests": sum(row["requests"] for row in subset),
                        "utility": sum(row["utility"] for row in subset),
                        "invalid": sum(row["trace"]["status"] == "invalid_action" for row in subset),
                        "model_calls": sum(decision["model_calls"] for row in subset for decision in row["trace"]["decisions"]),
                        "model_positions": sum(decision["model_positions"] for row in subset for decision in row["trace"]["decisions"])}
    return {"schema": "public-action-evaluation-v1", "dataset_sha256": data.identity,
            "query_cost": query_cost, "budget": asdict(budget), "totals": totals,
            "episodes": rows, "qualification": "development synthetic; not a capability promotion"}


ALGORITHM_HYPOTHESES = (
    {"arm": "action-weighted", "control": "ce", "mechanism": "Increase initial action-token loss relative to EOS; V4 never emitted QUERY.",
     "falsifier": "No held-out complete-action utility improvement at matched updates/tokens, or worsened known-answer accuracy.",
     "budget": "same model, split, seeds, sampled rows, updates and supervised token exposures; report weight denominator"},
    {"arm": "uncertainty-first", "control": "fixed-mixture", "mechanism": "Teach missing/conflicted QUERY before later balanced state mixtures.",
     "falsifier": "No held-out utility improvement or persistent redundant queries on directly answerable worlds.",
     "budget": "same row multiset per epoch and same updates/tokens; only order differs; no test-driven curriculum"},
)


def curriculum_indices(data: ActionDataset, *, seed: int, arm: str, epoch: int = 0) -> tuple[int, ...]:
    """Comparable full-epoch multisets; the initial epoch teaches QUERY first.

    End a matched comparison on full epochs, or separately report unequal row
    exposures if an interruption leaves either arm at a partial epoch.
    """
    if arm not in {"fixed-mixture", "uncertainty-first"} or type(epoch) is not int or epoch < 0:
        raise ValueError("unknown curriculum or invalid epoch")
    indices = list(range(len(data.examples)))
    random.Random(int(digest([VERSION, seed, epoch]), 16)).shuffle(indices)
    if arm == "uncertainty-first" and epoch == 0:
        indices.sort(key=lambda index: data.examples[index].action != "QUERY")
    return tuple(indices)
