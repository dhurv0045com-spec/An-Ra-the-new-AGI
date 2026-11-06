"""Run the grounded episode contract with an existing frozen ELENDRIL checkpoint."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from .codec import BOS, EOS, TOKEN_IDS, canonical, digest, encode
from .cognition import (Budget, ExperienceEvent, Goal, MemoryVersion, OperationCandidate,
                        PublicEnvironment, Truth, belief, run_episode, verify)
from .join import join_fixture, parse, solve


CHECKPOINT_SOURCE = "6615886b1a37fd535682bdee38157e4c44cd0783"
CHECKPOINT_LINEAGE = "two-read-join-v1-binding-7"
ACTION_SYMBOLS = {"Q": "QUERY", "R": "RETRIEVE", "S": "VERIFY"}


@dataclass(frozen=True, slots=True)
class Case:
    goal: Goal
    environment: PublicEnvironment
    truth: Truth
    display: str


def _truth(goal, env, answer, extra=()):
    public = (*env.initial, *(event for channel in env.channels.values() for event in channel), *extra)
    return Truth(goal.episode, answer, tuple(event.identity for event in public))


def route_cases() -> tuple[tuple[Case, Case], ...]:
    """Distinct historical graph worlds share one explicitly reusable public fact."""
    rows = join_fixture("development")
    graphs = [(row, *parse(row.visible.prompt)) for row in rows]
    chosen = []
    used = set()
    for target, edges, root in graphs:
        if target.visible.cluster_id in used or len({a for a, _ in edges}) != len(edges):
            continue
        middle = solve(edges, root, 1)
        shared = (middle, target.truth.gold)
        for prior, prior_edges, prior_root in graphs:
            if (prior.visible.cluster_id in (used | {target.visible.cluster_id})
                    or shared not in prior_edges or prior.truth.gold == target.truth.gold
                    or len({a for a, _ in prior_edges}) != len(prior_edges)):
                continue
            scope = "published-route-fact-" + str(len(chosen))
            reusable = ExperienceEvent(scope, shared[0], "route-directory", 0, shared[1])

            def build(row, graph, query, *, omit):
                episode = row.visible.task_id
                goal = Goal(episode, "route", query, episode, scope,
                            tuple(a for a, _ in graph), "#" if "#" in row.visible.prompt else ";")
                events = tuple(reusable if edge == shared else ExperienceEvent(
                    episode, edge[0], "local-route-map", index, edge[1])
                    for index, edge in enumerate(graph) if not (omit and edge == shared))
                env = PublicEnvironment(events, {})
                return Case(goal, env, _truth(goal, env, row.truth.gold, (reusable,) if omit else ()),
                            "Find the terminal of a two-link route from " + query + ".")

            chosen.append((build(prior, prior_edges, prior_root, omit=False),
                           build(target, edges, root, omit=True)))
            used.update((target.visible.cluster_id, prior.visible.cluster_id))
            break
        if len(chosen) == 4:
            return tuple(chosen)
    raise ValueError("insufficient distinct route pairs")


def revision_cases() -> tuple[Case, ...]:
    """Independent changing-state cases with missing, conflicted, and null values."""
    cases = []
    for index, answer in enumerate(("D", "F", None, "K")):
        episode = f"revision-world-{index}"
        goal = Goal(episode, "revision", "A", episode, "state-publication-" + str(index), ("A", "B"))
        initial = (ExperienceEvent(episode, "B", "facility-directory", 0, "E"),)
        if index != 1:
            initial += (ExperienceEvent(episode, "A", "operator", 1, "C"),
                        ExperienceEvent(episode, "A", "sensor", 1, "H"))
        update = (ExperienceEvent(episode, "A", "operator", 2, answer),
                  ExperienceEvent(episode, "A", "sensor", 2, answer))
        env = PublicEnvironment(initial, {"request-current-status": update})
        cases.append(Case(goal, env, _truth(goal, env, answer),
                          "Check the current state of facility A. The operator and sensor may disagree."))
    return tuple(cases)


class FixedPolicy:
    """Public-evidence control with explicit query and two-edge logic."""

    origin = "fixed_baseline"
    model_identity = None

    def propose(self, state, legal, channels, positions):
        if state.goal.family == "revision" and "QUERY" in legal:
            return OperationCandidate("QUERY", channels[0])
        value = belief(state.events, state.goal.subject).value
        if value is not None and state.goal.family == "route":
            value = belief(state.events, value).value
        return OperationCandidate("ANSWER", value) if value is not None else OperationCandidate("ABSTAIN")


class NeuralCore:
    """The original two-read model, loaded from a hash-verified frozen checkpoint."""

    origin = "model"

    def __init__(self, root: Path, checkpoint_sha256: str):
        import re
        import torch
        from v5_model.core import initialize
        from v5_training.checkpoint import CheckpointStore
        from .associative import ADDRESS_WIDTH, WINDOW, AssociativeDecoder
        from .training import MODEL, _load, configure_cpu

        if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None:
            raise ValueError("expected a full checkpoint SHA-256")
        if not (root / CHECKPOINT_LINEAGE / "objects" / checkpoint_sha256 / "manifest.json").is_file():
            raise ValueError("frozen checkpoint was not found")
        configure_cpu()
        store = CheckpointStore(root, CHECKPOINT_LINEAGE)
        state, payloads = store.restore(checkpoint_sha256)
        expected = digest([MODEL.sha256(), "associative-copy-v2", "two-read", WINDOW, ADDRESS_WIDTH])
        if (state.identities.source_commit != CHECKPOINT_SOURCE
                or state.identities.model_spec_sha256 != expected
                or not state.complete or state.global_update != 384):
            raise ValueError("checkpoint identity does not match ELD-DEV-004")
        self.model = AssociativeDecoder(initialize(MODEL, 7), 7, mode="two-read")
        self.model.load_state_dict(_load(payloads["model.bin"]), strict=True)
        self.model.eval().requires_grad_(False)
        self.model_identity = checkpoint_sha256
        self._torch = torch

    def weights_sha256(self) -> str:
        from hashlib import sha256

        fingerprint = sha256()
        for name, parameter in sorted(self.model.state_dict().items()):
            fingerprint.update(name.encode() + b"\0")
            fingerprint.update(parameter.detach().contiguous().reshape(-1).view(self._torch.uint8).numpy().tobytes())
        return fingerprint.hexdigest()

    def propose(self, state, legal, channels, positions):
        from .training import MODEL, logits_for

        torch = self._torch
        ids = (BOS, *encode(state.render()))
        if len(ids) + 1 > MODEL.context_length or 2 * len(ids) + 1 > positions:
            return None

        def scores(ids):
            tokens = torch.tensor([ids], device="cpu")
            logits = logits_for(self.model, tokens, torch.zeros_like(tokens))[0, -1]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite model proposal")
            return logits.softmax(-1)

        with torch.inference_mode():
            distribution = scores(ids)
            first = int(distribution.argmax())
            confidence = float(distribution[first])
            symbol = next((token for token, identity in TOKEN_IDS.items() if identity == first), None)
            op = ACTION_SYMBOLS.get(symbol)
            if op:
                return OperationCandidate(op, channels[0] if op == "QUERY" and channels else None,
                                          confidence, len(ids), 1, (first,))
            tail = scores((*ids, first))
            second = int(tail.argmax())
            confidence *= float(tail[second])
        operation = "INVALID"
        argument = None
        if second == EOS and symbol == "!":
            operation = "ABSTAIN"
        elif second == EOS and symbol is not None and "A" <= symbol <= "P":
            operation, argument = "ANSWER", symbol
        return OperationCandidate(operation, argument, confidence, 2 * len(ids) + 1, 2, (first, second))


def run_demo(policy) -> dict:
    rows = []
    before = policy.weights_sha256() if isinstance(policy, NeuralCore) else None

    def execute(case, arm, memory=None, prefetch=False):
        trace = run_episode(case.goal, case.environment, policy, memory=memory, prefetch=prefetch,
                            budget=Budget())
        outcome = verify(trace, case.truth)
        rows.append({"arm": arm, "display": case.display, "trace": asdict(trace),
                     "trace_sha256": trace.identity, "outcome": asdict(outcome),
                     "memory_sha256": memory.identity if memory is not None else None})
        return trace, outcome

    for prior, target in route_cases():
        trace, outcome = execute(prior, "admission")
        memory = MemoryVersion().consolidate(trace, outcome)
        memory = MemoryVersion.from_record(json.loads(canonical(asdict(memory))), memory.identity)
        execute(target, "no-memory")
        execute(target, "memory", memory, prefetch=True)
        missing_slot = next(subject for subject in target.goal.slots
                            if subject not in {event.subject for event in target.environment.initial})
        # Same-length extra context, but the decoy has a separate symbol and no path role.
        decoy_subject = next(symbol for symbol in "ABCDEFGHIJKLMNOP" if symbol not in target.goal.slots)
        decoy = ExperienceEvent(target.goal.scope, decoy_subject, "irrelevant-public-note", 0, "A")
        slots = tuple(decoy_subject if subject == missing_slot else subject for subject in target.goal.slots)
        noisy = replace(target, goal=replace(target.goal, slots=slots),
                        environment=PublicEnvironment((*target.environment.initial, decoy), {}))
        execute(noisy, "same-token-extra-context")
    for case in revision_cases():
        execute(case, "public-query")

    after = policy.weights_sha256() if isinstance(policy, NeuralCore) else None
    if before != after:
        raise ValueError("frozen model weights changed during the episode")
    totals = {}
    for row in rows:
        key = row["trace"]["state"]["goal"]["family"] + "/" + row["arm"]
        item = totals.setdefault(key, {"correct": 0, "episodes": 0, "requests": 0,
                                       "model_positions": 0, "model_calls": 0})
        item["correct"] += row["outcome"]["correct"]
        item["episodes"] += 1
        item["requests"] += row["trace"]["requests"]
        for decision in row["trace"]["decisions"]:
            item["model_positions"] += decision["model_positions"]
            item["model_calls"] += decision["model_calls"]
    return {"schema": "elendril-cognitive-demo-v1", "model": policy.model_identity,
            "weights_sha256_before": before, "weights_sha256_after": after,
            "totals": totals, "episodes": rows}
