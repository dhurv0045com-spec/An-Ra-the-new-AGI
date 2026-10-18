from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable

from .contracts import ACTION_COSTS, Action, ActionDescriptor, Policy, PublicView
from .generator import GeneratedWorld
from .rules import CATALOG, RuleSpec, all_assignments, evaluate_rule, overridden_assignment


def _action_from_descriptor(descriptor: ActionDescriptor, *, prediction: int | None = None) -> Action:
    arguments = dict(descriptor.arguments)
    if descriptor.kind == "submit":
        if prediction is None:
            prediction = arguments["answer"]
        arguments["answer"] = prediction
    if descriptor.kind == "stop":
        prediction = None
    return Action.make(descriptor.kind, prediction=prediction, **arguments)


def _target_candidates(view: PublicView) -> list[tuple[int, int, int]]:
    public_probes = {tuple(item) for item in view.probe_contexts}
    values = [list(item) for item in all_assignments() if tuple(item) not in public_probes]
    for event in view.transcript:
        feedback = event.feedback
        if not event.accepted or feedback.get("kind") != "read":
            continue
        try:
            index = view.variable_names.index(feedback["variable"])
            value = feedback["value"]
        except (ValueError, KeyError):
            continue
        values = [assignment for assignment in values if assignment[index] == value]
    return [tuple(item) for item in values] or list(all_assignments())


def _posterior(view: PublicView) -> list[RuleSpec]:
    posterior = list(CATALOG)
    for event in view.transcript:
        feedback = event.feedback
        if not event.accepted or feedback.get("kind") not in {"probe", "intervene"}:
            continue
        if "outcome" not in feedback:
            continue
        outcome = feedback["outcome"]
        if feedback["kind"] == "probe":
            context = tuple(feedback["context"])
            posterior = [spec for spec in posterior if evaluate_rule(spec, context) == outcome]
        else:
            context = tuple(feedback["context"])
            variable_name = feedback["intervention"]["variable"]
            variable = view.variable_names.index(variable_name)
            value = feedback["intervention"]["value"]
            counterfactual = overridden_assignment(context, variable, value)
            posterior = [spec for spec in posterior if evaluate_rule(spec, counterfactual) == outcome]
    return posterior or list(CATALOG)


def _answer_probability(view: PublicView, posterior: Iterable[RuleSpec] | None = None, candidates: Iterable[tuple[int, int, int]] | None = None) -> float:
    specs = list(posterior if posterior is not None else _posterior(view))
    assignments = list(candidates if candidates is not None else _target_candidates(view))
    if not specs or not assignments:
        return 0.5
    positive = sum(spec.evaluate(assignment) for spec in specs for assignment in assignments)
    total = len(specs) * len(assignments)
    return positive / total


def _entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(probability * math.log2(probability) + (1.0 - probability) * math.log2(1.0 - probability))


def _filtered_candidates(view: PublicView, variable: str, value: int) -> list[tuple[int, int, int]]:
    try:
        index = view.variable_names.index(variable)
    except ValueError:
        return _target_candidates(view)
    return [item for item in _target_candidates(view) if item[index] == value]


def _expected_entropy(
    view: PublicView,
    descriptor: ActionDescriptor,
    posterior: list[RuleSpec],
    candidates: list[tuple[int, int, int]],
) -> float:
    if descriptor.kind == "read":
        variable = descriptor.arguments["variable"]
        branches = ((0, _filtered_candidates(view, variable, 0)), (1, _filtered_candidates(view, variable, 1)))
        total = max(1, len(candidates))
        return sum(
            (len(branch) / total) * _entropy(_answer_probability(view, posterior, branch))
            for _, branch in branches
        )
    if descriptor.kind in {"probe", "intervene"}:
        if descriptor.kind == "probe":
            context = tuple(descriptor.arguments["context"])
            assignment = context
        else:
            context = tuple(descriptor.arguments["context"])
            variable = view.variable_names.index(descriptor.arguments["variable"])
            value = descriptor.arguments["value"]
            assignment = overridden_assignment(context, variable, value)
        expected = 0.0
        for outcome in (0, 1):
            branch = [spec for spec in posterior if evaluate_rule(spec, assignment) == outcome]
            if not branch:
                continue
            expected += len(branch) / len(posterior) * _entropy(_answer_probability(view, branch, candidates))
        return expected
    return _entropy(_answer_probability(view, posterior, candidates))


def _prediction_for_action(
    view: PublicView,
    descriptor: ActionDescriptor,
    posterior: list[RuleSpec],
    candidates: list[tuple[int, int, int]],
) -> int:
    if descriptor.kind == "submit":
        return int(_answer_probability(view, posterior, candidates) >= 0.5)
    if descriptor.kind == "read":
        variable = view.variable_names.index(descriptor.arguments["variable"])
        probability = sum(item[variable] for item in candidates) / max(1, len(candidates))
        return int(probability >= 0.5)
    if descriptor.kind == "probe":
        assignment = tuple(descriptor.arguments["context"])
    else:
        context = tuple(descriptor.arguments["context"])
        variable = view.variable_names.index(descriptor.arguments["variable"])
        value = descriptor.arguments["value"]
        assignment = overridden_assignment(context, variable, value)
    probability = sum(evaluate_rule(spec, assignment) for spec in posterior) / max(1, len(posterior))
    return int(probability >= 0.5)


def _lookahead_action_key(action: tuple[str, tuple[tuple[str, object], ...], int]) -> tuple[object, ...]:
    kind, arguments, _ = action
    return (kind, tuple(arguments))


def _lookahead_initial_state(view: PublicView) -> tuple[tuple[RuleSpec, ...], tuple[tuple[int, int, int], ...], frozenset[tuple[object, ...]], int]:
    used: set[tuple[object, ...]] = set()
    for event in view.transcript:
        feedback = event.feedback
        if not event.accepted:
            continue
        if feedback.get("kind") == "read":
            used.add(("read", feedback["variable"]))
        elif feedback.get("kind") == "probe":
            used.add(("probe", tuple(feedback["context"])))
        elif feedback.get("kind") == "intervene":
            used.add(
                (
                    "intervene",
                    tuple(feedback["context"]),
                    feedback["intervention"]["variable"],
                    feedback["intervention"]["value"],
                )
            )
    return (
        tuple(_posterior(view)),
        tuple(_target_candidates(view)),
        frozenset(used),
        view.remaining_budget,
    )


def _lookahead_actions(
    view: PublicView,
    state: tuple[tuple[RuleSpec, ...], tuple[tuple[int, int, int], ...], frozenset[tuple[object, ...]], int],
    intervention_contexts: tuple[tuple[int, int, int], ...],
    allow_intervention: bool,
) -> list[tuple[str, tuple[tuple[str, object], ...], int]]:
    _, _, used, remaining = state
    actions: list[tuple[str, tuple[tuple[str, object], ...], int]] = []
    for variable in view.variable_names:
        key = ("read", variable)
        if key not in used and ACTION_COSTS["read"] + ACTION_COSTS["submit"] <= remaining:
            actions.append(("read", (("variable", variable),), ACTION_COSTS["read"]))
    for context in view.probe_contexts:
        key = ("probe", tuple(context))
        if key not in used and ACTION_COSTS["probe"] + ACTION_COSTS["submit"] <= remaining:
            actions.append(("probe", (("context", tuple(context)),), ACTION_COSTS["probe"]))
    if allow_intervention:
        for context in intervention_contexts:
            for variable in view.variable_names:
                for value in (0, 1):
                    key = ("intervene", tuple(context), variable, value)
                    if key not in used and ACTION_COSTS["intervene"] + ACTION_COSTS["submit"] <= remaining:
                        actions.append(
                            (
                                "intervene",
                                (("context", tuple(context)), ("variable", variable), ("value", value)),
                                ACTION_COSTS["intervene"],
                            )
                        )
    return actions


def _lookahead_branches(
    state: tuple[tuple[RuleSpec, ...], tuple[tuple[int, int, int], ...], frozenset[tuple[object, ...]], int],
    action: tuple[str, tuple[tuple[str, object], ...], int],
    view: PublicView,
) -> list[tuple[int, tuple[tuple[RuleSpec, ...], tuple[tuple[int, int, int], ...], frozenset[tuple[object, ...]], int]]]:
    posterior, candidates, used, remaining = state
    kind, argument_items, cost = action
    arguments = dict(argument_items)
    branches: list[tuple[int, tuple[tuple[RuleSpec, ...], tuple[tuple[int, int, int], ...], frozenset[tuple[object, ...]], int]]] = []
    if kind == "read":
        index = view.variable_names.index(arguments["variable"])
        key = ("read", arguments["variable"])
        for outcome in (0, 1):
            next_candidates = tuple(item for item in candidates if item[index] == outcome)
            if next_candidates:
                branches.append((outcome, (posterior, next_candidates, used | {key}, remaining - cost)))
        return branches
    if kind == "probe":
        context = tuple(arguments["context"])
        key = ("probe", context)
        for outcome in (0, 1):
            next_posterior = tuple(spec for spec in posterior if evaluate_rule(spec, context) == outcome)
            if next_posterior:
                branches.append((outcome, (next_posterior, candidates, used | {key}, remaining - cost)))
        return branches
    context = tuple(arguments["context"])
    variable = view.variable_names.index(arguments["variable"])
    value = arguments["value"]
    assignment = overridden_assignment(context, variable, value)
    key = ("intervene", context, arguments["variable"], value)
    for outcome in (0, 1):
        next_posterior = tuple(spec for spec in posterior if evaluate_rule(spec, assignment) == outcome)
        if next_posterior:
            branches.append((outcome, (next_posterior, candidates, used | {key}, remaining - cost)))
    return branches


def _lookahead_terminal_value(
    posterior: tuple[RuleSpec, ...],
    candidates: tuple[tuple[int, int, int], ...],
) -> float:
    if not posterior or not candidates:
        return 0.5
    probability = sum(spec.evaluate(candidate) for spec in posterior for candidate in candidates) / (len(posterior) * len(candidates))
    return max(probability, 1.0 - probability)


class TransparentLookaheadPolicy(Policy):
    identity = "transparent-lookahead/v1"
    uses_hidden_truth = False

    def __init__(self, *, max_expansions: int = 2500) -> None:
        self.max_expansions = max_expansions
        self._view: PublicView | None = None

    def act(self, view: PublicView) -> Action:
        self._view = view
        state = _lookahead_initial_state(view)
        intervention_contexts = tuple(
            sorted(
                {
                    tuple(descriptor.arguments["context"])
                    for descriptor in view.legal_actions
                    if descriptor.kind == "intervene"
                }
            )
        )
        allow_intervention = bool(intervention_contexts)
        memo: dict[tuple[object, ...], float] = {}
        expansions = 0

        def value(current: tuple[tuple[RuleSpec, ...], tuple[tuple[int, int, int], ...], frozenset[tuple[object, ...]], int]) -> float:
            nonlocal expansions
            key = (
                tuple(spec.semantic_sha256 for spec in current[0]),
                current[1],
                tuple(sorted(current[2], key=repr)),
                current[3],
            )
            if key in memo:
                return memo[key]
            if current[3] <= ACTION_COSTS["submit"]:
                result = _lookahead_terminal_value(current[0], current[1])
                memo[key] = result
                return result
            if expansions >= self.max_expansions:
                return _lookahead_terminal_value(current[0], current[1])
            expansions += 1
            best = _lookahead_terminal_value(current[0], current[1])
            for action in _lookahead_actions(view, current, intervention_contexts, allow_intervention):
                branches = _lookahead_branches(current, action, view)
                if not branches:
                    continue
                expected = sum(value(branch_state) for _, branch_state in branches) / len(branches)
                best = max(best, expected)
            memo[key] = best
            return best

        actions = _lookahead_actions(view, state, intervention_contexts, allow_intervention)
        if not actions:
            submit = next((item for item in view.legal_actions if item.kind == "submit"), None)
            if submit is not None:
                probability = _answer_probability(view, state[0], state[1])
                return _action_from_descriptor(submit, prediction=int(probability >= 0.5))
            return _action_from_descriptor(view.legal_actions[0])
        ranked: list[tuple[float, int, str, tuple[str, tuple[tuple[str, object], ...], int]]] = []
        for action in actions:
            branches = _lookahead_branches(state, action, view)
            if not branches:
                continue
            expected = sum(value(branch_state) for _, branch_state in branches) / len(branches)
            ranked.append((expected, -action[2], repr(_lookahead_action_key(action)), action))
        _, _, _, selected = max(ranked, key=lambda item: item[:3])
        kind, argument_items, _ = selected
        arguments = dict(argument_items)
        if kind == "submit":
            probability = _answer_probability(view, state[0], state[1])
            submit = next(item for item in view.legal_actions if item.kind == "submit")
            return _action_from_descriptor(submit, prediction=int(probability >= 0.5))
        descriptor = ActionDescriptor(
            kind=kind,
            arguments=arguments,
            cost=ACTION_COSTS[kind],
            outcome_domain=(0, 1),
            mode="interventional" if kind == "intervene" else "observational",
            preconditions=(),
            state_changes=(),
        )
        prediction = _prediction_for_action(view, descriptor, list(state[0]), list(state[1]))
        return _action_from_descriptor(descriptor, prediction=prediction)


class InformationGainPolicy(Policy):
    identity = "transparent-information-gain/v1"
    uses_hidden_truth = False

    def __init__(self) -> None:
        self._view: PublicView | None = None

    def act(self, view: PublicView) -> Action:
        posterior = _posterior(view)
        candidates = _target_candidates(view)
        current = _entropy(_answer_probability(view, posterior, candidates))
        nonterminal = [
            item
            for item in view.legal_actions
            if item.kind not in {"submit", "stop"}
            and item.cost + ACTION_COSTS["submit"] <= view.remaining_budget
        ]
        if current <= 1e-12 or not nonterminal:
            submit = next((item for item in view.legal_actions if item.kind == "submit"), None)
            if submit is not None:
                return _action_from_descriptor(
                    submit,
                    prediction=int(_answer_probability(view, posterior, candidates) >= 0.5),
                )
            return _action_from_descriptor(view.legal_actions[0])
        scored: list[tuple[float, float, int, ActionDescriptor]] = []
        for index, descriptor in enumerate(nonterminal):
            expected = _expected_entropy(view, descriptor, posterior, candidates)
            gain = max(0.0, current - expected)
            cost = max(1, descriptor.cost)
            scored.append((gain / cost, gain, -index, descriptor))
        best_gain, _, _, selected = max(scored, key=lambda item: (item[0], item[1], item[2]))
        if best_gain <= 1e-12:
            submit = next((item for item in view.legal_actions if item.kind == "submit"), None)
            if submit is not None:
                return _action_from_descriptor(
                    submit,
                    prediction=int(_answer_probability(view, posterior, candidates) >= 0.5),
                )
        prediction = _prediction_for_action(view, selected, posterior, candidates)
        return _action_from_descriptor(selected, prediction=prediction)


class RandomPolicy(Policy):
    identity = "random-legal/v1"
    uses_hidden_truth = False

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self, episode_id: str) -> None:
        self._rng = random.Random(self.seed + int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16))

    def act(self, view: PublicView) -> Action:
        descriptor = self._rng.choice(view.legal_actions)
        prediction = None
        if descriptor.kind in {"read", "probe", "intervene"}:
            prediction = self._rng.randint(0, 1)
        return _action_from_descriptor(descriptor, prediction=prediction)


class FixedSchedulePolicy(Policy):
    identity = "fixed-schedule/v1"
    uses_hidden_truth = False

    def __init__(self) -> None:
        self._read: set[str] = set()
        self._probes: set[tuple[int, ...]] = set()

    def reset(self, episode_id: str) -> None:
        self._read = set()
        self._probes = set()

    def act(self, view: PublicView) -> Action:
        for descriptor in view.legal_actions:
            if descriptor.kind == "read" and descriptor.arguments["variable"] not in self._read:
                self._read.add(descriptor.arguments["variable"])
                return _action_from_descriptor(descriptor, prediction=0)
        for descriptor in view.legal_actions:
            if descriptor.kind == "probe":
                context = tuple(descriptor.arguments["context"])
                if context not in self._probes:
                    self._probes.add(context)
                    return _action_from_descriptor(descriptor, prediction=0)
        submit = next((item for item in view.legal_actions if item.kind == "submit"), None)
        if submit is not None:
            probability = _answer_probability(view)
            return _action_from_descriptor(submit, prediction=int(probability >= 0.5))
        return _action_from_descriptor(view.legal_actions[0])


class NoExperimentPolicy(Policy):
    identity = "no-experiment/v1"
    uses_hidden_truth = False

    def act(self, view: PublicView) -> Action:
        submit = next((item for item in view.legal_actions if item.kind == "submit"), None)
        if submit is not None:
            answer = int(_answer_probability(view) >= 0.5)
            return _action_from_descriptor(submit, prediction=answer)
        return _action_from_descriptor(view.legal_actions[0])


class UncertaintyPolicy(Policy):
    identity = "uncertainty-seeking/v1"
    uses_hidden_truth = False

    def act(self, view: PublicView) -> Action:
        candidates = [
            item
            for item in view.legal_actions
            if item.kind in {"probe", "intervene"}
            and len(item.outcome_domain) > 1
            and item.cost + ACTION_COSTS["submit"] <= view.remaining_budget
        ]
        if not candidates:
            candidates = [
                item
                for item in view.legal_actions
                if item.kind == "read" and item.cost + ACTION_COSTS["submit"] <= view.remaining_budget
            ]
        if candidates:
            return _action_from_descriptor(candidates[0], prediction=0)
        submit = next((item for item in view.legal_actions if item.kind == "submit"), None)
        if submit is not None:
            return _action_from_descriptor(submit, prediction=int(_answer_probability(view) >= 0.5))
        return _action_from_descriptor(view.legal_actions[0])


class SurfaceShortcutPolicy(Policy):
    identity = "surface-position-shortcut/v1"
    uses_hidden_truth = False

    def __init__(self) -> None:
        self._read = False

    def reset(self, episode_id: str) -> None:
        self._read = False

    def act(self, view: PublicView) -> Action:
        if not self._read:
            read = next((item for item in view.legal_actions if item.kind == "read"), None)
            if read is not None:
                self._read = True
                return _action_from_descriptor(read, prediction=0)
        submit = next((item for item in view.legal_actions if item.kind == "submit"), None)
        if submit is not None:
            return _action_from_descriptor(submit, prediction=0)
        return _action_from_descriptor(view.legal_actions[0])


class TaskIdentityShortcutPolicy(Policy):
    identity = "task-identity-shortcut/v1"
    uses_hidden_truth = False

    def act(self, view: PublicView) -> Action:
        submit = next((item for item in view.legal_actions if item.kind == "submit"), None)
        if submit is None:
            return _action_from_descriptor(view.legal_actions[0])
        digest = hashlib.sha256(view.episode_id.encode()).hexdigest()
        answer = int(digest[-1], 16) % 2
        return _action_from_descriptor(submit, prediction=answer)


class OraclePolicy(Policy):
    identity = "evaluator-oracle/v1"
    uses_hidden_truth = True

    def __init__(self, world: GeneratedWorld) -> None:
        self.world = world

    def act(self, view: PublicView) -> Action:
        submit = next((item for item in view.legal_actions if item.kind == "submit"), None)
        if submit is not None:
            return _action_from_descriptor(submit, prediction=self.world.hidden_answer)
        raise RuntimeError("oracle cannot act after the episode terminates")


def policy_factories(seed: int):
    return {
        "random": lambda world: RandomPolicy(seed + 1),
        "fixed": lambda world: FixedSchedulePolicy(),
        "uncertainty": lambda world: UncertaintyPolicy(),
        "information_gain": lambda world: InformationGainPolicy(),
        "no_experiment": lambda world: NoExperimentPolicy(),
        "surface_shortcut": lambda world: SurfaceShortcutPolicy(),
        "task_identity_shortcut": lambda world: TaskIdentityShortcutPolicy(),
        "oracle": lambda world: OraclePolicy(world),
    }


def extended_policy_factories(seed: int):
    factories = policy_factories(seed)
    factories["lookahead"] = lambda world: TransparentLookaheadPolicy()
    return factories
