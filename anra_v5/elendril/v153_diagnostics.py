"""V.153 controlled development benchmark: 1-4 hops without row/tick shortcuts.

No neural training or inference is performed by this module. Every reported
contrast shares one synthetic world. The suite is a *development diagnostic*,
not a new sealed test set or evidence of internal neural composition.
"""
from __future__ import annotations

from dataclasses import replace
import random

from .codec import canonical, digest
from .cognition import CognitiveState, ExperienceEvent, Goal, PublicEnvironment, Truth
from .v6_data import SYMBOLS, V6Case, _partition, _state_key, case_to_record, public_action, render_public
from .v71_data import ChainSurface, _admitted, assert_training_disjoint

HOPS = (1, 2, 3, 4)
VARIANTS = (
    "original", "reversed", "shifted-timestamps",
    "remove-distractors", "replace-distractors", "remove-required-edge",
)


def _make_world(rng, seed, serial):
    """Create a unique unambiguous four-edge graph with four irrelevant edges."""
    nodes = rng.sample(SYMBOLS, 13)
    path_nodes, distractor_subjects, distractor_values = nodes[:5], nodes[5:9], nodes[9:13]
    ticks = rng.sample(range(120, 900), 8)
    world = digest(["v153-world-v1", seed, serial, path_nodes, distractor_subjects, ticks])
    scope = "world-" + world
    path = [
        ExperienceEvent(scope, path_nodes[i], "map", ticks[i], path_nodes[i + 1])
        for i in range(4)
    ]
    noise = [
        ExperienceEvent(scope, distractor_subjects[i], "map", ticks[4 + i], distractor_values[i])
        for i in range(4)
    ]
    # Make the maximum-timestamp fact irrelevant, without altering the true path.
    all_events = path + noise
    maximum = max(range(8), key=lambda i: all_events[i].tick)
    if maximum < 4:
        all_events[maximum], all_events[6] = (
            replace(all_events[maximum], tick=all_events[6].tick),
            replace(all_events[6], tick=all_events[maximum].tick),
        )
    path, noise = all_events[:4], all_events[4:]
    middle = [*path, noise[2], noise[3]]
    rng.shuffle(middle)
    events = (noise[0], *middle, noise[1])
    return world, tuple(path_nodes), tuple(events), tuple(e.subject for e in noise)


def _build_case(world, nodes, events, noise_subjects, hops, variant):
    key = digest(["v153-case-v1", world, hops, variant])
    scope = "world-" + world
    goal = Goal(key, "route", nodes[0], scope, "memory-" + world, nodes)
    env_events = events
    if variant == "shifted-timestamps":
        env_events = tuple(replace(e, tick=e.tick + 9000) for e in events)
    elif variant == "remove-distractors":
        env_events = tuple(e for e in events if e.subject not in noise_subjects)
    elif variant == "replace-distractors":
        replacements = [s for s in SYMBOLS if s not in nodes]
        env_events = tuple(
            replace(e, value=replacements[(replacements.index(e.value) + 1) % len(replacements)])
            if e.subject in noise_subjects else e
            for e in events
        )
    elif variant == "remove-required-edge":
        env_events = tuple(e for e in events if e.subject != nodes[hops - 1])
    elif variant not in ("original", "reversed"):
        raise ValueError("unsupported variant")
    env = PublicEnvironment(env_events, {})
    expected = None if variant == "remove-required-edge" else nodes[hops]
    case = V6Case(
        key, world, "route", "route", goal, env,
        Truth(key, expected, _admitted(env, None)),
        ChainSurface(hops=hops, order="reverse" if variant == "reversed" else "original"),
        pair_id=digest(["v153-pair", world, hops]), panel="v153-" + variant,
    )
    return case


def build_suite(*, seed=153071, worlds=16):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be uint32")
    if type(worlds) is not int or not 1 <= worlds <= 32:
        raise ValueError("worlds must be 1..32")
    rng = random.Random(int(digest(["v153-route-seed-v1", seed]), 16))
    suite = {f"hop-{h}|{v}": [] for h in HOPS for v in VARIANTS}
    admitted = set()
    attempts = 0
    while len(admitted) < worlds:
        attempts += 1
        if attempts > worlds * 100:
            raise ValueError("could not construct enough unique validation worlds")
        world, nodes, events, noise = _make_world(rng, seed, attempts)
        root = _build_case(world, nodes, events, noise, 4, "original")
        key = _state_key(CognitiveState(root.goal).observe(events), (), False)
        if _partition(key) != "validation" or key in admitted:
            continue
        admitted.add(key)
        for h in HOPS:
            for variant in VARIANTS:
                suite[f"hop-{h}|{variant}"].append(
                    _build_case(world, nodes, events, noise, h, variant)
                )
    return {name: tuple(cases) for name, cases in suite.items()}


def planned_contrasts(suite):
    keys = {f"hop-{h}|{v}" for h in HOPS for v in VARIANTS}
    if set(suite) != keys:
        raise ValueError("suite is missing a cohort or has an unexpected cohort")
    contrasts = []
    for h in HOPS:
        base = suite[f"hop-{h}|original"]
        for variant in VARIANTS[1:]:
            changed = suite[f"hop-{h}|{variant}"]
            if len(base) != len(changed) or any(
                a.world_id != b.world_id for a, b in zip(base, changed)
            ):
                raise ValueError("paired cases are not aligned")
            contrasts.append({
                "left": f"hop-{h}|original", "right": f"hop-{h}|{variant}",
                "factor": variant, "pairs": len(base),
            })
    return contrasts


def audit(suite, *, seed=153071, train=None):
    """Validate reproducibility, public-answer correctness and shortcut controls."""
    lengths = {len(v) for v in suite.values()}
    if len(lengths) != 1:
        raise ValueError("cohorts must all have the same number of worlds")
    count = next(iter(lengths))
    rebuilt = build_suite(seed=seed, worlds=count)
    if canonical({k: [case_to_record(c) for c in v] for k, v in suite.items()}) != canonical(
        {k: [case_to_record(c) for c in v] for k, v in rebuilt.items()}
    ):
        raise ValueError("suite differs from registered deterministic construction")
    contrasts = planned_contrasts(suite)
    shortcut_hits = {"last_visible_row": 0, "max_timestamp": 0}
    for name, cases in suite.items():
        for case in cases:
            state = CognitiveState(case.goal).observe(case.environment.initial)
            answer = public_action(state, (), case.surface, False)
            expected = ("ABSTAIN", None) if case.truth.expected is None else (
                "ANSWER", case.truth.expected
            )
            if answer != expected:
                raise ValueError("public teacher cannot solve the declared case")
            if len(render_public(state, (), case.surface).encode("utf-8")) + 6 > 512:
                raise ValueError("benchmark prompt exceeds S3 context")
            if set(case.truth.admitted_public) != {
                e.identity for e in case.environment.initial
            }:
                raise ValueError("truth/source admission does not match public events")
            if name.endswith("|original"):
                ordered = case.environment.initial
                if ordered[-1].value == case.truth.expected:
                    shortcut_hits["last_visible_row"] += 1
                if max(ordered, key=lambda e: e.tick).value == case.truth.expected:
                    shortcut_hits["max_timestamp"] += 1
    if any(shortcut_hits.values()):
        raise ValueError("simple row/timestamp copying solves an original case")
    disjoint = (
        {"status": "NOT_RUN", "reason": "actual training data not supplied"}
        if train is None else assert_training_disjoint(train, suite)
    )
    return {
        "schema": "elendril-v153-route-audit-v1", "status": "PASS",
        "seed": seed, "worlds": count, "cohorts": len(suite),
        "case_records": sum(len(v) for v in suite.values()),
        "planned_contrasts": len(contrasts),
        "shortcut_hits": shortcut_hits,
        "training_disjoint": disjoint,
        "suite_sha256": digest({k: [case_to_record(c) for c in v] for k, v in sorted(suite.items())}),
        "qualification": "Controlled development worlds; no neural inference, no sealed generalization and no claim of an internal mechanism.",
    }
