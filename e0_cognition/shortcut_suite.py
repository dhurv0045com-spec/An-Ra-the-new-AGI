"""Binding shortcut suite: competing trivial hypotheses (M30).

Every baseline predicts from task text alone, never from latent pairings or
gold. The harness fits frequency/template priors on a FIT cohort and scores
on a disjoint EVAL cohort, then judges each baseline against chance with
Wilson intervals (M31: uncertainty, not point estimates). A generator earns
qualification only when trivial features fail while the truth solver stays
perfect. Pair-destroyed controls are scored as pairs: the base answer must
change while bag features stay near-identical.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Mapping

from .statistics import wilson_interval

_WORD = re.compile(r"[A-Za-z0-9-]+")


def _facts(task: dict[str, Any]) -> list[str]:
    facts = task.get("facts")
    if isinstance(facts, (list, tuple)) and facts:
        return [str(fact) for fact in facts]
    return [line for line in str(task.get("facts_text", "")).split("\n") if line.strip()]


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _overlap(words: list[str], candidate: str) -> int:
    bag: dict[str, int] = {}
    for word in words:
        bag[word] = bag.get(word, 0) + 1
    return sum(bag.get(word, 0) for word in _words(candidate))


def _argmax(candidates: list[str], scores: list[float]) -> str:
    best = 0
    for index in range(1, len(scores)):
        if scores[index] > scores[best]:
            best = index
    return candidates[best]


def predict_truth(task: dict[str, Any]) -> str:
    """Ceiling reference: always gold. Excluded from gating by construction."""

    return str(task["gold"])


def predict_bag_of_words(task: dict[str, Any]) -> str:
    """E0 semantics: per-candidate containing-context Jaccard with the query."""

    query = set(word.lower() for word in _words(str(task["query"])))
    best: tuple[float, str] = (-1.0, task["candidates"][0])
    for candidate in task["candidates"]:
        containing = " ".join(fact for fact in _facts(task) if candidate in fact)
        tokens = set(word.lower() for word in _words(containing))
        union = query | tokens
        score = len(query & tokens) / len(union) if union else 0.0
        if (score, candidate) > best:
            best = (score, candidate)
    return best[1]


def predict_lexical_overlap(task: dict[str, Any]) -> str:
    """E0 semantics: best sentence by query overlap, then candidate in it."""

    query_tokens = set(word.lower() for word in _words(str(task["query"])))
    scored = []
    for index, fact in enumerate(_facts(task)):
        overlap = len(query_tokens & set(word.lower() for word in _words(fact)))
        scored.append((overlap, -index, fact))
    fact = max(scored)[2]
    hits = [candidate for candidate in task["candidates"] if candidate in fact]
    return hits[-1] if hits else task["candidates"][0]


def predict_latest_fact(task: dict[str, Any]) -> str:
    """E0 semantics: candidate from the last fact mentioning any candidate."""

    for fact in reversed(_facts(task)):
        hits = [candidate for candidate in task["candidates"] if candidate in fact]
        if hits:
            return hits[-1]
    return task["candidates"][0]


def predict_nearest_position(task: dict[str, Any]) -> str:
    """E0 semantics: candidate whose last context mention is nearest the query."""

    context = "\n".join(_facts(task))
    ranked = [(context.rfind(candidate), candidate) for candidate in task["candidates"]]
    present = [item for item in ranked if item[0] >= 0]
    return max(present)[1] if present else task["candidates"][0]


def predict_first_candidate(task: dict[str, Any]) -> str:
    return task["candidates"][0]


def predict_last_candidate(task: dict[str, Any]) -> str:
    return task["candidates"][-1]


def predict_query_only(task: dict[str, Any]) -> str:
    words = _words(str(task["query"]))
    return _argmax(task["candidates"], [_overlap(words, c) for c in task["candidates"]])


def predict_facts_only(task: dict[str, Any]) -> str:
    words = _words(str(task["facts_text"]))
    return _argmax(task["candidates"], [_overlap(words, c) for c in task["candidates"]])


def predict_unordered_set(task: dict[str, Any]) -> str:
    universe = set(_words(str(task["facts_text"])) + _words(str(task["query"])))
    return _argmax(
        task["candidates"],
        [len(set(_words(c)) & universe) for c in task["candidates"]],
    )


def predict_position(task: dict[str, Any], which: str) -> str:
    """Pick the value from a fixed fact position (first/middle/last)."""

    sentences = [line for line in str(task["facts_text"]).split("\n") if line.strip()]
    if not sentences:
        return task["candidates"][0]
    index = {"first": 0, "last": -1, "middle": len(sentences) // 2}[which]
    sentence = sentences[index]
    for candidate in task["candidates"]:
        if candidate in sentence:
            return candidate
    return task["candidates"][0]


def predict_target_distance(task: dict[str, Any], nearest: bool = True) -> str:
    """Pick the candidate closest to/farthest from the query entity mention."""

    query_words = _words(str(task["query"]))
    context_words = _words(str(task["facts_text"]))
    entity_positions = [
        index for index, word in enumerate(context_words) if word in query_words
    ]
    anchor = entity_positions[0] if entity_positions else 0
    scored = []
    for candidate in task["candidates"]:
        positions = [
            index for index, word in enumerate(context_words)
            if word in set(_words(candidate))
        ]
        distance = min((abs(index - anchor) for index in positions), default=10**9)
        scored.append(-distance if nearest else distance)
    return _argmax(task["candidates"], scored)


class FrequencyPrior:
    """Cohort-fit value/entity frequency prior (fit on FIT, applied on EVAL)."""

    def __init__(self, key: str) -> None:
        if key not in {"value", "entity"}:
            raise ValueError("frequency prior key must be value or entity")
        self.key = key
        self.counts: dict[str, int] = {}

    def fit(self, tasks: list[dict[str, Any]]) -> None:
        for task in tasks:
            for candidate in task["candidates"]:
                self.counts[candidate] = self.counts.get(candidate, 0) + (1 if self.key == "value" else 0)
        entities = [str(task.get("query_entity", "")) for task in tasks]
        if self.key == "entity":
            self.counts = {}
            for entity in entities:
                if entity:
                    self.counts[entity] = self.counts.get(entity, 0) + 1

    def predict(self, task: dict[str, Any]) -> str:
        return _argmax(
            task["candidates"], [float(self.counts.get(c, 0)) for c in task["candidates"]]
        )


class TemplatePrior:
    """Majority gold per surface template, fit on FIT only."""

    def __init__(self) -> None:
        self.majority: dict[str, str] = {}
        self.fallback = ""

    def fit(self, tasks: list[dict[str, Any]]) -> None:
        from collections import Counter

        by_template: dict[str, Counter] = {}
        for task in tasks:
            by_template.setdefault(str(task.get("grammar", "")), Counter())[str(task["gold"])] += 1
        self.majority = {template: counter.most_common(1)[0][0] for template, counter in by_template.items()}
        overall: Counter = Counter()
        for counter in by_template.values():
            overall.update(counter)
        self.fallback = overall.most_common(1)[0][0] if overall else ""

    def predict(self, task: dict[str, Any]) -> str:
        guess = self.majority.get(str(task.get("grammar", "")), self.fallback)
        if guess in task["candidates"]:
            return guess
        return task["candidates"][0]


class CentroidProbe:
    """Nearest-centroid linear model over cheap scalar features."""

    FEATURES = ("prompt_len", "n_candidates", "gold_len", "max_overlap", "query_len")

    def _vector(self, task: dict[str, Any]) -> list[float]:
        words = _words(str(task["facts_text"]) + " " + str(task["query"]))
        overlaps = sorted((_overlap(words, c) for c in task["candidates"]), reverse=True)
        return [
            float(len(str(task["facts_text"]))),
            float(len(task["candidates"])),
            float(len(str(task["gold"]))),
            float(overlaps[0]) if overlaps else 0.0,
            float(len(_words(str(task["query"])))),
        ]

    def fit(self, tasks: list[dict[str, Any]]) -> None:
        sums: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        for task in tasks:
            gold = str(task["gold"])
            vector = self._vector(task)
            sums.setdefault(gold, [0.0] * len(vector))
            counts[gold] = counts.get(gold, 0) + 1
            for index, value in enumerate(vector):
                sums[gold][index] += value
        if len(sums) < 2:
            raise ValueError("centroid probe needs at least two gold classes to fit")
        self.centroids = {
            gold: [total / counts[gold] for total in totals] for gold, totals in sums.items()
        }

    def predict(self, task: dict[str, Any]) -> str:
        vector = self._vector(task)
        best, best_distance = task["candidates"][0], math.inf
        for candidate in task["candidates"]:
            centroid = self.centroids.get(candidate)
            if centroid is None:
                continue
            distance = sum((a - b) ** 2 for a, b in zip(vector, centroid))
            if distance < best_distance:
                best, best_distance = candidate, distance
        return best


BASELINES: dict[str, Callable[[dict[str, Any]], str]] = {
    "truth_solver": predict_truth,
    "bag_of_words": predict_bag_of_words,
    "lexical_overlap": predict_lexical_overlap,
    "latest_fact": predict_latest_fact,
    "nearest_position": predict_nearest_position,
    "first_candidate": predict_first_candidate,
    "last_candidate": predict_last_candidate,
    "query_only": predict_query_only,
    "facts_only": predict_facts_only,
    "unordered_set": predict_unordered_set,
    "position_first": lambda task: predict_position(task, "first"),
    "position_middle": lambda task: predict_position(task, "middle"),
    "position_last": lambda task: predict_position(task, "last"),
    "target_nearest": lambda task: predict_target_distance(task, True),
    "target_farthest": lambda task: predict_target_distance(task, False),
}


def run_suite(
    fit_tasks: list[dict[str, Any]], eval_tasks: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """Fit priors on FIT, score every baseline on EVAL; return accuracies."""

    if not fit_tasks or not eval_tasks:
        raise ValueError("shortcut suite needs nonempty fit and eval cohorts")
    value_prior = FrequencyPrior("value")
    value_prior.fit(fit_tasks)
    template_prior = TemplatePrior()
    template_prior.fit(fit_tasks)
    probe = CentroidProbe()
    try:
        probe.fit(fit_tasks)
        probe_ready = True
    except ValueError:
        probe_ready = False
    results: dict[str, dict[str, float]] = {}
    for name, baseline in BASELINES.items():
        correct = sum(1 for task in eval_tasks if baseline(task) == task["gold"])
        results[name] = {"accuracy": correct / len(eval_tasks), "n": len(eval_tasks)}
    for name, predictor in (
        ("value_frequency", value_prior.predict),
        ("surface_template", template_prior.predict),
    ):
        correct = sum(1 for task in eval_tasks if predictor(task) == task["gold"])
        results[name] = {"accuracy": correct / len(eval_tasks), "n": len(eval_tasks)}
    if probe_ready:
        correct = sum(1 for task in eval_tasks if probe.predict(task) == task["gold"])
        results["linear_centroid"] = {"accuracy": correct / len(eval_tasks), "n": len(eval_tasks)}
    return results


def pair_sensitivity(
    predict: Callable[[dict[str, Any]], str],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, float]:
    """Base-vs-control flip behavior: a structure-blind model cannot flip correctly."""

    if not pairs:
        raise ValueError("pair sensitivity needs base/control pairs")
    base_correct = 0
    both_correct = 0
    flipped_right = 0
    for base, control in pairs:
        base_guess = predict(base)
        control_guess = predict(control)
        base_ok = base_guess == base["gold"]
        control_ok = control_guess == control["gold"]
        base_correct += int(base_ok)
        both_correct += int(base_ok and control_ok)
        flipped_right += int(base_ok and control_ok and control_guess != base_guess)
    total = len(pairs)
    return {
        "base_accuracy": base_correct / total,
        "pair_accuracy": both_correct / total,
        "flip_rate": flipped_right / total,
    }


def qualify_pairs(
    pair_accuracies: Mapping[str, float],
    pair_counts: Mapping[str, int],
    *,
    null_ceiling: float,
    max_excess: float,
    truth_pair_accuracy: float = 1.0,
) -> dict[str, object]:
    """Gate pair accuracies against a structure-blind null ceiling.

    The null is NOT uniform chance: it is the best score achievable with
    addressing but no selection (e.g. 0.25 for two-version interference:
    perfect entity retrieval plus a random version guess). A baseline passes
    only when its Wilson UPPER bound stays within the ceiling plus margin;
    wide intervals fail closed. Truth must be perfect (validity).
    """

    from .statistics import wilson_interval

    if truth_pair_accuracy != 1.0:
        return {
            "null_ceiling": null_ceiling,
            "verdict": "GENERATOR_NOT_QUALIFIED",
            "reason": "truth solver is imperfect; the generator is invalid",
        }
    verdicts: dict[str, object] = {}
    for name, accuracy in pair_accuracies.items():
        total = int(pair_counts[name])
        _lower, upper = wilson_interval(int(round(accuracy * total)), total)
        verdicts[name] = {
            "pair_accuracy": accuracy,
            "n": total,
            "excess_upper_bound": upper - null_ceiling,
            "pass": bool(upper <= null_ceiling + max_excess),
        }
    passed = all(entry["pass"] for entry in verdicts.values())  # type: ignore[union-attr]
    return {
        "null_ceiling": null_ceiling,
        "max_excess_allowed": max_excess,
        "baselines": verdicts,
        "verdict": "GENERATOR_QUALIFIED" if passed else "GENERATOR_NOT_QUALIFIED",
    }


def qualify_against_suite(
    results: Mapping[str, Mapping[str, float]],
    *,
    chance: float,
    max_excess: float,
    exclude: tuple[str, ...] = ("truth_solver",),
) -> dict[str, object]:
    """Judge baselines with Wilson upper bounds, never point estimates.

    A baseline passes the gate (generator survives it) only when the UPPER
    bound of its excess over chance stays within max_excess. Wide intervals
    from small samples fail closed toward NOT_QUALIFIED.
    """

    verdicts: dict[str, object] = {}
    worst_excess = 0.0
    for name, result in results.items():
        if name in exclude:
            continue
        correct = int(round(float(result["accuracy"]) * int(result["n"])))
        total = int(result["n"])
        _lower, upper = wilson_interval(correct, total)
        excess_upper = upper - chance
        verdicts[name] = {
            "accuracy": float(result["accuracy"]),
            "n": total,
            "excess_upper_bound": excess_upper,
            "pass": bool(excess_upper <= max_excess),
        }
        worst_excess = max(worst_excess, float(result["accuracy"]) - chance)
    passed = all(entry["pass"] for entry in verdicts.values())  # type: ignore[union-attr]
    return {
        "chance": chance,
        "max_excess_allowed": max_excess,
        "baselines": verdicts,
        "worst_point_excess": worst_excess,
        "verdict": "GENERATOR_QUALIFIED" if passed else "GENERATOR_NOT_QUALIFIED",
    }


__all__ = [
    "BASELINES",
    "CentroidProbe",
    "FrequencyPrior",
    "TemplatePrior",
    "pair_sensitivity",
    "predict_bag_of_words",
    "predict_truth",
    "qualify_against_suite",
    "qualify_pairs",
    "run_suite",
]
