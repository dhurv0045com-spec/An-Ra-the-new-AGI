"""Independent evaluation and candidate promotion (B09).

Correctness is always recomputed from raw predictions — forged success fields
are structurally impossible here because outcome records carry no success
field at all. Paired goal metrics expose query-blind behavior even when
aggregate accuracy looks acceptable. Family-level reporting prevents an
aggregate improvement from hiding a protected-family regression. Promotion
defaults to no-promotion whenever evidence is insufficient.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class EvaluationError(ValueError):
    """An evaluation input violates the scoring contract."""


PROMOTION_DECISIONS = frozenset({"accept", "reject", "diagnose", "no_promotion"})
OUTCOME_ROLES = frozenset({"primary", "swapped"})


@dataclass(frozen=True)
class RawOutcome:
    """One immutable raw evaluation outcome. No derived success field exists.

    ``prediction`` is the raw generated answer; ``stopped_on_eos`` is the raw
    stopping evidence. ``correct`` is never stored — it is recomputed.
    """

    outcome_id: str
    pool: str                 # measurement | confirmation | sealed (storage-separated)
    split: str
    family: str
    task_semantic_id: str
    prediction: str | None
    stopped_on_eos: bool
    label: str
    cost: float
    pair_group_id: str | None = None
    role: str | None = None   # for paired items: primary | swapped
    confidence: float | None = None  # optional declared confidence for Brier

    def __post_init__(self) -> None:
        for name in ("outcome_id", "pool", "split", "family", "task_semantic_id", "label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise EvaluationError(f"{name} must be a nonempty string")
        if self.prediction is not None and not isinstance(self.prediction, str):
            raise EvaluationError("prediction must be a string or None")
        if not isinstance(self.stopped_on_eos, bool):
            raise EvaluationError("stopped_on_eos must be a boolean")
        if not isinstance(self.cost, (int, float)) or isinstance(self.cost, bool) or self.cost < 0:
            raise EvaluationError("cost must be a nonnegative number")
        if self.role is not None and self.role not in OUTCOME_ROLES:
            raise EvaluationError(f"role must be one of {sorted(OUTCOME_ROLES)}")
        if (self.role is None) != (self.pair_group_id is None) and self.pair_group_id \
                and self.role is None:
            raise EvaluationError("paired outcomes must declare their role")
        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool) \
                    or not 0.0 <= float(self.confidence) <= 1.0:
                raise EvaluationError("confidence must lie within [0, 1]")

    def complete_answer_correct(self) -> bool:
        """Exact answer AND valid stopping, recomputed from raw fields.

        A prediction that matches the label but stopped on the cap instead of
        EOS is not a complete answer and never counts as correct.
        """
        return self.stopped_on_eos and self.prediction == self.label

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def complete_answer_metrics(outcomes: Sequence[RawOutcome]) -> dict[str, Any]:
    """Exact+EOS accuracy, exact-only rate and cost totals, recomputed."""
    if not outcomes:
        raise EvaluationError("no outcomes to score")
    total = len(outcomes)
    exact = sum(1 for outcome in outcomes
                if outcome.prediction == outcome.label)
    complete = sum(1 for outcome in outcomes if outcome.complete_answer_correct())
    eos_invalid_but_exact = exact - sum(
        1 for outcome in outcomes if outcome.complete_answer_correct())
    return {
        "count": total,
        "complete_answer_rate": complete / total,
        "exact_answer_rate": exact / total,
        "exact_but_stopping_invalid": eos_invalid_but_exact,
        "cost_total": float(sum(outcome.cost for outcome in outcomes)),
        "recomputed": True,
    }


def paired_goal_metrics(outcomes: Sequence[RawOutcome]) -> dict[str, Any]:
    """Both-correct rate, same-answer rate and goal-swap gap over pairs.

    Each pair group must contain exactly two outcomes, one primary and one
    swapped rendering. Missing members or duplicated roles reject — a partial
    pair can never silently enter a metric.
    """
    groups: dict[str, list[RawOutcome]] = {}
    for outcome in outcomes:
        if outcome.pair_group_id is None:
            continue
        groups.setdefault(outcome.pair_group_id, []).append(outcome)
    if not groups:
        raise EvaluationError("no paired outcomes supplied")
    both_correct = 0
    same_answer = 0
    primary_correct = 0
    swapped_correct = 0
    for group_id, members in sorted(groups.items()):
        if len(members) != 2:
            raise EvaluationError(
                f"pair {group_id!r} has {len(members)} outcomes; exactly 2 required")
        roles = sorted(member.role for member in members)
        if roles != ["primary", "swapped"]:
            raise EvaluationError(
                f"pair {group_id!r} must contain one primary and one swapped outcome; "
                f"found {roles}")
        primary = next(member for member in members if member.role == "primary")
        swapped = next(member for member in members if member.role == "swapped")
        if primary.complete_answer_correct():
            primary_correct += 1
        if swapped.complete_answer_correct():
            swapped_correct += 1
        if primary.complete_answer_correct() and swapped.complete_answer_correct():
            both_correct += 1
        if primary.prediction is not None and primary.prediction == swapped.prediction:
            same_answer += 1
    count = len(groups)
    return {
        "pairs": count,
        "both_correct_rate": both_correct / count,
        "same_answer_rate": same_answer / count,
        "goal_swap_gap": (primary_correct - swapped_correct) / count,
        "primary_accuracy": primary_correct / count,
        "swapped_accuracy": swapped_correct / count,
    }


def family_metrics(outcomes: Sequence[RawOutcome]) -> dict[str, dict[str, float]]:
    """Per-family complete-answer rates; the unit of retention reporting."""
    by_family: dict[str, list[RawOutcome]] = {}
    for outcome in outcomes:
        by_family.setdefault(outcome.family, []).append(outcome)
    report: dict[str, dict[str, float]] = {}
    for family, members in sorted(by_family.items()):
        correct = sum(1 for member in members if member.complete_answer_correct())
        report[family] = {
            "count": len(members),
            "complete_answer_rate": correct / len(members),
            "cost_total": float(sum(member.cost for member in members)),
        }
    return report


@dataclass(frozen=True)
class RetentionFinding:
    family: str
    reference_rate: float
    candidate_rate: float
    delta: float
    regressed: bool


def retention_report(reference: Mapping[str, dict[str, float]],
                     candidate: Mapping[str, dict[str, float]], *,
                     regression_margin: float) -> dict[str, Any]:
    """Family-level retention; the worst family is always surfaced.

    An aggregate mean cannot override a protected-family failure: the report
    carries the worst delta and any family beyond the declared margin.
    """
    findings: list[RetentionFinding] = []
    for family in sorted(set(reference) | set(candidate)):
        ref = reference.get(family, {}).get("complete_answer_rate", 0.0)
        cand = candidate.get(family, {}).get("complete_answer_rate", 0.0)
        delta = cand - ref
        findings.append(RetentionFinding(
            family=family, reference_rate=ref, candidate_rate=cand, delta=delta,
            regressed=delta < -regression_margin))
    if not findings:
        raise EvaluationError("retention report requires at least one family")
    worst = min(findings, key=lambda finding: finding.delta)
    return {
        "families": [finding.__dict__ for finding in findings],
        "worst_family": worst.family,
        "worst_delta": worst.delta,
        "regressed_families": [finding.family for finding in findings if finding.regressed],
        "mean_delta": sum(finding.delta for finding in findings) / len(findings),
        "regression_margin": regression_margin,
    }


def brier_score(outcomes: Sequence[RawOutcome]) -> dict[str, Any] | None:
    """Brier score over declared confidences; None when confidence is absent."""
    scored = [outcome for outcome in outcomes if outcome.confidence is not None]
    if not scored:
        return None
    total = sum((float(outcome.confidence) - (1.0 if outcome.complete_answer_correct() else 0.0)) ** 2
                for outcome in scored)
    return {"brier": total / len(scored), "count": len(scored)}


# --- formation gates and clustered uncertainty (B2.1) ------------------------

def sustained_gate(evaluations: Sequence[tuple[int, float]], *, threshold: float,
                   consecutive: int = 3) -> dict[str, Any]:
    """Preregistered sustained-threshold semantics over an evaluation series.

    ``evaluations`` are ``(update, score)`` pairs in evaluation order. The
    gate counts only as the FIRST of ``consecutive`` evaluations at or above
    the bar: ``onset_update`` is the first evaluation of the streak and
    ``confirmed_update`` the one where the streak reaches the required
    length. Peak or single-evaluation claims are structurally excluded —
    ``max_score`` is recorded for auditing only and is never a claim.
    """
    if consecutive < 1:
        raise EvaluationError("consecutive must be at least 1")
    if not evaluations:
        raise EvaluationError("sustained_gate requires at least one evaluation")
    streak = 0
    onset: int | None = None
    confirmed: int | None = None
    for update, score in evaluations:
        if score >= threshold:
            streak += 1
            if streak == 1:
                onset = int(update)
            if streak >= consecutive and confirmed is None:
                confirmed = int(update)
                break
        else:
            streak = 0
            onset = None
    return {
        "threshold": threshold,
        "consecutive": consecutive,
        "onset_update": onset,
        "confirmed_update": confirmed,
        "sustained_confirmed": confirmed is not None,
        "formation_auc": sum(score for _, score in evaluations) / len(evaluations),
        "max_score_for_audit_only": max(score for _, score in evaluations),
        "peak_claims_forbidden": True,
    }


def clustered_bootstrap_delta(reference_outcomes: Sequence[RawOutcome],
                              candidate_outcomes: Sequence[RawOutcome], *,
                              iterations: int = 1000, seed: int = 0,
                              confidence_level: float = 0.95) -> dict[str, Any]:
    """Bootstrap the paired complete-answer delta by semantic world cluster.

    Resamples clusters (``task_semantic_id``), never individual rows, so all
    evaluations of one world move together (W08: correlated rows are not
    independent evidence). Reference and candidate outcomes must pair on
    task ids; unmatched tasks reject.
    """
    import random as _random

    if iterations < 1:
        raise EvaluationError("iterations must be at least 1")
    if not 0.0 < confidence_level < 1.0:
        raise EvaluationError("confidence_level must lie in (0, 1)")
    ref: dict[str, list[int]] = {}
    cand: dict[str, list[int]] = {}
    for outcome in reference_outcomes:
        ref.setdefault(outcome.task_semantic_id, []).append(
            1 if outcome.complete_answer_correct() else 0)
    for outcome in candidate_outcomes:
        cand.setdefault(outcome.task_semantic_id, []).append(
            1 if outcome.complete_answer_correct() else 0)
    unmatched = sorted(set(ref) ^ set(cand))
    if unmatched:
        raise EvaluationError(
            f"reference/candidate tasks do not pair: {unmatched[:4]}")
    cluster_deltas: dict[str, float] = {}
    for task in ref:
        r = sum(ref[task]) / len(ref[task])
        c = sum(cand[task]) / len(cand[task])
        cluster_deltas[task] = c - r
    tasks = sorted(cluster_deltas)
    rng = _random.Random(seed)
    sampled: list[float] = []
    for _ in range(iterations):
        draw = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        sampled.append(sum(cluster_deltas[task] for task in draw) / len(tasks))
    sampled.sort()
    alpha = (1.0 - confidence_level) / 2.0
    low = sampled[max(0, int(alpha * iterations))]
    high = sampled[min(iterations - 1, int((1 - alpha) * iterations))]
    point = sum(cluster_deltas.values()) / len(tasks)
    return {
        "delta": point,
        "ci_low": low,
        "ci_high": high,
        "confidence_level": confidence_level,
        "clusters": len(tasks),
        "iterations": iterations,
        "unit": "semantic world cluster",
        "ci_includes_zero": low <= 0.0 <= high,
    }


@dataclass(frozen=True)
class PromotionConfig:
    """Configured promotion margins; thresholds are frozen before evaluation."""

    primary_margin: float = 0.05
    regression_margin: float = 0.02
    min_pairs_for_goal_check: int = 8
    both_correct_floor: float = 0.5
    min_clusters: int = 2

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def decide_promotion(*, reference_metrics: Mapping[str, Any],
                     candidate_metrics: Mapping[str, Any],
                     reference_family: Mapping[str, dict[str, float]],
                     candidate_family: Mapping[str, dict[str, float]],
                     paired: Mapping[str, Any] | None,
                     config: PromotionConfig,
                     evidence_complete: bool,
                     sealed_fresh: bool | None = None,
                     clustered_uncertainty: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Frozen-margin promotion decision; insufficient evidence defaults to
    no_promotion. A family regression can never be hidden by the aggregate.
    Underpowered evidence (too few clusters, or a confidence interval that
    straddles zero) can never produce a confident accept."""
    reasons: list[str] = []
    if not evidence_complete:
        reasons.append("evidence_incomplete")
    ref_rate = reference_metrics.get("complete_answer_rate", 0.0)
    cand_rate = candidate_metrics.get("complete_answer_rate", 0.0)
    if cand_rate - ref_rate < config.primary_margin:
        reasons.append(
            f"primary_margin_not_met: {cand_rate:.4f} - {ref_rate:.4f} < "
            f"{config.primary_margin}")
    retention = retention_report(reference_family, candidate_family,
                                 regression_margin=config.regression_margin)
    if retention["regressed_families"]:
        reasons.append(
            f"family_regression: {retention['regressed_families']} "
            f"(worst {retention['worst_family']} delta {retention['worst_delta']:.4f})")
    if paired is not None and paired.get("pairs", 0) < config.min_pairs_for_goal_check:
        reasons.append("insufficient_pairs_for_goal_check")
    elif paired is not None and paired.get("both_correct_rate", 0.0) < config.both_correct_floor:
        reasons.append("both_correct_floor_not_met")
    if sealed_fresh is False:
        reasons.append("sealed_pool_reuse_detected")
    if clustered_uncertainty is not None:
        if clustered_uncertainty.get("clusters", 0) < config.min_clusters:
            reasons.append(
                f"underpowered: {clustered_uncertainty['clusters']} clusters < "
                f"{config.min_clusters}")
        elif clustered_uncertainty.get("ci_includes_zero") \
                and cand_rate - ref_rate >= config.primary_margin:
            reasons.append("bootstrap_ci_includes_zero")
    decision = "accept" if not reasons else (
        "diagnose" if len(reasons) == 1 and reasons[0] == "insufficient_pairs_for_goal_check"
        else "no_promotion")
    return {
        "decision": decision,
        "reasons": reasons,
        "candidate_complete_answer_rate": cand_rate,
        "reference_complete_answer_rate": ref_rate,
        "retention": retention,
        "thresholds_frozen": True,
    }
