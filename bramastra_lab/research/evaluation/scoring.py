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
import math
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
    case_id: str = ""         # explicit evaluation case id, separate from clusters

    def __post_init__(self) -> None:
        for name in ("outcome_id", "pool", "split", "family", "task_semantic_id", "label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise EvaluationError(f"{name} must be a nonempty string")
        if not isinstance(self.case_id, str) or not self.case_id:
            raise EvaluationError(
                "case_id must be a nonempty string identifying the exact evaluation "
                "case, separate from the semantic world/cluster identity")
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


def _pair_outcomes_by_case(reference_outcomes: Sequence[RawOutcome],
                           candidate_outcomes: Sequence[RawOutcome]) -> dict[str, tuple[RawOutcome, RawOutcome]]:
    """Pair reference/candidate outcomes on exact case identity (B2.2 R6).

    A pair requires the same ``task_semantic_id`` AND ``case_id``, with equal
    label, family, split and cost. Duplicates on either side and unmatched
    or mismatched cases reject: different cases inside one world can never
    produce a delta.
    """
    def index(outcomes: Sequence[RawOutcome], side: str) -> dict[tuple[str, str], RawOutcome]:
        indexed: dict[tuple[str, str], RawOutcome] = {}
        for outcome in outcomes:
            key = (outcome.task_semantic_id, outcome.case_id)
            if key in indexed:
                raise EvaluationError(
                    f"duplicate {side} outcome for case {key[1]!r} in world {key[0]!r}")
            indexed[key] = outcome
        return indexed

    ref_index = index(reference_outcomes, "reference")
    cand_index = index(candidate_outcomes, "candidate")
    unmatched = sorted(set(ref_index) ^ set(cand_index))
    if unmatched:
        raise EvaluationError(
            f"reference/candidate cases do not pair: {unmatched[:4]}")
    pairs: dict[str, tuple[RawOutcome, RawOutcome]] = {}
    for (task, case), ref_outcome in ref_index.items():
        cand_outcome = cand_index[(task, case)]
        mismatches = [name for name, a, b in (
            ("label", ref_outcome.label, cand_outcome.label),
            ("family", ref_outcome.family, cand_outcome.family),
            ("split", ref_outcome.split, cand_outcome.split),
            ("cost", ref_outcome.cost, cand_outcome.cost)) if a != b]
        if mismatches:
            raise EvaluationError(
                f"paired case {case!r} in world {task!r} differs on {mismatches}")
        pairs[f"{task}\x00{case}"] = (ref_outcome, cand_outcome)
    return pairs


def clustered_bootstrap_delta(reference_outcomes: Sequence[RawOutcome],
                              candidate_outcomes: Sequence[RawOutcome], *,
                              iterations: int = 1000, seed: int = 0,
                              confidence_level: float = 0.95) -> dict[str, Any]:
    """Bootstrap the paired complete-answer delta by semantic world cluster.

    Outcomes first pair on exact case identity (same task, same case id,
    equal label/family/split/cost — duplicates and mismatches reject), then
    clusters (``task_semantic_id``) are resampled, never individual rows, so
    all evaluations of one world move together (W08).
    """
    import random as _random

    if iterations < 1:
        raise EvaluationError("iterations must be at least 1")
    if not 0.0 < confidence_level < 1.0:
        raise EvaluationError("confidence_level must lie in (0, 1)")
    pairs = _pair_outcomes_by_case(reference_outcomes, candidate_outcomes)
    cluster_deltas: dict[str, list[float]] = {}
    for key, (ref_outcome, cand_outcome) in pairs.items():
        task = key.split("\x00")[0]
        delta = (1 if cand_outcome.complete_answer_correct() else 0) \
            - (1 if ref_outcome.complete_answer_correct() else 0)
        cluster_deltas.setdefault(task, []).append(float(delta))
    cluster_means = {task: sum(values) / len(values)
                     for task, values in cluster_deltas.items()}
    tasks = sorted(cluster_means)
    rng = _random.Random(seed)
    sampled: list[float] = []
    for _ in range(iterations):
        draw = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        sampled.append(sum(cluster_means[task] for task in draw) / len(tasks))
    sampled.sort()
    alpha = (1.0 - confidence_level) / 2.0
    low = sampled[max(0, int(alpha * iterations))]
    high = sampled[min(iterations - 1, int((1 - alpha) * iterations))]
    point = sum(cluster_means.values()) / len(tasks)
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


@dataclass(frozen=True)
class EvaluationProtocol:
    """The declared protocol an evaluation runs under (B2.2 R6).

    ``paired_goal`` makes pair receipts mandatory for promotion; protocols
    without paired goals record that pair criteria are explicitly not
    applicable instead of silently bypassing the gate.
    ``requires_uncertainty`` makes a clustered-uncertainty receipt mandatory.
    """

    protocol_id: str
    paired_goal: bool = False
    requires_uncertainty: bool = False
    max_new_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_id, str) or not self.protocol_id.strip():
            raise EvaluationError("protocol_id must be a nonempty string")
        if self.max_new_tokens is not None \
                and (not isinstance(self.max_new_tokens, int)
                     or isinstance(self.max_new_tokens, bool) or self.max_new_tokens <= 0):
            raise EvaluationError("max_new_tokens must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_id": self.protocol_id, "paired_goal": self.paired_goal,
                "requires_uncertainty": self.requires_uncertainty,
                "max_new_tokens": self.max_new_tokens}


@dataclass(frozen=True)
class EvidenceBundle:
    """One validated evidence bundle binding a promotion claim together."""

    parent_identity: str
    child_identity: str
    protocol: EvaluationProtocol
    pool: str
    data_identity: str
    reference_metrics: Mapping[str, Any]
    candidate_metrics: Mapping[str, Any]
    reference_family: Mapping[str, dict[str, float]]
    candidate_family: Mapping[str, dict[str, float]]
    paired: Mapping[str, Any] | None = None
    clustered_uncertainty: Mapping[str, Any] | None = None
    sealed_fresh: bool | None = None
    mechanism_cluster_evidence: bool = False

    def __post_init__(self) -> None:
        for name in ("parent_identity", "child_identity", "pool", "data_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EvaluationError(f"evidence bundle {name} must be a nonempty string")
        if self.parent_identity == self.child_identity:
            raise EvaluationError("promotion parent and child must differ")
        if not isinstance(self.protocol, EvaluationProtocol):
            raise EvaluationError("evidence bundle requires an EvaluationProtocol")
        if not isinstance(self.reference_metrics, Mapping) \
                or not isinstance(self.candidate_metrics, Mapping):
            raise EvaluationError("evidence bundle requires metric mappings")
        for side, metrics in (("reference", self.reference_metrics),
                              ("candidate", self.candidate_metrics)):
            _validate_metric_mapping(metrics, f"{side} metrics")
        for side, families in (("reference", self.reference_family),
                               ("candidate", self.candidate_family)):
            _validate_family_metrics(families, f"{side} family metrics")
        if self.paired is not None:
            _validate_paired_metrics(self.paired)
        if self.clustered_uncertainty is not None:
            _validate_clustered_uncertainty(self.clustered_uncertainty)


def _require_finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise EvaluationError(f"{where} must be a finite number")
    return float(value)


def _validate_metric_mapping(metrics: Mapping[str, Any], where: str) -> None:
    rate = metrics.get("complete_answer_rate")
    if rate is None:
        raise EvaluationError(f"{where} must report complete_answer_rate")
    checked = _require_finite_number(rate, f"{where}.complete_answer_rate")
    if not 0.0 <= checked <= 1.0:
        raise EvaluationError(f"{where}.complete_answer_rate must lie in [0, 1]")


def _validate_family_metrics(families: Mapping[str, Any], where: str) -> None:
    if not isinstance(families, Mapping) or not families:
        raise EvaluationError(f"{where} must be a nonempty mapping")
    for family, entry in families.items():
        if not isinstance(family, str) or not family:
            raise EvaluationError(f"{where} family names must be nonempty strings")
        if not isinstance(entry, Mapping):
            raise EvaluationError(f"{where}.{family} must be an object")
        rate = entry.get("complete_answer_rate")
        if rate is None:
            raise EvaluationError(f"{where}.{family} must report complete_answer_rate")
        checked = _require_finite_number(rate, f"{where}.{family}.complete_answer_rate")
        if not 0.0 <= checked <= 1.0:
            raise EvaluationError(
                f"{where}.{family}.complete_answer_rate must lie in [0, 1]")
        count = entry.get("count")
        if count is not None:
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise EvaluationError(
                    f"{where}.{family}.count must be a nonnegative integer")


def _validate_paired_metrics(paired: Mapping[str, Any]) -> None:
    if not isinstance(paired, Mapping):
        raise EvaluationError("paired receipt must be an object")
    pairs = paired.get("pairs")
    if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 0:
        raise EvaluationError("paired receipt pairs must be a nonnegative integer")
    for key in ("both_correct_rate", "same_answer_rate", "primary_accuracy",
                "swapped_accuracy"):
        if key in paired:
            checked = _require_finite_number(paired[key], f"paired.{key}")
            if not 0.0 <= checked <= 1.0:
                raise EvaluationError(f"paired.{key} must lie in [0, 1]")
    if "goal_swap_gap" in paired:
        gap = _require_finite_number(paired["goal_swap_gap"], "paired.goal_swap_gap")
        if not -1.0 <= gap <= 1.0:
            raise EvaluationError("paired.goal_swap_gap must lie in [-1, 1]")


def _validate_clustered_uncertainty(uncertainty: Mapping[str, Any]) -> None:
    if not isinstance(uncertainty, Mapping):
        raise EvaluationError("clustered uncertainty must be an object")
    clusters = uncertainty.get("clusters")
    if not isinstance(clusters, int) or isinstance(clusters, bool) or clusters < 1:
        raise EvaluationError("clustered uncertainty clusters must be a positive integer")
    for key in ("delta", "ci_low", "ci_high"):
        if key not in uncertainty:
            raise EvaluationError(f"clustered uncertainty must report {key}")
        _require_finite_number(uncertainty[key], f"clustered uncertainty.{key}")
    if uncertainty["ci_low"] > uncertainty["ci_high"]:
        raise EvaluationError(
            "clustered uncertainty interval is inverted: ci_low exceeds ci_high")
    # The includes-zero flag is DERIVED from the bounds, never trusted from
    # the caller (B2.2 chief F3).
    uncertainty["ci_includes_zero"] = bool(
        uncertainty["ci_low"] <= 0.0 <= uncertainty["ci_high"])


def decide_promotion(evidence: EvidenceBundle, config: PromotionConfig) -> dict[str, Any]:
    """Protocol-bound promotion from a validated evidence bundle.

    Missing required receipts are insufficient evidence, never a bypass
    (B2.2 R6): a paired-goal protocol without a pair receipt and an
    uncertainty-requiring protocol without an uncertainty receipt both
    refuse. Pair criteria are explicitly not applicable under protocols
    without paired goals, and that is recorded. A family regression can
    never be hidden by the aggregate.
    """
    reasons: list[str] = []
    if evidence.protocol.paired_goal:
        if evidence.paired is None:
            reasons.append("missing_required_pair_receipt")
        elif evidence.paired.get("pairs", 0) < config.min_pairs_for_goal_check:
            reasons.append("insufficient_pairs_for_goal_check")
        elif evidence.paired.get("both_correct_rate", 0.0) < config.both_correct_floor:
            reasons.append("both_correct_floor_not_met")
    else:
        pair_not_applicable = True
    if evidence.protocol.requires_uncertainty and evidence.clustered_uncertainty is None:
        reasons.append("missing_required_uncertainty_receipt")
    if evidence.clustered_uncertainty is not None:
        min_clusters = config.min_clusters
        if evidence.clustered_uncertainty.get("clusters", 0) < min_clusters:
            reasons.append(
                f"underpowered: {evidence.clustered_uncertainty['clusters']} clusters < "
                f"{min_clusters}")
        ref_rate_early = evidence.reference_metrics.get("complete_answer_rate", 0.0)
        cand_rate_early = evidence.candidate_metrics.get("complete_answer_rate", 0.0)
        aggregate_gain = cand_rate_early - ref_rate_early
        ci_low = evidence.clustered_uncertainty["ci_low"]
        ci_high = evidence.clustered_uncertainty["ci_high"]
        # The interval must agree in direction with the aggregate gain: an
        # entirely negative interval with a positive aggregate gain is an
        # inconsistent comparison, never a favorable selection (F3). An
        # accept additionally requires a positive lower bound.
        if aggregate_gain >= config.primary_margin:
            if ci_high <= 0.0:
                reasons.append("aggregate_interval_inconsistent: positive gain with "
                               "an entirely negative interval")
            elif ci_low <= 0.0:
                reasons.append("bootstrap_ci_includes_zero")
    if evidence.sealed_fresh is False:
        reasons.append("sealed_pool_reuse_detected")
    reference_metrics = evidence.reference_metrics
    candidate_metrics = evidence.candidate_metrics
    ref_rate = reference_metrics.get("complete_answer_rate", 0.0)
    cand_rate = candidate_metrics.get("complete_answer_rate", 0.0)
    if cand_rate - ref_rate < config.primary_margin:
        reasons.append(
            f"primary_margin_not_met: {cand_rate:.4f} - {ref_rate:.4f} < "
            f"{config.primary_margin}")
    retention = retention_report(evidence.reference_family, evidence.candidate_family,
                                 regression_margin=config.regression_margin)
    if retention["regressed_families"]:
        reasons.append(
            f"family_regression: {retention['regressed_families']} "
            f"(worst {retention['worst_family']} delta {retention['worst_delta']:.4f})")
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
        "protocol": evidence.protocol.to_dict(),
        "pair_criteria": ("applied" if evidence.protocol.paired_goal
                          else "not_applicable_for_this_protocol"),
        "mechanism_cluster_evidence": evidence.mechanism_cluster_evidence,
        "pool": evidence.pool,
        "parent_identity": evidence.parent_identity,
        "child_identity": evidence.child_identity,
        "data_identity": evidence.data_identity,
    }
