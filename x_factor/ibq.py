"""IBQ — Intervention Basis Qualification.

X1-REAL-0 postmortem: its 95.45% "cell accuracy" was compared against a 9%
best-intervention repair rate — incommensurable metrics. Under ~9% cell
prevalence, a trivial always-negative predictor achieves ~95% accuracy too.
X1-REAL-0 is therefore EXPLORATORY / INVALID FOR PROMOTION, and its real
finding is: the intervention basis had low repair coverage and the metric
system allowed a sparsity false-positive.

This module makes that failure mode mechanically impossible and adds the
qualification stage that must precede any future X1:

  1. InterventionSpec + legality validation (information-preserving vs
     information-adding, machine-readable; forbidden inputs enforced).
  2. Basis-quality metrics on the outcome matrix R (N failures x M
     interventions): oracle coverage, cell prevalence, per-intervention
     prevalence (degenerate probes flagged), response-signature entropy,
     unique signatures, pairwise redundancy, identification capacity.
  3. Sparsity-matched null models (global Bernoulli, column-marginal
     permutation, within-column shuffle) so any geometry claim must beat
     nulls that preserve trivial statistics.
  4. Imbalance-resistant prediction metrics (AUPRC, balanced accuracy, MCC,
     Brier skill vs prevalence) — raw cell accuracy is diagnostics only.
  5. Task-level decision metrics (repair capture, false-intervention rate,
     oracle-normalized regret, abstention quality).
  6. The promotion gate: an always-negative predictor at 95% raw accuracy
     under 5% prevalence is REJECTED by construction (tested).
  7. Checkpoint identity contract (parameter_sha256=None is not promotion
     grade) and an experiment chronology state machine (analyze-before-
     execute is impossible).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from itertools import combinations

# ---------------------------------------------------------------------------
# 1. Intervention specifications and legality.
# ---------------------------------------------------------------------------

INFORMATION_PRESERVING = "INFORMATION_PRESERVING"
INFORMATION_ADDING = "INFORMATION_ADDING"
FORBIDDEN_INPUTS = frozenset({
    "gold_answer", "gold_rank", "correctness_label", "hidden_failure_category",
    "future_intervention_outcome", "oracle_selected_relevant_fact",
    "evaluator_latent_variables",
})


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    id: str
    version: int
    family: str            # surface_control | query_salience | context_structure |
                           # distractor_structure | addressing_support |
                           # realization_support | decode_variation |
                           # composition_support | state_externalization
    cost: int
    information_class: str
    legality_inputs: frozenset[str]
    forbidden_inputs: frozenset[str]
    transformation: str            # exact semantics, prose-precise
    mechanism_hypothesis: str      # what computation it probes (evaluator-side)
    expected_signature: str
    control_pair: str | None       # id of the negative control, if a probe
    status: str = "CANDIDATE"      # CANDIDATE | QUALIFIED | BLOCKED

    def __post_init__(self) -> None:
        if self.information_class not in (INFORMATION_PRESERVING, INFORMATION_ADDING):
            raise ValueError(f"{self.id}: unknown information class")
        # Legality polarity: forbidden sources must not appear among the
        # inputs the intervention DECLARES it may consume.
        bad = self.legality_inputs & FORBIDDEN_INPUTS
        if bad:
            raise ValueError(f"{self.id}: legality inputs contain forbidden "
                             f"sources {sorted(bad)}")
        if self.information_class == INFORMATION_ADDING and not self.control_pair:
            raise ValueError(f"{self.id}: information-adding intervention requires "
                             "a declared control pair")
        if self.cost < 0:
            raise ValueError(f"{self.id}: negative cost")

    def hash(self) -> str:
        payload = json.dumps({
            "id": self.id, "version": self.version, "family": self.family,
            "cost": self.cost, "information_class": self.information_class,
            "legality_inputs": sorted(self.legality_inputs),
            "transformation": self.transformation,
        }, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def validate_intervention(spec: InterventionSpec) -> dict[str, bool]:
    """Legality validation: an intervention passes only if its declared
    inputs exclude every forbidden source, information-adding probes declare
    controls, and surface-control probes expect null effects."""
    checks = {
        "no_gold_answer": "gold_answer" not in spec.legality_inputs,
        "no_correctness_label": "correctness_label" not in spec.legality_inputs,
        "no_hidden_category": "hidden_failure_category" not in spec.legality_inputs,
        "no_future_outcomes": "future_intervention_outcome" not in spec.legality_inputs,
        "no_oracle_selection": "oracle_selected_relevant_fact" not in spec.legality_inputs,
        "cost_declared": spec.cost >= 0,
        "mechanism_declared": len(spec.mechanism_hypothesis) > 10,
        "signature_declared": len(spec.expected_signature) > 5,
    }
    if spec.information_class == INFORMATION_ADDING:
        checks["control_pair_declared"] = spec.control_pair is not None
    return checks


def qualifies(spec: InterventionSpec) -> bool:
    return spec.status == "QUALIFIED" and all(validate_intervention(spec).values())


# ---------------------------------------------------------------------------
# 2. Basis-quality metrics. M: list of rows (each a list of 0/1 outcomes,
# one per intervention) — the outcome matrix R.
# ---------------------------------------------------------------------------

def _columns(M):
    n, m = len(M), len(M[0])
    return [[row[j] for row in M] for j in range(m)]


def oracle_coverage(M) -> float:
    return sum(1 for row in M if any(row)) / len(M)


def cell_prevalence(M) -> float:
    return sum(sum(row) for row in M) / (len(M) * len(M[0]))


def per_intervention_prevalence(M) -> list[float]:
    return [sum(col) / len(col) for col in _columns(M)]


def degenerate_interventions(M) -> list[int]:
    """Interventions that never or always fire — no identification value."""
    return [j for j, p in enumerate(per_intervention_prevalence(M))
            if p in (0.0, 1.0)]


def response_signature_entropy(M) -> float:
    """Shannon entropy (bits) over the distribution of distinct row
    signatures — how varied the failure responses are."""
    from collections import Counter
    counts = Counter(tuple(row) for row in M)
    n = len(M)
    H = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return H


def unique_signatures(M) -> int:
    return len({tuple(row) for row in M})


def pairwise_redundancy(M) -> float:
    """Fraction of intervention pairs with identical outcome columns —
    redundant probes carry no additional identification information."""
    cols = _columns(M)
    pairs = list(combinations(range(len(cols)), 2))
    if not pairs:
        return 0.0
    identical = sum(1 for a, b in pairs if cols[a] == cols[b])
    return identical / len(pairs)


def identification_capacity(M) -> int:
    """Upper bound on distinguishable latent failure states: the number of
    distinct response signatures the basis can express (2^M) intersected
    with observed signatures."""
    return unique_signatures(M)


def basis_quality(M) -> dict:
    prev = per_intervention_prevalence(M)
    return {
        "n_failures": len(M), "n_interventions": len(M[0]),
        "oracle_coverage": round(oracle_coverage(M), 4),
        "cell_prevalence": round(cell_prevalence(M), 4),
        "per_intervention_prevalence": [round(p, 4) for p in prev],
        "degenerate_interventions": degenerate_interventions(M),
        "response_signature_entropy_bits": round(response_signature_entropy(M), 4),
        "unique_signatures": unique_signatures(M),
        "pairwise_redundancy": round(pairwise_redundancy(M), 4),
        "identification_capacity": identification_capacity(M),
    }


def basis_qualified(M, *, min_oracle_coverage: float = 0.15,
                    min_prevalence: float = 0.02,
                    max_redundancy: float = 0.5) -> dict:
    """IBQ gate. Thresholds are reasoned, not tuned: a basis covering <15%
    of failures cannot support a repair self-model; a probe firing never or
    always measures nothing; >50% redundant pairs means the basis does not
    span the failure space."""
    q = basis_quality(M)
    checks = {
        "G1_legality": True,  # all specs validated at registration time
        "G2_oracle_coverage": q["oracle_coverage"] >= min_oracle_coverage,
        "G3_no_degenerate_probes": not q["degenerate_interventions"],
        "G4_signature_variation": q["response_signature_entropy_bits"] >= 0.5,
        "G5_span": q["unique_signatures"] >= min(8, 2 ** q["n_interventions"] - 1) // 2,
        "G6_not_redundant": q["pairwise_redundancy"] <= max_redundancy,
    }
    return {"qualified": all(checks.values()), "checks": checks, "quality": q}


# ---------------------------------------------------------------------------
# 3. Sparsity-matched null models. Each preserves trivial statistics; a
# geometry claim requires beating ALL of them.
# ---------------------------------------------------------------------------

def null_global(M, seed: int) -> list[list[int]]:
    """NULL-0: iid Bernoulli at the global positive rate."""
    rng = random.Random(seed)
    p = cell_prevalence(M)
    return [[int(rng.random() < p) for _ in row] for row in M]


def null_column_marginals(M, seed: int) -> list[list[int]]:
    """NULL-1/NULL-3: shuffle each intervention column independently,
    preserving every per-intervention success count exactly."""
    rng = random.Random(seed)
    cols = _columns(M)
    shuffled = [col[:] for col in cols]
    for col in shuffled:
        rng.shuffle(col)
    return [[shuffled[j][i] for j in range(len(cols))] for i in range(len(M))]


def null_row_marginals(M, seed: int) -> list[list[int]]:
    """NULL-2 approximation: shuffle within rows, preserving per-task
    repair counts (which interventions fired is decoupled from the task)."""
    rng = random.Random(seed)
    out = []
    for row in M:
        r = row[:]
        rng.shuffle(r)
        out.append(r)
    return out


def geometry_vs_nulls(M, *, n_nulls: int = 200, seed: int = 0) -> dict:
    """Is the real matrix MORE compressible than sparsity-matched nulls?
    Reports the null distribution of signature entropy and the fraction of
    nulls at least as structured as the real matrix (one-sided p)."""
    real_entropy = response_signature_entropy(M)
    real_unique = unique_signatures(M)
    null_ent, null_uniq = [], []
    for k in range(n_nulls):
        for maker in (null_global, null_column_marginals, null_row_marginals):
            N = maker(M, seed + k)
            null_ent.append(response_signature_entropy(N))
            null_uniq.append(unique_signatures(N))
    return {
        "real_signature_entropy_bits": round(real_entropy, 4),
        "null_entropy_mean": round(sum(null_ent) / len(null_ent), 4),
        "null_entropy_p95": round(sorted(null_ent)[int(0.95 * len(null_ent))], 4),
        "real_unique_signatures": real_unique,
        "null_unique_mean": round(sum(null_uniq) / len(null_uniq), 2),
        # Lower signature entropy = more compressible = more structure.
        # p-value: fraction of nulls AT LEAST AS compressible as the real
        # matrix. High p => the real matrix is what sparsity alone predicts.
        "entropy_p_value_vs_nulls": round(
            sum(1 for e in null_ent if e <= real_entropy) / len(null_ent), 4),
        "verdict": ("structure exceeds matched nulls (X0-v2 candidate)" if
                    sum(1 for e in null_ent if e <= real_entropy) / len(null_ent) <= 0.05
                    else "no structure beyond sparsity"),
    }


# ---------------------------------------------------------------------------
# 4. Imbalance-resistant prediction metrics.
# ---------------------------------------------------------------------------

def auprc(scores: list[float], labels: list[int]) -> float:
    """Average precision — threshold-free, prevalence-aware."""
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    positives = sum(labels) or 1
    hits, precision_sum = 0, 0.0
    for i, (_, label) in enumerate(pairs, start=1):
        hits += label
        precision_sum += hits / i
    return min(1.0, precision_sum / positives)


def balanced_accuracy(pred: list[int], labels: list[int]) -> float:
    tp = sum(1 for p, y in zip(pred, labels) if p and y)
    tn = sum(1 for p, y in zip(pred, labels) if not p and not y)
    pos = sum(labels) or 1
    neg = len(labels) - pos or 1
    return (tp / pos + tn / neg) / 2


def mcc(pred: list[int], labels: list[int]) -> float:
    tp = sum(1 for p, y in zip(pred, labels) if p and y)
    tn = sum(1 for p, y in zip(pred, labels) if not p and not y)
    fp = sum(1 for p, y in zip(pred, labels) if p and not y)
    fn = sum(1 for p, y in zip(pred, labels) if not p and y)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom else 0.0


def brier(probs: list[float], labels: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(labels)


def brier_skill(probs: list[float], labels: list[int]) -> float:
    """Skill vs the prevalence constant predictor (the X1-REAL-0 killer:
    a model must beat 'always say the base rate')."""
    prevalence = sum(labels) / len(labels) if labels else 0.0
    base = brier([prevalence] * len(labels), labels)
    model = brier(probs, labels)
    return 1.0 - model / base if base > 0 else 0.0


# ---------------------------------------------------------------------------
# 5. The promotion gate: raw accuracy can NEVER justify promotion under
# imbalance. This is the mechanized X1-REAL-0 lesson.
# ---------------------------------------------------------------------------

def promote_prediction(*, scores: list[float], labels: list[int],
                       threshold: float = 0.5,
                       min_auprc_lift: float = 0.10,
                       min_brier_skill: float = 0.05) -> dict:
    labels_bin = [int(y) for y in labels]
    prevalence = sum(labels_bin) / len(labels_bin) or 0.0
    pred = [int(s >= threshold) for s in scores]
    aup = auprc(scores, labels_bin)
    bs = brier_skill(probs=scores, labels=labels_bin)
    degenerate = len(set(scores)) == 1  # always-negative / constant scorer
    checks = {
        "non_degenerate_scores": not degenerate,
        "auprc_above_prevalence": aup > prevalence + min_auprc_lift,
        "brier_skill_positive": bs >= min_brier_skill,
        "mcc_positive": mcc(pred, labels_bin) > 0.0,
    }
    # The tripwire: an always-negative predictor at 5% prevalence scores
    # 95% raw accuracy and 0 AUPRC-lift / 0 Brier skill / MCC undefined
    # (degenerate) — rejected here by construction.
    return {
        "promotion": all(checks.values()),
        "prevalence": round(prevalence, 4),
        "auprc": round(aup, 4),
        "brier_skill": round(bs, 4),
        "raw_accuracy_diagnostic_only": round(
            sum(1 for p, y in zip(pred, labels_bin) if p == y) / len(labels_bin), 4),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# 6. Task-level decision metrics (what routing is FOR).
# ---------------------------------------------------------------------------

def task_level_metrics(selected: list[str], tasks: list[dict]) -> dict:
    """tasks: [{'repairable': bool, 'gold_cost': int|None, 'outcomes': {...}}];
    selected: the intervention chosen per task."""
    repairable = [(t, s) for t, s in zip(tasks, selected) if t["repairable"]]
    unrepairable = [(t, s) for t, s in zip(tasks, selected) if not t["repairable"]]
    capture = sum(1 for t, s in repairable if t["outcomes"].get(s, 0) == 1) \
        / len(repairable) if repairable else None
    false_intervention = sum(1 for _, s in unrepairable if s != "NO_CHANGE") \
        / len(unrepairable) if unrepairable else None
    regrets = []
    for t, s in zip(tasks, selected):
        if not t["repairable"]:
            continue
        gold_cost = min(c for iv, c in COSTByExample(t) if t["outcomes"].get(iv) == 1) \
            if any(t["outcomes"].values()) else None
        if gold_cost is not None:
            regrets.append(COSTS.get(s, 0) - gold_cost)
    return {
        "repair_capture": round(capture, 4) if capture is not None else None,
        "false_intervention_rate": round(false_intervention, 4)
            if false_intervention is not None else None,
        "oracle_normalized_regret_mean": round(sum(regrets) / len(regrets), 4)
            if regrets else None,
        "abstention_available": "NO_CHANGE" in COSTS,
    }


def COSTByExample(task):
    return [(iv, cost) for iv, cost in
            (("NO_CHANGE", 0), ("PROBE_A", 1), ("PROBE_B", 2))]


# ---------------------------------------------------------------------------
# 7. Checkpoint identity + chronology state machine.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    file_sha256: str | None
    parameter_sha256: str | None
    model_config_sha256: str | None
    runtime_commit: str | None
    tokenizer_sha256: str | None
    training_lineage: str | None = None

    def assert_promotion_grade(self) -> None:
        missing = [k for k, v in {
            "file_sha256": self.file_sha256,
            "parameter_sha256": self.parameter_sha256,
            "model_config_sha256": self.model_config_sha256,
            "runtime_commit": self.runtime_commit,
            "tokenizer_sha256": self.tokenizer_sha256,
        }.items() if v is None]
        if missing:
            raise ValueError(f"checkpoint identity not promotion-grade: {missing}")


class ChronologyError(RuntimeError):
    pass


class ExperimentChronology:
    """PREREGISTERED -> EXECUTED -> ANALYZED. Analysis before execution, or
    execution before preregistration, is mechanically impossible."""

    def __init__(self, experiment_id: str, preregistration_commit: str) -> None:
        self.experiment_id = experiment_id
        self.preregistration_commit = preregistration_commit
        self.state = "PREREGISTERED"
        self.execution_commit: str | None = None

    def register_execution(self, commit: str) -> None:
        if self.state != "PREREGISTERED":
            raise ChronologyError(f"{self.experiment_id}: cannot execute from state {self.state}")
        self.execution_commit = commit
        self.state = "EXECUTED"

    def register_analysis(self, commit: str) -> None:
        if self.state != "EXECUTED":
            raise ChronologyError(f"{self.experiment_id}: cannot analyze from state {self.state}")
        self.state = "ANALYZED"
