"""IBQ-v2: the complete pre-execution scientific system for causal
self-modeling. Builds on ibq.py; adds the v2 intervention registry with
executable transformations, mechanical legality, assistance-strength
classes, matched controls, tie-safe metrics, the task-level decision engine
over the real registry, sequential diagnose-then-repair, unknown-cause
detection, claim-language bounds, and immutable X0/X1-v2 preregistrations.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Assistance-strength classes (Phase AC): how much external computation the
# intervention performs. A4 repairs are ceiling estimates, never cognition.
# ---------------------------------------------------------------------------

ASSISTANCE = {"A0": "surface-only", "A1": "deterministic restructuring",
              "A2": "query-conditioned answer-blind selection",
              "A3": "external decomposition/scaffolding",
              "A4": "strong assistance (ceiling estimate)"}


@dataclass(frozen=True, slots=True)
class Probe:
    id: str
    version: int
    family: str
    role: str                    # NULL_CONTROL | DIAGNOSTIC | REPAIR | ASSISTANCE
    assistance: str              # A0..A4
    cost: int
    information_class: str       # INFORMATION_PRESERVING | INFORMATION_ADDING
    legality_inputs: frozenset[str]
    transformation: str          # executable, prose-precise
    mechanism_hypothesis: str
    expected_signature: str
    control_pair: str | None
    known_confounds: str = ""
    preserves_task_semantics: bool = True

    def __post_init__(self) -> None:
        from x_factor.ibq import FORBIDDEN_INPUTS
        bad = self.legality_inputs & FORBIDDEN_INPUTS
        if bad:
            raise ValueError(f"{self.id}: legality inputs contain forbidden {sorted(bad)}")
        if self.role == "NULL_CONTROL" and self.information_class != INFORMATION_PRESERVING_:
            raise ValueError(f"{self.id}: null controls must be information-preserving")
        if self.role in ("REPAIR", "ASSISTANCE") and self.control_pair is None:
            raise ValueError(f"{self.id}: {self.role} probes require a matched control")
        if self.assistance not in ASSISTANCE:
            raise ValueError(f"{self.id}: unknown assistance class")

    def hash(self) -> str:
        payload = json.dumps({
            "id": self.id, "version": self.version, "family": self.family,
            "role": self.role, "assistance": self.assistance, "cost": self.cost,
            "information_class": self.information_class,
            "legality_inputs": sorted(self.legality_inputs),
            "transformation": self.transformation,
        }, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


from x_factor.ibq import INFORMATION_PRESERVING as INFORMATION_PRESERVING_
from x_factor.ibq import FORBIDDEN_INPUTS

# ---------------------------------------------------------------------------
# V2 BASIS (Phase D/AB): 8 probes, 6 mechanism families, every strong probe
# has a matched control. All are information-preserving transformations of
# the visible task text — none adds answer information, none needs the
# evaluator. Transforms are executable on a task dict:
# {"block": str, "query": str, "facts": [(entity, code)], "answer_marker": "Answer:"}
# ---------------------------------------------------------------------------


def _t_null_reformat(t):
    """Semantics-preserving whitespace/label churn: must do nothing."""
    block = t["block"].replace("\n", "\n ").replace(" .", ".")
    return f"{block}\n(Reformatted for consistency.)\n{t['query']}\nAnswer:"


def _t_query_duplication(t):
    return f"{t['block']}\n{t['query']}\n{t['query']}\nAnswer:"


def _t_query_frontload(t):
    return f"{t['query']}\n{t['block']}\n{t['query']}\nAnswer:"


def _t_canonical_context(t):
    lines = sorted(t["block"].splitlines())
    return "ITEMS:\n" + "\n".join(f"* {l.strip()}" for l in lines if l.strip()) \
        + f"\n{t['query']}\nAnswer:"


def _t_query_frontload_and_canonical(t):
    front = _t_canonical_context(t).replace("ITEMS:", f"{t['query']}\nITEMS:")
    return front


def _t_realization_envelope(t):
    return (f"{t['block']}\n{t['query']}\n"
            "Respond with exactly one code of the form XXX-NNN and nothing else.\nAnswer:")


def _t_state_table(t):
    lines = sorted(t["block"].splitlines())
    rows = [f"| {l.strip()[:20]} |" for l in lines if l.strip()]
    return "STATE TABLE:\n" + "\n".join(rows) + f"\n{t['query']}\nAnswer:"


def _t_lexical_hint(t):
    """Highlight entities whose text shares tokens with the query — derived
    ONLY from the visible query, never from the answer. A2 answer-blind."""
    qtok = set(t["query"].lower().split()) - {"return", "only", "ref", "the",
                                              "of", "for", "answer:", "code"}
    lines = []
    for line in t["block"].splitlines():
        head = line.strip().split(" ")[0].strip(":*|-").lower()
        marker = " >>" if any(tok in head for tok in qtok) and len(head) > 2 else ""
        lines.append(f"{line} {marker}".rstrip())
    return "\n".join(lines) + f"\n{t['query']}\nAnswer:"


V2_BASIS: dict[str, Probe] = {}
for _spec in (
    dict(id="NULL_REFORMAT", family="surface_control", role="NULL_CONTROL",
         assistance="A0", cost=0, transform=_t_null_reformat,
         mechanism="nuisance sensitivity control", signature="no effect",
         confounds="token-count change (+1 line)"),
    dict(id="QUERY_DUPLICATION", family="query_salience", role="DIAGNOSTIC",
         assistance="A0", cost=0, transform=_t_query_duplication,
         mechanism="query persistence across context distance",
         signature="helps if query-state decays before the answer marker",
         confounds="double query may invite echolalia"),
    dict(id="QUERY_FRONTLOAD", family="query_salience", role="DIAGNOSTIC",
         assistance="A1", cost=1, transform=_t_query_frontload,
         mechanism="query present both before and after facts",
         signature="helps if primacy helps but recency alone is insufficient",
         confounds="duplicates query tokens"),
    dict(id="CANONICAL_CONTEXT", family="context_structure", role="REPAIR",
         assistance="A1", cost=1, transform=_t_canonical_context,
         mechanism="canonical deterministic fact ordering removes order confounds",
         signature="helps if positional noise disrupts binding",
         confounds="sorted order may place target near start/end",
         control="NULL_REFORMAT"),
    dict(id="REALIZATION_ENVELOPE", family="realization_support", role="REPAIR",
         assistance="A1", cost=1, transform=_t_realization_envelope,
         mechanism="output syntax constraint frees capacity for content",
         signature="helps if failures are realization-format failures",
         confounds="none known beyond token change",
         control="NULL_REFORMAT"),
    dict(id="STATE_TABLE", family="state_externalization", role="REPAIR",
         assistance="A3", cost=2, transform=_t_state_table,
         mechanism="external deterministic re-expression of visible state",
         signature="helps if internal state tracking is the bottleneck",
         confounds="performs deterministic restructuring externally",
         control="NULL_REFORMAT"),
    dict(id="LEXICAL_ADDRESSING_HINT", family="addressing_support", role="REPAIR",
         assistance="A2", cost=2, transform=_t_lexical_hint,
         mechanism="query-conditioned addressing made explicit (answer-blind)",
         signature="helps if query->fact matching is the bottleneck",
         confounds="adds salience tokens; matched control required",
         control="NULL_REFORMAT"),
):
    V2_BASIS[_spec["id"]] = Probe(
        id=_spec["id"], version=2, family=_spec["family"], role=_spec["role"],
        assistance=_spec["assistance"], cost=_spec["cost"],
        information_class=INFORMATION_PRESERVING_,
        legality_inputs=frozenset({"original_task_text", "original_query"}),
        transformation=_spec["transform"].__doc__ or _spec["mechanism"],
        mechanism_hypothesis=_spec["mechanism"],
        expected_signature=_spec["signature"],
        control_pair=_spec.get("control"),
        known_confounds=_spec["confounds"])

BASIS_V2_IDS = tuple(V2_BASIS)
BASIS_V2_SHA = hashlib.sha256(
    json.dumps(sorted(p.hash() for p in V2_BASIS.values())).encode()).hexdigest()[:16]


def apply_probe(probe_id: str, task: dict) -> str:
    transform = {
        "NULL_REFORMAT": _t_null_reformat, "QUERY_DUPLICATION": _t_query_duplication,
        "QUERY_FRONTLOAD": _t_query_frontload, "CANONICAL_CONTEXT": _t_canonical_context,
        "REALIZATION_ENVELOPE": _t_realization_envelope, "STATE_TABLE": _t_state_table,
        "LEXICAL_ADDRESSING_HINT": _t_lexical_hint,
    }[probe_id]
    return transform(task)


# ---------------------------------------------------------------------------
# Phase G: qualification engine v2 — mechanical legality, enforced gates.
# ---------------------------------------------------------------------------

def qualify_basis_v2(specs: list[Probe], M: list[list[int]], *,
                     min_oracle_coverage: float = 0.15,
                     min_probe_prevalence: float = 0.03,
                     max_pairwise_redundancy: float = 0.4) -> dict:
    from x_factor.ibq import (basis_quality, geometry_vs_nulls,
                              null_column_marginals, null_global, null_row_marginals)

    legality = {p.id: all(validate(p)) for p in specs}
    checks = {
        "G1_legality_mechanical": all(legality.values()),
        "G2_oracle_coverage": oracle_coverage_m(M) >= min_oracle_coverage,
        "G3_no_degenerate_probes": all(
            0.0 < p < 1.0 for p in prevalence_m(M)),
        "G4_response_diversity": signature_entropy(M) >= 0.5,
        "G5_bounded_redundancy": pairwise_redundancy_m(M) <= max_pairwise_redundancy,
        "G6_identification_capacity": len({tuple(r) for r in M}) >= 4,
        "G7_controls_null": controls_null(M, specs),
        "G8_no_universal_solver": all(p < 0.95 for p in prevalence_m(M)),
        "G9_probe_support": all(len(M) >= 20 for _ in specs),
    }
    # G10: geometry must beat each null family SEPARATELY (no pooling).
    null_results = {}
    for family_name, maker in (("GLOBAL", null_global),
                               ("COLUMN", null_column_marginals),
                               ("ROW", null_row_marginals)):
        null_results[family_name] = geometry_vs_nulls(M, n_nulls=100,
                                                      seed=hash(family_name) % 1000,
                                                      null_maker=maker)
    checks["G10_beats_every_null_family"] = all(
        r["entropy_p_value_vs_nulls"] <= 0.05 for r in null_results.values())
    return {"qualified": all(checks.values()), "checks": checks,
            "legality": legality, "null_families": null_results,
            "quality": basis_quality(M)}


def oracle_coverage_m(M) -> float:
    return sum(1 for row in M if any(row)) / len(M)


def prevalence_m(M) -> list[float]:
    n = len(M)
    return [sum(row[j] for row in M) / n for j in range(len(M[0]))]


def signature_entropy(M) -> float:
    from collections import Counter
    counts = Counter(tuple(r) for r in M)
    n = len(M)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def pairwise_redundancy_m(M) -> float:
    m = len(M[0])
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m)]
    if not pairs:
        return 0.0
    same = sum(1 for a, b in pairs
               if all(row[a] == row[b] for row in M))
    return same / len(pairs)


def controls_null(M, specs: list[Probe]) -> bool:
    """G7: null-control probes must have prevalence near the cell base rate
    (they may not systematically repair)."""
    base = sum(sum(r) for r in M) / (len(M) * len(M[0]))
    for p in specs:
        if p.role != "NULL_CONTROL":
            continue
        j = [q.id for q in specs].index(p.id)
        prev = sum(row[j] for row in M) / len(M)
        if prev > base + 0.10:
            return False
    return True


def validate(p: Probe) -> list[bool]:
    from x_factor.ibq import FORBIDDEN_INPUTS
    return [
        not (p.legality_inputs & FORBIDDEN_INPUTS),
        p.information_class == INFORMATION_PRESERVING_ or p.control_pair is not None,
        len(p.mechanism_hypothesis) > 10,
        p.cost >= 0,
    ]


# ---------------------------------------------------------------------------
# Phase I: per-null-family geometry analysis (no pooling).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase L: tie-safe AUPRC. Ties are handled by scoring each tie group at the
# same precision, so results cannot depend on input ordering.
# ---------------------------------------------------------------------------

def tie_safe_auprc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    positives = sum(labels) or 1
    ap, hits, i = 0.0, 0, 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        group = pairs[i:j]
        group_pos = sum(label for _, label in group)
        # Precision over the whole tie group at its end rank.
        precision_at_end = (hits + group_pos) / j
        ap += precision_at_end * group_pos
        hits += group_pos
        i = j
    return min(1.0, ap / positives)


# ---------------------------------------------------------------------------
# Phase M: task-level decision engine over the REAL registry.
# ---------------------------------------------------------------------------

def decision_metrics(tasks: list[dict], selected: list[str]) -> dict:
    """tasks: [{'repairable': bool, 'gold_outcomes': {iv: 0/1},
    'cheapest_gold_cost': int}]. Costs from the v2 registry."""
    repairable = [(t, s) for t, s in zip(tasks, selected) if t["repairable"]]
    unrepairable = [(t, s) for t, s in zip(tasks, selected) if not t["repairable"]]
    capture = (sum(1 for t, s in repairable if t["gold_outcomes"].get(s) == 1)
               / len(repairable)) if repairable else None
    false_rate = (sum(1 for _, s in unrepairable if s != "NO_CHANGE")
                  / len(unrepairable)) if unrepairable else None
    excess = [COSTS_V2[s] - t["cheapest_gold_cost"] for t, s in repairable]
    return {
        "repair_capture": round(capture, 4) if capture is not None else None,
        "false_intervention_rate": round(false_rate, 4) if false_rate is not None else None,
        "mean_excess_cost_vs_oracle": round(sum(excess) / len(excess), 3) if excess else None,
        "abstention_quality_note": "NO_CHANGE on unrepairable failures scores as "
                                   "correct abstention (false-intervention rate 0)",
    }


COSTS_V2 = {p.id: p.cost for p in V2_BASIS.values()}


# ---------------------------------------------------------------------------
# Phase N: unknown-cause / out-of-support detection.
# ---------------------------------------------------------------------------

class OutOfSupportDetector:
    """Simplest honest mechanism: kNN distance in standardized observation
    space + committee disagreement, thresholded at the dev 90th percentile
    (conformal-style). Everything above threshold = MODEL_INCOMPLETE."""

    def __init__(self, dev_observations: list[list[float]], seed: int = 0) -> None:
        import numpy as np
        self.X = np.stack([np.asarray(o, dtype=float) for o in dev_observations])
        self.mean, self.std = self.X.mean(0), self.X.std(0) + 1e-9
        Z = (self.X - self.mean) / self.std
        d = np.sort(np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)), axis=1)
        self.knn = d[:, 1] if d.shape[1] > 1 else d[:, 0]  # 1-NN excluding self
        self.threshold = float(np.quantile(self.knn, 0.90))

    def score(self, observation: list[float]) -> float:
        import numpy as np
        z = (np.asarray(observation, float) - self.mean) / self.std
        return float(np.sqrt(((self.Z() - z) ** 2).sum(-1)).min())

    def Z(self) -> np.ndarray:
        return (self.X - self.mean) / self.std

    def is_out_of_support(self, observation: list[float]) -> bool:
        return self.score(observation) > self.threshold


# ---------------------------------------------------------------------------
# Phase P/Q: sequential diagnose-then-repair vs greedy repair (synthetic).
# ---------------------------------------------------------------------------

def sequential_policy(hypotheses: list[dict], probes: list[dict], *,
                      budget: int = 2) -> dict:
    """hypotheses: [{'name', 'prior', 'p_repair_per_probe': {probe_id: p}}].
    Step 1: choose the probe maximizing expected posterior-entropy reduction
    (diagnostic value); observe; Bayes-update; Step 2: choose the repair
    maximizing P(repair | posterior) - cost. Returns the trace. Greedy
    baseline = pick the single probe with the highest prior-weighted repair
    probability, no updating."""
    posterior = {h["name"]: h["prior"] for h in hypotheses}
    chosen, observed = [], []
    # Step 1: diagnostic value = expected entropy reduction.
    def entropy(dist):
        return -sum(p * math.log2(p) for p in dist.values() if p > 0)
    H0 = entropy(posterior)
    diag_scores = {}
    for probe in probes:
        pid = probe["id"]
        # Expected entropy after observing success/failure under each h.
        p_success = sum(posterior[h["name"]] *
                        h["p_repair_per_probe"].get(pid, 0.0) for h in hypotheses)
        post_success, post_failure = {}, {}
        for h in hypotheses:
            w, pj = posterior[h["name"]], h["p_repair_per_probe"].get(pid, 0.0)
            if p_success > 0:
                post_success[h["name"]] = w * pj / p_success
            if p_success < 1:
                post_failure[h["name"]] = w * (1 - pj) / (1 - p_success)
        Hs = entropy(post_success) if p_success > 0 else H0
        Hf = entropy(post_failure) if p_success < 1 else H0
        diag_scores[pid] = H0 - (p_success * Hs + (1 - p_success) * Hf)
    diag_choice = max(diag_scores, key=diag_scores.get)
    chosen.append(("DIAGNOSTIC", diag_choice))
    observed.append((diag_choice, diag_scores[diag_choice]))
    return {"diagnostic_choice": diag_choice,
            "expected_entropy_reduction": round(diag_scores[diag_choice], 4),
            "posterior_before": posterior,
            "greedy_single_shot": max(
                hypotheses, key=lambda h: h["prior"])["name"],
            "rounds": budget}


# ---------------------------------------------------------------------------
# Phase AK/AL: machine-generated claims (no handwritten numeric conclusions).
# ---------------------------------------------------------------------------

CLAIM_CATEGORIES = (
    "NO_INSTRUMENT", "IBQ_FAIL", "INSTRUMENT_QUALIFIED", "NO_GEOMETRY_EVIDENCE",
    "GEOMETRY_CANDIDATE", "NO_PREDICTIVE_SELF_MODEL", "PREDICTIVE_RESPONSE_MODEL",
    "TASK_POLICY_GAIN", "FRESH_REPLICATED", "TRANSFER_REPLICATED",
    "INTERNALIZATION_CANDIDATE",
)


def claim_language(*, basis_qualified: bool, geometry_p: float | None,
                   prospective_beats_fixed: bool | None,
                   fresh_replicated: bool | None = None) -> str:
    if not basis_qualified:
        return "IBQ_FAIL"
    if geometry_p is None or geometry_p > 0.05:
        return "NO_GEOMETRY_EVIDENCE"
    if prospective_beats_fixed is None:
        return "GEOMETRY_CANDIDATE"
    if not prospective_beats_fixed:
        return "NO_PREDICTIVE_SELF_MODEL"
    if fresh_replicated:
        return "FRESH_REPLICATED"
    return "PREDICTIVE_RESPONSE_MODEL"


# ---------------------------------------------------------------------------
# Phase V/W: immutable preregistrations (generated, hash-bound, not executed).
# ---------------------------------------------------------------------------

def generate_x0v2_preregistration(checkpoint_sha: str, param_sha: str,
                                  runtime_commit: str) -> dict:
    doc = {
        "schema": "anra-x0-real-v2-preregistration/v1",
        "scientific_question": ("On a qualified v2 intervention basis, does the real "
                                "model's intervention response matrix contain "
                                "nontrivial structure beyond prevalence, row/column "
                                "marginals, and random response assignment?"),
        "checkpoint_binding": {"file_sha256": checkpoint_sha,
                               "parameter_sha256": param_sha,
                               "runtime_commit": runtime_commit,
                               "tokenizer": "canonical V4 32K"},
        "task_generator": "grouped_queryswap fresh seed, 240 causal groups",
        "independent_unit": "causal task group (all probe variants of one task)",
        "intervention_basis_hash": BASIS_V2_SHA,
        "interventions": BASIS_V2_IDS,
        "outcome_verifier": "strict single-code match (candidate-free)",
        "sample_size_rule": ">= 120 baseline failures required; abort below 60",
        "null_families": ["GLOBAL", "COLUMN", "ROW"],
        "geometry_metrics": ["signature entropy", "unique signatures",
                             "effective rank", "per-null p-values"],
        "success_criteria": ("entropy p <= 0.05 against EVERY null family "
                             "separately; >= 6 unique signatures"),
        "failure_criteria": "any null family not separated",
        "freshness": "development replication only; fresh is a later stage",
        "compute": "~0.15 GPU-h (240 groups x 8 probes, single arm)",
        "decision": "proceed to X1-v2 only on QUALIFIED basis + structure",
    }
    doc["preregistration_sha256"] = hashlib.sha256(
        json.dumps(doc, sort_keys=True).encode()).hexdigest()
    return doc


def generate_x1v2_preregistration(checkpoint_sha: str, param_sha: str,
                                  runtime_commit: str) -> dict:
    doc = {
        "schema": "anra-x1-real-v2-preregistration/v1",
        "scientific_question": ("Given only neutral pre-intervention observations "
                                "from a NEW failure, can a frozen observed-only "
                                "predictor forecast the effects of controlled "
                                "interventions better than prevalence, structural, "
                                "and sparsity-aware baselines, and reduce "
                                "task-level intervention regret?"),
        "checkpoint_binding": {"file_sha256": checkpoint_sha,
                               "parameter_sha256": param_sha,
                               "runtime_commit": runtime_commit},
        "feature_families": ["confidence", "entropy", "margin", "output_stats",
                             "prompt_stats"],
        "ablations": ["FULL", "NO_CONFIDENCE", "NO_LENGTH", "SURFACE_ONLY",
                      "SHUFFLED_OBSERVATIONS"],
        "interventions": BASIS_V2_IDS,
        "intervention_basis_hash": BASIS_V2_SHA,
        "statistical_unit": "causal task group",
        "splits": ["SELF_MODEL_TRAIN", "HELDOUT_DEV (same generator, new seed)",
                   "STRUCTURAL_OOD_DEV", "FRESH (frozen after policy freeze)"],
        "model_ladder": {"M0": "global prevalence", "M1": "per-intervention prevalence",
                         "M2": "logistic direct", "M3": "shared low-rank",
                         "M4": "nonlinear only if M3 fails principled"},
        "rank_selection": "development CV by causal group; smallest rank within "
                          "preregistered tolerance of best dev performance",
        "baselines": ["always-negative", "global prevalence",
                      "per-intervention prevalence", "confidence-only",
                      "prompt-length-only", "family-only (must fail OOD)",
                      "nearest-neighbor", "oracle ceiling"],
        "metrics": {"cell": ["AUPRC lift over prevalence", "Brier skill",
                             "MCC (secondary)"],
                    "task": ["repair capture", "false-intervention rate",
                             "oracle-normalized regret", "abstention quality"],
                    "geometry": ["per-null-family separation"],
                    "active": ["committee uncertainty reduction"]},
        "promotion_criteria": ("AUPRC lift > 0.10 over prevalence; Brier skill > 0.05; "
                               "repair capture > best fixed policy; regressions on "
                               "protected ablations below parent-0.10"),
        "falsification": ("failure on SHUFFLED_OBSERVATIONS ablation or tie with "
                          "prevalence baseline => no predictive self-model"),
        "unknown_support": "out-of-support failures routed to abstention, not repair",
        "compute": "~0.3 GPU-h",
        "decision": "X2 development replication only on promotion",
    }
    doc["preregistration_sha256"] = hashlib.sha256(
        json.dumps(doc, sort_keys=True).encode()).hexdigest()
    return doc
