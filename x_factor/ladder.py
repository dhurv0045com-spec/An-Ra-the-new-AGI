"""The preregistered X-ladder: causal cognitive self-modeling experiments.

Every rung states its objective, dangerous assumption, controls, baselines,
metric, promotion criterion, falsification criterion, freshness rule,
compute estimate, and the decision it produces. Rungs execute in order;
a rung whose promotion fails sends the program to its falsification branch
rather than to the next rung.
"""

from __future__ import annotations

LADDER_SCHEMA = "anra-x-factor-ladder/v1"


def build_ladder() -> dict[str, object]:
    rungs = [
        {
            "id": "X0", "objective": "Do intervention outcomes expose stable low-rank latent structure at all?",
            "assumption": "the synthetic latent-factor physics is a fair template for real cognition",
            "implementation": "outcome_matrix over deterministic world.py physics; rank/spectrum analysis; learned fingerprint must beat family shortcut CROSS-FAMILY on synthetic data",
            "controls": ["family shortcut (must fail cross-family)", "NO_CHANGE floor", "label-shuffled control (must fail)"],
            "baselines": ["ALWAYS_*", "RANDOM", "FAMILY_SHORTCUT"],
            "metric": "cross-family top-1 repair accuracy; outcome-matrix rank",
            "promotion": "learned cross-family accuracy > best fixed policy AND > family shortcut out-of-family, both on synthetic + later real fixtures",
            "falsification": "outcome matrix full-rank / no factor structure => fingerprints are noise; abandon representation",
            "freshness": "synthetic proof may run anytime; real-model X0 requires a fresh fixture drawn after policy freeze",
            "compute": "synthetic: CPU minutes; real: 1 P35 arm x 200 failures x 5 interventions ~ 0.1 GPU-h",
            "decision": "proceed to X1 or abandon the representation",
        },
        {
            "id": "X1", "objective": "Can an observed-only learner predict intervention outcomes on held-out tasks?",
            "assumption": "observed gap signals carry enough signal without hidden labels",
            "implementation": "train_fingerprint on development tasks; evaluate on held-out same-family tasks",
            "controls": ["leakage audit on every feature record", "label-shuffled learner"],
            "baselines": ["ALWAYS_*", "RANDOM", "CONFIDENCE_ONLY", "FAMILY_SHORTCUT"],
            "metric": "top-1, regret vs oracle, pairwise ranking accuracy, Brier",
            "promotion": "beats every fixed policy and FAMILY_SHORTCUT on held-out data, p < 0.01 paired",
            "falsification": "no better than ALWAYS_FULL_REPLAY => observed state is uninformative",
            "freshness": "held-out tasks drawn after learner freeze; no peeking",
            "compute": "pure CPU, minutes: reuse one P35 arm's traces over 200 failures x 5 interventions",
            "decision": "does observed failure state carry causal information?",
        },
        {
            "id": "X2", "objective": "Does the prediction generalize to FRESH task instances (new seeds, same generators)?",
            "assumption": "X1 was not fixture memorization",
            "implementation": "frozen learner from X1 evaluated on a fresh fixture generated after freeze; identity committed before evaluation",
            "controls": ["fixture identity committed pre-evaluation", "shuffled-label control"],
            "baselines": ["fixed policies", "FAMILY_SHORTCUT", "oracle ceiling"],
            "metric": "top-1, regret, calibration (Brier), effect-sign accuracy",
            "promotion": "direction and magnitude of X1 replicate within CI",
            "falsification": "accuracy collapses to fixed-policy level on fresh data",
            "freshness": "absolute: fresh split sealed until learner is frozen",
            "compute": "0.05 GPU-h: one P35 arm, 200 failures x 5 interventions, teacher-forced",
            "decision": "is X1 real or leakage?",
        },
        {
            "id": "X3", "objective": "Cross-FAMILY generalization: train on some families, evaluate on unseen families.",
            "assumption": "latent factors, not surface templates, carry the causal signal",
            "implementation": "leave-one-family-out training/evaluation over all families",
            "controls": ["FAMILY_SHORTCUT MUST FAIL here (structural negative control)", "format-shuffled control"],
            "baselines": ["fixed policies", "in-family learned (upper reference)"],
            "metric": "cross-family top-1 + regret; gap to in-family",
            "promotion": "cross-family accuracy > best fixed policy on every held-out family",
            "falsification": "family shortcut ties the learner cross-family => we learned templates, not cognition",
            "freshness": "held-out families never touched during training",
            "compute": "CPU only: leave-one-family-out retraining of the logistic heads, minutes",
            "decision": "is the learned structure causal or cosmetic?",
        },
        {
            "id": "X4", "objective": "Does the same cognitive representation transfer across checkpoints?",
            "assumption": "failure geometry is a property of the task, not the weights",
            "implementation": "learner trained on checkpoint A outcomes, evaluated on checkpoint B failures (core-exp precedent: v7 policy transferred)",
            "controls": ["checkpoint-shuffled control"],
            "baselines": ["fixed policies on B", "B-native learner (upper ref)"],
            "metric": "top-1 delta vs fixed policies on B",
            "promotion": "transferred learner beats fixed policies on B",
            "falsification": "transfer fails => self-model is checkpoint-local",
            "freshness": "B outcomes unseen until transfer evaluation",
            "compute": "0.1 GPU-h: outcome traces on checkpoint B for 200 failures x 5 interventions",
            "decision": "is the self-model portable infrastructure?",
        },
        {
            "id": "X5", "objective": "Can the learner beat STRONG structural heuristics with cost pressure?",
            "assumption": "cost-adjusted decisions do not collapse to NO_CHANGE",
            "implementation": "sweep cost_weight; compare against hand-routed and confidence-only at equal cost budgets",
            "controls": ["cost-matched comparison", "NO_CHANGE floor"],
            "baselines": ["HAND_ROUTED", "CONFIDENCE_ONLY", "cost-matched ALWAYS_*"],
            "metric": "cost-adjusted score; repair efficiency per unit cost",
            "promotion": "higher cost-adjusted score than every structural heuristic",
            "falsification": "no cost regime where the learner wins",
            "freshness": "same fixture discipline as X2",
            "compute": "CPU only: leave-one-family-out retraining of the logistic heads, minutes",
            "decision": "is the self-model practically useful, not just statistically significant?",
        },
        {
            "id": "X6", "objective": "Can successful external repairs become NATIVE core capability (internalization)?",
            "assumption": "repair-conditioned examples teach the underlying computation, not the crutch",
            "implementation": "distill verified repair trajectories into SFT groups where the intervention's context change is REMOVED but the answer target stays; strict raw-core evaluation after",
            "controls": ["no-intervention control child", "rehearsal >= 50%", "parent-relative retention floors"],
            "baselines": ["pre-internalization core", "SFT on raw failures only"],
            "metric": "raw-core accuracy on the bottleneck primitive, retention vector",
            "promotion": "raw-core gain on target primitive with all protected families within parent-0.10",
            "falsification": "gain exists only with the intervention present => dependence, not internalization",
            "freshness": "sealed evaluation suite frozen before internalization training",
            "compute": "~50 updates local GPU (~10 min)",
            "decision": "does intervention evidence convert into training signal?",
        },
        {
            "id": "X7", "objective": "After internalization, does dependence on the external intervention FALL?",
            "assumption": "X6 gains are computation, not context sensitivity",
            "implementation": "re-measure intervention effectiveness on the internalized child: if the Core now performs the computation, intervention lift should shrink while raw accuracy stays up",
            "controls": ["lift measured with identical instruments", "unrelated-primitive lift should stay flat"],
            "baselines": ["pre-internalization lift profile"],
            "metric": "query-lift / intervention-lift delta, raw accuracy delta",
            "promotion": "target-primitive lift shrinks AND raw accuracy holds AND other lifts unchanged",
            "falsification": "raw accuracy requires the intervention at full strength => distillation failed",
            "freshness": "same sealed suite as X6",
            "compute": "0.05 GPU-h: one P35 arm, 200 failures x 5 interventions, teacher-forced",
            "decision": "has external repair become internal cognition?",
        },
    ]
    required = {"objective", "assumption", "implementation", "controls", "baselines",
                "metric", "promotion", "falsification", "freshness", "compute", "decision"}
    for rung in rungs:
        missing = required - set(rung)
        if missing:
            raise ValueError(f"rung {rung['id']} incomplete: {sorted(missing)}")
    return {"schema": LADDER_SCHEMA, "rungs": rungs,
            "discipline": "a failed promotion sends the program to the falsification branch; rungs may not be reordered after outcomes"}


def experiment(id_: str) -> dict:
    for rung in build_ladder()["rungs"]:
        if rung["id"] == id_:
            return rung
    raise KeyError(id_)
