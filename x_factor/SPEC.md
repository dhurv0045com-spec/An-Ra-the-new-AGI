# SYNTHETIC WORLD SPECIFICATION — Causal Cognitive Self-Modeling

> **STATUS:** SYNTHETIC_MECHANISM_DEMONSTRATION. This spec describes the
> deterministic toy world used to validate the software framework. It is NOT
> the real-model causal specification. See REAL_MODEL_CAUSAL_SPEC.md for
> the actual scientific protocol principles.

# X FACTOR — Causal Cognitive Self-Modeling

**Thesis.** A failed execution's observed state predicts which cognitive
intervention repairs it, because failures carry low-rank latent causal
structure that interventions address factor-wise. The end state is not a
failure classifier: it is a model that predicts *future* intervention
outcomes — which intervention, how much effect, when nothing is needed —
and whose predictions are graded by actual counterfactual interventions.

**Representation decision (replacing the raw fingerprint vector).** The
fingerprint F(x) = [predicted effect per intervention] is *derived*, not
stored: the primary object is the task × intervention outcome matrix, whose
low-rank structure (rank ≤ |latent factors| + 1 under the physics of
`world.py`) is what the learner must exploit. Why: rank deficiency is
mechanically testable (X0), factor names never hard-code into the learner,
and the representation degrades gracefully — if the matrix is full-rank,
there is no causal structure and the program stops. A raw feature vector
would have hidden this test.

**Leakage law (enforced in code, `contracts.py`).** A policy sees only
`ObservedFailureFeatures` (observed factor-gap signals, candidate/arity/
format counters, confidence). Forbidden everywhere in policy-visible
records: correctness labels, gold answers, gold ranks, hidden required-
factor sets, family identity, and any intervention outcome. The audit
`assert_observation_legality` rejects forbidden keys in serialized
evidence, including nested evaluator blocks. Outcomes of *training* tasks
are legal intervention evidence; outcomes of *evaluation* tasks are not.

**Intervention semantics.** Each intervention supplies a set of latent
factors; NO_CHANGE supplies nothing (the do-nothing control). Repair =
all missing factors supplied. Effect = signed, magnitude-bearing (repair +
penalty per residual missing factor). The registry is data, not taxonomy:
the learned representation may discover any structure over outcomes.

**Falsification law.** The program abandons the mechanism if any of:
(X0) outcome matrices show no low-rank structure; (X2) held-out
predictions collapse to fixed-policy level; (X3) the family shortcut ties
the learner cross-family (we learned templates, not cognition); (X7)
internalized accuracy still requires the external intervention at full
strength. A textual explanation of failure is never evidence; only
predicted-vs-actual intervention outcomes count.

**Metric set (fixed):** top-1 repair accuracy · regret vs oracle ·
pairwise ranking accuracy · effect-sign accuracy · Brier calibration ·
cost-adjusted score (repair − λ·cost). Discrimination is expected in the
cost-adjusted axis: FULL_REPLAY repairs everything at cost 4; intelligence
is picking the cheapest covering intervention.

**Structural negative control.** Surface family is sampled independently
of latent requirements, so a family-ID shortcut policy dominates in-family
and collapses cross-family. Proven in tests: the learned observed-only
policy beats every fixed policy and the shortcut on cost-adjusted score.
Any future real-data result where the shortcut survives X3 means we
learned templates, not cognition.

**The ladder (preregistered, `ladder.py`).** X0 structure exists → X1
held-out prediction → X2 fresh instances → X3 cross-family → X4
cross-checkpoint → X5 cost-pressure vs strong heuristics → X6
internalization (repair-conditioned SFT, ≥50% rehearsal, retention
floors) → X7 dependence fall. Each rung carries objective, assumption,
controls, baselines, metrics, promotion, falsification, freshness rule,
compute estimate, and its decision. A failed promotion routes to the
falsification branch; rungs may not be reordered after outcomes.

**Training connection (conservative).** The smallest mechanism testing
"can successful external repairs become native capability": distill
verified repair trajectories into SFT where the intervention's context
change is removed but the target stays (X6), then re-measure intervention
lift on the child (X7). Internalization is demonstrated only if raw-core
accuracy rises while intervention lift falls and protected retention holds
within parent − 0.10.

**AGI boundary (precise).** Ordinary models learn task mappings; adaptive
systems learn behavior; An-Ra is attempting to learn a causal model of how
its own failures respond to interventions, and to convert that knowledge
into native computation. Demonstrated, that is a real ingredient for
adaptive cognition. It is not AGI, and this program does not claim it.

**Status on this branch.** Contracts, leakage law, world physics,
policies, oracle isolation, metrics, learner, and the full ladder are
implemented and test-proven on the synthetic world (8 tests). Every
real-model rung is preregistered and intentionally NOT executed here.
