# W08 — Independent examiner and claim gates

**Status:** ready against W01 documented records; integration depends on W01/W02. **Effort:** 3–5 hours. **Role:** evaluation/statistical engineer. **Compute:** CPU synthetic outcome checks and small inference fixtures.

Read contracts, experiment registry and charter. Own `research/evaluation/`, examiner tests and `engineering/reports/W08/`. Reuse reviewed prototype statistics through an explicit adapter where correct; do not fork several incompatible score definitions.

## Deliverable

Implement independent task allocation, matched-arm scoring, world-clustered uncertainty, per-family reporting, retention comparisons and a configurable promotion rule. Separate training probes, development, strategy validation and confirmation access. The candidate has no write access to examiner rules or outcome labels.

## Required behavior

Validate exact paired keys, label/family consistency, duplicate records, probability ranges, legal query counts and complete answer termination where applicable. Recompute correctness from predictions. Preserve raw rows and failed/incomplete runs.

Promotion rules take explicit practical gain and retention margins plus a declared finite testing budget. Missing margins, inadequate evidence or failed prerequisites return insufficient evidence, never pass. Report cost and cumulative retention history alongside the decision.

## Acceptance evidence

- Hand-calculated fixtures agree with accuracy, Brier and paired deltas.
- Cluster bootstrap resamples worlds, not correlated query rows; seed-level variability remains a separate report.
- Mismatched keys, inconsistent labels, invalid probabilities and forged correctness flags are rejected.
- A high mean gain with a protected-family violation is rejected under the configured rule.
- One-world or underpowered fixtures cannot produce a confident promotion claim.
- Repeated candidate/pool use is tracked; pools consumed by development are not silently reused as independent confirmation.
- Learner access tests demonstrate the intended boundary, with process-isolation limitations stated accurately.

Do not invent universal AGI thresholds. The examiner establishes specific claims under a named protocol. A promotion decision cannot certify general intelligence or guarantee future retention outside measured families.
