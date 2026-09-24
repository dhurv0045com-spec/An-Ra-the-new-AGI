# Goal-specific comparisons

There is no universal branch league table. A score is meaningful only inside a stated goal and family. `unknown` is not zero.

## Rubric

- 5 = strong, directly evidenced and current for the stated scope; 4 = good with bounded caveats; 3 = mixed/partial; 2 = substantial blocker or stale authority; 1 = weak or contradicted; `unknown` = not evidenced.
- Scientific maturity is not implementation completeness. A branch may be operationally useful and scientifically negative, or promising and unexecuted.
- Scores are reviewed profile judgments, never LOC, commits, model size, benchmark, or last-modified proxies.

## Best branch for understanding overall research direction

**Question:** Which ref best explains the current research thesis, its evidence boundaries, and the next discriminating experiment without treating prose as proof?

**Recommendation:** `refs/remotes/origin/Gandiva` (confidence: medium).

**Why:** Gandiva is the current owner-facing integration point: it combines the first-principles BRAMASTRA thesis, an evidence audit that preserves negative/inconclusive findings, and the latest K8 implementation work. It is a reading recommendation, not a scientific winner.

**Tradeoffs:** Citadel and Arkenstone are stronger bounded evidence sources; BRAMASTRA is more operator-focused; Triquetra and Cymek expose deeper mechanism and evidence ledgers. None establishes AGI.

| Candidate | goal clarity | specification completeness | implementation completeness | reproducibility | evaluation quality | provenance and custody | hardware qualification | scientific result maturity | negative evidence handling | operational readiness | unresolved blockers | stale instruction risk | unique contribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `refs/remotes/origin/Gandiva` | strong (5) | strong (5) | bounded (4) | bounded (4) | strong (4) | strong (4) | unknown | mixed (3) | strong (5) | bounded (4) | mixed (3) | mixed (3) | strong (5) |
| `refs/remotes/origin/BRAMASTRA` | strong (4) | strong (4) | strong (4) | bounded (4) | bounded (4) | strong (4) | bounded (3) | mixed (2) | strong (4) | strong (4) | mixed (3) | mixed (3) | strong (4) |
| `refs/remotes/origin/Arkenstone` | strong (5) | strong (5) | strong (4) | strong (5) | strong (5) | strong (5) | bounded (4) | strong (4) | strong (5) | strong (4) | mixed (3) | mixed (3) | strong (5) |
| `refs/remotes/origin/citadel` | strong (5) | strong (5) | strong (4) | strong (5) | strong (5) | strong (5) | bounded (3) | strong (4) | strong (5) | mixed (3) | mixed (3) | low (2) | strong (5) |
| `refs/remotes/origin/triquetra` | strong (4) | strong (4) | bounded (3) | strong (4) | strong (5) | strong (4) | bounded (3) | strong (4) | strong (5) | bounded (3) | mixed (3) | mixed (3) | strong (5) |
| `refs/remotes/origin/cyhex-hermes` | strong (4) | strong (5) | strong (4) | bounded (4) | strong (5) | strong (5) | bounded (4) | strong (4) | strong (5) | mixed (3) | mixed (3) | mixed (3) | strong (5) |

**Evidence anchors:**
- `refs/remotes/origin/Gandiva:BRAMASTRA.md` § The central decision
- `refs/remotes/origin/Gandiva:docs/bramastra/EVIDENCE.md` § Consequences
- `refs/remotes/origin/Gandiva:AGENTS.md` § Mission and authority

**Smallest resolving comparison:** If the goal is evidence audit rather than overall direction, use Citadel and the relevant branch-specific ledger; if it is mechanism discovery, use Arkenstone's current state.

## Best branch for a specified near-term engineering task

**Question:** For finishing the owner-launched two-T4 K8 build and preserving a resumable evidence bundle, which ref is the best starting point?

**Recommendation:** `refs/remotes/origin/Gandiva` (confidence: medium).

**Why:** Gandiva has the latest E5/K8 fixes and the explicit final experiment-build work order. BRAMASTRA is a focused alternative for K8 notebook operations; Cymek-beta is the right branch only for Formation-Mux checkpoint recovery.

**Tradeoffs:** Gandiva is broader and therefore more complex; BRAMASTRA is narrower but behind the latest fixes; recovery work is blocked on external checkpoint custody and must not be mixed with a new experiment.

| Candidate | goal clarity | specification completeness | implementation completeness | reproducibility | evaluation quality | provenance and custody | hardware qualification | scientific result maturity | negative evidence handling | operational readiness | unresolved blockers | stale instruction risk | unique contribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `refs/remotes/origin/Gandiva` | strong (5) | strong (5) | bounded (4) | bounded (4) | strong (4) | strong (4) | unknown | mixed (3) | strong (5) | bounded (4) | mixed (3) | mixed (3) | strong (5) |
| `refs/remotes/origin/BRAMASTRA` | strong (4) | strong (4) | strong (4) | bounded (4) | bounded (4) | strong (4) | bounded (3) | mixed (2) | strong (4) | strong (4) | mixed (3) | mixed (3) | strong (4) |
| `refs/remotes/origin/cymek-beta` | strong (5) | strong (5) | strong (4) | bounded (3) | mixed (3) | strong (5) | bounded (4) | mixed (2) | strong (4) | mixed (3) | strong (2) | low (2) | strong (5) |
| `refs/remotes/origin/cymek-cs-transfer-001` | strong (4) | strong (5) | strong (4) | bounded (4) | strong (5) | strong (5) | bounded (4) | strong (4) | strong (5) | mixed (3) | mixed (3) | mixed (3) | strong (5) |

**Evidence anchors:**
- `refs/remotes/origin/Gandiva:AGENTS.md` § Active work
- `refs/remotes/origin/Gandiva:docs/bramastra/EXPERIMENTS.md` § Execution order
- `refs/remotes/origin/BRAMASTRA:BRAMASTRA.md` § Operating within 100 TPU-hours a week

**Smallest resolving comparison:** The smallest resolving comparison is a read-only diff of the final K8 work order, launcher pins, and receipt requirements between Gandiva and BRAMASTRA.

## Best branch for strongest completed experimental evidence

**Question:** Which family has the strongest completed, repeated, receipt-backed evidence within a clearly bounded scientific scope?

**Recommendation:** `refs/remotes/origin/Arkenstone` (confidence: medium).

**Why:** Arkenstone has the densest sequence of executed, matched, audited bounded experiments and explicit replication/failure records. This is not a claim of the best hypothesis or AGI; Citadel is stronger for audit authority and Cyhex for recent executed negative outcomes.

**Tradeoffs:** Arkenstone's breadth creates a larger stale-document and family-boundary risk; Citadel's evidence is more explicitly audited but narrower; Cyhex has recent hardware executions with unresolved reproduction gaps.

| Candidate | goal clarity | specification completeness | implementation completeness | reproducibility | evaluation quality | provenance and custody | hardware qualification | scientific result maturity | negative evidence handling | operational readiness | unresolved blockers | stale instruction risk | unique contribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `refs/remotes/origin/Arkenstone` | strong (5) | strong (5) | strong (4) | strong (5) | strong (5) | strong (5) | bounded (4) | strong (4) | strong (5) | strong (4) | mixed (3) | mixed (3) | strong (5) |
| `refs/remotes/origin/citadel` | strong (5) | strong (5) | strong (4) | strong (5) | strong (5) | strong (5) | bounded (3) | strong (4) | strong (5) | mixed (3) | mixed (3) | low (2) | strong (5) |
| `refs/remotes/origin/cyhex-hermes` | strong (4) | strong (5) | strong (4) | bounded (4) | strong (5) | strong (5) | bounded (4) | strong (4) | strong (5) | mixed (3) | mixed (3) | mixed (3) | strong (5) |
| `refs/remotes/origin/triquetra` | strong (4) | strong (4) | bounded (3) | strong (4) | strong (5) | strong (4) | bounded (3) | strong (4) | strong (5) | bounded (3) | mixed (3) | mixed (3) | strong (5) |

**Evidence anchors:**
- `refs/remotes/origin/Arkenstone:docs/arkenstone/CURRENT_STATE.md` § Evidence that should be treated as current
- `refs/remotes/origin/Arkenstone:docs/arkenstone/EXPERIMENT_LOG.md` § Experiments
- `refs/remotes/origin/citadel:docs/citadel/EVIDENCE_LEDGER.md` § E3

**Smallest resolving comparison:** Compare the same preregistered estimand, hardware scope, seed count, and custody fields across one Arkenstone result, one Citadel audit claim, and one Cyhex result before calling any family strongest.

## Best branch for data and evidence integrity

**Question:** Which ref should an auditor read to identify contamination, shortcut, stale-green, and provenance failures?

**Recommendation:** `refs/remotes/origin/citadel` (confidence: high).

**Why:** Citadel's research protocol and evidence ledger explicitly separate documentary claims from receipt-backed measurements and preserve negative results; the Cyhex integrity branch is the narrower custody specialist.

**Tradeoffs:** Citadel audits inherited claims rather than producing new capability evidence; Cyhex integrity verifies bundles but cannot explain the scientific discrepancy.

| Candidate | goal clarity | specification completeness | implementation completeness | reproducibility | evaluation quality | provenance and custody | hardware qualification | scientific result maturity | negative evidence handling | operational readiness | unresolved blockers | stale instruction risk | unique contribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `refs/remotes/origin/citadel` | strong (5) | strong (5) | strong (4) | strong (5) | strong (5) | strong (5) | bounded (3) | strong (4) | strong (5) | mixed (3) | mixed (3) | low (2) | strong (5) |
| `refs/remotes/origin/codex/cyhex-integrity-audit` | strong (5) | strong (4) | strong (4) | strong (5) | strong (4) | strong (5) | unknown | mixed (3) | strong (5) | bounded (3) | mixed (3) | low (2) | strong (5) |
| `refs/remotes/origin/triquetra` | strong (4) | strong (4) | bounded (3) | strong (4) | strong (5) | strong (4) | bounded (3) | strong (4) | strong (5) | bounded (3) | mixed (3) | mixed (3) | strong (5) |

**Evidence anchors:**
- `refs/remotes/origin/citadel:docs/citadel/RESEARCH_PROTOCOL.md` § Receipts are immutable; ledgers are authoritative
- `refs/remotes/origin/citadel:docs/citadel/NEGATIVE_RESULTS.md` § N1 to N20

**Smallest resolving comparison:** Use Citadel for claim authority and Cyhex integrity for bundle custody; a single branch need not own both functions.

## Best branch for Formation-Mux recovery

**Question:** Which ref contains the safest exact continuation path for the partial Formation-Mux session?

**Recommendation:** `refs/remotes/origin/cymek-beta` (confidence: high).

**Why:** The beta branch contains the pinned recovery notebook, preflight, session report, and explicit missing-checkpoint blocker.

**Tradeoffs:** Recovery is blocked on the original saved Output tree; no branch can replace that external artifact or change the frozen protocol without creating a new experiment.

| Candidate | goal clarity | specification completeness | implementation completeness | reproducibility | evaluation quality | provenance and custody | hardware qualification | scientific result maturity | negative evidence handling | operational readiness | unresolved blockers | stale instruction risk | unique contribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `refs/remotes/origin/cymek-beta` | strong (5) | strong (5) | strong (4) | bounded (3) | mixed (3) | strong (5) | bounded (4) | mixed (2) | strong (4) | mixed (3) | strong (2) | low (2) | strong (5) |
| `refs/remotes/origin/cymek-next-core-architecture` | strong (4) | strong (5) | strong (4) | bounded (4) | strong (5) | strong (5) | bounded (4) | strong (4) | strong (5) | mixed (3) | mixed (3) | mixed (3) | strong (5) |
| `refs/remotes/origin/cyhex-hermes` | strong (4) | strong (5) | strong (4) | bounded (4) | strong (5) | strong (5) | bounded (4) | strong (4) | strong (5) | mixed (3) | mixed (3) | mixed (3) | strong (5) |

**Evidence anchors:**
- `refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/RECOVERY_RUNBOOK.md` § Status
- `refs/remotes/origin/cymek-beta:artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/SESSION_REPORT.md` § What did not complete

**Smallest resolving comparison:** The smallest resolving action is external: obtain and hash the complete Kaggle saved Output tree, then run the pinned preflight without changing protocol.

## Best branch for the bounded transfer architecture decision

**Question:** Which record should control the current vocabulary/architecture constraint after CS-TRANSFER-001?

**Recommendation:** `refs/heads/research/evidence-consolidation-2026-09-25` (confidence: high).

**Why:** The evidence-consolidation branch preserves the completed result, correction, provenance, and active architecture constraint in one decision record.

**Tradeoffs:** The decision is explicitly bounded and interaction/seed-sensitive; it does not authorize production scale or broad cognition claims.

| Candidate | goal clarity | specification completeness | implementation completeness | reproducibility | evaluation quality | provenance and custody | hardware qualification | scientific result maturity | negative evidence handling | operational readiness | unresolved blockers | stale instruction risk | unique contribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `refs/heads/research/evidence-consolidation-2026-09-25` | strong (4) | strong (4) | bounded (3) | bounded (4) | strong (4) | strong (5) | bounded (3) | strong (4) | strong (5) | mixed (3) | mixed (3) | low (2) | strong (4) |
| `refs/remotes/origin/cyhex-hermes` | strong (4) | strong (5) | strong (4) | bounded (4) | strong (5) | strong (5) | bounded (4) | strong (4) | strong (5) | mixed (3) | mixed (3) | mixed (3) | strong (5) |
| `refs/remotes/origin/cymek-next-core-architecture` | strong (4) | strong (5) | strong (4) | bounded (4) | strong (5) | strong (5) | bounded (4) | strong (4) | strong (5) | mixed (3) | mixed (3) | mixed (3) | strong (5) |

**Evidence anchors:**
- `refs/heads/research/evidence-consolidation-2026-09-25:docs/research/CS_TRANSFER_001_FINAL_EVIDENCE.md` § Decision
- `refs/heads/research/evidence-consolidation-2026-09-25:docs/research/CS_TRANSFER_001_FINAL_EVIDENCE.md` § Claim ceiling

**Smallest resolving comparison:** A fresh preregistered matched comparison with more seeds and independent custody is required to resolve the remaining interaction.

## Additional decision views

- **Data integrity:** use Citadel's evidence ledger/protocol for contamination and shortcut audit; use the Cyhex integrity audit for custody and receipt verification. Neither is production promotion authority.
- **Formation-Mux recovery:** use the Cymek-beta recovery runbook and session report; the required original checkpoint tree is external and the campaign remains partial.
- **Architecture decision:** use the CS-TRANSFER-001 decision for its bounded vocabulary conclusion and retain its explicit prohibition on production-scale authorization.

## Non-comparisons

Do not compare an evidence archive, training branch, operator notebook, and Core runtime as candidates for the same scientific claim. Their valid comparison is functional: custody, implementation, execution, or decision authority.
