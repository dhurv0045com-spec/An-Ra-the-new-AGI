# ARKENSTONE

**Mission:** discover mechanisms that produce transferable cognition per parameter,
per token, and per compute — knowledge no other branch contains. Not a recombination
of known ideas; a laboratory for what remains unknown.

- **Branch:** `Arkenstone` (isolated worktree; commits only here).
- **Base:** `origin/cymek` at `28bf57a0d299a2c13a99fe0046616c00a1b8530c` (see
  `BRANCH_PROVENANCE.json` for all inspected branch SHAs at creation).
- **Central question:** what causes An-Ra to acquire exact, transferable symbolic
  computation rather than merely lowering token-prediction loss?
- **Standing anomaly inherited from Citadel:** T1/T1C — loss falls (10.1 → 1.3-1.9)
  across objectives, corpora, and 2.3x scale at 4M tokens, while held-out AND train
  exact-match stay ~0. Train exact ~0 means the model never even fits seen rows.
- **Updated after ARK-004A-R reanalysis:** the claimed selectivity precursor was DIRECTIONALLY INVERTED in prose (higher early tens-selectivity actually associates with LATER G90) and is a transition MARKER, not a precursor (does not precede OOD in 3/4 seeds); ARK-004B CANCELLED. What stands: transition universal (7 seeds), timing seed-variable and decoupled from memorization speed, post-G90 instability real (seed 101 collapse) — retention is now the frontier (ARK-005).
- **Current highest-value uncertainty:** where is the FIRST exact-match lift-off
  point? lift-off thresholds now measured (ARK-001/002B/004A).

## Branch relations (external gap review GAP 3/6)
See `BRANCH_RELATIONS` content in `branch_relations.py` and the review
assessment in `EXTERNAL_GAP_REVIEW_ASSESSMENT.md` — per-branch
proves/done/connects-to, maintained here because no other branch may change.
Arkenstone's ledgers are self-verifying: `verify_ledgers.py` re-checks every
referenced artifact and receipt hash and stamps this file
(`verified-at-commit:`) — drift is mechanical, not manual (GAP 2/4 pattern).

## Progress tracking

- `PROGRESS.md` — brief dated entries, most recent first
- `IMPROVEMENTS.md` — every adopted improvement (dated, attributed)
- `FAILURES.md` — every experiment failure and falsified claim (dated, attributed)

## Documents
| File | Purpose |
|---|---|
| `UNIFIED_EVIDENCE_MAP.md` | everything branches demonstrated/failed/never-tested |
| `COGNITION_BOTTLENECK_GRAPH.md` | dependency chain + ranked bottlenecks |
| `MECHANISM_TOURNAMENT.md` | candidate mechanisms with verdicts |
| `AGI_FEATURE_LEDGER.md` | per-feature evidence/cost/novelty records |
| `NEGATIVE_RESULTS.md` | failures, preserved forever |
| `EXPERIMENT_LOG.md` | every experiment, one line each |
| `NOVELTY_REGISTER.md` | novelty classification per claim |
| `BRANCH_PROVENANCE.json` | machine-readable branch/environment provenance |

## Experiments
- `experiments/ARK-001/` — micro lift-off mapping (executed, analyzed)
- `experiments/ARK-002/` — grokking continuation + ERRATUM_002a (see also ARK-002B for the multi-seed replication)
- `experiments/ARK-002B/` — independent T2 replication, separated init/order seeds, commutation-free manifest
- `experiments/BINDING-V2-REDTEAM/` — independent verification of cymek's binding-v2 qualification (CONSISTENT)
- `experiments/ARK-004A/` — developmental transition mapping + precursor discovery (executed)

## Rules (inherited, non-negotiable)
Loss is a diagnostic, never proof of cognition. Execution artifacts beat prose.
Failures are preserved. Reproductions are labeled reproductions. Every claim gets
a novelty class. Branch isolation is absolute.

---

stamped-at-commit: 986ddea02082d92e33f877f7d30684b9dd345d4b
