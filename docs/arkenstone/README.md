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
- **Current highest-value uncertainty:** where is the FIRST exact-match lift-off
  point? No branch has ever measured a lift-off threshold on the simplest task.

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

## Rules (inherited, non-negotiable)
Loss is a diagnostic, never proof of cognition. Execution artifacts beat prose.
Failures are preserved. Reproductions are labeled reproductions. Every claim gets
a novelty class. Branch isolation is absolute.
