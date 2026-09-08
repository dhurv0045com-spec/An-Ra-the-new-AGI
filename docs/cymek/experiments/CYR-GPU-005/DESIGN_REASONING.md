# CYR-GPU-005 — DESIGN REASONING

Date: 2026-09-08. Written before execution; part of the executable freeze.

## What went wrong before, in one sentence

CYR-GPU-004 compared "two independent training runs with different LRs"
and was about to sell that as "the effect of LR after learning" — the
fork contract below is what turns it into the real question.

## The causal question

Arkenstone ARK-007R/ARK-010 established, at Micro T2 scale, that LOW LR
protects an acquired capability (0/12 vs 9/12 collapse) and that HIGH LR
is usually required to recover it (8/9 vs 2/9). What NO evidence yet
covers is the switching policy: does a state-dependent rule
(state-conditional LOW retention, HIGH re-acquisition) beat either fixed
policy, and is FIXED_TIME's win condition distinguishable from ordinary
time-based decay? CYR-GPU-005 tests exactly that, on THE SAME learned
capability.

## Why the central variable might not be "LR" (section 20)

LR is a dial; the mechanism candidates behind its effect are parameter
displacement, update magnitude relative to loss-surface curvature,
gradient noise scale, and Adam second-moment dynamics. Rather than
re-design the arms around an unproven mechanism, the experiment keeps
the replicated LR framing and MEASURES the mechanism candidates
(displacement, relative displacement, update norms, moment norms,
HIGH/LOW exposure split) at every evaluation — section 24's red team is
built into the receipts. If LOW's protection is pure near-freezing, the
displacement ledger will say so and DECISION.json will carry that
reading instead of a consolidation story.

## The fork contract (sections 5–8)

One acquisition per seed; G90_CONFIRMED checkpoint captures model,
optimizer, RNG, cursor, counters, and a flat parameter snapshot; four
forks restore those bytes and are proven byte-identical
(`PARENT_EQUIVALENCE.json`) before their first update; the future
example stream is pre-generated as prefix + tail with SHAs, and every
arm's actually-consumed batches are hash-compared. This is enforced by
code and by tests, not by operator discipline.

## Design choices an auditor might challenge

- **Three acquisition parents, not more.** Replication of the PARENT
  is what guards against "G90 by luck"; 3 × (1 acquisition + 4 matched
  forks) fits the 120–150-minute Colab window with real dose
  (2–6M acquisition / 0.5–2M continuation tokens). More parents at
  starved dose would repeat CYR-GPU-004's mistake in reverse.
- **Plumbing gate override exists in smoke mode.** The TINY pipeline
  cannot genuinely reach G90 in 1–3 updates, and a plumbing test that
  never exercises the fork paths is worthless. The override is
  structural: it can only be requested in smoke mode, is stamped
  `PLUMBING_SMOKE_ONLY` in every receipt, is tested to be refused in
  full mode, and never touches the metric-gate code itself (the real
  generated-behavior gate still runs).
- **T2 canonical-pair closure is declared, not hidden.** For
  commutative addition the grammar is closed under reordering — every
  eval canonical pair exists as a reversed train row. The preregistered
  holdout axis is the ordered question (first-operand tens band 6–7
  never trained). The audit reports the crossing count as a declared
  property and fails closed on everything the claim actually rests on:
  ordered-row crossing, duplicates, same-split reversals, answer
  distribution shift vs the grammar universe, tens-band overlap.
- **Answer "balance" means distribution matching, not uniformity.** The
  grammar's answer distribution is inherently triangular (answers ≤ 99,
  pair counts 1–20). Eval splits are proportionally stratified draws;
  the audit bounds total-variation distance against the universe
  (0.20 full mode), with an explicit receipted bound for 16-row smoke
  fixtures.
- **Transfer is reported, not squeezed.** The binding/registry family
  uses query-only variants (ARK-009's diagnostic reversed fact ORDER
  while swapping the query — a confound). At zero event rate the stage
  reports `NOT_INFORMATIVE`; a null transfer with no event rate is
  information-less, not negative.
- **FIXED_TIME switches at τ = 0.5 of ACTUAL continuation tokens.**
  Any step- or capacity-based switch would re-import the nominal-token
  fiction the program exists to kill; the receipt carries the switch
  point and the lr trace proves the switch happened on actual tokens.
- **XLA fixed NOW, certified LATER.** The accumulation-boundary defect
  was deterministic engineering (early microstep gradients scaled by
  powers of R); the fix ships with a mathematical oracle and a negative
  regression. Hardware status remains `IMPLEMENTED_PENDING_PRE500M_TPU` —
  CPU equivalence is necessary, never sufficient.
