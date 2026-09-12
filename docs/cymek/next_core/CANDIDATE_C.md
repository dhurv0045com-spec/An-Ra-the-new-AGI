# CANDIDATE C — ALTERNATIVE-EXPLANATION CORE (conditional: R1C World C)

**Claim:** the strongest Core **if** R1C falsifies inactive-softmax competition and the class-space effect instead lives in tied embedding/output geometry, initialization, or optimization interaction.

## What changes vs Candidate A

Still no new blocks. The output-head/vocabulary fields unblock toward *geometry* interventions, and the experiment queue reorders.

| Field | Value | Status in this world |
|---|---|---|
| output head | candidates ranked by simplicity under World C: (1) **untied output** at matched param count (isolates output-burden from embedding); (2) **reduced physical vocabulary** (if R1B transfer shows the matrix size itself binds); (3) output-init scaling as the cheapest probe | BLOCKED → unlocked by the World-C dissection experiment |
| initialization | inactive-row init scales become a first-class experimental variable (currently one Normal(0,0.02) for all rows) | PROVISIONAL |
| optimizer | unchanged; only if gradient diagnostics implicate the tied-matrix gradient partition does an optimizer-side treatment become designable | PROVISIONAL |
| vocabulary | 24,576 stays default unless the dissection implicates matrix size directly | BLOCKED |

## The World-C dissection (minimum experiment, not yet designed in full)

Keep physical matrix fixed at 24,576; manipulate *initialization/gradient flow of inactive rows* (e.g. zero-init inactive rows, frozen inactive rows, reduced-LR inactive rows) with the same dual structural/functional endpoints as R1C. If none of these reproduce the R1/R1B effect, the effect is declared **seed/regime-specific** (R1B's seed sensitivity supports this fallback) and Candidate A remains the winner with the class-space question downgraded to a described phenomenon.

## Costs / risks / falsifier

- cheapest interventions are init-time only (zero compute overhead);
- risk: confounding "geometry" with "effective learning rate of inactive rows" — the dissection must separate row-freeze from row-scale;
- falsifier: if World-C interventions also fail, the effect is not mechanism-robust → keep Candidate A permanently and treat the class-space result as substrate-specific.

## Why it must wait

Identical logic to Candidate B: its justification is evidence that does not exist. It is designed now, built never-before-evidence.
