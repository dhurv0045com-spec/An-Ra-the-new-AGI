# ESOES Research Map

## What ESOES is

ESOES is the clean-sheet architecture laboratory for An-Ra V5. It exists because V4/PGE improved language-model loss without reliably producing contextual binding or composition, while targeted SFT and EXP demonstrated that controlled query interventions can expose and alter specific failure mechanisms.

## Current phase

- Design iterations completed: **4 of 4**
- Ground blueprint: **v0.1**
- Production V5 implementation: **not started**
- Major V5 training: **not authorized**
- Current iteration: **closed; no Iteration 5**

## Canonical knowledge system

Read in this order:

1. [`EVIDENCE_BASE.md`](EVIDENCE_BASE.md) — internal receipts and public research ESOES may reason from.
2. [`ITERATIONS.md`](ITERATIONS.md) — four bounded attacks that produced Ground Blueprint v0.1.
3. [`V5_MASTER_BLUEPRINT.md`](V5_MASTER_BLUEPRINT.md) — authoritative human-readable design.
4. [`DECISIONS.md`](DECISIONS.md) — frozen, provisional, experimental, and open decisions; changes require explicit reopening.
5. [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) — only uncertainties capable of materially changing the program.

Post-Iteration-4 engineering specifications live in [`../../blueprints/`](../../blueprints/). They describe what may eventually be built; they are not implementation.

Historical files (`EVIDENCE_AND_CONTEXT.md`, `V5_COGNITION_FIRST_BLUEPRINT.md`, `DECISION_LOG.md`, `FREEZE_CHECKLIST.md`, `STATUS.md`, `report-source.md`, and `AGENT_HANDOFF.md`) are retained to avoid destructive loss. They are non-canonical and may not override the six documents above.

## Status language

- **[FROZEN]** — fixed for Ground Blueprint v0.1; reopen only through `DECISIONS.md`.
- **[PROVISIONAL]** — bounded working choice, not permission for a major run.
- **[EXPERIMENT REQUIRED]** — a named pre-implementation test owns the decision.
- **[OPEN]** — intentionally unresolved or deferred.

## Forbidden now

- no production trainer, data pipeline, tokenizer, or model implementation;
- no V5 checkpoint lineage or major training run;
- no copying VNext code into ESOES because it already exists;
- no tuning on sealed evaluation fixtures;
- no calling an assisted runtime result native Core cognition;
- no fifth open-ended architecture iteration.

Next action: implement **E0 only in the next authorized phase**—the benchmark/generator certification described in `blueprints/EXPERIMENTS.md`—then stop for its evidence before E1.
