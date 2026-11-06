# ELENDRIL — CODEX RESEARCH MANDATE (BOOTSTRAP)

**6 November 2026.** This is the repository-native starting charter for Codex. The expanded plan and copy-ready master prompt are delivered separately in the AN-RA research conversation. This document is self-contained enough to begin an honest first iteration.

## Mission
Build a serious and extensible, reproducible research platform for cumulative, resource-efficient learning. The long-term objective is to investigate whether a relatively small independently trained neural system can acquire reusable skills, retain them, transfer them to new tasks, and eventually improve its memory/strategy, weights, and architecture by independently verified interventions. This is a research hypothesis, not an achieved AGI/RSI result.

## Git and branch authority
- Work only on **Elendril**, branched from **cymek-upgraded** commit **feac2948466832516b7f048435483fd95e53b078**.
- Do **not** use main as the research authority. Do not rewrite, merge, delete, or force-update other research branches.
- Treat `research/evidence-consolidation-2026-09-25` as a major evidence reference, not automatically as executable integration code.
- Do not merge entire branch histories when a single vetted component suffices; record exact donor commits and tests.

## Research sources to inspect before deciding code
- `research/evidence-consolidation-2026-09-25/docs/research/`: experiment evidence ledger, evidence reversals, belief registry, negative results, evidence gaps, source manifest, dependency graph, decision tree and next-phase memo. Reported September 24 status includes 90 ledger entries.
- `cymek-upgraded/docs/cymek/research/`: Master AGI Construction Knowledge, Current State, cross-branch audit.
- `cymek-upgraded/docs/cymek/experiments/`: all CYR-GPU experiment reports, FORMATION-MUX-001, METRIC-RES-001, failure/negative/mechanism ledgers.
- `cymek-upgraded/docs/cymek/next_core/`: architecture decisions, bottleneck model and current-state override.
- Inspect `Arkenstone`, `triquetra`, `BRAMASTRA`, `Gandiva`, `citadel`, Cyhex branches and other pertinent records without assuming branches serve the same role.

**Known bounded evidence to reverify, not blindly assume:** R1C completed 24/24 and rejected inactive-softmax sufficiency; CS-TRANSFER-001 completed with PARTIAL_OR_INTERACTION; V5.1 canary-v2 mechanically passed but failed identity formation; Formation-Mux S5 completed 24/24 yet identity/copy was floor-limited; METRIC-RES-001 requires preserved S5 checkpoints absent from the reviewed compact bundle; Arkenstone showed bounded micro retention effects and unresolved new-skill formation; Brahmastra EOS helped terminal behavior but query-blind baselines explained an apparent binding gain; Triquetra exposed value-recency confounds; Citadel identified data/eval shortcuts; K8 was engineering-partial. Distinguish primary raw receipts from transcriptions, plans, and outdated handoffs.

## Engineering execution: do the work
1. **Recover evidence and custody.** Enumerate hypotheses/experiments, exact commits, execution status, raw data and checkpoint locations. Flag contradictions and missing external bytes; never fabricate replacements or report an unexecuted run as completed.
2. **Choose Cymek baseline.** Audit real model/trainer/tokenizer/sampler/checkpoint/evaluator implementations. Choose the smallest robust dense decoder-only baseline and frozen training recipe together. Reuse verified mechanisms; do not adopt speculative modules into the canonical baseline without a test.
3. **Qualify mechanical correctness.** Implement/repair tests for causal masking, prompt/answer/EOS alignment, reversible tokens, parameter and optimizer updates, finite gradients, deterministic samples, exact uninterrupted-vs-resumed continuation, checkpoint identity, scorer oracles and negative controls. Show a *learning* positive control, not merely a forward pass.
4. **Design audited development benchmark.** Identity/copy, query-sensitive binding, order/state, composition, missing-information abstention, retention and prospective forward transfer. Reject query-blind, recency/position, format-copy, repeated-world, leakage and scorer shortcuts. Build development fixtures, never tune on sealed scores.
5. **Run bounded pilots** when available with explicit seeds, matched training budgets, raw predictions, inference and verifier costs, per-family learning curves, and visible stop reasons. If hardware/artifacts are unavailable, give reproducible commands and classify the blocker honestly.
6. **Draft causal experiment and adaptive-lab protocol.** Forward-transfer: A learned, B learned/retained, unseen A+B, then C; compare scratch, fixed replay, memory and parameter adaptation at matched resources. Longer term: an independent verifier/promoter gates memory, weight and architectural interventions.
7. **Deliver actual changed code, tests, reports and handoff**, not only recommendations. Record every claim's evidence ceiling. Commit reversible, well-scoped changes to Elendril.

## Gates
- G0 evidence/custody reconciled; G1 model/training mechanics pass; G2 deterministic + exact resume; G3 evaluation/anti-shortcut validity; G4 bounded positive-control learning away from floor; G5 prospective matched causal tests; G6 independent promotion/rollback; G7 separate owner authorization for large compute and scale.
- Without G0-G4, do not interpret negative/positive cognitive outcomes from an unqualified instrument.
- Do not consume sealed fixtures for design, mutate historical receipts, spend large/paid GPU/TPU allocations, destroy checkpoints or launch large training without explicit authorization.

## Required durable files
Create and maintain `docs/elendril/BRANCH_LINEAGE.md`, `RESEARCH_EVIDENCE_MATRIX.md`, `ARTIFACT_CUSTODY_REGISTER.md`, `EVIDENCE_CONFLICTS.md`, `BASELINE_SELECTION.md`, `CORRECTNESS_QUALIFICATION.md`, `BENCHMARK_SPEC.md`, `NEXT_EXPERIMENT_DECISION.md`, and `SESSION_HANDOFF.md`, plus tests/code in coherent existing modules.

## Reporting rule
For each material change: (i) changed files/SHAs, (ii) commands and observed tests/results, (iii) resource budget, (iv) failure categories, (v) measured claim and limits, (vi) next cheapest decisive action. Do not invent charts or imply AGI from CI, loss or benchmark accuracy.

**Begin with Cymek and build a research instrument worthy of adversarial review. Small models are laboratories, never the project's ambition ceiling.**