# ESOES Status

Updated: 2026-08-29

Branch phase: **GROUND BLUEPRINT v0.2 — E0 DEVELOPMENT CERTIFIED, SEALED GATE OPEN**

Canonical research blueprint: [`V5_MASTER_BLUEPRINT.md`](V5_MASTER_BLUEPRINT.md)

Research source ledger: [`EVIDENCE_BASE.md`](EVIDENCE_BASE.md)

Large V5 training authorized: **NO**

Frozen V5 spec exists: **NO**

Provisional V5-A: **195.08M dense, 28×768, 4k full attention, 24,576 byte-fallback vocabulary, 4.0B audited tokens.** Exact architecture, tokenizer, data fraction, objective, LR, batch, and schedule remain experiment-gated.

Current evidence says the program should improve data quality, token sufficiency, causal query supervision, and behavioral checkpoint selection before scaling parameters.

Executed this phase: VNext implementation artifacts were removed; EXP v10/v11 and the old VIE bank were reclassified; an executable E0 development suite, separate training generator, metrics, controls, independent surface solver, chance/power calibration, certificate, and 14 regression/property tests were added.

Next action: **finish the E0 exit gate.** Add context-position/output-format balancing, preregister paired/exact metric procedures, and source-disjoint natural fixtures; then create Tier 2 under external seed custody and commit only its hash. Do not spend TPU time on model comparisons first.

Required sequence: E0 → P35 tokenizer/architecture/data screens → M102 replication → freeze review → target-TPU canaries → V5-A main run.
