# ESOES Status

Updated: 2026-08-29

Branch phase: **GROUND BLUEPRINT v0.3 — 250M CODE/INFRASTRUCTURE CONTRACT EXECUTABLE**

Canonical research blueprint: [`V5_MASTER_BLUEPRINT.md`](V5_MASTER_BLUEPRINT.md)

Research source ledger: [`EVIDENCE_BASE.md`](EVIDENCE_BASE.md)

Large V5 training authorized: **NO**

Frozen V5 spec exists: **NO**

Provisional V5-A: **250.22M dense, 26×896, 14Q/7KV, FFN 2368, affine QK norm, 4k full attention, 24,576 byte-fallback vocabulary, 5.0B audited tokens.** Exact shape, tokenizer, data fraction, objective, LR, batch, and schedule remain experiment-gated.

Current evidence says the program should improve data quality, token sufficiency, causal query supervision, and behavioral checkpoint selection before scaling parameters.

Executed through this phase: VNext contamination is removed; EXP evidence is corrected; E0 now covers candidate/context positions, answer formats, independent solvers, and machine-preregistered statistics; sealed custody fails closed; E1 has an artifact-bound static audit/Pareto harness; and exact 250M model/run/lineage contracts are executable.

Verification: **29 CPU tests pass**, both committed receipts reproduce, and package import boundaries pass.

Next action: **external inputs are required.** An independent custodian must create T2 outside Git, and real 16k/24k/32k tokenizer artifacts plus hash-bound corpus receipts must be supplied. Then run E1 static audits and bounded P35 comparisons. Do not start the 250M main run.

Required sequence: E0 → P35 tokenizer/architecture/data screens → M102 replication → freeze review → target-TPU canaries → V5-A main run.
