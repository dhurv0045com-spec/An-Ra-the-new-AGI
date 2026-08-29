# ESOES Status

Updated: 2026-08-29

Branch phase: **STEP 2 COMPLETE — PRE-FREEZE EXPERIMENT DESIGN**

Canonical research blueprint: [`V5_MASTER_BLUEPRINT.md`](V5_MASTER_BLUEPRINT.md)

Research source ledger: [`report-source.md`](report-source.md)

Large V5 training authorized: **NO**

Frozen V5 spec exists: **NO**

Provisional V5-A: **195.08M dense, 28×768, 4k full attention, 24,576 byte-fallback vocabulary, 4.0B audited tokens.** Exact architecture, tokenizer, data fraction, objective, LR, batch, and schedule remain experiment-gated.

Current evidence says the program should improve data quality, token sufficiency, causal query supervision, and behavioral checkpoint selection before scaling parameters.

Next action: **E0 benchmark and generator certification.** Freeze the representation/selection/realization metrics, OOD axes, direct-retrieval controls, split hashes, leakage tests, and sealed Tier 2 suite before using TPU time.

Required sequence: E0 → P35 tokenizer/architecture/data screens → M102 replication → freeze review → target-TPU canaries → V5-A main run.
