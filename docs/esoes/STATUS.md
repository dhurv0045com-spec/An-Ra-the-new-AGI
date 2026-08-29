# ESOES Status

Updated: 2026-08-30

Branch phase: **GROUND BLUEPRINT v0.4 — SHORTCUT-RESISTANT E0 CONTRACT EXECUTABLE**

Canonical research blueprint: [`V5_MASTER_BLUEPRINT.md`](V5_MASTER_BLUEPRINT.md)

Research source ledger: [`EVIDENCE_BASE.md`](EVIDENCE_BASE.md)

Large V5 training authorized: **NO**

Frozen V5 spec exists: **NO**

Provisional V5-A: **250.22M dense, 26×896, 14Q/7KV, FFN 2368, affine QK norm, 4k full attention, 24,576 byte-fallback vocabulary, 5.0B audited tokens.** Exact shape, tokenizer, data fraction, objective, LR, batch, and schedule remain experiment-gated.

Current evidence says the program should improve data quality, token sufficiency, causal query supervision, and behavioral checkpoint selection before scaling parameters.

Executed through this phase: VNext contamination is removed; EXP evidence is corrected; E0 now randomizes semantic state serialization, covers intermediate/rollback/precedence queries, holds out rule structures across splits, pools positional/fixed-rule/bag-of-words red-team seeds, reports difficulty curves, and has raw/assisted/intervention-dependence plus replication contracts; sealed custody fails closed; E1 has an artifact-bound static audit/Pareto/matched-budget harness, a real V4 baseline, truncation proxy, and independently trained local-development candidates; E2 has a fail-closed iso-parameter/fractional plan plus replicated CUDA kernel receipts; E3 has a staged data/objective plan with exact mixture arithmetic and causal promotion gates; and exact 250M model/run/lineage contracts are executable.

Verification: **67 CPU tests pass**, including compressed tokenizer artifact/hash/determinism, replicated full-stack receipts, and current signal-propagation receipt hashes; E0/V5 receipts plus E1/E2/E3 plans reproduce, and package import boundaries pass. A 20-seed E0 certificate sweep is 20/20 PASS. The repository CUDA environment is operational with PyTorch 2.11.0+cu128 on the RTX 4050; a 2048² FP32 matmul probe measured 4.85 TFLOP/s CUDA versus 0.211 TFLOP/s CPU. Three replicated bf16 attention receipts and three randomized exact-stack receipts pass correctness/backend checks through 4k context. Paired five-seed CUDA/three-seed CPU initialization canaries plus a separate three-seed native-4k CUDA replication support `1/sqrt(2L)` residual-output scaling by sharply reducing residual growth and gradient imbalance. Wide/shallow remains the local execution winner, but no shape is promoted without cognition evidence.

Next action: **external promotion inputs are still required.** The independent local tokenizer tournament strengthens the 24k planning center (+0.85% held-out tokens versus 32k for 7.34M fewer embedding parameters), but its legacy/local corpus cannot authorize E1. An independent custodian must create T2 outside Git and supply a representative hash-bound tokenizer corpus. Then repeat 16k/24k/32k static audits and run matched P35 comparisons. The target TPU stack must also prove an efficient native GQA path or use a measured alternative. Do not start the 250M main run.

Required sequence: E0 → P35 tokenizer/architecture/data screens → M102 replication → freeze review → target-TPU canaries → V5-A main run.
