# ESOES Status

Updated: 2026-08-31

Branch phase: **GROUND BLUEPRINT v0.4 — E0 DEVELOPMENT CONTRACT REPAIRED; TRAINING SYSTEM OPEN**

Canonical research blueprint: [`V5_MASTER_BLUEPRINT.md`](V5_MASTER_BLUEPRINT.md)

Research source ledger: [`EVIDENCE_BASE.md`](EVIDENCE_BASE.md)

Large V5 training authorized: **NO**

Frozen V5 spec exists: **NO**

Provisional V5-A: **250.22M dense, 26×896, 14Q/7KV, FFN 2368, affine QK norm, 4k full attention, 24,576 byte-fallback vocabulary, 5.0B audited tokens.** Exact shape, tokenizer, data fraction, objective, LR, batch, and schedule remain experiment-gated.

Current evidence says the program should improve data quality, token sufficiency, causal query supervision, and behavioral checkpoint selection before scaling parameters.

Executed through this phase: VNext contamination is removed; EXP evidence is corrected; E0 now randomizes semantic state serialization, places every query between/after events, covers intermediate/rollback/precedence queries, balances eight competing target values, holds out rule structures across splits, and fail-closes on six pooled positional/lexical/bag-of-words controls. It reports difficulty curves and has raw/assisted/intervention-dependence plus replication contracts; sealed custody fails closed. E1 has an artifact-bound static audit/Pareto/matched-budget harness, a real V4 baseline, truncation proxy, and independently trained local-development candidates; E2 has a fail-closed iso-parameter/fractional plan plus replicated CUDA kernel receipts; E3 has a staged data/objective plan with exact mixture arithmetic and causal promotion gates; and exact 250M model/run/lineage contracts are executable.

Verification: **84 CPU tests pass**. Exact-head GitHub Actions is green at `3f8f80a` after replacing platform-dependent raw JSON receipt hashes with canonical semantic hashes and adding CRLF/LF regression coverage. The repaired E0 receipt is schema 2 / generator 0.4.0: pooled bag-of-words fell from the reproduced false-green 81.77% to 4.17%, and lexical overlap from 71.09% to 10.16%, versus a 10.37% casewise null; all six state controls satisfy null + 10 points. Compressed tokenizer artifact/hash/determinism and perturbation-sweep receipts, replicated full-stack receipts, and current initialization/QK/precision/RoPE/real-update/cursor receipt hashes also pass; E0/V5 receipts plus E1/E2/E3 plans reproduce, and package import boundaries pass. The repository CUDA environment is operational with PyTorch 2.11.0+cu128 on the RTX 4050. Paired CPU/CUDA canaries support `1/sqrt(2L)` residual scaling, QK projection-scale control, native-4k RoPE implementation conformance, close local BF16/FP32 forward/backward parity through native P35 2k context, real-update/save-resume wiring with FP32 master state across all three E2 shape arms, and deterministic content-addressed cursor continuation. Wide/shallow remains the local execution winner, but no shape, learned-quality, 4k-V5, target-TPU, or long-run precision claim is promoted without the relevant evidence.

Next action: **certify the model↔E0 scoring adapter before spending P35 training compute.** Deterministic oracle/broken scorers and random-weight P35 models must expose candidate-length/tokenization/position bias across 16k/24k/32k and CPU/CUDA. Then build one atomic TrainingState checkpoint transaction joining model, optimizer, scheduler, RNG, cursor, token ledger, and identity manifests. External natural/T2 custody and a representative tokenizer corpus remain required for promotion. Do not start the 250M main run.

Required sequence: E0 → P35 tokenizer/architecture/data screens → M102 replication → freeze review → target-TPU canaries → V5-A main run.
