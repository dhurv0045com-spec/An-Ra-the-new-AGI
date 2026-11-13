# ELENDRIL V8 — Design Decisions and Implementation Contract

**Date:** 2026-11-14 (Asia/Calcutta)
**Base commit:** `0c9cb284d2c8c949e17ac342959c6f2bf351bd30` (`origin/Elendril`)
**Scientific parent:** `724af4cb508173e0eb3f4397d5dcb4a42a66712f` (V7.17)
**Implementation branch:** `codex/v8-mechanism-lab`

## 1. Core Architectural & Experimental Decisions

1. **S20 minimum backbone preserved:** All research runs use S20 (`width=512`, `layers=7`, `query_heads=8`, `kv_heads=2`, `head_dimension=64`, `ffn_width=1408`, `context_length=640`, `vocabulary_size=260`, `19,871,616` parameters). Tiny CPU models (`<= 100k` parameters) are restricted to mechanical unit/integration tests. No S3/S8/S9 research runs are admitted.
2. **Pre-optimizer model factory & shared backbone initialization:** For any `init_seed`, the backbone (`v5_model.core.initialize(base_spec, init_seed)`) is constructed first, then optional heads are initialized in a separate `torch.random.fork_rng()` seeded deterministically by `(architecture_id, init_seed)`. Backbone initial tensors are bit-for-bit identical across `B0`, `J`, `F`, `P`, and `F-cap`.
3. **Separated controls and causal contrasts:**
   - `data-control`: `B0/D0` (blocked, 9472 samples, seed 606) vs `B0/D1` (uniform, 3584 coverage-controlled worlds across 7 families).
   - `head-screen`: `J/D1` (`+9,747` params, joint 19-way linear head) vs `F/D1` (`+10,260` params, factorized `[4]` op head + `[16]` val head, 19-way joint decoding and algebraic null verification).
   - `reader-screen`: `P/D1` (`+76,309` params, pointer attention over public `A–P` tokens + sigmoid gate initialized at bias `-2.0`) vs `F-cap/D1` (matched residual SiLU bottleneck width `r=64`, `+76,372` params, `0.08%` delta <= `1%`).
   - `replication`: fresh seeds `(101, 211, 307)` on the best eligible candidate and exact relevant comparator, plus owner-supplied confirmation seed.
   - `archive-diagnostic`: conditional on genuine V7.17 FULL archive custody; defaults to `PARENT_WEIGHTS_UNAVAILABLE` when absent.
4. **Actor–Truth Firewall:** Actors receive only `PublicInput(text_utf8, query_available, memory_available, action_budget, observation_version)`. Truth, channels, hidden payloads, world IDs, task families, and gold hop counts are structurally excluded.
5. **Exact Optimizer & Schedule Lock:** Verified from `v5_training/optimizer.py` and `foundation_training.py`: `AdamW(lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)` with semantic `no_decay` grouping, cosine decay to terminal ratio `0.1`, gradient clipping `1.0`, `FP32`, logical `batch_size=16`, `micro_batch_size=2`, `4` epochs.
6. **Inactive Head Gradient & Optimizer Contract:** On batches lacking any `ANSWER` target, `val_head` / pointer / bottleneck parameters legitimately receive `grad is None`. Expected active parameter sets are declared per-batch/per-objective rather than injecting synthetic zero gradients, and optimizer step validation checks per-parameter step invariants consistently with PyTorch AdamW behavior and resume.
7. **24-Scenario Ledger & Conditional Stages:** All 24 scenarios (`F01..F04, G05..G08, R09..R12, E13..E16, A17..A20, C21..C24`) have explicit machine-readable statuses (`NOT_IMPLEMENTED`, `IMPLEMENTED_NOT_RUN`, `PREREQUISITE_NOT_QUALIFIED`, `EXECUTED`). Conditional W8 modules (`replay.py`, `memory.py`, `transfer.py`) and recurrence `U` refuse admission unless prerequisite qualification artifacts are supplied and verified.
8. **Exports & Notebook Custody:** V8 uses `ELENDRIL_V8_ANALYSIS.zip`, `ELENDRIL_V8_FULL.zip`, and `FINAL_STATUS.json`. The Kaggle notebook defaults to `RUN_STUDY=False`, `PHASE="data-control"`, `MINUTES=330`, pinned to an immutable 40-hex commit SHA.
