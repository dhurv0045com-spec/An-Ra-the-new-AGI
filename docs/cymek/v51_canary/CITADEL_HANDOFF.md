# CITADEL HANDOFF (independent audit package)

Task 3 requests independent audit. **No self-declared approval.** Everything below is machine-readable under `experiments/V5_1_CANARY/`.

## What Citadel should verify

1. **Preregistration integrity** — `PREREGISTRATION.json` committed before the final result; amendment 1 (data volume) documented with its fail-closed trigger; thresholds unchanged across amendments.
2. **Data instrument** — re-run `anra_v5.v51_canary_data.build_dataset(seed=20260913, worlds_per_family=2000)`; confirm split hashes match `receipts/DATA.json`; confirm zero cross-split collisions (`contamination_screen`); confirm worst shortcut baseline < 0.35 (`shortcut_baselines`); confirm reference solvers equal rendered answers.
3. **Split discipline** — group-level (latent-world) splitting; sealed split generated first, hashed, and consumed only in `receipts/FINALIZATION.json`.
4. **Parameter accounting** — `tools/next_core_compute_model.py`: Rung A = 10,227,456, Rung B = 42,092,544, V5-A = 250,216,960; instantiated models equal analytic counts exactly (receipts/MODEL.json, PREFLIGHT.json).
5. **Production path** — the runner exercises `v5_data` (manifest → tokenizer → pack → stream), `v5_model.core`, `v5_objectives.causal_lm_loss`, `v5_training.optimizer.build_adamw_optimizer`, `ProductionTrainingBackend.step` (clip 1.0 + 1e-4 certificate), `CheckpointStore` atomic publication, `certify_update` per update. No parallel training path exists.
6. **WSD execution** — `receipts/TRAINING.json` trace: `lr_expected == lr_actual` at every update, zero rewarm events, all three phases exercised, frozen 5B `lr_at` domain probes recorded.
7. **Exact resume** — `receipts/RESUME.json` + `tests/test_v51_canary.py::test_fresh_process_exact_resume_is_bitwise`: fresh subprocess resume reproduces model.bin/optimizer.bin/ledger/cursor/scheduler bitwise (parent pointer excluded as documented). Ambient-RNG pinning (`torch.manual_seed`) is the fix for cross-process nondeterminism — verify it is present in `make_backend`.
8. **Corruption/incompatibility rejection** — corrupted component byte → restore raises; identity drift → scan FAIL_CLOSED; injected crash preserves last-known-good and requires documented staging recovery.
9. **Sealed discipline** — dev used for iteration; sealed consumed once at finalize; verify `FINALIZATION.json` seals hash matches the prepare-time hash.
10. **Claim ceiling** — the only allowed positive claim is the integration-qualification sentence in FINALIZATION.json. No cognition, production-readiness, 500M, or output-space claims.

## Not in scope for Citadel here

Cognition qualification of the resulting checkpoint (Triquetra's authority, via readiness v2); production-corpus readiness (separate blocker); R1C/ARK-020 interpretation.
