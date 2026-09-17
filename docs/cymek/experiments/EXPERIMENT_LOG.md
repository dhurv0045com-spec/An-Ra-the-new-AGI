# Cymek experiment log (chronological, append-only)

## 2026-09-07 — closure cycle (branch cymek-500m-readiness)
- Bucket-pure execution, durable milestones, mixture scheduler, eval
  hooks, XLA adapter (pending), persistent mirror, import-plane fix.
  See experiment.md E1–E11 and
  artifacts/v5/cymek_500m_closure_test_receipt.json.

## 2026-09-08 — CYR-GPU-005 freeze cycle
- CYR-GPU-004 SUPERSEDED_BEFORE_EXECUTION (fatal fork-contract defect
  confirmed against live run_v4: independent acquisitions per LR, dead
  parent snapshot, no future-stream identity). See
  CYR-GPU-004/SUPERSEDED.md.
- Production XLA accumulation defect fixed (per-microstep all-reduce of
  the ACCUMULATED buffer scaled early microstep gradients by powers of
  the replica count); one collective at the accumulation boundary now,
  with a CPU oracle + negative regression. Hardware status stays
  IMPLEMENTED_PENDING_PRE500M_TPU.
- CYR-GPU-005 executable freeze (COMMIT A) + hash-bound preregistration
  (COMMIT B). RUN_READINESS=true. GPU evidence: NONE YET.

## 2026-09-12 — R1C launch failure root-caused and repaired (Priority Zero)
- Exact campaign failure audited: arm `S1_MASK_8192` aborted with
  `abort CLIP_BREACH: post-clip global norm 1.0000042915344238 exceeds 1.0`
  — float32 reduction-order noise (~4.3e-6 at 4.13M params) between the
  fused clip path and the certificate's recomputed norm, against a 1e-6
  tolerance. Implementation numerics only; no scientific information.
- Repair: `_NORM_TOLERANCE`/`_CLIP_TOLERANCE` 1e-6 -> 1e-4 (derivation
  documented). End-to-end engineering preflight added
  (tests/test_v5_cyr_gpu014_r1c_e2e_preflight.py): drives the ACTUAL
  campaign executable through all six arms on CPU with tiny fixtures;
  reproduced the failure before the repair, completes after.
- State: R1C_READY_FOR_OPERATOR_CUDA_RUN (RUN_READINESS_V4.json; executable
  6653b4ce — v3's binding predated the tolerance repair and is superseded).
- Constants single-sourced (CLIP_NORM_TOLERANCE in v5_training/step.py);
  v5_training code changed after the test receipts' tested commits, so the
  exact-head receipt meta-checks are STALE-BY-DESIGN until the next full
  suite run refreshes them (documented, not hidden — repo precedent).

## CYR-GPU-001 — status: SUPERSEDED_BEFORE_EXECUTION
- Preregistered design + tournament runner committed; never executed;
  superseded by the 002 rebuild (shared contexts, free-gen primary,
  dose resolver).

## CYR-GPU-002 — status: SUPERSEDED_BEFORE_EXECUTION
- Tournament stack + cycle test receipt exist (artifacts/v5/
  cyr_gpu_002_test_receipt.json); superseded by 003 without operator
  execution.

## CYR-GPU-003 — status: SUPERSEDED_BEFORE_EXECUTION
- Pre-execution audit found 13 defects (3 blockers) before any GPU
  execution; superseded by 004.

## CYR-GPU-004 — status: SUPERSEDED_BEFORE_EXECUTION
- Corrected runner still launched independent acquisitions per LR and
  never restored the parent snapshot; killed by the 005 audit. See
  CYR-GPU-004/SUPERSEDED.md.

## CYR-GPU-005 — status: PREREGISTERED, PENDING_OPERATOR_EXECUTION
- Shared-parent fork campaign (HIGH / LOW / FIXED_TIME / HYST from ONE
  G90 parent per seed), hash-bound preregistration, RUN_READINESS=true.
- No outcomes yet. Live results append below with dates when the
  operator returns the bundle. Do NOT edit preregistered sections
  after execution starts.

## CYR-GPU-011 / CYR-GPU-012 — status: both EXECUTED; discrepancy ledgered, cause UNKNOWN
- 2026-09-17 (branch codex/cyhex-integrity-audit). Both original bundles
  verify against their tracked receipts (SHA-256, byte count, exact member
  list, JSON parse). The duplicate-member verifier defect repaired in this
  branch does not affect either archive.
- Machine-checkable difference ledger
  (CYR-GPU-012/CYR011_VS_CYR012_LEDGER.json, schema
  `anra-cyr-repro-ledger/v1`): 15 fields classified. PROVEN IDENTICAL:
  torch 2.11.0+cu128, model_seed 3301, order_seed 4701, batch_rows 64,
  semantic stream digest (identity, not row contents), initial_l2
  46.88997268676758 (norm only), g90 null. PROVEN DIFFERENT: frozen
  executable, GPU (Tesla T4 vs RTX 4050 Laptop), updates 8,081 vs 18,000,
  row presentations 517,184 vs 1,152,000, g50_confirm_update 2,200 vs null
  (outcome difference, not itself causal evidence), final_model_sha256.
  UNKNOWN: initial_tensor_sha256 (replayed below), kernel_selection (no
  kernel trace recorded in either bundle).
- Initial-tensor replay (CYR-GPU-012/CYR011_VS_CYR012_INITIAL_REPLAY.json):
  frozen-source CPU rebuild under seed 3301 matches the recorded anchor
  exactly (norm_delta 0.0; byte-stable repeat digest
  `e263801d…504df04`; seed 3302 digests differently). Verdict
  INITIALIZATION_REPLAYS_MATCHING_NORM — supports, does not prove,
  identical historical initialization; no initial-tensor hash exists in
  either bundle.
- Frozen-executable diff `0a97257e..1a1624e` touches only probes/stopping/
  operator/tests — model, optimizer, and data math are byte-identical.
  First OBSERVED divergence is bounded at the first scheduled evaluation
  (update-200 traces: dev_controller 9.375% vs 4.6875% with near-identical
  relative displacement 0.2367 vs 0.2280); no cause identified.
- Remaining direct test of the environment hypothesis is a controlled
  cross-GPU pair run (T4 vs RTX 4050, update 200) — requires owner
  authorization; not launched. No causal ranking claimed; no retraining
  performed; production/frontier gates unchanged.

### Interpretation correction — applies to the entry above
- Per `REPLAY_SOURCE_AUDIT.md`, historical initial-tensor identity remains
  UNKNOWN after norm replay; kernel selection is not the only unknown.
- Byte-identical source files do not establish identical runtime math;
  similar displacement magnitudes do not establish similar trajectories.
- A controlled cross-GPU comparison is one possible environment-hypothesis
  test, not the only direct test. No such run was launched by this audit.
