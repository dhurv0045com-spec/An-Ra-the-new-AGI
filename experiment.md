# Cymek 500M campaign-engine experiments

Branch: `cymek-500m-readiness`. Single real chain under test:
documents -> manifest -> pack -> sampler order -> layout -> exact-token
microsteps -> accumulation -> one optimizer update -> trainer state machine
-> checkpoint transactions -> resume verification -> campaign receipt.

## Machine (do not exceed without asking)
- CPU: AMD Ryzen 7 170, 8C/16T @ 3.2 GHz — throttle with
  `torch.set_num_threads(8)`, `OMP/MKL_NUM_THREADS=8`, BelowNormal priority.
- RAM: ~15 GB usable.
- GPU: NVIDIA RTX 4050 Laptop, 6 GB. `torch.cuda.is_available() == True`
  via `.venv-cuda` (torch 2.11+cu128). CPU venv: `.venv` (torch 2.13+cpu).
- Device switch for the suite: `ANRA_TEST_DEVICE=cuda` (default `cpu`).

## Completed (measured)

### E1 — Contract suite, throttled CPU: 33/33
`python tests/test_production_entry.py` with 8 torch threads, BelowNormal.
Covers: exact completion, determinism, over-budget fail-closed, 5
resume/identity guards, 4-microstep full-update proof, partial-tail exactness,
milestones, rotation, recovery cadence, LR wiring incl. no-rewarm, precision
contract, TPU fail-closed, contamination commitment, mode labels, certificate
completeness, banned-symbol scan, helper purity, compressed e2e
(fresh -> recovery -> milestone -> stop -> resume -> partial -> complete),
session COMPLETE/TIMEBOX/RESUMABLE, already-complete, identity freeze,
supply accounting.

### E2 — Contract suite, RTX 4050: 27/33 (6 environmental OOM, 0 code failures)
Same suite with `ANRA_TEST_DEVICE=cuda`. All 6 failures are
`torch.OutOfMemoryError` on the frozen 131,072-token full update, in fresh
processes too — not fragmentation.
Measured: one 32,768-token microstep peaks at ~5 GB transient
(3.2 GB full-vocab FP32 logit upcast in `causal_lm_loss` + 1.6 GB bf16 +
grads/activations) vs 6 GB total shared with the OS. Conclusion: the frozen
microstep does not fit a 6 GB laptop GPU. Data-center GPUs/TPUs fit it;
no production-code change was made to accommodate test hardware.

### E3 — CUDA determinism probe: PASS, 2.8 s for two campaigns
Two fresh identical campaigns on CUDA: losses and per-update receipts
bit-identical. Small-model GPU path is deterministic in practice here.

### E4 — Interrupted == uninterrupted, byte-for-byte: PASS (CPU)
Stop after update 1 with `max_updates=1`, resume to completion, compare
against an uninterrupted run: `model.bin`, `optimizer.bin`, `rng.bin`,
`cursor.json`, `ledger.json` hashes identical.

### E5 — Checkpoint rotation: PASS (CPU)
Three publishes with no milestones/recovery configured: store ends with
exactly the head generation; older generations pruned; head never orphaned
(`prune()` refuses to drop LATEST).

### E6 — Session timebox: PASS (CPU)
`max_session_minutes=0.000001` stops after update 1 with TIMEBOX/RESUMABLE;
a second `run_500m_session` call resumes to COMPLETE. Milestone files written
once, never duplicated; `SESSION_RECEIPT.json` + `HEARTBEAT.json` present.

### E7 — Regression suites (CPU venv): all green
`test_v5_training` (7), `test_v5_step_schedule_trainer`,
`test_v5_production_backend`, `test_v5_runner`, `test_v5_checkpoint_adapter`,
`test_v5_data`, `test_v5_data_pipeline`, `test_v5_stream_cursor`,
`test_v5_durability_canary`, `test_v5_stream_resume` — 70+ tests pass.

### E8 — Canary artifact refresh
`artifacts/v5/training_transaction_canary.json` regenerated via
`python -m v5_training.transaction_canary`. Proven by diff: the ONLY change
vs the committed artifact is `implementation_sha256` (pre-existing `prune()`
addition to `checkpoint.py`); all behavior/checks identical.

### E9 — Bugs caught by execution (both fixed, both covered by tests)
1. Window real-token count != loss supervised count (BOS/segment-starts
   excluded by the loss). Fix: predict the supervised total with the loss's
   own keep rule; backend cross-checks and fails closed on drift.
2. Resume published against the grandparent (`writer fence rejected stale
   parent`). Fix: `train(..., resume_parent_sha256=<restored head>)`.

## Planned (not run)
- **P1 — Microstep memory vs width/vocab.** Measure peak transient per
  microstep for wider models; feeds PRE500M memory-fit evidence. Needs >6 GB GPU.
- **P2 — bf16 vs fp32 trajectory.** Same seed/budget on CPU-fp32 vs CUDA-bf16;
  compare loss curves and final param hashes. Expect small divergence; quantify.
- **P3 — Multi-session soak (CPU).** 500k-token budget across many
  stop/resume cycles with mixed MANUAL_BOUNDARY/TIMEBOX stops; assert ledger
  == budget and single head at end.
- **P4 — PRE500M TPU certification.** Collectives, 8-replica topology, memory
  fit, bf16 execution on real TPU. BLOCKED: needs TPU hardware.
- **P5 — Throughput curve.** Tokens/sec vs microstep size on a data-center
  GPU; validates the 8x4x4096 topology choice. Needs data-center GPU.
- **P6 — Full repo marathon suite.** Whole-repo pytest at exact HEAD.
  Skipped deliberately (laptop load); run on a bigger machine or overnight.
- **P7 — Real-corpus production run.** Requires DATA_READY 500M first-party
  corpus + frozen contamination benchmarks. BLOCKED by supply (honestly
  reported by `materialize_first_party`: DATA_NOT_READY).

## Load rules for this machine
1. Check specs and state the cost estimate BEFORE any training-step run.
2. Default to cheap tests; heavy runs only on explicit approval, throttled,
   one at a time.
3. Never modify production math to fit test hardware.
