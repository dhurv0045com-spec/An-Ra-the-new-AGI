# Cymek 500M closure-cycle experiments

Branch: `cymek-500m-readiness`. Machine: Ryzen 7 170 (8C/16T), ~15 GB RAM,
RTX 4050 Laptop 6 GB. CPU venv `.venv` (torch 2.13+cpu); CUDA venv
`.venv-cuda` (torch 2.11+cu128, `tokenizers` present). Training runs:
6 torch threads, `OMP/MKL_NUM_THREADS=6`, foreground only, one test at a
time, RAM checked before/after (9–11 GB free throughout; no swap event).
Suite device switch: `ANRA_TEST_DEVICE=cuda` (default `cpu`).

## Completed (measured)

### E1 — Bucket-lane unit suite: 10/10 (CPU, seconds)
`python tests/test_v5_bucket_cursor.py`. Single-bucket physical windows,
split-window resume prefix equality, missing-bucket fail-closed, lane
exhaustion, epoch determinism + reproducibility, mixed-cell rejection,
segregated family lanes, remainder accounting, cursor round-trip, receipt.

### E2 — Mixture + lifecycle suite: 7/7 (CPU, seconds)
`python tests/test_v5_mixture.py`. 500M allocation exact
(325M/100M/75M), deficit-scheduler exactness on unit steps, convergence
bound, purity/resumability, offline-demand == live-assignment proof,
lifecycle order + evidence gates.

### E3 — Durability/mirror/eval/XLA/topology suite: 9/9 (CPU, seconds)
`python tests/test_v5_durability_contracts.py`. Mirror round-trip,
interrupted copy, corrupt mirror, stale/foreign pointers, train/eval
collision, eval schedule + ingest idempotence + tamper rejection,
promotion delegation, replica sharding/divisibility/padding, XLA
fail-closed paths.

### E4 — Equivalence suite: 2/2 (CPU torch, ~1 min)
`python tests/test_v5_training_equivalence.py`. Activation checkpointing
off-vs-on: bit-identical logits AND gradients, inference path unchanged.
Replica-sharded (3 unequal replicas) vs single-shot global accumulation:
same denominator → loss/grads/params equal within 1e-5 (kernel tiling
justifies tolerance, not bit equality).

### E5 — Production-entry contract suite: 49/49 (throttled CPU + CUDA legs)
`python tests/test_production_entry.py`. Prior 33-test coverage carried
over and extended: exact single-bucket microstep shapes (64/32/16/8 rows,
width == bucket, frozen plan certified), supercycle order, partial-tail
pad-path receipts, durable cross-session milestone protection (object
survives rotation, receipt verifies, restore works), dangling/tampered
milestone rejection, rotation, recovery, LR wiring + no-rewarm, precision
per device, TPU fail-closed, contamination commitment + content binding,
mixture/freeze/provenance production gates, frozen-mixture end-to-end
(65/20/15 consumed exactly per plan), shortfall DATA_NOT_READY, epoch
replay (multi-epoch merge) + forbidden-replay failure, certificate,
banned-symbol scan, EOS packing contract (one-token doc → 2 supervised;
exact fill; ragged tail), e2e, sessions, V5A exact count, bf16 diagnostic,
soak, exact-head receipt.

### E5b — Cognition nested scheduling + mixture resume: PASS
Five planned cells (one per nested sub-family) consumed exactly;
interrupted vs uninterrupted runs byte-identical incl. cursor/ledger.
Demand planner proven to restart from restored counters (unit proof),
closing a planner-vs-executor resume divergence found live.

### E6 — Multi-session soak: PASS (throttled CPU, one shot)
`test_multi_session_soak_state_machine`. Budget 264144 across 3 sessions
(TIMEBOX, TIMEBOX, COMPLETE): exact final tokens + ledger, 3 losses,
milestone objects preserved + verified, no duplicate milestone files,
LR equals schedule at both resume points, epoch 0 / zero replays.

### E7 — P2 bf16-vs-fp32 diagnostic: delta 5.1e-5 (CUDA + CPU, ~2 min)
Same seed/docs/budget 4000: CUDA-bf16 loss 10.125044, CPU-fp32 loss
10.124993. Diagnostic only; no cross-precision hash equality claimed.

### E8 — V5A_250M instantiation: PASS (CPU, one shot)
`initialize(V5A_250M)` → 250,216,960 live parameters == receipt total.
~1 GB transient, freed after.

### E9 — Relevant regression suites: all green (CPU venv)
training (7), step/schedule/trainer, production_backend, runner,
checkpoint_adapter, data, data_pipeline, stream_cursor, stream_resume,
durability_canary, contracts (incl. import boundaries + transaction
canary), model, objectives, foundry, mutation_canary, cli, status,
launch_readiness, evaluation, evaluation_protocol, promotion, registry,
redteam, subject_v2, tokenizer — 200+ tests, 0 failures.

### E10 — Bugs caught by execution (all fixed, all covered)
1. Window real tokens != loss supervised tokens (BOS/segment-starts) →
   predict-with-loss-rule + backend cross-check.
2. Resume published against grandparent (stale writer fence) →
   `resume_parent_sha256`.
3. Checkpoint cursor-component check compared dict-with-tuples against
   JSON lists → compare through canonical JSON.
4. Overwrote tracked `v5_data/mixture.py`; restored and merged (kept
   `allocate`/`bucket_plan`/exports, added scheduler).
5. Moved `_load_corpus/_load_tokenizer` to `v5_data/corpus_loading.py`
   (fixes the pre-existing v5_data→v5_training plane violation; training
   modules re-export, all importers verified).
6. Test-pack bucket drift (BPE rate misestimated twice) → measured
   calibration (`_sized_text`, exact-full rows).

### E11 — Canary artifacts regenerated (behavior-identical, hash-only diffs)
`training_transaction_canary.json` (covers state.py + checkpoint.py
changes: resume-parent param is additive, prune auto-union, cursor
compare normalization — canary semantics unchanged, PASS).

## OOM classification (corrected per audit — no overclaim)
- RTX 4050 6 GB, frozen 32,768-token microstep, this implementation:
  **DOES_NOT_FIT** (measured ~5 GB transient vs 6 GB shared; fails in
  fresh processes, not fragmentation).
- Larger data-center GPUs: **UNMEASURED**.
- TPU: **UNMEASURED** (XLA adapter reports IMPLEMENTED_PENDING_PRE500M_TPU).
- No production math was changed to fit test hardware.

## CYR-GPU-001 cycle (research tournament preparation, PENDING execution)

- Audited Arkenstone@4911b84 (ARK-011 confirmed UNEXECUTED, notebook never
  run) and BRAMASTRA@90ee31a live; matrix in
  `docs/cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md`.
- Hardened Cymek this cycle: complete-answer contract tests (EOS-final
  segments, EOS supervised, generation EOS/cap stops), XLA selection
  through run_campaign (fail-closed, receipt-bound, rank-sharded math),
  auto-mirror + mirror recovery in run_campaign/run_500m_session.
  Prior-cycle work verified intact, not redone: bucket lanes, mixture
  scheduler, durable milestones, persistent mirror, eval hooks, corpus
  loading plane, activation checkpointing (full regression re-run green).
- Built version-controlled tournament (pure logic + torch executors
  split for import planes): exact-sized counterfactual worlds, matched
  pair/shuffled sampler, fork-based LR tournament (HIGH/fixed/state/
  hysteretic/MID), displacement diagnostics, sustained metrics,
  heuristic baselines, redteam gates, hardware resolver, failure-proof
  packaging, Colab guard (smoke default).
- Preregistered CYR-GPU-001 (PLAN/THREATS/README + hash-bound
  PREREGISTRATION.json) + thin 3-cell notebook.
- Local validation only: unit suites, tiny CPU smoke, notebook JSON +
  cell-compile checks, receipt schema checks. NO heavy/local/GPU/TPU
  training this cycle. Status: PENDING_OPERATOR_EXECUTION.
- Full validation (unit + entry suites) deferred to Colab CELL 0 gate;
  repo closure receipt goes stale-by-design on new code and refreshes
  from Colab evidence next cycle (documented, not hidden).

## Planned / blocked (not run)
- **P1** microstep memory vs width (needs >6 GB GPU).
- **P4** PRE500M TPU certification (needs TPU hardware).
- **P5** throughput curve (needs data-center GPU).
- **P6** whole-repo marathon (all *relevant* suites ran, E9; research
  stacks e0/e1/e2/e3, remote, and hardware benchmarks untouched —
  out of scope for this closure cycle, stated not implied).
- **P7** real-corpus production run (BLOCKED: DATA_NOT_READY supply).

## Load rules used
Specs checked before every training run; one test at a time; foreground
so aborts kill workers; no uncommitted-background batches after the first
orphan incident; V5A-size loads run once.
