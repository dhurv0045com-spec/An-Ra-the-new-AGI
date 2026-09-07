# Agent brief: CYMEK 500M closure cycle handoff

Branch: `cymek-500m-readiness`. Date: 2026-09-07.
Target end state: CYMEK SOFTWARE READY_FOR_CITADEL_REAUDIT.
This is NOT the 500M training run. No training started. No merges.

## PROVEN LOCALLY (executed, receipted in experiment.md)
- Bucket-pure microsteps: every executed row width == requested frozen
  bucket; frozen per-replica counts certified on full rows (64/32/16/8);
  sparse/partial tails take an explicit pad path, never silent certification.
- 20-microstep supercycle order from cumulative tokens (stronger than stored
  state: nothing to drift); accumulation math = global eligible-token
  denominator; replica-sharded == single-shot within 1e-5 on unequal fixtures.
- Canonical WSD schedule, no rewarm across resumes (asserted at two resume
  points); exact partial tails, no overshoot; fresh/resume from LATEST with
  full identity-drift rejection (docs, seed, budget, specs, source commit,
  lanes, mixture plan).
- Durable milestones: persisted in-store, survive sessions, rotation cannot
  prune them; receipts self-verify against restored checkpoints; dangling and
  tampered receipts rejected.
- Persistent mirror adapter: verified copy → fenced pointer → reread;
  interrupted/corrupt/stale/foreign failures all fail closed.
- Mixture: exact 65/20/15 allocation (325M/100M/75M at 500M), deficit
  scheduling with planner==executor proof, shortfall fails DATA_NOT_READY,
  epoch replay merges exactly (or forbidden → DATA_NOT_READY).
- Data contracts: lifecycle states to RUNNABLE, exact+near dedup before
  split (frozen policy), benchmark-CONTENT-bound contamination identity,
  strict raw-source provenance in production, EOS/packing boundary tests.
- Tokenizer: artifact-backed identity required, freeze SHA bound, no
  fallback in production.
- Eval boundary: Cymek exposes schedule/manifests/interface/ingestion;
  go/no-go stays in v5_promotion; Triquetra owns cognition. No train/eval
  collisions possible without abort.
- Target preflight for V5A_250M (exact 250,216,960 params instantiated and
  counted); activation checkpointing bit-identical off-vs-on.
- Import planes enforced by test (fixed one live violation + one
  pre-existing v5_data→v5_training violation by moving corpus loading to
  the data plane with re-exports).
- Soak: 3 sessions (TIMEBOX/TIMEBOX/COMPLETE), exact budget + ledger,
  milestones preserved + verified, no rewarm, no replay.
- bf16-vs-fp32 one-update delta 5.1e-5 (diagnostic, no equality claimed).
- 49/49 entry tests + 200+ relevant regression tests green; head-bound
  test receipt (`artifacts/v5/cymek_500m_closure_test_receipt.json`).

## IMPLEMENTED BUT UNPROVEN ON TPU
- XLA replicated adapter (discovery, frozen-topology gate, rank-0 writer,
  SUM collective + single synchronized step): imports and fail-closed paths
  tested, status IMPLEMENTED_PENDING_PRE500M_TPU. Zero TPU evidence present.
- Real data-parallel execution (8 replicas as 8 executors): only the
  single-device global emulation + the sharding oracle exist locally.

## BLOCKED BY DATA
- 500M first-party supply: DATA_NOT_READY (honest audit preserved).
  No filler generated. No corpus built or processed this cycle.

## BLOCKED BY EVALUATION
- Sealed/fresh datasets, production scoring policy, Triquetra evaluators:
  hooks + identities only. Nothing staged, nothing consumed.

## BLOCKED BY EXTERNAL IDENTITY
- `launch_readiness.json` untouched: main_training_authorized=false,
  production_launcher_implemented=false, E1–E6 pending, missing external
  identities outstanding. Correct as-is; flips require real gates.

## NOT IMPLEMENTED (deliberately)
- 5B-run final-partial arithmetic in the campaign path (generic tails only).
- Cognition qualification, scoring research, remote infrastructure.

## Launch checklist (unchanged, in order)
1. 500M unique first-party tokens → READY + frozen benchmarks.
2. PRE500M TPU certification on real hardware.
3. Soak-at-scale + divergence acceptance.
4. Staged sealed/fresh eval with scoring policy.
5. Subset dry-run of the full campaign config.
6. Only then: session loop to 500M, watching receipts.

## Machine rules (this laptop: Ryzen 7 170, 15 GB, RTX 4050 6 GB)
- Verification machine, not training machine. One test at a time,
  foreground, throttled, RAM checked. 4050 verdict: small tests fast;
  frozen 32k microstep DOES_NOT_FIT (measured); larger GPUs UNMEASURED.
- Never modify production math to fit test hardware.

## Map
- Engine: `v5_training/production_entry.py`.
- Lanes/cursor/mixture/lifecycle: `v5_data/bucket_cursor.py`,
  `v5_data/mixture.py`, `v5_data/lifecycle.py`, `v5_data/corpus_loading.py`.
- Checkpoints/mirror/receipts: `v5_training/checkpoint.py`,
  `persistent_store.py`, `test_receipt.py`. XLA: `xla_adapter.py`.
  Topology: `topology_map.py`. Eval hooks: `v5_evaluation/campaign.py`.
- Proof: `tests/test_production_entry.py` (49) + unit suites.
- Measurements: `experiment.md`.
