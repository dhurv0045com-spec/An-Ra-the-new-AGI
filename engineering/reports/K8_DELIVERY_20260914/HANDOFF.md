# K8 delivery handoff — D1–D5 with production-path evidence

Date: 2026-09-14. Agent: BRAMASTRA implementation lead. Baseline `44f1676`
(V2 chief review + D1–D5 execution order). This handoff implements the missing
phase handlers and the eight numbered corrections; it does not claim hardware
results. Prior handoffs preserved as history. Push: normal fast-forward on
`BRAMASTRA`. No local training: delivery probe observes
`optimizer_step_calls_observed: 0`; old CPU ledger 206/200 untouched.

## Chief counterexamples — failing-before / passing-after

Reran `engineering/reports/K8_V2_CHIEF_20260914/probe.py` (zero models, zero
updates) after the delivery commit:

| Check | Before (`probe-001.json` @30d0fcd) | After (this delivery) |
|---|---|---|
| E1 reserved together | 4 jobs (both slots, 2/GPU) | slot1 only (`E1-A-1701`, `E1-B-1701`); 3rd refused: exclusive occupancy |
| timeout marker | `worker_finished_after_timeout: true` (survived) | **false** (terminated, `exitcode=-15`) |
| live lease + old age | `old_live_lease_reclaimed: true` | **false** (refused regardless of age) |
| device after isolation | received `cuda:1`, visible `1` | received **local `cuda:0`**, physical `cuda:1`, visible `1` |

## D1 — Supervisor and slot execution: DELIVERED (CPU-proven)

Changed functions:
- `runner._phase_plan` — explicit ordered slots: E1 two counterbalanced slots
  (A1701+B1701, B1702+A1702); E3 slots T0-1701+T1-1702 then T1-1701+T0-1702;
  E4 slots S0-1701+S1-1702 then S1-1701+S0-1702 (reversed treatment order,
  GPU0 owns seed1701). E0/E2/E5/E6 single slots.
- `runner.run_campaign` — reserves/launches ONLY the current slot, persists
  `slot_completed`/`slot_failed` + per-job receipts, recovers individual jobs
  without repeating accepted outcomes, per-phase success predicates (E2 binds
  cases + checkpoint with zero updates; E6 is export), fenced lease token,
  `_preflight` gate (exit 2 before expensive work).
- `supervisor.CampaignLedger.reserve` — exclusive occupancy: second OPEN job
  on the same physical GPU refused even when seconds fit; different GPUs may
  overlap; sequential slots admit after close.
- `process_supervision.slot_plan_for_phase`, `physical_to_local_device`
  (cuda:N → visible N, local cuda:0), `_child_main` (visibility BEFORE torch
  import, physical preserved, UUID reported), `_run_one_in_process` (owned
  spawn Process + join/terminate/bounded re-join/kill fallback; timed_out only
  with termination status), `run_phase_concurrently` (one slot at a time,
  duplicate-physical refusal).
- `supervisor.SupervisorLease` — ownership token (`acquire` returns uuid,
  `release(token)` fenced, legacy `release()` kept); age NEVER authorizes
  takeover; liveness via non-terminating probes only (POSIX signal 0, Windows
  OpenProcess QUERY_LIMITED_INFORMATION via ctypes; indeterminate → refuse).

Delivery probe (production launcher + finite CPU doubles, no training):
`e1_slot_count 2`, sizes `[2,2]`, `slot1_reserved_together [E1-A-1701,
E1-B-1701]`, `slot2_same_gpu_while_open_refused true`,
`slot2_admits_after_slot1_close true`, `production_timeout_raised true`,
`terminated_child_wrote_marker false`, `lease_token_issued true`,
`old_live_lease_reclaimed false`, `fenced_release_rejects_foreign_token true`,
`physical_cuda1_maps_local_cuda0 true`, `worker_local_device cuda:0`,
`worker_physical_device cuda:1`. Overlap of different devices still proved via
V2 probe (`concurrent_overlap 0.406s`).

## D2 — One learning-window and checkpoint interface: DELIVERED (CPU-proven)

Changed functions:
- `learning/k8_trainer.K8Trainer` — single-window contract:
  `_require_clean_boundary` (second accumulate before finalize refused),
  `accumulate` / `accumulate_full_window` (route with per-term OWN
  denominators, ONE scaled backward, disabled-term execution refused, missing
  eligible weighted terms refused, pair rows pooled with weight checks),
  `finalize_update` (no re-division; consumes pair rows via `_apply_pair_term`
  when pair weight > 0, refuses pair rows when weight = 0 and missing pair
  data when weight > 0; steps exactly once; failed report never erases step),
  `_clear_pending`, `state_payload` (+ config/arch/RNG/profile/allocation),
  `load_state_payload` (+ `expected_allocation` live-reservation validation
  and `expected_config_identity` check; no new allowance).
- `campaigns/worker._run_e0` — explicit `training_step_full` =
  accumulate + finalize per update (both branches call it); committed/
  attempted derived from actual counters (no hardcoded 6);
  `_run_resume_in_child` — resumed continuation runs IN THE CHILD with the
  same device/profile/precision + allocation/config validation, returning
  checksum + counters (parent never continues a restore in-process).

Delivery probe (backward-only, no steps): `second_accumulate_before_finalize_
refused true`, `disabled_term_execution_refused true`,
`missing_eligible_term_refused true`, `backward_equivalent true` (max diff
0.0, token-only accumulate vs full-window), `accumulate_stepped false`,
checkpoint fields present (config/arch/RNG/allocation),
`wrong_config_identity_refused true`; E0 source evidence
(`finalize_update()` present, counters-derived, no hardcoded 6, child
continuation present). GPU parameter-change + AMP checks remain owner-launch.

## D3 — Data compiler and environment adapters: DELIVERED (CPU-proven)

Changed functions (`data/k8_bundle.py`):
- One declared canonical identity across all pools: rows carry
  `canonical_identity`; `splits.json` persists `_claimed` sets; validation
  uses stored identities (never a weaker reconstruction).
- Meta drawn from the same identity space with primary-pool exclusion
  (`exclude_canonical` over all primary claimed keys); protected families use
  concrete mechanism references.
- `_assert_public_separation` at write time (public rows never carry answer /
  target_value; hidden rule bodies stay private; tool feedback legitimacy
  distinguished from leakage).
- `_verifier_consistent` still recomputed per mechanism; inconsistent rows
  now FAIL preparation (refuse unqualified bundle).
- Manifest binds real source bytes (`source_bytes_identity` via
  `source_closure_sha256`) + file hashes + audit.

Delivery probe: `mini_bundle_valid true`, `splits_carry_declared_canonical
true`, `meta_resolved true`, `public_renderer_separation true`,
`manifest_binds_source_bytes true`, `tool_sample_verified true`.
Full 4096/256/256/128 + 256/64 preparation remains a Kaggle offline step.

## D4 — Actual E1–E5 executors: DELIVERED (production modules + double evidence)

New package `campaigns/phases/` behind `worker._dispatch_phase_executor`
(typed `JobInput`/`PhaseResult`; missing evidence → failed with actual counts;
E2 binds cases + checkpoint with zero updates — no positive-update gate).

- `phases/ops.py` — `ProductionOps` (real GPU training/generation/checkpoints;
  never executed locally) + `RecordingDoubleOps` (records init/treatments/
  streams/checkpoints/generations/budgets; zero optimizer updates).
- `phases/e1.py` — paired random init per seed, A/B weights
  (A token-only, B token/world/action/value/pair), identical per-seed stream
  from the real bundle, checkpoint fractions, free-gen eval. Doubles prove
  both slots traced, treatment weights recorded, checkpoint callbacks fired,
  actual counts returned.
- `phases/e2.py` — frozen E1 parents, policy/workspace/planner adapters,
  contradiction/inquiry/goal controls; real mini-env steps with injected
  responses; action/call/node budgets enforced; optimizer path verified zero.
- `phases/e3.py` — separate E1-B clones per child (isolation verified),
  declared replay mixture, tool acquisition + old-task eval; sampling
  receipts, `verify_tool` on real tool rows, sealed-confirmation exclusion
  from the training stream.
- `phases/e4.py` — E1-B fork independent of E3; S0/S1 gate config;
  `migrate_from_parent` zero-gate + nonzero-path + full segment/action/value
  contract checks; device-correct migration/restoration; slot ordering from
  the runner.
- `phases/e5.py` — archive_P0 trials with snapshots before choices,
  P0/P1/P_fixed compiled identities, dispatch application per method,
  anchor-stability check, fresh-confirmation choices bound before outcomes;
  rejects future outcomes, changed anchors, unapplied recipes; lineage + cost
  events verified.

Delivery probe (all via `RecordingDoubleOps`, zero steps): `e1a/b_completed
true` with actual counts `[2,2]`, checkpoint callbacks + treatment weights
recorded, `e1_both_slots_traced true`; `e2_completed true`,
`e2_zero_optimizer_path true`, `e2_evaluated_cases 2`,
`e2_no_positive_update_gate true`; `e3_completed true`, `e3_tool_verified
true`; `e4_completed true`, `e4_gates_enabled true`; `e5_completed true`,
`e5_trials 3`, `e5_archive_bound true`; `missing_evidence_fails true`.
Production GPU execution of these handlers is owner-launch; no local learning
was performed to prove them.

## D5 — Packaging and delivery review: DELIVERED (CPU-proven)

- `runner._preflight` — validates bundle (`validate_bundle`) + handler
  imports before E0; missing manifest/bundle/handlers → exit 2 (never silent
  success). Probe: missing bundle refused, valid bundle passes.
- Notebook `notebooks/bramastra_k8.ipynb` — subprocess arglists, no `!`
  shell, `RUN_DIR` retained across e0/full/summarize/export, returncodes
  checked (structural probe true on all four).
- `phases/e6.py` + `campaigns/k8.cmd_export` — canonical
  source/data/protocol/results/payload bundle (5 JSON files); verification of
  hashes, required E1-B parents for both seeds, failed-run preservation, and
  checkpoint-record completeness. Probe: `e6_status completed`,
  `e6_export_verified true`, `e6_missing_ledger_fails true`. Full `.pt`
  durability reload on owner storage remains launch-gated; export failure
  never reports campaign success (runner returns nonzero).

## Verification (actual commands run, zero training)

```powershell
$env:PYTHONPATH = (Get-Location).Path
python engineering/reports/K8_V2_CHIEF_20260914/probe.py
# -> slot1 only, terminatedWashington false marker, live lease false, local cuda:0

python engineering/reports/K8_DELIVERY_20260914/probe_delivery.py
# -> 74 keys, all expected, optimizer_step_calls_observed 0 (see probe-delivery-001.json)

python -m pytest tests/test_research_k8.py -q -p no:cacheprovider -o addopts=
# -> 17 passed

python -m pytest tests/test_research_k8.py tests/test_research_data.py tests/test_research_learning.py tests/test_research_meta_rsi.py tests/test_research_model_wrapper.py tests/test_research_checkpoint.py -q -p no:cacheprovider -o addopts=
# -> 91 passed, 5 skipped

python -m pytest tests/ -q -p no:cacheprovider -o addopts= --ignore=tests/test_v5_training.py --ignore=tests/test_v5_distributed.py -k "not slow and not cuda and not tpu"
# -> 525 passed, same 10 pre-existing e1/e2/v5 receipt-hash failures, 11 skipped
```

Resource accounting: K8 allocation **0 updates consumed locally**
(`AdamW.step` patched and counted in both probes); old CPU ledger 206/200
unchanged; no new allocation; no historical reset.

## Files changed

- `campaigns/runner.py` — slots, per-phase success, fenced lease, preflight.
- `campaigns/supervisor.py` — exclusive occupancy, token lease + fenced release.
- `campaigns/process_supervision.py` — owned processes, device mapping, slots.
- `learning/k8_trainer.py` — single-window contract, pair consumer, checkpoint.
- `campaigns/worker.py` — real dispatch, E0 finalize + child continuation, E6.
- `campaigns/phases/__init__.py`, `types.py`, `ops.py`, `e1.py`, `e2.py`,
  `e3.py`, `e4.py`, `e5.py`, `e6.py` — **new** D4 executors.
- `data/k8_bundle.py` — unified canonical identity, meta exclusion, renderer
  separation, source-bytes manifest, strict validation.
- `campaigns/k8.py` — (unchanged export core; preflight via runner).
- `engineering/reports/K8_DELIVERY_20260914/probe_delivery.py`,
  `probe-delivery-001.json` — 74-key zero-training receipt.

## R01–R08 mapping (remaining partials are hardware-only)

R01 partial→control closed + learned handlers delivered (GPU run pending);
R02 closed (live-race beyond unit scope remains unproven by construction);
R03 delivered (live two-T4 overlap/UUID pending); R04/R05 delivered
(parameter-change/AMP/GPU equality pending); R06 delivered (full-size Kaggle
prepare pending); R07 preserved + E4/E5 wired (learned E5 confirmation
pending); R08 delivered (live Kaggle path + `.pt` reload pending). Launch
remains blocked pending the owner's authorized GPU campaign — no session was
spent testing this pipeline.
