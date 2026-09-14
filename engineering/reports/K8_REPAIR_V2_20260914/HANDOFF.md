# K8 repair V2 handoff — R01–R08 honest dispositions

Date: 2026-09-14. Agent: BRAMASTRA implementation lead. Baseline `f955d50`
(Second K8 review). This handoff supersedes the unsupported CLOSED labels in
[the prior repair handoff](../K8_REPAIR_20260914/HANDOFF.md) (preserved as
history) and responds point-by-point to
[the second review](../K8_SECOND_REVIEW_20260914/REVIEW.md) and the original
[ R01–R08 criteria](../K8_CHIEF_20260914/REVIEW.md). No new experiment or
compute allocation is claimed. Push: normal fast-forward on `BRAMASTRA`.

## Executive status

Control-plane and data/model fail-closed defects are repaired with exact
changed-function evidence below and zero local optimizer updates
(`optimizer_step_calls_observed: 0`, ledger 206/200 untouched — this phase
consumed zero optimizer updates). Focused suite: **17/17 K8 tests pass**;
broader focused set **91 passed / 5 skipped** (k8 + meta_rsi + learning +
data + wrapper + checkpoint); full suite **525 passed / 10 pre-existing
baseline failures / 11 skipped** (same 10 e1/e2/v5 receipt-hash failures as
the prior baseline, unrelated to K8). CUDA-only checks remain pending owner
launch. Launch acceptance remains blocked pending GPU verification — this
handoff does not claim launch-ready.

Probe: `engineering/reports/K8_REPAIR_V2_20260914/probe_v2.py` with receipt
`probe-v2-001.json` (47 keys, zero steps, zero accelerator work). Run:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python engineering/reports/K8_REPAIR_V2_20260914/probe_v2.py
python -m pytest tests/test_research_k8.py -q -p no:cacheprovider -o addopts=
```

## Failing-before / passing-after (second-review probe vs V2 probe)

| Check | Before (`probe-001.json` @3914dc4) | After (`probe-v2-001.json`) |
|---|---|---|
| same_id_deadline_extension | 0.0 (passing, preserved) | 0.0 |
| new allocation in same dir | 60.0s extension (defect) | **rejected** (`new_id_same_directory_rejected: true`, `SupervisorError`) |
| two zero-work receipts pass gate | true (defect) | **false** (qualified gate) |
| two qualified distinct receipts pass | — | true (new positive control) |
| failed E0 exit code | 0 (defect) | **1** (fail-closed) |
| missing bundle exit code | — (ran into E0) | 2 `DATA_NOT_READY` (fail before expensive work) |
| E2 inherits success | completed zero-work (defect) | **failed** `refused_missing_executor` (`e2_inherits_success: false`) |
| E1 status | `blocked_pending_e0` unconditional | **failed** `refused_missing_executor` (explicit, nonzero) |
| rule rows 4096 | 4096 by names (diversity defect) | 4096 with canonical semantic dedup |
| NAND(false,false)->true accepted | false (defect) | **true** |
| all ten rule types verify | 4/10 (defect) | **true** (10/10) |
| renamings share canonical key | — (counted names) | true |
| canonical distinct (256-sample) | — | 256/256 |
| mini bundle valid | — | true |
| tool heldout distinct execution | same composition (defect) | **true** (single_filter vs filter_then_aggregate_then_check + step structure) |
| meta resolved | unresolved strings (defect) | **true** (executable mechanism refs) |
| `verifier_consistent` | hardcoded true (defect) | independently recomputed per mechanism |
| scorer gated path | bypass (`decoder.forward_hidden`, defect) | **true** via `_model_hidden` → `forward_hidden`; bypass **false** |
| M0/M1 distinct | identical (defect) | **true** (different coefficients + strict mapping) |
| mismatched recipe rejected | loose mapping (defect) | **true** |
| overlap proved (2×0.4s) | synchronous only (defect) | **0.406s < 0.7s, true** |
| hang terminated | no enforcement (defect) | **true** (1.031s boundary) |
| export full bundle | ledger-only (defect) | **true** (5 files) |
| optimizer steps observed | 0 | **0** |

## Corrections to prior unsupported closure claims

- **R05 CLOSED (same function)** — withdrawn. `git diff c225c2d..3914dc4 --
  bramastra_lab/research/campaigns/worker.py` was empty; the E0 repair was
  not committed. V2 actually rewrites `worker.py`: `_run_e0` (started fix,
  canonical `training_step_full` in BOTH branches, strong checksum,
  subprocess restore proof), `_run_learned_phase` (explicit refusal), `_run_e6`,
  `full_profile_descriptor`, `_strong_checksum`, `_verify_payload_in_subprocess`.
- **R04 CLOSED (route_window imported)** — withdrawn. Import alone never
  consumed the router. V2 consumes it: `K8Trainer.accumulate` routes the token
  term; new `accumulate_full_window` builds ONE `SupervisionWindow` BEFORE
  backward, calls `route_window` (per-term OWN denominators), then ONE scaled
  backward. `finalize_update` no longer re-divides normalized gradients.
- **R07 CLOSED (all scorers use gated path)** — withdrawn.
  `learning/k8_scoring.py:49` called `model.decoder.forward_hidden` directly.
  V2 routes all scorers through `_model_hidden` → `model.forward_hidden`
  (base implements it; `GatedReuseModel` overrides to `_decoder_with_reuse`
  BEFORE final norm). No `decoder.forward_hidden` remains in scoring.
- **R03 PARTIALLY CLOSED (design)** — withdrawn as implementation. V2 adds
  real `campaigns/process_supervision.py` (spawn isolation, GPU visibility
  before torch import, concurrent paired slots, absolute deadlines, 450
  cutoff, hang termination, UUID report) wired into `runner.py`.
- **R06 CLOSED (4096 rows)** — withdrawn as diversity. Row count by names is
  not mechanism diversity. V2 implements canonical semantic keys, ten
  verifiers, cross-pool rejection, held-out execution difference, computed
  `verifier_consistent`, genuine trajectories, executable meta.
- **R08 CLOSED (notebook + ledger)** — withdrawn as packaging. Export wrote
  only the ledger. V2 writes the full bundle (below) and fails before
  expensive work on missing requirements.
- **Test counts as integration evidence** — withdrawn. 17 focused tests never
  closed execution paths; V2 keeps them as non-learning unit evidence plus
  the new zero-training probe above (changed-function + negative controls).

## R01–R08 dispositions (honest, per-package)

### R01 — Phases and E0 admission gate: PARTIALLY CLOSED (control plane closed; learned execution pending GPU)
Changed functions:
- `supervisor.CampaignLedger.qualified_phase_receipts` (new),
  `supervisor.CampaignLedger.phase_success` (distinct qualified devices +
  checkpoint + updates + exposure + device_seconds, not row count).
- `runner.run_campaign` (qualified gate, per-job `completed_jobs` resume
  instead of per-phase names, `CAMPAIGN_FAILED` + return 1 on
  failure, return 2 on `DATA_NOT_READY`/`ALLOCATION_REJECTED`, ledger close
  in finally, training-cutoff skip, concurrent dispatch).
- `worker.run_worker_phase` (unknown-phase refuse, deadline refuse),
  `worker._run_learned_phase` (manifest check + explicit
  `refused_missing_executor` failed — never `blocked_pending_e0`
  masquerading or zero-work success; E2 no longer inherits success),
  `worker._run_e6` (ledger + manifest verification).
Failing-before: fake-E0 failure dispatched E1–E5 and exited 0; E2 completed
with zero work. Passing-after: `failed_e0_exit_code: 1`,
`e2_inherits_success: false`, `e1_status: failed`,
`two_zero_work_receipts_on_same_gpu_pass_gate: false`,
`two_qualified_distinct_receipts_pass_gate: true`. Focused: 17/17 K8 tests.
Remaining (GPU-only): real E1–E5 learned executors behind the refusal gate +
live E6 artifact export on owner data; launch still blocked.

### R02 — Allocation and job identity: CLOSED (CPU-provable parts)
Changed functions:
- `supervisor.CampaignLedger.record_allocation` (one immutable allocation per
  run dir; second ID raises; same-ID deadline never moves; e0→full shares ID).
- `supervisor.CampaignLedger.reserve` (compatible-retry validation: same
  job_id with different worker/device/phase/arm/seed/parent raises;
  idempotent compatible retry returns existing; per-device occupancy:
  same-device overlap refused, different-device concurrent admitted).
- `supervisor.CampaignLedger.phase_success` (above) + `phase_consumption`
  (unchanged aggregation).
- `supervisor.SupervisorLease.acquire` (verified stale recovery: mtime age
  or dead PID → remove once and retry; live holder still refuses).
- `runner.run_campaign` (per-job resume, elapsed `device_seconds` on
  exception instead of zero, ledger close on every path).
Failing-before: new-ID extension 60.0s; duplicate IDs unchecked; global sum
blocked two-GPU admission; zero-work receipts passed; lock had no recovery.
Passing-after: `same_id_deadline_extension: 0.0`,
`new_id_same_directory_rejected: true`,
`incompatible_retry_rejected: true`, `compatible_retry_idempotent: true`,
per-device test updated (j1+j2 admitted, j3 same-GPU refused).
Remaining: live two-supervisor race under true concurrency (SQLite UNIQUE +
checks are best-effort locally); no local ledger reset performed.

### R03 — Parallel workers and deadlines: PARTIALLY CLOSED (supervision real; live two-T4 pending)
New module `campaigns/process_supervision.py` (real production code):
- `campaign_training_cutoff` (450), `phase_absolute_deadlines` (start-derived,
  training clamped, E6 full 480), `_child_execute` (GPU visibility BEFORE
  torch import in fresh spawn), `_run_one_in_process` (hard timeout +
  termination), `run_phase_concurrently` (paired-slot overlap, hang →
  timed_out, no export borrowing; test doubles allowed ONLY at expensive
  model boundary), `verify_physical_devices` (UUIDs, not indices).
Wired into `runner.py` (absolute timeouts, cutoff skip, concurrent E0+).
Failing-before: synchronous nested loops, descriptive caps, device strings
without isolation. Passing-after: `concurrent_overlap_proved: true`
(0.406s), `hang_terminated: true` (1.031s), `training_cutoff_minutes: 450.0`,
`e0_deadline_absolute: true`, `training_clamped: true`.
Remaining (GPU-only): live two-T4 overlap/separation + UUID verification +
minute-450 hard stop on hardware (owner allocation).

### R04 — Canonical learning boundary: PARTIALLY CLOSED (boundary real; parameter-change + AMP pending E0 GPU)
Changed functions:
- `learning/k8_trainer.K8Trainer.__init__` (`require_allocation` flag,
  `attempted_updates`, `_pending_normalized`).
- `learning/k8_trainer.K8Trainer._admit_update` (missing allocation refused
  when required).
- `learning/k8_trainer.K8Trainer.accumulate` (token term via `route_window`,
  pair_rows pooled, single scaled backward).
- `learning/k8_trainer.K8Trainer.accumulate_full_window` (new: window BEFORE
  backward, `route_window` per-term denominators, ONE scaled backward).
- `learning/k8_trainer.K8Trainer.finalize_update` (no re-division when
  normalized; attempted accounting; LR applied; scaler.step/update; failed
  report after commit never erases step).
- `learning/k8_trainer.K8Trainer.state_payload/load_state_payload`
  (attempted/skipped/scaler/allocation round-trip).
Failing-before: answer-only, `pair_rows` unused, `route_window` unimported
use, single-denominator division, missing allocation admitted.
Passing-after: `accumulate_consumes_route_window: true`,
`full_window_single_backward: true`,
`full_window_backward_carries_gradients: true`,
`pending_normalized_flag: true`, `missing_allocation_refused: true`,
`AllocationGateTests` still pass (exhausted/deadline refuse).
Remaining (GPU-only): production-path attributable-gradient + window-
normalization equivalence on full model, real parameter-change and AMP
checks in E0 (owner launch).

### R05 — E0 comparison: PARTIALLY CLOSED (orchestration real; GPU equality + throughput pending)
Changed functions: `worker._run_e0` (above) + `_strong_checksum`
(model+optimizer+counters+schedule SHA256, not parameter sum),
`_payload_identity`, `_verify_payload_in_subprocess` (spawn reload proof),
`full_profile_descriptor` (tiny probe vs development calibration, no local
training), `training_step_full` canonical closure used in BOTH branches.
Failing-before: divergent treatments (answer-only resume), mixed
scaled/unscaled, tiny-only, same-process restore, weak sum, undefined
`started`. Passing-after (orchestration, zero-training): identical-function
use is structural (both branches call `training_step_full` →
`accumulate_full_window`); checksum/subprocess/profile are code-present and
unit-imported; no local E0 executed per policy (probe uses doubles).
Remaining (GPU-only): full-profile 6-update comparison, checksum agreement,
next-batch equivalence and throughput calibration on T4 (owner launch, 6
updates/worker charged to K8 allocation, caps retained).

### R06 — Data diversity and qualification: CLOSED
Changed functions (`data/k8_bundle.py`):
- `generate_rule_inquiry_mechanism` (+ relevant subset + negations),
  `canonical_rule_key` (renamings grouped), `_canonical_key_for_mechanism`
  (function class + distractor COUNT + noise PROCESS; names excluded),
  `verify_rule` (+ nand/nor/xnor/majority/exactly_one/at_least_two with
  negations/relevant), `generate_tool_mechanism` (held-out
  filter_then_aggregate_then_check with genuinely different steps/predicates),
  `verify_tool` (two-part held-out answer), `_mechanisms_for_family`
  (canonical dedup + `held_out_composition` + `exclude_canonical` +
  explicit `insufficient_inventory` error), `_materialize_trajectory`
  (action/result history), `_verifier_consistent` (recomputed, not hardcoded),
  `build_k8_bundle` (cross-pool canonical rejection, held-out flag passed,
  trajectories, computed flags, executable meta with real mechanism refs),
  `validate_bundle` + `_validate_disjointness/_validate_verifier_flags/
  _validate_tool_heldout/_validate_meta` (wrong answers, renamed duplicates,
  unresolved meta, undersized pools and shared tool compositions fail).
Failing-before: 4 verifiers, name-counted 4096, overlap unrejected, shared
tool composition, hardcoded true, label-only trajectories, unresolved meta.
Passing-after: `generated_rule_rows: 4096`,
`nand_false_false_target_true_accepted: true`,
`all_ten_rule_types_verify: true`, `renamings_share_canonical_key: true`,
`canonical_distinct_256: 256`, `mini_bundle_valid: true`,
`tool_compositions_distinct: true`, `meta_resolved: true`.
No GPU needed; full 4096 + 128-confirmation qualification remains a Kaggle
`prepare`+`validate` step (not run locally).

### R07 — Architecture and proposer semantics: PARTIALLY CLOSED (semantics real; learned E5 pending)
Changed functions:
- `models.IntegratedModel.forward_hidden` (new canonical hidden path).
- `models/gated.GatedReuseModel._decoder_with_reuse` (padding + attention
  masks forwarded, reuse BEFORE final norm), `GatedReuseModel.forward`
  (full contract: segment_ids/action_span_ends/action_mask/return_hidden/
  return_value preserved through gated hidden), `GatedReuseModel.forward_hidden`
  (segment/attention support).
- `learning/k8_scoring._model_hidden` (all scorers via `forward_hidden`;
  direct `decoder.forward_hidden` removed) — `score_candidates_trainable`
  live-gradient preserved.
- `metalearning/dispatch._METHOD_PROGRAMS` (M0 anchor vs M1 reduced-LR
  variant distinct; M2 clip), `_program_to_method_id` (strict lineage, loose
  JSON raises), `dispatch_method_to_trainer` (compiled program identity must
  match declared lineage; M1 halves LR; M2 requires gates).
Failing-before: bypassed gates, dropped segment/output contract, identical
M0/M1, loose mapping, unchecked caller recipes. Passing-after:
`scorer_uses_gated_path: true`, `scorer_bypasses_gates: false`,
`gated_preserves_segment_contract: true`, `gated_reuse_before_norm: true`,
`scorer_live_gradient: true`, `m0_m1_distinct: true`, `m0_maps_m0/m1_maps_m1:
true`, `mismatched_recipe_rejected: true`, `GatedArchitectureTests` pass.
Remaining (GPU/learning): model/scorer combined gradients through shared vs
extra path on full model + actual E5 proposer/successor execution (owner
allocation); parsing alone never claimed as I04 completion.

### R08 — Packaging and handoff: PARTIALLY CLOSED (bundle real; live Kaggle path pending)
Changed functions: `campaigns/k8.cmd_export` (full bundle:
`campaign_ledger.json` + `phase_results.json` (qualified counts/devices) +
`allocation.json` + `protocol.json` (frozen 480/450/export + admission) +
`restore_evidence.json` (checkpoint inventory + GPU-gated note); refuses with
exit 2 when ledger missing), `runner.run_campaign` + `k8.cmd_run`
(`DATA_NOT_READY` exit 2 on missing manifest; setup fails before expensive
work). Notebook `notebooks/bramastra_k8.ipynb` already uses
`subprocess.run` argument lists (zero `!` shell calls, verified by inspection;
no change needed in V2).
Failing-before: ledger-only export, shell fragments (previously fixed),
unsupported closure claims. Passing-after: `export_exit_code: 0`,
`export_full_bundle: true` (5 files), `missing_data_exit_code: 2`.
Remaining (owner-side): end-to-end Kaggle prepare→validate→e0→full→export
with restorable `.pt` durability proof (not run locally per policy).

## Focused verification (actual commands run)

```powershell
$env:PYTHONPATH = (Get-Location).Path
python engineering/reports/K8_REPAIR_V2_20260914/probe_v2.py
# -> 47 keys, all true/expected, optimizer_step_calls_observed 0 (see probe-v2-001.json)

python -m pytest tests/test_research_k8.py -q -p no:cacheprovider -o addopts=
# -> 17 passed in ~4s

python -m pytest tests/test_research_k8.py tests/test_research_meta_rsi.py tests/test_research_learning.py tests/test_research_data.py tests/test_research_model_wrapper.py tests/test_research_checkpoint.py -q -p no:cacheprovider -o addopts=
# -> 91 passed, 5 skipped

python -m pytest tests/ -q -p no:cacheprovider -o addopts= --ignore=tests/test_v5_training.py --ignore=tests/test_v5_distributed.py -k "not slow and not cuda and not tpu"
# -> 525 passed, 10 pre-existing baseline failures (e1/e2/v5 receipt hashes), 11 skipped
```

Resource accounting: old CPU ledger **206/200 unchanged**; K8 allocation
**0 updates consumed locally** (probe patches `AdamW.step` and asserts 0;
E0 never executed locally); no new allocation; no historical-ledger reset.

## Files changed (production + evidence)

- `bramastra_lab/research/campaigns/supervisor.py` — immutable binding,
  compatible retry, per-device, qualified receipts, stale lease.
- `bramastra_lab/research/campaigns/runner.py` — qualified gate, per-job
  resume, exit codes, cutoff, concurrent dispatch, durable consumption, close.
- `bramastra_lab/research/campaigns/process_supervision.py` — **new** (R03).
- `bramastra_lab/research/campaigns/worker.py` — real E0/E1-E6 handlers,
  strong identity, subprocess restore.
- `bramastra_lab/research/learning/k8_trainer.py` — canonical boundary,
  require_allocation, attempted accounting.
- `bramastra_lab/research/learning/k8_scoring.py` — gated path.
- `bramastra_lab/research/data/k8_bundle.py` — qualified data.
- `bramastra_lab/research/models/gated.py`, `models/wrapper.py` — contract.
- `bramastra_lab/research/metalearning/dispatch.py` — distinct lineages.
- `bramastra_lab/research/campaigns/k8.py` — full export.
- `tests/test_research_k8.py` — per-device capacity expectation corrected
  (two-GPU admission + same-GPU refusal per R02).
- `engineering/reports/K8_REPAIR_V2_20260914/probe_v2.py`,
  `probe-v2-001.json` — zero-training diagnostics + receipt.

## Remaining GPU-only checks (explicitly pending owner launch)

- Actual CUDA gradient flow through the full development-profile model with AMP.
- Real T4 resume equivalence (strong checksums, counters, next batch) + throughput.
- Two-worker concurrent dispatch with physical UUIDs + minute-450 hard stop.
- Full 4096/256/256/128 + 256/64 tool bundle `prepare`+`validate` on Kaggle.
- E1–E5 learned results and E5 proposer/successor lineage confirmation.
- Live Kaggle export durability + fresh-process restore of `.pt` payloads.

Launch acceptance remains blocked until those run under the owner's
allocation. No local training was performed to produce this handoff.
